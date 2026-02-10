from __future__ import annotations

"""
Shared evaluation entry point for CLI and Trainer auto-eval.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from src.core.checkpoint import load_checkpoint
from src.core.logging import get_logger
from src.data.dataloader import make_dataloader
from src.eval.calibration import compute_ece_from_logits
from src.eval.funnel import funnel_consistency
from src.eval.metrics import compute_binary_metrics, sigmoid
from src.models.build import build_model
from src.train.infer import infer_to_parquet
from src.utils.feature_meta import build_model_feature_meta
from src.utils.metrics_schema import apply_esmm_schema

LOG = get_logger(__name__)


def _move_features_to_device(features: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """
    Copy of the lightweight mover in train.infer to avoid circular imports.
    """
    fields = {}
    for base, fd in features["fields"].items():
        fields[base] = {
            "indices": fd["indices"].to(device),
            "offsets": fd["offsets"].to(device),
            "weights": fd["weights"].to(device) if fd.get("weights") is not None else None,
        }
    return {"fields": fields, "field_names": features["field_names"]}


def _resolve_enabled_heads(model_cfg: Dict[str, Any]) -> list[str]:
    """
    Resolve enabled heads with fallbacks and stable ordering.
    """
    heads = model_cfg.get("enabled_heads") or model_cfg.get("tasks")
    if not heads:
        heads = ["ctr", "cvr"]
    return sorted([str(h).lower() for h in heads])


def _build_eval_metadata(cfg: Dict[str, Any], config_path: Optional[str | Path]) -> Dict[str, Any]:
    model_cfg = cfg.get("model", {})
    enabled_heads = _resolve_enabled_heads(model_cfg)
    model_name = model_cfg.get("name", "unknown_model")
    exp_name = f"{model_name}__heads={'-'.join(enabled_heads)}"
    data_seed = cfg.get("data", {}).get("seed")
    runtime_seed = cfg.get("runtime", {}).get("seed")
    seed = data_seed if data_seed is not None else runtime_seed

    metadata = {
        "model_name": model_name,
        "enabled_heads": enabled_heads,
        "exp_name": exp_name,
        "config_path": str(config_path) if config_path else None,
    }
    metadata["seed"] = seed if seed is not None else None
    return metadata


def _resolve_run_dir(run_dir: Optional[str | Path], ckpt_path: Optional[str | Path], cfg: Dict[str, Any]) -> Path:
    if run_dir is not None:
        return Path(run_dir)
    if ckpt_path is not None:
        return Path(ckpt_path).resolve().parent
    # Fallback: derive from experiment name with timestamp
    exp_name = cfg.get("experiment", {}).get("name", "eval")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("runs") / f"{exp_name}_eval_{ts}"


def _build_dataloader(cfg: Dict[str, Any], split: str, feature_meta: Dict[str, Any]):
    data_cfg = cfg.get("data", {})
    # Use num_workers_valid for eval to avoid multiprocessing issues
    num_workers = int(data_cfg.get("num_workers_valid", data_cfg.get("num_workers", 0)))
    # Only set prefetch_factor if num_workers > 0 (required by PyTorch DataLoader)
    prefetch = int(data_cfg.get("prefetch_factor", 2)) if num_workers > 0 else None
    return make_dataloader(
        split=split,
        batch_size=int(data_cfg.get("batch_size", 256)),
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        persistent_workers=bool(data_cfg.get("persistent_workers", False)),
        seed=data_cfg.get("seed"),
        feature_meta=feature_meta,
        debug=bool(data_cfg.get("debug", False)),
        prefetch_factor=prefetch,
    )


def run_eval(
    cfg: Dict[str, Any],
    split: str,
    ckpt_path: Optional[str | Path] = None,
    run_dir: Optional[str | Path] = None,
    save_preds: bool = False,
    max_batches: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
    config_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """
    Main evaluation routine. Returns the eval summary dict and also writes
    eval.json under the resolved run_dir.
    """
    log = logger or LOG
    device = torch.device(cfg.get("runtime", {}).get("device", "cpu"))
    metadata = _build_eval_metadata(cfg, config_path)
    enabled_heads_list = metadata["enabled_heads"]
    enabled_heads = set(enabled_heads_list)
    resolved_run_dir = _resolve_run_dir(run_dir, ckpt_path, cfg)
    use_esmm = bool(cfg.get("use_esmm", False))
    esmm_eps = float(cfg.get("esmm", {}).get("eps", 1e-8))

    # Build feature meta + loader + model
    feature_meta = build_model_feature_meta(Path(cfg["data"]["metadata_path"]), cfg.get("embedding", {}))
    loader = _build_dataloader(cfg, split=split, feature_meta=feature_meta)
    model = build_model(cfg).to(device)

    if ckpt_path:
        load_checkpoint(ckpt_path, model, optimizer=None, map_location=device, strict=False)
        log.info("Loaded checkpoint from %s", ckpt_path)

    # Switch to eval mode to disable dropout/BN training behavior
    model.eval()

    # Negative sampling correction: if training used neg_keep_prob < 1, we need to
    # shift logits to correct for the sampling bias in probability estimates.
    # Formula: logit_corrected = logit_raw - log((1-r)/r) where r = neg_keep_prob
    # This does NOT affect AUC (rank-preserving), but fixes logloss/ECE/pred_mean.
    sampling_mode = str(cfg.get("sampling", {}).get("negative_sampling", "keep_prob")).lower()
    neg_keep_prob = float(cfg.get("data", {}).get("neg_keep_prob_train", 1.0))
    # Only disable correction if sampling is explicitly disabled
    if sampling_mode in {"none", "off", "disable", "disabled"}:
        neg_keep_prob = 1.0
    logit_correction = 0.0
    if neg_keep_prob < 1.0 and neg_keep_prob > 0.0:
        # Correction term to shift from sampled distribution to original distribution
        logit_correction = float(np.log((1.0 - neg_keep_prob) / neg_keep_prob))
        log.info("Applying logit correction=%.4f for neg_keep_prob=%.3f (use_esmm=%s)", logit_correction, neg_keep_prob, use_esmm)

    # Forward pass to collect logits/labels for metrics
    use_ctr = "ctr" in enabled_heads
    use_cvr = "cvr" in enabled_heads

    y_ctr_list = [] if use_ctr else None
    y_cvr_list = [] if use_cvr else None
    click_mask_list = [] if use_cvr else None
    y_ctcvr_list = [] if use_cvr and use_esmm else None
    ctr_logit_list = [] if use_ctr else None
    cvr_logit_list = [] if use_cvr else None
    ctcvr_logit_list = [] if use_cvr and use_esmm else None
    ctcvr_logit_res_list = [] if use_cvr and use_esmm else None
    ctr_wide_logit_list = [] if use_ctr else None
    ctr_parts_list = [] if use_ctr else None
    logit_parts_decomposable = None

    with torch.no_grad():
        for step, (labels, features, meta) in enumerate(loader):
            if max_batches is not None and step >= max_batches:
                break
            labels_dev = {k: v.to(device) for k, v in labels.items() if torch.is_tensor(v)}
            features_dev = _move_features_to_device(features, device)

            outputs = model(features_dev)

            if use_ctr:
                ctr_logit = outputs["ctr"]
                if ctr_logit.dim() > 1:
                    ctr_logit = ctr_logit.view(-1)
                y_ctr_list.append(labels_dev["y_ctr"].cpu().numpy())
                ctr_logit_list.append(ctr_logit.cpu().numpy())
                if "ctr_logit_parts" in outputs:
                    parts = outputs["ctr_logit_parts"]
                    ctr_parts_list.append(
                        {
                            "wide": parts.get("wide"),
                            "fm": parts.get("fm"),
                            "deep": parts.get("deep"),
                            "total": parts.get("total"),
                        }
                    )
                if "logit_parts_decomposable" in outputs:
                    logit_parts_decomposable = bool(outputs["logit_parts_decomposable"])
                if ctr_wide_logit_list is not None and "ctr_logit_parts" in outputs:
                    wide_comp = outputs["ctr_logit_parts"].get("wide")
                    if wide_comp is not None:
                        if wide_comp.dim() > 1:
                            wide_comp = wide_comp.view(-1)
                        ctr_wide_logit_list.append(wide_comp.cpu().numpy())

            if use_cvr:
                cvr_logit = outputs["cvr"]
                if cvr_logit.dim() > 1:
                    cvr_logit = cvr_logit.view(-1)
                if use_esmm:
                    cvr_logit_list.append(cvr_logit.cpu().numpy())
                    if ctcvr_logit_list is not None:
                        ctr_logit_for_joint = outputs["ctr"]
                        if ctr_logit_for_joint.dim() > 1:
                            ctr_logit_for_joint = ctr_logit_for_joint.view(-1)
                        p_ctcvr = sigmoid(ctr_logit_for_joint) * sigmoid(cvr_logit)
                        p_ctcvr = torch.clamp(p_ctcvr, min=esmm_eps, max=1.0 - esmm_eps)
                        ctcvr_logit = torch.log(p_ctcvr / (1.0 - p_ctcvr))
                        ctcvr_logit_list.append(ctcvr_logit.cpu().numpy())
                        # Residual CTCVR logit (logit-space additive)
                        r_logit = outputs.get("residual_logit")
                        if r_logit is not None and ctcvr_logit_res_list is not None:
                            r_logit_flat = r_logit.view(-1) if r_logit.dim() > 1 else r_logit
                            ctcvr_logit_res = (
                                ctr_logit_for_joint + cvr_logit + r_logit_flat
                            )
                            ctcvr_logit_res_list.append(ctcvr_logit_res.cpu().numpy())
                    if y_ctcvr_list is not None and "y_ctcvr" in labels_dev:
                        y_ctcvr_list.append(labels_dev["y_ctcvr"].cpu().numpy())

                    y_cvr_list.append(labels_dev["y_cvr"].cpu().numpy())
                    # click mask equivalent: clicked rows (y_ctr==1)
                    click_mask_list.append(labels_dev["y_ctr"].cpu().numpy())
                else:
                    y_cvr_list.append(labels_dev["y_cvr"].cpu().numpy())
                    click_mask_list.append(labels_dev["click_mask"].cpu().numpy())
                    cvr_logit_list.append(cvr_logit.cpu().numpy())
                    if y_ctcvr_list is not None and "y_ctcvr" in labels_dev:
                        y_ctcvr_list.append(labels_dev["y_ctcvr"].cpu().numpy())

    y_ctr = np.concatenate(y_ctr_list) if use_ctr and y_ctr_list else np.array([])
    y_cvr = np.concatenate(y_cvr_list) if use_cvr and y_cvr_list else np.array([])
    click_mask = np.concatenate(click_mask_list) if use_cvr and click_mask_list else np.array([])
    y_ctcvr = np.concatenate(y_ctcvr_list) if y_ctcvr_list else None
    ctr_logit_raw = np.concatenate(ctr_logit_list) if use_ctr and ctr_logit_list else np.array([])
    cvr_logit = np.concatenate(cvr_logit_list) if use_cvr and cvr_logit_list else np.array([])
    ctcvr_logit = np.concatenate(ctcvr_logit_list) if ctcvr_logit_list else None
    ctcvr_logit_res = np.concatenate(ctcvr_logit_res_list) if ctcvr_logit_res_list else None
    residual_available = ctcvr_logit_res is not None
    ctr_wide_logit_raw = np.concatenate(ctr_wide_logit_list) if ctr_wide_logit_list else np.array([])

    # Apply logit correction to CTR predictions (negative sampling affects CTR label distribution)
    # CVR is evaluated on clicked subset only, which is not affected by CTR negative sampling
    ctr_logit = ctr_logit_raw - logit_correction if ctr_logit_raw.size > 0 else ctr_logit_raw
    ctr_wide_logit = ctr_wide_logit_raw - logit_correction if ctr_wide_logit_raw.size > 0 else ctr_wide_logit_raw

    empty_metrics = {"logloss": None, "auc": None, "pos_rate": None, "pred_mean": None, "n": 0}
    ctr_metrics = compute_binary_metrics(y_ctr, ctr_logit) if use_ctr else empty_metrics
    if use_cvr:
        if use_esmm:
            cvr_metrics_masked = compute_binary_metrics(
                y_cvr, cvr_logit, mask=(y_ctr > 0.5)
            )
            ctcvr_metrics = (
                compute_binary_metrics(y_ctcvr, ctcvr_logit)
                if (y_ctcvr is not None and ctcvr_logit is not None)
                else empty_metrics
            )
            # Residual CTCVR metrics (dual-AUC)
            ctcvr_metrics_res = (
                compute_binary_metrics(y_ctcvr, ctcvr_logit_res)
                if (y_ctcvr is not None and residual_available)
                else empty_metrics
            )
        else:
            cvr_metrics_masked = compute_binary_metrics(y_cvr, cvr_logit, mask=click_mask > 0.5)
            ctcvr_metrics = empty_metrics
    else:
        cvr_metrics_masked = empty_metrics
        ctcvr_metrics = empty_metrics

    ctr_wide_metrics = compute_binary_metrics(y_ctr, ctr_wide_logit) if (use_ctr and ctr_wide_logit.size) else empty_metrics

    ece_ctr = compute_ece_from_logits(y_ctr, ctr_logit) if use_ctr else {"ece": None, "bins": []}
    if use_cvr:
        if use_esmm:
            ece_cvr_masked = compute_ece_from_logits(y_cvr, cvr_logit, mask=(y_ctr > 0.5))
        else:
            ece_cvr_masked = compute_ece_from_logits(y_cvr, cvr_logit, mask=click_mask > 0.5)
    else:
        ece_cvr_masked = {"ece": None, "bins": []}

    pred_ctr_prob = sigmoid(torch.from_numpy(ctr_logit)).numpy() if use_ctr and ctr_logit.size > 0 else np.array([])
    if use_cvr and cvr_logit.size > 0:
        pred_cvr_prob = sigmoid(torch.from_numpy(cvr_logit)).numpy()
    else:
        pred_cvr_prob = np.array([])
    funnel_stats = None
    if use_ctr and use_cvr and pred_ctr_prob.size and pred_cvr_prob.size:
        funnel_stats = funnel_consistency(
            {
                "pred_ctr": pred_ctr_prob,
                "pred_cvr": pred_cvr_prob,
                "y_ctcvr": y_ctcvr,
            },
            has_ctcvr_label=y_ctcvr is not None,
        )

    # Optional predictions export (fresh loader to avoid Iterable exhaustion)
    preds_summary = None
    if save_preds:
        preds_path = resolved_run_dir / f"preds_{split}.parquet"
        preds_loader = _build_dataloader(cfg, split=split, feature_meta=feature_meta)
        preds_summary = infer_to_parquet(
            model,
            preds_loader,
            device=device,
            out_path=preds_path,
            max_batches=max_batches,
            include_ctcvr=bool(use_ctr and use_cvr),
            split=split,
            enabled_heads=enabled_heads_list,
        )

    def _stack_parts(parts_list, key):
        if not parts_list:
            return None
        vals = []
        for p in parts_list:
            v = p.get(key)
            if v is None:
                continue
            if torch.is_tensor(v):
                v = v.view(-1).cpu().numpy()
            vals.append(v)
        if not vals:
            return None
        return np.concatenate(vals)

    def _part_stats(component: Optional[np.ndarray], total: Optional[np.ndarray]) -> Dict[str, Optional[float]]:
        if component is None or component.size == 0:
            return {"mean": None, "std": None, "min": None, "max": None, "var_ratio": None, "corr_with_total": None}
        stats = {
            "mean": float(component.mean()),
            "std": float(component.std()),
            "min": float(component.min()),
            "max": float(component.max()),
        }
        var_total = float(total.var()) if total is not None and total.size else 0.0
        stats["var_ratio"] = float(component.var() / var_total) if var_total > 0 else None
        if total is not None and total.size == component.size and component.size > 1:
            try:
                corr = float(np.corrcoef(component, total)[0, 1])
            except Exception:
                corr = None
        else:
            corr = None
        stats["corr_with_total"] = corr
        return stats

    ctr_parts_stats = None
    if ctr_parts_list:
        wide_comp = _stack_parts(ctr_parts_list, "wide")
        fm_comp = _stack_parts(ctr_parts_list, "fm")
        deep_comp = _stack_parts(ctr_parts_list, "deep")
        total_comp = _stack_parts(ctr_parts_list, "total")
        ctr_parts_stats = {
            "wide": _part_stats(wide_comp, total_comp),
            "fm": _part_stats(fm_comp, total_comp),
            "deep": _part_stats(deep_comp, total_comp),
            "total": _part_stats(total_comp, total_comp),
        }

    resolved_run_dir.mkdir(parents=True, exist_ok=True)
    eval_path = resolved_run_dir / "eval.json"

    result = {
        **metadata,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "split": split,
        "ckpt_path": str(ckpt_path) if ckpt_path else None,
        "run_dir": str(resolved_run_dir),
        "ctr_auc": ctr_metrics["auc"] if use_ctr else None,
        "cvr_auc_masked": cvr_metrics_masked["auc"] if use_cvr else None,
        "ctcvr_auc": ctcvr_metrics_res["auc"] if (use_cvr and use_esmm and residual_available) else (ctcvr_metrics["auc"] if use_cvr and use_esmm else None),
        "ctcvr_auc_mul": ctcvr_metrics["auc"] if use_cvr and use_esmm else None,
        "ctcvr_auc_res": ctcvr_metrics_res["auc"] if (use_cvr and use_esmm and residual_available) else None,
        "ctr_logloss": ctr_metrics["logloss"] if use_ctr else None,
        "cvr_logloss_masked": cvr_metrics_masked["logloss"] if use_cvr else None,
        "ctcvr_logloss": ctcvr_metrics["logloss"] if use_cvr and use_esmm else None,
        "ctr_wide_only_auc": ctr_wide_metrics["auc"] if use_ctr else None,
        "ctr_wide_only_logloss": ctr_wide_metrics["logloss"] if use_ctr else None,
        "ece_ctr": ece_ctr["ece"] if use_ctr else None,
        "ece_cvr_masked": ece_cvr_masked["ece"] if use_cvr else None,
        "ctr": ctr_metrics,
        "cvr_masked": cvr_metrics_masked,
        "ctcvr": ctcvr_metrics if use_cvr and use_esmm else empty_metrics,
        "ece_ctr_bins": ece_ctr.get("bins") if use_ctr else [],
        "ece_cvr_bins": ece_cvr_masked.get("bins") if use_cvr else [],
        "funnel_gap_stats": funnel_stats,
        "preds": preds_summary,
        "n": int(y_ctr.shape[0]) if use_ctr else 0,
        "n_masked": int(int((click_mask > 0.5).sum())) if use_cvr and click_mask.size else 0,
        "ctr_logit_parts_stats": ctr_parts_stats,
        "logit_parts_decomposable": logit_parts_decomposable,
    }

    if use_esmm:
        # bridge legacy keys so schema enrichment can add clearer ESMM names without dropping old fields
        result.setdefault("n_rows", result.get("n"))
        result.setdefault("mask_cvr_sum", result.get("n"))  # exposures count per eval
        result.setdefault("n_masked_train", result.get("n_masked"))
        result.setdefault("auc_cvr", result.get("cvr_auc_masked"))
        result.setdefault("loss_cvr", result.get("ctcvr_logloss"))
        result = apply_esmm_schema(result, use_esmm=True)

    with eval_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    log_parts = [f"Eval split={split}"]
    if use_ctr:
        log_parts.append(f"ctr_auc={ctr_metrics['auc']}")
        log_parts.append(f"ctr_logloss={ctr_metrics['logloss']}")
        log_parts.append(f"ece_ctr={ece_ctr['ece']}")
    else:
        log_parts.append("ctr=disabled")
    if use_cvr:
        log_parts.append(f"cvr_auc_masked={cvr_metrics_masked['auc']}")
        log_parts.append(f"cvr_logloss_masked={cvr_metrics_masked['logloss']}")
        if use_esmm:
            if residual_available:
                log_parts.append(f"ctcvr_auc_mul={ctcvr_metrics['auc']}")
                log_parts.append(f"ctcvr_auc_res={ctcvr_metrics_res['auc']}")
            else:
                log_parts.append(f"ctcvr_auc={ctcvr_metrics['auc']}")
            log_parts.append(f"ctcvr_logloss={ctcvr_metrics['logloss']}")
            log_parts.append(f"auc_cvr_click={cvr_metrics_masked['auc']}")
            log_parts.append(f"n_exposure={result.get('n')}")
            log_parts.append(f"n_click={result.get('n_masked')}")
        log_parts.append(f"ece_cvr_masked={ece_cvr_masked['ece']}")
    else:
        log_parts.append("cvr=disabled")
    log.info(" ".join(log_parts))
    log.info("Eval written to %s", eval_path)

    return result


__all__ = ["run_eval"]
