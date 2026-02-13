from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

from src.models.build import build_model, compute_head_priors
from src.train.optim import build_optimizer_bundle
from src.utils.config import load_yaml


PRIOR_EPS = 1e-8


def _safe_float(value: Any, key: str) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        print(f"[warn] {key}={value!r} is not a valid float; treat as missing.")
        return None


def _logit(prob: float) -> float:
    p = min(max(float(prob), PRIOR_EPS), 1.0 - PRIOR_EPS)
    return math.log(p / (1.0 - p))


def _resolve_metadata_path(cfg: Dict[str, Any], metadata_arg: Optional[str]) -> Path:
    if metadata_arg:
        return Path(metadata_arg)
    cfg_path = cfg.get("data", {}).get("metadata_path")
    if not cfg_path:
        raise ValueError("metadata path is missing. Use --metadata or set data.metadata_path in config.")
    return Path(cfg_path)


def _load_metadata(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"metadata file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_train_rates(metadata: Dict[str, Any]) -> tuple[Optional[float], Optional[float], bool]:
    train_stats = ((metadata or {}).get("split_stats") or {}).get("train", {}) or {}
    ctr = _safe_float(train_stats.get("ctr"), "split_stats.train.ctr")
    ctcvr = _safe_float(train_stats.get("ctcvr"), "split_stats.train.ctcvr")
    used_cvr_fallback = False
    if ctcvr is None:
        ctcvr_from_cvr = _safe_float(train_stats.get("cvr"), "split_stats.train.cvr")
        if ctcvr_from_cvr is not None:
            ctcvr = ctcvr_from_cvr
            used_cvr_fallback = True
    return ctr, ctcvr, used_cvr_fallback


def _get_head_bias(model: Any, task: str) -> Optional[float]:
    towers = getattr(model, "towers", None)
    if towers is None or task not in towers:
        return None
    head = towers[task]
    bias = getattr(getattr(head, "out_proj", None), "bias", None)
    if bias is None:
        return None
    return float(bias.detach().view(-1)[0].cpu().item())


def _find_param_group_info(model: Any, dense_opt: Any, param_name: str) -> Optional[dict]:
    named_params = dict(model.named_parameters())
    if param_name not in named_params:
        return None
    target_id = id(named_params[param_name])
    for idx, group in enumerate(dense_opt.param_groups):
        if any(id(p) == target_id for p in group["params"]):
            return {
                "group_idx": idx,
                "weight_decay": float(group.get("weight_decay", 0.0)),
                "lr": float(group.get("lr", 0.0)),
            }
    return None


def _find_layernorm_bias_name(model: Any) -> Optional[str]:
    for name, _ in model.named_parameters():
        low = name.lower()
        if low.endswith("layernorm.bias") or low.endswith("norm.bias") or ".bn" in low and low.endswith(".bias"):
            return name
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect ESMM priors and AdamW decay/no_decay grouping.")
    parser.add_argument("--config", type=str, required=True, help="Path to run/config YAML.")
    parser.add_argument("--metadata", type=str, default=None, help="Path to metadata.json (optional override).")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    metadata_path = _resolve_metadata_path(cfg, args.metadata)
    cfg.setdefault("data", {})["metadata_path"] = str(metadata_path)

    metadata = _load_metadata(metadata_path)
    train_ctr_rate, train_ctcvr_rate, used_cvr_fallback = _extract_train_rates(metadata)
    train_cvr_click_rate = None
    if train_ctr_rate is not None and train_ctcvr_rate is not None:
        train_cvr_click_rate = train_ctcvr_rate / max(train_ctr_rate, PRIOR_EPS)

    head_priors = compute_head_priors(cfg, metadata)

    print("=== Train Rates ===")
    print(f"metadata_path: {metadata_path}")
    print(f"train_ctr_rate: {train_ctr_rate}")
    print(f"train_ctcvr_rate: {train_ctcvr_rate}")
    print(f"derived_train_cvr_click_rate: {train_cvr_click_rate}")
    print(f"used_train_cvr_as_ctcvr_fallback: {used_cvr_fallback}")
    print()

    print("=== Computed Head Priors ===")
    for task in ("ctr", "cvr", "ctcvr"):
        print(f"{task}: {head_priors.get(task)}")
    print()

    model = build_model(cfg, meta=metadata)
    model.eval()

    print("=== Head Bias Check (bias vs logit(prior)) ===")
    for task in ("ctr", "cvr", "ctcvr"):
        bias = _get_head_bias(model, task)
        prior = head_priors.get(task)
        if bias is None:
            print(f"{task}: bias=<missing>")
            continue
        if prior is None:
            print(f"{task}: bias={bias:.8f}, prior=<missing>")
            continue
        expected = _logit(prior)
        print(
            f"{task}: bias={bias:.8f}, logit(prior)={expected:.8f}, delta={bias - expected:+.3e}"
        )
    print()

    optim_bundle = build_optimizer_bundle(cfg, model, scaler=None)
    dense_opt = optim_bundle.dense_opt

    print("=== Dense Optimizer Param Groups ===")
    print(f"dense_param_groups: {len(dense_opt.param_groups)}")
    for idx, group in enumerate(dense_opt.param_groups):
        num_tensors = len(group["params"])
        num_elems = sum(p.numel() for p in group["params"])
        print(
            f"group[{idx}]: lr={float(group.get('lr', 0.0)):.6g}, "
            f"weight_decay={float(group.get('weight_decay', 0.0)):.6g}, "
            f"num_tensors={num_tensors}, num_elements={num_elems}"
        )
    print()

    check_names = [
        "towers.ctr.out_proj.bias",
        "towers.ctr.out_proj.weight",
        "towers.cvr.out_proj.bias",
        "towers.cvr.out_proj.weight",
        "towers.ctcvr.out_proj.bias",
        "towers.ctcvr.out_proj.weight",
    ]
    ln_bias_name = _find_layernorm_bias_name(model)
    if ln_bias_name:
        check_names.append(ln_bias_name)

    print("=== Key Parameter Group Assignment ===")
    for name in check_names:
        info = _find_param_group_info(model, dense_opt, name)
        if info is None:
            print(f"{name}: <missing or not in dense optimizer>")
            continue
        print(
            f"{name}: group={info['group_idx']}, wd={info['weight_decay']:.6g}, lr={info['lr']:.6g}"
        )


if __name__ == "__main__":
    main()
