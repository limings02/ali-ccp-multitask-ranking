from __future__ import annotations

from pathlib import Path
import json
import logging
import math
from typing import Any, Dict

import torch.nn as nn

from src.models.backbones.deepfm import DeepFMBackbone
from src.models.mtl.shared_bottom import SharedBottom
from src.models.mtl.mmoe import MMoE
from src.models.mtl.ple import PLE

try:
    from src.utils.feature_meta import build_model_feature_meta
except ImportError:  # pragma: no cover - fallback if module missing
    build_model_feature_meta = None


logger = logging.getLogger(__name__)
_PRIOR_EPS = 1e-8


def _resolve_feature_meta(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Build merged model_feature_meta using metadata.json + embedding config.
    """
    data_cfg = cfg.get("data", {})
    embedding_cfg = cfg.get("embedding", {})
    metadata_path = Path(data_cfg["metadata_path"])

    if build_model_feature_meta is None:
        raise ImportError("build_model_feature_meta not available; please ensure src.utils.feature_meta exists.")
    return build_model_feature_meta(metadata_path, embedding_cfg)


def _clip_prior(prob: float, eps: float = _PRIOR_EPS) -> float:
    prob = float(prob)
    if not math.isfinite(prob):
        raise ValueError(f"Invalid prior value: {prob}")
    return min(max(prob, eps), 1.0 - eps)


def _safe_read_rate(train_stats: Dict[str, Any], key: str) -> float | None:
    value = train_stats.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        logger.warning("compute_head_priors: split_stats.train.%s=%r is not a valid float; ignored.", key, value)
        return None


def compute_head_priors(cfg: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute head priors (probabilities) for output-layer bias initialization.

    ESMM semantics:
      - ctr head predicts P(click)
      - ctcvr head predicts P(click & conv)
      - cvr head predicts P(conv | click), so prior must be P(click & conv) / P(click)

    Metadata compatibility:
      - Prefer split_stats.train.ctcvr as exposure-level P(click & conv)
      - Fallback to split_stats.train.cvr as legacy exposure-level ctcvr alias
    """
    train_stats = ((metadata or {}).get("split_stats") or {}).get("train", {}) or {}
    use_esmm = bool(cfg.get("use_esmm", False))

    train_ctr_rate = _safe_read_rate(train_stats, "ctr")
    train_cvr_rate = _safe_read_rate(train_stats, "cvr")
    train_ctcvr_rate = _safe_read_rate(train_stats, "ctcvr")
    priors: Dict[str, float] = {}

    if use_esmm:
        if train_ctcvr_rate is None and train_cvr_rate is not None:
            train_ctcvr_rate = train_cvr_rate
            logger.warning(
                "compute_head_priors: split_stats.train.ctcvr missing; fallback to split_stats.train.cvr "
                "as exposure-level P(click&conv) for ESMM backward compatibility."
            )

        if train_ctr_rate is not None:
            priors["ctr"] = _clip_prior(train_ctr_rate)
        if train_ctcvr_rate is not None:
            priors["ctcvr"] = _clip_prior(train_ctcvr_rate)

        if train_ctr_rate is not None and train_ctcvr_rate is not None:
            # ESMM: CVR head target is P(conv|click), not exposure-level P(click&conv).
            cvr_click_rate = train_ctcvr_rate / max(train_ctr_rate, _PRIOR_EPS)
            priors["cvr"] = _clip_prior(cvr_click_rate)
        elif train_ctcvr_rate is not None:
            # Last-resort fallback when ctr rate is unavailable.
            priors["cvr"] = _clip_prior(train_ctcvr_rate)
            logger.warning(
                "compute_head_priors: split_stats.train.ctr missing; cannot derive P(conv|click). "
                "Fallback sets cvr prior to exposure-level P(click&conv)."
            )
        return priors

    if train_ctr_rate is not None:
        priors["ctr"] = _clip_prior(train_ctr_rate)
    if train_cvr_rate is not None:
        priors["cvr"] = _clip_prior(train_cvr_rate)
    if train_ctcvr_rate is not None:
        priors["ctcvr"] = _clip_prior(train_ctcvr_rate)
    return priors


def _load_metadata(cfg: Dict[str, Any]) -> Dict[str, Any]:
    data_cfg = cfg.get("data", {})
    metadata_path_val = data_cfg.get("metadata_path")
    if not metadata_path_val:
        return {}

    metadata_path = Path(metadata_path_val)
    try:
        with metadata_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Failed to load metadata for priors from %s: %s", metadata_path, exc)
        return {}


def _build_backbone(cfg: Dict[str, Any], feature_meta: Dict[str, Any]) -> nn.Module:
    model_cfg = cfg.get("model", {})
    # backward-compatible: model/backbone nesting
    backbone_cfg = model_cfg.get("backbone", model_cfg)
    embedding_cfg = cfg.get("embedding", {})

    def _pick(key: str, default=None):
        if key in backbone_cfg and backbone_cfg.get(key) is not None:
            return backbone_cfg.get(key)
        if key in model_cfg and model_cfg.get(key) is not None:
            return model_cfg.get(key)
        return default

    use_legacy = bool(_pick("use_legacy_pseudo_deepfm", True))
    return_parts = bool(_pick("return_logit_parts", False))
    sparse_grad = bool(embedding_cfg.get("sparse_grad", False))

    return DeepFMBackbone(
        feature_meta=feature_meta,
        deep_hidden_dims=list(_pick("deep_hidden_dims", [])),
        deep_dropout=float(_pick("deep_dropout", 0.0)),
        deep_activation=str(_pick("deep_activation", "relu")),
        deep_use_bn=bool(_pick("deep_use_bn", False)),
        fm_enabled=bool(_pick("fm_enabled", True)),
        fm_projection_dim=(
            None
            if _pick("fm_projection_dim") is None
            else int(_pick("fm_projection_dim"))
        ),
        out_dim=int(_pick("out_dim", 128)),
        use_legacy_pseudo_deepfm=use_legacy,
        return_logit_parts=return_parts,
        sparse_grad=sparse_grad,
    )


def build_model(cfg: Dict[str, Any], feature_map: Dict[str, Any] | None = None, meta: Dict[str, Any] | None = None) -> nn.Module:
    """
    Assemble model according to cfg. Supports SharedBottom (default) and MMoE.
    """
    model_cfg = cfg.get("model", {})
    enabled_heads = model_cfg.get("enabled_heads") or ["ctr", "cvr"]
    mtl = str(model_cfg.get("mtl", "sharedbottom")).lower()

    feature_meta = _resolve_feature_meta(cfg)
    backbone = _build_backbone(cfg, feature_meta)
    metadata = meta if isinstance(meta, dict) and meta else _load_metadata(cfg)
    label_priors = compute_head_priors(cfg, metadata)

    head_cfg = model_cfg.get("heads", {})
    head_cfg.setdefault("tasks", model_cfg.get("tasks", ["ctr", "cvr"]))
    head_cfg.setdefault("default", {})
    head_cfg["default"].setdefault("mlp_dims", model_cfg.get("tower_hidden_dims", []))
    head_cfg["default"].setdefault("dropout", model_cfg.get("head_dropout", 0.0))
    head_cfg["default"].setdefault("use_bn", model_cfg.get("head_use_bn", False))
    head_cfg["default"].setdefault("activation", model_cfg.get("head_activation", model_cfg.get("deep_activation", "relu")))

    per_head_add = model_cfg.get("backbone", {}).get("per_head_add") or model_cfg.get("per_head_add") or {}
    use_legacy = bool(model_cfg.get("backbone", {}).get("use_legacy_pseudo_deepfm", model_cfg.get("use_legacy_pseudo_deepfm", True)))
    return_parts = bool(model_cfg.get("backbone", {}).get("return_logit_parts", model_cfg.get("return_logit_parts", False)))

    if mtl in {"sharedbottom", "shared_bottom"}:
        return SharedBottom(
            backbone=backbone,
            head_cfg=head_cfg,
            enabled_heads=enabled_heads,
            use_legacy_pseudo_deepfm=use_legacy,
            return_logit_parts=return_parts,
            per_head_add=per_head_add,
            head_priors=label_priors,
        )

    if mtl == "mmoe":
        mmoe_cfg = model_cfg.get("mmoe", {})
        log_gates = bool(mmoe_cfg.get("log_gates", False))
        # ESMM residual head 配置: 从 esmm.residual 读取
        esmm_residual_cfg = cfg.get("esmm", {}).get("residual", {})
        return MMoE(
            backbone=backbone,
            head_cfg=head_cfg,
            mmoe_cfg=mmoe_cfg,
            enabled_heads=enabled_heads,
            use_legacy_pseudo_deepfm=use_legacy,
            return_logit_parts=return_parts,
            per_head_add=per_head_add,
            head_priors=label_priors,
            log_gates=log_gates,
            esmm_residual_cfg=esmm_residual_cfg,
        )

    # ========== PLE-Lite 分支 ==========
    # 与 MMoE 同级，作为对照组实验模型
    if mtl == "ple":
        ple_cfg = model_cfg.get("ple", {})
        log_gates = bool(ple_cfg.get("log_gates", False))
        # ESMM residual head 配置: 从 esmm.residual 读取
        esmm_residual_cfg = cfg.get("esmm", {}).get("residual", {})
        return PLE(
            backbone=backbone,
            head_cfg=head_cfg,
            ple_cfg=ple_cfg,
            enabled_heads=enabled_heads,
            use_legacy_pseudo_deepfm=use_legacy,
            return_logit_parts=return_parts,
            per_head_add=per_head_add,
            head_priors=label_priors,
            log_gates=log_gates,
            esmm_residual_cfg=esmm_residual_cfg,
        )

    raise ValueError(f"Unsupported model.mtl '{mtl}'. Expected 'sharedbottom', 'mmoe', or 'ple'.")


__all__ = ["build_model", "compute_head_priors"]
