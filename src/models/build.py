from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Dict, List

import torch.nn as nn

from src.models.backbones.deepfm import DeepFMBackbone
from src.models.mtl.shared_bottom import SharedBottom
from src.models.mtl.mmoe import MMoE
from src.models.mtl.ple import PLE
from src.utils.config import load_yaml

try:
    from src.utils.feature_meta import build_model_feature_meta
except ImportError:  # pragma: no cover - fallback if module missing
    build_model_feature_meta = None


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


def _load_label_priors(cfg: Dict[str, Any]) -> Dict[str, float]:
    """
    Read train split positive rates from metadata.json for bias init.
    Falls back to empty dict if missing.
    """
    try:
        data_cfg = cfg.get("data", {})
        metadata_path = Path(data_cfg["metadata_path"])
        with metadata_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        train_stats = meta.get("split_stats", {}).get("train", {})
        priors = {}
        if "ctr" in train_stats:
            priors["ctr"] = float(train_stats["ctr"])
        if "cvr" in train_stats:
            priors["cvr"] = float(train_stats["cvr"])
        return priors
    except Exception:
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
    name = model_cfg.get("name") or "deepfm_shared_bottom"
    mtl = str(model_cfg.get("mtl", "sharedbottom")).lower()

    feature_meta = _resolve_feature_meta(cfg)
    backbone = _build_backbone(cfg, feature_meta)
    label_priors = _load_label_priors(cfg)

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


__all__ = ["build_model"]
