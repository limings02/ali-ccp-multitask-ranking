from __future__ import annotations

from typing import Dict, List, Optional, Set

import torch
from torch import nn

from src.models.heads import TaskHead


class SharedBottom(nn.Module):
    """
    Shared-bottom multi-task model that wraps a backbone (e.g., DeepFM)
    and attaches per-task towers while preserving the wide (linear) term.
    """

    def __init__(
        self,
        backbone: nn.Module,
        head_cfg: Dict,
        enabled_heads: Optional[List[str]] = None,
        use_legacy_pseudo_deepfm: bool = True,
        return_logit_parts: bool = False,
        per_head_add: Optional[Dict[str, Dict[str, bool]]] = None,
        head_priors: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.use_legacy_pseudo_deepfm = bool(use_legacy_pseudo_deepfm)
        self.return_logit_parts = bool(return_logit_parts)
        self.per_head_add = per_head_add or {}
        self.head_priors = head_priors or {}
        # Debug flag (set externally) – only records stats, no forward change.
        self.debug_logit: bool = False
        self.last_ctr_debug: Optional[Dict[str, object]] = None

        tasks = head_cfg.get("tasks") or ["ctr", "cvr"]
        self.tasks: List[str] = tasks
        # Enabled heads are a subset (or equal) of all tasks; default to all tasks.
        enabled = enabled_heads or tasks
        self.enabled_heads: Set[str] = {h.lower() for h in enabled}

        self._head_cfg_resolved: Dict[str, Dict[str, object]] = {}

        def _pick_cfg(head_cfg: Dict, task: str, key: str, default):
            """
            Fallback only when value is missing or None (not any falsy).
            Priority: task-specific -> default block -> top-level -> default.
            """
            if isinstance(head_cfg.get(task), dict) and key in head_cfg[task] and head_cfg[task][key] is not None:
                return head_cfg[task][key]
            if isinstance(head_cfg.get("default"), dict) and key in head_cfg["default"] and head_cfg["default"][key] is not None:
                return head_cfg["default"][key]
            if key in head_cfg and head_cfg[key] is not None:
                return head_cfg[key]
            return default

        in_dim = getattr(backbone, "out_dim", head_cfg.get("in_dim"))
        # Classic mode uses deep_h; prefer deep_out_dim when available
        if not self.use_legacy_pseudo_deepfm and hasattr(backbone, "deep_out_dim"):
            in_dim = getattr(backbone, "deep_out_dim")
        if in_dim is None:
            raise ValueError("SharedBottom: backbone must expose out_dim or provide head_cfg['in_dim'].")

        # Normalize shared representation to keep tower inputs in a stable range.
        self.layernorm = nn.LayerNorm(in_dim)

        self.towers = nn.ModuleDict()
        for task in tasks:
            mlp_dims = list(_pick_cfg(head_cfg, task, "mlp_dims", []))
            dropout = float(_pick_cfg(head_cfg, task, "dropout", 0.0))
            use_bn = bool(_pick_cfg(head_cfg, task, "use_bn", False))
            activation = str(_pick_cfg(head_cfg, task, "activation", "relu"))

            self._head_cfg_resolved[task] = {
                "mlp_dims": mlp_dims,
                "dropout": dropout,
                "use_bn": use_bn,
                "activation": activation,
            }

            self.towers[task] = TaskHead(
                in_dim=in_dim,
                mlp_dims=mlp_dims,
                out=1,
                activation=activation,
                dropout=dropout,
                use_bn=use_bn,
            )
            # Bias init to prior logits to avoid saturated outputs at cold start.
            # Priority: user-provided priors -> defaults.
            prior_map = {"ctr": 0.07, "cvr": 0.01}
            prior = self.head_priors.get(task, prior_map.get(task, 0.5))
            prior = float(prior)
            # Clamp to keep logits finite even when prior=0/1.
            prior = min(max(prior, 1e-8), 1 - 1e-8)
            bias_init = float(torch.log(torch.tensor(prior / (1 - prior))))
            if self.towers[task].out_proj.bias is not None:
                nn.init.constant_(self.towers[task].out_proj.bias, bias_init)

    def _get_head_add_cfg(self, task: str) -> Dict[str, bool]:
        task = task.lower()
        cfg = self.per_head_add.get(task)
        if cfg is not None:
            return {
                "use_wide": bool(cfg.get("use_wide", False)),
                "use_fm": bool(cfg.get("use_fm", False)),
            }
        # Sensible defaults: CTR/CTCVR use wide+fm, CVR defaults off to avoid negative transfer
        if task == "cvr":
            return {"use_wide": False, "use_fm": False}
        return {"use_wide": True, "use_fm": True}

    def forward(self, features: Dict, dense_x: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: dict produced by dataloader (EmbeddingBag inputs).
            dense_x: currently unsupported and passed through to backbone for validation.
        Returns:
            Dict mapping task name -> logit tensor of shape [B].
        """
        out = self.backbone(features, dense_x=dense_x, return_aux=True)
        if "h" not in out or "logit_linear" not in out:
            raise KeyError("Backbone must return {'h', 'logit_linear'} when return_aux=True.")

        results: Dict[str, torch.Tensor] = {}
        logit_parts_decomposable = not self.use_legacy_pseudo_deepfm

        if self.use_legacy_pseudo_deepfm:
            # Original behaviour: head(h) + linear
            h = out["h"]
            h = self.layernorm(h)
            linear = out["logit_linear"]
            if linear.dim() == 2 and linear.size(-1) == 1:
                linear = linear.squeeze(-1)
            if linear.dim() != 1:
                raise ValueError(f"logit_linear must broadcast to [B]; got shape {tuple(linear.shape)}")

            for task, head in self.towers.items():
                if task not in self.enabled_heads:
                    continue
                task_logit = head(h) + linear
                results[task] = task_logit
                if self.return_logit_parts:
                    results[f"{task}_logit_parts"] = {
                        "wide": linear,
                        "fm": None,
                        "deep": None,
                        "total": task_logit,
                    }
                if self.debug_logit and task.lower() == "ctr":
                    with torch.no_grad():
                        flat = task_logit.detach().view(-1)
                        stats = None
                        if flat.numel() > 0:
                            qs = torch.quantile(flat, torch.tensor([0.99, 0.999], device=flat.device))
                            stats = {
                                "abs_max": float(flat.abs().max().item()),
                                "p99": float(qs[0].item()),
                                "p999": float(qs[1].item()),
                                "mean": float(flat.mean().item()),
                                "std": float(flat.std().item()),
                            }
                        self.last_ctr_debug = {
                            "total": stats,
                            "wide": None,
                            "fm": None,
                            "deep": None,
                        }
            results["logit_parts_decomposable"] = logit_parts_decomposable
            if "aux" in out:
                results["aux"] = out["aux"]
            return results

        # Classic explicit-sum path
        deep_h = out.get("deep_h")
        wide_logit = out.get("logit_linear")  # alias to wide
        fm_logit = out.get("fm_logit")

        if deep_h is None:
            raise KeyError("Classic mode requires 'deep_h' from backbone.")
        deep_h_norm = self.layernorm(deep_h)

        if wide_logit is not None and wide_logit.dim() == 2 and wide_logit.size(-1) == 1:
            wide_logit = wide_logit.squeeze(-1)
        if fm_logit is not None and fm_logit.dim() == 2 and fm_logit.size(-1) == 1:
            fm_logit = fm_logit.squeeze(-1)

        for task, head in self.towers.items():
            if task not in self.enabled_heads:
                continue
            add_cfg = self._get_head_add_cfg(task)
            wide_term = wide_logit if (add_cfg.get("use_wide", False) and wide_logit is not None) else 0.0
            fm_term = fm_logit if (add_cfg.get("use_fm", False) and fm_logit is not None) else 0.0
            task_deep_logit = head(deep_h_norm)
            task_logit = task_deep_logit + wide_term + fm_term
            results[task] = task_logit

            if self.return_logit_parts:
                # Ensure tensor types for consistency
                wide_comp = wide_term if torch.is_tensor(wide_term) else torch.tensor(wide_term, device=task_logit.device)
                fm_comp = fm_term if torch.is_tensor(fm_term) else torch.tensor(fm_term, device=task_logit.device)
                results[f"{task}_logit_parts"] = {
                    "wide": wide_comp,
                    "fm": fm_comp,
                    "deep": task_deep_logit,
                    "total": task_logit,
                }
            if self.debug_logit and task.lower() == "ctr":
                with torch.no_grad():
                    def _stat(t: torch.Tensor):
                        flat = t.detach().float().view(-1)
                        if flat.numel() == 0:
                            return None
                        qs = torch.quantile(flat, torch.tensor([0.99, 0.999], device=flat.device, dtype=flat.dtype))
                        return {
                            "abs_max": float(flat.abs().max().item()),
                            "p99": float(qs[0].item()),
                            "p999": float(qs[1].item()),
                            "mean": float(flat.mean().item()),
                            "std": float(flat.std().item()),
                        }

                    self.last_ctr_debug = {
                        "total": _stat(task_logit),
                        "wide": _stat(wide_term) if torch.is_tensor(wide_term) else None,
                        "fm": _stat(fm_term) if torch.is_tensor(fm_term) else None,
                        "deep": _stat(task_deep_logit),
                    }

        results["logit_parts_decomposable"] = logit_parts_decomposable
        if "aux" in out:
            results["aux"] = out["aux"]
        return results


__all__ = ["SharedBottom"]
