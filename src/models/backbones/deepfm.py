from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from src.models.backbones.layers import FeatureEmbedding, MLP


class DeepFMBackbone(nn.Module):
    """
    DeepFM backbone that can operate in two modes:
    - legacy "pseudo DeepFM": concat(deep_h, fm_vec) -> out_proj -> task heads (existing behaviour)
    - classic DeepFM baseline: expose wide_logit, fm_logit (eq-weight sum), deep_h for explicit logit decomposition
    """

    def __init__(
        self,
        feature_meta: Dict[str, Dict[str, Any]],
        deep_hidden_dims: List[int],
        deep_dropout: float,
        deep_activation: str,
        deep_use_bn: bool,
        fm_enabled: bool = True,
        fm_projection_dim: Optional[int] = None,
        out_dim: int = 128,
        use_legacy_pseudo_deepfm: bool = True,
        return_logit_parts: bool = False,
        sparse_grad: bool = False,
    ):
        super().__init__()
        self.feature_meta = feature_meta
        self.feat_emb = FeatureEmbedding(feature_meta, sparse_grad=sparse_grad)
        self.field_names_sorted: List[str] = sorted(feature_meta.keys())
        # Debug flags (set externally, e.g., in Trainer) – no effect on forward outputs.
        self.debug_linear: bool = False
        self.debug_fm: bool = False
        self.last_linear_debug: Optional[Dict[str, Any]] = None
        self.last_fm_debug: Optional[Dict[str, Any]] = None

        # Deep component
        self.deep_mlp = MLP(
            input_dim=self.feat_emb.concat_dim,
            hidden_dims=deep_hidden_dims,
            activation=deep_activation,
            dropout=deep_dropout,
            use_bn=deep_use_bn,
        )

        # FM component config
        self.fm_enabled = bool(fm_enabled)
        self.use_projection = False
        self.fm_dim = 0
        self.fm_proj: Optional[nn.ModuleDict] = None

        emb_dims = {base: int(meta["embedding_dim"]) for base, meta in feature_meta.items()}
        dim_set = set(emb_dims.values())
        projection_dim = fm_projection_dim

        if self.fm_enabled:
            if projection_dim is not None:
                self.use_projection = True
                self.fm_dim = int(projection_dim)
                self.fm_proj = nn.ModuleDict(
                    {
                        base: nn.Linear(dim, self.fm_dim, bias=False)
                        for base, dim in emb_dims.items()
                    }
                )
            elif len(dim_set) == 1:
                self.fm_dim = int(next(iter(dim_set)))
            else:
                self.fm_enabled = False
                self.fm_dim = 0
                print(
                    "DeepFMBackbone: FM disabled because embedding dims differ and fm_projection_dim is None."
                )

        final_in_dim = self.deep_mlp.output_dim + (self.fm_dim if self.fm_enabled else 0)
        self.out_proj = nn.Linear(final_in_dim, out_dim)
        self.out_dim = out_dim
        self.deep_out_dim = self.deep_mlp.output_dim

        # Linear (first-order) component: true sparse linear via EmbeddingBag
        self.linear_bags = nn.ModuleDict(
            {
                base: nn.EmbeddingBag(
                    num_embeddings=int(meta["num_embeddings"]),
                    embedding_dim=1,
                    # Sum + manual length normalization (to keep per_sample_weights supported).
                    mode="sum",
                    include_last_offset=False,
                )
                for base, meta in feature_meta.items()
            }
        )
        # Tighter init to keep initial logits in a learnable range.
        for emb in self.linear_bags.values():
            nn.init.normal_(emb.weight, mean=0.0, std=0.01)

        self.use_legacy_pseudo_deepfm = bool(use_legacy_pseudo_deepfm)
        self.return_logit_parts = bool(return_logit_parts)

    def _compute_fm(self, embedding_out: Dict[str, Any]) -> torch.Tensor:
        field_names = embedding_out["field_names"]
        emb_dict = embedding_out["emb_dict"]

        if self.use_projection and self.fm_proj is not None:
            proj_list = [self.fm_proj[base](emb_dict[base]) for base in field_names]
            fm_stack = torch.stack(proj_list, dim=1)
        else:
            fm_stack = embedding_out.get("emb_stack")
            if fm_stack is None:
                raise RuntimeError("FM stack is None while FM is enabled without projection.")

        sum_emb = fm_stack.sum(dim=1)
        sum_square = sum_emb * sum_emb
        square_sum = (fm_stack * fm_stack).sum(dim=1)
        fm_vec = 0.5 * (sum_square - square_sum)
        # Normalize by field count to prevent explosion when there are many features.
        num_fields = float(fm_stack.shape[1]) if fm_stack is not None else float(len(field_names))
        if num_fields > 0:
            fm_vec = fm_vec / num_fields
        return fm_vec

    def _compute_linear(self, features: Dict[str, Any]) -> torch.Tensor:
        field_names: List[str] = features["field_names"]
        fields = features["fields"]

        # Basic presence/dtype assertions per field to avoid silent misuse.
        for base in field_names:
            if base not in fields:
                raise KeyError(f"Feature {base} missing in features['fields'].")
            if base not in self.linear_bags:
                raise KeyError(f"Linear embedding for {base} not initialized.")

        # Batch size inferred from offsets (guaranteed include_last_offset=False so len == B)
        first_base = field_names[0]
        B = int(fields[first_base]["offsets"].shape[0])

        device = self.linear_bags[first_base].weight.device
        logit = torch.zeros((B, 1), device=device, dtype=self.linear_bags[first_base].weight.dtype)
        debug_stats: Dict[str, Any] = {} if self.debug_linear else None

        for base in field_names:
            fd = fields[base]
            idx = fd["indices"]
            offsets = fd["offsets"]
            wts = fd.get("weights")

            assert idx.dtype == torch.int64, f"{base}: indices dtype must be int64"
            assert offsets.dtype == torch.int64, f"{base}: offsets dtype must be int64"
            if wts is not None:
                assert wts.dtype == torch.float32, f"{base}: weights dtype must be float32"
                if wts.device != device:
                    wts = wts.to(device, non_blocking=True)

            if int(offsets.shape[0]) != B:
                raise ValueError(f"{base}: offsets length {offsets.shape[0]} != batch size {B}")

            if idx.device != device:
                idx = idx.to(device, non_blocking=True)
            if offsets.device != device:
                offsets = offsets.to(device, non_blocking=True)

            bag_out = self.linear_bags[base](idx, offsets, per_sample_weights=wts)
            # Normalize by bag length to avoid length-driven magnitude explosion while keeping mode='sum' for weights support.
            if offsets.numel() > 0:
                lengths = torch.empty_like(offsets, dtype=bag_out.dtype, device=device)
                if offsets.numel() > 1:
                    lengths[:-1] = offsets[1:] - offsets[:-1]
                lengths[-1] = idx.numel() - offsets[-1]
                lengths = lengths.clamp_min(1)
                bag_out = bag_out / lengths.unsqueeze(-1)
            logit = logit + bag_out

            if debug_stats is not None:
                flat = bag_out.detach().view(-1)
                l_flat = lengths.detach().view(-1)
                stat = {
                    "abs_max": float(flat.abs().max().item()) if flat.numel() else 0.0,
                    "p99": float(torch.quantile(flat, torch.tensor(0.99, device=flat.device)).item()) if flat.numel() else 0.0,
                    "mean": float(flat.mean().item()) if flat.numel() else 0.0,
                    "std": float(flat.std().item()) if flat.numel() else 0.0,
                    "len_p90": float(torch.quantile(l_flat, torch.tensor(0.90, device=l_flat.device)).item()) if l_flat.numel() else 0.0,
                    "len_p99": float(torch.quantile(l_flat, torch.tensor(0.99, device=l_flat.device)).item()) if l_flat.numel() else 0.0,
                    "len_max": float(l_flat.max().item()) if l_flat.numel() else 0.0,
                }
                if wts is not None and wts.numel() > 0:
                    w_flat = wts.detach().view(-1)
                    stat["w_abs_max"] = float(w_flat.abs().max().item())
                    stat["w_p99"] = float(torch.quantile(w_flat, torch.tensor(0.99, device=w_flat.device)).item())
                    stat["w_p999"] = float(torch.quantile(w_flat, torch.tensor(0.999, device=w_flat.device)).item())
                debug_stats[base] = stat

        if debug_stats is not None:
            self.last_linear_debug = debug_stats

        return logit

    def forward(
        self,
        features: Dict[str, Any],
        dense_x: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        if dense_x is not None:
            raise ValueError("dense_x is not supported in DeepFMBackbone backbone.")

        emb_out = self.feat_emb(features)
        deep_h = self.deep_mlp(emb_out["emb_concat"])

        # ----- Legacy pseudo-DeepFM path (existing behaviour) -----
        if self.use_legacy_pseudo_deepfm:
            parts = [deep_h]
            if self.fm_enabled:
                fm_vec = self._compute_fm(emb_out)
                parts.append(fm_vec)
            else:
                fm_vec = None

            h = torch.cat(parts, dim=1)
            h_out = self.out_proj(h)

            if not return_aux:
                return h_out

            linear_logit = self._compute_linear(features)
            out = {"h": h_out, "logit_linear": linear_logit, "deep_h": deep_h}
            if fm_vec is not None:
                out["fm_vec"] = fm_vec
            
            # ===== 新增：为 MMoEInputComposer 提供 embedding 聚合表示 =====
            emb_stack = emb_out.get("emb_stack")  # [B, F, D] or None
            out["emb_concat"] = emb_out["emb_concat"]  # 始终可用
            if emb_stack is not None:
                out["sum_emb"] = emb_stack.sum(dim=1)   # [B, D]
                out["mean_emb"] = emb_stack.mean(dim=1)  # [B, D]
            else:
                out["sum_emb"] = None
                out["mean_emb"] = None
            
            # legacy mode keeps behaviour; decomposition handled upstream if requested
            return out

        # ----- Classic DeepFM baseline path -----
        fm_vec = self._compute_fm(emb_out) if self.fm_enabled else None
        fm_logit = None
        if fm_vec is not None:
            # Expect shape [B, D]; sum over embedding dim to get scalar fm logit
            fm_logit = fm_vec.sum(dim=1, keepdim=True)
        if self.debug_fm and fm_vec is not None:
            flat = fm_logit.detach().view(-1) if fm_logit is not None else fm_vec.detach().view(-1)
            if flat.numel() > 0:
                qs = torch.quantile(flat, torch.tensor([0.99, 0.999], device=flat.device))
                self.last_fm_debug = {
                    "abs_max": float(flat.abs().max().item()),
                    "p99": float(qs[0].item()),
                    "p999": float(qs[1].item()),
                    "mean": float(flat.mean().item()),
                    "std": float(flat.std().item()),
                }

        wide_logit = self._compute_linear(features)

        if not return_aux:
            return deep_h

        out: Dict[str, torch.Tensor | None] = {
            "h": deep_h,  # keep key for compatibility; equals deep_h here
            "deep_h": deep_h,
            "logit_linear": wide_logit,  # for downstream addition
            "wide_logit": wide_logit,
            "fm_logit": fm_logit,
        }
        if fm_vec is not None:
            out["fm_vec"] = fm_vec
        
        # ===== 新增：为 MMoEInputComposer 提供 embedding 聚合表示 =====
        # emb_stack: [B, F, D]（仅当所有 embedding 维度相同时可用）
        # sum_emb: [B, D]（embedding sum pooling）
        # mean_emb: [B, D]（embedding mean pooling）
        # emb_concat: [B, F*D]（所有 embedding 拼接）
        emb_stack = emb_out.get("emb_stack")  # [B, F, D] or None
        out["emb_concat"] = emb_out["emb_concat"]  # 始终可用
        
        if emb_stack is not None:
            # 所有 embedding 维度相同，可以做 sum/mean
            out["sum_emb"] = emb_stack.sum(dim=1)   # [B, D]
            out["mean_emb"] = emb_stack.mean(dim=1)  # [B, D]
        else:
            # 各 embedding 维度不同，无法直接 sum/mean
            # 设为 None，composer 会在需要时报错
            out["sum_emb"] = None
            out["mean_emb"] = None
        
        return out


__all__ = ["DeepFMBackbone"]


def _smoke_test():
    torch.manual_seed(0)
    feature_meta = {
        "f0": {
            "num_embeddings": 10,
            "embedding_dim": 4,
            "mode": "sum",
            "is_multi_hot": False,
            "use_value": False,
        },
        "f1": {
            "num_embeddings": 12,
            "embedding_dim": 4,
            "mode": "sum",
            "is_multi_hot": True,
            "use_value": True,
        },
    }

    B = 3
    features = {
        "field_names": ["f0", "f1"],
        "fields": {
            "f0": {
                "indices": torch.tensor([1, 2, 3], dtype=torch.int64),
                "offsets": torch.tensor([0, 1, 2], dtype=torch.int64),
                "weights": None,
            },
            "f1": {
                "indices": torch.tensor([0, 1, 2, 3], dtype=torch.int64),
                "offsets": torch.tensor([0, 2, 4], dtype=torch.int64),
                "weights": torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32),
            },
        },
    }

    model = DeepFMBackbone(
        feature_meta=feature_meta,
        deep_hidden_dims=[8],
        deep_dropout=0.0,
        deep_activation="relu",
        deep_use_bn=False,
        fm_enabled=True,
        fm_projection_dim=None,
        out_dim=16,
    )

    out = model(features, return_aux=True)
    assert out["h"].shape == (B, 16)
    assert out["logit_linear"].shape == (B, 1)
    assert torch.isfinite(out["h"]).all()
    assert torch.isfinite(out["logit_linear"]).all()
    print("Smoke test passed: h", out["h"].shape, "logit_linear", out["logit_linear"].shape)


if __name__ == "__main__":  # pragma: no cover
    # Allow running as a script: add repo root to sys.path when __package__ is None.
    if __package__ is None:  # type: ignore
        import sys
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[3]
        sys.path.append(str(repo_root))
    _smoke_test()
