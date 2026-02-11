from __future__ import annotations

from typing import Any, Dict, List

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        activation: str = "relu",
        dropout: float = 0.0,
        use_bn: bool = False,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(dim))
            layers.append(self._make_activation(activation))
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = dim
        if not layers:
            layers.append(nn.Identity())
        self.model = nn.Sequential(*layers)
        self.output_dim = prev_dim

    @staticmethod
    def _make_activation(name: str) -> nn.Module:
        name = name.lower()
        if name == "relu":
            return nn.ReLU()
        if name == "gelu":
            return nn.GELU()
        if name == "tanh":
            return nn.Tanh()
        if name == "sigmoid":
            return nn.Sigmoid()
        raise ValueError(f"Unsupported activation: {name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class EmbeddingBagEncoder(nn.Module):
    """
    Thin wrapper around nn.EmbeddingBag with include_last_offset=False.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, mode: str = "sum", sparse: bool = False):
        super().__init__()
        self.sparse = bool(sparse)
        self.embedding = nn.EmbeddingBag(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            mode=mode,
            include_last_offset=False,
            sparse=self.sparse,
        )

    def forward(
        self,
        indices: torch.Tensor,
        offsets: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = self.embedding.weight.device
        idx = indices if indices.device == device else indices.to(device, non_blocking=True)
        off = offsets if offsets.device == device else offsets.to(device, non_blocking=True)
        wts = None
        if weights is not None:
            wts = weights if weights.device == device else weights.to(device, non_blocking=True)
        return self.embedding(idx, off, per_sample_weights=wts)


class FeatureEmbedding(nn.Module):
    """
    Encode per-field indices/offsets/weights into embeddings.
    """

    def __init__(self, feature_meta: Dict[str, Dict[str, Any]], sparse_grad: bool = False):
        super().__init__()
        self.feature_meta = feature_meta
        self.sparse_grad = bool(sparse_grad)
        self.embedders = nn.ModuleDict()
        self.embedding_dims: Dict[str, int] = {}

        for base, meta in feature_meta.items():
            num_embeddings = int(meta["num_embeddings"])
            emb_dim = int(meta["embedding_dim"])
            mode = str(meta.get("mode", "sum"))
            self.embedders[base] = EmbeddingBagEncoder(
                num_embeddings=num_embeddings,
                embedding_dim=emb_dim,
                mode=mode,
                sparse=self.sparse_grad,
            )
            self.embedding_dims[base] = emb_dim

        self.concat_dim = sum(self.embedding_dims.values())
        self.all_dims_equal = len(set(self.embedding_dims.values())) == 1

    def forward(self, features: Dict[str, Any]) -> Dict[str, Any]:
        field_names: List[str] = features["field_names"]
        fields = features["fields"]

        emb_list: List[torch.Tensor] = []
        emb_dict: Dict[str, torch.Tensor] = {}
        dims: List[int] = []

        for base in field_names:
            if base not in self.embedders:
                raise KeyError(f"Feature {base} missing in embedding table.")
            field_data = fields[base]
            offsets = field_data["offsets"]
            indices = field_data["indices"]
            lengths = torch.empty_like(offsets, dtype=torch.float32, device=indices.device)
            if offsets.numel() > 1:
                lengths[:-1] = offsets[1:] - offsets[:-1]
            lengths[-1] = indices.numel() - offsets[-1]
            lengths = lengths.clamp_min(1)
            emb = self.embedders[base](
                field_data["indices"],
                field_data["offsets"],
                field_data.get("weights"),
            )
            emb = emb / lengths.unsqueeze(-1)
            emb_dict[base] = emb
            emb_list.append(emb)
            dims.append(emb.shape[1])

        if not emb_list:
            raise ValueError("No embeddings produced; check feature inputs.")

        emb_concat = torch.cat(emb_list, dim=1)

        emb_stack = None
        if len(set(dims)) == 1:
            emb_stack = torch.stack(emb_list, dim=1)

        return {
            "emb_dict": emb_dict,
            "emb_concat": emb_concat,
            "emb_stack": emb_stack,
            "field_names": field_names,
        }


__all__ = ["MLP", "EmbeddingBagEncoder", "FeatureEmbedding"]
