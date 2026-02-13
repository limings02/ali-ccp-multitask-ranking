from __future__ import annotations

import json
import mmap
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def load_vectorized_split_metadata(split_dir: str | Path) -> Dict[str, Any]:
    split_path = Path(split_dir)
    meta_path = split_path / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Vectorized metadata not found: {meta_path}")
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"Failed to parse vectorized metadata: {meta_path}") from exc


class VectorizedRecDataset(Dataset):
    """
    Map-style dataset for vectorized storage.

    __getitem__ intentionally returns only row index to keep per-sample cost minimal.
    """

    def __init__(self, split_dir: str | Path):
        self.split_dir = Path(split_dir)
        self.meta = load_vectorized_split_metadata(self.split_dir)
        self.num_rows = int(self.meta["num_rows"])

    def __len__(self) -> int:
        return self.num_rows

    def __getitem__(self, idx: int) -> int:
        if idx < 0 or idx >= self.num_rows:
            raise IndexError(idx)
        return int(idx)

    def __getitems__(self, indices: List[int]) -> List[int]:
        """
        Batched fast path used by torch DataLoader map fetcher.
        Avoids per-sample __getitem__ Python dispatch for auto-collation.
        """
        if not indices:
            return []
        lo = min(indices)
        hi = max(indices)
        if lo < 0 or hi >= self.num_rows:
            raise IndexError(f"indices out of range: min={lo}, max={hi}, size={self.num_rows}")
        return [int(i) for i in indices]


class VectorizedBatchCollator:
    """
    Tensorized collator for vectorized ragged CSR features.

    For each field we only loop over fields, never over samples.
    """

    def __init__(self, split_dir: str | Path, include_entity_id: bool = True):
        self.split_dir = Path(split_dir)
        self.meta = load_vectorized_split_metadata(self.split_dir)
        self.field_names: List[str] = list(self.meta["field_names"])
        self.include_entity_id = bool(include_entity_id)
        self._any_use_value = any(
            bool((self.meta.get("fields", {}) or {}).get(name, {}).get("use_value", False))
            for name in self.field_names
        )
        self._opened = False

        self._y_ctr: Optional[np.ndarray] = None
        self._y_cvr: Optional[np.ndarray] = None
        self._click_mask: Optional[np.ndarray] = None
        self._row_id: Optional[np.ndarray] = None

        self._field_offsets_t: Dict[str, torch.Tensor] = {}
        self._field_indices_t: Dict[str, torch.Tensor] = {}
        self._field_weights_t: Dict[str, Optional[torch.Tensor]] = {}

        self._entity_offsets: Optional[np.ndarray] = None
        self._entity_file = None
        self._entity_mmap: Optional[mmap.mmap] = None

    def _np_load(self, name: str) -> np.ndarray:
        return np.load(self.split_dir / name, mmap_mode="r+")

    def _ensure_open(self) -> None:
        if self._opened:
            return

        self._y_ctr = self._np_load("y_ctr.npy")
        self._y_cvr = self._np_load("y_cvr.npy")
        self._click_mask = self._np_load("click_mask.npy")
        self._row_id = self._np_load("row_id.npy")
        if self.include_entity_id:
            self._entity_offsets = self._np_load("entity_id_offsets.npy")
            entity_path = self.split_dir / "entity_id_data.bin"
            self._entity_file = entity_path.open("rb")
            self._entity_mmap = mmap.mmap(self._entity_file.fileno(), 0, access=mmap.ACCESS_READ)

        field_meta = self.meta["fields"]
        for field_name in self.field_names:
            fmeta = field_meta[field_name]
            offsets = np.load(self.split_dir / fmeta["offsets_file"], mmap_mode="r+")
            indices = np.load(self.split_dir / fmeta["indices_file"], mmap_mode="r+")
            self._field_offsets_t[field_name] = torch.from_numpy(offsets)
            self._field_indices_t[field_name] = torch.from_numpy(indices)

            weights_file = fmeta.get("weights_file")
            if weights_file:
                weights = np.load(self.split_dir / weights_file, mmap_mode="r+")
                self._field_weights_t[field_name] = torch.from_numpy(weights)
            else:
                self._field_weights_t[field_name] = None

        self._opened = True

    def close(self) -> None:
        if self._entity_mmap is not None:
            self._entity_mmap.close()
            self._entity_mmap = None
        if self._entity_file is not None:
            self._entity_file.close()
            self._entity_file = None
        self._opened = False

    def __del__(self) -> None:  # pragma: no cover - cleanup guard
        try:
            self.close()
        except Exception:
            pass

    def _gather_entity_ids(self, batch_idx_np: np.ndarray) -> List[Any]:
        if not self.include_entity_id:
            return []
        assert self._entity_offsets is not None
        assert self._entity_mmap is not None
        starts = self._entity_offsets[batch_idx_np]
        ends = self._entity_offsets[batch_idx_np + 1]
        out: List[Any] = []
        for start, end in zip(starts.tolist(), ends.tolist()):
            payload = self._entity_mmap[int(start): int(end)]
            out.append(json.loads(payload.decode("utf-8")))
        return out

    @staticmethod
    def _is_contiguous_range(batch_idx_np: np.ndarray) -> bool:
        if batch_idx_np.ndim != 1 or batch_idx_np.size == 0:
            return False
        if batch_idx_np.size == 1:
            return True
        # Fast path for sequential sampler batches: [s, s+1, ..., s+B-1]
        return bool(np.all(np.diff(batch_idx_np) == 1))

    def __call__(self, batch_idx_list: List[int]):
        if not batch_idx_list:
            raise ValueError("Empty batch is not allowed.")
        self._ensure_open()

        assert self._y_ctr is not None
        assert self._y_cvr is not None
        assert self._click_mask is not None
        assert self._row_id is not None

        t0 = time.perf_counter()
        batch_idx_np = np.asarray(batch_idx_list, dtype=np.int64)

        # Keep ESMM funnel guards identical to existing parquet collator.
        y_ctr_raw = self._y_ctr[batch_idx_np].astype(np.float32, copy=False)
        y_cvr_raw = self._y_cvr[batch_idx_np].astype(np.float32, copy=False)
        valid_mask = ~((y_ctr_raw <= 0.0) & (y_cvr_raw > 0.0))
        if not np.any(valid_mask):
            raise ValueError("Batch empty after filtering funnel-inconsistent samples.")
        if not np.all(valid_mask):
            batch_idx_np = batch_idx_np[valid_mask]
            y_ctr = y_ctr_raw[valid_mask]
            y_cvr = y_cvr_raw[valid_mask]
        else:
            y_ctr = y_ctr_raw
            y_cvr = y_cvr_raw

        y_cvr = y_cvr.copy()
        y_cvr[y_ctr <= 0.0] = 0.0

        click_mask = self._click_mask[batch_idx_np].astype(np.float32, copy=False)
        row_id = self._row_id[batch_idx_np].astype(np.int64, copy=False)
        y_ctcvr = np.logical_and(y_ctr > 0.5, y_cvr > 0.5).astype(np.float32, copy=False)

        labels = {
            "y_ctr": torch.from_numpy(y_ctr),
            "y_cvr": torch.from_numpy(y_cvr),
            "y_ctcvr": torch.from_numpy(y_ctcvr),
            "click_mask": torch.from_numpy(click_mask),
            "row_id": torch.from_numpy(row_id),
        }

        batch_idx = torch.from_numpy(batch_idx_np)
        B = int(batch_idx.shape[0])
        contiguous_batch = self._is_contiguous_range(batch_idx_np)
        base_row = int(batch_idx_np[0])
        fields_out: Dict[str, Dict[str, torch.Tensor | None]] = {}

        for field_name in self.field_names:
            offsets_all = self._field_offsets_t[field_name]
            indices_all = self._field_indices_t[field_name]
            weights_all = self._field_weights_t[field_name]

            if contiguous_batch:
                # CSR contiguous fast path: reuse a single offsets slice and one packed
                # payload slice per field; avoids repeat_interleave/index_select overhead.
                offsets_win = offsets_all[base_row : base_row + B + 1]
                packed_start = int(offsets_win[0].item())
                packed_end = int(offsets_win[-1].item())
                total = int(packed_end - packed_start)
                out_offsets = (offsets_win[:-1] - packed_start).to(torch.int64)

                if total > 0:
                    out_indices = indices_all[packed_start:packed_end]
                    if weights_all is not None:
                        out_weights = weights_all[packed_start:packed_end]
                    elif self._any_use_value:
                        out_weights = torch.ones(total, dtype=torch.float32)
                    else:
                        out_weights = None
                else:
                    out_indices = torch.empty(0, dtype=torch.int64)
                    if weights_all is not None or self._any_use_value:
                        out_weights = torch.empty(0, dtype=torch.float32)
                    else:
                        out_weights = None
            else:
                # Generic gather for non-contiguous index lists.
                start = offsets_all.index_select(0, batch_idx)
                end = offsets_all.index_select(0, batch_idx + 1)
                lengths = end - start
                out_offsets = torch.zeros(B, dtype=torch.int64)
                if B > 1:
                    out_offsets[1:] = torch.cumsum(lengths[:-1], dim=0)

                total = int(lengths.sum().item())
                if total > 0:
                    start_rep = torch.repeat_interleave(start, lengths)
                    base_rep = torch.repeat_interleave(out_offsets, lengths)
                    intra = torch.arange(total, dtype=torch.int64)
                    global_pos = start_rep + (intra - base_rep)
                    out_indices = indices_all.index_select(0, global_pos)
                    if weights_all is not None:
                        out_weights = weights_all.index_select(0, global_pos)
                    elif self._any_use_value:
                        out_weights = torch.ones(total, dtype=torch.float32)
                    else:
                        out_weights = None
                else:
                    out_indices = torch.empty(0, dtype=torch.int64)
                    if weights_all is not None or self._any_use_value:
                        out_weights = torch.empty(0, dtype=torch.float32)
                    else:
                        out_weights = None

            fields_out[field_name] = {
                "indices": out_indices,
                "offsets": out_offsets,
                "weights": out_weights,
            }

        meta = {"_perf": {"collate_ms": (time.perf_counter() - t0) * 1000.0}}
        if self.include_entity_id:
            meta["entity_id"] = self._gather_entity_ids(batch_idx_np)
        features = {"field_names": self.field_names, "fields": fields_out}
        return labels, features, meta


__all__ = ["VectorizedRecDataset", "VectorizedBatchCollator", "load_vectorized_split_metadata"]
