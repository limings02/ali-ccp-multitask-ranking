"""
EmbeddingBag-friendly DataLoader utilities for processed parquet data.

Design goals
------------
- Reuse existing IterableDataset from `src.data.dataset` (no IO duplication).
- Produce per-field EmbeddingBag inputs `{indices, offsets, weights}` so model
  can hold one EmbeddingBag per feature without extra reshaping.
- Enforce deterministic base ordering to avoid silent feature shift:
    bases = sorted([k for k in row.keys() if k.startswith("f") and k.endswith("_idx")])
- Handle single-hot / multi-hot / empty bags uniformly; support optional values.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
import logging
from functools import partial

import numpy as np
import torch
from torch.autograd.profiler import record_function
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

# Prefer importing the dataset class; provide a clear error if name changes.
try:
    from src.data.dataset import ProcessedIterDataset
except Exception as exc:  # pragma: no cover - defensive
    raise ImportError(
        "Failed to import ProcessedIterDataset from src.data.dataset. "
        "Please ensure dataset.py exposes ProcessedIterDataset."
    ) from exc


# ---------- small helpers ----------

def _load_feature_meta(metadata_path: Path) -> Dict[str, Any]:
    """Load feature_meta from metadata.json; falls back to empty dict."""
    if not metadata_path.exists():
        return {}
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data.get("feature_meta", {}) or {}


def _ensure_list(obj: Any) -> List[Any]:
    """Normalize scalar or iterable to a Python list."""
    if obj is None:
        return []
    if isinstance(obj, (list, tuple)):
        return list(obj)
    if torch.is_tensor(obj):
        return obj.tolist()
    # Handle numpy arrays
    if hasattr(obj, '__array__'):
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    return [obj]


def _check_indices(indices: List[int], base: str, row_idx: int, meta: Dict[str, Any], debug: bool) -> None:
    for idx in indices:
        if idx is None:
            raise ValueError(f"{base}: idx is None at row {row_idx}")
        # Handle both scalar and array elements
        if hasattr(idx, '__len__') and not isinstance(idx, str):
            # idx is array/list, check all elements
            idx_arr = np.atleast_1d(idx)
            if (idx_arr < 0).any():
                neg_vals = idx_arr[idx_arr < 0].tolist()
                raise ValueError(f"{base}: negative idx {neg_vals} at row {row_idx}")
        else:
            # idx is scalar
            if idx < 0:
                raise ValueError(f"{base}: negative idx {idx} at row {row_idx}")
    if debug:
        upper = meta.get("num_embeddings") or meta.get("vocab_num_embeddings") or meta.get("hash_bucket")
        offset = meta.get("special_base_offset") or meta.get("base_offset") or 0
        if upper is not None:
            limit = offset + int(upper)
            # Flatten indices to handle nested arrays
            flat_indices = []
            for idx in indices:
                if hasattr(idx, '__iter__') and not isinstance(idx, str):
                    flat_indices.extend(list(np.atleast_1d(idx)))
                else:
                    flat_indices.append(idx)
            bad = [x for x in flat_indices if x >= limit]
            if bad:
                raise ValueError(f"{base}: {len(bad)} indices exceed num_embeddings={limit}, sample row {row_idx}, first5={bad[:5]}")


def _to_1d_int64(raw: Any, is_multi: bool) -> np.ndarray:
    """Convert scalar / list-like / tensor to contiguous int64 numpy array."""
    if raw is None:
        return np.empty(0, dtype=np.int64)
    if hasattr(raw, "as_py"):  # pyarrow scalar
        raw = raw.as_py()
    if is_multi:
        if torch.is_tensor(raw):
            arr = raw.detach().cpu().numpy()
        else:
            arr = np.asarray(raw)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        elif arr.ndim > 1:
            arr = arr.reshape(-1)
        return np.asarray(arr, dtype=np.int64)
    return np.asarray([int(raw)], dtype=np.int64)


def _to_1d_float32(raw: Any, is_multi: bool) -> np.ndarray:
    """Convert scalar / list-like / tensor to contiguous float32 numpy array."""
    if raw is None:
        return np.empty(0, dtype=np.float32)
    if hasattr(raw, "as_py"):  # pyarrow scalar
        raw = raw.as_py()
    if is_multi:
        if torch.is_tensor(raw):
            arr = raw.detach().cpu().numpy()
        else:
            arr = np.asarray(raw)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        elif arr.ndim > 1:
            arr = arr.reshape(-1)
        return np.asarray(arr, dtype=np.float32)
    return np.asarray([float(raw)], dtype=np.float32)


@dataclass(frozen=True)
class _FieldPlan:
    base: str
    idx_key: str
    val_key: str
    is_multi: bool
    use_value: bool
    max_len: int | None
    meta: Dict[str, Any]


class EmbeddingBagBatchCollator:
    """
    Fast collator for EmbeddingBag input packing.
    Key ideas:
      - Cache field planning (base -> idx/val/meta) once per worker.
      - Pre-allocate numpy arrays by total nnz per field (avoid repeated list.extend).
      - Use torch.from_numpy to avoid Python-list tensor construction overhead.
    """

    def __init__(self, feature_meta: Dict[str, Any], debug: bool = False) -> None:
        self.feature_meta = feature_meta
        self.debug = debug
        self._field_plans: List[_FieldPlan] | None = None
        self._field_names: List[str] | None = None
        self._any_use_value: bool = False

    def _build_field_plans(self, sample0: Dict[str, Any]) -> None:
        base_keys = sorted(k for k in sample0.keys() if k.startswith("f") and k.endswith("_idx"))
        bases = [k[:-4] for k in base_keys]
        if not bases:
            raise ValueError("No feature *_idx columns found in batch.")

        plans: List[_FieldPlan] = []
        any_use_value = False
        for base in bases:
            meta = self.feature_meta.get(base, {})
            is_multi = bool(meta.get("is_multi_hot", False))
            use_value = bool(meta.get("use_value", False))
            max_len = meta.get("max_len")
            if isinstance(max_len, float) and math.isnan(max_len):
                max_len = None
            if max_len is not None:
                try:
                    max_len = int(max_len)
                except Exception:
                    max_len = None
            plans.append(
                _FieldPlan(
                    base=base,
                    idx_key=f"{base}_idx",
                    val_key=f"{base}_val",
                    is_multi=is_multi,
                    use_value=use_value,
                    max_len=max_len,
                    meta=meta,
                )
            )
            any_use_value = any_use_value or use_value

        self._field_plans = plans
        self._field_names = [p.base for p in plans]
        self._any_use_value = any_use_value

    def __call__(self, batch: List[Dict]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        if not batch:
            raise ValueError("Empty batch is not allowed.")
        if self._field_plans is None:
            sample0 = batch[0]
            if not isinstance(sample0, dict):
                raise TypeError(f"Expected row dict from dataset, got {type(sample0)}.")
            self._build_field_plans(sample0)

        field_plans = self._field_plans or []
        field_names = self._field_names or []

        with record_function("dataloader.collate_embeddingbag"):
            t0 = time.perf_counter()

            # 1) labels/meta with ESMM guards
            y_ctr_raw = np.asarray([float(row.get("y_ctr", 0.0)) for row in batch], dtype=np.float32)
            y_cvr_raw = np.asarray(
                [0.0 if row.get("y_cvr") is None else float(row.get("y_cvr", 0.0)) for row in batch],
                dtype=np.float32,
            )
            valid_mask = ~((y_ctr_raw <= 0.0) & (y_cvr_raw > 0.0))
            if not np.any(valid_mask):
                raise ValueError("Batch empty after filtering funnel-inconsistent samples.")

            keep_idx = np.nonzero(valid_mask)[0]
            rows = [batch[int(i)] for i in keep_idx.tolist()]
            y_ctr = y_ctr_raw[keep_idx]
            y_cvr = y_cvr_raw[keep_idx]
            y_cvr[y_ctr <= 0.0] = 0.0

            B = int(len(rows))
            if B == 0:
                raise ValueError("Batch empty after filtering funnel-inconsistent samples.")

            click_mask = np.empty(B, dtype=np.float32)
            row_id = np.empty(B, dtype=np.int64)
            entity_id: List[Any] = []
            for i, row in enumerate(rows):
                cm = row.get("click_mask")
                if cm is None:
                    cm = 1.0 if y_ctr[i] > 0.0 else 0.0
                click_mask[i] = float(cm)
                row_id[i] = int(row.get("row_id", i))
                entity_id.append(row.get("entity_id"))

            y_ctcvr = np.logical_and(y_ctr > 0.5, y_cvr > 0.5).astype(np.float32, copy=False)
            labels = {
                "y_ctr": torch.from_numpy(y_ctr),
                "y_cvr": torch.from_numpy(y_cvr),
                "y_ctcvr": torch.from_numpy(y_ctcvr),
                "click_mask": torch.from_numpy(click_mask),
                "row_id": torch.from_numpy(row_id),
            }
            meta_out = {"entity_id": entity_id}

            # 2) per-field packing
            fields_out: Dict[str, Dict[str, torch.Tensor | None]] = {}
            for fp in field_plans:
                idx_rows: List[np.ndarray] = [np.empty(0, dtype=np.int64)] * B
                val_rows: List[np.ndarray] | None = [np.empty(0, dtype=np.float32)] * B if fp.use_value else None
                total_nnz = 0

                for row_idx, row in enumerate(rows):
                    idx_arr = _to_1d_int64(row.get(fp.idx_key), is_multi=fp.is_multi)
                    if fp.max_len is not None and idx_arr.size > fp.max_len:
                        idx_arr = idx_arr[: fp.max_len]
                    if self.debug:
                        _check_indices(idx_arr.tolist(), fp.base, row_idx, fp.meta, self.debug)
                    idx_rows[row_idx] = idx_arr
                    n = int(idx_arr.size)
                    total_nnz += n

                    if fp.use_value:
                        assert val_rows is not None
                        val_arr = _to_1d_float32(row.get(fp.val_key), is_multi=fp.is_multi)
                        if fp.max_len is not None and val_arr.size > fp.max_len:
                            val_arr = val_arr[: fp.max_len]
                        if val_arr.size == 0 and n > 0:
                            val_arr = np.ones(n, dtype=np.float32)
                        if val_arr.size != n:
                            raise ValueError(
                                f"{fp.base}: val/idx length mismatch at row {row_idx} ({val_arr.size} vs {n})"
                            )
                        val_rows[row_idx] = val_arr

                indices_np = np.empty(total_nnz, dtype=np.int64)
                offsets_np = np.empty(B, dtype=np.int64)
                weights_np = np.empty(total_nnz, dtype=np.float32) if self._any_use_value else None

                cursor = 0
                for row_idx in range(B):
                    offsets_np[row_idx] = cursor
                    idx_arr = idx_rows[row_idx]
                    n = int(idx_arr.size)
                    if n <= 0:
                        continue
                    indices_np[cursor : cursor + n] = idx_arr
                    if weights_np is not None:
                        if fp.use_value:
                            assert val_rows is not None
                            weights_np[cursor : cursor + n] = val_rows[row_idx]
                        else:
                            weights_np[cursor : cursor + n] = 1.0
                    cursor += n

                if self.debug:
                    assert cursor == total_nnz, f"{fp.base}: packed nnz mismatch ({cursor} vs {total_nnz})"
                    if B > 0:
                        assert offsets_np[0] == 0, f"{fp.base}: offsets[0] must be 0"
                        assert np.all(offsets_np[:-1] <= offsets_np[1:]), f"{fp.base}: offsets not non-decreasing"

                fields_out[fp.base] = {
                    "indices": torch.from_numpy(indices_np),
                    "offsets": torch.from_numpy(offsets_np),
                    "weights": torch.from_numpy(weights_np) if weights_np is not None else None,
                }

            features = {
                "fields": fields_out,
                "field_names": field_names,
            }
            meta_out["_perf"] = {"collate_ms": (time.perf_counter() - t0) * 1000.0}
            return labels, features, meta_out


# ---------- core collation ----------

def collate_fn_embeddingbag(
    batch: List[Dict],
    feature_meta: Dict[str, Any],
    debug: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    # Backward-compatible function API; make_dataloader uses cached class instance.
    return EmbeddingBagBatchCollator(feature_meta=feature_meta, debug=debug)(batch)


# ---------- DataLoader builder ----------

def _seed_worker(worker_id: int, base_seed: int) -> None:  # pragma: no cover - simple
    seed = base_seed + worker_id
    random.seed(seed)
    try:
        import numpy as np  # optional
    except ImportError:
        np = None
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)


def _collate_embeddingbag(batch: List[Dict], collator: EmbeddingBagBatchCollator) -> Tuple[Dict, Dict, Dict]:
    return collator(batch)


def _worker_init_with_seed(worker_id: int, base_seed: int | None = None, worker_cpu_threads: int = 1) -> None:
    if base_seed is not None:
        _seed_worker(worker_id, base_seed)
    # Avoid num_workers x OMP/MKL thread explosion on spawn-based platforms.
    if worker_cpu_threads > 0:
        os.environ.setdefault("OMP_NUM_THREADS", str(worker_cpu_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(worker_cpu_threads))
        os.environ.setdefault("OPENBLAS_NUM_THREADS", str(worker_cpu_threads))
        os.environ.setdefault("NUMEXPR_NUM_THREADS", str(worker_cpu_threads))
        try:
            torch.set_num_threads(worker_cpu_threads)
        except Exception:
            pass
        try:
            torch.set_num_interop_threads(1)
        except Exception:
            pass


class _SubsetIterDataset(IterableDataset):
    """
    Deterministic down-sampling wrapper for IterableDataset.
    Used to build a small, fixed validation subset for cheap AUC.
    """

    def __init__(
        self,
        base_ds: IterableDataset,
        subset_ratio: float | None = None,
        max_samples: int | None = None,
        seed: int | None = None,
    ) -> None:
        if subset_ratio is not None and (subset_ratio <= 0 or subset_ratio > 1):
            raise ValueError("subset_ratio must be in (0,1].")
        if max_samples is not None and max_samples <= 0:
            raise ValueError("max_samples must be positive.")
        self.base_ds = base_ds
        self.subset_ratio = subset_ratio
        self.max_samples = max_samples
        self.seed = seed if seed is not None else 2026

    def __iter__(self):
        info = get_worker_info()
        worker_seed = self.seed if info is None else (self.seed + info.id)
        rng = random.Random(worker_seed)
        yielded = 0
        for row in iter(self.base_ds):
            if self.max_samples is not None and yielded >= self.max_samples:
                break
            if self.subset_ratio is not None and rng.random() > self.subset_ratio:
                continue
            yielded += 1
            yield row

    def __len__(self) -> int:  # pragma: no cover - best effort estimate
        base_len = len(self.base_ds) if hasattr(self.base_ds, "__len__") else 0
        if self.max_samples is not None:
            base_len = min(base_len, self.max_samples)
        if self.subset_ratio is not None and base_len:
            return int(base_len * self.subset_ratio)
        return base_len


def make_dataloader(
    split: str,
    batch_size: int,
    num_workers: int,
    shuffle: bool | None = None,
    drop_last: bool = False,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    seed: int | None = None,
    feature_meta: Dict[str, Any] | None = None,
    debug: bool = False,
    neg_keep_prob_train: float = 1.0,
    subset_ratio: float | None = None,
    subset_max_samples: int | None = None,
    subset_seed: int | None = None,
    prefetch_factor: int | None = None,
    worker_cpu_threads: int = 1,
) -> DataLoader:
    """
    Build a DataLoader that yields (labels, features, meta) tuples
    suitable for EmbeddingBag-based models.
    """
    if split not in {"train", "valid"}:
        raise ValueError("split must be 'train' or 'valid'")
    if worker_cpu_threads <= 0:
        raise ValueError("worker_cpu_threads must be >= 1")

    data_dir = Path("data/processed") / split
    metadata_path = data_dir.parent / "metadata.json"
    fm = feature_meta or _load_feature_meta(metadata_path)

    # IterableDataset does not support shuffle=True; guard against misuse.
    if shuffle is None:
        shuffle = split == "train"
    if shuffle:
        raise ValueError("shuffle=True is not supported with IterableDataset; shuffle offline instead.")

    ds: IterableDataset = ProcessedIterDataset(
        data_dir,
        metadata_path=metadata_path,
        neg_keep_prob=neg_keep_prob_train if split == "train" else 1.0,
    )
    if split == "valid" and (subset_ratio is not None or subset_max_samples is not None):
        ds = _SubsetIterDataset(
            ds,
            subset_ratio=subset_ratio,
            max_samples=subset_max_samples,
            seed=subset_seed if subset_seed is not None else seed,
        )

    base_seed = seed if seed is not None else None
    worker_init = (
        partial(_worker_init_with_seed, base_seed=base_seed, worker_cpu_threads=worker_cpu_threads)
        if num_workers > 0
        else None
    )
    collator = EmbeddingBagBatchCollator(feature_meta=fm, debug=debug)
    collate = partial(_collate_embeddingbag, collator=collator)

    # Determine prefetch_factor: for high worker counts, prefetch 4 is usually better.
    if prefetch_factor is not None:
        effective_prefetch = prefetch_factor
    else:
        if num_workers <= 0:
            effective_prefetch = None
        elif num_workers >= 4:
            effective_prefetch = 4
        else:
            effective_prefetch = 2
    
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
        persistent_workers=bool(persistent_workers and num_workers > 0),
        collate_fn=collate,
        worker_init_fn=worker_init,
        prefetch_factor=effective_prefetch,
    )
    if num_workers > 0:
        logging.getLogger(__name__).info(
            "DataLoader spawn config: num_workers=%d persistent_workers=%s pin_memory=%s prefetch_factor=%s worker_cpu_threads=%d collate_fn=%s.%s",
            num_workers,
            bool(persistent_workers and num_workers > 0),
            pin_memory,
            effective_prefetch,
            worker_cpu_threads,
            collate.func.__module__ if isinstance(collate, partial) else type(collate).__module__,
            collate.func.__name__ if isinstance(collate, partial) else getattr(collate, "__name__", type(collate).__name__),
        )
    return dl


# ---------- quick self-check ----------

def _quick_test_embeddingbag_pack() -> None:
    """
    Minimal self-test: two samples with single-hot, multi-hot, and an empty bag.
    Ensures shapes/offsets/weights align with spec.
    """
    feature_meta = {
        "f0_301": {"is_multi_hot": False, "use_value": False},
        "f1_110_14": {"is_multi_hot": True, "use_value": True, "max_len": 3},
    }

    batch = [
        {
            "y_ctr": 1.0,
            "y_cvr": 0.0,
            "y_ctcvr": 0.0,
            "click_mask": 1.0,
            "row_id": 10,
            "entity_id": "uA",
            "f0_301_idx": 5,
            "f1_110_14_idx": [7, 8, 9, 10],  # will be truncated to 3
            "f1_110_14_val": [0.5, 0.4, 0.3, 0.2],
        },
        {
            "y_ctr": 0.0,
            "y_cvr": 0.0,
            "y_ctcvr": 0.0,
            "click_mask": 1.0,
            "row_id": 11,
            "entity_id": "uB",
            "f0_301_idx": 3,
            "f1_110_14_idx": [],  # empty bag allowed
        },
    ]

    labels, features, meta = collate_fn_embeddingbag(batch, feature_meta, debug=True)

    fields = features["fields"]
    # field order
    assert features["field_names"] == ["f0_301", "f1_110_14"]

    # f0_301 single-hot
    f0 = fields["f0_301"]
    assert f0["indices"].tolist() == [5, 3]
    assert f0["offsets"].tolist() == [0, 1]
    assert f0["weights"] is not None and f0["weights"].tolist() == [1.0, 1.0]  # filled because any_use_value=True

    # f1_110_14 multi-hot with truncation + empty bag
    f1 = fields["f1_110_14"]
    assert f1["indices"].tolist() == [7, 8, 9]
    assert f1["offsets"].tolist() == [0, 3]  # second sample empty -> length derived from last offset
    assert f1["weights"] is not None
    assert torch.allclose(f1["weights"], torch.tensor([0.5, 0.4, 0.3], dtype=torch.float32))

    assert meta["entity_id"] == ["uA", "uB"]


if __name__ == "__main__":  # pragma: no cover
    _quick_test_embeddingbag_pack()
