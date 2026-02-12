"""
Dataset/Loader helper for processed data.

Key behaviours:
- Multi-worker safe: shards parquet fragments per worker to avoid duplicate reads.
- Value handling: only pad/return *_val when FeatureSpec.use_value is True.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Iterator, List, Tuple, Optional
from dataclasses import dataclass, asdict, field

import logging
import pyarrow.dataset as ds
import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info


# ============================================================
# Resume State Management for Checkpoint Recovery
# ============================================================
@dataclass
class DatasetWorkerState:
    """
    Per-worker iteration state for resume support.
    - frag_idx: current fragment index (within worker's assigned fragments)
    - batch_idx: current batch index within the fragment
    - row_idx: current row index within the batch (for partial batch resume)
    - rng_state: pickled state of random.Random for deterministic neg sampling
    """
    frag_idx: int = 0
    batch_idx: int = 0
    row_idx: int = 0
    rng_state: Optional[Tuple] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "DatasetWorkerState":
        return cls(**d)


@dataclass
class DatasetResumeState:
    """
    Global state tracking for all workers. Used by Trainer to save/restore.
    Maps worker_id -> DatasetWorkerState dict
    """
    data_state: Dict[int, Dict] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {"data_state": self.data_state}

    @classmethod
    def from_dict(cls, d: Dict) -> "DatasetResumeState":
        return cls(data_state=d.get("data_state", {}))


def _load_feature_meta(meta_path: Path) -> Tuple[Dict[str, Dict[str, bool]], Dict[str, int]]:
    """
    Read metadata.json produced by processed_builder to recover per-field flags and row counts.
    Returns (feature_meta, rows_dict) where rows_dict maps split name -> row count.
    """
    if not meta_path.exists():
        return {}, {}
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}, {}
    return meta.get("feature_meta", {}), meta.get("rows", {})


class ProcessedIterDataset(IterableDataset):
    """
    Stream rows from processed parquet using pyarrow.dataset.
    Worker sharding is done by slicing the fragment list so each worker reads
    disjoint files and no examples are duplicated.
    
    Support resume from checkpoint: restore fragment/batch/row positions per worker.
    """

    def __init__(
        self,
        data_dir: str | Path,
        metadata_path: str | Path | None = None,
        neg_keep_prob: float = 1.0,
        resume_state: Optional[DatasetResumeState] = None,
    ):
        self.data_dir = Path(data_dir)
        self.dataset = ds.dataset(self.data_dir, format="parquet")
        # Sort fragments for stable order across resume
        self.fragments = sorted(
            list(self.dataset.get_fragments()),
            key=lambda f: str(f.path)
        )
        meta_path = Path(metadata_path) if metadata_path else self.data_dir.parent / "metadata.json"
        self.feature_meta, rows_dict = _load_feature_meta(meta_path)
        # P1-6: prefer pre-computed row count from metadata to avoid expensive count_rows()
        split_name = self.data_dir.name  # e.g., "train" or "valid"
        if split_name in rows_dict:
            self._count = int(rows_dict[split_name])
        else:
            self._count = self.dataset.count_rows()  # fallback
        self.split_name = split_name
        # Only meaningful for train split; for others we keep all samples.
        self.neg_keep_prob = float(neg_keep_prob)
        if self.neg_keep_prob < 0.0 or self.neg_keep_prob > 1.0:
            raise ValueError("neg_keep_prob must be in [0,1]")
        
        # Resume state
        self.resume_state = resume_state or DatasetResumeState()

    def __iter__(self) -> Iterator[Dict]:
        info = get_worker_info()
        worker_id = info.id if info is not None else 0
        num_workers = info.num_workers if info is not None else 1
        
        # Get fragments assigned to this worker
        fragments = self.fragments
        if info is not None and info.num_workers > 1:
            fragments = fragments[info.id :: info.num_workers]
        
        # Recover resume state for this worker
        worker_state_dict = self.resume_state.data_state.get(worker_id)
        if worker_state_dict:
            worker_state = DatasetWorkerState.from_dict(worker_state_dict)
        else:
            worker_state = DatasetWorkerState()
        
        # Initialize RNG
        rng = random.Random()
        if info is not None:
            rng.seed(info.seed ^ 0xABCDEF)
        
        # Restore RNG state if available (for deterministic neg sampling)
        if worker_state.rng_state is not None:
            try:
                rng.setstate(worker_state.rng_state)
            except Exception as e:
                _logger = logging.getLogger(__name__)
                _logger.warning(f"Failed to restore RNG state for worker {worker_id}: {e}")
                worker_state.rng_state = None
        
        # Iterate from saved position
        for frag_idx in range(worker_state.frag_idx, len(fragments)):
            frag = fragments[frag_idx]
            batches = list(frag.to_batches())
            
            # Start from saved batch index
            batch_start_idx = worker_state.batch_idx if frag_idx == worker_state.frag_idx else 0
            for batch_idx in range(batch_start_idx, len(batches)):
                batch = batches[batch_idx]
                
                try:
                    cols = {name: batch.column(name).to_numpy(zero_copy_only=False) 
                            for name in batch.schema.names}
                except Exception:
                    cols = batch.to_pydict()
                
                rows = batch.num_rows
                # Start from saved row index
                row_start_idx = worker_state.row_idx if (frag_idx == worker_state.frag_idx 
                                                          and batch_idx == worker_state.batch_idx) else 0
                
                for i in range(row_start_idx, rows):
                    row = {k: v[i] for k, v in cols.items()}
                    
                    if self.split_name == "train" and self.neg_keep_prob < 1.0:
                        # Down-sample only when y_ctr is negative.
                        y_ctr = row.get("y_ctr", None)
                        try:
                            is_negative = (y_ctr is not None) and (float(y_ctr) <= 0.0)
                        except Exception:
                            is_negative = False
                        if is_negative and rng.random() > self.neg_keep_prob:
                            continue
                    
                    # Update position for next resume
                    worker_state.frag_idx = frag_idx
                    worker_state.batch_idx = batch_idx
                    worker_state.row_idx = i + 1
                    worker_state.rng_state = rng.getstate()
                    
                    yield row
                
                # Reset row_idx after batch is done
                worker_state.row_idx = 0
            
            # Reset batch_idx after fragment is done
            worker_state.batch_idx = 0

    def __len__(self) -> int:
        return self._count


def _pad_indices(seqs: List[List[int]], pad_val: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = max((len(s) for s in seqs), default=0)
    out = torch.full((len(seqs), max_len), pad_val, dtype=torch.int64)
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.int64)
    for i, s in enumerate(seqs):
        if s:
            out[i, : len(s)] = torch.tensor(s, dtype=torch.int64)
    return out, lengths


def _pad_values(seqs: List[List[float]], pad_val: float = 0.0) -> torch.Tensor:
    max_len = max((len(s) for s in seqs), default=0)
    out = torch.full((len(seqs), max_len), pad_val, dtype=torch.float32)
    for i, s in enumerate(seqs):
        if s:
            out[i, : len(s)] = torch.tensor(s, dtype=torch.float32)
    return out


def _collate_factory(feature_meta: Dict[str, Dict[str, bool]]):
    """
    Build a collate_fn that honours per-field use_value flag.
    """

    def collate_fn(batch: List[Dict]):
        log = logging.getLogger(__name__)
        y_ctr_list: List[float] = []
        y_cvr_list: List[float] = []
        y_ctcvr_list: List[float] = []
        click_mask_list: List[float] = []
        row_id_list: List[int] = []
        entity_list: List = []
        funnel_bad = 0

        for b in batch:
            y_ctr = float(b.get("y_ctr", 0.0))
            try:
                y_cvr = float(b.get("y_cvr", 0.0))
            except Exception:
                y_cvr = 0.0
            if y_ctr <= 0.0 and y_cvr > 0.0:
                funnel_bad += 1
                y_cvr = 0.0
            if y_ctr <= 0.0:
                y_cvr = 0.0

            click_mask = b.get("click_mask")
            if click_mask is None:
                click_mask = 1.0 if y_ctr > 0.0 else 0.0
            click_mask = float(click_mask)

            y_ctcvr = 1.0 if (y_ctr > 0.5 and y_cvr > 0.5) else 0.0

            y_ctr_list.append(y_ctr)
            y_cvr_list.append(y_cvr)
            y_ctcvr_list.append(y_ctcvr)
            click_mask_list.append(click_mask)
            row_id_list.append(int(b.get("row_id", len(row_id_list))))
            entity_list.append(b.get("entity_id"))

        if funnel_bad:
            log.warning("funnel_bad detected: %d samples had y_cvr=1 while y_ctr=0; coerced y_cvr->0.", funnel_bad)

        labels = {
            "y_ctr": torch.tensor(y_ctr_list, dtype=torch.float32),
            "y_cvr": torch.tensor(y_cvr_list, dtype=torch.float32),
            "y_ctcvr": torch.tensor(y_ctcvr_list, dtype=torch.float32),
            "click_mask": torch.tensor(click_mask_list, dtype=torch.float32),
            "row_id": torch.tensor(row_id_list, dtype=torch.int64),
            "entity_id": entity_list,
        }

        features: Dict[str, Dict[str, torch.Tensor | None]] = {}
        # Iterate over keys once to avoid repeated string ops.
        for key in batch[0].keys():
            if not (key.startswith("f") and key.endswith("_idx")):
                continue
            base = key[:-4]
            meta = feature_meta.get(base, {})
            use_value = meta.get("use_value")
            if use_value is None:
                use_value = (base + "_val") in batch[0]  # fallback: infer from presence
            use_value = bool(use_value)
            is_multi = bool(meta.get("is_multi_hot", isinstance(batch[0][key], list)))
            if is_multi:
                idx_seqs = [b[key] for b in batch]
                idx_pad, lengths = _pad_indices(idx_seqs, pad_val=0)
                feat = {"type": "multi", "idx": idx_pad, "len": lengths}
                # P1-5: *_off column removed from processed data (was semantically mismatched)
                val_key = base + "_val"
                if use_value and val_key in batch[0]:
                    val_seqs = [b[val_key] for b in batch]
                    feat["val"] = _pad_values(val_seqs, pad_val=0.0)
                else:
                    feat["val"] = None
                features[base] = feat
            else:
                idx_tensor = torch.tensor([b[key] for b in batch], dtype=torch.int64)
                feat = {"type": "single", "idx": idx_tensor}
                val_key = base + "_val"
                if use_value and val_key in batch[0]:
                    feat["val"] = torch.tensor([b[val_key] for b in batch], dtype=torch.float32)
                else:
                    feat["val"] = None
                features[base] = feat
        return {"labels": labels, "features": features}

    return collate_fn


def build_dataloader(
    data_dir: str | Path,
    batch_size: int = 1024,
    shuffle: bool = False,
    num_workers: int = 0,
    metadata_path: str | Path | None = None,
) -> DataLoader:
    # P0-4: IterableDataset does not support shuffle=True in DataLoader
    if shuffle:
        raise ValueError(
            "shuffle=True is not supported with IterableDataset. "
            "For shuffling, either shuffle the parquet fragments offline, "
            "or use a map-style dataset."
        )
    ds_obj = ProcessedIterDataset(data_dir, metadata_path=metadata_path)
    collate = _collate_factory(ds_obj.feature_meta)
    return DataLoader(
        ds_obj,
        batch_size=batch_size,
        shuffle=False,  # always False for IterableDataset
        num_workers=num_workers,
        collate_fn=collate,
    )


# Backward-compatible default collate_fn (uses no metadata; val returned only if present).
collate_fn = _collate_factory({})


__all__ = ["ProcessedIterDataset", "build_dataloader", "collate_fn", "DatasetResumeState", "DatasetWorkerState"]
