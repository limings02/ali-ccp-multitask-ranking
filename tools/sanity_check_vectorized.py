from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pyarrow.dataset as ds
import torch

if __package__ is None or __package__ == "":  # pragma: no cover - script mode
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.dataloader import EmbeddingBagBatchCollator
from src.data.vectorized_dataset import VectorizedBatchCollator, load_vectorized_split_metadata


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sanity check: parquet pipeline vs vectorized pipeline on sampled indices."
    )
    parser.add_argument("--processed-root", type=str, default="data/processed")
    parser.add_argument("--processed-metadata", type=str, default="data/processed/metadata.json")
    parser.add_argument("--vectorized-root", type=str, default="data/vectorized")
    parser.add_argument("--split", type=str, default="valid", choices=["train", "valid"])
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-rows", type=int, default=65536)
    parser.add_argument(
        "--sample-from-first",
        type=int,
        default=None,
        help="Optional speed-up: sample only from first N rows instead of full split.",
    )
    return parser.parse_args()


def _fetch_rows_by_indices(
    split_dir: Path,
    indices: List[int],
    batch_rows: int,
) -> List[Dict[str, Any]]:
    dataset = ds.dataset(split_dir, format="parquet")
    targets = sorted([(idx, pos) for pos, idx in enumerate(indices)], key=lambda x: x[0])
    out: List[Dict[str, Any] | None] = [None] * len(indices)

    cursor = 0
    t = 0
    scanner = dataset.scanner(batch_size=batch_rows)
    for batch in scanner.to_batches():
        if t >= len(targets):
            break
        rows = int(batch.num_rows)
        start = cursor
        end = cursor + rows
        if targets[t][0] >= end:
            cursor = end
            continue

        # Materialize this record batch once, then index only selected rows.
        try:
            cols = {
                name: batch.column(name).to_numpy(zero_copy_only=False)
                for name in batch.schema.names
            }
        except Exception:
            cols = batch.to_pydict()

        while t < len(targets) and targets[t][0] < end:
            global_idx, pos = targets[t]
            local_idx = int(global_idx - start)
            out[pos] = {k: v[local_idx] for k, v in cols.items()}
            t += 1
        cursor = end

    if any(x is None for x in out):
        missing = [i for i, x in enumerate(out) if x is None]
        raise RuntimeError(f"Failed to fetch rows for positions: {missing[:10]}")
    return [x for x in out if x is not None]


def _segment(indices: torch.Tensor, offsets: torch.Tensor, row_idx: int) -> torch.Tensor:
    start = int(offsets[row_idx].item())
    end = int(offsets[row_idx + 1].item()) if row_idx + 1 < offsets.numel() else int(indices.numel())
    return indices[start:end]


def _compare_labels(old_labels: Dict[str, torch.Tensor], new_labels: Dict[str, torch.Tensor]) -> None:
    for key in ["y_ctr", "y_cvr", "y_ctcvr", "click_mask"]:
        if not torch.allclose(old_labels[key], new_labels[key], atol=1e-6, rtol=0):
            raise AssertionError(f"Label mismatch at {key}")
    if not torch.equal(old_labels["row_id"], new_labels["row_id"]):
        raise AssertionError("Label mismatch at row_id")


def _compare_features(
    old_features: Dict[str, Any],
    new_features: Dict[str, Any],
    sampled_indices: np.ndarray,
    kept_indices: np.ndarray,
    split_meta: Dict[str, Any],
    vectorized_split_dir: Path,
) -> None:
    if old_features["field_names"] != new_features["field_names"]:
        raise AssertionError("field_names order mismatch")

    for field_name in old_features["field_names"]:
        old_field = old_features["fields"][field_name]
        new_field = new_features["fields"][field_name]
        if not torch.equal(old_field["offsets"], new_field["offsets"]):
            raise AssertionError(f"{field_name}: offsets mismatch")
        if not torch.equal(old_field["indices"], new_field["indices"]):
            # Print first mismatch sample context for faster debugging.
            B = int(old_field["offsets"].numel())
            mismatch_row = None
            for i in range(B):
                old_seg = _segment(old_field["indices"], old_field["offsets"], i)
                new_seg = _segment(new_field["indices"], new_field["offsets"], i)
                if not torch.equal(old_seg, new_seg):
                    mismatch_row = i
                    break
            if mismatch_row is None:
                mismatch_row = 0

            raw_idx = int(kept_indices[mismatch_row])
            fmeta = split_meta["fields"][field_name]
            global_offsets = np.load(vectorized_split_dir / fmeta["offsets_file"], mmap_mode="r")
            g_start = int(global_offsets[raw_idx])
            g_end = int(global_offsets[raw_idx + 1])
            old_seg = _segment(old_field["indices"], old_field["offsets"], mismatch_row)
            new_seg = _segment(new_field["indices"], new_field["offsets"], mismatch_row)
            raise AssertionError(
                f"{field_name}: indices mismatch at batch_pos={mismatch_row} raw_idx={raw_idx} "
                f"global_start={g_start} global_end={g_end} "
                f"old_head={old_seg[:12].tolist()} new_head={new_seg[:12].tolist()} "
                f"sampled_head={sampled_indices[:8].tolist()}"
            )

        old_w = old_field.get("weights")
        new_w = new_field.get("weights")
        if (old_w is None) != (new_w is None):
            raise AssertionError(f"{field_name}: weights None mismatch")
        if old_w is not None and not torch.allclose(old_w, new_w, atol=1e-6, rtol=0):
            raise AssertionError(f"{field_name}: weights mismatch")


def main() -> None:
    args = _parse_args()
    split = args.split
    processed_root = Path(args.processed_root)
    processed_split_dir = processed_root / split
    vectorized_split_dir = Path(args.vectorized_root) / split

    source_meta = json.loads(Path(args.processed_metadata).read_text(encoding="utf-8"))
    feature_meta = source_meta.get("feature_meta", {})
    if not feature_meta:
        raise KeyError("processed metadata must contain feature_meta")

    split_meta = load_vectorized_split_metadata(vectorized_split_dir)
    split_total = min(
        int(source_meta.get("rows", {}).get(split, 0)),
        int(split_meta["num_rows"]),
    )
    if split_total <= 0:
        raise ValueError(f"Invalid row count for split={split}: {split_total}")
    sample_upper = min(split_total, int(args.sample_from_first)) if args.sample_from_first else split_total
    if sample_upper < args.k:
        raise ValueError(f"sample pool {sample_upper} smaller than k={args.k}")

    rng = np.random.default_rng(args.seed)
    sampled_indices = rng.choice(sample_upper, size=args.k, replace=False).astype(np.int64)

    # Old parquet pipeline (existing collator + rows from processed parquet).
    rows = _fetch_rows_by_indices(
        split_dir=processed_split_dir,
        indices=sampled_indices.tolist(),
        batch_rows=int(args.batch_rows),
    )
    old_collator = EmbeddingBagBatchCollator(feature_meta=feature_meta, debug=False)
    old_labels, old_features, old_meta = old_collator(rows)

    # New vectorized pipeline.
    new_collator = VectorizedBatchCollator(vectorized_split_dir)
    new_labels, new_features, new_meta = new_collator(sampled_indices.tolist())

    # Keep-mask for debug mapping (must match both collators).
    y_ctr_full = np.load(vectorized_split_dir / "y_ctr.npy", mmap_mode="r")
    y_cvr_full = np.load(vectorized_split_dir / "y_cvr.npy", mmap_mode="r")
    y_ctr_raw = y_ctr_full[sampled_indices]
    y_cvr_raw = y_cvr_full[sampled_indices]
    keep_mask = ~((y_ctr_raw <= 0.0) & (y_cvr_raw > 0.0))
    kept_indices = sampled_indices[keep_mask]

    _compare_labels(old_labels, new_labels)
    _compare_features(
        old_features=old_features,
        new_features=new_features,
        sampled_indices=sampled_indices,
        kept_indices=kept_indices,
        split_meta=split_meta,
        vectorized_split_dir=vectorized_split_dir,
    )
    if old_meta.get("entity_id") != new_meta.get("entity_id"):
        raise AssertionError("meta.entity_id mismatch")

    print(
        json.dumps(
            {
                "status": "ok",
                "split": split,
                "k_sampled": int(args.k),
                "k_kept_after_funnel_guard": int(len(kept_indices)),
                "seed": int(args.seed),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
