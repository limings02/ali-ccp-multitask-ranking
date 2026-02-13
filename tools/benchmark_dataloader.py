from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

if __package__ is None or __package__ == "":  # pragma: no cover - script mode
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.dataloader import make_dataloader


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark DataLoader only (no model forward).")
    parser.add_argument("--format", type=str, default="both", choices=["parquet", "vectorized", "both"])
    parser.add_argument("--split", type=str, default="valid", choices=["train", "valid"])
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--warmup-steps", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--pin-memory", dest="pin_memory", action="store_true", default=True)
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.add_argument(
        "--persistent-workers",
        dest="persistent_workers",
        action="store_true",
        default=True,
    )
    parser.add_argument("--no-persistent-workers", dest="persistent_workers", action="store_false")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--processed-root", type=str, default="data/processed")
    parser.add_argument("--vectorized-root", type=str, default="data/vectorized")
    parser.add_argument("--metadata-path", type=str, default="data/processed/metadata.json")
    return parser.parse_args()


def _load_feature_meta(metadata_path: Path) -> Dict[str, Any]:
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    return data.get("feature_meta", {}) or {}


def _bench_once(
    data_format: str,
    split: str,
    steps: int,
    warmup_steps: int,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    pin_memory: bool,
    persistent_workers: bool,
    seed: int,
    processed_root: str,
    vectorized_root: str,
    feature_meta: Dict[str, Any],
) -> Dict[str, Any]:
    loader = make_dataloader(
        split=split,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        seed=seed,
        feature_meta=feature_meta,
        debug=False,
        prefetch_factor=prefetch_factor,
        data_format=data_format,
        processed_root=processed_root,
        vectorized_root=vectorized_root,
    )

    collate_ms_values: List[float] = []
    measured = 0
    started = False
    t0 = 0.0

    for step_idx, (_, _, meta) in enumerate(loader):
        if step_idx == warmup_steps:
            t0 = time.perf_counter()
            started = True
        if started:
            measured += 1
            perf = meta.get("_perf") if isinstance(meta, dict) else None
            if isinstance(perf, dict) and perf.get("collate_ms") is not None:
                collate_ms_values.append(float(perf["collate_ms"]))
        if step_idx + 1 >= warmup_steps + steps:
            break

    elapsed = (time.perf_counter() - t0) if started else 0.0
    batches_per_sec = (measured / elapsed) if elapsed > 0 else 0.0

    return {
        "format": data_format,
        "split": split,
        "steps_target": steps,
        "steps_measured": measured,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "prefetch_factor": prefetch_factor,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "batches_per_sec": batches_per_sec,
        "collate_ms_mean": float(np.mean(collate_ms_values)) if collate_ms_values else None,
        "collate_ms_median": float(np.median(collate_ms_values)) if collate_ms_values else None,
    }


def main() -> None:
    args = _parse_args()
    num_workers = (
        int(args.num_workers)
        if args.num_workers is not None
        else max(0, min(6, os.cpu_count() or 1))
    )
    feature_meta = _load_feature_meta(Path(args.metadata_path))

    formats = [args.format] if args.format != "both" else ["parquet", "vectorized"]
    results = []
    for fmt in formats:
        results.append(
            _bench_once(
                data_format=fmt,
                split=args.split,
                steps=int(args.steps),
                warmup_steps=int(args.warmup_steps),
                batch_size=int(args.batch_size),
                num_workers=num_workers,
                prefetch_factor=int(args.prefetch_factor),
                pin_memory=bool(args.pin_memory),
                persistent_workers=bool(args.persistent_workers),
                seed=int(args.seed),
                processed_root=str(args.processed_root),
                vectorized_root=str(args.vectorized_root),
                feature_meta=feature_meta,
            )
        )

    summary: Dict[str, Any] = {"results": results}
    if len(results) == 2:
        old = next((x for x in results if x["format"] == "parquet"), None)
        new = next((x for x in results if x["format"] == "vectorized"), None)
        if old and new and old["batches_per_sec"] > 0:
            summary["speedup_batches_per_sec"] = new["batches_per_sec"] / old["batches_per_sec"]
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
