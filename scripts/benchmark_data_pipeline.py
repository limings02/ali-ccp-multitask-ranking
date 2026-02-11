from __future__ import annotations

import argparse
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List
import sys

import numpy as np
import torch
from torch.autograd.profiler import record_function

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataloader import make_dataloader
from src.models.build import build_model
from src.utils.config import load_yaml
from src.utils.feature_meta import build_model_feature_meta


def _to_device_labels(
    labels: Dict[str, torch.Tensor],
    device: torch.device,
    non_blocking: bool,
) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=non_blocking) for k, v in labels.items()}


def _to_device_features(
    features: Dict[str, Any],
    device: torch.device,
    non_blocking: bool,
) -> Dict[str, Any]:
    fields = {}
    for base, fd in features["fields"].items():
        fields[base] = {
            "indices": fd["indices"].to(device, non_blocking=non_blocking),
            "offsets": fd["offsets"].to(device, non_blocking=non_blocking),
            "weights": (
                fd["weights"].to(device, non_blocking=non_blocking)
                if fd.get("weights") is not None
                else None
            ),
        }
    return {"fields": fields, "field_names": features["field_names"]}


def _summary(values: List[float]) -> str:
    if not values:
        return "n=0"
    arr = np.asarray(values, dtype=np.float64)
    return (
        f"n={arr.size} mean={mean(values)*1000:.3f}ms "
        f"p50={np.quantile(arr, 0.50)*1000:.3f}ms "
        f"p90={np.quantile(arr, 0.90)*1000:.3f}ms "
        f"p99={np.quantile(arr, 0.99)*1000:.3f}ms"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark dataloader/data pipeline timings.")
    p.add_argument("--config", type=str, default="configs/experiments/mmoe_optim/midgate_mmoe.yaml")
    p.add_argument("--split", type=str, default="train", choices=["train", "valid"])
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--with-model-step", action="store_true", help="Include model forward/backward in t_train_step.")
    p.add_argument("--sync-cuda", action="store_true", help="Synchronize CUDA for accurate per-step timing.")
    p.add_argument("--num-workers", type=int, default=None, help="Override num_workers for benchmark.")
    p.add_argument("--num-workers-valid", type=int, default=None, help="Override num_workers_valid for benchmark.")
    p.add_argument("--pin-memory", type=int, default=None, help="Override pin_memory (1/0).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    data_cfg = cfg.get("data", {})
    embedding_cfg = cfg.get("embedding", {})
    metadata_path = Path(data_cfg["metadata_path"])
    feature_meta = build_model_feature_meta(metadata_path, embedding_cfg)

    train_workers = int(data_cfg.get("num_workers", 0))
    valid_workers = int(data_cfg.get("num_workers_valid", max(1, train_workers // 2)))
    if args.num_workers is not None:
        train_workers = int(args.num_workers)
    if args.num_workers_valid is not None:
        valid_workers = int(args.num_workers_valid)
    workers = train_workers if args.split == "train" else valid_workers
    prefetch_cfg = data_cfg.get("prefetch_factor")
    prefetch = int(prefetch_cfg) if (prefetch_cfg is not None and workers > 0) else None
    persistent_cfg = data_cfg.get("persistent_workers")
    persistent = bool(persistent_cfg) if persistent_cfg is not None else (workers > 0)

    pin_memory = bool(data_cfg.get("pin_memory", True))
    if args.pin_memory is not None:
        pin_memory = bool(int(args.pin_memory))

    loader = make_dataloader(
        split=args.split,
        batch_size=int(data_cfg.get("batch_size", 1024)),
        num_workers=workers,
        shuffle=False,
        drop_last=bool(data_cfg.get("drop_last", False)),
        pin_memory=pin_memory,
        persistent_workers=persistent,
        seed=data_cfg.get("seed"),
        feature_meta=feature_meta,
        debug=bool(data_cfg.get("debug", False)),
        neg_keep_prob_train=float(data_cfg.get("neg_keep_prob_train", 1.0)),
        prefetch_factor=prefetch,
        worker_cpu_threads=int(data_cfg.get("worker_cpu_threads", 1)),
    )

    device = torch.device(args.device)
    non_blocking = bool(device.type == "cuda" and getattr(loader, "pin_memory", False))

    model = None
    optimizer = None
    if args.with_model_step:
        model = build_model(cfg).to(device)
        model.train()
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=1e-6)

    t_next_batch: List[float] = []
    t_collate: List[float] = []
    t_h2d: List[float] = []
    t_train_step: List[float] = []

    loader_iter = iter(loader)
    for step in range(args.steps + args.warmup):
        t0 = time.perf_counter()
        try:
            labels, features, meta = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            labels, features, meta = next(loader_iter)
        t1 = time.perf_counter()

        with record_function("bench.h2d_copy"):
            labels_dev = _to_device_labels(labels, device, non_blocking=non_blocking)
            features_dev = _to_device_features(features, device, non_blocking=non_blocking)
        t2 = time.perf_counter()

        with record_function("bench.train_step"):
            if model is not None and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                out = model(features_dev)
                ctr = out["ctr"] if isinstance(out, dict) else out
                loss = ctr.float().mean()
                loss.backward()
                optimizer.step()
            else:
                _ = labels_dev["y_ctr"].float().mean()
        if device.type == "cuda" and args.sync_cuda:
            torch.cuda.synchronize(device)
        t3 = time.perf_counter()

        if step < args.warmup:
            continue

        t_next_batch.append(t1 - t0)
        collate_ms = 0.0
        if isinstance(meta, dict):
            perf = meta.get("_perf")
            if isinstance(perf, dict):
                collate_ms = float(perf.get("collate_ms", 0.0))
        t_collate.append(collate_ms / 1000.0)
        t_h2d.append(t2 - t1)
        t_train_step.append(t3 - t2)

    print(f"config={args.config}")
    print(
        f"split={args.split} steps={args.steps} warmup={args.warmup} device={device} "
        f"pin_memory={getattr(loader, 'pin_memory', False)} non_blocking={non_blocking} with_model_step={args.with_model_step}"
    )
    print("t_next_batch :", _summary(t_next_batch))
    print("t_collate    :", _summary(t_collate))
    print("t_h2d        :", _summary(t_h2d))
    print("t_train_step :", _summary(t_train_step))


if __name__ == "__main__":
    main()
