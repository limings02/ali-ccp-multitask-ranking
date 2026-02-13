from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict

import torch

from src.data.dataloader import make_dataloader
from src.models.backbones.deepfm import DeepFMBackbone
from src.utils.config import load_yaml
from src.utils.feature_meta import build_model_feature_meta


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepFM backbone smoke runner")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/models/backbone_deepfm.yaml",
        help="Path to YAML config.",
    )
    return parser.parse_args()


def _get_cfg(cfg: Dict[str, Any], key: str, default: Any):
    val = cfg.get(key, default)
    return default if val is None else val


def main() -> None:
    args = _parse_args()
    cfg = load_yaml(args.config)

    data_cfg = cfg.get("data", {})
    embedding_cfg = cfg.get("embedding", {})
    model_cfg = cfg.get("model", {})
    runtime_cfg = cfg.get("runtime", {})

    device = torch.device(runtime_cfg.get("device", "cpu"))
    steps = int(runtime_cfg.get("steps", 1))
    log_every = int(runtime_cfg.get("log_every", 1))
    return_aux = bool(runtime_cfg.get("return_aux", False))

    metadata_path = Path(data_cfg["metadata_path"])
    feature_meta = build_model_feature_meta(metadata_path, embedding_cfg)

    loader = make_dataloader(
        split="train",
        batch_size=int(_get_cfg(data_cfg, "batch_size", 128)),
        num_workers=int(_get_cfg(data_cfg, "num_workers", 0)),
        shuffle=False,
        drop_last=bool(_get_cfg(data_cfg, "drop_last", False)),
        pin_memory=bool(_get_cfg(data_cfg, "pin_memory", True)),
        persistent_workers=bool(
            data_cfg["persistent_workers"] if "persistent_workers" in data_cfg else int(_get_cfg(data_cfg, "num_workers", 0)) > 0
        ),
        seed=data_cfg.get("seed"),
        feature_meta=feature_meta,
        debug=bool(_get_cfg(data_cfg, "debug", False)),
        neg_keep_prob_train=float(_get_cfg(data_cfg, "neg_keep_prob_train", 1.0)),
        prefetch_factor=(int(data_cfg["prefetch_factor"]) if "prefetch_factor" in data_cfg else 4),
        worker_cpu_threads=int(_get_cfg(data_cfg, "worker_cpu_threads", 1)),
        data_format=str(_get_cfg(data_cfg, "format", "vectorized")),
        processed_root=str(_get_cfg(data_cfg, "processed_dir", "data/processed")),
        vectorized_root=str(_get_cfg(data_cfg, "vectorized_dir", "data/vectorized")),
    )

    model = DeepFMBackbone(
        feature_meta=feature_meta,
        deep_hidden_dims=list(model_cfg.get("deep_hidden_dims", [])),
        deep_dropout=float(_get_cfg(model_cfg, "deep_dropout", 0.0)),
        deep_activation=str(_get_cfg(model_cfg, "deep_activation", "relu")),
        deep_use_bn=bool(_get_cfg(model_cfg, "deep_use_bn", False)),
        fm_enabled=bool(_get_cfg(model_cfg, "fm_enabled", True)),
        fm_projection_dim=(
            None
            if model_cfg.get("fm_projection_dim") is None
            else int(model_cfg.get("fm_projection_dim"))
        ),
        out_dim=int(_get_cfg(model_cfg, "out_dim", 128)),
    ).to(device)

    model.train()
    loader_iter = iter(loader)

    for step in range(steps):
        try:
            labels, features, meta = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            labels, features, meta = next(loader_iter)

        start = time.perf_counter()
        model.zero_grad(set_to_none=True)
        out = model(features, return_aux=return_aux)
        if return_aux:
            h = out["h"]
            logit_linear = out["logit_linear"]
        else:
            h = out
            logit_linear = None

        loss = h.float().mean()
        loss.backward()
        elapsed = time.perf_counter() - start

        if (step + 1) % log_every == 0:
            logit_info = (
                f", logit_linear shape={tuple(logit_linear.shape)}"
                if logit_linear is not None
                else ""
            )
            print(
                f"[step {step + 1}/{steps}] "
                f"h shape={tuple(h.shape)}{logit_info}, "
                f"loss={loss.item():.4f}, time={elapsed:.3f}s"
            )

    print("DeepFM backbone smoke test finished.")


if __name__ == "__main__":
    main()
