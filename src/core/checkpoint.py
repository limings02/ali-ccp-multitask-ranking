from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import nn
from torch.optim import Optimizer
import logging

logger = logging.getLogger(__name__)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Any | None,
    cfg: Dict[str, Any],
    step: int,
    best_metric: float | None = None,
    extra: Dict[str, Any] | None = None,
) -> None:
    """
    Save checkpoint atomically: write to tmp then rename to avoid partial files.
    """
    target = Path(path)
    tmp = target.with_suffix(target.suffix + ".tmp")

    state = {
        "model": model.state_dict(),
        "cfg": cfg,
        "step": int(step),
        "best_metric": best_metric,
        "extra": extra or {},
        "torch_version": torch.__version__,
    }
    if optimizer is not None:
        # OptimizerBundle path (preferred)
        if hasattr(optimizer, "state_dict") and hasattr(optimizer, "has_sparse"):
            optim_sd = optimizer.state_dict()
            state["optimizers"] = {
                "dense": optim_sd.get("dense"),
                "sparse": optim_sd.get("sparse"),
            }
            if hasattr(optimizer, "scaler_state_dict"):
                scaler_sd = optimizer.scaler_state_dict()
                if scaler_sd is not None:
                    state["scaler"] = scaler_sd
        elif hasattr(optimizer, "state_dict"):
            # Legacy single-optimizer format
            state["optimizer"] = optimizer.state_dict()

    torch.save(state, tmp)
    os.replace(tmp, target)  # atomic on Windows/POSIX


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Any | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load checkpoint to model (and optimizer if provided).

    Args:
        map_location: passed to torch.load to place tensors (e.g., "cpu" or "cuda:0").
        strict: forwarded to load_state_dict; False allows missing/unexpected keys.
    """
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Checkpoint not found: {target}")

    # weights_only=False to allow loading config/extra dicts saved alongside state dicts.
    chkpt = torch.load(target, map_location=map_location, weights_only=False)

    missing, unexpected = model.load_state_dict(chkpt["model"], strict=strict)
    if strict is False and (missing or unexpected):
        # Logically allow but surface info for debugging.
        print(f"load_checkpoint: missing keys={missing}, unexpected keys={unexpected}")
    elif strict and (missing or unexpected):
        raise RuntimeError(f"State dict mismatch. Missing: {missing}, Unexpected: {unexpected}")

    if optimizer is not None:
        if "optimizers" in chkpt and hasattr(optimizer, "load_state_dict"):
            try:
                optimizer.load_state_dict(chkpt["optimizers"])
                logger.info("load_checkpoint: loaded optimizer bundle (dense%s)", " + sparse" if chkpt["optimizers"].get("sparse") is not None else "")
            except Exception as exc:
                logger.warning(
                    "load_checkpoint: failed to restore optimizer bundle state (%s); "
                    "skip optimizer state restore and continue with freshly initialized optimizer.",
                    exc,
                )
        elif "optimizer" in chkpt and hasattr(optimizer, "load_state_dict"):
            try:
                optimizer.load_state_dict(chkpt["optimizer"])
                logger.info("load_checkpoint: loaded legacy optimizer into dense slot")
            except Exception as exc:
                logger.warning(
                    "load_checkpoint: failed to restore legacy optimizer state (%s); "
                    "skip optimizer state restore and continue with freshly initialized optimizer.",
                    exc,
                )
        else:
            logger.warning("load_checkpoint: no optimizer state found in checkpoint.")

        if "scaler" in chkpt and hasattr(optimizer, "load_scaler_state"):
            optimizer.load_scaler_state(chkpt.get("scaler"))
            logger.info("load_checkpoint: loaded scaler state")
        elif hasattr(optimizer, "load_scaler_state"):
            logger.info("load_checkpoint: scaler state not present or amp disabled, skipping.")

    return {
        "step": chkpt.get("step"),
        "best_metric": chkpt.get("best_metric"),
        "cfg": chkpt.get("cfg"),
        "extra": chkpt.get("extra", {}),
    }


def latest_checkpoint(run_dir: Path, pattern: str = "*.pt") -> Optional[Path]:
    """
    Return the most recently modified checkpoint matching pattern, or None.
    """
    candidates = list(run_dir.glob(pattern))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


__all__ = ["save_checkpoint", "load_checkpoint", "latest_checkpoint"]
