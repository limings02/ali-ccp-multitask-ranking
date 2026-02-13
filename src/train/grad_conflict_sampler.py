"""
Gradient Conflict Sampler for Multi-Task Learning Health Monitoring.

This module implements a systematic, configurable gradient conflict sampler that
collects cosine similarity, norm, and ratio metrics between task gradients over
the course of a training epoch.

Design rationale:
- **Dense vs sparse separation**: Embedding parameters with sparse_grad=True produce
  sparse gradient tensors. Their norms and cosine similarities have different scale and
  meaning compared to dense MLP/expert parameters. Mixing them into a single statistic
  would make the metric hard to interpret and potentially misleading. We therefore
  track them separately and report combined metrics alongside.
- **Skipping samples with no secondary-task gradient**: In ESMM mode, the CTCVR loss
  depends on both CTR and CVR heads; however, at some training steps the CVR-related
  gradient may be effectively zero (e.g., no clicked samples in the batch). Such samples
  should NOT contribute to cosine/norm statistics, but their count must be tracked so
  the user can diagnose sampling quality.
- **EMA smoothing for conflict_rate**: The per-step binary indicator "conflict or not"
  is very noisy. An exponential moving average provides a smooth running estimate that
  is more useful for comparing across epochs or experiments.
- **sample_interval auto-computation**: To reach `grad_samples_target` within a known
  number of training steps, we compute `sample_interval = max(1, expected_steps // target)`.
  This avoids the pitfall of a fixed `diag_every` that may over- or under-sample.

Usage:
    sampler = GradConflictSampler(grad_samples_target=1000)
    sampler.init_epoch(expected_steps=39000)
    for step in range(total_steps):
        if sampler.should_sample(step):
            try:
                metrics = grad_diag.compute_metrics(losses_by_task)
                sampler.record_sample(metrics, step, use_esmm=True)
            except Exception:
                sampler.record_error(step)
    result = sampler.aggregate(epoch=1, global_step=39000)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class GradConflictSampler:
    """
    Stateful sampler that accumulates gradient conflict metrics within one epoch.

    Parameters:
        grad_samples_target: desired number of gradient samples per epoch (>= 500).
        conflict_ema_alpha: smoothing factor for conflict_rate EMA (0 < alpha <= 1).
    """

    grad_samples_target: int = 1000
    conflict_ema_alpha: float = 0.1

    # --- internal state (reset per epoch via init_epoch) ---
    _sample_interval: int = field(default=1, init=False, repr=False)
    _expected_steps: int = field(default=0, init=False, repr=False)

    # Accumulators — dense cosine
    _cos_dense_vals: List[float] = field(default_factory=list, init=False, repr=False)
    # Accumulators — sparse cosine
    _cos_sparse_vals: List[float] = field(default_factory=list, init=False, repr=False)
    # Accumulators — combined cosine
    _cos_all_vals: List[float] = field(default_factory=list, init=False, repr=False)

    # Norms per task
    _norm_task1_vals: List[float] = field(default_factory=list, init=False, repr=False)
    _norm_task2_vals: List[float] = field(default_factory=list, init=False, repr=False)

    # Norm ratio (task2 / task1)
    _norm_ratio_vals: List[float] = field(default_factory=list, init=False, repr=False)

    # Conflict counters
    _conflict_hits: int = field(default=0, init=False, repr=False)
    _conflict_hits_dense: int = field(default=0, init=False, repr=False)
    _conflict_hits_sparse: int = field(default=0, init=False, repr=False)

    # Counts
    _collected: int = field(default=0, init=False, repr=False)
    _skipped_no_secondary_grad: int = field(default=0, init=False, repr=False)
    _error_count: int = field(default=0, init=False, repr=False)

    # Shared param counts from last sample
    _shared_dense_count: int = field(default=0, init=False, repr=False)
    _shared_sparse_count: int = field(default=0, init=False, repr=False)

    # EMA state
    _conflict_ema: Optional[float] = field(default=None, init=False, repr=False)

    # Task name tracking (for ESMM vs non-ESMM)
    _task1_name: str = field(default="ctr", init=False, repr=False)
    _task2_name: str = field(default="cvr", init=False, repr=False)

    # Track whether task2 ever had a valid gradient (for diagnostics)
    _task2_has_grad_count: int = field(default=0, init=False, repr=False)


    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def init_epoch(self, expected_steps: int) -> None:
        """
        Reset state and compute sample_interval for a new epoch.

        Args:
            expected_steps: estimated number of training steps in this epoch.
        """
        self._expected_steps = expected_steps
        self._sample_interval = max(1, expected_steps // self.grad_samples_target)

        # Reset accumulators
        self._cos_dense_vals.clear()
        self._cos_sparse_vals.clear()
        self._cos_all_vals.clear()
        self._norm_task1_vals.clear()
        self._norm_task2_vals.clear()
        self._norm_ratio_vals.clear()
        self._conflict_hits = 0
        self._conflict_hits_dense = 0
        self._conflict_hits_sparse = 0
        self._collected = 0
        self._skipped_no_secondary_grad = 0
        self._error_count = 0
        self._shared_dense_count = 0
        self._shared_sparse_count = 0
        self._conflict_ema = None
        self._task2_has_grad_count = 0

        logger.info(
            "[GradConflictSampler] init_epoch: expected_steps=%d, "
            "sample_interval=%d, target=%d",
            expected_steps, self._sample_interval, self.grad_samples_target,
        )

    def should_sample(self, step_in_epoch: int) -> bool:
        """Whether to collect a gradient conflict sample at this step."""
        return step_in_epoch % self._sample_interval == 0

    @property
    def sample_interval(self) -> int:
        return self._sample_interval

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_sample(
        self,
        diag_metrics: Dict[str, Any],
        global_step: int,
        use_esmm: bool = False,
    ) -> None:
        """
        Record one gradient diagnostics sample.

        Args:
            diag_metrics: output from GradientDiagnostics.compute_metrics()
            global_step: current global training step (for logging)
            use_esmm: if True, the second task is "ctcvr" instead of "cvr"
        """
        self._task1_name = "ctr"
        self._task2_name = "ctcvr" if use_esmm else "cvr"

        # Shared param counts
        self._shared_dense_count = diag_metrics.get("shared_dense_count", 0)
        self._shared_sparse_count = diag_metrics.get("shared_sparse_count", 0)

        # Norms
        norm_task1 = diag_metrics.get(f"grad_norm_shared_{self._task1_name}")
        norm_task2 = diag_metrics.get(f"grad_norm_shared_{self._task2_name}")

        # Check if task2 gradient is present / valid
        task2_has_grad = norm_task2 is not None and norm_task2 > 0
        if task2_has_grad:
            self._task2_has_grad_count += 1

        if norm_task1 is None and norm_task2 is None:
            # Both tasks have no gradient — skip entirely
            self._skipped_no_secondary_grad += 1
            logger.debug(
                "[GradConflictSampler] skip step=%d: both tasks have no grad",
                global_step,
            )
            return

        if not task2_has_grad:
            # Task2 (CVR/CTCVR) has no valid gradient — skip but count
            self._skipped_no_secondary_grad += 1
            logger.debug(
                "[GradConflictSampler] skip step=%d: %s has no grad (norm=%s)",
                global_step, self._task2_name, norm_task2,
            )
            return

        # --- Valid sample: record norms ---
        if norm_task1 is not None:
            self._norm_task1_vals.append(norm_task1)
        if norm_task2 is not None:
            self._norm_task2_vals.append(norm_task2)
        if (
            norm_task1 is not None
            and norm_task2 is not None
            and norm_task1 > 0
        ):
            self._norm_ratio_vals.append(norm_task2 / norm_task1)

        # --- Cosine similarities ---
        cos_dense = diag_metrics.get("grad_cosine_shared_dense")
        cos_sparse = diag_metrics.get("grad_cosine_shared_sparse")
        cos_all = diag_metrics.get("grad_cosine_shared_all")

        if cos_dense is not None:
            self._cos_dense_vals.append(cos_dense)
            if cos_dense < 0:
                self._conflict_hits_dense += 1

        if cos_sparse is not None:
            self._cos_sparse_vals.append(cos_sparse)
            if cos_sparse < 0:
                self._conflict_hits_sparse += 1

        if cos_all is not None:
            self._cos_all_vals.append(cos_all)
            if cos_all < 0:
                self._conflict_hits += 1
            # Update EMA
            is_conflict = 1.0 if cos_all < 0 else 0.0
            if self._conflict_ema is None:
                self._conflict_ema = is_conflict
            else:
                alpha = self.conflict_ema_alpha
                self._conflict_ema = alpha * is_conflict + (1 - alpha) * self._conflict_ema

        self._collected += 1

    def record_skip_no_secondary_grad(self, global_step: int) -> None:
        """Record a skip because the secondary task has no gradient."""
        self._skipped_no_secondary_grad += 1
        logger.debug(
            "[GradConflictSampler] record_skip step=%d: no secondary grad",
            global_step,
        )

    def record_error(self, global_step: int) -> None:
        """Record an error during gradient diagnostics computation."""
        self._error_count += 1
        logger.warning(
            "[GradConflictSampler] error at step=%d (total errors=%d)",
            global_step, self._error_count,
        )

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def aggregate(self, epoch: int, global_step: int) -> Dict[str, Any]:
        """
        Produce the final metrics dict for this epoch.

        Returns a dict ready to be merged into the train_one_epoch return value.
        All fields are JSON-serializable Python scalars or None.
        """
        import torch

        def _percentiles(vals: List[float], qs=(0.1, 0.5, 0.9)):
            if not vals:
                return [None] * len(qs)
            t = torch.tensor(vals, dtype=torch.float64)
            q_tensor = torch.tensor(qs, dtype=torch.float64)
            ps = torch.quantile(t, q_tensor)
            return [float(p.item()) for p in ps]

        def _safe_mean(vals: List[float]) -> Optional[float]:
            return sum(vals) / len(vals) if vals else None

        # Negative conflict strength: E[cos | cos < 0]
        neg_cos_all = [c for c in self._cos_all_vals if c < 0]
        neg_conflict_strength_mean = _safe_mean(neg_cos_all) if neg_cos_all else None

        # Percentiles
        cos_dense_p10, cos_dense_p50, cos_dense_p90 = _percentiles(self._cos_dense_vals)
        cos_sparse_p10, cos_sparse_p50, cos_sparse_p90 = _percentiles(self._cos_sparse_vals)
        cos_all_p10, cos_all_p50, cos_all_p90 = _percentiles(self._cos_all_vals)
        ratio_p50_val, = _percentiles(self._norm_ratio_vals, qs=(0.5,))
        ratio_p90_val, = _percentiles(self._norm_ratio_vals, qs=(0.9,))

        # Determine insufficiency reason if applicable
        insufficient_reason = None
        if self._collected < self.grad_samples_target:
            reasons = []
            if self._skipped_no_secondary_grad > 0:
                reasons.append(
                    f"skipped_no_{self._task2_name}_grad={self._skipped_no_secondary_grad}"
                )
            if self._error_count > 0:
                reasons.append(f"errors={self._error_count}")
            total_attempts = self._collected + self._skipped_no_secondary_grad + self._error_count
            if total_attempts < self.grad_samples_target:
                reasons.append(
                    f"total_sample_opportunities={total_attempts}"
                    f"<target={self.grad_samples_target}"
                    f"(expected_steps={self._expected_steps},"
                    f"interval={self._sample_interval})"
                )
            insufficient_reason = "; ".join(reasons) if reasons else "unknown"

        result = {
            # ---- Metadata ----
            "global_step": global_step,
            "grad_samples_target": self.grad_samples_target,
            "grad_samples_collected": self._collected,
            "grad_samples_skipped_no_cvr_grad": self._skipped_no_secondary_grad,
            "grad_sample_interval_steps": self._sample_interval,
            "grad_conflict_error_count": self._error_count,
            # Legacy compat: keep grad_samples = collected
            "grad_samples": self._collected,

            # ---- Norms (task-agnostic names for consistency) ----
            "grad_norm_shared_ctr_mean": _safe_mean(self._norm_task1_vals),
            # In ESMM mode: cvr norm is always None (we compute ctcvr).
            # Record the second task explicitly and also fill the legacy cvr field
            # with a clear indication.
            "grad_norm_shared_cvr_mean": (
                _safe_mean(self._norm_task2_vals)
                if self._task2_name == "cvr"
                else None
            ),
            "grad_norm_shared_ctcvr_mean": (
                _safe_mean(self._norm_task2_vals)
                if self._task2_name == "ctcvr"
                else None
            ),
            "grad_norm_ratio_mean": _safe_mean(self._norm_ratio_vals),
            "grad_norm_ratio_p50": ratio_p50_val,
            "grad_norm_ratio_p90": ratio_p90_val,

            # ---- Cosine dense ----
            "grad_cosine_shared_dense_mean": _safe_mean(self._cos_dense_vals),
            "grad_cosine_dense_p10": cos_dense_p10,
            "grad_cosine_dense_p50": cos_dense_p50,
            "grad_cosine_dense_p90": cos_dense_p90,

            # ---- Cosine sparse ----
            "grad_cosine_shared_sparse_mean": _safe_mean(self._cos_sparse_vals),
            "grad_cosine_sparse_p10": cos_sparse_p10,
            "grad_cosine_sparse_p50": cos_sparse_p50,
            "grad_cosine_sparse_p90": cos_sparse_p90,

            # ---- Cosine combined ----
            "grad_cosine_shared_mean": _safe_mean(self._cos_all_vals),
            "grad_cosine_p10": cos_all_p10,
            "grad_cosine_p50": cos_all_p50,
            "grad_cosine_p90": cos_all_p90,

            # ---- Conflict rates ----
            "conflict_rate": (
                self._conflict_hits / self._collected
                if self._collected > 0
                else None
            ),
            "conflict_rate_ema": self._conflict_ema,
            "neg_conflict_strength_mean": neg_conflict_strength_mean,

            # ---- Shared param counts ----
            "shared_dense_count": self._shared_dense_count if self._collected > 0 else None,
            "shared_sparse_count": self._shared_sparse_count if self._collected > 0 else None,

            # ---- CVR/CTCVR grad availability ----
            "cvr_has_grad": self._task2_has_grad_count > 0,
            "cvr_valid_count": self._task2_has_grad_count,
        }

        # Log insufficiency warning
        if insufficient_reason is not None and self._collected < self.grad_samples_target:
            logger.warning(
                "[GradConflictSampler] epoch=%d: collected %d/%d samples "
                "(insufficient reason: %s)",
                epoch, self._collected, self.grad_samples_target, insufficient_reason,
            )
            result["grad_samples_insufficient_reason"] = insufficient_reason

        # Log summary
        logger.info(
            "[GradConflictSampler] epoch=%d summary: collected=%d/%d, "
            "skipped_no_grad=%d, errors=%d, interval=%d, "
            "conflict_rate=%.3f, ema=%.3f, cos_all_p50=%s",
            epoch,
            self._collected,
            self.grad_samples_target,
            self._skipped_no_secondary_grad,
            self._error_count,
            self._sample_interval,
            result["conflict_rate"] if result["conflict_rate"] is not None else float("nan"),
            self._conflict_ema if self._conflict_ema is not None else float("nan"),
            cos_all_p50,
        )

        return result


__all__ = ["GradConflictSampler"]
