from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Callable, TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch.autograd.profiler import record_function
from torch.nn.utils import clip_grad_norm_
from torch import amp as torch_amp
from src.eval.metrics import compute_binary_metrics
from src.train.grad_diag import GradientDiagnostics
from src.train.grad_conflict_sampler import GradConflictSampler
from src.utils.health_metrics import (
    compute_label_counts,
    compute_logit_stats,
    compute_prob_stats,
    aggregate_gate_metrics,
)

if TYPE_CHECKING:
    from src.train.optim import LRSchedulerBundle

EPS = 1e-12
_diag_logger = logging.getLogger(__name__)


def _to_device_labels(
    labels: Dict[str, torch.Tensor],
    device: torch.device,
    non_blocking: bool = False,
) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=non_blocking) for k, v in labels.items()}


def _to_device_features(
    features: Dict[str, Any],
    device: torch.device,
    non_blocking: bool = False,
) -> Dict[str, Any]:
    """
    Move indices/offsets/weights to target device; keep structure identical.
    """
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


def _select_shared_params(model, fallback_cap: int = 50) -> List[Tuple[str, torch.nn.Parameter]]:
    """
    选择共享参数用于梯度诊断。
    
    策略（按优先级）：
    1. 优先选择 embedding 参数（name 包含 emb/embedding）
    2. 然后选择 backbone 下的其他参数
    3. 最后选择其他非 head 参数（如 experts、gates、composer 等）
    
    排除规则：
    - 排除 heads.ctr.* / heads.cvr.* 等 head 专属参数
    - 排除不需要梯度的参数
    
    返回两类参数：
    - dense 参数：正常的权重矩阵
    - sparse 参数：embedding 参数（可能有 sparse_grad=True）
    
    注意：此函数是 health metrics 的关键依赖，若返回空列表则所有健康指标为 null。
    """
    # 收集所有候选共享参数（排除 heads）
    shared_candidates: List[Tuple[str, torch.nn.Parameter]] = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        name_low = name.lower()
        
        # 排除 head 专属参数（heads.ctr, heads.cvr, heads.ctcvr 等）
        if "heads." in name_low:
            # 进一步检查是否是特定任务的 head
            if any(f"heads.{task}" in name_low for task in ["ctr", "cvr", "ctcvr"]):
                continue
        
        shared_candidates.append((name, param))
    
    if not shared_candidates:
        _diag_logger.warning(
            "_select_shared_params: No shared parameters found! "
            "This will cause all health metrics to be null."
        )
        return []
    
    # 按优先级排序：embedding > backbone > 其他 (experts/gates/composer)
    def priority(item: Tuple[str, torch.nn.Parameter]) -> int:
        name_low = item[0].lower()
        if "emb" in name_low or "embedding" in name_low:
            return 0  # 最高优先级
        elif name_low.startswith("backbone"):
            return 1
        else:
            return 2  # experts, gates, composer 等
    
    shared_candidates.sort(key=priority)
    
    # 限制返回数量以控制计算开销
    result = shared_candidates[:fallback_cap]
    
    # 统计 dense 和 sparse 参数数量用于日志
    dense_count = sum(1 for _, p in result if not getattr(p, 'is_sparse', False))
    _diag_logger.debug(
        "_select_shared_params: Selected %d shared params (%d dense), "
        "total candidates %d, first few: %s",
        len(result), dense_count, len(shared_candidates),
        [n for n, _ in result[:5]]
    )
    
    return result


def _flatten_grads(params: Sequence[Tuple[str, torch.nn.Parameter]]) -> Tuple[torch.Tensor | None, int, int]:
    """
    Flatten dense gradients into a 1D tensor.

    Returns:
        flat_grad: concatenated dense grads or None if empty
        num_dense: number of dense grads included
        num_sparse_skipped: number of sparse grads skipped
    """
    grads = []
    num_sparse_skipped = 0
    num_dense = 0
    for _, p in params:
        if p.grad is None:
            continue
        if p.grad.is_sparse:
            num_sparse_skipped += 1
            continue
        num_dense += 1
        grads.append(p.grad.detach().reshape(-1))
    if not grads:
        return None, num_dense, num_sparse_skipped
    return torch.cat(grads), num_dense, num_sparse_skipped


def _compute_grad_percentiles(values: List[float]) -> Tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    tensor_vals = torch.tensor(values, device="cpu")
    p10, p50, p90 = torch.quantile(tensor_vals, torch.tensor([0.1, 0.5, 0.9]))
    return float(p10.item()), float(p50.item()), float(p90.item())


def train_one_epoch(
    model,
    loader,
    optimizer_bundle,
    loss_fn,
    device,
    logger,
    epoch: int,
    max_steps: Optional[int] = None,
    grad_clip_norm: Optional[float] = None,
    log_every: int = 50,
    global_step: int = 0,
    grad_diag_enabled: bool = False,
    grad_diag_every: Optional[int] = None,  # If None, uses log_every
    grad_diag_min_tasks: int = 2,
    eval_every_steps: Optional[int] = None,
    validate_fn: Optional[Callable[[], Dict[str, float]]] = None,
    eval_callback: Optional[Callable[[Dict[str, float], int, int], None]] = None,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    scaler: Optional[torch_amp.GradScaler] = None,
    amp_device_type: str = "cuda",
    debug_logit_every: Optional[int] = None,
    lr_scheduler_bundle: Optional["LRSchedulerBundle"] = None,
    cfg: Optional[Dict[str, Any]] = None,
    expert_health_diag: Optional[Any] = None,  # ExpertHealthDiagnostics
    grad_samples_target: int = 1000,
    conflict_ema_alpha: float = 0.1,
) -> Dict[str, float]:
    """
    Train for one epoch and emit aggregated metrics plus optional gradient diagnostics.

    Gradient diagnostics use the new dynamic shared parameter detection that supports
    both dense and sparse gradients. Diagnostics run at the cadence specified by
    grad_diag_every (defaults to log_every).
    """
    model.train()
    enabled_heads = getattr(loss_fn, "enabled_heads", {"ctr", "cvr"})
    use_esmm = bool(getattr(loss_fn, "use_esmm", False))
    esmm_version = str(getattr(loss_fn, "esmm_version", "v2"))
    has_ctr = "ctr" in enabled_heads
    has_cvr = "cvr" in enabled_heads

    tot_loss = tot_ctr = tot_cvr = tot_mask = 0.0
    tot_ctr_raw = tot_cvr_raw = 0.0
    tot_ctr_scaled = tot_cvr_scaled = 0.0
    steps = 0
    n_rows = 0
    n_masked = 0.0
    pos_masked = 0.0

    # ESMM-specific metrics accumulators
    tot_p_ctr_mean = tot_p_cvr_mean = tot_p_ctcvr_mean = 0.0
    tot_pos_rate_ctr = tot_pos_rate_cvr = tot_pos_rate_ctcvr = 0.0
    tot_loss_cvr_aux = 0.0
    esmm_metric_count = 0
    cvr_aux_count = 0

    # ============================================================
    # Gradient conflict sampler (replaces old inline accumulators)
    # Computes sample_interval from expected epoch steps to reach target.
    # ============================================================
    grad_diag: Optional[GradientDiagnostics] = None
    grad_sampler: Optional[GradConflictSampler] = None
    if grad_diag_enabled:
        grad_diag = GradientDiagnostics(
            model,
            min_tasks=grad_diag_min_tasks,
            cache_refresh="epoch",
        )
        grad_diag.set_epoch(epoch)
        grad_sampler = GradConflictSampler(
            grad_samples_target=grad_samples_target,
            conflict_ema_alpha=conflict_ema_alpha,
        )
        # Estimate expected steps for this epoch
        if max_steps is not None:
            expected_steps = min(len(loader), max_steps - global_step)
        else:
            expected_steps = len(loader)
        grad_sampler.init_epoch(max(1, expected_steps))
    ctr_raw_count = cvr_raw_count = 0

    # ============================================================
    # pos_weight_clip schedule configuration (改动 B)
    # ============================================================
    pos_weight_clip_schedule_cfg = {}
    pos_weight_clip_schedule_enabled = False
    if cfg is not None:
        loss_cfg = cfg.get("loss", {}) or {}
        pos_weight_clip_schedule_cfg = loss_cfg.get("pos_weight_clip_schedule", {}) or {}
        pos_weight_clip_schedule_enabled = bool(pos_weight_clip_schedule_cfg.get("enabled", False))
    
    # Store original clip values to restore at end (if schedule modifies them dynamically)
    original_pos_weight_clip_ctcvr = getattr(loss_fn, "pos_weight_clip_ctcvr", None)
    original_pos_weight_clip_ctr = getattr(loss_fn, "pos_weight_clip_ctr", None)
    
    # Log schedule config at epoch start if enabled
    if pos_weight_clip_schedule_enabled and step == 0 if 'step' in dir() else True:
        _diag_logger.info(
            "[pos_weight_clip_schedule] enabled=true type=%s target=%s milestones=%s values=%s",
            pos_weight_clip_schedule_cfg.get("type", "piecewise"),
            pos_weight_clip_schedule_cfg.get("target", "ctcvr"),
            pos_weight_clip_schedule_cfg.get("milestones"),
            pos_weight_clip_schedule_cfg.get("values"),
        )

    # LR scheduler log flag
    lr_scheduler_enabled = lr_scheduler_bundle is not None and lr_scheduler_bundle.enabled
    last_logged_lr: Dict[str, float] = {}
    ctr_scaled_count = cvr_scaled_count = 0
    
    # ===== Data cursor tracking for resume =====
    data_state_updates: Dict[int, Dict] = {}  # Maps worker_id -> {last_row_id, batch_size, ...}

    # Performance monitoring
    data_load_time = 0.0
    forward_time = 0.0
    backward_time = 0.0
    optim_time = 0.0
    next_batch_time = 0.0
    collate_time = 0.0
    h2d_time = 0.0
    train_step_time = 0.0
    h2d_non_blocking = bool(device.type == "cuda" and getattr(loader, "pin_memory", False))

    t_start = time.time()
    t_batch_start = time.perf_counter()

    for step, (labels, features, meta) in enumerate(loader):
        t_data_end = time.perf_counter()
        t_next_batch = t_data_end - t_batch_start
        data_load_time += t_next_batch
        next_batch_time += t_next_batch
        meta_perf = meta.get("_perf") if isinstance(meta, dict) else None
        if isinstance(meta_perf, dict):
            collate_ms = meta_perf.get("collate_ms")
            if collate_ms is not None:
                collate_time += float(collate_ms) / 1000.0
        t_step_start = time.perf_counter()
        
        # Update global_step in loss_fn for aux_focal warmup control
        current_global_step = global_step + step
        expert_diag_should_log_train = (
            expert_health_diag is not None
            and expert_health_diag.should_log(current_global_step, "train")
        )
        
        # Check if we've reached max_train_steps (for early stopping or resume)
        if max_steps is not None and current_global_step >= max_steps:
            break
        
        if hasattr(loss_fn, "global_step"):
            loss_fn.global_step = current_global_step

        # Update step in model for gate schedule (PLE temperature/noise decay)
        if hasattr(model, "set_step"):
            # 尝试从 lr_scheduler_bundle 获取 total_steps
            total_steps = getattr(lr_scheduler_bundle, "total_steps", 0) if lr_scheduler_bundle else 0
            if total_steps == 0 and cfg:
                # fallback: 从 cfg 中读取
                total_steps = cfg.get("optim", {}).get("lr_scheduler", {}).get("total_steps", 0)
            if total_steps > 0:
                model.set_step(current_global_step, total_steps)

        with record_function("train.h2d_copy"):
            t_h2d_start = time.perf_counter()
            labels = _to_device_labels(labels, device, non_blocking=h2d_non_blocking)
            features_dev = _to_device_features(features, device, non_blocking=h2d_non_blocking)
            h2d_time += time.perf_counter() - t_h2d_start

        optimizer_bundle.zero_grad(set_to_none=True)
        # Autocast only on CUDA to avoid CPU precision issues.
        t_forward_start = time.time()
        with torch_amp.autocast(device_type=amp_device_type, enabled=amp_enabled, dtype=amp_dtype):
            outputs = model(features_dev)
            batch = {"labels": labels, "features": features_dev, "meta": meta}
            loss, loss_dict = loss_fn.compute(outputs, batch)
            
            # ============================================================
            # 改动 C: Add gate regularization loss from MMoE
            # gate_reg_loss is computed in model.forward() and stored in aux
            # Only add during training; validation logs stats but doesn't add to loss
            # ============================================================
            aux = outputs.get("aux", {}) or {}
            gate_reg_loss = aux.get("gate_reg_loss")
            if gate_reg_loss is not None:
                loss = loss + gate_reg_loss
                loss_dict["gate_reg_loss"] = float(gate_reg_loss.detach().cpu().item())
                gate_entropy_mean = aux.get("gate_entropy_mean")
                gate_lb_kl = aux.get("gate_lb_kl")
                if gate_entropy_mean is not None:
                    loss_dict["gate_entropy_mean"] = float(gate_entropy_mean.cpu().item())
                if gate_lb_kl is not None:
                    loss_dict["gate_lb_kl"] = float(gate_lb_kl.cpu().item())
            
            # ============================================================
            # 新增: shared mass floor metrics（gate_shared_mass_mean_{task}, gate_shared_mass_floor_loss_{task}）
            # ============================================================
            for task in ["ctr", "cvr"]:
                mass_mean_key = f"gate_shared_mass_mean_{task}"
                mass_loss_key = f"gate_shared_mass_floor_loss_{task}"
                if mass_mean_key in aux:
                    loss_dict[mass_mean_key] = float(aux[mass_mean_key])
                if mass_loss_key in aux:
                    loss_dict[mass_loss_key] = float(aux[mass_loss_key])
            
            # ============================================================
            # 专家健康诊断: 收集 gate weights（在 backward 之前）
            # ============================================================
            if expert_diag_should_log_train:
                gates = aux.get("gates")
                if gates:
                    for task, gate_w in gates.items():
                        expert_health_diag.collect_gate_weights(task, gate_w)
        
        t_forward_end = time.time()
        forward_time += t_forward_end - t_forward_start

        # Debug-only logit stats (no effect on gradients).
        # 注意：使用 step（当前批次索引，0-based）计算 current step for debug
        debug_current_step = global_step + step + 1  # 1-based step for debugging
        should_debug_logits = (
            debug_logit_every is not None
            and debug_logit_every > 0
            and (debug_current_step % debug_logit_every == 0 or debug_current_step == 1)
        )
        if should_debug_logits and hasattr(logger, "info"):
            with torch.no_grad():
                def _stats(t: Optional[torch.Tensor]):
                    if t is None:
                        return None
                    flat = t.detach().float().reshape(-1)
                    if flat.numel() == 0:
                        return None
                    qs = torch.quantile(flat, torch.tensor([0.99, 0.999], device=flat.device))
                    return {
                        "mean": float(flat.mean().item()),
                        "std": float(flat.std().item()),
                        "abs_max": float(flat.abs().max().item()),
                        "p99": float(qs[0].item()),
                        "p999": float(qs[1].item()),
                    }

                ctr_logit = outputs.get("ctr")
                ctr_parts = outputs.get("ctr_logit_parts", {}) if isinstance(outputs, dict) else {}
                fm_logit = ctr_parts.get("fm")
                wide_logit = ctr_parts.get("wide")
                deep_logit = ctr_parts.get("deep")
                stats_total = _stats(ctr_logit)
                stats_fm = _stats(fm_logit)
                stats_wide = _stats(wide_logit)
                stats_deep = _stats(deep_logit)

                # Per-field wide debug collected inside backbone when enabled.
                linear_debug = getattr(getattr(model, "backbone", model), "last_linear_debug", None)
                fm_debug = getattr(getattr(model, "backbone", model), "last_fm_debug", None)
                log_parts = [
                    f"[dbg_logit] step={debug_current_step}",
                    f"ctr_total={stats_total}",
                    f"wide={stats_wide}",
                    f"fm={stats_fm}",
                    f"deep={stats_deep}",
                ]
                if linear_debug:
                    log_parts.append(f"wide_fields={linear_debug}")
                if fm_debug:
                    log_parts.append(f"fm_stats={fm_debug}")
                pos_w = {
                    "pos_weight_ctr": loss_dict.get("pos_weight_ctr"),
                    "pos_weight_cvr": loss_dict.get("pos_weight_cvr"),
                    "ctr_pos_rate": float(labels["y_ctr"].mean().item()),
                    "cvr_pos_rate": float(labels["y_cvr"].mean().item()),
                }
                log_parts.append(f"labels={pos_w}")
                logger.info(" ".join(log_parts))

        mask_cvr_sum_val_raw = loss_dict.get("mask_cvr_sum", None)
        mask_cvr_sum_val = None
        if mask_cvr_sum_val_raw is not None:
            if isinstance(mask_cvr_sum_val_raw, torch.Tensor):
                mask_cvr_sum_val = float(mask_cvr_sum_val_raw.item())
            else:
                mask_cvr_sum_val = float(mask_cvr_sum_val_raw)

        log_loss_debug = ((step + 1) % log_every == 0) or not loss.requires_grad
        if log_loss_debug and hasattr(logger, "debug"):
            logger.debug(
                "loss_debug step=%d type=%s requires_grad=%s device=%s mask_cvr_sum=%s enabled_heads=%s",
                step + 1,
                type(loss).__name__,
                getattr(loss, "requires_grad", None),
                getattr(loss, "device", None),
                mask_cvr_sum_val_raw,
                enabled_heads,
            )

        skip_cvr_only_empty_mask = has_cvr and (not has_ctr) and (mask_cvr_sum_val == 0.0)
        if skip_cvr_only_empty_mask:
            if log_loss_debug and hasattr(logger, "debug"):
                logger.debug("skip backward/step at step=%d because mask_cvr_sum==0 in CVR-only batch", step + 1)
        else:
            t_backward_start = time.time()
            if amp_enabled and scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                if grad_clip_norm is not None:
                    # unscale before clipping, otherwise clipping would operate on scaled grads.
                    optimizer_bundle.unscale_(scaler)
                    clip_grad_norm_(optimizer_bundle.dense_params, grad_clip_norm)
                t_backward_end = time.time()
                backward_time += t_backward_end - t_backward_start
                t_optim_start = time.time()
                optimizer_bundle.step(scaler)
                t_optim_end = time.time()
                optim_time += t_optim_end - t_optim_start
            else:
                loss.backward()
                if grad_clip_norm is not None:
                    clip_grad_norm_(optimizer_bundle.dense_params, grad_clip_norm)
                t_backward_end = time.time()
                backward_time += t_backward_end - t_backward_start
                t_optim_start = time.time()
                optimizer_bundle.step()
                t_optim_end = time.time()
                optim_time += t_optim_end - t_optim_start
            
            # ============================================================
            # LR Scheduler step (改动 A: step-based scheduling)
            # ============================================================
            if lr_scheduler_enabled:
                lr_scheduler_bundle.step()
                last_logged_lr = lr_scheduler_bundle.get_last_lr()
            
            # ============================================================
            # 专家健康诊断: 收集梯度信息（在 backward 之后、zero_grad 之前）
            # ============================================================
            if expert_diag_should_log_train:
                # 在 backward 完成后收集梯度诊断并记录
                expert_health_diag.compute_and_log(
                    step=current_global_step,
                    epoch=epoch,
                    phase="train",
                    extra_meta={"batch_step": step},
                )
                expert_health_diag.reset()

        # ============================================================
        # pos_weight_clip schedule update (改动 B)
        # Update clip values based on current_global_step
        # ============================================================
        if pos_weight_clip_schedule_enabled:
            from src.loss.bce import _effective_pos_weight
            
            sch_type = str(pos_weight_clip_schedule_cfg.get("type", "piecewise")).lower()
            sch_target = str(pos_weight_clip_schedule_cfg.get("target", "ctcvr")).lower()
            
            def _compute_scheduled_clip(step: int) -> float:
                if sch_type == "piecewise":
                    milestones = pos_weight_clip_schedule_cfg.get("milestones", [0])
                    values = pos_weight_clip_schedule_cfg.get("values", [400])
                    # Find the current value based on step
                    current_val = values[0] if values else 400.0
                    for i, milestone in enumerate(milestones):
                        if step >= milestone and i < len(values):
                            current_val = values[i]
                    return float(current_val)
                elif sch_type == "linear":
                    start_step = int(pos_weight_clip_schedule_cfg.get("start_step", 0))
                    end_step = int(pos_weight_clip_schedule_cfg.get("end_step", 40000))
                    start_value = float(pos_weight_clip_schedule_cfg.get("start_value", 400))
                    end_value = float(pos_weight_clip_schedule_cfg.get("end_value", 100))
                    if step <= start_step:
                        return start_value
                    elif step >= end_step:
                        return end_value
                    else:
                        progress = (step - start_step) / max(end_step - start_step, 1)
                        return start_value + (end_value - start_value) * progress
                else:
                    return 400.0  # fallback
            
            current_clip = _compute_scheduled_clip(current_global_step)
            
            # Apply to target(s)
            if sch_target in ("ctcvr", "both"):
                loss_fn.pos_weight_clip_ctcvr = current_clip
            if sch_target in ("ctr", "both"):
                loss_fn.pos_weight_clip_ctr = current_clip
            
            # Compute effective pos_weight for logging
            raw_ctcvr = getattr(loss_fn, "static_pos_weight_ctcvr", None) or 1.0
            effective_ctcvr, _ = _effective_pos_weight(raw_ctcvr, current_clip)
            loss_dict["pos_weight_clip_ctcvr_scheduled"] = current_clip
            loss_dict["effective_pos_weight_ctcvr"] = effective_ctcvr
            loss_dict["pos_weight_ctcvr_raw"] = raw_ctcvr

        # aggregate stats
        B = int(labels["y_ctr"].shape[0])
        n_rows += B
        if has_cvr:
            mask = labels["click_mask"]
            n_masked += float(mask.sum().item())
            pos_masked += float((labels["y_cvr"] * (mask > 0.5)).sum().item())

        tot_loss += loss_dict["loss_total"]
        tot_ctr += loss_dict.get("loss_ctr", 0.0)
        tot_cvr += loss_dict.get("loss_cvr", 0.0)
        tot_mask += loss_dict.get("mask_cvr_sum", 0.0)
        if loss_dict.get("loss_ctr_raw") is not None:
            tot_ctr_raw += loss_dict["loss_ctr_raw"]
            ctr_raw_count += 1
        if loss_dict.get("loss_cvr_raw") is not None:
            tot_cvr_raw += loss_dict["loss_cvr_raw"]
            cvr_raw_count += 1
        if loss_dict.get("loss_ctr_scaled") is not None:
            tot_ctr_scaled += loss_dict["loss_ctr_scaled"]
            ctr_scaled_count += 1
        if loss_dict.get("loss_cvr_scaled") is not None:
            tot_cvr_scaled += loss_dict["loss_cvr_scaled"]
            cvr_scaled_count += 1

        # Accumulate ESMM-specific metrics
        if use_esmm:
            if loss_dict.get("p_ctr_mean") is not None:
                tot_p_ctr_mean += loss_dict["p_ctr_mean"]
                tot_p_cvr_mean += loss_dict.get("p_cvr_mean", 0.0)
                tot_p_ctcvr_mean += loss_dict.get("p_ctcvr_mean", 0.0)
                tot_pos_rate_ctr += loss_dict.get("pos_rate_ctr", 0.0)
                tot_pos_rate_ctcvr += loss_dict.get("pos_rate_ctcvr", 0.0)
                if loss_dict.get("pos_rate_cvr") is not None:
                    tot_pos_rate_cvr += loss_dict["pos_rate_cvr"]
                esmm_metric_count += 1
            if loss_dict.get("loss_cvr_aux") is not None:
                tot_loss_cvr_aux += loss_dict["loss_cvr_aux"]
                cvr_aux_count += 1

        steps += 1
        
        # Reset batch timer for next iteration
        train_step_time += time.perf_counter() - t_step_start
        t_batch_start = time.perf_counter()

        # ============================================================
        # Gradient Conflict Sampling (auto-interval to reach target)
        # current_global_step 来自 global_step + step
        # ============================================================
        should_sample_grad = (
            grad_diag_enabled
            and grad_diag is not None
            and grad_sampler is not None
            and has_ctr
            and has_cvr
            and grad_sampler.should_sample(step)
        )
        if should_sample_grad:
            try:
                # Zero gradients before diagnostics
                optimizer_bundle.zero_grad(set_to_none=True)

                # Keep diagnostics in full precision
                with torch_amp.autocast(enabled=False, device_type=amp_device_type):
                    diag_outputs = model(features_dev)

                    # Build per-task losses for gradient computation.
                    # Use the same task weights as training loss so grad diagnostics
                    # are sensitive to lambda/weight changes.
                    loss_ctr_diag_raw = F.binary_cross_entropy_with_logits(
                        diag_outputs["ctr"].float(), labels["y_ctr"].float(), reduction="mean"
                    )

                    # For ESMM: use CTCVR loss; for non-ESMM: use masked CVR loss
                    if use_esmm:
                        log_p_ctr = F.logsigmoid(diag_outputs["ctr"].float())
                        log_p_cvr = F.logsigmoid(diag_outputs["cvr"].float())
                        log_p_ctcvr = log_p_ctr + log_p_cvr
                        y_ctcvr = labels["y_ctcvr"].float()
                        loss_ctcvr_diag_raw = -(
                            y_ctcvr * log_p_ctcvr
                            + (1 - y_ctcvr) * torch.log1p(-torch.exp(log_p_ctcvr).clamp(max=1 - 1e-7))
                        ).mean()
                        lambda_ctr_diag = float(getattr(loss_fn, "lambda_ctr", 1.0))
                        lambda_ctcvr_diag = float(getattr(loss_fn, "lambda_ctcvr", 1.0))
                        loss_ctr_diag = lambda_ctr_diag * loss_ctr_diag_raw
                        loss_ctcvr_diag = lambda_ctcvr_diag * loss_ctcvr_diag_raw
                        losses_by_task = {"ctr": loss_ctr_diag, "ctcvr": loss_ctcvr_diag}
                    else:
                        mask = labels["click_mask"].float()
                        mask_sum = mask.sum()
                        if mask_sum > 0:
                            loss_vec = F.binary_cross_entropy_with_logits(
                                diag_outputs["cvr"].float(), labels["y_cvr"].float(), reduction="none"
                            )
                            loss_cvr_diag_raw = (loss_vec * mask).sum() / (mask_sum + EPS)
                        else:
                            loss_cvr_diag_raw = torch.tensor(0.0, device=device, requires_grad=True)
                        w_ctr_diag = float(getattr(loss_fn, "w_ctr", 1.0))
                        w_cvr_diag = float(getattr(loss_fn, "w_cvr", 1.0))
                        loss_ctr_diag = w_ctr_diag * loss_ctr_diag_raw
                        loss_cvr_diag = w_cvr_diag * loss_cvr_diag_raw
                        losses_by_task = {"ctr": loss_ctr_diag, "cvr": loss_cvr_diag}

                # Compute gradient metrics
                diag_metrics = grad_diag.compute_metrics(losses_by_task)

                # Record sample in sampler (handles dense/sparse/conflict tracking)
                grad_sampler.record_sample(
                    diag_metrics, global_step=current_global_step, use_esmm=use_esmm
                )

                # Zero gradients after diagnostics
                optimizer_bundle.zero_grad(set_to_none=True)

            except Exception as e:
                _diag_logger.warning(
                    "grad_diag: error during computation at global_step=%d: %s",
                    current_global_step, e, exc_info=True
                )
                if grad_sampler is not None:
                    grad_sampler.record_error(current_global_step)
                optimizer_bundle.zero_grad(set_to_none=True)
        
        # ===== Collect data cursor for resume support =====
        if isinstance(meta, dict) and "_data_cursor" in meta:
            cursor_info = meta["_data_cursor"]
            worker_id = cursor_info.get("worker_id", 0)
            data_state_updates[worker_id] = cursor_info

        if (step + 1) % log_every == 0:
            elapsed = time.time() - t_start
            steps_per_sec = (step + 1) / elapsed if elapsed > 0 else 0.0
            samples_per_sec = n_rows / elapsed if elapsed > 0 else 0.0
            
            # Calculate timing percentages
            total_time = data_load_time + forward_time + backward_time + optim_time
            data_pct = 100 * data_load_time / total_time if total_time > 0 else 0
            forward_pct = 100 * forward_time / total_time if total_time > 0 else 0
            backward_pct = 100 * backward_time / total_time if total_time > 0 else 0
            optim_pct = 100 * optim_time / total_time if total_time > 0 else 0
            
            log_parts = [
                f"[train] epoch={epoch} step={step+1} global_step={current_global_step}",
                f"loss_total={loss_dict['loss_total']:.4f}",
            ]
            log_parts.append(f"mode={'esmm' if use_esmm else 'legacy'}")
            if has_ctr:
                log_parts.append(f"loss_ctr={loss_dict.get('loss_ctr', 0.0):.4f}")
            else:
                log_parts.append("loss_ctr=0.0000(disabled)")
            if has_cvr:
                if use_esmm:
                    log_parts.append(f"loss_ctcvr={loss_dict.get('loss_cvr', 0.0):.4f}")
                    log_parts.append(f"n_exposure_batch={loss_dict.get('mask_cvr_sum', 0.0):.1f}")
                    log_parts.append("aliases(loss_cvr->loss_ctcvr,mask_sum->n_exposure_batch)")
                else:
                    log_parts.append(f"loss_cvr={loss_dict.get('loss_cvr', 0.0):.4f}")
                    log_parts.append(f"mask_sum={loss_dict.get('mask_cvr_sum', 0.0):.1f}")
            else:
                log_parts.append("loss_cvr=0.0000(disabled)")
            if has_ctr:
                mode = loss_dict.get("pos_weight_mode")
                if mode:
                    log_parts.append(f"pos_weight_mode={mode}")
                ctr_pw = loss_dict.get("pos_weight_ctr")
                if ctr_pw is not None:
                    log_parts.append(f"ctr_pos_w={ctr_pw:.3f}")
                ctr_pw_raw = loss_dict.get("pos_weight_ctr_raw")
                if ctr_pw_raw is not None:
                    log_parts.append(f"ctr_pos_w_raw={ctr_pw_raw:.3f}")
                ctr_pw_eff = loss_dict.get("pos_weight_ctr_effective")
                if ctr_pw_eff is not None:
                    log_parts.append(f"ctr_pos_w_eff={ctr_pw_eff:.3f}")
                ctcvr_pw_raw = loss_dict.get("pos_weight_ctcvr_raw")
                if ctcvr_pw_raw is not None:
                    log_parts.append(f"ctcvr_pos_w_raw={ctcvr_pw_raw:.3f}")
                ctcvr_pw_eff = loss_dict.get("pos_weight_ctcvr_effective")
                if ctcvr_pw_eff is not None:
                    log_parts.append(f"ctcvr_pos_w_eff={ctcvr_pw_eff:.3f}")
            
            # ============================================================
            # 改动 A: Log current learning rate
            # ============================================================
            if lr_scheduler_enabled and last_logged_lr:
                lr_dense = last_logged_lr.get("lr_dense")
                lr_sparse = last_logged_lr.get("lr_sparse")
                if lr_dense is not None:
                    log_parts.append(f"dense_lr={lr_dense:.6g}")
                if lr_sparse is not None:
                    log_parts.append(f"sparse_lr={lr_sparse:.6g}")
            
            # ============================================================
            # 改动 B: Log pos_weight_clip schedule info
            # ============================================================
            if pos_weight_clip_schedule_enabled:
                ctcvr_clip = loss_dict.get("pos_weight_clip_ctcvr_scheduled")
                effective_ctcvr = loss_dict.get("effective_pos_weight_ctcvr")
                raw_ctcvr = loss_dict.get("pos_weight_ctcvr_raw")
                if ctcvr_clip is not None:
                    log_parts.append(f"ctcvr_clip={ctcvr_clip:.1f}")
                if effective_ctcvr is not None:
                    log_parts.append(f"effective_pos_w_ctcvr={effective_ctcvr:.2f}")
                if raw_ctcvr is not None:
                    log_parts.append(f"raw_pos_w_ctcvr={raw_ctcvr:.2f}")
            
            # ============================================================
            # 改动 C: Log gate regularization metrics
            # ============================================================
            gate_reg_loss_val = loss_dict.get("gate_reg_loss")
            gate_entropy_mean = loss_dict.get("gate_entropy_mean")
            gate_lb_kl = loss_dict.get("gate_lb_kl")
            if gate_reg_loss_val is not None:
                log_parts.append(f"gate_reg_loss={gate_reg_loss_val:.6f}")
            if gate_entropy_mean is not None:
                log_parts.append(f"gate_entropy_mean={gate_entropy_mean:.4f}")
            if gate_lb_kl is not None:
                log_parts.append(f"gate_lb_kl={gate_lb_kl:.6f}")
            
            # ============================================================
            # 新增: Log shared mass floor metrics
            # ============================================================
            for task in ["ctr", "cvr"]:
                mass_mean = loss_dict.get(f"gate_shared_mass_mean_{task}")
                mass_loss = loss_dict.get(f"gate_shared_mass_floor_loss_{task}")
                if mass_mean is not None:
                    log_parts.append(f"mass_shared_{task}={mass_mean:.4f}")
                if mass_loss is not None and mass_loss > 1e-8:
                    log_parts.append(f"mass_floor_loss_{task}={mass_loss:.6f}")
            
            # ============================================================
            # Residual Head: Log residual diagnostics
            # ============================================================
            if loss_dict.get("residual_enabled"):
                r_abs_mean = loss_dict.get("residual_r_abs_mean")
                r_abs_p95 = loss_dict.get("residual_r_abs_p95")
                corr_ctr = loss_dict.get("corr_r_ctr")
                corr_cvr = loss_dict.get("corr_r_cvr")
                if r_abs_mean is not None:
                    log_parts.append(f"r_abs_mean={r_abs_mean:.4f}")
                if r_abs_p95 is not None:
                    log_parts.append(f"r_abs_p95={r_abs_p95:.4f}")
                if corr_ctr is not None:
                    log_parts.append(f"corr_r_ctr={corr_ctr:.3f}")
                if corr_cvr is not None:
                    log_parts.append(f"corr_r_cvr={corr_cvr:.3f}")
                loss_res_reg = loss_dict.get("loss_res_reg")
                if loss_res_reg is not None and loss_res_reg > 1e-8:
                    log_parts.append(f"loss_res_reg={loss_res_reg:.6f}")
                loss_anchor = loss_dict.get("loss_anchor")
                if loss_anchor is not None and loss_anchor > 1e-8:
                    log_parts.append(f"loss_anchor={loss_anchor:.6f}")
            
            # Add performance metrics
            log_parts.append(f"samples={n_rows} steps/s={steps_per_sec:.2f} samples/s={samples_per_sec:.1f}")
            log_parts.append(f"time={elapsed:.2f}s")
            log_parts.append(f"timing(data={data_load_time:.1f}s[{data_pct:.0f}%] fwd={forward_time:.1f}s[{forward_pct:.0f}%] bwd={backward_time:.1f}s[{backward_pct:.0f}%] opt={optim_time:.1f}s[{optim_pct:.0f}%])")
            log_parts.append(
                f"timing_ext(next_batch={next_batch_time:.1f}s collate={collate_time:.1f}s h2d={h2d_time:.1f}s train_step={train_step_time:.1f}s)"
            )
            
            logger.info(" ".join(log_parts))
            
            # Reset timing counters after logging
            data_load_time = forward_time = backward_time = optim_time = 0.0
            next_batch_time = collate_time = h2d_time = train_step_time = 0.0

        if eval_every_steps and validate_fn and (current_global_step > 0 and current_global_step % eval_every_steps == 0):
            # Mid-epoch validation at the same cadence as training logs.
            # validate_fn must switch the model to eval/no_grad internally; we switch back to train afterwards.
            val_metrics = validate_fn()
            model.train()
            if eval_callback:
                eval_callback(val_metrics, current_global_step, epoch)
            if val_metrics and hasattr(logger, "info"):
                auc_primary = val_metrics.get("auc_primary")
                auc_ctr = val_metrics.get("auc_ctr")
                auc_cvr = val_metrics.get("auc_cvr")
                log_parts = [
                    f"[eval] epoch={epoch} step={step+1} global_step={current_global_step}",
                    f"loss_total={val_metrics.get('loss_total'):.4f}",
                ]
                if auc_primary is not None:
                    log_parts.append(f"auc_primary={auc_primary:.4f}")
                if auc_ctr is not None:
                    log_parts.append(f"auc_ctr={auc_ctr:.4f}")
                if auc_cvr is not None:
                    log_parts.append(f"auc_cvr={auc_cvr:.4f}")
                logger.info(" ".join(log_parts))

    dur = time.time() - t_start
    if steps == 0:
        return {}

    mean_ctr = tot_ctr / steps if has_ctr else None
    mean_cvr = tot_cvr / steps if has_cvr else None
    mean_ctr_raw = (tot_ctr_raw / ctr_raw_count) if ctr_raw_count > 0 else (None if has_ctr else None)
    mean_cvr_raw = (tot_cvr_raw / cvr_raw_count) if cvr_raw_count > 0 else (None if has_cvr else None)
    mean_ctr_scaled = (tot_ctr_scaled / ctr_scaled_count) if ctr_scaled_count > 0 else (None if has_ctr else None)
    mean_cvr_scaled = (tot_cvr_scaled / cvr_scaled_count) if cvr_scaled_count > 0 else (None if has_cvr else None)
    loss_ratio_scaled = None
    if mean_ctr_scaled is not None and mean_cvr_scaled is not None:
        loss_ratio_scaled = float(mean_cvr_scaled / (mean_ctr_scaled + EPS))

    # ============================================================
    # Aggregate gradient conflict metrics from sampler
    # ============================================================
    final_global_step = global_step + steps
    if grad_sampler is not None and grad_diag_enabled:
        grad_metrics = grad_sampler.aggregate(epoch=epoch, global_step=final_global_step)
    else:
        # No gradient diagnostics — fill with None for schema consistency
        grad_metrics = {
            "global_step": final_global_step,
            "grad_samples_target": None,
            "grad_samples_collected": None,
            "grad_samples_skipped_no_cvr_grad": None,
            "grad_sample_interval_steps": None,
            "grad_conflict_error_count": None,
            "grad_samples": None,
            "grad_norm_shared_ctr_mean": None,
            "grad_norm_shared_cvr_mean": None,
            "grad_norm_shared_ctcvr_mean": None,
            "grad_norm_ratio_mean": None,
            "grad_norm_ratio_p50": None,
            "grad_norm_ratio_p90": None,
            "grad_cosine_shared_dense_mean": None,
            "grad_cosine_dense_p10": None,
            "grad_cosine_dense_p50": None,
            "grad_cosine_dense_p90": None,
            "grad_cosine_shared_sparse_mean": None,
            "grad_cosine_sparse_p10": None,
            "grad_cosine_sparse_p50": None,
            "grad_cosine_sparse_p90": None,
            "grad_cosine_shared_mean": None,
            "grad_cosine_p10": None,
            "grad_cosine_p50": None,
            "grad_cosine_p90": None,
            "conflict_rate": None,
            "conflict_rate_ema": None,
            "neg_conflict_strength_mean": None,
            "shared_dense_count": None,
            "shared_sparse_count": None,
            "cvr_has_grad": None,
            "cvr_valid_count": None,
        }

    # ESMM-specific metrics
    esmm_metrics = {}
    if use_esmm and esmm_metric_count > 0:
        esmm_metrics = {
            "p_ctr_mean": tot_p_ctr_mean / esmm_metric_count,
            "p_cvr_mean": tot_p_cvr_mean / esmm_metric_count,
            "p_ctcvr_mean": tot_p_ctcvr_mean / esmm_metric_count,
            "pos_rate_ctr": tot_pos_rate_ctr / esmm_metric_count,
            "pos_rate_cvr": tot_pos_rate_cvr / esmm_metric_count if tot_pos_rate_cvr > 0 else None,
            "pos_rate_ctcvr": tot_pos_rate_ctcvr / esmm_metric_count,
            "loss_cvr_aux": (tot_loss_cvr_aux / cvr_aux_count) if cvr_aux_count > 0 else None,
            "esmm_version": esmm_version,
        }

    # ============================================================
    # 改动 A: Add lr_dense/lr_sparse to metrics
    # ============================================================
    lr_metrics = {}
    if lr_scheduler_enabled and last_logged_lr:
        lr_metrics["lr_dense"] = last_logged_lr.get("lr_dense")
        lr_metrics["lr_sparse"] = last_logged_lr.get("lr_sparse")
    
    # ============================================================
    # 改动 B: Add pos_weight_clip schedule info to metrics
    # ============================================================
    clip_schedule_metrics = {}
    if pos_weight_clip_schedule_enabled:
        clip_schedule_metrics["pos_weight_clip_ctcvr"] = loss_dict.get("pos_weight_clip_ctcvr_scheduled")
        clip_schedule_metrics["effective_pos_weight_ctcvr"] = loss_dict.get("effective_pos_weight_ctcvr")
        clip_schedule_metrics["pos_weight_ctcvr_raw"] = loss_dict.get("pos_weight_ctcvr_raw")
    
    # ============================================================
    # 改动 C: Add gate stabilization metrics
    # ============================================================
    gate_metrics = {}
    if loss_dict.get("gate_reg_loss") is not None:
        gate_metrics["gate_reg_loss"] = loss_dict.get("gate_reg_loss")
        gate_metrics["gate_entropy_mean"] = loss_dict.get("gate_entropy_mean")
        gate_metrics["gate_lb_kl"] = loss_dict.get("gate_lb_kl")

    return {
        "loss_total": tot_loss / steps,
        "loss_ctr": mean_ctr if has_ctr else 0.0,
        "loss_cvr": mean_cvr if has_cvr else 0.0,
        "loss_ctr_raw": mean_ctr_raw,
        "loss_cvr_raw": mean_cvr_raw,
        "loss_ctr_scaled": mean_ctr_scaled,
        "loss_cvr_scaled": mean_cvr_scaled,
        "loss_ratio_scaled": loss_ratio_scaled,
        "mask_cvr_sum": tot_mask / steps if has_cvr else 0.0,
        "steps": steps,
        "n_rows": n_rows,
        "n_masked_train": int(n_masked) if has_cvr else None,
        "pos_masked_train": int(pos_masked) if has_cvr else None,
        "steps_per_sec": steps / dur if dur > 0 else 0.0,
        **grad_metrics,
        **esmm_metrics,
        **lr_metrics,
        **clip_schedule_metrics,
        **gate_metrics,
        # ===== Include data state for resume support =====
        "data_state": {"data_state": data_state_updates} if data_state_updates else None,
    }


def validate(
    model,
    loader,
    loss_fn,
    device,
    logger,
    epoch: int,
    max_steps: Optional[int] = None,
    log_every: int = 200,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    amp_device_type: str = "cuda",
    calc_auc: bool = False,
    log_health_metrics: bool = False,
    expert_health_diag: Optional[Any] = None,  # ExpertHealthDiagnostics
) -> Dict[str, float]:
    model.eval()
    enabled_heads = getattr(loss_fn, "enabled_heads", {"ctr", "cvr"})
    use_esmm = bool(getattr(loss_fn, "use_esmm", False))
    esmm_eps = float(getattr(loss_fn, "esmm_eps", 1e-8))
    has_ctr = "ctr" in enabled_heads
    has_cvr = "cvr" in enabled_heads

    tot_loss = tot_ctr = tot_cvr = tot_mask = 0.0
    tot_ctr_raw = tot_cvr_raw = 0.0
    tot_ctr_scaled = tot_cvr_scaled = 0.0
    ctr_raw_count = cvr_raw_count = 0
    ctr_scaled_count = cvr_scaled_count = 0
    steps = 0
    n_rows = 0
    n_masked = 0.0
    pos_masked = 0.0
    t_start = time.time()

    y_ctr_list: List[torch.Tensor] = []
    y_cvr_list: List[torch.Tensor] = []
    y_ctcvr_list: List[torch.Tensor] = []
    click_mask_list: List[torch.Tensor] = []
    ctr_logit_list: List[torch.Tensor] = []
    cvr_logit_list: List[torch.Tensor] = []
    ctcvr_logit_list: List[torch.Tensor] = []

    # Health metrics collection (only when log_health_metrics=True)
    gate_ctr_list: List[torch.Tensor] = []
    gate_cvr_list: List[torch.Tensor] = []
    p_ctcvr_list: List[torch.Tensor] = []  # For CTCVR prob stats

    # ============================================================
    # Residual Head: 额外收集 residual 版本的 CTCVR logit
    # 用于同时输出 ctcvr_auc_mul 和 ctcvr_auc_res
    # ============================================================
    ctcvr_logit_res_list: List[torch.Tensor] = []
    residual_enabled = bool(getattr(loss_fn, "residual_enabled", False))
    h2d_non_blocking = bool(device.type == "cuda" and getattr(loader, "pin_memory", False))

    with torch.no_grad():
        for step, (labels, features, meta) in enumerate(loader):
            if max_steps is not None and step >= max_steps:
                break

            labels = _to_device_labels(labels, device, non_blocking=h2d_non_blocking)
            features_dev = _to_device_features(features, device, non_blocking=h2d_non_blocking)

            with torch_amp.autocast(device_type=amp_device_type, enabled=amp_enabled, dtype=amp_dtype):
                outputs = model(features_dev)
                batch = {"labels": labels, "features": features_dev, "meta": meta}
                loss, loss_dict = loss_fn.compute(outputs, batch)

            B = int(labels["y_ctr"].shape[0])
            n_rows += B
            if has_cvr:
                mask = labels["click_mask"]
                n_masked += float(mask.sum().item())
                pos_masked += float((labels["y_cvr"] * (mask > 0.5)).sum().item())

            if calc_auc:
                if has_ctr:
                    ctr_logit = outputs["ctr"]
                    if ctr_logit.dim() > 1:
                        ctr_logit = ctr_logit.view(-1)
                    ctr_logit_list.append(ctr_logit.detach().cpu())
                    y_ctr_list.append(labels["y_ctr"].detach().cpu())
                if has_cvr:
                    if use_esmm:
                        cvr_logit = outputs.get("cvr")
                        if cvr_logit is None:
                            raise KeyError("cvr logit missing in outputs during validation")
                        if cvr_logit.dim() > 1:
                            cvr_logit = cvr_logit.view(-1)
                        cvr_logit_list.append(cvr_logit.detach().cpu())
                        y_cvr_list.append(labels["y_cvr"].detach().cpu())
                        click_mask_list.append(labels["y_ctr"].detach().cpu())  # click subset mask

                        # Derived CTCVR for ESMM from p_ctr * p_cvr
                        ctr_logit = outputs["ctr"]
                        if ctr_logit.dim() > 1:
                            ctr_logit = ctr_logit.view(-1)
                        p_ctr = torch.sigmoid(ctr_logit)
                        p_cvr = torch.sigmoid(cvr_logit)
                        p_ctcvr = torch.clamp(p_ctr * p_cvr, min=esmm_eps, max=1.0 - esmm_eps)
                        ctcvr_logit = torch.log(p_ctcvr / (1.0 - p_ctcvr))
                        ctcvr_logit_list.append(ctcvr_logit.detach().cpu())
                        y_ctcvr_list.append(labels["y_ctcvr"].detach().cpu())

                        # Residual Head: 计算 residual 版本的 CTCVR logit
                        # ctcvr_logit_res = ctr_logit + cvr_logit + r_logit (logit 空间加法)
                        if residual_enabled and "residual_logit" in outputs:
                            r_logit = outputs["residual_logit"].detach()
                            if r_logit.dim() == 2:
                                r_logit = r_logit.squeeze(-1)
                            ctcvr_logit_res = ctr_logit + cvr_logit + r_logit
                            ctcvr_logit_res_list.append(ctcvr_logit_res.cpu())

                        # Collect p_ctcvr for health metrics
                        if log_health_metrics:
                            p_ctcvr_list.append(p_ctcvr.detach().cpu())
                    else:
                        cvr_logit = outputs["cvr"]
                        if cvr_logit.dim() > 1:
                            cvr_logit = cvr_logit.view(-1)
                        cvr_logit_list.append(cvr_logit.detach().cpu())
                        y_cvr_list.append(labels["y_cvr"].detach().cpu())
                        click_mask_list.append(labels["click_mask"].detach().cpu())

            # Collect gate weights for health metrics (MMoE only)
            if log_health_metrics and calc_auc:
                aux = outputs.get("aux", {})
                gates = aux.get("gates", {})
                if "ctr" in gates:
                    gate_ctr_list.append(gates["ctr"].detach().cpu())
                if "cvr" in gates:
                    gate_cvr_list.append(gates["cvr"].detach().cpu())
            
            # 专家健康诊断: 收集 gate weights（验证阶段）
            if expert_health_diag is not None and expert_health_diag.config.log_on_valid:
                aux = outputs.get("aux", {})
                gates = aux.get("gates", {})
                for task, gate_w in gates.items():
                    expert_health_diag.collect_gate_weights(task, gate_w)

            tot_loss += loss_dict["loss_total"]
            tot_ctr += loss_dict.get("loss_ctr", 0.0)
            tot_cvr += loss_dict.get("loss_cvr", 0.0)
            tot_mask += loss_dict.get("mask_cvr_sum", 0.0)
            if loss_dict.get("loss_ctr_raw") is not None:
                tot_ctr_raw += loss_dict["loss_ctr_raw"]
                ctr_raw_count += 1
            if loss_dict.get("loss_cvr_raw") is not None:
                tot_cvr_raw += loss_dict["loss_cvr_raw"]
                cvr_raw_count += 1
            if loss_dict.get("loss_ctr_scaled") is not None:
                tot_ctr_scaled += loss_dict["loss_ctr_scaled"]
                ctr_scaled_count += 1
            if loss_dict.get("loss_cvr_scaled") is not None:
                tot_cvr_scaled += loss_dict["loss_cvr_scaled"]
                cvr_scaled_count += 1
            steps += 1

            if (step + 1) % log_every == 0:
                elapsed = time.time() - t_start
                log_parts = [
                    f"[val] epoch={epoch} step={step+1}",
                    f"loss_total={loss_dict['loss_total']:.4f}",
                ]
                log_parts.append(f"mode={'esmm' if use_esmm else 'legacy'}")
                if has_ctr:
                    log_parts.append(f"loss_ctr={loss_dict.get('loss_ctr', 0.0):.4f}")
                else:
                    log_parts.append("loss_ctr=0.0000(disabled)")
                if has_cvr:
                    if use_esmm:
                        log_parts.append(f"loss_ctcvr={loss_dict.get('loss_cvr', 0.0):.4f}")
                        log_parts.append(f"n_exposure_batch={loss_dict.get('mask_cvr_sum', 0.0):.1f}")
                        log_parts.append("aliases(loss_cvr->loss_ctcvr,mask_sum->n_exposure_batch)")
                    else:
                        log_parts.append(f"loss_cvr={loss_dict.get('loss_cvr', 0.0):.4f}")
                        log_parts.append(f"mask_sum={loss_dict.get('mask_cvr_sum', 0.0):.1f}")
                else:
                    log_parts.append("loss_cvr=0.0000(disabled)")
                if has_ctr:
                    mode = loss_dict.get("pos_weight_mode")
                    if mode:
                        log_parts.append(f"pos_weight_mode={mode}")
                    ctr_pw = loss_dict.get("pos_weight_ctr")
                    if ctr_pw is not None:
                        log_parts.append(f"ctr_pos_w={ctr_pw:.3f}")
                    ctr_pw_raw = loss_dict.get("pos_weight_ctr_raw")
                    if ctr_pw_raw is not None:
                        log_parts.append(f"ctr_pos_w_raw={ctr_pw_raw:.3f}")
                    ctr_pw_eff = loss_dict.get("pos_weight_ctr_effective")
                    if ctr_pw_eff is not None:
                        log_parts.append(f"ctr_pos_w_eff={ctr_pw_eff:.3f}")
                    ctcvr_pw_raw = loss_dict.get("pos_weight_ctcvr_raw")
                    if ctcvr_pw_raw is not None:
                        log_parts.append(f"ctcvr_pos_w_raw={ctcvr_pw_raw:.3f}")
                    ctcvr_pw_eff = loss_dict.get("pos_weight_ctcvr_effective")
                    if ctcvr_pw_eff is not None:
                        log_parts.append(f"ctcvr_pos_w_eff={ctcvr_pw_eff:.3f}")
                log_parts.append(f"time={elapsed:.2f}s")
                logger.info(" ".join(log_parts))

    dur = time.time() - t_start
    if steps == 0:
        return {}

    mean_ctr = tot_ctr / steps if has_ctr else None
    mean_cvr = tot_cvr / steps if has_cvr else None
    mean_ctr_raw = (tot_ctr_raw / ctr_raw_count) if ctr_raw_count > 0 else (None if has_ctr else None)
    mean_cvr_raw = (tot_cvr_raw / cvr_raw_count) if cvr_raw_count > 0 else (None if has_cvr else None)
    mean_ctr_scaled = (tot_ctr_scaled / ctr_scaled_count) if ctr_scaled_count > 0 else (None if has_ctr else None)
    mean_cvr_scaled = (tot_cvr_scaled / cvr_scaled_count) if cvr_scaled_count > 0 else (None if has_cvr else None)
    loss_ratio_scaled = None
    if mean_ctr_scaled is not None and mean_cvr_scaled is not None:
        loss_ratio_scaled = float(mean_cvr_scaled / (mean_ctr_scaled + EPS))

    auc_ctr = auc_cvr = auc_ctcvr = auc_primary = None
    auc_ctcvr_mul = auc_ctcvr_res = None  # Residual Head: 双版本 CTCVR AUC
    if calc_auc:
        auc_vals = []
        ctr_metrics = cvr_metrics = ctcvr_metrics = None
        if has_ctr and y_ctr_list:
            ctr_metrics = compute_binary_metrics(
                torch.cat(y_ctr_list).numpy(), torch.cat(ctr_logit_list).numpy()
            )
            auc_ctr = ctr_metrics.get("auc")
            if auc_ctr is not None:
                auc_vals.append(auc_ctr)
        if has_cvr and y_cvr_list:
            if click_mask_list and cvr_logit_list:
                mask_arr = torch.cat(click_mask_list).numpy()
                cvr_metrics = compute_binary_metrics(
                    torch.cat(y_cvr_list).numpy(),
                    torch.cat(cvr_logit_list).numpy(),
                    mask=mask_arr > 0.5,
                )
                auc_cvr = cvr_metrics.get("auc")
            if auc_cvr is not None:
                auc_vals.append(auc_cvr)
        if use_esmm and y_ctcvr_list and ctcvr_logit_list:
            ctcvr_metrics = compute_binary_metrics(
                torch.cat(y_ctcvr_list).numpy(), torch.cat(ctcvr_logit_list).numpy()
            )
            auc_ctcvr = ctcvr_metrics.get("auc")
            # ctcvr_logit_list 始终是乘法版本 (p_ctr * p_cvr)
            auc_ctcvr_mul = auc_ctcvr

            # ============================================================
            # Residual Head: 额外计算 residual 版本的 CTCVR AUC
            # 用于对比乘法版本和残差修正版本的效果
            # ============================================================
            if residual_enabled and ctcvr_logit_res_list:
                ctcvr_res_metrics = compute_binary_metrics(
                    torch.cat(y_ctcvr_list).numpy(),
                    torch.cat(ctcvr_logit_res_list).numpy(),
                )
                auc_ctcvr_res = ctcvr_res_metrics.get("auc")
                # 当 residual 启用时, 主 CTCVR AUC 使用 residual 版本
                if auc_ctcvr_res is not None:
                    auc_ctcvr = auc_ctcvr_res
        if auc_vals:
            if use_esmm and auc_ctr is not None and auc_cvr is not None and auc_ctcvr is not None:
                # ESMM mode: weighted combination
                auc_primary = float(1.0 * auc_ctcvr + 0.2 * auc_ctr + 0.2 * auc_cvr)
            else:
                # Non-ESMM mode: simple average
                auc_primary = float(sum(auc_vals) / len(auc_vals))

    # Health monitoring metrics (only computed when log_health_metrics=True and calc_auc=True)
    health_metrics: Dict[str, Any] = {}
    if log_health_metrics and calc_auc:
        # A) Label counts and reliability flags
        if y_ctr_list and y_cvr_list and click_mask_list:
            all_y_ctr = torch.cat(y_ctr_list)
            all_y_cvr = torch.cat(y_cvr_list)
            all_click_mask = torch.cat(click_mask_list)
            all_y_ctcvr = torch.cat(y_ctcvr_list) if y_ctcvr_list else None
            label_counts = compute_label_counts(
                y_ctr=all_y_ctr,
                y_cvr=all_y_cvr,
                click_mask=all_click_mask,
                y_ctcvr=all_y_ctcvr,
            )
            health_metrics.update(label_counts)

        # B) Logit/Prob distribution stats
        if ctr_logit_list:
            all_ctr_logit = torch.cat(ctr_logit_list)
            health_metrics.update(compute_logit_stats(all_ctr_logit, "logit_ctr"))

        if cvr_logit_list:
            all_cvr_logit = torch.cat(cvr_logit_list)
            health_metrics.update(compute_logit_stats(all_cvr_logit, "logit_cvr"))

        # CTCVR probability stats (ESMM mode)
        if p_ctcvr_list:
            all_p_ctcvr = torch.cat(p_ctcvr_list)
            health_metrics.update(compute_prob_stats(all_p_ctcvr, "prob_ctcvr"))

        # C) MMoE gate health metrics
        if gate_ctr_list:
            health_metrics.update(aggregate_gate_metrics(gate_ctr_list, "ctr"))
        if gate_cvr_list:
            health_metrics.update(aggregate_gate_metrics(gate_cvr_list, "cvr"))
    
    # 专家健康诊断: 验证阶段结束时记录
    if expert_health_diag is not None and expert_health_diag.config.log_on_valid:
        expert_health_diag.compute_and_log(
            step=0,  # 验证阶段不关注 step
            epoch=epoch,
            phase="valid",
            extra_meta={"n_rows": n_rows, "steps": steps},
        )
        expert_health_diag.reset()

    return {
        "loss_total": tot_loss / steps,
        "loss_ctr": mean_ctr if has_ctr else 0.0,
        "loss_cvr": mean_cvr if has_cvr else 0.0,
        "loss_ctr_raw": mean_ctr_raw,
        "loss_cvr_raw": mean_cvr_raw,
        "loss_ctr_scaled": mean_ctr_scaled,
        "loss_cvr_scaled": mean_cvr_scaled,
        "loss_ratio_scaled": loss_ratio_scaled,
        "mask_cvr_sum": tot_mask / steps if has_cvr else 0.0,
        "steps": steps,
        "n_rows": n_rows,
        "n_masked_train": int(n_masked) if has_cvr else None,
        "pos_masked_train": int(pos_masked) if has_cvr else None,
        "steps_per_sec": steps / dur if dur > 0 else 0.0,
        "grad_norm_shared_ctr_mean": None,
        "grad_norm_shared_cvr_mean": None,
        "grad_norm_shared_ctcvr_mean": None,
        "grad_norm_ratio_mean": None,
        "grad_norm_ratio_p50": None,
        "grad_norm_ratio_p90": None,
        "grad_cosine_shared_dense_mean": None,
        "grad_cosine_dense_p10": None,
        "grad_cosine_dense_p50": None,
        "grad_cosine_dense_p90": None,
        "grad_cosine_shared_sparse_mean": None,
        "grad_cosine_sparse_p10": None,
        "grad_cosine_sparse_p50": None,
        "grad_cosine_sparse_p90": None,
        "grad_cosine_shared_mean": None,
        "grad_cosine_p10": None,
        "grad_cosine_p50": None,
        "grad_cosine_p90": None,
        "conflict_rate": None,
        "conflict_rate_ema": None,
        "neg_conflict_strength_mean": None,
        "grad_samples": None,
        "grad_samples_target": None,
        "grad_samples_collected": None,
        "grad_samples_skipped_no_cvr_grad": None,
        "grad_sample_interval_steps": None,
        "grad_conflict_error_count": None,
        "shared_dense_count": None,
        "shared_sparse_count": None,
        "cvr_has_grad": None,
        "cvr_valid_count": None,
        "auc_ctr": auc_ctr,
        "auc_cvr": auc_cvr,
        "auc_ctcvr": auc_ctcvr,
        "auc_ctcvr_mul": auc_ctcvr_mul,  # 乘法版本 CTCVR AUC (基线对比)
        "auc_ctcvr_res": auc_ctcvr_res,  # 残差修正版本 CTCVR AUC (residual head)
        "auc_primary": auc_primary,
        **health_metrics,
    }


__all__ = ["train_one_epoch", "validate"]
