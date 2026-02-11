from __future__ import annotations

import warnings
from typing import Any, Dict, Tuple, Optional

import torch
import torch.nn.functional as F

from src.loss.base import LossFn, get_labels


# ============================================================================
# Effective Pos-Weight Calculation (统一 clip 逻辑)
# ============================================================================

def _effective_pos_weight(raw: float, clip: float | None) -> tuple[float, bool]:
    """
    计算有效的 pos_weight，统一裁剪逻辑。
    
    Args:
        raw: 原始 pos_weight 值
        clip: 裁剪上限，如果为 None 或 <= 0 则不裁剪
        
    Returns:
        (effective_pos_weight, clipped_flag)
        - effective_pos_weight: 裁剪后的有效值
        - clipped_flag: 是否发生了裁剪
    
    注意：此函数是 pos_weight 生效的唯一入口，确保 loss 计算与日志打印使用同一套逻辑。
    """
    if clip is None or clip <= 0:
        return raw, False
    if raw > clip:
        return clip, True
    return raw, False


def _compute_dynamic_pos_weight(
    y: torch.Tensor,
    clip: float | None,
    eps: float = 1e-6,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, float, float]:
    """
    动态计算 pos_weight = neg_count / pos_count，并应用 clip。
    
    Args:
        y: 标签 tensor {0, 1}
        clip: 裁剪上限，如果为 None 或 <= 0 则不裁剪
        eps: 防除零的最小值
        device: 输出 tensor 的设备（默认使用 y.device）
        dtype: 输出 tensor 的数据类型
    
    Returns:
        (pos_weight_tensor, pos_weight_raw, pos_weight_effective)
        - pos_weight_tensor: 用于传入 BCE loss 的 tensor
        - pos_weight_raw: 裁剪前的原始值（float）
        - pos_weight_effective: 裁剪后的有效值（float）
    """
    pos_count = torch.clamp(y.sum(), min=eps)
    neg_count = torch.clamp((1.0 - y).sum(), min=eps)
    pw_raw = float((neg_count / pos_count).item())
    
    # 应用 clip
    pw_effective, _ = _effective_pos_weight(pw_raw, clip)
    
    # 返回 tensor 和 float
    pw_tensor = torch.tensor(pw_effective, device=device or y.device, dtype=dtype)
    return pw_tensor, pw_raw, pw_effective


# ============================================================================
# Auxiliary Focal Loss (Logits Version)
# ============================================================================

def focal_on_logits_aux(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: Optional[float] = None,
    detach_p_for_weight: bool = True,
    compute_fp32: bool = True,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Auxiliary Focal Loss computed from logits for numerical stability and AMP compatibility.
    
    Focal loss downweights easy examples (high confidence correct predictions):
    - For y=1 (positive): easy when p≈1 → (1-p) small → focal_factor≈0 → loss↓
    - For y=0 (negative): easy when p≈0 → (1-p)≈1 and p_t=(1-p)≈1 → focal_factor≈0 → loss↓
    
    The key insight: easy negatives (y=0, p≈0) get focal_factor = (1 - p_t)^gamma ≈ 0,
    strongly downweighting them. Hard examples retain full loss weight.
    
    Args:
        logits: Model raw logits (before sigmoid), shape [N] or [N, ...]
        targets: Binary labels {0, 1}, shape [N] or [N, ...]
        gamma: Focal loss focusing parameter. Higher gamma → stronger downweighting of easy examples.
               Typical: 1.0 or 2.0. gamma=0 → standard BCE.
        alpha: Optional class balancing factor. If provided:
               - Positives weighted by alpha
               - Negatives weighted by (1-alpha)
               Typical: 0.25 for positive class. None to disable.
        detach_p_for_weight: If True, detach focal_factor from gradient graph to avoid
                            extra gradient coupling. Recommended True for auxiliary losses.
        compute_fp32: If True, compute focal weights in fp32 for numerical stability under AMP.
        reduction: "mean", "sum", or "none".
    
    Returns:
        Focal loss tensor (scalar if reduction!="none").
    
    Implementation:
        1. BCE(logits, y) = -y*log(p) - (1-y)*log(1-p), computed via F.binary_cross_entropy_with_logits
        2. p_t = p*y + (1-p)*(1-y)  [probability of the true class]
        3. focal_factor = (1 - p_t)^gamma
        4. If detach_p_for_weight: focal_factor = focal_factor.detach()
        5. If alpha: alpha_t = alpha*y + (1-alpha)*(1-y); loss = alpha_t * focal_factor * BCE
           Else: loss = focal_factor * BCE
    
    Note:
        - This function does NOT apply pos_weight (to avoid double-weighting with main BCE).
        - For CTCVR auxiliary loss in ESMM: use this alongside BCEWithLogits(pos_weight=...).
    """
    # Convert targets to float {0, 1}
    targets = targets.float()
    
    # Compute in fp32 for stability if requested (important under AMP)
    if compute_fp32:
        logits_w = logits.float()
        targets_w = targets.float()
    else:
        logits_w = logits
        targets_w = targets
    
    # Standard BCE without pos_weight (reduction="none" to get per-sample losses)
    ce = F.binary_cross_entropy_with_logits(logits_w, targets_w, reduction="none")
    
    # Compute probability p = sigmoid(logits)
    p = torch.sigmoid(logits_w)
    
    # Compute p_t: probability assigned to the true class
    # p_t = p when y=1, p_t = (1-p) when y=0
    p_t = p * targets_w + (1.0 - p) * (1.0 - targets_w)
    
    # Focal factor: (1 - p_t)^gamma
    # Clamp p_t to [0, 1] for safety (should already be in range, but numerical safety)
    focal_factor = (1.0 - p_t.clamp(0.0, 1.0)).pow(gamma)
    
    # Optionally detach focal_factor to prevent gradient flow through it
    if detach_p_for_weight:
        focal_factor = focal_factor.detach()
    
    # Apply alpha balancing if provided
    if alpha is not None:
        # alpha_t = alpha for positives, (1-alpha) for negatives
        alpha_t = alpha * targets_w + (1.0 - alpha) * (1.0 - targets_w)
        loss = alpha_t * focal_factor * ce
    else:
        loss = focal_factor * ce
    
    # Reduction
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}")


# ============================================================================
# Numerically stable helpers for log-domain ESMM
# ============================================================================

def log1mexp(x: torch.Tensor) -> torch.Tensor:
    """
    Computes log(1 - exp(x)) in a numerically stable way for x <= 0.
    
    Reference: "Accurately Computing log(1 - exp(-|a|))" by Martin Maechler
    https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    
    For x close to 0 (x > -0.6931): use log(-expm1(x)) to avoid catastrophic cancellation.
    For x <= -0.6931: use log1p(-exp(x)) which is stable in this range.
    """
    # Ensure input is <= 0 for valid probability domain
    out = torch.empty_like(x)
    # Threshold: log(2) ≈ 0.6931
    close_to_zero = x > -0.6931
    out[close_to_zero] = torch.log(-torch.expm1(x[close_to_zero]))
    out[~close_to_zero] = torch.log1p(-torch.exp(x[~close_to_zero]))
    return out


def bce_from_logp(log_p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Computes BCE loss given log-probability log_p = log(p) and target y ∈ {0, 1}.
    
    BCE = -y * log(p) - (1-y) * log(1-p)
        = -y * log_p - (1-y) * log1mexp(log_p)
    
    This avoids exp(log_p) -> p -> log(p) round-trip and is more stable.
    """
    # Clamp log_p to avoid extreme values
    log_p = torch.clamp(log_p, max=-1e-7)  # ensure log_p < 0 (p < 1)
    neg_term = log1mexp(log_p)
    # BCE = -y*log_p - (1-y)*log(1-p)
    return -y * log_p - (1.0 - y) * neg_term


class MultiTaskBCELoss:
    """
    BCE losses for CTR and CVR with click-conditional masking.
    
    When use_esmm=True:
      - esmm_version="v2" (default): Standard ESMM using p_ctcvr = p_ctr * p_cvr with log-domain stability.
      - esmm_version="legacy": Old behavior (deprecated) treating CVR logit as CTCVR logit directly.
    
    CTCVR in standard ESMM is not directly predicted; it's computed as p_ctr * p_cvr.
    """

    def __init__(
        self,
        w_ctr: float = 1.0,
        w_cvr: float = 1.0,
        eps: float = 1e-6,
        use_esmm: bool = False,
        esmm_version: str = "v2",  # "v2" (standard) or "legacy"
        lambda_ctr: float = 1.0,
        lambda_ctcvr: float = 1.0,
        lambda_cvr_aux: float = 0.0,  # CVR aux loss on click=1 subset; 0 to disable
        esmm_eps: float = 1e-8,
        enabled_heads: Any = None,
        pos_weight_dynamic: bool = True,
        static_pos_weight_ctr: float | None = None,
        static_pos_weight_ctcvr: float | None = None,
        pos_weight_clip_ctr: float | None = None,
        pos_weight_clip_ctcvr: float | None = None,
        # ===== Aux Focal Configuration (方案1: ESMM 主链路 BCE + CTCVR Aux-Focal) =====
        aux_focal_enabled: bool = False,
        aux_focal_warmup_steps: int = 2000,
        aux_focal_target_head: str = "ctcvr",
        aux_focal_lambda: float = 0.1,
        aux_focal_gamma: float = 1.0,
        aux_focal_use_alpha: bool = False,
        aux_focal_alpha: float = 0.25,
        aux_focal_detach_p_for_weight: bool = True,
        aux_focal_compute_fp32: bool = True,
        aux_focal_log_components: bool = True,
        # ===== 负采样权重补偿 =====
        neg_sample_weight_correction: bool = False,
        neg_keep_prob_train: float = 1.0,
        # ===== ESMM Residual Head 配置 =====
        residual_enabled: bool = False,
        residual_res_reg_weight: float = 5e-4,
        residual_anchor_enabled: bool = False,
        residual_anchor_weight: float = 0.02,
        residual_anchor_eps: float = 1e-6,
        # Global step tracker (will be updated externally by trainer)
        global_step: int = 0,
    ):
        self.w_ctr = float(w_ctr)
        self.w_cvr = float(w_cvr)
        self.eps = float(eps)
        self.use_esmm = bool(use_esmm)
        self.esmm_version = str(esmm_version).lower()
        if self.use_esmm and self.esmm_version == "legacy":
            warnings.warn(
                "esmm_version='legacy' is deprecated and may produce incorrect CTCVR semantics. "
                "Migrate to esmm_version='v2' (standard ESMM: p_ctcvr = p_ctr * p_cvr).",
                DeprecationWarning,
                stacklevel=2,
            )
        self.lambda_ctr = float(lambda_ctr)
        self.lambda_ctcvr = float(lambda_ctcvr)
        self.lambda_cvr_aux = float(lambda_cvr_aux)
        self.esmm_eps = float(esmm_eps)
        self.pos_weight_dynamic = bool(pos_weight_dynamic)
        self.static_pos_weight_ctr = (
            float(static_pos_weight_ctr) if static_pos_weight_ctr is not None else None
        )
        # Reserved for future ESMM usage; stored to keep config round-trip stable.
        self.static_pos_weight_ctcvr = (
            float(static_pos_weight_ctcvr) if static_pos_weight_ctcvr is not None else None
        )
        self.pos_weight_clip_ctr = float(pos_weight_clip_ctr) if pos_weight_clip_ctr is not None else None
        self.pos_weight_clip_ctcvr = (
            float(pos_weight_clip_ctcvr) if pos_weight_clip_ctcvr is not None else None
        )
        default_heads = ["ctr", "cvr"]
        enabled_list = enabled_heads or default_heads
        self.enabled_heads = {h.lower() for h in enabled_list}
        
        # ===== Aux Focal Configuration =====
        self.aux_focal_enabled = bool(aux_focal_enabled)
        self.aux_focal_warmup_steps = int(aux_focal_warmup_steps)
        self.aux_focal_target_head = str(aux_focal_target_head).lower()
        self.aux_focal_lambda = float(aux_focal_lambda)
        self.aux_focal_gamma = float(aux_focal_gamma)
        self.aux_focal_use_alpha = bool(aux_focal_use_alpha)
        self.aux_focal_alpha = float(aux_focal_alpha)
        self.aux_focal_detach_p_for_weight = bool(aux_focal_detach_p_for_weight)
        self.aux_focal_compute_fp32 = bool(aux_focal_compute_fp32)
        self.aux_focal_log_components = bool(aux_focal_log_components)
        
        # ===== 负采样权重补偿配置 =====
        self.neg_sample_weight_correction = bool(neg_sample_weight_correction)
        self.neg_keep_prob_train = float(neg_keep_prob_train)
        if self.neg_sample_weight_correction and self.neg_keep_prob_train < 1.0:
            self.neg_sample_weight = 1.0 / self.neg_keep_prob_train
        else:
            self.neg_sample_weight = 1.0

        # ===== ESMM Residual Head 配置 =====
        # 当 residual_enabled=True 时, 模型会在 outputs 中返回 "residual_logit"
        # Loss 计算将使用 logit 空间加法来组合 CTCVR:
        #   ctcvr_logit = detach(ctr_logit) + detach(cvr_logit) + r_logit
        self.residual_enabled = bool(residual_enabled)
        self.residual_res_reg_weight = float(residual_res_reg_weight)
        self.residual_anchor_enabled = bool(residual_anchor_enabled)
        self.residual_anchor_weight = float(residual_anchor_weight)
        self.residual_anchor_eps = float(residual_anchor_eps)
        
        # Global step for warmup control (updated externally by trainer)
        self.global_step = int(global_step)

    def compute(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
        labels = get_labels(batch)

        y_ctr = labels.get("y_ctr")
        y_cvr = labels.get("y_cvr")
        y_ctcvr = labels.get("y_ctcvr")
        click_mask = labels.get("click_mask")

        # Base device for creating zeros without moving tensors repeatedly.
        base_device = None
        if outputs:
            base_device = next(iter(outputs.values())).device
        zero = lambda: torch.tensor(0.0, device=base_device)  # noqa: E731

        # ESMM branch: CTR + CTCVR (full exposure, static pos_weight)
        if self.use_esmm:
            return self._compute_esmm(outputs, labels, base_device, zero)

        loss_ctr = zero()
        loss_cvr = zero()
        mask_sum_val = 0.0
        loss_ctr_raw_val: float | None = None
        loss_cvr_raw_val: float | None = None
        loss_ctr_scaled_val: float | None = None
        loss_cvr_scaled_val: float | None = None
        pos_weight_ctr_val: float | None = None
        pos_weight_cvr_val: float | None = None
        pos_weight_mode = "dynamic" if self.pos_weight_dynamic else "static"

        if "ctr" in self.enabled_heads:
            ctr_logit = outputs["ctr"]
            assert y_ctr is not None, "y_ctr missing from labels"
            assert ctr_logit.shape == y_ctr.shape, "ctr logit/label shape mismatch"
            if self.pos_weight_dynamic:
                # 动态计算 CTR 正例权重：neg/pos
                pos_weight_ctr, _, _ = _compute_dynamic_pos_weight(
                    y_ctr, self.pos_weight_clip_ctr, eps=self.eps,
                    device=ctr_logit.device, dtype=ctr_logit.dtype
                )
            else:
                # Static mode falls back to 1.0 if not provided to avoid crashing old checkpoints.
                static_val = self.static_pos_weight_ctr if self.static_pos_weight_ctr is not None else 1.0
                pos_weight_ctr = torch.tensor(static_val, device=ctr_logit.device, dtype=ctr_logit.dtype)
                if self.pos_weight_clip_ctr is not None:
                    pos_weight_ctr = torch.clamp(pos_weight_ctr, max=self.pos_weight_clip_ctr)

            pos_weight_ctr_val = float(pos_weight_ctr.item())
            loss_ctr = F.binary_cross_entropy_with_logits(
                ctr_logit, y_ctr, reduction="mean", pos_weight=pos_weight_ctr
            )
            loss_ctr_raw_val = float(loss_ctr.item())
            loss_ctr_scaled_val = float(loss_ctr_raw_val * self.w_ctr)

        if "cvr" in self.enabled_heads:
            cvr_logit = outputs["cvr"]
            assert y_cvr is not None, "y_cvr missing from labels"
            assert click_mask is not None, "click_mask missing from labels"
            assert cvr_logit.shape == y_cvr.shape, "cvr logit/label shape mismatch"

            mask = click_mask
            mask_sum = mask.sum()
            mask_sum_val = float(mask_sum.item())
            if mask_sum_val > 0:
                # 仅在有点击的子集上计算 pos_weight，防止未曝光样本干扰
                pos_count_cvr = torch.clamp((y_cvr * mask).sum(), min=self.eps)
                neg_count_cvr = torch.clamp(((1.0 - y_cvr) * mask).sum(), min=self.eps)
                pos_weight_cvr = neg_count_cvr / pos_count_cvr
                pos_weight_cvr_val = float(pos_weight_cvr.item())
                loss_cvr_vec = F.binary_cross_entropy_with_logits(
                    cvr_logit, y_cvr, reduction="none", pos_weight=pos_weight_cvr
                )
                loss_cvr = (loss_cvr_vec * mask).sum() / (mask_sum + self.eps)
                loss_cvr_raw_val = float(loss_cvr.item())
                loss_cvr_scaled_val = float(loss_cvr_raw_val * self.w_cvr)
            else:
                loss_cvr = zero()

        # Total only sums enabled heads
        loss_total = zero()
        if "ctr" in self.enabled_heads:
            loss_total = loss_total + self.w_ctr * loss_ctr
        if "cvr" in self.enabled_heads:
            loss_total = loss_total + self.w_cvr * loss_cvr

        # CTCVR prob for logging (only when both heads are active)
        ctcvr_prob_mean: float | None = None
        if "ctr" in self.enabled_heads and "cvr" in self.enabled_heads:
            ctr_prob = torch.sigmoid(outputs["ctr"])
            cvr_prob = torch.sigmoid(outputs["cvr"])
            ctcvr_prob_mean = float((ctr_prob * cvr_prob).mean().item())

        loss_ratio_scaled: float | None = None
        if loss_ctr_scaled_val is not None and loss_cvr_scaled_val is not None:
            loss_ratio_scaled = float(loss_cvr_scaled_val / (loss_ctr_scaled_val + 1e-12))

        loss_dict = {
            "loss_total": float(loss_total.item()),
            "loss_ctr": float(loss_ctr.item()) if "ctr" in self.enabled_heads else 0.0,
            "loss_cvr": float(loss_cvr.item()) if "cvr" in self.enabled_heads else 0.0,
            "mask_cvr_sum": mask_sum_val if "cvr" in self.enabled_heads else 0.0,
            "ctcvr_prob_mean": ctcvr_prob_mean,
            "loss_ctr_raw": loss_ctr_raw_val,
            "loss_cvr_raw": loss_cvr_raw_val,
            "loss_ctr_scaled": loss_ctr_scaled_val,
            "loss_cvr_scaled": loss_cvr_scaled_val,
            "loss_ratio_scaled": loss_ratio_scaled,
            "pos_weight_ctr": pos_weight_ctr_val,
            "pos_weight_cvr": pos_weight_cvr_val,
            "pos_weight_mode": pos_weight_mode,
        }
        return loss_total, loss_dict

    def _compute_esmm(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
        base_device: torch.device | None,
        zero: callable,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Standard ESMM (v2) or legacy ESMM computation.
        
        Standard ESMM:
          - p_ctcvr = p_ctr * p_cvr (probability multiplication constraint)
          - loss_ctcvr = BCE(p_ctcvr, y_ctcvr) computed in log-domain for stability
          - Optional: loss_cvr_aux = BCE(cvr_logit, y_cvr) on click=1 subset
        
        Legacy ESMM (deprecated):
          - Treats CVR logit as CTCVR logit directly (incorrect semantics)
        """
        # Extract required tensors with strict validation
        ctr_logit = outputs.get("ctr")
        cvr_logit = outputs.get("cvr")
        
        # Strict contract: ESMM v2 requires both ctr_logit and cvr_logit
        if ctr_logit is None:
            raise KeyError(
                "ESMM requires 'ctr' logit in model outputs. "
                "Ensure your model returns outputs['ctr']."
            )
        
        if self.esmm_version == "v2":
            # Standard ESMM: must have cvr_logit, NOT ctcvr_logit
            if cvr_logit is None:
                raise KeyError(
                    "Standard ESMM (v2) requires 'cvr' logit (post-click conversion probability) "
                    "in model outputs. Do NOT provide 'ctcvr' directly; CTCVR is computed as "
                    "p_ctr * p_cvr. Ensure your model returns outputs['cvr']."
                )
        else:
            # Legacy mode: fallback to ctcvr if cvr not present
            if cvr_logit is None:
                cvr_logit = outputs.get("ctcvr")
            if cvr_logit is None:
                raise KeyError(
                    "Legacy ESMM requires 'cvr' or 'ctcvr' logit in model outputs."
                )
        
        y_ctr = labels.get("y_ctr")
        y_ctcvr = labels.get("y_ctcvr")
        y_cvr = labels.get("y_cvr")
        click_mask = labels.get("click_mask")
        
        if y_ctr is None:
            raise KeyError("ESMM requires 'y_ctr' in labels.")
        if y_ctcvr is None:
            raise KeyError("ESMM requires 'y_ctcvr' in labels.")
        
        # Shape validation
        if ctr_logit.shape != y_ctr.shape:
            raise ValueError(
                f"CTR logit shape {ctr_logit.shape} != y_ctr shape {y_ctr.shape}"
            )
        if cvr_logit.shape != y_ctcvr.shape:
            raise ValueError(
                f"CVR logit shape {cvr_logit.shape} != y_ctcvr shape {y_ctcvr.shape}"
            )

        # Compute in float32 for numerical stability (important for AMP)
        ctr_logit_f32 = ctr_logit.float()
        cvr_logit_f32 = cvr_logit.float()
        y_ctr_f32 = y_ctr.float()
        y_ctcvr_f32 = y_ctcvr.float()

        # ====================================================================
        # CTR pos_weight: 动态或静态计算
        # ====================================================================
        if self.pos_weight_dynamic:
            pos_weight_ctr, ctr_pw_raw, ctr_pw_effective = _compute_dynamic_pos_weight(
                y_ctr_f32, self.pos_weight_clip_ctr, eps=self.eps,
                device=ctr_logit.device, dtype=torch.float32
            )
        else:
            ctr_pw_raw = self.static_pos_weight_ctr if self.static_pos_weight_ctr is not None else 1.0
            ctr_pw_effective, _ = _effective_pos_weight(ctr_pw_raw, self.pos_weight_clip_ctr)
            pos_weight_ctr = torch.tensor(ctr_pw_effective, device=ctr_logit.device, dtype=torch.float32)
        
        # 记录 raw 和 effective 值，用于日志和 metrics
        pos_weight_ctr_raw_val = float(ctr_pw_raw)
        pos_weight_ctr_effective_val = float(ctr_pw_effective)
        ctr_clipped = (ctr_pw_raw != ctr_pw_effective)

        # CTR loss (standard BCE with logits)
        loss_ctr_bce = F.binary_cross_entropy_with_logits(
            ctr_logit_f32, y_ctr_f32, reduction="none", pos_weight=pos_weight_ctr
        )
        
        # 负采样权重补偿：对被保留的负样本乘以 1/neg_keep_prob
        if self.neg_sample_weight_correction and self.neg_keep_prob_train < 1.0:
            neg_sample_weight = torch.where(
                y_ctr_f32 > 0.5,
                torch.ones_like(loss_ctr_bce),  # 正样本权重 = 1
                torch.tensor(self.neg_sample_weight, device=loss_ctr_bce.device, dtype=torch.float32),  # 负样本权重补偿
            )
            loss_ctr_bce = loss_ctr_bce * neg_sample_weight
        
        loss_ctr = loss_ctr_bce.mean()

        # ============================================================
        # Residual head 变量预初始化 (在 v2/legacy 分支前)
        # 仅 ESMM v2 + residual_enabled 时会被实际赋值
        # ============================================================
        residual_logit = None
        loss_res_reg = zero()
        loss_anchor = zero()
        residual_metrics: Dict[str, float] = {}

        if self.esmm_version == "v2":
            # ============================================================
            # Standard ESMM: p_ctcvr = p_ctr * p_cvr
            # Computed in log-domain for numerical stability
            # ============================================================
            # log(p_ctr) = logsigmoid(ctr_logit)
            # log(p_cvr) = logsigmoid(cvr_logit)
            # log(p_ctcvr) = log(p_ctr) + log(p_cvr)
            log_p_ctr = F.logsigmoid(ctr_logit_f32)
            log_p_cvr = F.logsigmoid(cvr_logit_f32)
            log_p_ctcvr = log_p_ctr + log_p_cvr
            
            # ====================================================================
            # CTCVR pos_weight: 动态或静态计算
            # ====================================================================
            if self.pos_weight_dynamic:
                # 动态计算 CTCVR 正例权重
                _, ctcvr_pw_raw, ctcvr_pw_effective = _compute_dynamic_pos_weight(
                    y_ctcvr_f32, self.pos_weight_clip_ctcvr, eps=self.eps,
                    device=ctr_logit.device, dtype=torch.float32
                )
            else:
                ctcvr_pw_raw = self.static_pos_weight_ctcvr if self.static_pos_weight_ctcvr is not None else 1.0
                ctcvr_pw_effective, _ = _effective_pos_weight(ctcvr_pw_raw, self.pos_weight_clip_ctcvr)
            
            # 记录 raw 和 effective 值，用于日志和 metrics
            pos_weight_ctcvr_raw_val = float(ctcvr_pw_raw)
            pos_weight_ctcvr_effective_val = float(ctcvr_pw_effective)
            ctcvr_clipped = (ctcvr_pw_raw != ctcvr_pw_effective)
            
            # Weighted BCE from log-probability
            # BCE = -y * log(p) - (1-y) * log(1-p)
            # With pos_weight w: BCE = -w*y*log(p) - (1-y)*log(1-p)
            bce_elements = bce_from_logp(log_p_ctcvr, y_ctcvr_f32)
            if ctcvr_pw_effective != 1.0:
                # Apply pos_weight: increase weight of positive samples
                weight = torch.where(
                    y_ctcvr_f32 > 0.5,
                    torch.tensor(ctcvr_pw_effective, device=bce_elements.device, dtype=torch.float32),
                    torch.tensor(1.0, device=bce_elements.device, dtype=torch.float32),
                )
                bce_elements = bce_elements * weight
            loss_ctcvr_bce = bce_elements.mean()
            
            # ============================================================
            # 方案1: CTCVR Aux-Focal（配置化 + warmup）
            # 只在 ESMM v2 且 target_head="ctcvr" 时应用
            # ============================================================
            loss_ctcvr_focal_val: Optional[float] = None
            aux_focal_active = False
            
            if (
                self.aux_focal_enabled
                and self.aux_focal_target_head == "ctcvr"
                and self.global_step >= self.aux_focal_warmup_steps
            ):
                # Compute CTCVR logits from log_p_ctcvr
                # logit = log(p / (1-p)) = log(p) - log(1-p)
                # log(1-p) = log1mexp(log_p) for log_p <= 0
                log_p_ctcvr_clamped = torch.clamp(log_p_ctcvr, max=-1e-7)  # ensure < 0
                log_1m_p_ctcvr = log1mexp(log_p_ctcvr_clamped)
                ctcvr_logit = log_p_ctcvr_clamped - log_1m_p_ctcvr
                
                # Apply focal loss
                focal_alpha = self.aux_focal_alpha if self.aux_focal_use_alpha else None
                loss_ctcvr_focal = focal_on_logits_aux(
                    ctcvr_logit,
                    y_ctcvr_f32,
                    gamma=self.aux_focal_gamma,
                    alpha=focal_alpha,
                    detach_p_for_weight=self.aux_focal_detach_p_for_weight,
                    compute_fp32=self.aux_focal_compute_fp32,
                    reduction="mean",
                )
                loss_ctcvr_focal_val = float(loss_ctcvr_focal.item())
                
                # Combine: L_ctcvr = L_ctcvr_bce + lambda * L_ctcvr_focal
                loss_ctcvr = loss_ctcvr_bce + self.aux_focal_lambda * loss_ctcvr_focal
                aux_focal_active = True
            else:
                # Warmup phase or disabled: use only BCE
                loss_ctcvr = loss_ctcvr_bce
            
            # ============================================================
            # ESMM Residual Head: logit 空间加法修正 CTCVR
            # ctcvr_logit = detach(ctr_logit) + detach(cvr_logit) + r_logit
            #
            # 为什么 detach? 防止 L_ctcvr 反向更新 CTR/CVR head, 避免任务互相拖累.
            # 残差 r_logit 由 ResidualHead 输出, 梯度只流过 residual head
            # 和（可选的）shared/MMoE 表征.
            # ============================================================
            residual_logit = outputs.get("residual_logit") if self.residual_enabled else None
            
            if residual_logit is not None:
                r_logit = residual_logit.float()  # [B, 1]
                # 确保 shape 统一为 [B, 1]
                if r_logit.dim() == 1:
                    r_logit = r_logit.unsqueeze(-1)
                
                # logit 空间加法: detach CTR/CVR logit 以隔离梯度
                ctr_logit_det = ctr_logit_f32.detach()
                cvr_logit_det = cvr_logit_f32.detach()
                # 将 [B] -> [B, 1] 以匹配 r_logit 的形状
                if ctr_logit_det.dim() == 1:
                    ctr_logit_det = ctr_logit_det.unsqueeze(-1)
                if cvr_logit_det.dim() == 1:
                    cvr_logit_det = cvr_logit_det.unsqueeze(-1)
                
                ctcvr_logit_res = ctr_logit_det + cvr_logit_det + r_logit  # [B, 1]
                ctcvr_logit_res = ctcvr_logit_res.squeeze(-1)  # [B]
                
                # 用 residual 版本覆盖 CTCVR loss
                # 注意: 这里直接用 logit 空间的 BCE, 而非 log-domain
                loss_ctcvr_res_bce = F.binary_cross_entropy_with_logits(
                    ctcvr_logit_res, y_ctcvr_f32, reduction="none",
                )
                if ctcvr_pw_effective != 1.0:
                    weight = torch.where(
                        y_ctcvr_f32 > 0.5,
                        torch.tensor(ctcvr_pw_effective, device=ctcvr_logit_res.device, dtype=torch.float32),
                        torch.tensor(1.0, device=ctcvr_logit_res.device, dtype=torch.float32),
                    )
                    loss_ctcvr_res_bce = loss_ctcvr_res_bce * weight
                # 保留乘法版本的 loss 用于日志对比
                loss_ctcvr_mul = loss_ctcvr  # 保持引用
                # 替换为 residual 版本
                loss_ctcvr = loss_ctcvr_res_bce.mean()
                
                # ============================================================
                # Residual 正则: L_res_reg = mean(r_logit^2) * res_reg_weight
                # 鼓励残差保持小幅度, 避免过度纠正
                # ============================================================
                r_logit_flat = r_logit.squeeze(-1)  # [B]
                if self.residual_res_reg_weight > 0:
                    loss_res_reg = (r_logit_flat ** 2).mean() * self.residual_res_reg_weight
                
                # ============================================================
                # Anchor loss (可选): 让残差不偏离 ESMM 乘法太远
                # L_anchor = MSE(ctcvr_logit_res, logit_mul) * anchor_weight
                # ============================================================
                if self.residual_anchor_enabled:
                    eps_a = self.residual_anchor_eps
                    p_mul = torch.sigmoid(ctr_logit_det.squeeze(-1)) * torch.sigmoid(cvr_logit_det.squeeze(-1))
                    p_mul = torch.clamp(p_mul, min=eps_a, max=1.0 - eps_a)
                    logit_mul = torch.log(p_mul) - torch.log(1.0 - p_mul)
                    loss_anchor = F.mse_loss(ctcvr_logit_res, logit_mul) * self.residual_anchor_weight
                
                # ============================================================
                # Residual 诊断指标 (训练/评估均输出)
                # - residual_r_abs_mean: 平均绝对值, 反映修正幅度
                # - residual_r_abs_p95: P95 绝对值, 检测异常大修正
                # - residual_r_abs_max: 最大绝对值
                # - corr_r_ctr / corr_r_cvr: 与 CTR/CVR logit 的皮尔逊相关
                #   若相关性过高, 说明 residual 在重复 CTR/CVR 的工作
                # ============================================================
                r_abs = r_logit_flat.detach().abs()
                residual_metrics["residual_r_abs_mean"] = float(r_abs.mean().item())
                # P95: 用 quantile 计算
                if r_abs.numel() > 1:
                    residual_metrics["residual_r_abs_p95"] = float(
                        torch.quantile(r_abs.float(), 0.95).item()
                    )
                else:
                    residual_metrics["residual_r_abs_p95"] = float(r_abs.mean().item())
                residual_metrics["residual_r_abs_max"] = float(r_abs.max().item())
                
                # 皮尔逊相关系数 (batch 内, 数值稳定版本)
                def _pearson_corr(a: torch.Tensor, b: torch.Tensor) -> float:
                    """Batch-level Pearson correlation, 数值稳定."""
                    a = a.detach().float()
                    b = b.detach().float()
                    a_mean = a.mean()
                    b_mean = b.mean()
                    a_centered = a - a_mean
                    b_centered = b - b_mean
                    cov = (a_centered * b_centered).mean()
                    std_a = a_centered.pow(2).mean().sqrt()
                    std_b = b_centered.pow(2).mean().sqrt()
                    denom = std_a * std_b + 1e-8  # 防除零
                    return float((cov / denom).clamp(-1.0, 1.0).item())
                
                residual_metrics["corr_r_ctr"] = _pearson_corr(
                    r_logit_flat, ctr_logit_f32.detach()
                )
                residual_metrics["corr_r_cvr"] = _pearson_corr(
                    r_logit_flat, cvr_logit_f32.detach()
                )
                residual_metrics["loss_ctcvr_mul"] = float(loss_ctcvr_mul.detach().item())
                residual_metrics["loss_res_reg"] = float(loss_res_reg.detach().item())
                if self.residual_anchor_enabled:
                    residual_metrics["loss_anchor"] = float(loss_anchor.detach().item())
            
            # Probabilities for metrics (computed from log-domain)
            p_ctr = torch.sigmoid(ctr_logit_f32)
            p_cvr = torch.sigmoid(cvr_logit_f32)
            p_ctcvr = p_ctr * p_cvr  # This equals exp(log_p_ctcvr) but more readable
            
        else:
            # ============================================================
            # Legacy ESMM: treat CVR logit as CTCVR logit (DEPRECATED)
            # ====================================================================
            # CTCVR pos_weight: 动态或静态计算
            # ====================================================================
            if self.pos_weight_dynamic:
                pos_weight_ctcvr, ctcvr_pw_raw, ctcvr_pw_effective = _compute_dynamic_pos_weight(
                    y_ctcvr_f32, self.pos_weight_clip_ctcvr, eps=self.eps,
                    device=cvr_logit.device, dtype=torch.float32
                )
            else:
                ctcvr_pw_raw = self.static_pos_weight_ctcvr if self.static_pos_weight_ctcvr is not None else 1.0
                ctcvr_pw_effective, _ = _effective_pos_weight(ctcvr_pw_raw, self.pos_weight_clip_ctcvr)
                pos_weight_ctcvr = torch.tensor(ctcvr_pw_effective, device=cvr_logit.device, dtype=torch.float32)
            
            # 记录 raw 和 effective 值
            pos_weight_ctcvr_raw_val = float(ctcvr_pw_raw)
            pos_weight_ctcvr_effective_val = float(ctcvr_pw_effective)
            ctcvr_clipped = (ctcvr_pw_raw != ctcvr_pw_effective)
            
            loss_ctcvr = F.binary_cross_entropy_with_logits(
                cvr_logit_f32, y_ctcvr_f32, reduction="mean", pos_weight=pos_weight_ctcvr
            )
            
            p_ctr = torch.sigmoid(ctr_logit_f32)
            p_ctcvr = torch.sigmoid(cvr_logit_f32)
            p_cvr = p_ctcvr / (p_ctr + self.esmm_eps)

        # ============================================================
        # Optional: CVR auxiliary loss on click=1 subset (ESMM+aux)
        # This accelerates CVR tower learning by providing direct supervision
        # ============================================================
        loss_cvr_aux = zero()
        loss_cvr_aux_val: float | None = None
        cvr_aux_mask_sum = 0.0
        
        if self.lambda_cvr_aux > 0 and y_cvr is not None:
            # Use click_mask if available, otherwise use y_ctr as mask
            if click_mask is not None:
                aux_mask = click_mask.float()
            else:
                aux_mask = y_ctr_f32
            
            mask_sum = aux_mask.sum()
            cvr_aux_mask_sum = float(mask_sum.item())
            
            if mask_sum > 0:
                y_cvr_f32 = y_cvr.float()
                # BCE on clicked subset
                loss_vec = F.binary_cross_entropy_with_logits(
                    cvr_logit_f32, y_cvr_f32, reduction="none"
                )
                loss_cvr_aux = (loss_vec * aux_mask).sum() / (mask_sum + self.eps)
                loss_cvr_aux_val = float(loss_cvr_aux.item())

        # ============================================================
        # Total loss
        # ============================================================
        loss_total = (
            self.lambda_ctr * loss_ctr 
            + self.lambda_ctcvr * loss_ctcvr 
            + self.lambda_cvr_aux * loss_cvr_aux
        )

        # ============================================================
        # Residual 附加 loss: reg + anchor (仅当 residual_logit 存在时)
        # 这些 loss 权重已经在 residual_metrics 中记录,
        # 不再使用 lambda_ 前缀, 直接加到 total loss
        # ============================================================
        if residual_logit is not None:
            loss_total = loss_total + loss_res_reg + loss_anchor

        # ============================================================
        # Metrics for observability
        # ============================================================
        p_ctr_mean = float(p_ctr.mean().item())
        p_cvr_mean = float(p_cvr.mean().item())
        p_ctcvr_mean = float(p_ctcvr.mean().item())
        
        # Positive rates
        pos_rate_ctr = float(y_ctr_f32.mean().item())
        pos_rate_ctcvr = float(y_ctcvr_f32.mean().item())
        pos_rate_cvr: float | None = None
        if y_cvr is not None:
            pos_rate_cvr = float(y_cvr.float().mean().item())

        loss_dict = {
            "loss_total": float(loss_total.item()),
            "loss_ctr": float(loss_ctr.item()),
            "loss_cvr": float(loss_ctcvr.item()),  # kept key name for compatibility
            "loss_ctcvr": float(loss_ctcvr.item()),
            "loss_cvr_aux": loss_cvr_aux_val,
            "cvr_aux_mask_sum": cvr_aux_mask_sum,
            "mask_cvr_sum": float(y_ctr.numel()),  # exposures count
            # ====================================================================
            # pos_weight: raw (原始配置值) / effective (裁剪后实际使用值) / clipped (是否发生裁剪)
            # ====================================================================
            "pos_weight_ctr": pos_weight_ctr_effective_val,  # 向后兼容
            "pos_weight_ctr_raw": pos_weight_ctr_raw_val,
            "pos_weight_ctr_effective": pos_weight_ctr_effective_val,
            "pos_weight_ctr_clipped": ctr_clipped,
            "pos_weight_cvr": pos_weight_ctcvr_effective_val,  # legacy alias
            "pos_weight_ctcvr": pos_weight_ctcvr_effective_val,  # 向后兼容
            "pos_weight_ctcvr_raw": pos_weight_ctcvr_raw_val,
            "pos_weight_ctcvr_effective": pos_weight_ctcvr_effective_val,
            "pos_weight_ctcvr_clipped": ctcvr_clipped,
            "pos_weight_mode": f"{'dynamic' if self.pos_weight_dynamic else 'static'}_esmm_{self.esmm_version}",
            "p_ctr_mean": p_ctr_mean,
            "p_cvr_mean": p_cvr_mean,
            "p_ctcvr_mean": p_ctcvr_mean,
            "pos_rate_ctr": pos_rate_ctr,
            "pos_rate_cvr": pos_rate_cvr,
            "pos_rate_ctcvr": pos_rate_ctcvr,
            "esmm_version": self.esmm_version,
            "lambda_ctr": self.lambda_ctr,
            "lambda_ctcvr": self.lambda_ctcvr,
            "lambda_cvr_aux": self.lambda_cvr_aux,
        }
        
        # ===== Add aux_focal metrics (方案1) =====
        if self.aux_focal_log_components and self.esmm_version == "v2":
            loss_dict["aux_focal_enabled"] = self.aux_focal_enabled
            loss_dict["aux_focal_warmup_steps"] = self.aux_focal_warmup_steps
            loss_dict["aux_focal_on"] = aux_focal_active
            loss_dict["aux_focal_lambda"] = self.aux_focal_lambda
            loss_dict["aux_focal_gamma"] = self.aux_focal_gamma
            loss_dict["global_step"] = self.global_step
            
            # BCE and Focal components (only when focal is active)
            if aux_focal_active and loss_ctcvr_focal_val is not None:
                loss_dict["loss_ctcvr_bce"] = float(loss_ctcvr_bce.item())
                loss_dict["loss_ctcvr_focal"] = loss_ctcvr_focal_val
            elif self.aux_focal_enabled:
                # Warmup phase: only BCE, no focal yet
                loss_dict["loss_ctcvr_bce"] = float(loss_ctcvr_bce.item()) if self.esmm_version == "v2" else None
                loss_dict["loss_ctcvr_focal"] = 0.0
        
        # ===== Add residual head metrics =====
        if self.residual_enabled and residual_metrics:
            loss_dict["residual_enabled"] = True
            loss_dict.update(residual_metrics)
        elif self.residual_enabled:
            # residual enabled in config but no residual_logit in outputs
            # (should not happen, but be defensive)
            loss_dict["residual_enabled"] = True
        
        return loss_total, loss_dict


__all__ = ["MultiTaskBCELoss", "log1mexp", "bce_from_logp", "focal_on_logits_aux", "_effective_pos_weight", "_compute_dynamic_pos_weight"]
