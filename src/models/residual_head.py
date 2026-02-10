"""
ESMM Logit Residual Correction Head.

为 ESMM 的 CTCVR 预估引入可学习的 logit 残差修正:

    ctcvr_logit = detach(ctr_logit) + detach(cvr_logit) + r_logit

其中 r_logit 由本模块输出。

设计原理:
  - 为什么用 tanh 限幅? 防止残差 logit 爆炸, 保证训练稳定。
    alpha 参数控制残差最大幅度, 通过 config 可调。
  - 为什么输出的是 logit 而非概率? 后续在 logit 空间做加法组合,
    避免概率乘法的数值问题; 同时 BCE loss 直接接受 logit。
  - 为什么不加 sigmoid? ResidualHead 输出的是修正量 (residual),
    而非独立的概率预估, sigmoid 应由最终的 loss 函数统一处理。
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
from torch import nn

from src.models.backbones.layers import MLP

logger = logging.getLogger(__name__)


class ResidualHead(nn.Module):
    """
    Residual correction head for ESMM CTCVR prediction.

    输入: MMoE/PLE CVR 分支输出的表征 (进入 CVR tower 前的向量), shape [B, D].
    输出: r_logit, shape [B, 1], 经 alpha*tanh(.) 限幅后的 logit 残差.

    Architecture:
        [LayerNorm (optional)] -> MLP(d -> dims[0] -> ... -> dims[-1]) -> Linear(dims[-1], 1) -> alpha * tanh(.)
    """

    def __init__(
        self,
        in_dim: int,
        mlp_dims: List[int],
        dropout: float = 0.1,
        use_layernorm: bool = True,
        alpha: float = 1.0,
    ):
        """
        Args:
            in_dim: 输入表征维度 (来自 MMoE/PLE 的 CVR 分支输出).
            mlp_dims: MLP 隐藏层维度列表, 例如 [128, 64] 或 [64].
            dropout: MLP 内部 dropout 比率.
            use_layernorm: 是否在输入侧附加 LayerNorm (推荐 True, 稳定训练).
            alpha: tanh 限幅系数, r_logit = alpha * tanh(r_raw).
                   建议 0.5~2.0, 默认 1.0.
        """
        super().__init__()
        self.alpha = float(alpha)

        # 可选输入侧 LayerNorm, 稳定来自不同 batch 的表征分布
        self.layernorm = nn.LayerNorm(in_dim) if use_layernorm else nn.Identity()

        # MLP hidden layers: Linear + ReLU + Dropout
        self.mlp = MLP(
            input_dim=in_dim,
            hidden_dims=mlp_dims,
            activation="relu",
            dropout=dropout,
            use_bn=False,
        )

        # 输出投影到标量
        self.out_proj = nn.Linear(self.mlp.output_dim, 1)
        # 初始化: 小权重 + zero bias, 使训练初期残差趋近于 0, 不破坏已有 ESMM 乘法
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.01)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

        logger.info(
            "[ResidualHead] in_dim=%d mlp_dims=%s dropout=%.2f "
            "use_layernorm=%s alpha=%.2f",
            in_dim, mlp_dims, dropout, use_layernorm, self.alpha,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, D] 表征向量 (CVR 分支 MMoE/PLE 混合输出).

        Returns:
            r_logit: [B, 1] 残差 logit, 经 alpha * tanh(.) 限幅.
                     幅度被限制在 [-alpha, +alpha] 范围内.
        """
        h = self.layernorm(x)
        h = self.mlp(h)
        r_raw = self.out_proj(h)  # [B, 1]
        # tanh 限幅: 防止残差 logit 过大导致训练不稳定
        # alpha 控制最大修正幅度
        r_logit = self.alpha * torch.tanh(r_raw)
        return r_logit


def build_residual_head(in_dim: int, residual_cfg: Dict) -> Optional[ResidualHead]:
    """
    根据配置构建 ResidualHead. 如果 enabled=false 则返回 None.

    Args:
        in_dim: 输入维度 (MMoE/PLE CVR 分支输出维度).
        residual_cfg: esmm.residual 配置字典.

    Returns:
        ResidualHead 实例或 None.
    """
    if not residual_cfg or not bool(residual_cfg.get("enabled", False)):
        return None

    mlp_dims = list(residual_cfg.get("mlp_dims", [64]))
    dropout = float(residual_cfg.get("dropout", 0.1))
    use_layernorm = bool(residual_cfg.get("use_layernorm", True))
    alpha = float(residual_cfg.get("alpha", 1.0))

    return ResidualHead(
        in_dim=in_dim,
        mlp_dims=mlp_dims,
        dropout=dropout,
        use_layernorm=use_layernorm,
        alpha=alpha,
    )


__all__ = ["ResidualHead", "build_residual_head"]
