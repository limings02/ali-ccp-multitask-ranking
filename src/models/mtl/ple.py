"""
PLE-Lite: Progressive Layered Extraction (Lite version) for Multi-Task Learning.

本模块实现 PLE-Lite，作为 MMoE 的对照组实验模型。
核心特点：
  - shared experts: 所有任务共享的专家网络
  - task-specific experts: 每个任务专属的专家网络
  - 每个任务的 gate 只在 (shared + 本任务 specific) experts 上做 softmax

与 mmoe.py 对齐要求（改动 B）：
  - 保持相同的 __init__ 签名风格
  - forward 返回格式一致：Dict[str, Tensor]，包含 per-task logits 和 aux
  - gate 稳定化逻辑完全复用 MMoE
  - 新增 gate_reg_scope 参数：控制正则化是只针对 shared experts 还是全部 experts

Author: Auto-generated for PLE-Lite ablation study
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from torch import nn

from src.models.backbones.layers import MLP
from src.models.heads import TaskHead
from src.models.mtl.composer import MMoEInputComposer, build_composer_from_config

logger = logging.getLogger(__name__)


# =============================================================================
# Heterogeneous Expert Support (向前兼容扩展)
# =============================================================================
def _should_use_hetero_experts(ple_cfg: Dict) -> bool:
    """
    判断是否应该使用异构专家（新逻辑）。
    
    向前兼容策略：
      1. 如果 hetero_enabled 显式设置为 False：走旧逻辑
      2. 如果 experts 配置缺失或为空：走旧逻辑
      3. 如果 experts.shared 不存在或为空：走旧逻辑
      4. 否则：走异构专家逻辑
    
    Args:
        ple_cfg: PLE 配置字典
        
    Returns:
        True if should use heterogeneous experts, False for legacy homogeneous experts
    """
    # 显式开关：hetero_enabled=False 强制走旧逻辑（便于快速回退）
    hetero_enabled = ple_cfg.get("hetero_enabled")
    if hetero_enabled is False:
        return False
    
    # 检查 experts 配置是否存在且有效
    experts_cfg = ple_cfg.get("experts")
    if not experts_cfg:
        return False
    
    # 检查 shared experts 是否存在且非空
    shared_specs = experts_cfg.get("shared")
    if not shared_specs or not isinstance(shared_specs, list) or len(shared_specs) == 0:
        return False
    
    return True


def _validate_hetero_config(ple_cfg: Dict, tasks: List[str]) -> None:
    """
    校验异构专家配置的有效性。
    
    检查项：
      - experts.shared 必须存在且非空
      - len(experts.shared) 必须等于 shared_num_experts（如果指定）
      - 所有 expert spec 必须包含 type 字段
      
    Args:
        ple_cfg: PLE 配置字典
        tasks: 任务名称列表
        
    Raises:
        ValueError: 如果配置无效
    """
    experts_cfg = ple_cfg.get("experts", {})
    
    # 校验 shared experts
    shared_specs = experts_cfg.get("shared", [])
    if not shared_specs:
        raise ValueError("Heterogeneous experts enabled but experts.shared is empty")
    
    # 校验 shared_num_experts 一致性
    num_shared_config = ple_cfg.get("shared_num_experts")
    if num_shared_config is not None and len(shared_specs) != int(num_shared_config):
        raise ValueError(
            f"len(experts.shared)={len(shared_specs)} != shared_num_experts={num_shared_config}"
        )
    
    # 校验每个 expert spec 包含 type
    for i, spec in enumerate(shared_specs):
        if not isinstance(spec, dict) or "type" not in spec:
            raise ValueError(f"experts.shared[{i}] must be a dict with 'type' field, got {spec}")
    
    # 校验 private experts（如果存在）
    private_cfg = experts_cfg.get("private", {})
    for task in tasks:
        task_specs = private_cfg.get(task, [])
        for i, spec in enumerate(task_specs):
            if not isinstance(spec, dict) or "type" not in spec:
                raise ValueError(f"experts.private.{task}[{i}] must be a dict with 'type' field")


class _GatePLE(nn.Module):
    """
    Per-task gate for PLE, outputs softmax over (shared + task-specific) experts.
    
    与 mmoe.py 的 _Gate 对齐：
    - 支持 temperature scaling（>1 使 softmax 更平滑）
    - 支持 training noise（仅训练时添加高斯噪声）
    - 支持动态传入 temperature/noise_std（用于 schedule）
    """

    def __init__(
        self,
        in_dim: int,
        num_experts: int,  # = num_shared + num_task_specific
        gate_type: str = "linear",
        hidden_dims: Optional[List[int]] = None,
        temperature: float = 1.0,
        noise_std: float = 0.0,
    ):
        super().__init__()
        gate_type = gate_type.lower()
        self.num_experts = num_experts
        self.temperature = temperature
        self.noise_std = noise_std

        if gate_type == "linear":
            self.net = nn.Linear(in_dim, num_experts)
        elif gate_type == "mlp":
            hidden_dims = hidden_dims or [in_dim]
            self.net = nn.Sequential(
                MLP(input_dim=in_dim, hidden_dims=hidden_dims, activation="relu", dropout=0.0, use_bn=False),
                nn.Linear(hidden_dims[-1] if hidden_dims else in_dim, num_experts),
            )
        else:
            raise ValueError(f"Unsupported gate_type: {gate_type}")

    def forward(
        self,
        x: torch.Tensor,
        return_logits: bool = False,
        temperature_override: Optional[float] = None,
        noise_std_override: Optional[float] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with optional stabilization.
        
        Args:
            x: Input tensor [B, D]
            return_logits: If True, return (weights, logits) tuple
            temperature_override: If provided, use this temperature instead of self.temperature
            noise_std_override: If provided, use this noise_std instead of self.noise_std
            
        Returns:
            weights: Softmax weights [B, K]（K = num_shared + num_task_specific）
            logits (optional): Raw logits before softmax [B, K]
        """
        logits = self.net(x)

        # 使用 override 或默认值
        temperature = temperature_override if temperature_override is not None else self.temperature
        noise_std = noise_std_override if noise_std_override is not None else self.noise_std

        # Add noise during training（与 mmoe.py 对齐）
        if self.training and noise_std > 0:
            noise = torch.randn_like(logits) * noise_std
            logits = logits + noise

        # Apply temperature scaling
        if temperature != 1.0:
            logits = logits / temperature

        weights = torch.softmax(logits, dim=-1)

        if return_logits:
            return weights, logits
        return weights


class PLE(nn.Module):
    """
    PLE-Lite: Progressive Layered Extraction 的简化版本。
    
    核心结构（改动 1.2）：
      - shared_experts: ModuleList，所有任务共享
      - specific_experts: ModuleDict(task -> ModuleList)，每个任务专属
      - gates: ModuleDict(task -> _GatePLE)，每个任务一个 gate
    
    与 MMoE 接口对齐（改动 B）：
      - __init__ 签名尽量一致
      - forward 输出格式一致：Dict[str, Tensor]
      - 支持 enabled_heads / use_legacy_pseudo_deepfm / return_logit_parts / per_head_add / head_priors / log_gates
    
    新增参数（改动 D）：
      - gate_reg_scope: "shared_only" 或 "all"
        - "shared_only"（默认）：只对 shared experts 的 gate 权重做 entropy/kl 正则
        - "all"：对全部 experts 的 gate 权重做正则
    """

    def __init__(
        self,
        backbone: nn.Module,
        head_cfg: Dict,
        ple_cfg: Dict,  # 对应 mmoe_cfg，但字段名更清晰
        enabled_heads: Optional[List[str]] = None,
        use_legacy_pseudo_deepfm: bool = True,
        return_logit_parts: bool = False,
        per_head_add: Optional[Dict[str, Dict[str, bool]]] = None,
        head_priors: Optional[Dict[str, float]] = None,
        log_gates: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.use_legacy_pseudo_deepfm = bool(use_legacy_pseudo_deepfm)
        self.return_logit_parts = bool(return_logit_parts)
        self.per_head_add = per_head_add or {}
        self.head_priors = head_priors or {}
        self.log_gates = bool(log_gates)

        # 任务列表
        tasks = head_cfg.get("tasks") or ["ctr", "cvr"]
        enabled = enabled_heads or tasks
        self.tasks: List[str] = tasks
        self.enabled_heads: Set[str] = {h.lower() for h in enabled}

        # ========== 构建可选的 input composer（与 mmoe.py 对齐）==========
        self.composer: Optional[MMoEInputComposer] = build_composer_from_config(backbone, ple_cfg)

        # ========== 解析输入维度 ==========
        if self.composer is not None:
            in_dim = self.composer.out_dim
            logger.info(f"[PLE] Using composer, input dim = {in_dim}")
        else:
            # 旧逻辑：直接使用 deep_h 或 emb_concat
            input_source = str(ple_cfg.get("input", "deep_h")).lower()
            self.input_source = input_source

            deep_dim = getattr(backbone, "deep_out_dim", getattr(backbone, "out_dim", None))
            emb_concat_dim = getattr(getattr(backbone, "feat_emb", None), "concat_dim", None)

            if input_source == "deep_h":
                in_dim = deep_dim
            elif input_source == "emb_concat":
                in_dim = emb_concat_dim
            else:
                raise ValueError(f"Unsupported ple.input value: {input_source}")

            if in_dim is None:
                raise ValueError("PLE: could not infer input dimension from backbone.")
            logger.info(f"[PLE] Using legacy input_source='{input_source}', input dim = {in_dim}")

        self.in_dim = in_dim
        self.layernorm = nn.LayerNorm(in_dim)

        # ==========================================================================
        # 异构专家支持：检测是否启用新的异构专家逻辑
        # ==========================================================================
        self._use_hetero_experts = _should_use_hetero_experts(ple_cfg)
        
        if self._use_hetero_experts:
            # ==================== 新路径：异构专家 ====================
            _validate_hetero_config(ple_cfg, tasks)
            self._build_hetero_experts(ple_cfg, tasks, in_dim)
        else:
            # ==================== 旧路径：同构专家（完全保持原逻辑）====================
            self._build_homogeneous_experts(ple_cfg, tasks, in_dim)

        # ========== Gate 稳定化配置（与 mmoe.py 完全对齐）==========
        gate_stabilize_cfg = ple_cfg.get("gate_stabilize", {}) or {}
        self.gate_stabilize_enabled = bool(gate_stabilize_cfg.get("enabled", False))
        self.gate_temperature = float(gate_stabilize_cfg.get("temperature", 1.0))
        self.gate_noise_std = float(gate_stabilize_cfg.get("noise_std", 0.0))
        self.gate_eps = float(gate_stabilize_cfg.get("eps", 1e-8))
        self.entropy_reg_weight = float(gate_stabilize_cfg.get("entropy_reg_weight", 0.0))
        self.load_balance_kl_weight = float(gate_stabilize_cfg.get("load_balance_kl_weight", 0.0))
        self.gate_log_stats = bool(gate_stabilize_cfg.get("log_stats", True))

        # ========== 新增：gate 温度/噪声退火 schedule（任务4）==========
        # 支持线性退火：从 start 值线性衰减到 end 值
        # enabled=false 时完全不生效（向后兼容）
        schedule_cfg = gate_stabilize_cfg.get("schedule", {}) or {}
        self.gate_schedule_enabled = bool(schedule_cfg.get("enabled", False))
        if self.gate_schedule_enabled:
            self.gate_schedule_warm_frac = float(schedule_cfg.get("warm_frac", 0.2))
            self.gate_schedule_temp_start = float(schedule_cfg.get("temperature_start", self.gate_temperature))
            self.gate_schedule_temp_end = float(schedule_cfg.get("temperature_end", 1.0))
            self.gate_schedule_noise_start = float(schedule_cfg.get("noise_std_start", self.gate_noise_std))
            self.gate_schedule_noise_end = float(schedule_cfg.get("noise_std_end", 0.0))
            logger.info(
                "[PLE] gate_schedule enabled: warm_frac=%.2f temp(%.2f->%.2f) noise(%.3f->%.3f)",
                self.gate_schedule_warm_frac,
                self.gate_schedule_temp_start, self.gate_schedule_temp_end,
                self.gate_schedule_noise_start, self.gate_schedule_noise_end,
            )
        # 用于外部设置当前 step（由 trainer 在每个 step 更新）
        self._current_step: int = 0
        self._total_steps: int = 1  # 防止除零

        # ========== 新增：gate_reg_scope 配置（改动 D）==========
        # "shared_only"：只对 shared experts 的 gate 权重做正则（默认）
        # "all"：对全部 experts 的 gate 权重做正则
        self.gate_reg_scope = str(ple_cfg.get("gate_reg_scope", "shared_only")).lower()
        if self.gate_reg_scope not in {"shared_only", "all"}:
            raise ValueError(f"gate_reg_scope must be 'shared_only' or 'all', got '{self.gate_reg_scope}'")

        if self.gate_stabilize_enabled:
            logger.info(
                "[PLE] gate_stabilize enabled: temp=%.2f noise_std=%.3f "
                "entropy_reg=%.2e lb_kl_reg=%.2e eps=%.2e scope=%s",
                self.gate_temperature, self.gate_noise_std,
                self.entropy_reg_weight, self.load_balance_kl_weight, 
                self.gate_eps, self.gate_reg_scope
            )

        # ========== 新增：gate_shared_mass_floor 配置（shared mass 下限正则）==========
        # 目的：防止 private experts 吞噬 shared experts，避免 gate 早熟锁死
        # 当 enabled=false 或配置缺失时完全不生效（向后兼容）
        mass_floor_cfg = ple_cfg.get("gate_shared_mass_floor", {}) or {}
        self.mass_floor_enabled = bool(mass_floor_cfg.get("enabled", False))
        self.mass_floor_per_task: Dict[str, Dict[str, float]] = {}
        if self.mass_floor_enabled:
            per_task_cfg = mass_floor_cfg.get("per_task", {}) or {}
            for task in tasks:
                task_cfg = per_task_cfg.get(task, {})
                if task_cfg:
                    self.mass_floor_per_task[task] = {
                        "min_mass": float(task_cfg.get("min_mass", 0.3)),
                        "weight": float(task_cfg.get("weight", 1e-3)),
                    }
            logger.info(
                "[PLE] gate_shared_mass_floor enabled: per_task=%s",
                self.mass_floor_per_task
            )

        # ========== 构建 gates（每个任务一个）==========
        gate_type = str(ple_cfg.get("gate_type", "linear"))
        gate_hidden_dims = ple_cfg.get("gate_hidden_dims")

        self.gates = nn.ModuleDict()
        self._gate_num_experts: Dict[str, int] = {}  # 记录每个任务 gate 的 expert 数量（用于正则计算）

        for task in tasks:
            # 每个任务的 gate 输出维度 = shared + 该任务的 specific
            num_experts_for_task = self.num_shared_experts + self.num_specific_experts[task]
            self._gate_num_experts[task] = num_experts_for_task

            self.gates[task] = _GatePLE(
                in_dim=in_dim,
                num_experts=num_experts_for_task,
                gate_type=gate_type,
                hidden_dims=gate_hidden_dims,
                temperature=self.gate_temperature if self.gate_stabilize_enabled else 1.0,
                noise_std=self.gate_noise_std if self.gate_stabilize_enabled else 0.0,
            )

        # ========== 构建 towers/heads（与 mmoe.py 对齐）==========
        self._head_cfg_resolved: Dict[str, Dict[str, object]] = {}
        self.towers = nn.ModuleDict()

        for task in tasks:
            def _pick_cfg(cfg: Dict, task_name: str, key: str, default):
                if isinstance(cfg.get(task_name), dict) and key in cfg[task_name] and cfg[task_name][key] is not None:
                    return cfg[task_name][key]
                if isinstance(cfg.get("default"), dict) and key in cfg["default"] and cfg["default"][key] is not None:
                    return cfg["default"][key]
                if key in cfg and cfg[key] is not None:
                    return cfg[key]
                return default

            mlp_dims = list(_pick_cfg(head_cfg, task, "mlp_dims", []))
            dropout = float(_pick_cfg(head_cfg, task, "dropout", 0.0))
            use_bn = bool(_pick_cfg(head_cfg, task, "use_bn", False))
            activation = str(_pick_cfg(head_cfg, task, "activation", "relu"))

            self._head_cfg_resolved[task] = {
                "mlp_dims": mlp_dims,
                "dropout": dropout,
                "use_bn": use_bn,
                "activation": activation,
            }

            self.towers[task] = TaskHead(
                in_dim=in_dim,
                mlp_dims=mlp_dims,
                out=1,
                activation=activation,
                dropout=dropout,
                use_bn=use_bn,
            )

            # Bias 初始化（与 mmoe.py 对齐）
            prior_map = {"ctr": 0.07, "cvr": 0.01}
            prior = float(self.head_priors.get(task, prior_map.get(task, 0.5)))
            prior = min(max(prior, 1e-6), 1 - 1e-6)
            bias_init = float(torch.log(torch.tensor(prior / (1 - prior))))
            if self.towers[task].out_proj.bias is not None:
                nn.init.constant_(self.towers[task].out_proj.bias, bias_init)

    # =========================================================================
    # Expert Building Methods (同构/异构专家构建)
    # =========================================================================
    
    def _build_homogeneous_experts(
        self,
        ple_cfg: Dict[str, Any],
        tasks: List[str],
        in_dim: int,
    ) -> None:
        """
        构建同构专家（旧逻辑，保持完全向后兼容）。
        
        同构专家使用统一的 MLP 配置：
          - shared_num_experts: 共享专家数量
          - specific_num_experts: 每个任务的专属专家数量
          - expert_mlp_dims: MLP 隐藏层维度
          - dropout/activation/use_bn: MLP 配置
        """
        # ========== PLE 专属配置：shared + specific experts 数量 ==========
        # shared experts 数量（所有任务共享）
        self.num_shared_experts = int(ple_cfg.get("shared_num_experts", 4))
        
        # specific experts 数量（每个任务可以不同）
        specific_cfg = ple_cfg.get("specific_num_experts", {})
        if isinstance(specific_cfg, int):
            # 兼容简写：所有任务相同数量
            self.num_specific_experts: Dict[str, int] = {task: specific_cfg for task in tasks}
        else:
            # 字典形式：每个任务可配置
            default_specific = int(specific_cfg.get("default", 1))
            self.num_specific_experts = {
                task: int(specific_cfg.get(task, default_specific)) for task in tasks
            }

        # expert MLP 配置
        expert_mlp_dims = list(ple_cfg.get("expert_mlp_dims", []))
        expert_output_dims = expert_mlp_dims + [in_dim] if expert_mlp_dims else [in_dim]
        expert_dropout = float(ple_cfg.get("dropout", 0.0))
        expert_activation = str(ple_cfg.get("activation", "relu"))
        expert_use_bn = bool(ple_cfg.get("use_bn", False))

        # ========== 新增：private expert 可选使用更小容量 ==========
        private_expert_mlp_dims = ple_cfg.get("private_expert_mlp_dims")
        if private_expert_mlp_dims:
            private_expert_mlp_dims = list(private_expert_mlp_dims)
            private_expert_output_dims = private_expert_mlp_dims + [in_dim] if private_expert_mlp_dims else [in_dim]
        else:
            private_expert_output_dims = expert_output_dims

        logger.info(
            f"[PLE] Homogeneous experts: shared={self.num_shared_experts}, "
            f"specific={self.num_specific_experts}, expert_mlp_dims={expert_mlp_dims}"
            + (f", private_expert_mlp_dims={private_expert_mlp_dims}" if private_expert_mlp_dims else "")
        )

        # ========== 构建 shared experts ==========
        self.shared_experts = nn.ModuleList([
            MLP(
                input_dim=in_dim,
                hidden_dims=expert_output_dims[:-1],
                activation=expert_activation,
                dropout=expert_dropout,
                use_bn=expert_use_bn,
            )
            for _ in range(self.num_shared_experts)
        ])

        # ========== 构建 task-specific experts ==========
        self.specific_experts = nn.ModuleDict()
        for task in tasks:
            num_specific = self.num_specific_experts[task]
            self.specific_experts[task] = nn.ModuleList([
                MLP(
                    input_dim=in_dim,
                    hidden_dims=private_expert_output_dims[:-1],
                    activation=expert_activation,
                    dropout=expert_dropout,
                    use_bn=expert_use_bn,
                )
                for _ in range(num_specific)
            ])
        
        # 同构专家不需要 output align
        self._expert_output_aligner = None
        
        # 存储专家名称（用于日志/metrics）
        # 使用明确的 [SHARE] 和 [PRIV-{task}] 前缀区分共享和私有专家
        self._expert_names: Dict[str, List[str]] = {}
        for task in tasks:
            shared_names = [f"[SHARE]shared_{i}" for i in range(self.num_shared_experts)]
            private_names = [f"[PRIV-{task}]private_{i}" for i in range(self.num_specific_experts[task])]
            self._expert_names[task] = shared_names + private_names

    def _build_hetero_experts(
        self,
        ple_cfg: Dict[str, Any],
        tasks: List[str],
        in_dim: int,
    ) -> None:
        """
        构建异构专家（新逻辑）。
        
        异构专家配置示例:
          experts:
            shared:
              - {name: "mlp_deep", type: "mlp", dims: [256,128], ...}
              - {name: "cross_v2", type: "crossnet_v2", num_layers: 3, ...}
            private:
              ctr:
                - {name: "ctr_cross", type: "crossnet_v2", ...}
              cvr:
                - {name: "cvr_mlp", type: "mlp", ...}
        
        关键点:
          1. 所有 expert 输出必须对齐到 expert_out_dim
          2. 异构专家需要 ExpertOutputAlign 进行输出归一化
          3. 支持 MLP / CrossNet-v2 / Identity 等多种类型
        """
        # 延迟导入以避免循环依赖
        from src.models.mtl.experts import build_expert_list, PerTaskExpertAligner

        experts_cfg = ple_cfg.get("experts", {})
        
        # ========== 解析 expert_out_dim ==========
        # 如果未指定，默认使用 in_dim（与同构专家行为一致）
        expert_out_dim = int(ple_cfg.get("expert_out_dim", in_dim))
        self._expert_out_dim = expert_out_dim
        
        # ========== 构建 shared experts ==========
        shared_specs = experts_cfg.get("shared", [])
        self.num_shared_experts = len(shared_specs)
        
        self.shared_experts = build_expert_list(
            specs=shared_specs,
            in_dim=in_dim,
            out_dim=expert_out_dim,
        )
        
        # 记录 shared expert 名称（使用 [SHARE] 前缀标注）
        shared_expert_names = [
            f"[SHARE]{spec.get('name', f'shared_{i}')}" for i, spec in enumerate(shared_specs)
        ]
        
        # ========== 构建 task-specific experts ==========
        private_cfg = experts_cfg.get("private", {})
        self.specific_experts = nn.ModuleDict()
        self.num_specific_experts: Dict[str, int] = {}
        self._expert_names: Dict[str, List[str]] = {}

        # Allow specific_num_experts to override private expert counts in hetero mode
        specific_cfg = ple_cfg.get("specific_num_experts")
        requested_specific: Dict[str, Optional[int]] = {}
        if isinstance(specific_cfg, int):
            requested_specific = {task: int(specific_cfg) for task in tasks}
        elif isinstance(specific_cfg, dict):
            default_specific = specific_cfg.get("default", None)
            for task in tasks:
                if task in specific_cfg:
                    requested_specific[task] = int(specific_cfg.get(task))
                elif default_specific is not None:
                    requested_specific[task] = int(default_specific)
                else:
                    requested_specific[task] = None
        else:
            requested_specific = {task: None for task in tasks}
        
        for task in tasks:
            task_specs = private_cfg.get(task, [])
            requested = requested_specific.get(task)
            if requested is not None:
                if requested < 0:
                    raise ValueError(f"specific_num_experts[{task}] must be >= 0, got {requested}")
                if requested > len(task_specs):
                    raise ValueError(
                        f"specific_num_experts[{task}]={requested} exceeds "
                        f"available private specs={len(task_specs)}"
                    )
                if requested != len(task_specs):
                    logger.info(
                        "[PLE] Hetero experts: task %s private experts truncated to %d (from %d) due to specific_num_experts",
                        task, requested, len(task_specs)
                    )
                task_specs = task_specs[:requested]
            self.num_specific_experts[task] = len(task_specs)
            
            if task_specs:
                self.specific_experts[task] = build_expert_list(
                    specs=task_specs,
                    in_dim=in_dim,
                    out_dim=expert_out_dim,
                )
                # 使用 [PRIV-{task}] 前缀标注私有专家
                private_names = [
                    f"[PRIV-{task}]{spec.get('name', f'private_{i}')}" for i, spec in enumerate(task_specs)
                ]
            else:
                # 该任务没有 private experts
                self.specific_experts[task] = nn.ModuleList()
                private_names = []
            
            # 合并专家名称列表
            self._expert_names[task] = shared_expert_names + private_names
        
        # ========== 构建 Expert Output Aligner ==========
        # 异构专家输出对齐配置
        align_cfg = ple_cfg.get("expert_output_align", {}) or {}
        use_layernorm = bool(align_cfg.get("layernorm", True))  # 默认开启
        use_learnable_scale = bool(align_cfg.get("learnable_scale", True))  # 默认开启
        align_dropout = float(align_cfg.get("dropout", 0.0))
        
        # 计算每个任务的专家总数
        task_num_experts = {
            task: self.num_shared_experts + self.num_specific_experts[task]
            for task in tasks
        }
        
        self._expert_output_aligner = PerTaskExpertAligner(
            task_num_experts=task_num_experts,
            dim=expert_out_dim,
            layernorm=use_layernorm,
            learnable_scale=use_learnable_scale,
            dropout=align_dropout,
        )
        
        # 日志输出
        logger.info(
            f"[PLE] Heterogeneous experts: shared={self.num_shared_experts} "
            f"[{', '.join(shared_expert_names)}], "
            f"private={self.num_specific_experts}, "
            f"expert_out_dim={expert_out_dim}"
        )
        logger.info(
            f"[PLE] Expert output align: layernorm={use_layernorm}, "
            f"learnable_scale={use_learnable_scale}, dropout={align_dropout}"
        )

    def _get_head_add_cfg(self, task: str) -> Dict[str, bool]:
        """获取每个 head 是否添加 wide/fm logit 的配置（与 mmoe.py 对齐）。"""
        task = task.lower()
        cfg = self.per_head_add.get(task)
        if cfg is not None:
            return {"use_wide": bool(cfg.get("use_wide", False)), "use_fm": bool(cfg.get("use_fm", False))}
        if task == "cvr":
            return {"use_wide": False, "use_fm": False}
        return {"use_wide": True, "use_fm": True}

    def _select_input(self, features: Dict[str, torch.Tensor], backbone_out: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        选择 PLE 的输入张量（与 mmoe.py 对齐）。
        
        如果配置了 composer，则使用 composer 组装输入；
        否则走旧逻辑（直接使用 deep_h 或 emb_concat）。
        """
        # ===== 新路径：使用 composer =====
        if self.composer is not None:
            return self.composer(backbone_out)

        # ===== 旧路径：直接使用 deep_h 或 emb_concat =====
        input_source = getattr(self, "input_source", "deep_h")
        if input_source == "deep_h":
            x = backbone_out.get("deep_h")
            if x is None:
                x = backbone_out.get("h")
            if x is None:
                raise KeyError("PLE requires 'deep_h' (or 'h') from backbone when input=deep_h.")
            return x

        # emb_concat path
        if not hasattr(self.backbone, "feat_emb"):
            raise AttributeError("Backbone has no feat_emb; cannot use emb_concat as PLE input.")
        emb_out = self.backbone.feat_emb(features)
        return emb_out["emb_concat"]

    def forward(self, features: Dict, dense_x: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        PLE forward pass.
        
        与 mmoe.py forward 输出格式一致：
          - per-task logits: results[task] = [B]
          - aux: 可选包含 gates / gate_reg_loss / gate_entropy_mean / gate_lb_kl
        """
        out = self.backbone(features, dense_x=dense_x, return_aux=True)
        if "logit_linear" not in out:
            raise KeyError("Backbone must return 'logit_linear' when return_aux=True.")

        results: Dict[str, torch.Tensor] = {}
        logit_parts_decomposable = not self.use_legacy_pseudo_deepfm

        # 选择输入并做 LayerNorm（与 mmoe.py 一致）
        base_x = self._select_input(features, out)
        base_x = self.layernorm(base_x)

        # ========== 计算 shared experts 输出（一次计算，所有任务共享）==========
        shared_expert_outputs = [expert(base_x) for expert in self.shared_experts]

        # ========== 获取 wide/fm logit ==========
        wide_logit = out.get("logit_linear")
        fm_logit = out.get("fm_logit")
        if wide_logit is not None and wide_logit.dim() == 2 and wide_logit.size(-1) == 1:
            wide_logit = wide_logit.squeeze(-1)
        if fm_logit is not None and fm_logit.dim() == 2 and fm_logit.size(-1) == 1:
            fm_logit = fm_logit.squeeze(-1)

        # 用于 health monitoring
        gate_weights_dict: Dict[str, torch.Tensor] = {}

        # 用于 gate 正则化（保留梯度）
        gate_weights_for_reg: List[torch.Tensor] = []
        gate_num_shared_list: List[int] = []  # 记录每个任务的 shared experts 数量（用于 shared_only scope）

        # ========== 计算当前 step 的 temperature/noise_std（支持 schedule）==========
        current_temp, current_noise = self._get_scheduled_gate_params()

        # ========== 遍历每个任务 ==========
        for task, head in self.towers.items():
            if task not in self.enabled_heads:
                continue

            # 计算该任务的 specific experts 输出
            specific_expert_outputs = [expert(base_x) for expert in self.specific_experts[task]]

            # 拼接所有 expert 输出：[shared..., specific...]
            # 顺序：shared experts 在前，task-specific experts 在后
            all_expert_outputs = shared_expert_outputs + specific_expert_outputs

            # 获取 gate 权重（支持动态 temperature/noise_std）
            gate = self.gates[task]
            gate_w = gate(base_x, temperature_override=current_temp, noise_std_override=current_noise)  # [B, K]

            # 存储用于正则化（保留梯度）
            if self.gate_stabilize_enabled and self.training:
                gate_weights_for_reg.append(gate_w)
                gate_num_shared_list.append(self.num_shared_experts)

            # 存储 gate 权重用于 health monitoring
            if self.log_gates:
                gate_weights_dict[task] = gate_w.detach()

            # ========== Mixing：加权求和 ==========
            # stacked: [B, D, K]
            stacked = torch.stack(all_expert_outputs, dim=2)
            
            # ========== 异构专家输出对齐（仅在启用时生效）==========
            if self._expert_output_aligner is not None:
                stacked = self._expert_output_aligner(task, stacked)
            
            # gate_w: [B, K] -> [B, K, 1]
            weights = gate_w.unsqueeze(-1)  # [B, K, 1]
            # bmm: [B, D, K] @ [B, K, 1] -> [B, D, 1] -> squeeze -> [B, D]
            mixed = torch.bmm(stacked, weights).squeeze(-1)

            # 通过 task head 获取 logit
            task_logit = head(mixed)

            # ========== 添加 wide/fm logit（与 mmoe.py 对齐）==========
            if self.use_legacy_pseudo_deepfm:
                if wide_logit is not None:
                    task_logit = task_logit + wide_logit
                results[task] = task_logit
                if self.return_logit_parts:
                    results[f"{task}_logit_parts"] = {
                        "wide": wide_logit,
                        "fm": None,
                        "deep": task_logit - (wide_logit if wide_logit is not None else 0.0),
                        "total": task_logit,
                    }
                continue

            add_cfg = self._get_head_add_cfg(task)
            wide_term = wide_logit if (add_cfg.get("use_wide", False) and wide_logit is not None) else 0.0
            fm_term = fm_logit if (add_cfg.get("use_fm", False) and fm_logit is not None) else 0.0
            task_logit = task_logit + wide_term + fm_term
            results[task] = task_logit

            if self.return_logit_parts:
                wide_comp = wide_term if torch.is_tensor(wide_term) else torch.tensor(wide_term, device=task_logit.device)
                fm_comp = fm_term if torch.is_tensor(fm_term) else torch.tensor(fm_term, device=task_logit.device)
                results[f"{task}_logit_parts"] = {
                    "wide": wide_comp,
                    "fm": fm_comp,
                    "deep": task_logit - wide_comp - fm_comp,
                    "total": task_logit,
                }

        results["logit_parts_decomposable"] = logit_parts_decomposable

        # ========== 合并 aux 并添加 gate 相关信息 ==========
        aux = dict(out.get("aux", {})) if "aux" in out else {}
        if self.log_gates and gate_weights_dict:
            aux["gates"] = gate_weights_dict
            
            # ========== 新增：计算 per-expert routing metrics ==========
            # 对每个 task 计算各专家的 hit rate（被选为 top1 的比例）
            if hasattr(self, "_expert_names") and self._expert_names:
                expert_metrics = self._compute_expert_routing_metrics(gate_weights_dict)
                for k, v in expert_metrics.items():
                    aux[k] = v

        # ========== 计算 gate 正则化损失（与 mmoe.py 对齐，但支持 shared_only scope）==========
        if self.gate_stabilize_enabled and self.training and gate_weights_for_reg:
            gate_reg_loss, gate_entropy_mean, gate_lb_kl = self._compute_gate_reg(
                gate_weights_for_reg, gate_num_shared_list
            )
            aux["gate_reg_loss"] = gate_reg_loss  # 保留梯度用于反传
            aux["gate_entropy_mean"] = gate_entropy_mean.detach()
            aux["gate_lb_kl"] = gate_lb_kl.detach()

        # ========== 计算 shared mass floor 正则化损失（新增）==========
        # 只在 enabled=True 且 training 时生效，验证时只记录 metrics 不加 loss
        if self.mass_floor_enabled and gate_weights_dict:
            if self.training:
                mass_floor_loss, mass_floor_metrics = self._compute_shared_mass_floor_loss(gate_weights_dict)
                # 将 mass floor loss 合并到 gate_reg_loss（如果存在）或单独记录
                if "gate_reg_loss" in aux:
                    aux["gate_reg_loss"] = aux["gate_reg_loss"] + mass_floor_loss
                else:
                    aux["gate_reg_loss"] = mass_floor_loss
                aux["gate_shared_mass_floor_loss"] = mass_floor_loss.detach()
                # 记录 per-task metrics
                for k, v in mass_floor_metrics.items():
                    aux[k] = v
            else:
                # 验证时只计算 metrics，不加 loss
                _, mass_floor_metrics = self._compute_shared_mass_floor_loss(gate_weights_dict)
                for k, v in mass_floor_metrics.items():
                    aux[k] = v

        if aux:
            results["aux"] = aux

        return results

    def _compute_gate_reg(
        self,
        gate_weights_list: List[torch.Tensor],
        num_shared_list: List[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算 gate 正则化损失：entropy regularization + load-balance KL divergence.
        
        与 mmoe.py 的 _compute_gate_reg 对齐，但增加了 gate_reg_scope 支持。
        
        ========== 改动 D: gate_reg_scope 详解 ==========
        
        背景：PLE 中每个任务的 gate 权重维度为 K = E_s + E_t（shared + task-specific）。
        
        问题：如果直接对全部 K 个权重做 entropy/kl 正则，会导致：
          1. 每个任务的 K 不同（因为 E_t 可能不同），无法公平比较
          2. task-specific experts 本身就是为该任务设计的，强制它们均匀分布不合理
        
        解决方案（gate_reg_scope="shared_only"，默认）：
          1. 只取 gate_w 的前 E_s 个权重（对应 shared experts）
          2. 将这些权重重新归一化到和为 1（因为原始 gate_w 是在全部 K 个 experts 上做的 softmax）
          3. 只对归一化后的 shared 权重计算 entropy/kl 正则
        
        这样做的好处：
          - 保证正则化只影响 shared experts 的路由分布
          - 不干扰 task-specific experts 的学习
          - 各任务的正则化是可比的（都是在 E_s 个 experts 上计算）
        
        注意：正则 scope 只影响 gate_reg_loss 的计算，不影响 mixing（mixing 仍使用完整的 gate_w）。
        
        Args:
            gate_weights_list: List of gate weight tensors [B, K] for each task
            num_shared_list: List of num_shared_experts for each task（用于 shared_only scope 切片）
            
        Returns:
            gate_reg_loss: Combined regularization loss (keeps gradient)
            gate_entropy_mean: Mean entropy across all tasks (detached, for logging)
            gate_lb_kl: Load-balance KL divergence (detached, for logging)
        """
        eps = self.gate_eps
        device = gate_weights_list[0].device

        # 使用 float32 计算以保证数值稳定（AMP 兼容）
        total_entropy = torch.tensor(0.0, device=device, dtype=torch.float32)
        total_kl = torch.tensor(0.0, device=device, dtype=torch.float32)

        for gate_w, num_shared in zip(gate_weights_list, num_shared_list):
            # 转换为 float32
            gate_w_f32 = gate_w.float()

            # ========== 根据 gate_reg_scope 选择要正则化的权重 ==========
            if self.gate_reg_scope == "shared_only":
                # 只取 shared experts 的权重（前 num_shared 个）
                shared_w = gate_w_f32[:, :num_shared]  # [B, E_s]
                
                # 重新归一化：因为原始 gate_w 是在全部 K 个 experts 上做的 softmax，
                # shared_w 的和不一定为 1，需要重新归一化才能作为有效的概率分布
                shared_w = shared_w / (shared_w.sum(dim=-1, keepdim=True) + eps)  # [B, E_s]
                
                # 用于正则化的权重
                reg_w = shared_w
                num_experts_for_reg = num_shared
            else:
                # gate_reg_scope == "all"：对全部 experts 做正则化
                reg_w = gate_w_f32
                num_experts_for_reg = gate_w_f32.size(-1)

            # ========== Entropy: H = -sum(w * log(w + eps)) ==========
            # Higher entropy = more uniform distribution (desirable to avoid collapse)
            log_w = torch.log(reg_w + eps)
            entropy = -torch.sum(reg_w * log_w, dim=-1)  # [B]
            total_entropy = total_entropy + entropy.mean()

            # ========== Load-balance KL: KL(mean_w || uniform) ==========
            # Measures how far the average gate assignment is from uniform
            mean_w = reg_w.mean(dim=0)  # [E_s] or [K]
            # KL(mean_w || uniform) = sum(mean_w * (log(mean_w) - log(1/E)))
            #                       = sum(mean_w * log(mean_w)) + log(E)
            kl = torch.sum(mean_w * (torch.log(mean_w + eps) - math.log(1.0 / num_experts_for_reg)))
            total_kl = total_kl + kl

        num_tasks = len(gate_weights_list)
        mean_entropy = total_entropy / num_tasks
        mean_kl = total_kl / num_tasks

        # ========== 计算正则化损失 ==========
        # - Entropy term: we want HIGH entropy, so loss = -entropy (minimize -entropy = maximize entropy)
        # - KL term: we want LOW KL (close to uniform), so loss = +KL
        gate_reg_loss = torch.tensor(0.0, device=device, dtype=torch.float32)

        if self.entropy_reg_weight > 0:
            gate_reg_loss = gate_reg_loss + self.entropy_reg_weight * (-mean_entropy)

        if self.load_balance_kl_weight > 0:
            gate_reg_loss = gate_reg_loss + self.load_balance_kl_weight * mean_kl

        return gate_reg_loss, mean_entropy, mean_kl

    def _get_scheduled_gate_params(self) -> tuple[Optional[float], Optional[float]]:
        """
        获取当前 step 的 temperature 和 noise_std（支持线性退火 schedule）。
        
        如果 gate_schedule_enabled=False，返回 (None, None)，
        Gate 将使用其默认的 self.temperature 和 self.noise_std。
        
        Returns:
            (current_temp, current_noise): 当前 step 应使用的 temperature 和 noise_std
                                           如果 schedule 未启用则返回 (None, None)
        """
        if not self.gate_schedule_enabled:
            return None, None
        
        # 计算当前进度 [0, 1]
        progress = min(1.0, self._current_step / max(1, self._total_steps))
        
        # warmup 阶段保持 start 值
        warm_frac = self.gate_schedule_warm_frac
        if progress < warm_frac:
            # 在 warmup 期间保持 start 值
            current_temp = self.gate_schedule_temp_start
            current_noise = self.gate_schedule_noise_start
        else:
            # warmup 后线性退火到 end 值
            decay_progress = (progress - warm_frac) / (1.0 - warm_frac + 1e-8)
            decay_progress = min(1.0, max(0.0, decay_progress))
            
            current_temp = self.gate_schedule_temp_start + decay_progress * (
                self.gate_schedule_temp_end - self.gate_schedule_temp_start
            )
            current_noise = self.gate_schedule_noise_start + decay_progress * (
                self.gate_schedule_noise_end - self.gate_schedule_noise_start
            )
        
        return current_temp, current_noise

    def set_step(self, current_step: int, total_steps: int) -> None:
        """
        设置当前训练 step（由 trainer 在每个 step 调用）。
        
        用于 gate temperature/noise_std 的 schedule 计算。
        
        Args:
            current_step: 当前全局 step（从 0 开始）
            total_steps: 总训练 step 数
        """
        self._current_step = current_step
        self._total_steps = max(1, total_steps)

    def _compute_shared_mass_floor_loss(
        self,
        gate_weights_dict: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """
        计算 shared mass floor 正则化损失。
        
        对每个 task 的 gate weights w（shape [B, E_task]），计算：
          mass_shared = mean( sum_{j in [0, num_shared)} w[:, j] )
        增加惩罚：loss += weight * relu(min_mass - mass_shared)^2
        
        Args:
            gate_weights_dict: Dict[task -> gate_weights [B, K]]
            
        Returns:
            total_loss: 总的 mass floor loss（保留梯度）
            metrics: Dict 包含每个 task 的 gate_shared_mass_mean_{task} 和 gate_shared_mass_floor_loss_{task}
        """
        device = next(iter(gate_weights_dict.values())).device
        total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        metrics: Dict[str, float] = {}
        
        num_shared = self.num_shared_experts
        
        for task, gate_w in gate_weights_dict.items():
            # 只处理配置了 mass floor 的 task
            if task not in self.mass_floor_per_task:
                continue
            
            task_cfg = self.mass_floor_per_task[task]
            min_mass = task_cfg["min_mass"]
            weight = task_cfg["weight"]
            
            # 转换为 float32 以保证数值稳定（AMP 兼容）
            gate_w_f32 = gate_w.float()
            
            # 计算 shared mass：sum_{j in [0, num_shared)} w[:, j]
            # gate_w 的维度顺序：[shared_0, shared_1, ..., shared_{n-1}, private_0, ...]
            shared_mass = gate_w_f32[:, :num_shared].sum(dim=-1)  # [B]
            mean_shared_mass = shared_mass.mean()  # scalar
            
            # 惩罚：relu(min_mass - mass_shared)^2
            # 当 mass_shared < min_mass 时产生正惩罚
            floor_violation = torch.relu(min_mass - mean_shared_mass)
            task_loss = weight * (floor_violation ** 2)
            
            total_loss = total_loss + task_loss
            
            # 记录 metrics（detach 后转 float）
            metrics[f"gate_shared_mass_mean_{task}"] = float(mean_shared_mass.detach().cpu().item())
            metrics[f"gate_shared_mass_floor_loss_{task}"] = float(task_loss.detach().cpu().item())
        
        return total_loss, metrics

    def _compute_expert_routing_metrics(
        self,
        gate_weights_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        """
        计算 per-expert 路由统计指标（用于监控异构专家的利用率）。
        
        对每个 task 计算：
          - gate_{task}_expert_top1_share: Dict[expert_name -> top1 选择频率]
          - gate_{task}_expert_mean_weight: Dict[expert_name -> 平均权重]
          - gate_{task}_share_agg_*: share experts 的聚合指标
          - gate_{task}_private_agg_*: private experts 的聚合指标
        
        这些指标有助于诊断：
          - 某些专家是否被忽略（权重接近 0）
          - 某些专家是否垄断了所有流量（top1 share 接近 1）
          - 异构专家是否有分化（不同类型专家被不同程度使用）
          - Share vs Private experts 的流量分配是否合理
        
        Args:
            gate_weights_dict: Dict[task -> gate_weights [B, K]]
            
        Returns:
            Dict containing per-task per-expert routing statistics and aggregated share/private stats
        """
        metrics: Dict[str, Any] = {}
        
        for task, gate_w in gate_weights_dict.items():
            # 跳过未启用的 task
            if task not in self._expert_names:
                continue
            
            expert_names = self._expert_names[task]
            K = len(expert_names)
            
            # 确保维度匹配
            if gate_w.size(-1) != K:
                logger.warning(
                    f"Gate weights dim ({gate_w.size(-1)}) != expert count ({K}) for task {task}"
                )
                continue
            
            gate_w_f32 = gate_w.float()
            
            # ========== Top-1 Selection Share ==========
            # 计算每个 expert 被选为 top1 的频率
            top1_indices = torch.argmax(gate_w_f32, dim=-1)  # [B]
            top1_counts = torch.bincount(top1_indices, minlength=K).float()  # [K]
            top1_share = top1_counts / (top1_counts.sum() + 1e-8)  # 归一化
            
            # 转换为 expert_name -> share 的字典
            expert_top1_share = {
                name: float(top1_share[i].cpu().item())
                for i, name in enumerate(expert_names)
            }
            metrics[f"gate_{task}_expert_top1_share"] = expert_top1_share
            
            # ========== Mean Weight per Expert ==========
            # 计算每个 expert 的平均 gate 权重
            mean_weights = gate_w_f32.mean(dim=0)  # [K]
            expert_mean_weight = {
                name: float(mean_weights[i].cpu().item())
                for i, name in enumerate(expert_names)
            }
            metrics[f"gate_{task}_expert_mean_weight"] = expert_mean_weight
            
            # ========== Share vs Private 聚合指标 ==========
            # 根据 [SHARE] 和 [PRIV-*] 前缀分类
            share_indices = []
            private_indices = []
            
            for idx, name in enumerate(expert_names):
                if name.startswith("[SHARE]"):
                    share_indices.append(idx)
                elif name.startswith("[PRIV-"):
                    private_indices.append(idx)
            
            # Share experts 聚合统计
            if share_indices:
                share_weights = gate_w_f32[:, share_indices]  # [B, num_share]
                share_mean_weight = float(share_weights.mean().cpu().item())
                share_top1_count = (top1_indices.unsqueeze(1) == torch.tensor(share_indices, device=top1_indices.device)).sum().float()
                share_top1_share = float((share_top1_count / len(top1_indices)).cpu().item())
                
                metrics[f"gate_{task}_share_agg_mean_weight"] = share_mean_weight
                metrics[f"gate_{task}_share_agg_top1_share"] = share_top1_share
            
            # Private experts 聚合统计
            if private_indices:
                private_weights = gate_w_f32[:, private_indices]  # [B, num_private]
                private_mean_weight = float(private_weights.mean().cpu().item())
                private_top1_count = (top1_indices.unsqueeze(1) == torch.tensor(private_indices, device=top1_indices.device)).sum().float()
                private_top1_share = float((private_top1_count / len(top1_indices)).cpu().item())
                
                metrics[f"gate_{task}_private_agg_mean_weight"] = private_mean_weight
                metrics[f"gate_{task}_private_agg_top1_share"] = private_top1_share
            
            # ========== 额外指标：是否启用了异构专家 ==========
            metrics["hetero_experts_enabled"] = self._use_hetero_experts
        
        return metrics

    def get_expert_health_data(self) -> Dict[str, Any]:
        """
        获取专家健康诊断所需的模型内部数据。
        
        用于 ExpertHealthDiagnostics 收集诊断数据。
        
        Returns:
            Dict containing:
              - expert_modules: List[(name, module)] 所有专家模块
              - expert_names: Dict[task -> List[str]] 每任务专家名称
              - aligners: Dict[task -> module] 每任务的 output aligner
              - hetero_enabled: bool 是否启用异构专家
        """
        data: Dict[str, Any] = {
            "hetero_enabled": self._use_hetero_experts,
            "expert_names": self._expert_names.copy() if self._expert_names else {},
        }
        
        # 收集所有专家模块
        expert_modules: List[Tuple[str, torch.nn.Module]] = []
        
        # Shared experts
        for i, expert in enumerate(self.shared_experts):
            name = f"[SHARE]shared_{i}"
            # 尝试从配置获取实际名称
            if hasattr(self, "_expert_names"):
                for task_names in self._expert_names.values():
                    for n in task_names:
                        if n.startswith("[SHARE]") and f"_{i}" in n:
                            name = n
                            break
                    break
            expert_modules.append((name, expert))
        
        # Task-specific experts
        for task, experts in self.specific_experts.items():
            for i, expert in enumerate(experts):
                name = f"[PRIV-{task}]private_{i}"
                if task in self._expert_names:
                    private_names = [n for n in self._expert_names[task] if f"[PRIV-{task}]" in n]
                    if i < len(private_names):
                        name = private_names[i]
                expert_modules.append((name, expert))
        
        data["expert_modules"] = expert_modules
        
        # 收集 aligners
        aligners = {}
        if self._expert_output_aligner is not None:
            # PerTaskExpertAligner 包含多个任务的 aligner
            if hasattr(self._expert_output_aligner, "aligners"):
                for task, aligner in self._expert_output_aligner.aligners.items():
                    aligners[task] = aligner
            else:
                # 单个 aligner（兼容旧版本）
                aligners["default"] = self._expert_output_aligner
        
        data["aligners"] = aligners
        
        return data


__all__ = ["PLE"]
