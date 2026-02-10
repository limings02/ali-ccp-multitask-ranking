from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Set

import torch
from torch import nn

from src.models.backbones.layers import MLP
from src.models.heads import TaskHead
from src.models.mtl.composer import MMoEInputComposer, build_composer_from_config
from src.models.residual_head import build_residual_head

logger = logging.getLogger(__name__)


class _Gate(nn.Module):
    """
    Per-task gate that outputs a softmax over experts.
    
    Supports optional stabilization features (改动 C):
    - temperature: >1 makes softmax smoother, <1 makes it sharper
    - noise_std: Gaussian noise added to logits (training only)
    """

    def __init__(
        self, 
        in_dim: int, 
        num_experts: int, 
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

    def forward(self, x: torch.Tensor, return_logits: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with optional stabilization.
        
        Args:
            x: Input tensor [B, D]
            return_logits: If True, return (weights, logits) tuple
            
        Returns:
            weights: Softmax weights [B, E]
            logits (optional): Raw logits before softmax [B, E]
        """
        logits = self.net(x)
        
        # Add noise during training (改动 C: gate stabilization)
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise
        
        # Apply temperature scaling
        if self.temperature != 1.0:
            logits = logits / self.temperature
        
        weights = torch.softmax(logits, dim=-1)
        
        if return_logits:
            return weights, logits
        return weights


class MMoE(nn.Module):
    """
    Minimal MMoE block plugged in place of the shared-bottom towers.

    - Uses backbone-provided representations (default deep_h) as expert inputs.
    - Keeps head interface identical to SharedBottom: returns per-task logits with optional wide/FM additions.
    - Optionally returns gate weights for health monitoring (controlled by log_gates flag).
    """

    def __init__(
        self,
        backbone: nn.Module,
        head_cfg: Dict,
        mmoe_cfg: Dict,
        enabled_heads: Optional[List[str]] = None,
        use_legacy_pseudo_deepfm: bool = True,
        return_logit_parts: bool = False,
        per_head_add: Optional[Dict[str, Dict[str, bool]]] = None,
        head_priors: Optional[Dict[str, float]] = None,
        log_gates: bool = False,
        esmm_residual_cfg: Optional[Dict] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.use_legacy_pseudo_deepfm = bool(use_legacy_pseudo_deepfm)
        self.return_logit_parts = bool(return_logit_parts)
        self.per_head_add = per_head_add or {}
        self.head_priors = head_priors or {}
        self.log_gates = bool(log_gates)  # Flag to enable gate weight logging

        tasks = head_cfg.get("tasks") or ["ctr", "cvr"]
        enabled = enabled_heads or tasks
        self.tasks: List[str] = tasks
        self.enabled_heads: Set[str] = {h.lower() for h in enabled}

        # ---- 构建可选的 input composer ----
        # 当配置中有 add_fm_vec/add_emb 等新字段时，使用 composer 组装输入
        # 否则走旧逻辑（直接使用 deep_h）
        self.composer: Optional[MMoEInputComposer] = build_composer_from_config(backbone, mmoe_cfg)
        
        # ---- resolve input dim ----
        if self.composer is not None:
            # 使用 composer 输出维度
            in_dim = self.composer.out_dim
            logger.info(f"[MMoE] Using composer, input dim = {in_dim}")
        else:
            # 旧逻辑：直接使用 deep_h 或 emb_concat
            input_source = str(mmoe_cfg.get("input", "deep_h")).lower()
            self.input_source = input_source

            deep_dim = getattr(backbone, "deep_out_dim", getattr(backbone, "out_dim", None))
            emb_concat_dim = getattr(getattr(backbone, "feat_emb", None), "concat_dim", None)

            if input_source == "deep_h":
                in_dim = deep_dim
            elif input_source == "emb_concat":
                in_dim = emb_concat_dim
            else:
                raise ValueError(f"Unsupported mmoe.input value: {input_source}")

            if in_dim is None:
                raise ValueError("MMoE: could not infer input dimension from backbone; ensure deep_out_dim or feat_emb.concat_dim is available.")
            logger.info(f"[MMoE] Using legacy input_source='{input_source}', input dim = {in_dim}")

        self.in_dim = in_dim
        self.layernorm = nn.LayerNorm(in_dim)

        # ---- experts ----
        num_experts = int(mmoe_cfg.get("num_experts", 4))
        expert_mlp_dims = list(mmoe_cfg.get("expert_mlp_dims", []))
        expert_output_dims = expert_mlp_dims + [in_dim] if expert_mlp_dims else [in_dim]

        self.experts = nn.ModuleList(
            [
                MLP(
                    input_dim=in_dim,
                    hidden_dims=expert_output_dims,
                    activation=str(mmoe_cfg.get("activation", "relu")),
                    dropout=float(mmoe_cfg.get("dropout", 0.0)),
                    use_bn=bool(mmoe_cfg.get("use_bn", False)),
                )
                for _ in range(num_experts)
            ]
        )

        # ---- gates (per task) ----
        gate_type = str(mmoe_cfg.get("gate_type", "linear"))
        gate_hidden_dims = mmoe_cfg.get("gate_hidden_dims")
        
        # ============================================================
        # 改动 C: Gate stabilization configuration
        # ============================================================
        gate_stabilize_cfg = mmoe_cfg.get("gate_stabilize", {}) or {}
        self.gate_stabilize_enabled = bool(gate_stabilize_cfg.get("enabled", False))
        self.gate_temperature = float(gate_stabilize_cfg.get("temperature", 1.0))
        self.gate_noise_std = float(gate_stabilize_cfg.get("noise_std", 0.0))
        self.gate_eps = float(gate_stabilize_cfg.get("eps", 1e-8))
        self.entropy_reg_weight = float(gate_stabilize_cfg.get("entropy_reg_weight", 0.0))
        self.load_balance_kl_weight = float(gate_stabilize_cfg.get("load_balance_kl_weight", 0.0))
        self.gate_log_stats = bool(gate_stabilize_cfg.get("log_stats", True))
        self.num_experts = num_experts
        
        if self.gate_stabilize_enabled:
            logger.info(
                "[MMoE] gate_stabilize enabled: temp=%.2f noise_std=%.3f "
                "entropy_reg=%.2e lb_kl_reg=%.2e eps=%.2e",
                self.gate_temperature, self.gate_noise_std,
                self.entropy_reg_weight, self.load_balance_kl_weight, self.gate_eps
            )
        
        self.gates = nn.ModuleDict(
            {
                task: _Gate(
                    in_dim=in_dim, 
                    num_experts=num_experts, 
                    gate_type=gate_type, 
                    hidden_dims=gate_hidden_dims,
                    temperature=self.gate_temperature if self.gate_stabilize_enabled else 1.0,
                    noise_std=self.gate_noise_std if self.gate_stabilize_enabled else 0.0,
                ) 
                for task in tasks
            }
        )

        # ---- towers / heads ----
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

            prior_map = {"ctr": 0.07, "cvr": 0.01}
            prior = float(self.head_priors.get(task, prior_map.get(task, 0.5)))
            prior = min(max(prior, 1e-6), 1 - 1e-6)
            bias_init = float(torch.log(torch.tensor(prior / (1 - prior))))
            if self.towers[task].out_proj.bias is not None:
                nn.init.constant_(self.towers[task].out_proj.bias, bias_init)

        # ============================================================
        # ESMM Residual Head: 可选的 CTCVR logit 残差修正
        # 输入为 CVR 分支的 MMoE 混合表征（进入 CVR tower 前）
        # ============================================================
        self._residual_cfg = esmm_residual_cfg or {}
        self.residual_head = build_residual_head(in_dim, self._residual_cfg)
        self._residual_stop_grad_input = bool(self._residual_cfg.get("stop_grad_input", False))

    def _get_head_add_cfg(self, task: str) -> Dict[str, bool]:
        task = task.lower()
        cfg = self.per_head_add.get(task)
        if cfg is not None:
            return {"use_wide": bool(cfg.get("use_wide", False)), "use_fm": bool(cfg.get("use_fm", False))}
        if task == "cvr":
            return {"use_wide": False, "use_fm": False}
        return {"use_wide": True, "use_fm": True}

    def _select_input(self, features: Dict[str, torch.Tensor], backbone_out: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        选择 MMoE 的输入张量。
        
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
                raise KeyError("MMoE requires 'deep_h' (or 'h') from backbone when input=deep_h.")
            return x

        # emb_concat path: recompute via backbone.feat_emb to avoid changing backbone outputs
        if not hasattr(self.backbone, "feat_emb"):
            raise AttributeError("Backbone has no feat_emb; cannot use emb_concat as MMoE input.")
        emb_out = self.backbone.feat_emb(features)
        return emb_out["emb_concat"]

    def forward(self, features: Dict, dense_x: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        out = self.backbone(features, dense_x=dense_x, return_aux=True)
        if "logit_linear" not in out:
            raise KeyError("Backbone must return 'logit_linear' when return_aux=True.")

        results: Dict[str, torch.Tensor] = {}
        logit_parts_decomposable = not self.use_legacy_pseudo_deepfm

        base_x = self._select_input(features, out)
        base_x = self.layernorm(base_x)

        expert_outputs = [expert(base_x) for expert in self.experts]

        def _mix(gate_weights: torch.Tensor) -> torch.Tensor:
            # gate_weights: [B, E]; expert_outputs: list of [B, D]
            stacked = torch.stack(expert_outputs, dim=2)  # [B, D, E]
            weights = gate_weights.unsqueeze(1)  # [B,1,E]
            mixed = torch.bmm(stacked, weights.transpose(1, 2)).squeeze(-1)  # [B, D]
            return mixed

        wide_logit = out.get("logit_linear")
        fm_logit = out.get("fm_logit")
        if wide_logit is not None and wide_logit.dim() == 2 and wide_logit.size(-1) == 1:
            wide_logit = wide_logit.squeeze(-1)
        if fm_logit is not None and fm_logit.dim() == 2 and fm_logit.size(-1) == 1:
            fm_logit = fm_logit.squeeze(-1)

        # Collect gate weights if logging is enabled
        gate_weights_dict: Dict[str, torch.Tensor] = {}
        
        # ============================================================
        # 改动 C: Collect gate weights for regularization (keep gradient)
        # ============================================================
        gate_weights_for_reg: List[torch.Tensor] = []

        # ============================================================
        # Residual Head: 保存 CVR 分支 MMoE 混合表征, 用于残差修正
        # ============================================================
        cvr_mmoe_repr: Optional[torch.Tensor] = None

        for task, head in self.towers.items():
            if task not in self.enabled_heads:
                continue
            gate = self.gates[task]
            gate_w = gate(base_x)  # [B, E]
            
            # Store for regularization (keep gradient for training)
            if self.gate_stabilize_enabled and self.training:
                gate_weights_for_reg.append(gate_w)

            # Store gate weights for health monitoring if enabled
            if self.log_gates:
                gate_weights_dict[task] = gate_w.detach()

            task_h = _mix(gate_w)

            # 捕获 CVR 分支的 MMoE 混合输出, 供 ResidualHead 使用
            if task == "cvr" and self.residual_head is not None:
                cvr_mmoe_repr = task_h

            task_logit = head(task_h)

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

        # ============================================================
        # ESMM Residual Head: 计算残差 logit 并存入 results
        # 仅当 residual_head 存在且 CVR 分支已激活时生效
        # ============================================================
        if self.residual_head is not None and cvr_mmoe_repr is not None:
            # stop_grad_input: 若 True, 阻断 L_ctcvr 向 shared/MMoE 表征的梯度
            # 默认 False, 允许 ctcvr loss 更新 shared/moe 表征
            res_input = cvr_mmoe_repr.detach() if self._residual_stop_grad_input else cvr_mmoe_repr
            r_logit = self.residual_head(res_input)  # [B, 1]
            results["residual_logit"] = r_logit

        # Merge aux from backbone and add gate weights if logging is enabled
        aux = dict(out.get("aux", {})) if "aux" in out else {}
        if self.log_gates and gate_weights_dict:
            aux["gates"] = gate_weights_dict
        
        # ============================================================
        # 改动 C: Compute gate regularization loss (entropy + load-balance KL)
        # Only computed during training when gate_stabilize is enabled
        # ============================================================
        if self.gate_stabilize_enabled and self.training and gate_weights_for_reg:
            gate_reg_loss, gate_entropy_mean, gate_lb_kl = self._compute_gate_reg(gate_weights_for_reg)
            aux["gate_reg_loss"] = gate_reg_loss  # Keep gradient for backprop
            aux["gate_entropy_mean"] = gate_entropy_mean.detach()
            aux["gate_lb_kl"] = gate_lb_kl.detach()
        
        if aux:
            results["aux"] = aux

        return results
    
    def _compute_gate_reg(self, gate_weights_list: List[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute gate regularization: entropy regularization + load-balance KL divergence.
        
        改动 C: 工业常见的 gate 稳定化正则项
        
        Args:
            gate_weights_list: List of gate weight tensors [B, E] for each task
            
        Returns:
            gate_reg_loss: Combined regularization loss (keeps gradient)
            gate_entropy_mean: Mean entropy across all tasks (detached, for logging)
            gate_lb_kl: Load-balance KL divergence (detached, for logging)
        """
        eps = self.gate_eps
        num_experts = self.num_experts
        
        # Compute in float32 for numerical stability (AMP compatibility)
        total_entropy = torch.tensor(0.0, device=gate_weights_list[0].device, dtype=torch.float32)
        total_kl = torch.tensor(0.0, device=gate_weights_list[0].device, dtype=torch.float32)
        
        for gate_w in gate_weights_list:
            # Convert to float32 for stability
            gate_w_f32 = gate_w.float()
            
            # Entropy: H = -sum(w * log(w + eps))
            # Higher entropy = more uniform distribution (desirable to avoid collapse)
            log_w = torch.log(gate_w_f32 + eps)
            entropy = -torch.sum(gate_w_f32 * log_w, dim=-1)  # [B]
            total_entropy = total_entropy + entropy.mean()
            
            # Load-balance KL: KL(mean_w || uniform)
            # Measures how far the average gate assignment is from uniform
            mean_w = gate_w_f32.mean(dim=0)  # [E]
            uniform = torch.ones_like(mean_w) / num_experts
            # KL(mean_w || uniform) = sum(mean_w * (log(mean_w) - log(1/E)))
            # = sum(mean_w * log(mean_w)) + log(E)
            kl = torch.sum(mean_w * (torch.log(mean_w + eps) - math.log(1.0 / num_experts)))
            total_kl = total_kl + kl
        
        num_tasks = len(gate_weights_list)
        mean_entropy = total_entropy / num_tasks
        mean_kl = total_kl / num_tasks
        
        # Regularization loss:
        # - Entropy term: we want HIGH entropy, so loss = -entropy (minimize -entropy = maximize entropy)
        # - KL term: we want LOW KL (close to uniform), so loss = +KL
        gate_reg_loss = torch.tensor(0.0, device=gate_weights_list[0].device, dtype=torch.float32)
        
        if self.entropy_reg_weight > 0:
            # Negative entropy (we want to maximize entropy, so minimize -entropy)
            gate_reg_loss = gate_reg_loss + self.entropy_reg_weight * (-mean_entropy)
        
        if self.load_balance_kl_weight > 0:
            gate_reg_loss = gate_reg_loss + self.load_balance_kl_weight * mean_kl
        
        return gate_reg_loss, mean_entropy, mean_kl
    
    def get_expert_health_data(self) -> Dict[str, Any]:
        """
        获取专家健康诊断所需的模型内部数据。
        
        用于 ExpertHealthDiagnostics 收集诊断数据。
        
        Returns:
            Dict containing:
              - expert_modules: List[(name, module)] 所有专家模块
              - expert_names: Dict[task -> List[str]] 每任务专家名称
              - aligners: Dict[task -> module] (MMoE 无 aligner，返回空字典)
              - hetero_enabled: bool (MMoE 专家同构，返回 False)
        """
        from typing import List, Tuple
        
        data: Dict[str, Any] = {
            "hetero_enabled": False,  # MMoE 专家同构
            "expert_names": {},
        }
        
        # 收集所有专家模块
        expert_modules: List[Tuple[str, torch.nn.Module]] = []
        for i, expert in enumerate(self.experts):
            name = f"expert_{i}"
            expert_modules.append((name, expert))
        
        data["expert_modules"] = expert_modules
        
        # 为每个任务构建专家名称列表（MMoE 中所有任务共享同一组专家）
        expert_names_list = [f"expert_{i}" for i in range(len(self.experts))]
        for task in self.tasks:
            data["expert_names"][task] = expert_names_list
        
        # MMoE 无 output aligner
        data["aligners"] = {}
        
        return data


__all__ = ["MMoE"]
