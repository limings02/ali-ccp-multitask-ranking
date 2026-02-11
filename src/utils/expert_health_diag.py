"""
Expert Health Diagnostics Module for Heterogeneous PLE.

专家健康诊断模块，用于监控异构专家的健康状态，辅助判断是否需要迁移到 HoME 架构。

诊断指标包括：
1. 专家利用率诊断：死亡检测、垄断检测、Gini系数
2. 专家输出分布监控：per-expert统计、专家间相似度
3. 专家梯度监控：per-expert梯度范数、消失/爆炸检测
4. 异构类型特化分析：按类型聚合、跨任务特化分数
5. Output Aligner监控：learnable_scale统计

诊断结果写入 run_dir/expert_health_diag.jsonl，与 metrics.jsonl 分离。

Author: Expert Health Diagnostics for PLE-Lite
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Dataclasses
# =============================================================================

@dataclass
class UtilizationConfig:
    """专家利用率诊断配置"""
    enabled: bool = True
    dead_mean_threshold: float = 0.01       # mean_weight < 1% 认定 soft-dead
    dead_p95_threshold: float = 0.01        # 可选更严格：p95 < 1%
    dead_require_p95: bool = False          # True: dead 需同时满足 mean+p95
    rank_dom_top1_threshold: float = 0.95   # top1_share > 95% 才考虑 rank dominance
    rank_dom_uniform_eps: float = 0.03      # |mean_weight - 1/K| < eps 视为近均匀
    rank_dom_std_eps: float = 0.02          # std(mean_weight) < eps 视为整体近均匀
    monopoly_top1_threshold: float = 0.95   # top1_share > 95%
    monopoly_mean_threshold: Optional[float] = None
    # None 时按 K 动态计算: max(0.5, 1/K + 0.2)
    compute_gini: bool = True


@dataclass
class OutputStatsConfig:
    """专家输出分布监控配置"""
    enabled: bool = True
    per_expert_stats: bool = True
    cross_expert_similarity: bool = True
    similarity_alert_threshold: float = 0.95


@dataclass
class GradientConfig:
    """专家梯度监控配置"""
    enabled: bool = True
    per_expert_grad_norm: bool = True
    vanish_threshold: float = 1e-6
    explode_threshold: float = 100.0


@dataclass
class TypeSpecializationConfig:
    """异构类型特化分析配置"""
    enabled: bool = True
    aggregate_by_type: bool = True
    cross_task_specialization: bool = True


@dataclass
class AlignerConfig:
    """Output Aligner 监控配置"""
    enabled: bool = True
    log_scale_stats: bool = True
    log_distribution_change: bool = False


@dataclass
class ExpertHealthDiagConfig:
    """专家健康诊断总配置"""
    enabled: bool = True
    log_interval: int = 1000
    log_on_valid: bool = True
    
    utilization: UtilizationConfig = field(default_factory=UtilizationConfig)
    output_stats: OutputStatsConfig = field(default_factory=OutputStatsConfig)
    gradient: GradientConfig = field(default_factory=GradientConfig)
    type_specialization: TypeSpecializationConfig = field(default_factory=TypeSpecializationConfig)
    aligner: AlignerConfig = field(default_factory=AlignerConfig)
    
    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "ExpertHealthDiagConfig":
        """从配置字典构建"""
        if not cfg:
            return cls(enabled=False)
        
        util_cfg = dict(cfg.get("utilization", {}))
        # Backward compatibility for old config keys.
        if "dead_threshold" in util_cfg and "dead_mean_threshold" not in util_cfg:
            util_cfg["dead_mean_threshold"] = util_cfg.pop("dead_threshold")
        if "monopoly_threshold" in util_cfg and "monopoly_top1_threshold" not in util_cfg:
            util_cfg["monopoly_top1_threshold"] = util_cfg.pop("monopoly_threshold")
        
        return cls(
            enabled=cfg.get("enabled", True),
            log_interval=cfg.get("log_interval", 1000),
            log_on_valid=cfg.get("log_on_valid", True),
            utilization=UtilizationConfig(**util_cfg),
            output_stats=OutputStatsConfig(**cfg.get("output_stats", {})),
            gradient=GradientConfig(**cfg.get("gradient", {})),
            type_specialization=TypeSpecializationConfig(**cfg.get("type_specialization", {})),
            aligner=AlignerConfig(**cfg.get("aligner", {})),
        )


# =============================================================================
# Core Diagnostic Functions
# =============================================================================

def compute_gini_coefficient(values: List[float]) -> float:
    """
    计算 Gini 系数，衡量专家负载不均衡度。
    
    Gini = 0: 完全均衡（所有专家负载相同）
    Gini = 1: 完全不均衡（单个专家承担所有负载）
    
    Args:
        values: 每个专家的使用比例（如 top1_share）
        
    Returns:
        Gini 系数 [0, 1]
    """
    if not values or len(values) < 2:
        return 0.0
    
    arr = np.array(sorted(values), dtype=np.float64)
    n = len(arr)
    total = arr.sum()
    
    if total < 1e-12:
        return 0.0
    
    # Gini coefficient formula
    cumsum = np.cumsum(arr)
    gini = (2 * np.sum((np.arange(1, n + 1) * arr)) / (n * total)) - (n + 1) / n
    return float(np.clip(gini, 0.0, 1.0))


def compute_expert_utilization(
    gate_weights: torch.Tensor,
    expert_names: List[str],
    config: UtilizationConfig,
) -> Dict[str, Any]:
    """
    计算专家利用率诊断指标。
    
    Args:
        gate_weights: Gate 权重 [B, K]
        expert_names: 专家名称列表 [K]
        config: 利用率诊断配置
        
    Returns:
        诊断指标字典
    """
    if not config.enabled:
        return {}
    
    metrics: Dict[str, Any] = {}
    K = gate_weights.size(-1)
    
    gate_w_f32 = gate_weights.float()
    
    # Top-1 选择频率
    top1_indices = torch.argmax(gate_w_f32, dim=-1)  # [B]
    top1_counts = torch.bincount(top1_indices, minlength=K).float()
    B = float(gate_weights.size(0))
    top1_share = (top1_counts / (B + 1e-8)).cpu().numpy()
    
    # Per-expert top1 share
    expert_top1_share = {
        name: float(top1_share[i]) for i, name in enumerate(expert_names)
    }
    metrics["expert_top1_share"] = expert_top1_share
    
    # 平均权重
    mean_weights = gate_w_f32.mean(dim=0).cpu().numpy()
    expert_mean_weight = {
        name: float(mean_weights[i]) for i, name in enumerate(expert_names)
    }
    metrics["expert_mean_weight"] = expert_mean_weight
    
    # Per-expert p95 权重（soft routing 使用强度）
    if gate_w_f32.size(0) > 0:
        p95_weights = torch.quantile(
            gate_w_f32,
            torch.tensor(0.95, device=gate_w_f32.device, dtype=gate_w_f32.dtype),
            dim=0,
        ).cpu().numpy()
    else:
        p95_weights = np.zeros_like(mean_weights)
    expert_weight_p95 = {
        name: float(p95_weights[i]) for i, name in enumerate(expert_names)
    }
    metrics["expert_weight_p95"] = expert_weight_p95
    
    # 死亡专家检测（soft-dead）：基于 mean_weight，可选叠加 p95 条件
    dead_by_mean = {
        name for name, w in expert_mean_weight.items()
        if w < config.dead_mean_threshold
    }
    dead_by_p95 = {
        name for name, w in expert_weight_p95.items()
        if w < config.dead_p95_threshold
    }
    if config.dead_require_p95:
        dead_experts = sorted(dead_by_mean & dead_by_p95)
    else:
        dead_experts = sorted(dead_by_mean)
    metrics["dead_experts"] = dead_experts
    metrics["dead_expert_count"] = len(dead_experts)
    metrics["dead_experts_by_mean"] = sorted(dead_by_mean)
    metrics["dead_experts_by_p95"] = sorted(dead_by_p95)
    
    # 垄断专家检测：top1_share + mean_weight 双条件
    monopoly_mean_th = (
        config.monopoly_mean_threshold
        if config.monopoly_mean_threshold is not None
        else max(0.5, 1.0 / max(K, 1) + 0.2)
    )
    monopoly_experts = [
        name
        for i, name in enumerate(expert_names)
        if top1_share[i] > config.monopoly_top1_threshold and mean_weights[i] > monopoly_mean_th
    ]
    metrics["monopoly_experts"] = monopoly_experts
    metrics["monopoly_expert_count"] = len(monopoly_experts)
    metrics["monopoly_mean_threshold_used"] = float(monopoly_mean_th)
    
    # Rank dominance: top1 极端，但 mean_weight 仍接近均匀（argmax 近似并列造成）
    uniform_mean = 1.0 / max(K, 1)
    mean_weight_std = float(np.std(mean_weights)) if K > 0 else 0.0
    near_uniform_global = mean_weight_std < config.rank_dom_std_eps
    rank_dominant_experts = []
    for i, name in enumerate(expert_names):
        if top1_share[i] <= config.rank_dom_top1_threshold:
            continue
        near_uniform_local = abs(mean_weights[i] - uniform_mean) < config.rank_dom_uniform_eps
        if near_uniform_local or near_uniform_global:
            rank_dominant_experts.append(name)
    metrics["rank_dominant_experts"] = rank_dominant_experts
    metrics["rank_dominant_count"] = len(rank_dominant_experts)
    metrics["rank_dominance_alert"] = len(rank_dominant_experts) > 0
    metrics["mean_weight_std"] = mean_weight_std
    
    # Gini 系数
    if config.compute_gini:
        gini = compute_gini_coefficient(list(top1_share))
        metrics["gini_coefficient"] = gini
        # Gini > 0.5 表示明显的负载不均衡
        metrics["load_imbalance_alert"] = gini > 0.5
    
    return metrics


def compute_expert_output_stats(
    expert_outputs: List[torch.Tensor],
    expert_names: List[str],
    config: OutputStatsConfig,
) -> Dict[str, Any]:
    """
    计算专家输出分布统计。
    
    Args:
        expert_outputs: 每个专家的输出列表，每个 [B, D]
        expert_names: 专家名称列表
        config: 输出统计配置
        
    Returns:
        诊断指标字典
    """
    if not config.enabled or not expert_outputs:
        return {}
    
    metrics: Dict[str, Any] = {}
    
    # Per-expert 统计
    if config.per_expert_stats:
        per_expert_stats = {}
        for name, out in zip(expert_names, expert_outputs):
            out_flat = out.detach().float()
            stats = {
                "mean": float(out_flat.mean().cpu().item()),
                "std": float(out_flat.std().cpu().item()),
                "sparsity": float((out_flat.abs() < 1e-5).float().mean().cpu().item()),
                "abs_max": float(out_flat.abs().max().cpu().item()),
            }
            per_expert_stats[name] = stats
        metrics["per_expert_output_stats"] = per_expert_stats
    
    # 专家间余弦相似度
    if config.cross_expert_similarity and len(expert_outputs) >= 2:
        similarities = []
        high_sim_pairs = []
        
        for i in range(len(expert_outputs)):
            for j in range(i + 1, len(expert_outputs)):
                # 展平后计算余弦相似度
                out_i = expert_outputs[i].detach().float().reshape(-1)
                out_j = expert_outputs[j].detach().float().reshape(-1)
                
                norm_i = out_i.norm() + 1e-8
                norm_j = out_j.norm() + 1e-8
                sim = float((out_i @ out_j / (norm_i * norm_j)).cpu().item())
                
                similarities.append(sim)
                
                # 高相似度告警
                if sim > config.similarity_alert_threshold:
                    high_sim_pairs.append({
                        "expert_i": expert_names[i],
                        "expert_j": expert_names[j],
                        "similarity": sim,
                    })
        
        metrics["output_similarity_mean"] = float(np.mean(similarities)) if similarities else None
        metrics["output_similarity_max"] = float(np.max(similarities)) if similarities else None
        metrics["output_similarity_min"] = float(np.min(similarities)) if similarities else None
        metrics["high_similarity_pairs"] = high_sim_pairs
        metrics["expert_collapse_alert"] = len(high_sim_pairs) > 0
    
    return metrics


def compute_expert_gradient_stats(
    expert_modules: List[Tuple[str, torch.nn.Module]],
    config: GradientConfig,
) -> Dict[str, Any]:
    """
    计算专家梯度统计。
    
    Args:
        expert_modules: (专家名称, 专家模块) 列表
        config: 梯度监控配置
        
    Returns:
        诊断指标字典
    """
    if not config.enabled:
        return {}
    
    metrics: Dict[str, Any] = {}
    per_expert_grad = {}
    vanish_experts = []
    explode_experts = []
    
    for name, module in expert_modules:
        grad_norm_sq = 0.0
        param_count = 0
        
        for param in module.parameters():
            if param.grad is not None:
                if param.grad.is_sparse:
                    # 稀疏梯度使用 coalesce 后计算
                    grad_norm_sq += param.grad.coalesce().values().norm().item() ** 2
                else:
                    grad_norm_sq += param.grad.norm().item() ** 2
                param_count += 1
        
        grad_norm = grad_norm_sq ** 0.5
        
        per_expert_grad[name] = {
            "grad_norm": grad_norm,
            "param_count": param_count,
        }
        
        # 检测梯度消失/爆炸
        if param_count > 0:
            if grad_norm < config.vanish_threshold:
                vanish_experts.append(name)
                per_expert_grad[name]["status"] = "vanishing"
            elif grad_norm > config.explode_threshold:
                explode_experts.append(name)
                per_expert_grad[name]["status"] = "exploding"
            else:
                per_expert_grad[name]["status"] = "normal"
    
    metrics["per_expert_gradient"] = per_expert_grad
    metrics["gradient_vanish_experts"] = vanish_experts
    metrics["gradient_explode_experts"] = explode_experts
    metrics["gradient_health_alert"] = len(vanish_experts) > 0 or len(explode_experts) > 0
    
    return metrics


def compute_type_specialization(
    gate_weights_per_task: Dict[str, torch.Tensor],
    expert_names_per_task: Dict[str, List[str]],
    config: TypeSpecializationConfig,
) -> Dict[str, Any]:
    """
    计算异构专家类型特化分析。
    
    Args:
        gate_weights_per_task: {task: gate_weights [B, K]}
        expert_names_per_task: {task: expert_names [K]}
        config: 类型特化分析配置
        
    Returns:
        诊断指标字典
    """
    if not config.enabled:
        return {}
    
    metrics: Dict[str, Any] = {}
    
    # 自动检测专家类型（基于名称）
    def infer_type(name: str) -> str:
        name_lower = name.lower()
        if "mlp" in name_lower:
            return "mlp"
        elif "cross" in name_lower:
            return "crossnet"
        elif "identity" in name_lower:
            return "identity"
        else:
            return "other"
    
    # 按类型聚合
    if config.aggregate_by_type:
        type_stats_per_task = {}
        
        for task, gate_w in gate_weights_per_task.items():
            expert_names = expert_names_per_task.get(task, [])
            if not expert_names:
                continue
            
            gate_w_f32 = gate_w.float()
            top1_indices = torch.argmax(gate_w_f32, dim=-1)
            K = gate_w.size(-1)
            top1_counts = torch.bincount(top1_indices, minlength=K).float()
            B = float(gate_w.size(0))
            top1_share = (top1_counts / (B + 1e-8)).cpu().numpy()
            mean_weights = gate_w_f32.mean(dim=0).cpu().numpy()
            
            # 按类型聚合
            type_top1 = {}
            type_mean_weight = {}
            
            for i, name in enumerate(expert_names):
                etype = infer_type(name)
                if etype not in type_top1:
                    type_top1[etype] = []
                    type_mean_weight[etype] = []
                type_top1[etype].append(top1_share[i])
                type_mean_weight[etype].append(mean_weights[i])
            
            task_type_stats = {}
            for etype in type_top1:
                task_type_stats[etype] = {
                    "avg_top1_share": float(np.mean(type_top1[etype])),
                    "avg_mean_weight": float(np.mean(type_mean_weight[etype])),
                    "expert_count": len(type_top1[etype]),
                }
            
            type_stats_per_task[task] = task_type_stats
        
        metrics["type_aggregation"] = type_stats_per_task
    
    # 跨任务特化分数
    if config.cross_task_specialization and len(gate_weights_per_task) >= 2:
        # 找出共享专家（出现在所有任务中的专家）
        tasks = list(gate_weights_per_task.keys())
        if len(tasks) >= 2:
            # 简化：只比较前两个有完整信息的任务
            task1, task2 = None, None
            for task in tasks:
                if task in expert_names_per_task and expert_names_per_task[task]:
                    if task1 is None:
                        task1 = task
                    elif task2 is None:
                        task2 = task
                        break
            
            # 如果找到了两个有效任务，才进行比较
            if task1 is not None and task2 is not None:
                names1 = set(expert_names_per_task.get(task1, []))
                names2 = set(expert_names_per_task.get(task2, []))
                
                # 计算 task1 和 task2 的 top1_share
                def get_top1_share_dict(gate_w, names):
                    gate_w_f32 = gate_w.float()
                    K = gate_w.size(-1)
                    top1_indices = torch.argmax(gate_w_f32, dim=-1)
                    top1_counts = torch.bincount(top1_indices, minlength=K).float()
                    B = float(gate_w.size(0))
                    top1_share = (top1_counts / (B + 1e-8)).cpu().numpy()
                    return {name: float(top1_share[i]) for i, name in enumerate(names)}
                
                share1 = get_top1_share_dict(gate_weights_per_task[task1], expert_names_per_task[task1])
                share2 = get_top1_share_dict(gate_weights_per_task[task2], expert_names_per_task[task2])
                
                # 找共享专家（名称包含 SHARE）
                shared_names = [n for n in names1 if "[SHARE]" in n]
                
                specialization_scores = {}
                for name in shared_names:
                    if name in share1 and name in share2:
                        # 特化分数 = 两个任务对该专家使用差异的绝对值
                        spec_score = abs(share1[name] - share2[name])
                        specialization_scores[name] = {
                            f"{task1}_share": share1[name],
                            f"{task2}_share": share2[name],
                            "specialization_score": spec_score,
                        }
                
                metrics["cross_task_specialization"] = {
                    "tasks_compared": [task1, task2],
                    "shared_expert_scores": specialization_scores,
                }
    
    return metrics


def compute_aligner_stats(
    aligners: Dict[str, torch.nn.Module],
    config: AlignerConfig,
) -> Dict[str, Any]:
    """
    计算 Output Aligner 统计。
    
    Args:
        aligners: {task: ExpertOutputAlign 模块}
        config: Aligner 监控配置
        
    Returns:
        诊断指标字典
    """
    if not config.enabled:
        return {}
    
    metrics: Dict[str, Any] = {}
    
    if config.log_scale_stats:
        aligner_stats = {}
        
        for task, aligner in aligners.items():
            # 检查是否有 learnable scale
            scales = getattr(aligner, "scales", None)
            if scales is not None and isinstance(scales, torch.nn.Parameter):
                scale_data = scales.detach().float().cpu()
                aligner_stats[task] = {
                    "scale_mean": float(scale_data.mean().item()),
                    "scale_std": float(scale_data.std().item()),
                    "scale_min": float(scale_data.min().item()),
                    "scale_max": float(scale_data.max().item()),
                    "scale_values": scale_data.tolist(),
                }
                
                # 检测异常：scale 过小或过大
                if scale_data.min().item() < 0.1 or scale_data.max().item() > 10:
                    aligner_stats[task]["scale_alert"] = True
        
        metrics["aligner_scale_stats"] = aligner_stats
    
    return metrics


# =============================================================================
# Main Diagnostics Class
# =============================================================================

class ExpertHealthDiagnostics:
    """
    专家健康诊断管理器。
    
    负责：
    1. 收集各类诊断数据
    2. 计算诊断指标
    3. 写入诊断日志文件
    
    Usage:
        diag = ExpertHealthDiagnostics(config, run_dir)
        
        # 在训练循环中
        diag.collect_gate_weights(task, gate_weights)
        diag.collect_expert_outputs(task, expert_outputs, expert_names)
        
        # 每 N steps 或验证时
        if should_log:
            diag.compute_and_log(step, epoch, phase="train")
            diag.reset()
    """
    
    def __init__(
        self,
        config: ExpertHealthDiagConfig,
        run_dir: Path,
    ):
        self.config = config
        self.run_dir = Path(run_dir)
        self.log_path = self.run_dir / "expert_health_diag.jsonl"
        
        # 收集缓冲区
        self._gate_weights: Dict[str, List[torch.Tensor]] = {}
        self._expert_outputs: Dict[str, List[torch.Tensor]] = {}  # 最后一个 batch 的输出
        self._expert_names: Dict[str, List[str]] = {}
        self._expert_modules: List[Tuple[str, torch.nn.Module]] = []
        self._aligners: Dict[str, torch.nn.Module] = {}
        
        # 初始化日志文件
        if config.enabled:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"[ExpertHealthDiag] Initialized, log path: {self.log_path}")
    
    def collect_gate_weights(self, task: str, gate_weights: torch.Tensor) -> None:
        """收集 gate 权重（累积多个 batch）"""
        if not self.config.enabled:
            return
        if task not in self._gate_weights:
            self._gate_weights[task] = []
        # 只保留必要数据，避免内存爆炸
        self._gate_weights[task].append(gate_weights.detach().cpu())
    
    def collect_expert_outputs(
        self,
        task: str,
        expert_outputs: List[torch.Tensor],
        expert_names: List[str],
    ) -> None:
        """收集专家输出（只保留最后一个 batch 的统计）"""
        if not self.config.enabled or not self.config.output_stats.enabled:
            return
        # 直接覆盖，只保留最新 batch（输出分布统计不需要累积）
        self._expert_outputs[task] = [out.detach().cpu() for out in expert_outputs]
        self._expert_names[task] = expert_names
    
    def set_expert_modules(self, expert_modules: List[Tuple[str, torch.nn.Module]]) -> None:
        """设置专家模块引用（用于梯度监控）"""
        self._expert_modules = expert_modules
    
    def set_aligners(self, aligners: Dict[str, torch.nn.Module]) -> None:
        """设置 aligner 模块引用"""
        self._aligners = aligners
    
    def set_expert_names(self, task: str, names: List[str]) -> None:
        """设置专家名称"""
        self._expert_names[task] = names
    
    def compute_and_log(
        self,
        step: int,
        epoch: int,
        phase: str = "train",
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        计算所有诊断指标并写入日志文件。
        
        Args:
            step: 当前全局 step
            epoch: 当前 epoch
            phase: "train" 或 "valid"
            extra_meta: 额外的元数据
            
        Returns:
            所有诊断指标的字典
        """
        if not self.config.enabled:
            return {}
        
        all_metrics: Dict[str, Any] = {
            "step": step,
            "epoch": epoch,
            "phase": phase,
        }
        if extra_meta:
            all_metrics.update(extra_meta)
        
        # 1. 专家利用率诊断
        if self.config.utilization.enabled:
            for task, gate_w_list in self._gate_weights.items():
                if not gate_w_list:
                    continue
                # 合并所有 batch
                all_gate_w = torch.cat(gate_w_list, dim=0)
                expert_names = self._expert_names.get(task, [f"expert_{i}" for i in range(all_gate_w.size(-1))])
                
                util_metrics = compute_expert_utilization(
                    all_gate_w, expert_names, self.config.utilization
                )
                all_metrics[f"utilization_{task}"] = util_metrics
        
        # 2. 专家输出分布监控
        if self.config.output_stats.enabled:
            for task, outputs in self._expert_outputs.items():
                if not outputs:
                    continue
                expert_names = self._expert_names.get(task, [f"expert_{i}" for i in range(len(outputs))])
                
                output_metrics = compute_expert_output_stats(
                    outputs, expert_names, self.config.output_stats
                )
                all_metrics[f"output_stats_{task}"] = output_metrics
        
        # 3. 专家梯度监控
        if self.config.gradient.enabled and self._expert_modules:
            grad_metrics = compute_expert_gradient_stats(
                self._expert_modules, self.config.gradient
            )
            all_metrics["gradient"] = grad_metrics
        
        # 4. 异构类型特化分析
        if self.config.type_specialization.enabled:
            gate_weights_dict = {}
            for task, gate_w_list in self._gate_weights.items():
                if gate_w_list:
                    gate_weights_dict[task] = torch.cat(gate_w_list, dim=0)
            
            if gate_weights_dict:
                type_metrics = compute_type_specialization(
                    gate_weights_dict, self._expert_names, self.config.type_specialization
                )
                all_metrics["type_specialization"] = type_metrics
        
        # 5. Aligner 监控
        if self.config.aligner.enabled and self._aligners:
            aligner_metrics = compute_aligner_stats(
                self._aligners, self.config.aligner
            )
            all_metrics["aligner"] = aligner_metrics
        
        # 生成汇总告警
        alerts = self._generate_alerts(all_metrics)
        all_metrics["alerts"] = alerts
        
        # 写入日志文件
        self._write_log(all_metrics)
        
        return all_metrics
    
    def _generate_alerts(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """汇总生成告警信息"""
        alerts = {
            "has_alert": False,
            "summary": [],
        }
        
        # 检查各类告警
        for key, value in metrics.items():
            if not isinstance(value, dict):
                continue
            
            # 死亡专家
            if "dead_expert_count" in value and value["dead_expert_count"] > 0:
                alerts["has_alert"] = True
                alerts["summary"].append(
                    f"{key}: {value['dead_expert_count']} dead experts "
                    f"(soft-dead by mean_weight/p95, {value.get('dead_experts', [])})"
                )
            
            # 垄断专家
            if "monopoly_expert_count" in value and value["monopoly_expert_count"] > 0:
                alerts["has_alert"] = True
                alerts["summary"].append(
                    f"{key}: {value['monopoly_expert_count']} monopoly experts "
                    f"(by top1_share+mean_weight, {value.get('monopoly_experts', [])})"
                )
            
            # Rank dominance（解释 top1 极端但权重仍近均匀）
            if "rank_dominant_count" in value and value["rank_dominant_count"] > 0:
                alerts["has_alert"] = True
                alerts["summary"].append(
                    f"{key}: {value['rank_dominant_count']} rank-dominant experts "
                    f"({value.get('rank_dominant_experts', [])}); top1 extreme but mean_weight near-uniform "
                    f"(likely argmax tie/near-tie)"
                )
            
            # 负载不均衡
            if value.get("load_imbalance_alert"):
                alerts["has_alert"] = True
                gini = value.get("gini_coefficient", "N/A")
                alerts["summary"].append(f"{key}: high load imbalance (Gini={gini:.3f})")
            
            # 专家坍缩
            if value.get("expert_collapse_alert"):
                alerts["has_alert"] = True
                pairs = value.get("high_similarity_pairs", [])
                alerts["summary"].append(f"{key}: expert collapse detected ({len(pairs)} high-sim pairs)")
            
            # 梯度异常
            if value.get("gradient_health_alert"):
                alerts["has_alert"] = True
                vanish = value.get("gradient_vanish_experts", [])
                explode = value.get("gradient_explode_experts", [])
                alerts["summary"].append(
                    f"gradient: {len(vanish)} vanishing, {len(explode)} exploding"
                )
        
        return alerts
    
    def _write_log(self, metrics: Dict[str, Any]) -> None:
        """写入 JSONL 日志文件"""
        try:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"[ExpertHealthDiag] Failed to write log: {e}")
    
    def reset(self) -> None:
        """重置收集缓冲区"""
        self._gate_weights.clear()
        self._expert_outputs.clear()
        # 保留 expert_names, expert_modules, aligners（这些是静态引用）
    
    def should_log(self, step: int, phase: str = "train") -> bool:
        """判断当前 step 是否应该记录诊断"""
        if not self.config.enabled:
            return False
        
        if phase == "valid":
            return self.config.log_on_valid
        
        return step > 0 and step % self.config.log_interval == 0


__all__ = [
    "ExpertHealthDiagConfig",
    "ExpertHealthDiagnostics",
    "compute_gini_coefficient",
    "compute_expert_utilization",
    "compute_expert_output_stats",
    "compute_expert_gradient_stats",
    "compute_type_specialization",
    "compute_aligner_stats",
]
