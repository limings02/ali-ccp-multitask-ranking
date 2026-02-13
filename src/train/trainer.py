from __future__ import annotations

import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import amp as torch_amp
import yaml

from src.core.checkpoint import load_checkpoint, save_checkpoint
from src.data.dataloader import make_dataloader
from src.loss.bce import MultiTaskBCELoss
from src.models.build import build_model
from src.eval.run_eval import run_eval
from src.train.loops import train_one_epoch, validate
from src.train.optim import build_optimizer_bundle, build_lr_scheduler_bundle, LRSchedulerBundle
from src.train.best_selector import BestSelector
from src.utils.feature_meta import build_model_feature_meta
from src.utils.metrics_schema import apply_esmm_schema


def _setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(f"trainer_{log_path}")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_enabled_heads(model_cfg: Dict[str, Any]) -> list[str]:
    heads = model_cfg.get("enabled_heads") or model_cfg.get("tasks")
    if not heads:
        heads = ["ctr", "cvr"]
    return sorted([str(h).lower() for h in heads])


def get_exp_tags(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unified experiment identifier used in every metrics.jsonl row.
    """
    model_cfg = cfg.get("model", {})
    model_name = model_cfg.get("name", "unknown_model")
    enabled = model_cfg.get("enabled_heads") or model_cfg.get("tasks") or ["ctr", "cvr"]
    enabled = sorted([str(h).lower() for h in enabled])
    exp_name = f"{model_name}__heads={'-'.join(enabled)}"
    seed = cfg.get("data", {}).get("seed", cfg.get("runtime", {}).get("seed"))
    return {"model_name": model_name, "enabled_heads": enabled, "exp_name": exp_name, "seed": seed}


class Trainer:
    def __init__(self, cfg: Dict[str, Any], config_path: Optional[str | Path] = None):
        self.cfg = cfg
        self.config_path = Path(config_path).resolve() if config_path else None

        self.use_esmm = bool(cfg.get("use_esmm", False))
        sampling_cfg = cfg.get("sampling", {}) or {}
        self.negative_sampling = str(sampling_cfg.get("negative_sampling", "keep_prob")).lower()

        tags = get_exp_tags(cfg)
        self.enabled_heads = tags["enabled_heads"]
        self.model_name = tags["model_name"]
        self.exp_name = tags["exp_name"]
        self.seed_value = tags["seed"]
        self.metrics_meta = {**tags, "config_path": str(self.config_path) if self.config_path else None}
        self.metrics_meta["mode"] = "esmm" if self.use_esmm else "non-esmm"
        exp_name = cfg.get("experiment", {}).get("name", "exp")
        resume_path = cfg.get("runtime", {}).get("resume_path")
        if resume_path:
            # Resume: reuse the run directory of the checkpoint for appending logs
            self.run_dir = Path(resume_path).resolve().parent
        else:
            # Fresh training: create new run directory with timestamp
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir = Path("runs") / f"{exp_name}_{ts}"
        
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Save config snapshot
        (self.run_dir / "config.yaml").write_text(
            yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
        )

        self.logger = _setup_logger(self.run_dir / "train.log")
        seed = int(cfg.get("runtime", {}).get("seed", 2026))
        _set_seed(seed)

        self.device = torch.device(cfg.get("runtime", {}).get("device", "cpu"))
        if self.device.type == "cuda" and not torch.cuda.is_available():
            # Keep training runnable on machines without CUDA; AMP will be disabled automatically.
            self.device = torch.device("cpu")
            logging.getLogger(__name__).warning("CUDA requested but not available; falling back to CPU and disabling AMP.")
        runtime_cfg = cfg.get("runtime", {})
        self.amp_enabled = bool(runtime_cfg.get("amp", False)) and self.device.type == "cuda"
        self.amp_dtype = torch.float16
        self.amp_device_type = self.device.type
        # GradScaler keeps fp32 master weights while applying scaled grads; disabled on CPU automatically.
        self.scaler = torch_amp.GradScaler(enabled=self.amp_enabled, device="cuda")
        self.debug_logit_every = int(runtime_cfg.get("debug_logit_every", 0) or 0)
        self.debug_linear_stats = bool(runtime_cfg.get("debug_linear_stats", False))
        self.debug_fm_stats = bool(runtime_cfg.get("debug_fm_stats", False))
        self.debug_dataloader_assert = bool(runtime_cfg.get("debug_dataloader_assert", False))

        # feature_meta for dataloader/model
        metadata_path = Path(cfg["data"]["metadata_path"])
        embedding_cfg = cfg.get("embedding", {})
        self.feature_meta = build_model_feature_meta(metadata_path, embedding_cfg)

        loss_cfg = cfg.get("loss", {})
        static_pos_weight_cfg = loss_cfg.get("static_pos_weight", {}) or {}
        pos_weight_clip_cfg = loss_cfg.get("pos_weight_clip", {}) or {}

        # Resolve negative sampling keep prob (honor config unless explicitly disabled)
        neg_keep_prob_cfg = float(cfg.get("data", {}).get("neg_keep_prob_train", 1.0))
        if self.negative_sampling in {"none", "off", "disable", "disabled"}:
            self.neg_keep_prob_train = 1.0
        else:
            self.neg_keep_prob_train = neg_keep_prob_cfg

        self.logger.info(
            "mode=%s pos_weight_dynamic=%s pos_weight_ctr=%s pos_weight_ctcvr=%s neg_sampling=%s neg_keep_prob_train=%.3f",
            "esmm" if self.use_esmm else "non-esmm",
            bool(loss_cfg.get("pos_weight_dynamic", True)),
            static_pos_weight_cfg.get("ctr"),
            static_pos_weight_cfg.get("ctcvr"),
            self.negative_sampling,
            self.neg_keep_prob_train,
        )
        data_cfg = cfg.get("data", {})
        default_workers = max(0, min(6, os.cpu_count() or 1))
        train_workers = int(data_cfg.get("num_workers", default_workers))
        valid_workers = int(data_cfg.get("num_workers_valid", max(1, train_workers // 2)))
        data_format = str(data_cfg.get("format", "vectorized")).lower()
        processed_root = str(data_cfg.get("processed_dir", "data/processed"))
        vectorized_root = str(data_cfg.get("vectorized_dir", "data/vectorized"))
        prefetch_cfg = data_cfg.get("prefetch_factor")
        train_prefetch = int(prefetch_cfg) if prefetch_cfg is not None else (4 if train_workers > 0 else None)
        valid_prefetch = int(prefetch_cfg) if prefetch_cfg is not None else (4 if valid_workers > 0 else None)
        worker_cpu_threads = int(data_cfg.get("worker_cpu_threads", 1))
        persistent_cfg = data_cfg.get("persistent_workers")
        train_persistent = bool(persistent_cfg) if persistent_cfg is not None else (train_workers > 0)
        valid_persistent = bool(persistent_cfg) if persistent_cfg is not None else (valid_workers > 0)

        self.train_loader = make_dataloader(
            split="train",
            batch_size=int(cfg["data"]["batch_size"]),
            num_workers=train_workers,
            shuffle=False,
            drop_last=bool(cfg["data"].get("drop_last", False)),
            pin_memory=bool(cfg["data"].get("pin_memory", True)),
            persistent_workers=train_persistent,
            seed=cfg["data"].get("seed"),
            feature_meta=self.feature_meta,
            debug=bool(cfg["data"].get("debug", False) or self.debug_dataloader_assert),
            neg_keep_prob_train=self.neg_keep_prob_train,
            prefetch_factor=train_prefetch,
            worker_cpu_threads=worker_cpu_threads,
            # ===== Resume state for strict checkpoint recovery =====
            resume_state=getattr(self, "data_resume_state", None),
            enable_data_cursor=True,
            include_entity_id=False,
            data_format=data_format,
            processed_root=processed_root,
            vectorized_root=vectorized_root,
        )
        # Use separate num_workers for validation to avoid memory issues
        # Default to half of train workers or max 2 to prevent OOM with persistent_workers
        self.valid_loader = make_dataloader(
            split="valid",
            batch_size=int(cfg["data"]["batch_size"]),
            num_workers=valid_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=bool(cfg["data"].get("pin_memory", True)),
            persistent_workers=valid_persistent,
            seed=cfg["data"].get("seed"),
            feature_meta=self.feature_meta,
            debug=bool(cfg["data"].get("debug", False) or self.debug_dataloader_assert),
            prefetch_factor=valid_prefetch,
            worker_cpu_threads=worker_cpu_threads,
            include_entity_id=False,
            data_format=data_format,
            processed_root=processed_root,
            vectorized_root=vectorized_root,
        )

        self.model = build_model(cfg).to(self.device)
        # Enable debug hooks on backbone/head if supported.
        if hasattr(self.model, "backbone"):
            setattr(self.model.backbone, "debug_linear", self.debug_linear_stats or self.debug_logit_every > 0)
            setattr(self.model.backbone, "debug_fm", self.debug_fm_stats or self.debug_logit_every > 0)
        if hasattr(self.model, "debug_logit"):
            self.model.debug_logit = self.debug_logit_every > 0

        opt_cfg = cfg.get("optim", {})
        self.optim_bundle = build_optimizer_bundle(cfg, self.model, scaler=self.scaler)

        # ===== Build LR Scheduler (warmup + decay, step-based) =====
        runtime_cfg = cfg.get("runtime", {})
        epochs = int(runtime_cfg.get("epochs", 1))
        max_train_steps = runtime_cfg.get("max_train_steps")
        self.lr_scheduler_bundle = build_lr_scheduler_bundle(
            cfg,
            self.optim_bundle,
            train_loader_len=len(self.train_loader),
            epochs=epochs,
            max_steps=max_train_steps,
            logger_obj=self.logger,
        )

        # ===== Load aux_focal configuration (方案1: ESMM 主链路 BCE + CTCVR Aux-Focal) =====
        aux_focal_cfg = loss_cfg.get("aux_focal", {})
        
        self.loss_fn = MultiTaskBCELoss(
            w_ctr=float(loss_cfg.get("w_ctr", 1.0)),
            w_cvr=float(loss_cfg.get("w_cvr", 1.0)),
            eps=float(loss_cfg.get("eps", 1e-6)),
            use_esmm=self.use_esmm,
            esmm_version=str(cfg.get("esmm", {}).get("version", "v2")),  # "v2" (standard) or "legacy"
            lambda_ctr=float(cfg.get("esmm", {}).get("lambda_ctr", 1.0)),
            lambda_ctcvr=float(cfg.get("esmm", {}).get("lambda_ctcvr", 1.0)),
            lambda_cvr_aux=float(cfg.get("esmm", {}).get("lambda_cvr_aux", 0.0)),  # CVR aux loss
            esmm_eps=float(cfg.get("esmm", {}).get("eps", 1e-8)),
            enabled_heads=self.enabled_heads,
            pos_weight_dynamic=bool(loss_cfg.get("pos_weight_dynamic", True)),
            static_pos_weight_ctr=(
                float(static_pos_weight_cfg["ctr"]) if "ctr" in static_pos_weight_cfg else None
            ),
            static_pos_weight_ctcvr=(
                float(static_pos_weight_cfg["ctcvr"]) if "ctcvr" in static_pos_weight_cfg else None
            ),
            pos_weight_clip_ctr=(
                float(pos_weight_clip_cfg["ctr"]) if "ctr" in pos_weight_clip_cfg else None
            ),
            pos_weight_clip_ctcvr=(
                float(pos_weight_clip_cfg["ctcvr"]) if "ctcvr" in pos_weight_clip_cfg else None
            ),
            # Aux Focal parameters (defaults to disabled for backward compatibility)
            aux_focal_enabled=bool(aux_focal_cfg.get("enabled", False)),
            aux_focal_warmup_steps=int(aux_focal_cfg.get("warmup_steps", 2000)),
            aux_focal_target_head=str(aux_focal_cfg.get("target_head", "ctcvr")),
            aux_focal_lambda=float(aux_focal_cfg.get("lambda", 0.1)),
            aux_focal_gamma=float(aux_focal_cfg.get("gamma", 1.0)),
            aux_focal_use_alpha=bool(aux_focal_cfg.get("use_alpha", False)),
            aux_focal_alpha=float(aux_focal_cfg.get("alpha", 0.25)),
            aux_focal_detach_p_for_weight=bool(aux_focal_cfg.get("detach_p_for_weight", True)),
            aux_focal_compute_fp32=bool(aux_focal_cfg.get("compute_fp32", True)),
            aux_focal_log_components=bool(aux_focal_cfg.get("log_components", True)),
            # 负采样权重补偿
            neg_sample_weight_correction=bool(loss_cfg.get("neg_sample_weight_correction", False)),
            neg_keep_prob_train=self.neg_keep_prob_train,
            # ESMM Residual Head 配置
            residual_enabled=bool(cfg.get("esmm", {}).get("residual", {}).get("enabled", False)),
            residual_res_reg_weight=float(cfg.get("esmm", {}).get("residual", {}).get("res_reg_weight", 5e-4)),
            residual_anchor_enabled=bool(cfg.get("esmm", {}).get("residual", {}).get("anchor", {}).get("enabled", False)),
            residual_anchor_weight=float(cfg.get("esmm", {}).get("residual", {}).get("anchor", {}).get("weight", 0.02)),
            residual_anchor_eps=float(cfg.get("esmm", {}).get("residual", {}).get("anchor", {}).get("eps", 1e-6)),
            global_step=0,  # Will be updated in training loop
        )

        # Log ESMM configuration
        if self.use_esmm:
            esmm_cfg = cfg.get("esmm", {})
            self.logger.info(
                "ESMM config: version=%s lambda_ctr=%.2f lambda_ctcvr=%.2f lambda_cvr_aux=%.2f",
                esmm_cfg.get("version", "v2"),
                esmm_cfg.get("lambda_ctr", 1.0),
                esmm_cfg.get("lambda_ctcvr", 1.0),
                esmm_cfg.get("lambda_cvr_aux", 0.0),
            )
            
            # ====================================================================
            # 打印 pos_weight 配置（区分动态/静态模式）
            # ====================================================================
            pos_weight_dynamic = bool(loss_cfg.get("pos_weight_dynamic", True))
            ctr_clip = float(pos_weight_clip_cfg.get("ctr")) if "ctr" in pos_weight_clip_cfg else None
            ctcvr_clip = float(pos_weight_clip_cfg.get("ctcvr")) if "ctcvr" in pos_weight_clip_cfg else None
            
            if pos_weight_dynamic:
                # 动态模式：训练时按 batch 实时计算 neg/pos，这里只显示 clip 配置
                self.logger.info(
                    "[pos_weight] Mode: DYNAMIC (computed per-batch from neg/pos ratio)"
                )
                self.logger.info(
                    "[pos_weight] Clip limits - CTR: %s | CTCVR: %s",
                    f"{ctr_clip:.1f}" if ctr_clip is not None else "None (no limit)",
                    f"{ctcvr_clip:.1f}" if ctcvr_clip is not None else "None (no limit)"
                )
                self.logger.info(
                    "[pos_weight] 💡 Actual dynamic values are logged in training metrics: "
                    "pos_weight_ctr_raw, pos_weight_ctr_effective, pos_weight_ctcvr_raw, pos_weight_ctcvr_effective"
                )
            else:
                # 静态模式：使用配置的固定值
                from src.loss.bce import _effective_pos_weight
                ctr_raw = float(static_pos_weight_cfg.get("ctr", 1.0))
                ctcvr_raw = float(static_pos_weight_cfg.get("ctcvr", 1.0))
                ctr_effective, ctr_clipped = _effective_pos_weight(ctr_raw, ctr_clip)
                ctcvr_effective, ctcvr_clipped = _effective_pos_weight(ctcvr_raw, ctcvr_clip)
                
                self.logger.info(
                    "[pos_weight] Mode: STATIC (fixed values from config)"
                )
                self.logger.info(
                    "[pos_weight] CTR: raw=%.2f clip=%s effective=%.2f clipped=%s",
                    ctr_raw, ctr_clip, ctr_effective, ctr_clipped
                )
                self.logger.info(
                    "[pos_weight] CTCVR: raw=%.2f clip=%s effective=%.2f clipped=%s",
                    ctcvr_raw, ctcvr_clip, ctcvr_effective, ctcvr_clipped
                )
            
            # Log Aux Focal configuration if enabled
            if aux_focal_cfg.get("enabled", False):
                self.logger.info(
                    "Aux Focal config: enabled=true warmup_steps=%d target_head=%s lambda=%.3f gamma=%.1f use_alpha=%s alpha=%.3f",
                    aux_focal_cfg.get("warmup_steps", 2000),
                    aux_focal_cfg.get("target_head", "ctcvr"),
                    aux_focal_cfg.get("lambda", 0.1),
                    aux_focal_cfg.get("gamma", 1.0),
                    aux_focal_cfg.get("use_alpha", False),
                    aux_focal_cfg.get("alpha", 0.25),
                )

            # Log Residual Head configuration if enabled
            residual_cfg = cfg.get("esmm", {}).get("residual", {})
            if residual_cfg.get("enabled", False):
                anchor_cfg = residual_cfg.get("anchor", {})
                self.logger.info(
                    "Residual Head config: enabled=true mlp_dims=%s dropout=%.2f "
                    "use_layernorm=%s alpha=%.2f stop_grad_input=%s "
                    "res_reg_weight=%.2e anchor_enabled=%s anchor_weight=%.3f",
                    residual_cfg.get("mlp_dims", [64]),
                    residual_cfg.get("dropout", 0.1),
                    residual_cfg.get("use_layernorm", True),
                    residual_cfg.get("alpha", 1.0),
                    residual_cfg.get("stop_grad_input", False),
                    residual_cfg.get("res_reg_weight", 5e-4),
                    anchor_cfg.get("enabled", False),
                    anchor_cfg.get("weight", 0.02),
                )

        self.best_metric: float = float("-inf")  # tracks best full AUC (kept for legacy compatibility)
        self.global_step = 0

        # Health metrics logging (disabled by default for backward compatibility)
        self.log_health_metrics = bool(runtime_cfg.get("log_health_metrics", False))
        if self.log_health_metrics:
            self.logger.info("Health metrics logging enabled (log_health_metrics=true)")

        # ===== Expert Health Diagnostics (专家健康诊断) =====
        from src.utils.expert_health_diag import ExpertHealthDiagConfig, ExpertHealthDiagnostics
        expert_health_cfg = runtime_cfg.get("expert_health_diag", {})
        self.expert_health_diag_config = ExpertHealthDiagConfig.from_dict(expert_health_cfg)
        self.expert_health_diag: Optional[ExpertHealthDiagnostics] = None
        
        if self.expert_health_diag_config.enabled:
            self.expert_health_diag = ExpertHealthDiagnostics(
                config=self.expert_health_diag_config,
                run_dir=self.run_dir,
            )
            # 设置模型内部数据引用（如果模型支持）
            # 支持 PLE: model.ple.get_expert_health_data()
            # 支持 MMoE: model.get_expert_health_data() (when model is MMoE directly)
            health_data = None
            if hasattr(self.model, "ple") and hasattr(self.model.ple, "get_expert_health_data"):
                health_data = self.model.ple.get_expert_health_data()
            elif hasattr(self.model, "get_expert_health_data"):
                health_data = self.model.get_expert_health_data()
            
            if health_data is not None:
                self.expert_health_diag.set_expert_modules(health_data.get("expert_modules", []))
                self.expert_health_diag.set_aligners(health_data.get("aligners", {}))
                for task, names in health_data.get("expert_names", {}).items():
                    self.expert_health_diag.set_expert_names(task, names)
                self.logger.info(
                    "[ExpertHealthDiag] Initialized: log_interval=%d, log_on_valid=%s",
                    self.expert_health_diag_config.log_interval,
                    self.expert_health_diag_config.log_on_valid,
                )
            else:
                self.logger.warning(
                    "[ExpertHealthDiag] Enabled but model does not support expert health data collection"
                )

        # ===== Best model selection strategy (gate or legacy auc_primary) =====
        best_selection_cfg = runtime_cfg.get("best_selection", {})
        strategy = best_selection_cfg.get("strategy", "auc_primary")
        self.best_selector = BestSelector(
            strategy=strategy,
            primary_key=best_selection_cfg.get("primary_key", "auc_ctcvr"),
            aux_keys=best_selection_cfg.get("aux_keys", ["auc_ctr", "auc_cvr"]),
            use_primary_ma=best_selection_cfg.get("use_primary_ma", False),
            ma_window=best_selection_cfg.get("ma_window", 5),
            tol_primary=best_selection_cfg.get("tol_primary", 0.0),
            tol_aux=best_selection_cfg.get("tol_aux", {}),
            confirm_times=best_selection_cfg.get("confirm_times", 1),
            cooldown_evals=best_selection_cfg.get("cooldown_evals", 0),
            log_decision=best_selection_cfg.get("log_decision", True),
            logger=self.logger,
        )
        self.logger.info(f"BestSelector initialized: strategy={strategy}")

        resume_path = cfg.get("runtime", {}).get("resume_path")
        if resume_path:
            info = load_checkpoint(resume_path, self.model, self.optim_bundle, map_location=self.device, strict=False)
            self.best_metric = info.get("best_metric", self.best_metric)
            self.global_step = info.get("step", 0) or 0
            # Restore lr_scheduler state if present
            extra = info.get("extra", {}) or {}
            if "lr_scheduler" in extra and self.lr_scheduler_bundle.enabled:
                self.lr_scheduler_bundle.load_state_dict(extra.get("lr_scheduler"))
                self.logger.info("Restored lr_scheduler state from checkpoint")
            # Restore best_selector state if present
            if "best_selector" in extra:
                self.best_selector.load_state(extra["best_selector"])
                self.logger.info("Restored best_selector state from checkpoint")
            
            # ===== Restore dataset resume state for strict checkpoint recovery =====
            self.data_resume_state = extra.get("data_state")
            if self.data_resume_state:
                self.logger.info(f"Restored data_state for {len(self.data_resume_state)} workers")
            
            self.logger.info(f"Resumed from {resume_path}, step={self.global_step}, best_metric={self.best_metric}")
        else:
            self.data_resume_state = None

    def _write_metrics(self, metrics: Dict[str, Any]) -> None:
        payload = {
            **self.metrics_meta,
            "amp_enabled": self.amp_enabled,
            "amp_dtype": "fp16" if self.amp_enabled else "none",
            **apply_esmm_schema(metrics, self.use_esmm),
        }
        if self.amp_enabled and hasattr(self, "scaler") and self.scaler is not None:
            payload["scaler_scale"] = float(self.scaler.get_scale())
        with (self.run_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

    def run(self) -> None:
        runtime = self.cfg.get("runtime", {})
        max_train_steps = runtime.get("max_train_steps")
        if max_train_steps is None:
            raise ValueError("max_train_steps must be specified in config")
        
        epochs = int(runtime.get("epochs", 1))
        log_every = int(runtime.get("log_every", 50))
        max_valid_steps = runtime.get("max_valid_steps")
        grad_clip_norm = runtime.get("grad_clip_norm")
        grad_diag_enabled = bool(runtime.get("grad_diag_enabled", False))
        eval_every_steps = log_every
        save_last = bool(runtime.get("save_last", True))
        
        # 计算每个 epoch 的期望步数
        epoch_steps = max_train_steps // epochs
        
        # 根据 global_step 计算起始 epoch（从 0 开始便于计算）
        start_epoch = self.global_step // epoch_steps
        
        self.logger.info(f"Resuming training from global_step={self.global_step}, start_epoch={start_epoch + 1}/{epochs}")
        
        for epoch in range(start_epoch, epochs):
            # 计算当前 epoch 的起始和结束步数
            epoch_start_step = epoch * epoch_steps
            epoch_end_step = (epoch + 1) * epoch_steps if epoch < epochs - 1 else max_train_steps
            epoch_num = epoch + 1  # 用于日志，epoch 从 1 计数
            
            self.logger.info(f"Epoch {epoch_num}/{epochs}: global steps {epoch_start_step}-{epoch_end_step}")
            
            def _validate_full():
                return validate(
                    self.model,
                    self.valid_loader,
                    self.loss_fn,
                    self.device,
                    self.logger,
                    epoch=epoch_num,
                    max_steps=max_valid_steps,
                    log_every=max(1, log_every * 4),
                    amp_enabled=self.amp_enabled,
                    amp_dtype=self.amp_dtype,
                    amp_device_type=self.amp_device_type,
                    calc_auc=True,
                    log_health_metrics=self.log_health_metrics,
                    expert_health_diag=self.expert_health_diag,
                )

            def _on_eval(val_metrics: Dict[str, Any], g_step: int, ep: int) -> None:
                record = {"epoch": ep, "split": "valid", "global_step": g_step, **val_metrics}
                self._write_metrics(record)
                
                # Use BestSelector to determine whether to update best checkpoint
                should_update, decision_info = self.best_selector.should_update_best(val_metrics, g_step)
                
                # Log decision info to metrics.jsonl for analysis
                decision_record = {
                    "epoch": ep,
                    "split": "valid_decision",
                    "global_step": g_step,
                    **decision_info,
                }
                self._write_metrics(decision_record)
                
                if should_update:
                    # Update legacy best_metric for backward compatibility (auc_primary)
                    full_auc = val_metrics.get("auc_primary")
                    if full_auc is not None:
                        self.best_metric = full_auc
                    
                    eval_extra = {"epoch": ep, "kind": "full"}
                    if self.lr_scheduler_bundle.enabled:
                        eval_extra["lr_scheduler"] = self.lr_scheduler_bundle.state_dict()
                    # Save best_selector state for resume support
                    eval_extra["best_selector"] = self.best_selector.get_state()
                    
                    # ===== Include data_state for strict resume =====
                    if hasattr(self, "last_data_state") and self.last_data_state:
                        eval_extra["data_state"] = self.last_data_state
                    
                    save_checkpoint(
                        self.run_dir / "ckpt_best.pt",
                        self.model,
                        self.optim_bundle,
                        cfg=self.cfg,
                        step=g_step,
                        best_metric=self.best_metric,
                        extra=eval_extra,
                    )
                    # Best selector already logged detailed info, so keep this simple
                    self.logger.info(f"[OK] Saved best checkpoint at step {g_step}")

            # Resolve gradient diagnostics parameters
            grad_diag_every = runtime.get("grad_diag_every")  # None means use log_every
            grad_diag_min_tasks = int(runtime.get("grad_diag_min_tasks", 2))
            grad_samples_target = int(runtime.get("grad_samples_target", 1000))
            conflict_ema_alpha = float(runtime.get("conflict_ema_alpha", 0.1))

            train_metrics = train_one_epoch(
                self.model,
                self.train_loader,
                self.optim_bundle,
                self.loss_fn,
                self.device,
                self.logger,
                epoch=epoch_num,
                max_steps=epoch_end_step,  # 限制该 epoch 的最大步数
                grad_clip_norm=grad_clip_norm,
                log_every=log_every,
                global_step=self.global_step,
                grad_diag_enabled=grad_diag_enabled,
                grad_diag_every=grad_diag_every,
                grad_diag_min_tasks=grad_diag_min_tasks,
                eval_every_steps=eval_every_steps,
                validate_fn=_validate_full,
                eval_callback=_on_eval,
                amp_enabled=self.amp_enabled,
                amp_dtype=self.amp_dtype,
                scaler=self.scaler,
                amp_device_type=self.amp_device_type,
                debug_logit_every=self.debug_logit_every,
                lr_scheduler_bundle=self.lr_scheduler_bundle,
                cfg=self.cfg,
                expert_health_diag=self.expert_health_diag,
                grad_samples_target=grad_samples_target,
                conflict_ema_alpha=conflict_ema_alpha,
            )
            self.global_step += train_metrics.get("steps", 0)
            train_record = {"epoch": epoch_num, "split": "train", **train_metrics}
            self._write_metrics(train_record)
            
            # ===== Collect data state from train metrics for resume support =====
            if "data_state" in train_metrics:
                self.last_data_state = train_metrics["data_state"]
            else:
                self.last_data_state = None

            valid_metrics = validate(
                self.model,
                self.valid_loader,
                self.loss_fn,
                self.device,
                self.logger,
                epoch=epoch_num,
                max_steps=max_valid_steps,
                log_every=log_every * 4,
                amp_enabled=self.amp_enabled,
                amp_dtype=self.amp_dtype,
                calc_auc=True,
                log_health_metrics=self.log_health_metrics,
                expert_health_diag=self.expert_health_diag,
            )
            valid_record = {"epoch": epoch_num, "split": "valid", **valid_metrics}
            self._write_metrics(valid_record)

            # checkpointing
            # Prepare extra dict with lr_scheduler state
            ckpt_extra = {"epoch": epoch_num}
            if self.lr_scheduler_bundle.enabled:
                ckpt_extra["lr_scheduler"] = self.lr_scheduler_bundle.state_dict()
            
            # ===== Include data_state for strict resume =====
            if hasattr(self, "last_data_state") and self.last_data_state:
                ckpt_extra["data_state"] = self.last_data_state
            
            if save_last:
                save_checkpoint(
                    self.run_dir / "ckpt_last.pt",
                    self.model,
                    self.optim_bundle,
                    cfg=self.cfg,
                    step=self.global_step,
                    best_metric=self.best_metric,
                    extra=ckpt_extra,
                )
            if valid_metrics:
                full_auc = valid_metrics.get("auc_primary")
                if full_auc is not None and full_auc > self.best_metric:
                    self.best_metric = full_auc
                    save_checkpoint(
                        self.run_dir / "ckpt_best.pt",
                        self.model,
                        self.optim_bundle,
                        cfg=self.cfg,
                        step=self.global_step,
                        best_metric=self.best_metric,
                        extra={**ckpt_extra, "kind": "full"},
                    )
                    self.logger.info(f"New best at epoch {epoch_num}: auc={self.best_metric:.4f}")
            
            # 检查是否达到总步数限制
            if self.global_step >= max_train_steps:
                self.logger.info(f"Reached max_train_steps={max_train_steps}, stopping training")
                break

        self.logger.info("Training finished.")
        # Optional auto-eval after training completes
        runtime = self.cfg.get("runtime", {})
        if runtime.get("auto_eval", False):
            ckpt_choice = str(runtime.get("auto_eval_ckpt", "best")).lower()
            ckpt_name = "ckpt_best.pt" if ckpt_choice == "best" else "ckpt_last.pt"
            ckpt_path = self.run_dir / ckpt_name
            if not ckpt_path.exists():
                self.logger.warning("Auto-eval skipped: checkpoint not found at %s", ckpt_path)
                return
            split = runtime.get("auto_eval_split", "valid")
            save_preds = bool(runtime.get("auto_eval_save_preds", True))
            self.logger.info(
                "Auto-eval starting on split=%s using %s (save_preds=%s)", split, ckpt_name, save_preds
            )
            eval_config_path = self.config_path or (self.run_dir / "config.yaml")
            run_eval(
                cfg=self.cfg,
                split=split,
                ckpt_path=ckpt_path,
                run_dir=self.run_dir,
                save_preds=save_preds,
                max_batches=None,
                logger=self.logger,
                config_path=eval_config_path,
            )


__all__ = ["Trainer"]
