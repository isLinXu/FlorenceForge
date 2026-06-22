#!/usr/bin/env python3
"""
Florence Forge — 多任务训练器（模块化训练栈）

组合 ``TrainingLoop``、``CheckpointManager``、``DeviceConfigurator`` 等子模块，
提供 FSDP/DeepSpeed、激活重计算、异步 checkpoint、LoRA 与多任务训练能力。
"""

import csv
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_scheduler

from ._accelerator_compat import Accelerator
from ..core.callbacks import create_default_callbacks
from ..core.config import TrainingConfig
from ..data.dataset import MultiTaskDataset
from ..data.loader import TaskDataLoader
from ..utils.training_logging import (
    format_epoch_summary,
    format_training_complete,
    format_training_start,
)
from .async_checkpoint import AsyncCheckpointSaver
from .checkpoint_manager import CheckpointManager
from .deepspeed_plugin import DeepSpeedPlugin
from .device_config import DeviceConfigurator
from .fsdp_plugin import FSDPPlugin
from .gradient_checkpoint_optimizer import (
    ActivationRecomputePolicy,
    GradientCheckpointOptimizer,
)
from .gradient_validator import GradientValidationConfig, GradientValidator
from .lora_manager import LoRAManager
from .memory_monitor import MemoryMonitor, MemoryMonitorConfig
from .model_merger import ModelMerger
from .scheduler import TaskScheduler
from .training_loop import TrainingLoop

logger = logging.getLogger(__name__)


class MultiTaskTrainer:
    """多任务训练器

    职责分离到独立组件：
    - DeviceConfigurator: 设备与混合精度
    - CheckpointManager: 检查点管理
    - TrainingLoop: 训练/验证循环
    - GradientCheckpointOptimizer: 梯度检查点（4 档策略）
    - AsyncCheckpointSaver: 异步检查点保存
    - FSDPPlugin / DeepSpeedPlugin: 分布式训练
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset: MultiTaskDataset,
        val_dataset: Optional[MultiTaskDataset] = None,
        config: Optional[TrainingConfig] = None,
        accelerator: Optional[Accelerator] = None,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config or TrainingConfig()

        self.device_config = DeviceConfigurator(self.config)
        self.fsdp_plugin = FSDPPlugin()
        self.deepspeed_plugin = DeepSpeedPlugin()

        if accelerator is None:
            self.accelerator = self._create_accelerator()
        else:
            self.accelerator = accelerator

        self.checkpoint_manager = CheckpointManager(
            model=self.model,
            config=self.config,
            accelerator=self.accelerator,
        )
        self.training_loop = TrainingLoop(
            model=self.model,
            config=self.config,
            accelerator=self.accelerator,
        )

        recompute_policy = self._resolve_recompute_policy()
        self.gradient_optimizer = GradientCheckpointOptimizer(
            model=self.model,
            config=self.config,
            policy=recompute_policy,
        )
        self.async_checkpoint_saver = AsyncCheckpointSaver(
            checkpoint_dir=str(Path(self.config.output_dir) / "checkpoints"),
            max_checkpoints=getattr(self.config, "max_checkpoints", 5),
            async_save=getattr(self.config, "async_checkpoint", True),
        )

        self.optimizer = None
        self.lr_scheduler = None
        self.train_dataloader = None
        self.val_dataloader = None

        self.task_scheduler = None
        self.lora_manager = None
        self.model_merger = None
        self.callback_manager = None
        self.gradient_validator = None
        self.memory_monitor = None

        self.csv_file = None
        self.csv_writer = None

        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float("inf")

    @property
    def output_dir(self) -> Path:
        return Path(self.config.output_dir)

    def _resolve_recompute_policy(self) -> ActivationRecomputePolicy:
        policy_name = getattr(self.config, "activation_recompute_policy", None)
        if policy_name:
            try:
                return ActivationRecomputePolicy[policy_name]
            except KeyError:
                logger.warning("未知的激活重计算策略: %s，使用 off", policy_name)

        model_settings = getattr(self.config, "model_settings", None)
        if model_settings is None:
            return ActivationRecomputePolicy.off

        strategy = getattr(model_settings, "activation_checkpointing_strategy", "none")
        strategy_map = {
            "none": ActivationRecomputePolicy.off,
            "selective": ActivationRecomputePolicy.low,
            "auto": ActivationRecomputePolicy.medium,
            "full": ActivationRecomputePolicy.high,
        }
        if strategy in strategy_map:
            return strategy_map[strategy]
        if getattr(model_settings, "gradient_checkpointing", False):
            return ActivationRecomputePolicy.high
        return ActivationRecomputePolicy.off

    def _should_emit_console_log(self) -> bool:
        return self.accelerator is None or getattr(
            self.accelerator, "is_local_main_process", True
        )

    def _create_accelerator(self) -> Accelerator:
        self.device_config.setup_device()
        mixed_precision = self.device_config.determine_mixed_precision()

        accel_kwargs = {
            "mixed_precision": mixed_precision,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "log_with": "tensorboard" if self.config.logging_dir else None,
            "project_dir": self.config.logging_dir,
        }

        dist_config = self.config.distributed_settings
        if dist_config.enabled or dist_config.strategy != "none":
            plugin = self.device_config.build_distributed_plugin(dist_config)
            if plugin is not None:
                if dist_config.strategy == "fsdp":
                    accel_kwargs["fsdp_plugin"] = plugin
                elif dist_config.strategy == "deepspeed":
                    accel_kwargs["deepspeed_plugin"] = plugin
            elif dist_config.strategy == "fsdp" and not self.fsdp_plugin.is_available:
                logger.warning("FSDP 不可用，回退到单卡训练")
            elif dist_config.strategy == "deepspeed" and not self.deepspeed_plugin.is_available:
                logger.warning("DeepSpeed 不可用，回退到单卡训练")

        return Accelerator(**accel_kwargs)

    def setup_training(self) -> None:
        logger.info("Setting up training environment...")

        self.gradient_optimizer.enable_gradient_checkpointing()
        self.gradient_optimizer.disable_kv_cache_for_training()
        self._setup_dataloaders()
        self._setup_optimizer()
        self._setup_lr_scheduler()

        model_config = getattr(self.config, "model_config", {})
        if isinstance(model_config, dict):
            lora_config = model_config.get("lora_config")
        else:
            lora_config = getattr(model_config, "lora_config", None)

        if self.config.use_lora and not getattr(self.model, "is_peft_model", False):
            task_types = list(getattr(self.train_dataset, "task_indices", {}).keys())
            self.lora_manager = LoRAManager(lora_config)
            self.model_merger = ModelMerger(self.lora_manager)
            if task_types:
                first_task = task_types[0]
                self.model = self.lora_manager.apply_lora_to_model(self.model, first_task)
                for task_type in task_types[1:]:
                    self.lora_manager.add_adapter_to_model(self.model, task_type)
                if hasattr(self.lora_manager, "print_trainable_parameters"):
                    self.lora_manager.print_trainable_parameters(self.model)
        elif self.config.use_lora:
            self.lora_manager = LoRAManager(lora_config)
            self.model_merger = ModelMerger(self.lora_manager)
            logger.info("检测到模型已在加载阶段注入 LoRA，复用现有 PEFT 模型")

        if getattr(self.config, "use_task_scheduler", False):
            task_types = list(getattr(self.train_dataset, "task_indices", {}).keys())
            self.task_scheduler = TaskScheduler(
                task_types=task_types,
                config=getattr(self.config, "task_scheduling_settings", None),
            )

        if getattr(self.config, "use_callbacks", False):
            self.callback_manager = create_default_callbacks(self.config)
            self.training_loop.callback_manager = self.callback_manager

        if getattr(self.config, "enable_gradient_validation", False):
            self.gradient_validator = GradientValidator(
                self.model, GradientValidationConfig()
            )

        if getattr(self.config, "enable_memory_monitoring", False):
            self.memory_monitor = MemoryMonitor(MemoryMonitorConfig())

        self.model, self.optimizer, self.train_dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader
        )
        if self.val_dataloader is not None:
            self.val_dataloader = self.accelerator.prepare(self.val_dataloader)
        self.lr_scheduler = self.accelerator.prepare(self.lr_scheduler)
        self._init_csv_logger()
        logger.info("Training environment setup complete")

    def _setup_dataloaders(self) -> None:
        data = self.config.data_settings
        device_type = self.device_config.device_type or self.device_config.setup_device()
        pin_memory = bool(data.pin_memory and device_type == "cuda")
        self.train_dataloader = TaskDataLoader(
            dataset=self.train_dataset,
            batch_size=data.batch_size,
            shuffle=data.shuffle,
            num_workers=data.num_workers,
            pin_memory=pin_memory,
            drop_last=data.drop_last,
        )
        if self.val_dataset is not None:
            self.val_dataloader = TaskDataLoader(
                dataset=self.val_dataset,
                batch_size=data.batch_size,
                shuffle=False,
                num_workers=data.num_workers,
                pin_memory=pin_memory,
            )

    def _setup_optimizer(self) -> None:
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
        )

    def _setup_lr_scheduler(self) -> None:
        num_training_steps = len(self.train_dataloader) * self.config.num_epochs
        num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)
        self.lr_scheduler = get_scheduler(
            name=self.config.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

    def _init_csv_logger(self) -> None:
        if not self.config.logging_dir:
            return
        log_dir = Path(self.config.logging_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        csv_path = log_dir / "training_metrics.csv"
        self.csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        fieldnames = ["epoch", "step", "loss", "learning_rate", "val_loss"]
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.csv_writer.writeheader()

    def train(self) -> Dict[str, Any]:
        training_start_time = time.perf_counter()
        if self._should_emit_console_log():
            logger.info(
                format_training_start(
                    epochs=self.config.num_epochs,
                    batch_size=self.config.data_settings.batch_size,
                    gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                    logging_steps=self.config.logging_steps,
                )
            )

        if self.optimizer is None:
            self.setup_training()

        if self.callback_manager:
            self.callback_manager.on_train_begin()

        try:
            for epoch in range(self.config.num_epochs):
                self.current_epoch = epoch
                train_metrics = self.training_loop.train_epoch(
                    train_dataloader=self.train_dataloader,
                    optimizer=self.optimizer,
                    lr_scheduler=self.lr_scheduler,
                    epoch=epoch,
                    gradient_validator=self.gradient_validator,
                    memory_monitor=self.memory_monitor,
                )

                val_metrics = None
                if self.val_dataloader is not None:
                    val_metrics = self.training_loop.validate_epoch(
                        val_dataloader=self.val_dataloader,
                        epoch=epoch,
                    )

                self._record_epoch_metrics(epoch, train_metrics, val_metrics)

                is_best = False
                if val_metrics and "val_loss" in val_metrics:
                    is_best = val_metrics["val_loss"] < self.best_metric
                    if is_best:
                        self.best_metric = val_metrics["val_loss"]

                self.checkpoint_manager.save_checkpoint(
                    epoch=epoch,
                    optimizer=self.optimizer,
                    lr_scheduler=self.lr_scheduler,
                    metrics={**train_metrics, **(val_metrics or {})},
                    is_best=is_best,
                    lora_manager=self.lora_manager,
                )

                if val_metrics and self.config.early_stopping_patience > 0:
                    if self.training_loop.should_early_stop(
                        current_metric=val_metrics.get("val_loss", float("inf")),
                        patience=self.config.early_stopping_patience,
                    ):
                        logger.info("触发早停")
                        break

                if self.training_loop._max_steps_reached():
                    logger.info("已达到 max_steps=%s，停止训练", self.config.max_steps)
                    break
        finally:
            if self.callback_manager:
                self.callback_manager.on_train_end()
            self._maybe_restore_best_model()
            self.checkpoint_manager.save_final_model(
                merge_lora=self.config.use_lora,
                lora_manager=self.lora_manager,
            )
            self._cleanup()

        if self._should_emit_console_log():
            logger.info(
                format_training_complete(
                    completed_steps=self.training_loop.global_step,
                    elapsed_seconds=time.perf_counter() - training_start_time,
                    best_metric=self.best_metric,
                )
            )
        return self._get_training_summary()

    def _record_epoch_metrics(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]],
    ) -> None:
        if self._should_emit_console_log():
            logger.info(
                format_epoch_summary(
                    epoch=epoch + 1,
                    total_epochs=self.config.num_epochs,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                )
            )
        if self.csv_writer:
            self.csv_writer.writerow(
                {
                    "epoch": epoch,
                    "step": self.training_loop.global_step,
                    "loss": train_metrics.get("loss", 0.0),
                    "learning_rate": train_metrics.get("learning_rate", 0.0),
                    "val_loss": val_metrics.get("val_loss", "") if val_metrics else "",
                }
            )
            self.csv_file.flush()

    def _save_config(self) -> None:
        """保存训练配置与数据集统计到 output_dir。"""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.config.save_to_file(output_dir / "training_config.json")
        dataset_stats = {
            "train_dataset": self.train_dataset.get_task_statistics(),
            "val_dataset": (
                self.val_dataset.get_task_statistics() if self.val_dataset else None
            ),
        }
        with open(output_dir / "dataset_statistics.json", "w", encoding="utf-8") as f:
            json.dump(dataset_stats, f, indent=2, ensure_ascii=False)

    def _get_training_summary(self) -> Dict[str, Any]:
        summary = {
            "total_epochs": self.current_epoch + 1,
            "global_steps": self.training_loop.global_step,
            "best_metric": self.best_metric,
            "best_checkpoint": (
                str(self.checkpoint_manager.get_best_checkpoint_path())
                if self.checkpoint_manager.get_best_checkpoint_path()
                else None
            ),
        }
        try:
            summary["final_learning_rate"] = self.training_loop._get_current_lr()
        except Exception:
            summary["final_learning_rate"] = None
        summary["use_lora"] = getattr(self.config, "use_lora", False)
        summary["tasks"] = getattr(self.config, "tasks", [])
        return summary

    def _maybe_restore_best_model(self) -> None:
        if not getattr(self.config, "load_best_model_at_end", False):
            return
        best_path = self.checkpoint_manager.get_best_checkpoint_path()
        if not best_path or not Path(best_path).exists():
            return
        try:
            self.checkpoint_manager.load_checkpoint(
                checkpoint_path=best_path,
                optimizer=None,
                lr_scheduler=None,
            )
        except Exception as exc:
            logger.warning("恢复最佳检查点失败: %s", exc)

    def _cleanup(self) -> None:
        if self.csv_file:
            self.csv_file.close()
        self.async_checkpoint_saver.shutdown(wait=True)
        self.checkpoint_manager.cleanup()

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        metadata = self.checkpoint_manager.load_checkpoint(
            checkpoint_path=checkpoint_path,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
        )
        self.current_epoch = metadata.get("epoch", 0)

    def save_merged_model(self, output_dir: Union[str, Path]) -> None:
        if self.model_merger is None:
            logger.warning("模型合并器未初始化")
            return
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_to_merge = (
            self.accelerator.unwrap_model(self.model)
            if self.accelerator is not None
            else self.model
        )
        merged_model = self.model_merger.merge_all_adapters(model_to_merge)
        if self.accelerator is not None:
            self.accelerator.save_model(merged_model, output_dir)
        elif hasattr(merged_model, "save_pretrained"):
            merged_model.save_pretrained(output_dir)
        else:
            torch.save(merged_model.state_dict(), output_dir / "pytorch_model.bin")
