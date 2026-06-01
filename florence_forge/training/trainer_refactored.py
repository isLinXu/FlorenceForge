#!/usr/bin/env python3
"""
Florence Forge — 多任务训练器（v2，模块化重构版）

⚠️ 当前状态（2026-05-21）
=========================
本文件提供 **v2 训练栈**，与 `trainer.py`（v1，单文件 god class）并行存在。

- v1 入口：`florence_forge.training.trainer.MultiTaskTrainer`（默认导出，1579 行）
- v2 入口：本文件 + `training_loop.py` + `checkpoint_manager.py`（模块化）

v2 通过组合 `TrainingLoop` 与 `CheckpointManager` 实现更清晰的职责拆分；
`tests/test_training_integration.py` 测的是 v2 体系。

两者**功能上不完全等价**：v1 含 FSDP/DeepSpeed Plugin、激活值重计算 4 档策略、
异步 checkpoint 等高级特性，v2 目前聚焦核心训练循环。

迁移路线（P1）：把 v1 的高级特性逐步移植到 v2，最终统一到 v2 并删除 v1。
在此之前两套并存，**请不要混用**。

提供完整的多任务训练功能，采用模块化架构
"""

import os
import csv
import time
import json
import logging
from typing import Optional, Dict, Any, Union
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler

from ._accelerator_compat import Accelerator

from ..core.config import TrainingConfig
from ..data.dataset import MultiTaskDataset
from ..data.loader import TaskDataLoader
from .scheduler import TaskScheduler
from .lora_manager import LoRAManager
from .visualizer import TrainingVisualizer
from .monitoring import TrainingMonitor
from .gradient_validator import GradientValidator, GradientValidationConfig
from .memory_monitor import MemoryMonitor, MemoryMonitorConfig
from ..core.callbacks import CallbackManager, create_default_callbacks

# 新的模块化组件
from .device_config import DeviceConfigurator
from .checkpoint_manager import CheckpointManager
from .training_loop import TrainingLoop
from .gradient_checkpoint_optimizer import GradientCheckpointOptimizer
from ..utils.training_logging import (
    format_epoch_summary,
    format_training_complete,
    format_training_start,
)

logger = logging.getLogger(__name__)


class MultiTaskTrainer:
    """多任务训练器（重构版）
    
    采用模块化架构，将职责分离到独立组件：
    - DeviceConfigurator: 设备和混合精度配置
    - CheckpointManager: 检查点管理
    - TrainingLoop: 训练和验证循环
    - GradientCheckpointOptimizer: 梯度检查点优化
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataset: MultiTaskDataset,
        val_dataset: Optional[MultiTaskDataset] = None,
        config: Optional[TrainingConfig] = None,
        accelerator: Optional[Accelerator] = None
    ):
        """初始化训练器
        
        Args:
            model: 多任务模型
            train_dataset: 训练数据集
            val_dataset: 验证数据集
            config: 训练配置
            accelerator: Accelerate 加速器
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config or TrainingConfig()
        
        # 初始化设备配置器
        self.device_config = DeviceConfigurator(self.config)
        
        # 初始化加速器
        if accelerator is None:
            self.accelerator = self._create_accelerator()
        else:
            self.accelerator = accelerator
        
        # 初始化子组件
        self.checkpoint_manager = CheckpointManager(
            model=self.model,
            config=self.config,
            accelerator=self.accelerator
        )
        
        self.training_loop = TrainingLoop(
            model=self.model,
            config=self.config,
            accelerator=self.accelerator
        )
        
        self.gradient_optimizer = GradientCheckpointOptimizer(
            model=self.model,
            config=self.config
        )
        
        # 训练组件（延迟初始化）
        self.optimizer = None
        self.lr_scheduler = None
        self.train_dataloader = None
        self.val_dataloader = None
        
        # 可选组件
        self.task_scheduler = None
        self.lora_manager = None
        self.callback_manager = None
        self.gradient_validator = None
        self.memory_monitor = None
        
        # CSV 日志
        self.csv_file = None
        self.csv_writer = None
        self.csv_buffer = []
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('inf')

    def _should_emit_console_log(self) -> bool:
        """Return whether this process should emit user-facing progress lines."""
        return self.accelerator is None or getattr(
            self.accelerator,
            "is_local_main_process",
            True,
        )
    
    def _create_accelerator(self) -> Accelerator:
        """创建加速器实例
        
        Returns:
            配置好的 Accelerator 实例
        """
        # 设置设备
        device_type = self.device_config.setup_device()
        
        # 确定混合精度
        mixed_precision = self.device_config.determine_mixed_precision()
        
        # 构建参数
        accel_kwargs = {
            "mixed_precision": mixed_precision,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "log_with": "tensorboard" if self.config.logging_dir else None,
            "project_dir": self.config.logging_dir
        }
        
        # 分布式插件
        dist_config = self.config.distributed_settings
        if dist_config.enabled or dist_config.strategy != "none":
            plugin = self.device_config.build_distributed_plugin(dist_config)
            if plugin is not None:
                if dist_config.strategy == "fsdp":
                    accel_kwargs["fsdp_plugin"] = plugin
                elif dist_config.strategy == "deepspeed":
                    accel_kwargs["deepspeed_plugin"] = plugin
        
        return Accelerator(**accel_kwargs)
    
    def setup_training(self) -> None:
        """设置训练环境
        
        初始化所有训练组件：优化器、调度器、数据加载器等
        """
        logger.info("🔧 设置训练环境...")
        
        # 1. 梯度检查点
        self.gradient_optimizer.enable_gradient_checkpointing()
        self.gradient_optimizer.disable_kv_cache_for_training()
        
        # 2. 数据加载器
        self._setup_dataloaders()
        
        # 3. 优化器
        self._setup_optimizer()
        
        # 4. 学习率调度器
        self._setup_lr_scheduler()
        
        # 5. LoRA 管理器
        if self.config.use_lora:
            from .lora_manager import LoRAManager
            self.lora_manager = LoRAManager(self.model, self.config.lora)
            self.lora_manager.apply_lora()
        
        # 6. 任务调度器
        if hasattr(self.config, 'use_task_scheduler') and self.config.use_task_scheduler:
            self.task_scheduler = TaskScheduler(self.train_dataset, self.config)
        
        # 7. 回调管理器
        if self.config.use_callbacks:
            self.callback_manager = create_default_callbacks(self.config)
            self.training_loop.callback_manager = self.callback_manager
        
        # 8. 梯度验证器（调试）
        if hasattr(self.config, 'enable_gradient_validation') and self.config.enable_gradient_validation:
            grad_config = GradientValidationConfig()
            self.gradient_validator = GradientValidator(self.model, grad_config)
        
        # 9. 内存监控
        if hasattr(self.config, 'enable_memory_monitoring') and self.config.enable_memory_monitoring:
            mem_config = MemoryMonitorConfig()
            self.memory_monitor = MemoryMonitor(mem_config)
        
        # 10. Accelerator prepare
        self.model, self.optimizer, self.train_dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader
        )
        
        if self.val_dataloader is not None:
            self.val_dataloader = self.accelerator.prepare(self.val_dataloader)
        
        self.lr_scheduler = self.accelerator.prepare(self.lr_scheduler)
        
        # 11. CSV 日志
        self._init_csv_logger()
        
        logger.info("✅ 训练环境设置完成")
    
    def _setup_dataloaders(self) -> None:
        """设置数据加载器"""
        self.train_dataloader = TaskDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        if self.val_dataset is not None:
            self.val_dataloader = TaskDataLoader(
                dataset=self.val_dataset,
                batch_size=self.config.eval_batch_size or self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=True
            )
    
    def _setup_optimizer(self) -> None:
        """设置优化器"""
        # 参数分组（可选）
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon
        )
    
    def _setup_lr_scheduler(self) -> None:
        """设置学习率调度器"""
        num_training_steps = len(self.train_dataloader) * self.config.num_epochs
        num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)
        
        self.lr_scheduler = get_scheduler(
            name=self.config.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    
    def _init_csv_logger(self) -> None:
        """初始化 CSV 日志"""
        if self.config.logging_dir:
            log_dir = Path(self.config.logging_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            csv_path = log_dir / "training_metrics.csv"
            self.csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
            
            fieldnames = ['epoch', 'step', 'loss', 'learning_rate', 'val_loss']
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
            self.csv_writer.writeheader()
    
    def train(self) -> Dict[str, Any]:
        """执行完整训练流程
        
        Returns:
            训练摘要字典
        """
        training_start_time = time.perf_counter()
        if self._should_emit_console_log():
            logger.info(
                format_training_start(
                    epochs=self.config.num_epochs,
                    batch_size=self.config.batch_size,
                    gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                    logging_steps=self.config.logging_steps,
                )
            )
        
        # 设置训练环境（如果尚未设置）
        if self.optimizer is None:
            self.setup_training()
        
        # 触发训练开始回调
        if self.callback_manager:
            self.callback_manager.on_train_begin()
        
        try:
            for epoch in range(self.config.num_epochs):
                self.current_epoch = epoch
                
                # 训练一个 epoch
                train_metrics = self.training_loop.train_epoch(
                    train_dataloader=self.train_dataloader,
                    optimizer=self.optimizer,
                    lr_scheduler=self.lr_scheduler,
                    epoch=epoch,
                    gradient_validator=self.gradient_validator,
                    memory_monitor=self.memory_monitor
                )
                
                # 验证
                val_metrics = None
                if self.val_dataloader is not None:
                    val_metrics = self.training_loop.validate_epoch(
                        val_dataloader=self.val_dataloader,
                        epoch=epoch
                    )
                
                # 记录指标
                self._record_epoch_metrics(epoch, train_metrics, val_metrics)
                
                # 保存检查点
                is_best = False
                if val_metrics and 'val_loss' in val_metrics:
                    is_best = val_metrics['val_loss'] < self.best_metric
                    if is_best:
                        self.best_metric = val_metrics['val_loss']
                
                self.checkpoint_manager.save_checkpoint(
                    epoch=epoch,
                    optimizer=self.optimizer,
                    lr_scheduler=self.lr_scheduler,
                    metrics={**train_metrics, **(val_metrics or {})},
                    is_best=is_best
                )
                
                # 早停检查
                if val_metrics and self.config.early_stopping_patience > 0:
                    if self.training_loop.should_early_stop(
                        current_metric=val_metrics.get('val_loss', float('inf')),
                        patience=self.config.early_stopping_patience
                    ):
                        logger.info("🛑 触发早停")
                        break

                # max_steps 硬上限：优先于 num_epochs，达到后终止外层 epoch 循环
                if self.training_loop._max_steps_reached():
                    logger.info(
                        "🏁 已达到 max_steps=%s，停止训练（num_epochs 未跑满）",
                        self.config.max_steps,
                    )
                    break
        
        finally:
            # 触发训练结束回调
            if self.callback_manager:
                self.callback_manager.on_train_end()

            # load_best_model_at_end：保存最终模型前先恢复最佳检查点权重
            self._maybe_restore_best_model()

            # 保存最终模型
            self.checkpoint_manager.save_final_model(
                merge_lora=self.config.use_lora,
                lora_manager=self.lora_manager
            )
            
            # 清理资源
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
        val_metrics: Optional[Dict[str, float]]
    ) -> None:
        """记录 epoch 指标"""
        if self._should_emit_console_log():
            logger.info(
                format_epoch_summary(
                    epoch=epoch + 1,
                    total_epochs=self.config.num_epochs,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                )
            )
        
        # CSV 日志
        if self.csv_writer:
            row = {
                'epoch': epoch,
                'step': self.training_loop.global_step,
                'loss': train_metrics.get('loss', 0.0),
                'learning_rate': train_metrics.get('learning_rate', 0.0),
                'val_loss': val_metrics.get('val_loss', '') if val_metrics else ''
            }
            self.csv_writer.writerow(row)
            self.csv_file.flush()
    
    def _get_training_summary(self) -> Dict[str, Any]:
        """获取训练摘要"""
        return {
            'total_epochs': self.current_epoch + 1,
            'global_steps': self.training_loop.global_step,
            'best_metric': self.best_metric,
            'best_checkpoint': str(self.checkpoint_manager.get_best_checkpoint_path()) if self.checkpoint_manager.get_best_checkpoint_path() else None
        }
    
    def _maybe_restore_best_model(self) -> None:
        """若启用 load_best_model_at_end，则在收尾前恢复最佳检查点权重。

        与 v1 行为对齐：训练结束后，最终模型应是验证指标最优的那一份，
        而非最后一个 epoch 的权重。
        """
        if not getattr(self.config, "load_best_model_at_end", False):
            return

        best_path = self.checkpoint_manager.get_best_checkpoint_path()
        if not best_path or not Path(best_path).exists():
            logger.info("ℹ️  未找到最佳检查点，保留最后一个 epoch 的权重")
            return

        try:
            self.checkpoint_manager.load_checkpoint(
                checkpoint_path=best_path,
                optimizer=None,
                lr_scheduler=None,
            )
            logger.info("✅ 已恢复最佳检查点权重：%s", best_path)
        except Exception as exc:  # 恢复失败不应阻断收尾流程
            logger.warning("⚠️  恢复最佳检查点失败，保留当前权重：%s", exc)

    def _cleanup(self) -> None:
        """清理资源"""
        if self.csv_file:
            self.csv_file.close()
        
        self.checkpoint_manager.cleanup()
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """加载检查点
        
        Args:
            checkpoint_path: 检查点路径
        """
        metadata = self.checkpoint_manager.load_checkpoint(
            checkpoint_path=checkpoint_path,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler
        )
        
        self.current_epoch = metadata.get('epoch', 0)
        logger.info(f"✅ 检查点已加载：Epoch {self.current_epoch}")
    
    def save_merged_model(self, output_dir: Union[str, Path]) -> None:
        """保存合并后的模型
        
        Args:
            output_dir: 输出目录
        """
        if self.lora_manager is None:
            logger.warning("⚠️  未使用 LoRA，无需合并")
            return
        
        from .model_merger import ModelMerger
        merger = ModelMerger(self.model, self.lora_manager)
        merger.merge_and_save(output_dir)
