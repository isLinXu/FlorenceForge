#!/usr/bin/env python3
"""
Florence Forge - 多数据集多任务训练器

专门为多数据集多任务训练场景设计的高级训练器
"""

import os
import json
import logging
import time
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ._accelerator_compat import Accelerator

from ..core.config import TrainingConfig
from ..data.multi_dataset_manager import MultiDatasetManager
from ..data.dataset import MultiTaskDataset
from ..data.loader import TaskDataLoader
from .trainer import MultiTaskTrainer
from .scheduler import TaskScheduler
from .gradient_validator import GradientValidator, GradientValidationConfig
from .memory_monitor import MemoryMonitor, MemoryMonitorConfig
from ..utils.training_logging import (
    format_training_step,
    resolve_total_steps,
)

logger = logging.getLogger(__name__)

class MultiDatasetTrainer(MultiTaskTrainer):
    """多数据集多任务训练器
    
    继承自MultiTaskTrainer，增加了多数据集协调功能
    """
    
    def __init__(
        self,
        model: nn.Module,
        dataset_manager: MultiDatasetManager,
        config: Optional[TrainingConfig] = None,
        accelerator: Optional[Accelerator] = None,
        task_types: Optional[List[str]] = None
    ):
        """初始化多数据集训练器

        Args:
            model: 多任务模型（Florence2MultiTaskModel 或任何兼容 nn.Module 的模型）
            dataset_manager: 多数据集管理器
            config: 训练配置
            accelerator: Accelerate加速器
            task_types: 要训练的任务类型列表，None表示训练所有任务
        """
        self.dataset_manager = dataset_manager
        self.task_types = task_types
        self.config = config or TrainingConfig()
        
        # 创建训练和验证数据集
        train_dataset, val_dataset, _ = self._create_datasets()
        
        # 调用父类初始化
        super().__init__(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=self.config,
            accelerator=accelerator
        )
        
        # 多数据集特有的状态
        self.dataset_performance = defaultdict(lambda: defaultdict(list))
        self.dataset_performance_history_limit = int(
            getattr(self.config, "dataset_performance_history_limit", 1000) or 0
        )
        self.dataset_weights = {}
        self.current_dataset_strategy = "balanced"
        
        # 梯度验证器（多数据集训练中更重要）
        opt_settings = self.config.optimization_settings
        if hasattr(opt_settings, 'max_grad_norm') and opt_settings.max_grad_norm > 0:
            grad_config = GradientValidationConfig(
                max_grad_norm_threshold=opt_settings.max_grad_norm * 1.5,  # 稍微宽松一些
                log_frequency=50,  # 多数据集训练中更频繁的检查
                stats_save_frequency=200,
                monitor_layer_gradients=True  # 启用层级监控
            )
            self.gradient_validator = GradientValidator(self.model, grad_config)
        else:
            self.gradient_validator = None
        
        # 内存监控器（多数据集训练中内存压力更大）
        memory_config = MemoryMonitorConfig(
            enable_monitoring=True,
            log_frequency=25,  # 更频繁的内存监控
            warning_threshold_percent=75.0,  # 更严格的警告阈值
            critical_threshold_percent=85.0,
            enable_gpu_monitoring=True,
            auto_cleanup=True,
            save_stats=True,
            enable_continuous_monitoring=False  # 避免过多的监控开销
        )
        self.memory_monitor = MemoryMonitor(memory_config)
        
        logger.info("多数据集训练器初始化完成")
    
    def _create_datasets(self) -> tuple:
        """创建训练和验证数据集
        
        Returns:
            (训练集, 验证集, 测试集) 元组
        """
        logger.info("正在创建多数据集...")
        
        # 验证配置
        validation_result = self.dataset_manager.validate_configuration()
        if validation_result["errors"]:
            raise ValueError(f"数据集配置错误: {validation_result['errors']}")
        
        if validation_result["warnings"]:
            for warning in validation_result["warnings"]:
                logger.warning(warning)
        
        # 创建统一数据集
        processor = getattr(self, 'processor', None)
        full_dataset = self.dataset_manager.create_unified_dataset(
            task_types=self.task_types,
            processor=processor
        )
        
        # 创建训练/验证划分
        train_dataset, val_dataset, test_dataset = self.dataset_manager.create_balanced_split(
            val_ratio=self.config.eval_ratio if hasattr(self.config, 'eval_ratio') else 0.2,
            stratify_by_task=True,
            stratify_by_dataset=True,
            random_seed=self.config.seed if hasattr(self.config, 'seed') else None
        )
        
        logger.info(
            f"数据集创建完成 - 训练: {len(train_dataset)}, "
            f"验证: {len(val_dataset)} 样本"
        )
        
        return train_dataset, val_dataset, test_dataset
    
    def setup_training(self) -> None:
        """设置训练组件（重写父类方法）"""
        logger.info("正在设置多数据集训练组件...")
        
        # 调用父类设置
        super().setup_training()
        
        # 初始化数据集权重
        self._initialize_dataset_weights()
        
        # 设置数据集特定的任务调度器
        self._setup_dataset_aware_scheduler()
        
        logger.info("多数据集训练组件设置完成")
    
    def _initialize_dataset_weights(self) -> None:
        """初始化数据集权重"""
        dataset_stats = self.dataset_manager.get_dataset_statistics()
        
        for dataset_name, stats in dataset_stats["datasets"].items():
            if stats["loaded"]:
                # 基于数据集大小和优先级计算初始权重
                sample_count = stats.get("total_samples", 1)
                priority = stats.get("priority", 1)
                
                # 权重 = 优先级 / sqrt(样本数)，避免大数据集完全主导
                weight = priority / (sample_count ** 0.5)
                self.dataset_weights[dataset_name] = weight
        
        logger.info(f"数据集权重初始化: {self.dataset_weights}")
    
    def _setup_dataset_aware_scheduler(self) -> None:
        """设置数据集感知的任务调度器"""
        # 获取所有任务类型和对应的数据集
        task_dataset_mapping = {}
        
        for task_type, mapping in self.dataset_manager.task_mappings.items():
            task_dataset_mapping[task_type] = mapping.datasets
        
        # 扩展任务调度器以支持数据集信息
        if hasattr(self.task_scheduler, 'set_dataset_mapping'):
            self.task_scheduler.set_dataset_mapping(task_dataset_mapping)
    
    def _train_epoch(self) -> Dict[str, float]:
        """训练一个epoch（重写父类方法）
        
        Returns:
            训练指标字典
        """
        self.model.train()
        epoch_metrics = defaultdict(list)
        dataset_metrics = defaultdict(lambda: defaultdict(list))
        if not hasattr(self, "_progress_log_start_time"):
            self._progress_log_start_time = time.perf_counter()
        total_steps = resolve_total_steps(
            self.train_dataloader,
            self.config.num_epochs,
            self.config.max_steps,
        )
        
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {self.current_epoch + 1}/{self.config.num_epochs}",
            disable=not self.accelerator.is_local_main_process
        )
        
        for step, batch in enumerate(progress_bar):
            step_start_time = time.time()
            
            # 跳过空批次
            if batch is None or batch.get("is_empty", False):
                continue
            
            # 处理批次数据
            if isinstance(batch, list):
                sample = batch[0]
            else:
                sample = batch
            
            # 获取任务类型和数据集信息
            task_type = self._extract_task_type(sample)
            dataset_name = self._extract_dataset_name(sample)
            
            # 切换LoRA适配器（如果使用）
            if self.lora_manager and hasattr(self.model, 'set_adapter'):
                self.lora_manager.switch_adapter(self.model, task_type)
            
            # 前向传播
            loss = self._forward_pass(sample)
            
            # 反向传播
            self._backward_pass(loss)
            
            # 记录指标
            step_time = time.time() - step_start_time
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 更新任务和数据集性能
            self.task_scheduler.update_task_performance(task_type, loss.item())
            self._record_dataset_performance(dataset_name, task_type, loss.item())
            
            # 记录步骤指标
            should_log = (
                self.global_step % self.config.logging_steps == 0 or
                self.global_step == 1 or
                (self.global_step % max(1, len(self.train_dataloader) // 3) == 0)
            )
            if should_log:
                grad_norm = self._calculate_grad_norm()
                self._record_step_metrics(
                    step=self.global_step,
                    epoch=self.current_epoch,
                    task_type=task_type,
                    loss=loss.item(),
                    learning_rate=current_lr,
                    grad_norm=grad_norm,
                    time_per_step=step_time,
                    dataset_name=dataset_name
                )
                if self.accelerator.is_local_main_process:
                    logger.info(
                        format_training_step(
                            completed_step=self.global_step + 1,
                            total_steps=total_steps,
                            epoch=self.current_epoch + 1,
                            total_epochs=self.config.num_epochs,
                            metrics={
                                "loss": loss.item(),
                                "learning_rate": current_lr,
                                "grad_norm": grad_norm,
                                "time_per_step": step_time,
                            },
                            task_type=task_type,
                            dataset_name=dataset_name,
                            elapsed_seconds=time.perf_counter() - self._progress_log_start_time,
                        )
                    )
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{current_lr:.2e}',
                'task': task_type,
                'dataset': dataset_name
            })
            
            # 收集指标
            epoch_metrics['loss'].append(loss.item())
            epoch_metrics['learning_rate'].append(current_lr)
            dataset_metrics[dataset_name]['loss'].append(loss.item())
            dataset_metrics[dataset_name][task_type].append(loss.item())
            
            self.global_step += 1
            
            # 检查是否达到最大步数
            if self.config.max_steps and self.global_step >= self.config.max_steps:
                break
        
        # 计算epoch平均指标
        avg_metrics = {
            key: sum(values) / len(values)
            for key, values in epoch_metrics.items()
        }
        
        # 添加数据集特定指标
        avg_metrics['dataset_metrics'] = {
            dataset: {
                metric: sum(values) / len(values)
                for metric, values in metrics.items()
            }
            for dataset, metrics in dataset_metrics.items()
        }
        
        # 动态调整数据集权重
        self._adjust_dataset_weights(dataset_metrics)
        
        return avg_metrics

    def _record_dataset_performance(
        self,
        dataset_name: str,
        task_type: str,
        loss: float,
    ) -> None:
        """记录数据集-任务级 loss，并限制内存历史长度。"""
        losses = self.dataset_performance[dataset_name][task_type]
        losses.append(loss)
        if (
            self.dataset_performance_history_limit > 0
            and len(losses) > self.dataset_performance_history_limit
        ):
            del losses[:-self.dataset_performance_history_limit]
        elif self.dataset_performance_history_limit == 0:
            losses.clear()
    
    def _extract_task_type(self, sample: Dict[str, Any]) -> str:
        """从样本中提取任务类型"""
        if isinstance(sample, dict) and 'task_type' in sample:
            return sample['task_type']
        elif hasattr(sample, 'task_type'):
            return sample.task_type
        else:
            # 备选方案：使用调度器
            return self.task_scheduler.select_task(self.current_epoch)
    
    def _extract_dataset_name(self, sample: Dict[str, Any]) -> str:
        """从样本中提取数据集名称"""
        if isinstance(sample, dict):
            metadata = sample.get('metadata', {})
            if isinstance(metadata, dict):
                return metadata.get('dataset_name', 'unknown')
        
        return 'unknown'
    
    def _forward_pass(self, sample: Dict[str, Any]) -> torch.Tensor:
        """执行前向传播"""
        device = torch.device("cpu")
        input_ids = sample["input_ids"].to(device)
        pixel_values = sample["pixel_values"].to(device)
        attention_mask = sample.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        with self.accelerator.accumulate(self.model):
            outputs = self.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                labels=input_ids
            )
            return outputs.loss
    
    def _backward_pass(self, loss: torch.Tensor) -> None:
        """执行反向传播"""
        with self.accelerator.accumulate(self.model):
            self.accelerator.backward(loss)
            
            # 梯度裁剪
            if self.config.optimization_settings.max_grad_norm > 0:
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.optimization_settings.max_grad_norm
                )
            
            # 优化器步骤
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()
    
    def _calculate_grad_norm(self) -> float:
        """计算梯度范数"""
        if self.config.optimization_settings.max_grad_norm <= 0:
            return 0.0
        
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        
        return total_norm ** (1. / 2)
    
    def _record_step_metrics(
        self,
        step: int,
        epoch: int,
        task_type: str,
        loss: float,
        learning_rate: float,
        grad_norm: float,
        time_per_step: float,
        dataset_name: str = "unknown"
    ) -> None:
        """记录步骤指标（扩展版本）"""
        # 调用父类方法
        super()._record_step_metrics(
            step, epoch, task_type, loss, learning_rate, grad_norm, time_per_step
        )
        
        # 记录数据集特定指标
        dataset_csv_path = self.output_dir / "dataset_step_metrics.csv"
        
        # 如果文件不存在，创建并写入标题
        if not dataset_csv_path.exists():
            import csv
            with open(dataset_csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'step', 'epoch', 'task_type', 'dataset_name', 'loss',
                    'learning_rate', 'grad_norm', 'time_per_step'
                ])
        
        # 追加数据
        import csv
        with open(dataset_csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                step, epoch, task_type, dataset_name, loss,
                learning_rate, grad_norm, time_per_step
            ])
    
    def _adjust_dataset_weights(self, dataset_metrics: Dict[str, Dict[str, List[float]]]) -> None:
        """动态调整数据集权重"""
        if not self.config.adaptive_dataset_weighting:
            return
        
        # 计算每个数据集的平均损失
        dataset_avg_loss = {}
        for dataset_name, metrics in dataset_metrics.items():
            if 'loss' in metrics and metrics['loss']:
                dataset_avg_loss[dataset_name] = sum(metrics['loss']) / len(metrics['loss'])
        
        if len(dataset_avg_loss) < 2:
            return
        
        # 基于相对性能调整权重
        min_loss = min(dataset_avg_loss.values())
        max_loss = max(dataset_avg_loss.values())
        
        if max_loss - min_loss > 0.1:  # 只有在损失差异显著时才调整
            for dataset_name, avg_loss in dataset_avg_loss.items():
                # 性能差的数据集获得更高权重
                relative_performance = (avg_loss - min_loss) / (max_loss - min_loss)
                new_weight = 1.0 + relative_performance * 0.5  # 最多增加50%权重
                
                self.dataset_weights[dataset_name] = new_weight
            
            logger.info(f"数据集权重已调整: {self.dataset_weights}")
    
    def get_dataset_performance_summary(self) -> Dict[str, Any]:
        """获取数据集性能摘要
        
        Returns:
            数据集性能摘要
        """
        summary = {
            "dataset_weights": self.dataset_weights.copy(),
            "dataset_performance": {},
            "task_dataset_distribution": {}
        }
        
        # 计算每个数据集的性能统计
        for dataset_name, task_metrics in self.dataset_performance.items():
            dataset_summary = {}
            for task_type, losses in task_metrics.items():
                if losses:
                    dataset_summary[task_type] = {
                        "avg_loss": sum(losses) / len(losses),
                        "min_loss": min(losses),
                        "max_loss": max(losses),
                        "sample_count": len(losses)
                    }
            summary["dataset_performance"][dataset_name] = dataset_summary
        
        # 统计任务-数据集分布
        for task_type, mapping in self.dataset_manager.task_mappings.items():
            summary["task_dataset_distribution"][task_type] = {
                "datasets": mapping.datasets,
                "weights": mapping.weights,
                "sampling_strategy": mapping.sampling_strategy
            }
        
        return summary
    
    def save_dataset_performance(self, file_path: Optional[Union[str, Path]] = None) -> None:
        """保存数据集性能信息
        
        Args:
            file_path: 保存路径，None表示使用默认路径
        """
        if file_path is None:
            file_path = self.output_dir / "dataset_performance.json"
        
        performance_data = {
            "training_summary": self.get_dataset_performance_summary(),
            "dataset_manager_stats": self.dataset_manager.get_dataset_statistics(),
            "training_config": self.config.to_dict(),
            "final_metrics": self._get_training_summary()
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(performance_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"数据集性能信息已保存到: {file_path}")
    
    def train(self) -> Dict[str, Any]:
        """开始训练（重写父类方法）
        
        Returns:
            训练结果字典
        """
        logger.info("[data] preparing multi-dataset training configuration")
        
        # 保存数据集配置
        config_path = self.output_dir / "dataset_config.json"
        self.dataset_manager.save_configuration(config_path)
        
        # 调用父类训练方法
        result = super().train()
        
        # 保存数据集性能信息
        self.save_dataset_performance()
        
        # 添加数据集特定的结果
        result["dataset_performance"] = self.get_dataset_performance_summary()
        result["dataset_statistics"] = self.dataset_manager.get_dataset_statistics()
        
        return result
    
    @classmethod
    def from_config(
        cls,
        model: nn.Module,
        dataset_config_path: Union[str, Path],
        training_config: Optional[TrainingConfig] = None,
        task_types: Optional[List[str]] = None,
        accelerator: Optional[Accelerator] = None
    ) -> 'MultiDatasetTrainer':
        """从配置文件创建训练器

        Args:
            model: 多任务模型（Florence2MultiTaskModel 或任何兼容 nn.Module 的模型）
            dataset_config_path: 数据集配置文件路径
            training_config: 训练配置
            task_types: 要训练的任务类型列表
            accelerator: Accelerate加速器

        Returns:
            多数据集训练器实例
        """
        # 加载数据集管理器
        dataset_manager = MultiDatasetManager.load_configuration(dataset_config_path)
        
        # 创建训练器
        trainer = cls(
            model=model,
            dataset_manager=dataset_manager,
            config=training_config,
            accelerator=accelerator,
            task_types=task_types
        )
        
        return trainer
