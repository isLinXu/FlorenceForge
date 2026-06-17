#!/usr/bin/env python3
"""Florence Forge - 多数据集多任务训练器"""

import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch.nn as nn

from ._accelerator_compat import Accelerator
from ..core.config import TrainingConfig
from ..data.multi_dataset_manager import MultiDatasetManager
from .gradient_validator import GradientValidationConfig, GradientValidator
from .memory_monitor import MemoryMonitor, MemoryMonitorConfig
from .scheduler import TaskScheduler
from .trainer import MultiTaskTrainer

logger = logging.getLogger(__name__)


class MultiDatasetTrainer(MultiTaskTrainer):
    """多数据集多任务训练器

    在 ``MultiTaskTrainer`` 基础上增加多数据集协调、性能跟踪与权重调整。
    """

    def __init__(
        self,
        model: nn.Module,
        dataset_manager: MultiDatasetManager,
        config: Optional[TrainingConfig] = None,
        accelerator: Optional[Accelerator] = None,
        task_types: Optional[List[str]] = None,
    ):
        self.dataset_manager = dataset_manager
        self.task_types = task_types
        self.config = config or TrainingConfig()

        train_dataset, val_dataset, _ = self._create_datasets()
        super().__init__(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=self.config,
            accelerator=accelerator,
        )

        self.dataset_performance = defaultdict(lambda: defaultdict(list))
        self.dataset_performance_history_limit = int(
            getattr(self.config, "dataset_performance_history_limit", 1000) or 0
        )
        self.dataset_weights: Dict[str, float] = {}
        self.current_dataset_strategy = "balanced"

        opt_settings = self.config.optimization_settings
        if hasattr(opt_settings, "max_grad_norm") and opt_settings.max_grad_norm > 0:
            self.gradient_validator = GradientValidator(
                self.model,
                GradientValidationConfig(
                    max_grad_norm_threshold=opt_settings.max_grad_norm * 1.5,
                    log_frequency=50,
                    stats_save_frequency=200,
                    monitor_layer_gradients=True,
                ),
            )

        mem_config = MemoryMonitorConfig(
            enable_monitoring=True,
            log_frequency=25,
            warning_threshold_percent=75.0,
            critical_threshold_percent=85.0,
            enable_gpu_monitoring=True,
            auto_cleanup=True,
            save_stats=True,
            enable_continuous_monitoring=False,
        )
        self.memory_monitor = MemoryMonitor(mem_config)
        logger.info("多数据集训练器初始化完成")

    def _create_datasets(self):
        logger.info("正在创建多数据集...")
        validation_result = self.dataset_manager.validate_configuration()
        if validation_result["errors"]:
            raise ValueError(f"数据集配置错误: {validation_result['errors']}")
        for warning in validation_result.get("warnings", []):
            logger.warning(warning)

        full_dataset = self.dataset_manager.create_unified_dataset(
            task_types=self.task_types,
            processor=getattr(self, "processor", None),
        )
        train_dataset, val_dataset, test_dataset = self.dataset_manager.create_balanced_split(
            val_ratio=getattr(self.config, "eval_ratio", 0.2),
            stratify_by_task=True,
            stratify_by_dataset=True,
            random_seed=getattr(self.config, "seed", None),
        )
        logger.info(
            "数据集创建完成 - 训练: %s, 验证: %s 样本",
            len(train_dataset),
            len(val_dataset),
        )
        return train_dataset, val_dataset, test_dataset

    def setup_training(self) -> None:
        super().setup_training()
        self._initialize_dataset_weights()
        self._setup_dataset_aware_scheduler()

    def _initialize_dataset_weights(self) -> None:
        dataset_stats = self.dataset_manager.get_dataset_statistics()
        for dataset_name, stats in dataset_stats["datasets"].items():
            if stats["loaded"]:
                sample_count = stats.get("total_samples", 1)
                priority = stats.get("priority", 1)
                self.dataset_weights[dataset_name] = priority / (sample_count ** 0.5)
        logger.info("数据集权重初始化: %s", self.dataset_weights)

    def _setup_dataset_aware_scheduler(self) -> None:
        if self.task_scheduler is None:
            return
        task_dataset_mapping = {
            task_type: mapping.datasets
            for task_type, mapping in self.dataset_manager.task_mappings.items()
        }
        if hasattr(self.task_scheduler, "set_dataset_mapping"):
            self.task_scheduler.set_dataset_mapping(task_dataset_mapping)

    def _record_dataset_performance(
        self, dataset_name: str, task_type: str, loss: float
    ) -> None:
        losses = self.dataset_performance[dataset_name][task_type]
        losses.append(loss)
        if (
            self.dataset_performance_history_limit > 0
            and len(losses) > self.dataset_performance_history_limit
        ):
            del losses[: -self.dataset_performance_history_limit]
        elif self.dataset_performance_history_limit == 0:
            losses.clear()

    def get_dataset_performance_summary(self) -> Dict[str, Any]:
        summary = {
            "dataset_weights": self.dataset_weights.copy(),
            "dataset_performance": {},
            "task_dataset_distribution": {},
        }
        for dataset_name, task_metrics in self.dataset_performance.items():
            dataset_summary = {}
            for task_type, losses in task_metrics.items():
                if losses:
                    dataset_summary[task_type] = {
                        "avg_loss": sum(losses) / len(losses),
                        "min_loss": min(losses),
                        "max_loss": max(losses),
                        "sample_count": len(losses),
                    }
            summary["dataset_performance"][dataset_name] = dataset_summary
        for task_type, mapping in self.dataset_manager.task_mappings.items():
            summary["task_dataset_distribution"][task_type] = {
                "datasets": mapping.datasets,
                "weights": mapping.weights,
                "sampling_strategy": mapping.sampling_strategy,
            }
        return summary

    def save_dataset_performance(self, file_path: Optional[Union[str, Path]] = None) -> None:
        if file_path is None:
            file_path = self.output_dir / "dataset_performance.json"
        performance_data = {
            "training_summary": self.get_dataset_performance_summary(),
            "dataset_manager_stats": self.dataset_manager.get_dataset_statistics(),
            "training_config": self.config.to_dict(),
            "final_metrics": self._get_training_summary(),
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(performance_data, f, indent=2, ensure_ascii=False)

    def train(self) -> Dict[str, Any]:
        config_path = self.output_dir / "dataset_config.json"
        self.dataset_manager.save_configuration(config_path)
        result = super().train()
        self.save_dataset_performance()
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
        accelerator: Optional[Accelerator] = None,
    ) -> "MultiDatasetTrainer":
        dataset_manager = MultiDatasetManager.load_configuration(dataset_config_path)
        return cls(
            model=model,
            dataset_manager=dataset_manager,
            config=training_config,
            accelerator=accelerator,
            task_types=task_types,
        )
