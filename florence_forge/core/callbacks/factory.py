"""Convenience factory for building the default callback list from a config."""

from __future__ import annotations

from typing import List

from .base import TrainerCallback
from .builtin import (
    CheckpointCallback,
    EarlyStoppingCallback,
    LoggingCallback,
    TensorBoardCallback,
)
from .integrations import MoECallback, MonitoringCallback


def create_default_callbacks(config, monitor=None) -> List[TrainerCallback]:
    """根据 TrainingConfig 创建默认 Callback 列表

    Args:
        config: 训练配置
        monitor: 可选的 TrainingMonitor 实例。如果提供，将创建
                 MonitoringCallback 替代独立的 TensorBoardCallback。
    """
    callbacks: List[TrainerCallback] = []

    # 日志
    if hasattr(config, "logging_steps"):
        callbacks.append(LoggingCallback(logging_steps=config.logging_steps))

    # 检查点
    if hasattr(config, "save_steps") and hasattr(config, "save_total_limit"):
        callbacks.append(
            CheckpointCallback(
                save_steps=config.save_steps,
                save_total_limit=config.save_total_limit,
            )
        )

    # 早停（如果配置了）
    if getattr(config, "early_stopping_patience", 0) > 0:
        callbacks.append(
            EarlyStoppingCallback(
                monitor=config.metric_for_best_model,
                patience=config.early_stopping_patience,
                threshold=config.early_stopping_threshold,
                mode="min" if not config.greater_is_better else "max",
            )
        )

    # 监控——统一入口（WandB/SwanLab/TensorBoard）
    log_frequency = getattr(config, "logging_steps", 10)
    if monitor is not None:
        # 使用 MonitoringCallback 统一管理所有监控工具
        callbacks.append(MonitoringCallback(
            monitor=monitor,
            log_frequency=log_frequency,
            log_gradients=getattr(config, "log_gradients", False),
        ))
    elif getattr(config, "use_tensorboard", True):
        # 向后兼容：没有 monitor 时使用独立的 TensorBoardCallback
        callbacks.append(TensorBoardCallback())

    # MoE 诊断回调
    if getattr(config, "use_moe", False):
        callbacks.append(MoECallback())

    return callbacks
