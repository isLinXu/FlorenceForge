"""
FlorenceForge 统一 Callback 系统

为训练器/评估器提供生命周期钩子，支持：
- 日志/监控（TensorBoard、WandB、打印）
- 检查点策略（按 step/epoch 保存、最大保留数）
- 早停
- 梯度裁剪/监控
- MoE 路由诊断
- 自定义扩展

设计参考：HuggingFace TrainerCallback + Keras Callback。

历史上所有实现集中在单个 ``core/callbacks.py`` 文件中；现已按职责拆分为子包
（``base`` / ``builtin`` / ``integrations`` / ``factory``）。为保持向后兼容，
所有公共符号仍从 ``florence_forge.core.callbacks`` 直接导出。
"""

from __future__ import annotations

from .base import CallbackManager, TrainerCallback
from .builtin import (
    CheckpointCallback,
    EarlyStoppingCallback,
    GradientClipCallback,
    LoggingCallback,
    TensorBoardCallback,
)
from .factory import create_default_callbacks
from .integrations import MoECallback, MonitoringCallback

__all__ = [
    "TrainerCallback",
    "CallbackManager",
    "LoggingCallback",
    "CheckpointCallback",
    "EarlyStoppingCallback",
    "GradientClipCallback",
    "TensorBoardCallback",
    "MoECallback",
    "MonitoringCallback",
    "create_default_callbacks",
]
