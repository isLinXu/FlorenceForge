"""FlorenceForge核心模块

包含模型、配置和任务定义等核心组件
"""

from .model import Florence2MultiTaskModel
from .config import (
    LoRAConfig,
    ModelConfig,
    DataConfig,
    OptimizationConfig,
    TaskSchedulingConfig,
    TrainingConfig,
    EvaluationConfig
)
from .tasks import FLORENCE2_TASKS

__all__ = [
    'Florence2MultiTaskModel',
    'LoRAConfig',
    'ModelConfig',
    'DataConfig',
    'OptimizationConfig',
    'TaskSchedulingConfig',
    'TrainingConfig',
    'EvaluationConfig',
    'FLORENCE2_TASKS'
]