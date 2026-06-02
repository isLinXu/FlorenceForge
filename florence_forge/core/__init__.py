"""FlorenceForge 核心模块导出入口。"""

from importlib import import_module

from .config import (
    LoRAConfig,
    ModelConfig,
    DataConfig,
    OptimizationConfig,
    TaskSchedulingConfig,
    TrainingConfig,
    EvaluationConfig,
    DistributedConfig,
)
from .tasks import FLORENCE2_TASKS

__all__ = [
    "Florence2MultiTaskModel",
    "LoRAConfig",
    "ModelConfig",
    "DataConfig",
    "OptimizationConfig",
    "TaskSchedulingConfig",
    "TrainingConfig",
    "EvaluationConfig",
    "DistributedConfig",
    "FLORENCE2_TASKS",
]

_LAZY_EXPORTS = {
    "Florence2MultiTaskModel": ("florence_forge.core.model", "Florence2MultiTaskModel"),
}


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        module = import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
