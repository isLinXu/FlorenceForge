"""FlorenceForge 训练模块导出入口。

并存说明（2026-05-21）
---------------------
仓库内有 **两套训练栈**：

- **v1（默认导出）**：`trainer.py` 单文件主流版 → `MultiTaskTrainer`
- **v2（模块化）**：`trainer_refactored.py` + `training_loop.py` + `checkpoint_manager.py`
  由 `tests/test_training_integration.py` 验证，可通过 ``MultiTaskTrainerV2`` /
  ``TrainerV2`` 显式导入：

    >>> from florence_forge.training import MultiTaskTrainerV2
    >>> from florence_forge.training.training_loop import TrainingLoop
    >>> from florence_forge.training.checkpoint_manager import CheckpointManager

详情参见各文件顶部 docstring。

迁移计划（v1 → v2 的正式弃用时间线）见 ``docs/v1_v2_Migration_Timeline.md``。
"""

from importlib import import_module

__all__ = [
    "MultiTaskTrainer",
    "MultiTaskTrainerV2",
    "TrainerV2",
    "TaskScheduler",
    "LoRAManager",
    "ModelMerger",
    "TrainingConfig",
    "ConfigManager",
    "create_default_config",
    "load_config_from_file",
    "CheckpointManager",
    "create_checkpoint_manager",
    "save_model_only",
    "load_model_only",
]

# v1 默认导出（main path）
_LAZY_EXPORTS = {
    "MultiTaskTrainer": ("florence_forge.training.trainer", "MultiTaskTrainer"),
    "MultiTaskTrainerV2": ("florence_forge.training.trainer_refactored", "MultiTaskTrainer"),
    "TrainerV2": ("florence_forge.training.trainer_refactored", "MultiTaskTrainer"),
    "TaskScheduler": ("florence_forge.training.scheduler", "TaskScheduler"),
    "LoRAManager": ("florence_forge.training.lora_manager", "LoRAManager"),
    "ModelMerger": ("florence_forge.training.model_merger", "ModelMerger"),
    "TrainingConfig": ("florence_forge.training.config", "TrainingConfig"),
    "ConfigManager": ("florence_forge.cli.config_manager", "ConfigManager"),
    "create_default_config": ("florence_forge.training.config", "create_default_config"),
    "load_config_from_file": ("florence_forge.training.config", "load_config_from_file"),
    # 注意：CheckpointManager 默认指向 v2（OO 生命周期版，供 trainer_refactored 使用）。
    # v1 同名 class 仍可通过 `from florence_forge.training.checkpoint import CheckpointManager` 显式获取。
    "CheckpointManager": ("florence_forge.training.checkpoint_manager", "CheckpointManager"),
    "create_checkpoint_manager": ("florence_forge.training.checkpoint", "create_checkpoint_manager"),
    "save_model_only": ("florence_forge.training.checkpoint", "save_model_only"),
    "load_model_only": ("florence_forge.training.checkpoint", "load_model_only"),
}


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        module = import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
