"""FlorenceForge 训练模块导出入口。

v2.0.0 起仅保留模块化训练栈（``trainer_refactored`` + ``training_loop`` + ``checkpoint_manager``）。

    >>> from florence_forge.training import MultiTaskTrainer

历史 v1 ``trainer.py`` 已移除；请使用 ``MultiTaskTrainer``（v2）。
迁移说明见 ``docs/v1_v2_Migration_Timeline.md``。
"""

from importlib import import_module

__all__ = [
    "MultiTaskTrainer",
    "MultiTaskTrainerV2",
    "Trainer",
    "TrainerV2",
    "TaskScheduler",
    "LoRAManager",
    "ModelMerger",
    "TrainingConfig",
    "ConfigManager",
    "create_default_config",
    "load_config_from_file",
    "CheckpointManager",
    "DirectoryCheckpointManager",
    "create_checkpoint_manager",
    "save_model_only",
    "load_model_only",
]

_LAZY_EXPORTS = {
    "MultiTaskTrainer": ("florence_forge.training.trainer_refactored", "MultiTaskTrainer"),
    "MultiTaskTrainerV2": ("florence_forge.training.trainer_refactored", "MultiTaskTrainer"),
    "Trainer": ("florence_forge.training.trainer_refactored", "MultiTaskTrainer"),
    "TrainerV2": ("florence_forge.training.trainer_refactored", "MultiTaskTrainer"),
    "TaskScheduler": ("florence_forge.training.scheduler", "TaskScheduler"),
    "LoRAManager": ("florence_forge.training.lora_manager", "LoRAManager"),
    "ModelMerger": ("florence_forge.training.model_merger", "ModelMerger"),
    "TrainingConfig": ("florence_forge.training.config", "TrainingConfig"),
    "ConfigManager": ("florence_forge.cli.config_manager", "ConfigManager"),
    "create_default_config": ("florence_forge.training.config", "create_default_config"),
    "load_config_from_file": ("florence_forge.training.config", "load_config_from_file"),
    "CheckpointManager": ("florence_forge.training.checkpoint_manager", "CheckpointManager"),
    "DirectoryCheckpointManager": (
        "florence_forge.training.checkpoint",
        "DirectoryCheckpointManager",
    ),
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
    if name in {"MultiTaskTrainerV1", "TrainerV1"}:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}; "
            "v1 trainer.py was removed in v2.0.0 — use MultiTaskTrainer instead."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
