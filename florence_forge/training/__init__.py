"""FlorenceForge 训练模块导出入口。"""

from importlib import import_module

__all__ = [
    "MultiTaskTrainer",
    "TaskScheduler",
    "LoRAManager",
    "ModelMerger",
    "TrainingConfig",
    "ConfigManager",
    "create_default_config",
    "load_config_from_file",
    "CheckpointManager",
    "save_model_only",
    "load_model_only",
    "FSDPPlugin",
    "DeepSpeedPlugin",
    "DeviceConfigurator",
    "GradientCheckpointOptimizer",
    "ActivationRecomputePolicy",
    "AsyncCheckpointSaver",
    "TrainingLoop",
]

_LAZY_EXPORTS = {
    "MultiTaskTrainer": ("florence_forge.training.trainer", "MultiTaskTrainer"),
    "TaskScheduler": ("florence_forge.training.scheduler", "TaskScheduler"),
    "LoRAManager": ("florence_forge.training.lora_manager", "LoRAManager"),
    "ModelMerger": ("florence_forge.training.model_merger", "ModelMerger"),
    "TrainingConfig": ("florence_forge.training.config", "TrainingConfig"),
    "ConfigManager": ("florence_forge.cli.config_manager", "ConfigManager"),
    "create_default_config": ("florence_forge.training.config", "create_default_config"),
    "load_config_from_file": ("florence_forge.training.config", "load_config_from_file"),
    "CheckpointManager": ("florence_forge.training.checkpoint_manager", "CheckpointManager"),
    "save_model_only": ("florence_forge.training.checkpoint_manager", "save_model_only"),
    "load_model_only": ("florence_forge.training.checkpoint_manager", "load_model_only"),
    "FSDPPlugin": ("florence_forge.training.fsdp_plugin", "FSDPPlugin"),
    "DeepSpeedPlugin": ("florence_forge.training.deepspeed_plugin", "DeepSpeedPlugin"),
    "DeviceConfigurator": ("florence_forge.training.device_config", "DeviceConfigurator"),
    "GradientCheckpointOptimizer": (
        "florence_forge.training.gradient_checkpoint_optimizer",
        "GradientCheckpointOptimizer",
    ),
    "ActivationRecomputePolicy": (
        "florence_forge.training.gradient_checkpoint_optimizer",
        "ActivationRecomputePolicy",
    ),
    "AsyncCheckpointSaver": ("florence_forge.training.async_checkpoint", "AsyncCheckpointSaver"),
    "TrainingLoop": ("florence_forge.training.training_loop", "TrainingLoop"),
}


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        module = import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
