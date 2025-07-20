"""Training configuration management module.

This module provides configuration validation and management for training processes.
"""

import os
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Union, Any
from pathlib import Path

# 导入核心配置类
try:
    from .monitoring import MonitoringConfig
except ImportError:
    MonitoringConfig = None
try:
    from ..core.config import ModelConfig, DataConfig, OptimizationConfig, TaskSchedulingConfig
except ImportError:
    # 如果导入失败，创建简单的配置类
    @dataclass
    class ModelConfig:
        model_name: str = "microsoft/Florence-2-base"
        trust_remote_code: bool = True
        torch_dtype: str = "auto"
        device_map: str = "auto"
        attn_implementation: str = "eager"  # 使用默认的eager实现
        use_lora: bool = True
        
        def to_dict(self):
            return {
                "model_name": self.model_name,
                "trust_remote_code": self.trust_remote_code,
                "torch_dtype": self.torch_dtype,
                "device_map": self.device_map,
                "attn_implementation": self.attn_implementation,
                "use_lora": self.use_lora
            }
    
    @dataclass
    class DataConfig:
        batch_size: int = 8
        num_workers: int = 4
        
        def to_dict(self):
            return {
                "batch_size": self.batch_size,
                "num_workers": self.num_workers
            }
    
    @dataclass
    class OptimizationConfig:
        learning_rate: float = 1e-4
        weight_decay: float = 0.01
        
        def to_dict(self):
            return {
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay
            }
    
    @dataclass
    class TaskSchedulingConfig:
        strategy: str = "round_robin"
        temperature: float = 1.0
        update_frequency: int = 100
        curriculum_start_epoch: int = 0
        curriculum_end_epoch: int = 10
        
        def to_dict(self):
            return {
                "strategy": self.strategy,
                "temperature": self.temperature,
                "update_frequency": self.update_frequency,
                "curriculum_start_epoch": self.curriculum_start_epoch,
                "curriculum_end_epoch": self.curriculum_end_epoch
            }

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration class with validation."""
    
    # Model configuration
    model_name: str = "microsoft/Florence-2-base"
    model_revision: Optional[str] = None
    
    # Training parameters
    learning_rate: float = 1e-4
    batch_size: int = 4
    num_epochs: int = 3
    max_steps: Optional[int] = None
    warmup_steps: int = 500
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Data configuration
    train_data_path: str = ""
    val_data_path: str = ""
    max_seq_length: int = 1024
    image_size: int = 768
    
    # Task configuration
    tasks: List[str] = field(default_factory=lambda: ["caption"])
    task_weights: Dict[str, float] = field(default_factory=dict)
    
    # Output configuration
    output_dir: str = "./outputs"
    logging_dir: Optional[str] = None
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100
    save_total_limit: int = 3
    
    # Hardware configuration
    device: str = "auto"
    mixed_precision: bool = True
    use_bf16: bool = False
    use_fp16: bool = True
    dataloader_num_workers: int = 4
    
    # Optimization configuration
    optimizer: str = "adamw"
    scheduler: str = "linear"
    
    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    greater_is_better: bool = False
    metric_for_best_model: str = "eval_loss"
    
    # 监控配置
    enable_wandb: bool = False
    enable_swanlab: bool = False
    enable_tensorboard: bool = True
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    swanlab_project: Optional[str] = None
    swanlab_experiment_name: Optional[str] = None
    
    # LoRA配置
    use_lora: bool = False
    lora_config: Optional[Dict[str, Any]] = None
    save_merged_model: bool = False  # 是否保存合并后的模型
    merge_strategy: str = "linear"  # 合并策略: linear, weighted
    export_formats: List[str] = field(default_factory=lambda: ["pytorch"])  # 导出格式
    
    # 配置对象（在 __post_init__ 中创建）
    model_config: Optional[ModelConfig] = field(default=None, init=False)
    data_config: Optional[DataConfig] = field(default=None, init=False)
    optimization_config: Optional[OptimizationConfig] = field(default=None, init=False)
    monitoring_config: Optional['MonitoringConfig'] = field(default=None, init=False)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # 创建子配置对象
        self.model_config = ModelConfig(
            model_name=self.model_name,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation="eager",  # 使用默认的eager实现，避免flash_attn依赖
            use_lora=True
        )
        
        self.data_config = DataConfig(
            batch_size=self.batch_size,
            num_workers=self.dataloader_num_workers
        )
        
        self.optimization_config = OptimizationConfig(
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        self.task_scheduling_config = TaskSchedulingConfig(
            strategy="round_robin",
            temperature=1.0,
            update_frequency=100
        )
        
        # 创建监控配置对象
        if MonitoringConfig is not None:
            self.monitoring_config = MonitoringConfig(
                enable_wandb=self.enable_wandb,
                enable_swanlab=self.enable_swanlab,
                enable_tensorboard=self.enable_tensorboard,
                wandb_project=self.wandb_project,
                wandb_entity=self.wandb_entity,
                wandb_run_name=self.wandb_run_name,
                swanlab_project=self.swanlab_project,
                swanlab_experiment_name=self.swanlab_experiment_name,
                tensorboard_log_dir=self.logging_dir,
                log_frequency=self.logging_steps
            )
        
        self.validate()
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        # Validate learning rate
        if not 0 < self.learning_rate < 1:
            raise ValueError(f"Learning rate must be between 0 and 1, got {self.learning_rate}")
        
        # Validate batch size
        if self.batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {self.batch_size}")
        
        # Validate epochs
        if self.num_epochs <= 0:
            raise ValueError(f"Number of epochs must be positive, got {self.num_epochs}")
        
        # Validate data paths
        if self.train_data_path and not os.path.exists(self.train_data_path):
            logger.warning(f"Training data path does not exist: {self.train_data_path}")
        
        if self.val_data_path and not os.path.exists(self.val_data_path):
            logger.warning(f"Validation data path does not exist: {self.val_data_path}")
        
        # Validate tasks
        valid_tasks = {"caption", "detection", "ocr", "segmentation"}
        for task in self.tasks:
            if task not in valid_tasks:
                raise ValueError(f"Invalid task '{task}'. Valid tasks: {valid_tasks}")
        
        # Validate task weights
        if self.task_weights:
            for task, weight in self.task_weights.items():
                if task not in self.tasks:
                    logger.warning(f"Task weight specified for task '{task}' not in task list")
                if weight <= 0:
                    raise ValueError(
                        f"Task weight must be positive, "
                        f"got {weight} for task '{task}'"
                    )
        
        # Set default task weights if not provided
        if not self.task_weights:
            self.task_weights = {task: 1.0 for task in self.tasks}
        
        # Validate optimizer
        valid_optimizers = {"adamw", "adam", "sgd"}
        if self.optimizer not in valid_optimizers:
            raise ValueError(f"Invalid optimizer '{self.optimizer}'. Valid optimizers: {valid_optimizers}")
        
        # Validate scheduler
        valid_schedulers = {"linear", "cosine", "constant", "polynomial"}
        if self.scheduler not in valid_schedulers:
            raise ValueError(f"Invalid scheduler '{self.scheduler}'. Valid schedulers: {valid_schedulers}")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info("Configuration validation completed successfully")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        config_dict = {
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "max_steps": self.max_steps,
            "warmup_steps": self.warmup_steps,
            "weight_decay": self.weight_decay,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_grad_norm": self.max_grad_norm,
            "train_data_path": self.train_data_path,
            "val_data_path": self.val_data_path,
            "max_seq_length": self.max_seq_length,
            "image_size": self.image_size,
            "tasks": self.tasks,
            "task_weights": self.task_weights,
            "output_dir": self.output_dir,
            "logging_dir": self.logging_dir,
            "save_steps": self.save_steps,
            "logging_steps": self.logging_steps,
            "eval_steps": self.eval_steps,
            "save_total_limit": self.save_total_limit,
            "device": self.device,
            "mixed_precision": self.mixed_precision,
            "use_bf16": self.use_bf16,
            "use_fp16": self.use_fp16,
            "dataloader_num_workers": self.dataloader_num_workers,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_threshold": self.early_stopping_threshold,
            "greater_is_better": self.greater_is_better
        }
        
        # 添加子配置对象
        if self.model_config:
            config_dict["model_config"] = self.model_config.to_dict()
        if self.data_config:
            config_dict["data_config"] = self.data_config.to_dict()
        if self.optimization_config:
            config_dict["optimization_config"] = self.optimization_config.to_dict()
        if hasattr(self, 'task_scheduling_config') and self.task_scheduling_config:
            config_dict["task_scheduling_config"] = self.task_scheduling_config.to_dict() if hasattr(self.task_scheduling_config, 'to_dict') else self.task_scheduling_config.__dict__
            
        return config_dict
    
    def save_to_file(self, file_path: str) -> None:
        """保存配置到文件"""
        import json
        
        config_dict = self.to_dict()
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create configuration from dictionary."""
        # 获取TrainingConfig支持的参数
        import inspect
        signature = inspect.signature(cls.__init__)
        valid_params = set(signature.parameters.keys()) - {'self'}
        
        # 过滤配置字典，只保留支持的参数
        filtered_config = {k: v for k, v in config_dict.items() if k in valid_params}
        
        # 记录被过滤掉的参数
        filtered_out = set(config_dict.keys()) - valid_params
        if filtered_out:
            logger.warning(f"忽略不支持的配置参数: {filtered_out}")
        
        return cls(**filtered_config)
    
    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> 'TrainingConfig':
        """Load configuration from JSON file."""
        with open(json_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def save_json(self, json_path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Configuration saved to {json_path}")
    
    def update(self, **kwargs) -> None:
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Unknown configuration parameter: {key}")
        
        # Re-validate after update
        self.validate()
    
    def get_effective_batch_size(self) -> int:
        """Get effective batch size considering gradient accumulation."""
        return self.batch_size * self.gradient_accumulation_steps
    
    def get_total_steps(self, num_samples: int) -> int:
        """Calculate total training steps."""
        steps_per_epoch = num_samples // self.get_effective_batch_size()
        return steps_per_epoch * self.num_epochs


class ConfigManager:
    """Configuration manager for training processes."""
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        """Initialize configuration manager."""
        self.config = config or TrainingConfig()
        self.config_history: List[Dict[str, Any]] = []
    
    def load_config(self, config_path: Union[str, Path]) -> None:
        """Load configuration from file."""
        try:
            self.config = TrainingConfig.from_json(config_path)
            logger.info(f"Configuration loaded from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
            raise
    
    def save_config(self, config_path: Union[str, Path]) -> None:
        """Save current configuration to file."""
        try:
            self.config.save_json(config_path)
        except Exception as e:
            logger.error(f"Failed to save configuration to {config_path}: {e}")
            raise
    
    def update_config(self, **kwargs) -> None:
        """Update configuration and save to history."""
        # Save current config to history
        self.config_history.append(self.config.to_dict())
        
        # Update configuration
        self.config.update(**kwargs)
        
        logger.info(f"Configuration updated with {len(kwargs)} parameters")
    
    def rollback_config(self) -> bool:
        """Rollback to previous configuration."""
        if not self.config_history:
            logger.warning("No configuration history available for rollback")
            return False
        
        previous_config = self.config_history.pop()
        self.config = TrainingConfig.from_dict(previous_config)
        logger.info("Configuration rolled back to previous state")
        return True
    
    def validate_config(self) -> bool:
        """Validate current configuration."""
        try:
            self.config.validate()
            return True
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary."""
        return {
            "model": self.config.model_name,
            "tasks": self.config.tasks,
            "learning_rate": self.config.learning_rate,
            "batch_size": self.config.batch_size,
            "effective_batch_size": self.config.get_effective_batch_size(),
            "num_epochs": self.config.num_epochs,
            "output_dir": self.config.output_dir,
            "mixed_precision": self.config.mixed_precision
        }


def create_default_config() -> TrainingConfig:
    """Create default training configuration."""
    return TrainingConfig()


def load_config_from_file(config_path: Union[str, Path]) -> TrainingConfig:
    """Load training configuration from file."""
    config_path = Path(config_path)
    
    if config_path.suffix.lower() in ['.yaml', '.yml']:
        # 加载YAML配置文件
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return TrainingConfig.from_dict(config_dict)
    elif config_path.suffix.lower() == '.json':
        # 加载JSON配置文件
        return TrainingConfig.from_json(config_path)
    else:
        # 默认尝试YAML格式
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return TrainingConfig.from_dict(config_dict)


def validate_config_file(config_path: Union[str, Path]) -> bool:
    """Validate configuration file."""
    try:
        config = TrainingConfig.from_json(config_path)
        config.validate()
        return True
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return False