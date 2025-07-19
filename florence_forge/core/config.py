"""FlorenceForge配置模块

定义训练、模型和数据处理的各种配置
"""

import json
import yaml
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

@dataclass
class LoRAConfig:
    """LoRA配置"""
    r: int = 32
    lora_alpha: int = 32
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "target_modules": self.target_modules,
            "lora_dropout": self.lora_dropout,
            "bias": self.bias,
            "task_type": self.task_type
        }

@dataclass
class ModelConfig:
    """模型配置"""
    model_name: str = "microsoft/Florence-2-large"
    trust_remote_code: bool = True
    torch_dtype: str = "auto"
    device_map: str = "auto"
    attn_implementation: str = "flash_attention_2"
    
    # LoRA配置
    use_lora: bool = True
    lora_config: LoRAConfig = field(default_factory=LoRAConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "model_name": self.model_name,
            "trust_remote_code": self.trust_remote_code,
            "torch_dtype": self.torch_dtype,
            "device_map": self.device_map,
            "attn_implementation": self.attn_implementation,
            "use_lora": self.use_lora,
            "lora_config": self.lora_config.to_dict()
        }

@dataclass
class DataConfig:
    """数据配置"""
    batch_size: int = 4
    num_workers: int = 4
    pin_memory: bool = True
    shuffle: bool = True
    drop_last: bool = True
    
    # 数据增强
    use_augmentation: bool = False
    augmentation_prob: float = 0.5
    
    # 数据平衡
    use_balanced_sampling: bool = True
    max_samples_per_task: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "shuffle": self.shuffle,
            "drop_last": self.drop_last,
            "use_augmentation": self.use_augmentation,
            "augmentation_prob": self.augmentation_prob,
            "use_balanced_sampling": self.use_balanced_sampling,
            "max_samples_per_task": self.max_samples_per_task
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataConfig':
        """从字典创建实例"""
        config = cls()
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config

@dataclass
class OptimizationConfig:
    """优化器配置"""
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # 学习率调度
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    warmup_steps: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "adam_beta1": self.adam_beta1,
            "adam_beta2": self.adam_beta2,
            "adam_epsilon": self.adam_epsilon,
            "max_grad_norm": self.max_grad_norm,
            "lr_scheduler_type": self.lr_scheduler_type,
            "warmup_ratio": self.warmup_ratio,
            "warmup_steps": self.warmup_steps
        }

@dataclass
class TaskSchedulingConfig:
    """任务调度配置"""
    strategy: str = "round_robin"  # round_robin, weighted, curriculum
    temperature: float = 1.0
    update_frequency: int = 100
    
    # 课程学习配置
    curriculum_start_epoch: int = 0
    curriculum_end_epoch: int = 10
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "strategy": self.strategy,
            "temperature": self.temperature,
            "update_frequency": self.update_frequency,
            "curriculum_start_epoch": self.curriculum_start_epoch,
            "curriculum_end_epoch": self.curriculum_end_epoch
        }

@dataclass
class TrainingConfig:
    """完整的训练配置"""
    # 基础训练参数
    num_epochs: int = 10
    max_steps: Optional[int] = None
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100
    
    # 输出目录
    output_dir: str = "./outputs"
    logging_dir: Optional[str] = None
    
    # 设备配置
    device: str = "auto"  # auto, cpu, cuda, cuda:0, cuda:1, etc.
    
    # 混合精度训练
    use_fp16: bool = False
    use_bf16: bool = True
    
    # 梯度累积
    gradient_accumulation_steps: int = 1
    
    # 检查点
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # 早停
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.001
    
    # 任务配置
    tasks: List[str] = field(default_factory=lambda: ["CAPTION"])
    task_weights: Dict[str, float] = field(default_factory=dict)
    train_data_path: Optional[str] = None
    val_data_path: Optional[str] = None
    
    # 子配置
    model_config: ModelConfig = field(default_factory=ModelConfig)
    data_config: DataConfig = field(default_factory=DataConfig)
    optimization_config: OptimizationConfig = field(default_factory=OptimizationConfig)
    task_scheduling_config: TaskSchedulingConfig = field(default_factory=TaskSchedulingConfig)
    
    # 实验配置
    experiment_name: Optional[str] = None
    run_name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """后处理初始化"""
        if self.logging_dir is None:
            self.logging_dir = f"{self.output_dir}/logs"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "num_epochs": self.num_epochs,
            "max_steps": self.max_steps,
            "eval_steps": self.eval_steps,
            "save_steps": self.save_steps,
            "logging_steps": self.logging_steps,
            "output_dir": self.output_dir,
            "logging_dir": self.logging_dir,
            "device": self.device,
            "use_fp16": self.use_fp16,
            "use_bf16": self.use_bf16,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "save_total_limit": self.save_total_limit,
            "load_best_model_at_end": self.load_best_model_at_end,
            "metric_for_best_model": self.metric_for_best_model,
            "greater_is_better": self.greater_is_better,
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_threshold": self.early_stopping_threshold,
            "tasks": self.tasks,
            "task_weights": self.task_weights,
            "train_data_path": self.train_data_path,
            "val_data_path": self.val_data_path,
            "model_config": self.model_config.to_dict(),
            "data_config": self.data_config.to_dict(),
            "optimization_config": self.optimization_config.to_dict(),
            "task_scheduling_config": self.task_scheduling_config.to_dict(),
            "experiment_name": self.experiment_name,
            "run_name": self.run_name,
            "tags": self.tags
        }
    
    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """保存配置到文件
        
        Args:
            file_path: 文件路径
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 根据文件扩展名选择格式
        if file_path.suffix.lower() in ['.yaml', '.yml']:
            self.save_to_yaml(file_path)
        else:
            self.save_to_json(file_path)
    
    def save_to_json(self, file_path: Union[str, Path]) -> None:
        """保存配置到JSON文件
        
        Args:
            file_path: JSON文件路径
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    def save_to_yaml(self, file_path: Union[str, Path]) -> None:
        """保存配置到YAML文件
        
        Args:
            file_path: YAML文件路径
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = self.to_dict()
        # 添加元数据
        config_dict['_metadata'] = {
            'created_at': datetime.now().isoformat(),
            'config_version': '1.0',
            'description': 'Florence-2 Multi-task Training Configuration'
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, 
                     allow_unicode=True, indent=2, sort_keys=False)
    
    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> 'TrainingConfig':
        """从文件加载配置
        
        Args:
            file_path: 文件路径
            
        Returns:
            训练配置实例
        """
        file_path = Path(file_path)
        
        # 根据文件扩展名选择加载方式
        if file_path.suffix.lower() in ['.yaml', '.yml']:
            return cls.load_from_yaml(file_path)
        else:
            return cls.load_from_json(file_path)
    
    @classmethod
    def load_from_json(cls, file_path: Union[str, Path]) -> 'TrainingConfig':
        """从JSON文件加载配置
        
        Args:
            file_path: JSON文件路径
            
        Returns:
            训练配置实例
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def load_from_yaml(cls, file_path: Union[str, Path]) -> 'TrainingConfig':
        """从YAML文件加载配置
        
        Args:
            file_path: YAML文件路径
            
        Returns:
            训练配置实例
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        # 移除元数据（如果存在）
        config_dict.pop('_metadata', None)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """从字典创建配置
        
        Args:
            config_dict: 配置字典
            
        Returns:
            训练配置实例
        """
        # 提取子配置
        model_config_dict = config_dict.pop('model_config', {})
        data_config_dict = config_dict.pop('data_config', {})
        optimization_config_dict = config_dict.pop('optimization_config', {})
        task_scheduling_config_dict = config_dict.pop('task_scheduling_config', {})
        
        # 创建子配置实例
        model_config = ModelConfig(**model_config_dict)
        if 'lora_config' in model_config_dict:
            model_config.lora_config = LoRAConfig(**model_config_dict['lora_config'])
        
        data_config = DataConfig(**data_config_dict)
        optimization_config = OptimizationConfig(**optimization_config_dict)
        task_scheduling_config = TaskSchedulingConfig(**task_scheduling_config_dict)
        
        # 创建主配置实例
        return cls(
            model_config=model_config,
            data_config=data_config,
            optimization_config=optimization_config,
            task_scheduling_config=task_scheduling_config,
            **config_dict
        )


@dataclass
class EvaluationConfig:
    """评估配置类
    
    定义模型评估相关的参数
    """
    
    # 基础评估设置
    batch_size: int = 8
    num_workers: int = 4
    device: str = "auto"  # auto, cpu, cuda, mps
    
    # 评估数据设置
    eval_split: str = "test"  # train, val, test
    max_samples: Optional[int] = None  # 限制评估样本数量
    shuffle: bool = False
    
    # 指标计算设置
    metrics: List[str] = None  # 要计算的指标列表
    save_predictions: bool = True
    save_detailed_results: bool = False
    
    # 输出设置
    output_dir: str = "./evaluation_results"
    save_format: str = "json"  # json, csv, both
    
    # 可视化设置
    generate_plots: bool = True
    plot_format: str = "png"  # png, pdf, svg
    max_visualization_samples: int = 100
    
    # 任务特定设置
    task_configs: Dict[str, Dict[str, Any]] = None
    
    # 性能设置
    use_amp: bool = False  # 自动混合精度
    compile_model: bool = False  # torch.compile优化
    
    def __post_init__(self):
        """初始化后处理"""
        if self.metrics is None:
            self.metrics = ["accuracy", "precision", "recall", "f1"]
        
        if self.task_configs is None:
            self.task_configs = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典
        
        Returns:
            配置字典
        """
        return {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "device": self.device,
            "eval_split": self.eval_split,
            "max_samples": self.max_samples,
            "shuffle": self.shuffle,
            "metrics": self.metrics,
            "save_predictions": self.save_predictions,
            "save_detailed_results": self.save_detailed_results,
            "output_dir": self.output_dir,
            "save_format": self.save_format,
            "generate_plots": self.generate_plots,
            "plot_format": self.plot_format,
            "max_visualization_samples": self.max_visualization_samples,
            "task_configs": self.task_configs,
            "use_amp": self.use_amp,
            "compile_model": self.compile_model
        }
    
    def save_to_json(self, json_path: Union[str, Path]) -> None:
        """保存到JSON文件
        
        Args:
            json_path: JSON文件路径
        """
        import json
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    def save_to_yaml(self, yaml_path: Union[str, Path]) -> None:
        """保存到YAML文件
        
        Args:
            yaml_path: YAML文件路径
        """
        import yaml
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)
    
    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> 'EvaluationConfig':
        """从JSON文件加载配置
        
        Args:
            json_path: JSON文件路径
            
        Returns:
            评估配置实例
        """
        import json
        
        with open(json_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'EvaluationConfig':
        """从YAML文件加载配置
        
        Args:
            yaml_path: YAML文件路径
            
        Returns:
            评估配置实例
        """
        import yaml
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(**config_dict)