"""FlorenceForge YAML配置系统

提供统一的YAML配置文件支持，整合训练配置、多数据集配置和多任务配置
"""

import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime

from .config import TrainingConfig, DataConfig, ModelConfig, LoRAConfig, OptimizationConfig, TaskSchedulingConfig
from .tasks import validate_task_name, list_all_tasks
from ..data.multi_dataset_manager import MultiDatasetManager, DatasetInfo, TaskDatasetMapping

logger = logging.getLogger(__name__)


@dataclass
class YAMLDatasetConfig:
    """YAML数据集配置"""
    name: str
    path: str
    task_types: List[str]
    format: str = "custom"
    weight: float = 1.0
    priority: int = 1
    max_samples: Optional[int] = None
    preprocessing: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dataset_info(self) -> DatasetInfo:
        """转换为DatasetInfo"""
        return DatasetInfo(
            name=self.name,
            path=self.path,
            task_types=self.task_types,
            format=self.format,
            weight=self.weight,
            priority=self.priority,
            max_samples=self.max_samples,
            preprocessing=self.preprocessing,
            metadata=self.metadata
        )
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'YAMLDatasetConfig':
        """从字典创建实例"""
        return cls(
            name=data["name"],
            path=data["path"],
            task_types=data["task_types"],
            format=data.get("format", "custom"),
            weight=data.get("weight", 1.0),
            priority=data.get("priority", 1),
            max_samples=data.get("max_samples"),
            preprocessing=data.get("preprocessing", {}),
            metadata=data.get("metadata", {})
        )


@dataclass
class YAMLTaskMapping:
    """YAML任务映射配置"""
    task_type: str
    datasets: List[str]
    weights: Dict[str, float] = field(default_factory=dict)
    sampling_strategy: str = "balanced"
    
    def to_task_dataset_mapping(self) -> TaskDatasetMapping:
        """转换为TaskDatasetMapping"""
        # 如果没有指定权重，设置为均等权重
        if not self.weights:
            self.weights = {dataset: 1.0 for dataset in self.datasets}
        
        return TaskDatasetMapping(
            task_type=self.task_type,
            datasets=self.datasets,
            weights=self.weights,
            sampling_strategy=self.sampling_strategy
        )
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'YAMLTaskMapping':
        """从字典创建实例"""
        return cls(
            task_type=data["task_type"],
            datasets=data["datasets"],
            weights=data.get("weights", {}),
            sampling_strategy=data.get("sampling_strategy", "balanced")
        )


@dataclass
class FlorenceForgeYAMLConfig:
    """FlorenceForge完整YAML配置"""
    
    # 元数据
    config_version: str = "1.0"
    description: str = "Florence-2 Multi-task Training Configuration"
    created_at: Optional[str] = None
    
    # 项目配置
    project_name: str = "florence_forge_training"
    experiment_name: Optional[str] = None
    output_dir: str = "./outputs"
    image_base_path: str = ""
    
    # 训练配置
    training: Optional[Dict[str, Any]] = None
    
    # 数据集配置
    datasets: List[YAMLDatasetConfig] = field(default_factory=list)
    
    # 任务映射配置
    task_mappings: List[YAMLTaskMapping] = field(default_factory=list)
    
    # 启用的任务类型
    enabled_tasks: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """初始化后处理"""
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        
        # 如果没有指定启用的任务，从任务映射中推断
        if not self.enabled_tasks and self.task_mappings:
            self.enabled_tasks = [mapping.task_type for mapping in self.task_mappings]
    
    def validate(self) -> Dict[str, List[str]]:
        """验证配置
        
        Returns:
            包含错误和警告的字典
        """
        errors = []
        warnings = []
        
        # 验证任务类型
        all_valid_tasks = list_all_tasks()
        for task_type in self.enabled_tasks:
            if not validate_task_name(task_type):
                errors.append(f"无效的任务类型: {task_type}. 支持的任务类型: {all_valid_tasks}")
        
        # 验证数据集配置
        dataset_names = set()
        for dataset in self.datasets:
            if dataset.name in dataset_names:
                errors.append(f"重复的数据集名称: {dataset.name}")
            dataset_names.add(dataset.name)
            
            # 验证数据集的任务类型
            for task_type in dataset.task_types:
                if not validate_task_name(task_type):
                    errors.append(f"数据集 {dataset.name} 包含无效的任务类型: {task_type}")
            
            # 检查路径是否存在
            if not Path(dataset.path).exists():
                warnings.append(f"数据集路径不存在: {dataset.path}")
        
        # 验证任务映射
        mapped_tasks = set()
        for mapping in self.task_mappings:
            if mapping.task_type in mapped_tasks:
                errors.append(f"重复的任务映射: {mapping.task_type}")
            mapped_tasks.add(mapping.task_type)
            
            # 验证映射的数据集是否存在
            for dataset_name in mapping.datasets:
                if dataset_name not in dataset_names:
                    errors.append(f"任务映射 {mapping.task_type} 引用了不存在的数据集: {dataset_name}")
            
            # 验证数据集是否支持该任务类型
            for dataset in self.datasets:
                if dataset.name in mapping.datasets:
                    if mapping.task_type not in dataset.task_types:
                        errors.append(
                            f"数据集 {dataset.name} 不支持任务类型 {mapping.task_type}"
                        )
        
        # 检查启用的任务是否都有映射
        for task_type in self.enabled_tasks:
            if task_type not in mapped_tasks:
                warnings.append(f"启用的任务类型 {task_type} 没有对应的数据集映射")
        
        return {"errors": errors, "warnings": warnings}
    
    def to_training_config(self) -> TrainingConfig:
        """转换为TrainingConfig
        
        Returns:
            训练配置实例
        """
        if self.training is None:
            # 使用默认配置
            config = TrainingConfig()
        else:
            config = TrainingConfig.from_dict(self.training)
        
        # 更新输出目录和实验名称
        config.output_dir = self.output_dir
        if self.experiment_name:
            config.experiment_name = self.experiment_name
        
        return config
    
    def to_multi_dataset_manager(self) -> MultiDatasetManager:
        """转换为MultiDatasetManager
        
        Returns:
            多数据集管理器实例
        """
        # 创建数据配置
        if self.training and "data_config" in self.training:
            data_config = DataConfig.from_dict(self.training["data_config"])
        else:
            data_config = DataConfig()
        
        # 创建管理器
        manager = MultiDatasetManager(
            image_base_path=self.image_base_path,
            config=data_config
        )
        
        # 注册数据集
        for dataset_config in self.datasets:
            manager.register_dataset(dataset_config.to_dataset_info())
        
        # 设置任务映射
        for mapping_config in self.task_mappings:
            mapping = mapping_config.to_task_dataset_mapping()
            manager.map_task_to_datasets(
                task_type=mapping.task_type,
                dataset_names=mapping.datasets,
                weights=mapping.weights,
                sampling_strategy=mapping.sampling_strategy
            )
        
        return manager
    
    def save_to_yaml(self, file_path: Union[str, Path]) -> None:
        """保存到YAML文件
        
        Args:
            file_path: YAML文件路径
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 转换为字典
        config_dict = {
            "config_version": self.config_version,
            "description": self.description,
            "created_at": self.created_at,
            "project_name": self.project_name,
            "experiment_name": self.experiment_name,
            "output_dir": self.output_dir,
            "image_base_path": self.image_base_path,
            "enabled_tasks": self.enabled_tasks,
            "datasets": [
                {
                    "name": ds.name,
                    "path": ds.path,
                    "task_types": ds.task_types,
                    "format": ds.format,
                    "weight": ds.weight,
                    "priority": ds.priority,
                    "max_samples": ds.max_samples,
                    "preprocessing": ds.preprocessing,
                    "metadata": ds.metadata
                }
                for ds in self.datasets
            ],
            "task_mappings": [
                {
                    "task_type": tm.task_type,
                    "datasets": tm.datasets,
                    "weights": tm.weights,
                    "sampling_strategy": tm.sampling_strategy
                }
                for tm in self.task_mappings
            ]
        }
        
        # 添加训练配置
        if self.training:
            config_dict["training"] = self.training
        
        # 保存到YAML文件
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, 
                     allow_unicode=True, indent=2, sort_keys=False)
    
    @classmethod
    def load_from_yaml(cls, file_path: Union[str, Path]) -> 'FlorenceForgeYAMLConfig':
        """从YAML文件加载配置
        
        Args:
            file_path: YAML文件路径
            
        Returns:
            配置实例
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> 'FlorenceForgeYAMLConfig':
        """从文件加载配置（支持YAML和JSON）
        
        Args:
            file_path: 配置文件路径
            
        Returns:
            配置实例
        """
        file_path = Path(file_path)
        
        if file_path.suffix.lower() in ['.yaml', '.yml']:
            return cls.load_from_yaml(file_path)
        elif file_path.suffix.lower() == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            return cls.from_dict(config_dict)
        else:
            raise ValueError(f"不支持的文件格式: {file_path.suffix}")
    
    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """保存到文件（根据扩展名自动选择格式）
        
        Args:
            file_path: 输出文件路径
        """
        file_path = Path(file_path)
        
        if file_path.suffix.lower() in ['.yaml', '.yml']:
            self.save_to_yaml(file_path)
        elif file_path.suffix.lower() == '.json':
            self.save_to_json(file_path)
        else:
            # 默认保存为YAML
            file_path = file_path.with_suffix('.yaml')
            self.save_to_yaml(file_path)
    
    def save_to_json(self, file_path: Union[str, Path]) -> None:
        """保存到JSON文件
        
        Args:
            file_path: JSON文件路径
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 转换为字典
        config_dict = self.to_dict()
        
        # 保存到JSON文件
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典
        
        Returns:
            配置字典
        """
        config_dict = {
            "config_version": self.config_version,
            "description": self.description,
            "created_at": self.created_at,
            "project_name": self.project_name,
            "experiment_name": self.experiment_name,
            "output_dir": self.output_dir,
            "image_base_path": self.image_base_path,
            "enabled_tasks": self.enabled_tasks,
            "datasets": [
                {
                    "name": ds.name,
                    "path": ds.path,
                    "task_types": ds.task_types,
                    "format": ds.format,
                    "weight": ds.weight,
                    "priority": ds.priority,
                    "max_samples": ds.max_samples,
                    "preprocessing": ds.preprocessing,
                    "metadata": ds.metadata
                }
                for ds in self.datasets
            ],
            "task_mappings": [
                {
                    "task_type": tm.task_type,
                    "datasets": tm.datasets,
                    "weights": tm.weights,
                    "sampling_strategy": tm.sampling_strategy
                }
                for tm in self.task_mappings
            ]
        }
        
        # 添加训练配置
        if self.training:
            config_dict["training"] = self.training
        
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'FlorenceForgeYAMLConfig':
        """从字典创建配置
        
        Args:
            config_dict: 配置字典
            
        Returns:
            配置实例
        """
        # 解析数据集配置
        datasets = [
            YAMLDatasetConfig.from_dict(ds_dict)
            for ds_dict in config_dict.get("datasets", [])
        ]
        
        # 解析任务映射配置
        task_mappings = [
            YAMLTaskMapping.from_dict(tm_dict)
            for tm_dict in config_dict.get("task_mappings", [])
        ]
        
        return cls(
            config_version=config_dict.get("config_version", "1.0"),
            description=config_dict.get("description", "Florence-2 Multi-task Training Configuration"),
            created_at=config_dict.get("created_at"),
            project_name=config_dict.get("project_name", "florence_forge_training"),
            experiment_name=config_dict.get("experiment_name"),
            output_dir=config_dict.get("output_dir", "./outputs"),
            image_base_path=config_dict.get("image_base_path", ""),
            enabled_tasks=config_dict.get("enabled_tasks", []),
            datasets=datasets,
            task_mappings=task_mappings,
            training=config_dict.get("training")
        )
    
    @classmethod
    def create_example_config(cls) -> 'FlorenceForgeYAMLConfig':
        """创建示例配置
        
        Returns:
            示例配置实例
        """
        return cls(
            project_name="florence2_multitask_example",
            experiment_name="caption_and_detection",
            output_dir="./outputs/example_training",
            image_base_path="/path/to/images",
            enabled_tasks=["CAPTION", "OD"],
            datasets=[
                YAMLDatasetConfig(
                    name="coco_captions",
                    path="/path/to/coco/captions",
                    task_types=["CAPTION"],
                    format="coco",
                    weight=1.0,
                    priority=1,
                    max_samples=50000,
                    preprocessing={
                        "resize": {"height": 384, "width": 384},
                        "normalize": True
                    },
                    metadata={
                        "description": "COCO图像标题数据集",
                        "version": "2017"
                    }
                ),
                YAMLDatasetConfig(
                    name="coco_detection",
                    path="/path/to/coco/detection",
                    task_types=["OD"],
                    format="coco",
                    weight=1.0,
                    priority=1,
                    max_samples=30000,
                    preprocessing={
                        "resize": {"height": 384, "width": 384},
                        "normalize": True
                    },
                    metadata={
                        "description": "COCO目标检测数据集",
                        "version": "2017"
                    }
                )
            ],
            task_mappings=[
                YAMLTaskMapping(
                    task_type="CAPTION",
                    datasets=["coco_captions"],
                    weights={"coco_captions": 1.0},
                    sampling_strategy="balanced"
                ),
                YAMLTaskMapping(
                    task_type="OD",
                    datasets=["coco_detection"],
                    weights={"coco_detection": 1.0},
                    sampling_strategy="balanced"
                )
            ],
            training={
                "num_epochs": 10,
                "eval_steps": 500,
                "save_steps": 1000,
                "logging_steps": 100,
                "use_bf16": True,
                "gradient_accumulation_steps": 2,
                "model_config": {
                    "model_name": "microsoft/Florence-2-large",
                    "use_lora": True,
                    "lora_config": {
                        "r": 32,
                        "lora_alpha": 32,
                        "lora_dropout": 0.05
                    }
                },
                "data_config": {
                    "batch_size": 4,
                    "num_workers": 4,
                    "use_augmentation": True,
                    "augmentation_prob": 0.5
                },
                "optimization_config": {
                    "learning_rate": 1e-5,
                    "weight_decay": 0.01,
                    "lr_scheduler_type": "cosine",
                    "warmup_ratio": 0.1
                }
            }
        )


def create_yaml_config_template(output_path: Union[str, Path]) -> None:
    """创建YAML配置模板文件
    
    Args:
        output_path: 输出文件路径
    """
    example_config = FlorenceForgeYAMLConfig.create_example_config()
    example_config.save_to_yaml(output_path)
    print(f"YAML配置模板已创建: {output_path}")


def validate_yaml_config(config_path: Union[str, Path]) -> bool:
    """验证YAML配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        是否有效
    """
    try:
        config = FlorenceForgeYAMLConfig.load_from_yaml(config_path)
        validation_result = config.validate()
        
        if validation_result["errors"]:
            print("配置验证失败:")
            for error in validation_result["errors"]:
                print(f"  错误: {error}")
            return False
        
        if validation_result["warnings"]:
            print("配置验证警告:")
            for warning in validation_result["warnings"]:
                print(f"  警告: {warning}")
        
        print("配置验证通过!")
        return True
        
    except Exception as e:
        logger.exception("配置验证失败")
        return False