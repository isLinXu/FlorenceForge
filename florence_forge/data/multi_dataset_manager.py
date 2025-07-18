#!/usr/bin/env python3
"""
Florence Forge - 多数据集管理器

提供对多个数据集的统一管理和协调功能，支持复杂的多任务多数据集训练场景
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from collections import defaultdict
from dataclasses import dataclass

from .dataset import MultiTaskDataset, TaskSample
from .builder import DatasetBuilder
from ..core.config import DataConfig
from ..core.tasks import validate_task_name

logger = logging.getLogger(__name__)

@dataclass
class DatasetInfo:
    """数据集信息"""
    name: str
    path: str
    task_types: List[str]
    weight: float = 1.0
    priority: int = 1  # 优先级，数字越大优先级越高
    metadata: Dict[str, Any] = None
    format: str = "custom"  # 数据集格式
    max_samples: Optional[int] = None  # 最大样本数
    preprocessing: Dict[str, Any] = None  # 预处理配置
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.preprocessing is None:
            self.preprocessing = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "path": self.path,
            "task_types": self.task_types,
            "weight": self.weight,
            "priority": self.priority,
            "metadata": self.metadata,
            "format": self.format,
            "max_samples": self.max_samples,
            "preprocessing": self.preprocessing
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetInfo':
        """从字典创建实例"""
        return cls(
            name=data["name"],
            path=data["path"],
            task_types=data["task_types"],
            weight=data.get("weight", 1.0),
            priority=data.get("priority", 1),
            metadata=data.get("metadata", {}),
            format=data.get("format", "custom"),
            max_samples=data.get("max_samples"),
            preprocessing=data.get("preprocessing", {})
        )

@dataclass
class TaskDatasetMapping:
    """任务-数据集映射"""
    task_type: str
    datasets: List[str]  # 数据集名称列表
    weights: Union[Dict[str, float], List[float]]  # 每个数据集的权重
    sampling_strategy: str = "balanced"  # 采样策略
    
    def __post_init__(self):
        # 如果weights是列表，转换为字典
        if isinstance(self.weights, list):
            if len(self.weights) != len(self.datasets):
                raise ValueError("权重列表长度必须与数据集列表长度相同")
            self.weights = {dataset: weight for dataset, weight in zip(self.datasets, self.weights)}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "task_type": self.task_type,
            "datasets": self.datasets,
            "weights": self.weights,
            "sampling_strategy": self.sampling_strategy
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskDatasetMapping':
        """从字典创建实例"""
        return cls(
            task_type=data["task_type"],
            datasets=data["datasets"],
            weights=data["weights"],
            sampling_strategy=data.get("sampling_strategy", "balanced")
        )
    
class MultiDatasetManager:
    """多数据集管理器
    
    统一管理多个数据集，支持复杂的任务-数据集映射关系
    """
    
    def __init__(
        self,
        image_base_path: str = "",
        config: Optional[DataConfig] = None
    ):
        """初始化多数据集管理器
        
        Args:
            image_base_path: 图像文件基础路径
            config: 数据配置
        """
        self.image_base_path = Path(image_base_path)
        self.config = config or DataConfig()
        
        # 数据集注册表
        self.datasets: Dict[str, DatasetInfo] = {}
        self.loaded_datasets: Dict[str, MultiTaskDataset] = {}
        
        # 任务-数据集映射
        self.task_mappings: Dict[str, TaskDatasetMapping] = {}
        
        # 统计信息
        self.dataset_stats: Dict[str, Dict[str, Any]] = {}
        
        logger.info("多数据集管理器初始化完成")
    
    def register_dataset(
        self,
        dataset_info: Union[DatasetInfo, Dict[str, Any]] = None,
        name: str = None,
        path: Union[str, Path] = None,
        task_types: List[str] = None,
        weight: float = 1.0,
        priority: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
        format: str = "custom",
        max_samples: Optional[int] = None,
        preprocessing: Optional[Dict[str, Any]] = None
    ) -> 'MultiDatasetManager':
        """注册数据集
        
        Args:
            dataset_info: DatasetInfo实例或字典
            name: 数据集名称
            path: 数据集路径
            task_types: 支持的任务类型列表
            weight: 数据集权重
            priority: 优先级
            metadata: 元数据
            format: 数据集格式
            max_samples: 最大样本数
            preprocessing: 预处理配置
            
        Returns:
            自身实例
        """
        # 处理不同的输入方式
        if dataset_info is not None:
            if isinstance(dataset_info, dict):
                info = DatasetInfo.from_dict(dataset_info)
            else:
                info = dataset_info
        else:
            # 使用单独的参数创建DatasetInfo
            if name is None or path is None or task_types is None:
                raise ValueError("必须提供dataset_info或完整的参数集合")
            
            info = DatasetInfo(
                name=name,
                path=str(path),
                task_types=task_types,
                weight=weight,
                priority=priority,
                metadata=metadata or {},
                format=format,
                max_samples=max_samples,
                preprocessing=preprocessing or {}
            )
        
        # 验证任务类型
        for task_type in info.task_types:
            if not validate_task_name(task_type):
                raise ValueError(f"未知任务类型: {task_type}")
        
        # 验证路径（如果路径存在的话）
        dataset_path = Path(info.path)
        if not dataset_path.exists():
            logger.warning(f"数据集路径不存在: {dataset_path}")
        
        # 注册数据集
        self.datasets[info.name] = info
        logger.info(f"已注册数据集: {info.name}, 任务类型: {info.task_types}")
        
        return self
    
    def register_multiple_datasets(
        self,
        dataset_configs: List[Dict[str, Any]]
    ) -> 'MultiDatasetManager':
        """批量注册数据集
        
        Args:
            dataset_configs: 数据集配置列表
            
        Returns:
            自身实例
        """
        for config in dataset_configs:
            self.register_dataset(**config)
        
        return self
    
    def map_task_to_datasets(
        self,
        task_type: str,
        dataset_names: List[str],
        weights: Optional[Dict[str, float]] = None,
        sampling_strategy: str = "balanced"
    ) -> 'MultiDatasetManager':
        """映射任务到多个数据集
        
        Args:
            task_type: 任务类型
            dataset_names: 数据集名称列表
            weights: 每个数据集的权重
            sampling_strategy: 采样策略
            
        Returns:
            自身实例
        """
        # 验证任务类型
        if not validate_task_name(task_type):
            raise ValueError(f"未知任务类型: {task_type}")
        
        # 验证数据集是否已注册且支持该任务
        for dataset_name in dataset_names:
            if dataset_name not in self.datasets:
                raise ValueError(f"数据集未注册: {dataset_name}")
            
            dataset_info = self.datasets[dataset_name]
            if task_type not in dataset_info.task_types:
                raise ValueError(
                    f"数据集 {dataset_name} 不支持任务类型 {task_type}"
                )
        
        # 设置默认权重
        if weights is None:
            weights = {name: 1.0 for name in dataset_names}
        
        # 创建映射
        mapping = TaskDatasetMapping(
            task_type=task_type,
            datasets=dataset_names,
            weights=weights,
            sampling_strategy=sampling_strategy
        )
        
        self.task_mappings[task_type] = mapping
        logger.info(f"已映射任务 {task_type} 到数据集: {dataset_names}")
        
        return self
    
    def load_dataset(self, name: str) -> MultiTaskDataset:
        """加载单个数据集
        
        Args:
            name: 数据集名称
            
        Returns:
            加载的数据集
        """
        if name in self.loaded_datasets:
            return self.loaded_datasets[name]
        
        if name not in self.datasets:
            raise ValueError(f"数据集未注册: {name}")
        
        dataset_info = self.datasets[name]
        
        # 为每个任务类型创建数据配置
        data_configs = []
        for task_type in dataset_info.task_types:
            data_configs.append({
                "task_type": task_type,
                "data_path": dataset_info.path,
                "weight": dataset_info.weight
            })
        
        # 创建数据集
        dataset = MultiTaskDataset(
            data_configs=data_configs,
            image_base_path=str(self.image_base_path),
            config=self.config
        )
        
        self.loaded_datasets[name] = dataset
        
        # 收集统计信息
        self.dataset_stats[name] = dataset.get_task_statistics()
        
        logger.info(f"已加载数据集: {name}")
        return dataset
    
    def create_unified_dataset(
        self,
        task_types: Optional[List[str]] = None,
        processor=None
    ) -> MultiTaskDataset:
        """创建统一的多任务数据集
        
        Args:
            task_types: 要包含的任务类型列表，None表示包含所有任务
            processor: 数据处理器
            
        Returns:
            统一的多任务数据集
        """
        if task_types is None:
            task_types = list(self.task_mappings.keys())
        
        # 收集所有数据配置
        all_data_configs = []
        
        for task_type in task_types:
            if task_type not in self.task_mappings:
                logger.warning(f"任务类型 {task_type} 没有映射的数据集，跳过")
                continue
            
            mapping = self.task_mappings[task_type]
            
            # 为每个映射的数据集创建配置
            for dataset_name in mapping.datasets:
                dataset_info = self.datasets[dataset_name]
                
                # 计算最终权重（数据集权重 × 映射权重）
                final_weight = (
                    dataset_info.weight * 
                    mapping.weights.get(dataset_name, 1.0)
                )
                
                config = {
                    "task_type": task_type,
                    "data_path": dataset_info.path,
                    "weight": final_weight,
                    "dataset_name": dataset_name,
                    "priority": dataset_info.priority,
                    "sampling_strategy": mapping.sampling_strategy
                }
                
                all_data_configs.append(config)
        
        if not all_data_configs:
            raise ValueError("没有找到有效的数据配置")
        
        # 创建统一数据集
        unified_dataset = MultiTaskDataset(
            data_configs=all_data_configs,
            image_base_path=str(self.image_base_path),
            config=self.config,
            processor=processor
        )
        
        logger.info(
            f"已创建统一数据集，包含 {len(task_types)} 个任务类型，"
            f"{len(all_data_configs)} 个数据配置"
        )
        
        return unified_dataset
    
    def create_task_specific_dataset(
        self,
        task_type: str,
        processor=None
    ) -> MultiTaskDataset:
        """创建特定任务的数据集
        
        Args:
            task_type: 任务类型
            processor: 数据处理器
            
        Returns:
            任务特定的数据集
        """
        if task_type not in self.task_mappings:
            raise ValueError(f"任务类型 {task_type} 没有映射的数据集")
        
        return self.create_unified_dataset([task_type], processor)
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """获取数据集统计信息
        
        Returns:
            统计信息字典
        """
        stats = {
            "registered_datasets": len(self.datasets),
            "loaded_datasets": len(self.loaded_datasets),
            "task_mappings": len(self.task_mappings),
            "datasets": {},
            "task_coverage": {},
            "total_samples_estimate": 0
        }
        
        # 数据集详细信息
        for name, info in self.datasets.items():
            dataset_stat = {
                "path": info.path,
                "task_types": info.task_types,
                "weight": info.weight,
                "priority": info.priority,
                "loaded": name in self.loaded_datasets
            }
            
            # 如果已加载，添加详细统计
            if name in self.dataset_stats:
                dataset_stat.update(self.dataset_stats[name])
                stats["total_samples_estimate"] += dataset_stat.get("total_samples", 0)
            
            stats["datasets"][name] = dataset_stat
        
        # 任务覆盖情况
        for task_type, mapping in self.task_mappings.items():
            stats["task_coverage"][task_type] = {
                "datasets": mapping.datasets,
                "weights": mapping.weights,
                "sampling_strategy": mapping.sampling_strategy
            }
        
        return stats
    
    def validate_configuration(self) -> Dict[str, List[str]]:
        """验证配置的完整性
        
        Returns:
            验证结果，包含错误和警告信息
        """
        errors = []
        warnings = []
        
        # 检查数据集路径
        for name, info in self.datasets.items():
            if not Path(info.path).exists():
                errors.append(f"数据集 {name} 的路径不存在: {info.path}")
        
        # 检查任务映射
        all_tasks = set()
        for info in self.datasets.values():
            all_tasks.update(info.task_types)
        
        mapped_tasks = set(self.task_mappings.keys())
        unmapped_tasks = all_tasks - mapped_tasks
        
        if unmapped_tasks:
            warnings.append(f"以下任务类型没有映射: {list(unmapped_tasks)}")
        
        # 检查映射的数据集是否存在
        for task_type, mapping in self.task_mappings.items():
            for dataset_name in mapping.datasets:
                if dataset_name not in self.datasets:
                    errors.append(
                        f"任务 {task_type} 映射的数据集 {dataset_name} 未注册"
                    )
        
        return {"errors": errors, "warnings": warnings}
    
    def save_configuration(self, config_path: Union[str, Path]) -> None:
        """保存配置到文件
        
        Args:
            config_path: 配置文件路径
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_data = {
            "image_base_path": str(self.image_base_path),
            "data_config": self.config.to_dict(),
            "datasets": {
                name: {
                    "path": info.path,
                    "task_types": info.task_types,
                    "weight": info.weight,
                    "priority": info.priority,
                    "metadata": info.metadata
                }
                for name, info in self.datasets.items()
            },
            "task_mappings": {
                task_type: {
                    "datasets": mapping.datasets,
                    "weights": mapping.weights,
                    "sampling_strategy": mapping.sampling_strategy
                }
                for task_type, mapping in self.task_mappings.items()
            },
            "statistics": self.get_dataset_statistics()
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"配置已保存到: {config_path}")
    
    @classmethod
    def load_configuration(
        cls,
        config_path: Union[str, Path]
    ) -> 'MultiDatasetManager':
        """从配置文件加载
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            多数据集管理器实例
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        # 创建管理器实例
        manager = cls(
            image_base_path=config_data.get("image_base_path", ""),
            config=DataConfig.from_dict(config_data.get("data_config", {}))
        )
        
        # 注册数据集
        for name, dataset_config in config_data.get("datasets", {}).items():
            manager.register_dataset(
                name=name,
                path=dataset_config["path"],
                task_types=dataset_config["task_types"],
                weight=dataset_config.get("weight", 1.0),
                priority=dataset_config.get("priority", 1),
                metadata=dataset_config.get("metadata", {})
            )
        
        # 设置任务映射
        for task_type, mapping_config in config_data.get("task_mappings", {}).items():
            manager.map_task_to_datasets(
                task_type=task_type,
                dataset_names=mapping_config["datasets"],
                weights=mapping_config.get("weights"),
                sampling_strategy=mapping_config.get("sampling_strategy", "balanced")
            )
        
        logger.info(f"配置已从文件加载: {config_path}")
        return manager
    
    def create_balanced_split(
        self,
        val_ratio: float = 0.2,
        test_ratio: float = 0.0,
        stratify_by_task: bool = True,
        stratify_by_dataset: bool = True,
        random_seed: Optional[int] = None
    ) -> Tuple[MultiTaskDataset, MultiTaskDataset, Optional[MultiTaskDataset]]:
        """创建平衡的数据集划分
        
        Args:
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            stratify_by_task: 是否按任务分层
            stratify_by_dataset: 是否按数据集分层
            random_seed: 随机种子
            
        Returns:
            (训练集, 验证集, 测试集) 元组，测试集可能为None
        """
        import random
        
        if random_seed is not None:
            random.seed(random_seed)
        
        # 创建统一数据集
        full_dataset = self.create_unified_dataset()
        
        # 实现分层划分逻辑
        train_indices = []
        val_indices = []
        test_indices = [] if test_ratio > 0 else None
        
        if stratify_by_task and stratify_by_dataset:
            # 按任务和数据集双重分层
            for task_type in full_dataset.task_indices.keys():
                task_samples = full_dataset.get_task_samples(task_type)
                
                # 按数据集分组
                dataset_groups = defaultdict(list)
                for idx in task_samples:
                    sample = full_dataset.samples[idx]
                    dataset_name = sample.metadata.get("dataset_name", "unknown")
                    dataset_groups[dataset_name].append(idx)
                
                # 对每个数据集组进行划分
                for dataset_name, indices in dataset_groups.items():
                    random.shuffle(indices)
                    
                    if test_ratio > 0:
                        test_split = int(len(indices) * test_ratio)
                        val_split = int(len(indices) * val_ratio)
                        
                        test_indices.extend(indices[:test_split])
                        val_indices.extend(indices[test_split:test_split + val_split])
                        train_indices.extend(indices[test_split + val_split:])
                    else:
                        val_split = int(len(indices) * val_ratio)
                        val_indices.extend(indices[:val_split])
                        train_indices.extend(indices[val_split:])
        
        elif stratify_by_task:
            # 仅按任务分层
            for task_type in full_dataset.task_indices.keys():
                task_indices = full_dataset.get_task_samples(task_type)
                random.shuffle(task_indices)
                
                if test_ratio > 0:
                    test_split = int(len(task_indices) * test_ratio)
                    val_split = int(len(task_indices) * val_ratio)
                    
                    test_indices.extend(task_indices[:test_split])
                    val_indices.extend(task_indices[test_split:test_split + val_split])
                    train_indices.extend(task_indices[test_split + val_split:])
                else:
                    val_split = int(len(task_indices) * val_ratio)
                    val_indices.extend(task_indices[:val_split])
                    train_indices.extend(task_indices[val_split:])
        
        else:
            # 随机划分
            all_indices = list(range(len(full_dataset)))
            random.shuffle(all_indices)
            
            if test_ratio > 0:
                test_split = int(len(all_indices) * test_ratio)
                val_split = int(len(all_indices) * val_ratio)
                
                test_indices = all_indices[:test_split]
                val_indices = all_indices[test_split:test_split + val_split]
                train_indices = all_indices[test_split + val_split:]
            else:
                val_split = int(len(all_indices) * val_ratio)
                val_indices = all_indices[:val_split]
                train_indices = all_indices[val_split:]
        
        # 创建子集
        train_dataset = full_dataset.create_subset(train_indices)
        val_dataset = full_dataset.create_subset(val_indices)
        test_dataset = full_dataset.create_subset(test_indices) if test_indices else None
        
        logger.info(
            f"数据集划分完成 - 训练集: {len(train_dataset)} 样本, "
            f"验证集: {len(val_dataset)} 样本"
            + (f", 测试集: {len(test_dataset)} 样本" if test_dataset else "")
        )
        
        return train_dataset, val_dataset, test_dataset