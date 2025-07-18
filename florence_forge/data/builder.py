"""FlorenceForge数据集构建器模块

提供便捷的数据集创建、配置和管理功能
"""

import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Tuple

from .dataset import MultiTaskDataset
from ..core.config import DataConfig
from ..core.tasks import validate_task_name, list_all_tasks

logger = logging.getLogger(__name__)

class DatasetBuilder:
    """数据集构建器
    
    提供灵活的数据集创建和配置功能
    """
    
    def __init__(self, image_base_path: str = "", config: Optional[DataConfig] = None):
        """初始化构建器
        
        Args:
            image_base_path: 图像文件基础路径
            config: 数据配置
        """
        self.image_base_path = Path(image_base_path)
        self.config = config or DataConfig()
        self.data_configs: List[Dict[str, Any]] = []
        self.processor = None
    
    def set_processor(self, processor) -> 'DatasetBuilder':
        """设置数据处理器
        
        Args:
            processor: 数据处理器
            
        Returns:
            自身实例
        """
        self.processor = processor
        return self
    
    def add_task_data(
        self,
        task_type: str,
        data_path: Union[str, Path],
        weight: float = 1.0,
        **kwargs
    ) -> 'DatasetBuilder':
        """添加任务数据
        
        Args:
            task_type: 任务类型
            data_path: 数据文件路径
            weight: 任务权重
            **kwargs: 其他配置参数
            
        Returns:
            自身实例
        """
        if not validate_task_name(task_type):
            raise ValueError(f"未知任务类型: {task_type}")
        
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {data_path}")
        
        config = {
            "task_type": task_type,
            "data_path": str(data_path),
            "weight": weight,
            **kwargs
        }
        
        self.data_configs.append(config)
        logger.info(f"已添加任务数据: {task_type} -> {data_path}")
        
        return self
    
    def add_multiple_tasks(
        self,
        task_configs: List[Dict[str, Any]]
    ) -> 'DatasetBuilder':
        """批量添加任务数据
        
        Args:
            task_configs: 任务配置列表
            
        Returns:
            自身实例
        """
        for config in task_configs:
            self.add_task_data(**config)
        
        return self
    
    def from_config_file(self, config_path: Union[str, Path]) -> 'DatasetBuilder':
        """从配置文件加载任务数据
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            自身实例
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        # 更新基础配置
        if "image_base_path" in config_data:
            self.image_base_path = Path(config_data["image_base_path"])
        
        if "data_config" in config_data:
            self.config = DataConfig.from_dict(config_data["data_config"])
        
        # 添加任务数据
        if "tasks" in config_data:
            self.add_multiple_tasks(config_data["tasks"])
        
        logger.info(f"已从配置文件加载: {config_path}")
        return self
    
    def auto_discover_tasks(
        self,
        data_directory: Union[str, Path],
        pattern: str = "*.jsonl"
    ) -> 'DatasetBuilder':
        """自动发现任务数据文件
        
        Args:
            data_directory: 数据目录
            pattern: 文件匹配模式
            
        Returns:
            自身实例
        """
        data_dir = Path(data_directory)
        if not data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {data_dir}")
        
        # 查找匹配的文件
        data_files = list(data_dir.glob(pattern))
        
        for file_path in data_files:
            # 尝试从文件名推断任务类型
            file_stem = file_path.stem.upper()
            
            # 查找匹配的任务类型
            matching_tasks = []
            for task_name in list_all_tasks():
                if task_name in file_stem or file_stem in task_name:
                    matching_tasks.append(task_name)
            
            if len(matching_tasks) == 1:
                task_type = matching_tasks[0]
                self.add_task_data(task_type, file_path)
                logger.info(f"自动发现任务: {task_type} -> {file_path}")
            elif len(matching_tasks) > 1:
                logger.warning(f"文件 {file_path} 匹配多个任务类型: {matching_tasks}")
            else:
                logger.warning(f"无法为文件 {file_path} 确定任务类型")
        
        return self
    
    def set_task_weights(self, weights: Dict[str, float]) -> 'DatasetBuilder':
        """设置任务权重
        
        Args:
            weights: 任务权重字典
            
        Returns:
            自身实例
        """
        for config in self.data_configs:
            task_type = config["task_type"]
            if task_type in weights:
                config["weight"] = weights[task_type]
        
        return self
    
    def filter_tasks(self, task_types: List[str]) -> 'DatasetBuilder':
        """过滤任务类型
        
        Args:
            task_types: 要保留的任务类型列表
            
        Returns:
            自身实例
        """
        self.data_configs = [
            config for config in self.data_configs
            if config["task_type"] in task_types
        ]
        
        logger.info(f"已过滤任务，保留: {task_types}")
        return self
    
    def limit_samples_per_task(self, max_samples: int) -> 'DatasetBuilder':
        """限制每个任务的样本数量
        
        Args:
            max_samples: 最大样本数
            
        Returns:
            自身实例
        """
        self.config.max_samples_per_task = max_samples
        return self
    
    def enable_balanced_sampling(self, enabled: bool = True) -> 'DatasetBuilder':
        """启用平衡采样
        
        Args:
            enabled: 是否启用
            
        Returns:
            自身实例
        """
        self.config.use_balanced_sampling = enabled
        return self
    
    def build(self) -> MultiTaskDataset:
        """构建数据集
        
        Returns:
            多任务数据集实例
        """
        if not self.data_configs:
            raise ValueError("没有添加任何任务数据")
        
        logger.info(f"正在构建数据集，包含 {len(self.data_configs)} 个任务")
        
        dataset = MultiTaskDataset(
            data_configs=self.data_configs,
            image_base_path=str(self.image_base_path),
            config=self.config,
            processor=self.processor
        )
        
        return dataset
    
    def build_train_val_split(
        self,
        val_ratio: float = 0.2,
        stratify: bool = True,
        random_seed: Optional[int] = None
    ) -> Tuple[MultiTaskDataset, MultiTaskDataset]:
        """构建训练和验证数据集
        
        Args:
            val_ratio: 验证集比例
            stratify: 是否按任务类型分层
            random_seed: 随机种子
            
        Returns:
            (训练集, 验证集) 元组
        """
        import random
        
        if random_seed is not None:
            random.seed(random_seed)
        
        # 先构建完整数据集
        full_dataset = self.build()
        
        if stratify:
            # 按任务类型分层划分
            train_indices = []
            val_indices = []
            
            for task_type in full_dataset.task_indices.keys():
                task_indices = full_dataset.get_task_samples(task_type)
                random.shuffle(task_indices)
                
                split_point = int(len(task_indices) * (1 - val_ratio))
                train_indices.extend(task_indices[:split_point])
                val_indices.extend(task_indices[split_point:])
        else:
            # 随机划分
            all_indices = list(range(len(full_dataset)))
            random.shuffle(all_indices)
            
            split_point = int(len(all_indices) * (1 - val_ratio))
            train_indices = all_indices[:split_point]
            val_indices = all_indices[split_point:]
        
        # 创建子集
        train_dataset = full_dataset.create_subset(train_indices)
        val_dataset = full_dataset.create_subset(val_indices)
        
        logger.info(
            f"数据集划分完成 - 训练集: {len(train_dataset)} 样本, "
            f"验证集: {len(val_dataset)} 样本"
        )
        
        return train_dataset, val_dataset
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取构建器统计信息
        
        Returns:
            统计信息字典
        """
        task_counts = defaultdict(int)
        total_weight = 0
        
        for config in self.data_configs:
            task_type = config["task_type"]
            weight = config.get("weight", 1.0)
            
            # 快速统计文件行数（样本数）
            try:
                with open(config["data_path"], 'r', encoding='utf-8') as f:
                    count = sum(1 for _ in f)
                task_counts[task_type] += count
                total_weight += weight
            except Exception as e:
                logger.warning(f"无法统计文件 {config['data_path']}: {e}")
        
        return {
            "num_tasks": len(self.data_configs),
            "task_types": list(task_counts.keys()),
            "estimated_samples": dict(task_counts),
            "total_estimated_samples": sum(task_counts.values()),
            "total_weight": total_weight,
            "image_base_path": str(self.image_base_path),
            "config": self.config.to_dict()
        }
    
    def save_config(self, config_path: Union[str, Path]) -> None:
        """保存构建器配置
        
        Args:
            config_path: 配置文件路径
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_data = {
            "image_base_path": str(self.image_base_path),
            "data_config": self.config.to_dict(),
            "tasks": self.data_configs,
            "statistics": self.get_statistics()
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"构建器配置已保存到: {config_path}")
    
    def clear(self) -> 'DatasetBuilder':
        """清空所有配置
        
        Returns:
            自身实例
        """
        self.data_configs.clear()
        logger.info("已清空所有任务配置")
        return self
    
    def copy(self) -> 'DatasetBuilder':
        """创建构建器副本
        
        Returns:
            新的构建器实例
        """
        new_builder = DatasetBuilder(
            image_base_path=str(self.image_base_path),
            config=self.config
        )
        new_builder.data_configs = self.data_configs.copy()
        new_builder.processor = self.processor
        
        return new_builder