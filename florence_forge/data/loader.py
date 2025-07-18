"""FlorenceForge任务数据加载器模块

提供高效的数据加载、批处理和采样功能
"""

import torch
from torch.utils.data import DataLoader, Sampler
import random
import logging
import numpy as np
from typing import Optional, List, Iterator, Callable, Dict, Any
from PIL import Image

from .dataset import MultiTaskDataset
from ..core.config import DataConfig

logger = logging.getLogger(__name__)

class TaskBalancedSampler(Sampler):
    """任务平衡采样器
    
    确保每个批次中包含来自不同任务的样本
    """
    
    def __init__(
        self,
        dataset: MultiTaskDataset,
        batch_size: int,
        drop_last: bool = True,
        shuffle: bool = True,
        seed: Optional[int] = None
    ):
        """初始化采样器
        
        Args:
            dataset: 多任务数据集
            batch_size: 批次大小
            drop_last: 是否丢弃最后不完整的批次
            shuffle: 是否打乱顺序
            seed: 随机种子
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        
        self.task_indices = dataset.task_indices
        self.task_weights = dataset.task_weights
        
        # 计算总批次数
        self.num_samples = len(dataset)
        if self.drop_last:
            self.num_batches = self.num_samples // self.batch_size
        else:
            self.num_batches = (self.num_samples + self.batch_size - 1) // self.batch_size
    
    def __iter__(self) -> Iterator[int]:
        """生成采样索引"""
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
        
        # 为每个任务创建索引池
        task_pools = {}
        for task_type, indices in self.task_indices.items():
            pool = indices.copy()
            if self.shuffle:
                random.shuffle(pool)
            task_pools[task_type] = pool
        
        # 计算任务权重
        task_types = list(self.task_indices.keys())
        if self.task_weights:
            weights = [self.task_weights.get(task, 1.0) for task in task_types]
        else:
            weights = [1.0] * len(task_types)
        
        # 归一化权重
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # 生成批次
        for _ in range(self.num_batches):
            batch_indices = []
            
            for _ in range(self.batch_size):
                # 根据权重选择任务
                task_type = np.random.choice(task_types, p=weights)
                
                # 从任务池中获取样本
                if task_pools[task_type]:
                    idx = task_pools[task_type].pop(0)
                    batch_indices.append(idx)
                else:
                    # 如果任务池为空，重新填充
                    pool = self.task_indices[task_type].copy()
                    if self.shuffle:
                        random.shuffle(pool)
                    task_pools[task_type] = pool
                    
                    if pool:
                        idx = task_pools[task_type].pop(0)
                        batch_indices.append(idx)
            
            # 如果批次不完整且需要丢弃，则跳过
            if len(batch_indices) < self.batch_size and self.drop_last:
                continue
            
            yield from batch_indices
    
    def __len__(self) -> int:
        """返回采样器长度"""
        return self.num_batches * self.batch_size if self.drop_last else self.num_samples

class TaskRoundRobinSampler(Sampler):
    """任务轮询采样器
    
    按轮询方式从不同任务中采样
    """
    
    def __init__(
        self,
        dataset: MultiTaskDataset,
        shuffle: bool = True,
        seed: Optional[int] = None
    ):
        """初始化采样器
        
        Args:
            dataset: 多任务数据集
            shuffle: 是否打乱任务内顺序
            seed: 随机种子
        """
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        
        self.task_indices = dataset.task_indices
        self.num_samples = len(dataset)
    
    def __iter__(self) -> Iterator[int]:
        """生成采样索引"""
        if self.seed is not None:
            random.seed(self.seed)
        
        # 为每个任务创建索引迭代器
        task_iterators = {}
        for task_type, indices in self.task_indices.items():
            task_list = indices.copy()
            if self.shuffle:
                random.shuffle(task_list)
            task_iterators[task_type] = iter(task_list)
        
        task_types = list(self.task_indices.keys())
        if self.shuffle:
            random.shuffle(task_types)
        
        # 轮询生成样本
        samples_generated = 0
        task_idx = 0
        
        while samples_generated < self.num_samples:
            current_task = task_types[task_idx % len(task_types)]
            
            try:
                idx = next(task_iterators[current_task])
                yield idx
                samples_generated += 1
            except StopIteration:
                # 当前任务已耗尽，重新创建迭代器
                task_list = self.task_indices[current_task].copy()
                if self.shuffle:
                    random.shuffle(task_list)
                task_iterators[current_task] = iter(task_list)
                
                # 尝试再次获取
                try:
                    idx = next(task_iterators[current_task])
                    yield idx
                    samples_generated += 1
                except StopIteration:
                    # 如果任务为空，记录警告并跳过
                    logger.warning(f"任务 {current_task} 的数据已完全耗尽，跳过此轮采样")
                    # 从任务列表中移除空任务，避免无限循环
                    if current_task in task_types:
                        task_types.remove(current_task)
                    
                    # 如果所有任务都为空，提前结束
                    if not task_types:
                        logger.error("所有任务数据都已耗尽，提前结束采样")
                        break
            
            task_idx += 1
    
    def __len__(self) -> int:
        """返回采样器长度"""
        return self.num_samples

def collate_fn(batch):
    """自定义批处理函数
    
    Args:
        batch: 批次数据列表
        
    Returns:
        批处理后的数据
    """
    from torch.utils.data.dataloader import default_collate
    import torch
    
    # 检查空批次
    if not batch:
        # 创建一个最小的有效批次
        dummy_sample = {
            "input_ids": torch.tensor([[0]], dtype=torch.long),
            "pixel_values": torch.zeros((1, 3, 224, 224), dtype=torch.float32),
            "attention_mask": torch.tensor([[1]], dtype=torch.long),
            "task_type": "CAPTION",  # 默认任务类型
            "is_empty": True
        }
        return dummy_sample
    
    # 提取任务类型信息
    task_types = []
    for item in batch:
        if 'task_type' in item:
            task_types.append(item['task_type'])
    
    # 处理PIL图像对象
    processed_batch = []
    for item in batch:
        processed_item = {}
        for key, value in item.items():
            if isinstance(value, Image.Image):
                # 跳过PIL图像，因为它们已经被processor处理了
                continue
            else:
                processed_item[key] = value
        processed_batch.append(processed_item)
    
    # 对于非空批次，使用默认的collate函数
    try:
        result = default_collate(processed_batch)
        result["is_empty"] = False
        
        # 添加任务类型信息
        if task_types:
            # 如果批次中所有样本都是同一任务类型，使用该类型
            if len(set(task_types)) == 1:
                result["task_type"] = task_types[0]
            else:
                # 如果是混合任务，保留所有任务类型
                result["task_types"] = task_types
                result["task_type"] = task_types[0]  # 使用第一个作为主要任务类型
        
        return result
    except Exception as e:
        # 如果默认collate失败，返回第一个样本
        sample = processed_batch[0] if processed_batch else batch[0]
        sample["is_empty"] = False
        
        # 确保包含任务类型
        if 'task_type' not in sample and task_types:
            sample['task_type'] = task_types[0]
        
        return sample

class TaskDataLoader:
    """任务数据加载器
    
    提供多种采样策略和批处理选项
    """
    
    def __init__(
        self,
        dataset: MultiTaskDataset,
        config: Optional[DataConfig] = None,
        sampling_strategy: str = "balanced",
        **kwargs
    ):
        """初始化数据加载器
        
        Args:
            dataset: 多任务数据集
            config: 数据配置
            sampling_strategy: 采样策略 ("balanced", "round_robin", "random")
            **kwargs: 其他DataLoader参数
        """
        self.dataset = dataset
        self.config = config or DataConfig()
        self.sampling_strategy = sampling_strategy
        
        # 设置默认参数
        self.dataloader_kwargs = {
            "batch_size": self.config.batch_size,
            "num_workers": self.config.num_workers,
            "pin_memory": self.config.pin_memory,
            "drop_last": self.config.drop_last,
            "collate_fn": collate_fn  # 使用自定义的collate函数
        }
        
        # 更新用户提供的参数
        self.dataloader_kwargs.update(kwargs)
        
        # 设置采样器
        self._setup_sampler()
    
    def _setup_sampler(self) -> None:
        """设置采样器"""
        if self.sampling_strategy == "balanced":
            self.sampler = TaskBalancedSampler(
                dataset=self.dataset,
                batch_size=self.dataloader_kwargs["batch_size"],
                drop_last=self.dataloader_kwargs["drop_last"],
                shuffle=self.config.shuffle
            )
            # 使用自定义采样器时，不能同时设置shuffle
            self.dataloader_kwargs["shuffle"] = False
            self.dataloader_kwargs["sampler"] = self.sampler
        
        elif self.sampling_strategy == "round_robin":
            self.sampler = TaskRoundRobinSampler(
                dataset=self.dataset,
                shuffle=self.config.shuffle
            )
            self.dataloader_kwargs["shuffle"] = False
            self.dataloader_kwargs["sampler"] = self.sampler
        
        elif self.sampling_strategy == "random":
            # 使用默认的随机采样
            self.sampler = None
            self.dataloader_kwargs["shuffle"] = self.config.shuffle
        
        else:
            raise ValueError(f"未知采样策略: {self.sampling_strategy}")
    
    def get_dataloader(self) -> DataLoader:
        """获取PyTorch DataLoader
        
        Returns:
            配置好的DataLoader
        """
        return DataLoader(self.dataset, **self.dataloader_kwargs)
    
    def __iter__(self):
        """迭代器接口"""
        return iter(self.get_dataloader())
    
    def __len__(self) -> int:
        """返回批次数量"""
        if self.sampler:
            return len(self.sampler) // self.dataloader_kwargs["batch_size"]
        else:
            dataset_size = len(self.dataset)
            batch_size = self.dataloader_kwargs["batch_size"]
            if self.dataloader_kwargs["drop_last"]:
                return dataset_size // batch_size
            else:
                return (dataset_size + batch_size - 1) // batch_size
    
    def get_task_distribution(self, num_batches: int = 10) -> Dict[str, float]:
        """分析任务分布
        
        Args:
            num_batches: 分析的批次数量
            
        Returns:
            任务分布统计
        """
        task_counts = defaultdict(int)
        total_samples = 0
        
        dataloader = self.get_dataloader()
        
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            
            for task_type in batch["task_types"]:
                task_counts[task_type] += 1
                total_samples += 1
        
        # 计算比例
        task_distribution = {
            task: count / total_samples
            for task, count in task_counts.items()
        }
        
        return task_distribution
    
    def set_epoch(self, epoch: int) -> None:
        """设置训练轮次（用于分布式训练）
        
        Args:
            epoch: 当前轮次
        """
        if hasattr(self.sampler, 'set_epoch'):
            self.sampler.set_epoch(epoch)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取数据加载器统计信息
        
        Returns:
            统计信息字典
        """
        return {
            "dataset_size": len(self.dataset),
            "batch_size": self.dataloader_kwargs["batch_size"],
            "num_batches": len(self),
            "sampling_strategy": self.sampling_strategy,
            "num_workers": self.dataloader_kwargs["num_workers"],
            "pin_memory": self.dataloader_kwargs["pin_memory"],
            "drop_last": self.dataloader_kwargs["drop_last"],
            "dataset_statistics": self.dataset.get_task_statistics()
        }