"""FlorenceForge任务数据加载器模块

提供高效的数据加载、批处理和采样功能
"""

import torch
import os
from torch.utils.data import DataLoader, Sampler, DistributedSampler
import random
import logging
import numpy as np
from typing import Optional, List, Iterator, Callable, Dict, Any
from collections import defaultdict, deque

from .dataset import MultiTaskDataset
from .collate import Florence2Collator, collate_fn
from ..core.config import DataConfig

logger = logging.getLogger(__name__)


def _get_distributed_info() -> tuple[int, int, int]:
    """获取分布式训练信息

    自动检测当前是否处于分布式环境，返回 (world_size, rank, local_rank)。
    如果未初始化分布式，则返回 (1, 0, 0)。

    Returns:
        (world_size, rank, local_rank) 元组
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        # local_rank 无法直接从 torch.distributed 获取，从环境变量推断
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return world_size, rank, local_rank
    return 1, 0, 0

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
        """生成采样索引（预计算优化版本）"""
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        # 为每个任务创建索引池（使用 deque 支持 O(1) pop）
        task_pools = {}
        for task_type, indices in self.task_indices.items():
            pool = deque(indices)
            if self.shuffle:
                # 转为 list 打乱后重新构建 deque
                shuffled = list(pool)
                random.shuffle(shuffled)
                pool = deque(shuffled)
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

        # 预计算所有任务选择（比逐样本 np.random.choice 快 10-50 倍）
        total_indices_needed = self.num_batches * self.batch_size
        task_choices = np.random.choice(
            len(task_types), size=total_indices_needed, p=weights
        )

        # 按预计算的选择分配索引
        idx = 0
        for batch_idx in range(self.num_batches):
            batch_indices = []

            for sample_idx in range(self.batch_size):
                if idx >= len(task_choices):
                    break
                task_type = task_types[task_choices[idx]]
                idx += 1

                # 从任务池中获取样本 (deque.popleft() O(1))
                if task_pools[task_type]:
                    batch_indices.append(task_pools[task_type].popleft())
                else:
                    # 如果任务池为空，重新填充
                    pool = deque(self.task_indices[task_type])
                    if self.shuffle:
                        shuffled = list(pool)
                        random.shuffle(shuffled)
                        pool = deque(shuffled)
                    task_pools[task_type] = pool

                    if pool:
                        batch_indices.append(task_pools[task_type].popleft())

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


class DistributedTaskSampler(Sampler):
    """分布式任务采样器包装器

    将任意底层采样器（如 TaskBalancedSampler、TaskRoundRobinSampler）
    包装为分布式版本，确保多进程训练中每个 rank 只处理不同的数据子集。

    实现方式：先由底层采样器生成完整的 epoch 索引序列，
    再按 rank 和 world_size 切片分配，避免重复采样。

    用法示例::

        base_sampler = TaskBalancedSampler(dataset, batch_size=4)
        dist_sampler = DistributedTaskSampler(
            base_sampler, world_size=4, rank=0, shuffle=True
        )
        loader = DataLoader(dataset, sampler=dist_sampler, ...)
    """

    def __init__(
        self,
        sampler: Sampler,
        world_size: int,
        rank: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ):
        """初始化分布式采样器

        Args:
            sampler: 底层采样器（如 TaskBalancedSampler）
            world_size: 分布式世界大小（进程总数）
            rank: 当前进程 rank
            shuffle: 是否在每个 epoch 打乱顺序
            seed: 随机种子（配合 epoch 使用）
            drop_last: 是否丢弃不完整的尾部数据
        """
        self.sampler = sampler
        self.world_size = world_size
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        # 计算当前 rank 应处理的样本数
        full_length = len(sampler)
        if drop_last:
            self.num_samples = full_length // world_size
        else:
            self.num_samples = (full_length + world_size - 1) // world_size

        self.total_size = self.num_samples * world_size

    def __iter__(self) -> Iterator[int]:
        """生成分布式索引序列"""
        # 1. 获取底层采样器的完整索引序列
        indices = list(self.sampler)

        # 2. 如果需要，填充到 world_size 的整数倍（保证每个 rank 样本数一致）
        if not self.drop_last:
            # 循环填充
            padding_size = self.total_size - len(indices)
            if padding_size > 0:
                indices += indices[:padding_size]
        else:
            # 截断到 world_size 整数倍
            indices = indices[: self.total_size]

        # 3. 根据 epoch 设置随机种子进行打乱（保证不同 epoch 顺序不同）
        if self.shuffle:
            # 使用确定性随机，保证所有 rank 看到相同的打乱顺序
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            # 将 indices 转为 tensor 进行随机打乱，再转回 list
            indices_tensor = torch.tensor(indices, dtype=torch.int64)
            perm = torch.randperm(len(indices_tensor), generator=g)
            indices = indices_tensor[perm].tolist()

        # 4. 按 rank 切片
        start_idx = self.rank * self.num_samples
        end_idx = start_idx + self.num_samples
        rank_indices = indices[start_idx:end_idx]

        return iter(rank_indices)

    def __len__(self) -> int:
        """返回当前 rank 的样本数"""
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """设置当前 epoch（用于同步打乱顺序）

        应在每个 epoch 开始时调用，确保不同 epoch 的采样顺序不同。

        Args:
            epoch: 当前训练轮次
        """
        self.epoch = epoch
        # 同时传播给底层采样器（如果支持）
        if hasattr(self.sampler, 'set_epoch'):
            self.sampler.set_epoch(epoch)


class TaskDataLoader:
    """任务数据加载器
    
    提供多种采样策略和批处理选项，支持分布式训练。
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
        
        # 检测分布式环境
        self._distributed, self._world_size, self._rank, self._local_rank = self._detect_distributed()
        
        # 设置默认参数
        # pin_memory仅在CUDA/MPS设备可用时有效
        mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        pin_memory = self.config.pin_memory and (torch.cuda.is_available() or mps_available)
        
        # 使用 Florence2Collator 替代旧 collate_fn，支持动态 padding
        collator = Florence2Collator()

        self.dataloader_kwargs = {
            "batch_size": self.config.batch_size,
            "num_workers": self.config.num_workers,
            "pin_memory": pin_memory,
            "drop_last": self.config.drop_last,
            "collate_fn": collator  # 使用专业的 Florence2Collator
        }

        # DataLoader 性能调优参数
        if self.config.num_workers > 0:
            if self.config.prefetch_factor is not None:
                self.dataloader_kwargs["prefetch_factor"] = self.config.prefetch_factor
            if self.config.persistent_workers:
                self.dataloader_kwargs["persistent_workers"] = True
        
        # 更新用户提供的参数
        self.dataloader_kwargs.update(kwargs)

        if (
            getattr(self.dataset, "processor", None) is not None
            and self.dataloader_kwargs.get("num_workers", 0) > 0
        ):
            logger.warning(
                "检测到数据集依赖 processor 进行在线编码，已将 num_workers 设为 0，"
                "避免 spawn worker 丢失 processor 后返回未编码样本。"
            )
            self.dataloader_kwargs["num_workers"] = 0
            self.dataloader_kwargs.pop("prefetch_factor", None)
            self.dataloader_kwargs.pop("persistent_workers", None)
        
        # 设置采样器
        self._setup_sampler()
    
    def _detect_distributed(self) -> tuple[bool, int, int, int]:
        """检测分布式训练环境

        优先使用配置中的显式设置，否则自动检测 torch.distributed 环境。

        Returns:
            (is_distributed, world_size, rank, local_rank)
        """
        # 1. 如果配置显式启用分布式，使用配置值
        if self.config.distributed:
            ws = self.config.world_size or 1
            r = self.config.rank or 0
            lr = self.config.local_rank or 0
            logger.info(f"分布式训练已显式启用: world_size={ws}, rank={r}, local_rank={lr}")
            return True, ws, r, lr

        # 2. 自动检测 torch.distributed 环境
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            ws = torch.distributed.get_world_size()
            r = torch.distributed.get_rank()
            lr = int(os.environ.get("LOCAL_RANK", 0))
            logger.info(f"自动检测到分布式环境: world_size={ws}, rank={r}, local_rank={lr}")
            return True, ws, r, lr

        # 3. 检测环境变量（torchrun / launch 方式）
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            ws = int(os.environ["WORLD_SIZE"])
            r = int(os.environ["RANK"])
            lr = int(os.environ.get("LOCAL_RANK", 0))
            if ws > 1:
                logger.info(f"通过环境变量检测到分布式: world_size={ws}, rank={r}, local_rank={lr}")
                return True, ws, r, lr

        # 4. 非分布式环境
        return False, 1, 0, 0
    
    def _setup_sampler(self) -> None:
        """设置采样器（含分布式包装）"""
        # 1. 创建底层采样器
        if self.sampling_strategy == "balanced":
            base_sampler = TaskBalancedSampler(
                dataset=self.dataset,
                batch_size=self.dataloader_kwargs["batch_size"],
                drop_last=self.dataloader_kwargs["drop_last"],
                shuffle=self.config.shuffle
            )
            # 使用自定义采样器时，不能同时设置shuffle
            self.dataloader_kwargs["shuffle"] = False
        
        elif self.sampling_strategy == "round_robin":
            base_sampler = TaskRoundRobinSampler(
                dataset=self.dataset,
                shuffle=self.config.shuffle
            )
            self.dataloader_kwargs["shuffle"] = False
        
        elif self.sampling_strategy == "random":
            # 使用默认的随机采样
            base_sampler = None
            self.dataloader_kwargs["shuffle"] = self.config.shuffle
        
        else:
            raise ValueError(f"未知采样策略: {self.sampling_strategy}")

        # 2. 如果处于分布式环境，包装为 DistributedTaskSampler
        if self._distributed and base_sampler is not None:
            seed = self.config.distributed_seed or 42
            self.sampler = DistributedTaskSampler(
                sampler=base_sampler,
                world_size=self._world_size,
                rank=self._rank,
                shuffle=self.config.shuffle,
                seed=seed,
                drop_last=self.config.drop_last,
            )
            self.dataloader_kwargs["sampler"] = self.sampler
            logger.info(
                f"已启用分布式采样: rank={self._rank}/{self._world_size}, "
                f"本地样本数={len(self.sampler)}"
            )
        elif base_sampler is not None:
            self.sampler = base_sampler
            self.dataloader_kwargs["sampler"] = self.sampler
        else:
            self.sampler = None
    
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
        batch_size = self.dataloader_kwargs["batch_size"]
        if self.sampler:
            if self.dataloader_kwargs["drop_last"]:
                return len(self.sampler) // batch_size
            return (len(self.sampler) + batch_size - 1) // batch_size
        else:
            dataset_size = len(self.dataset)
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
            
            # 兼容不同的批次结构
            if "task_types" in batch:
                # 多任务类型批次
                for task_type in batch["task_types"]:
                    task_counts[task_type] += 1
                    total_samples += 1
            elif "task_type" in batch:
                # 单任务类型批次
                task_type = batch["task_type"]
                batch_size = batch["input_ids"].size(0) if "input_ids" in batch else 1
                task_counts[task_type] += batch_size
                total_samples += batch_size
            else:
                logger.warning(f"批次 {i} 中未找到任务类型信息")
        
        # 计算比例
        if total_samples > 0:
            task_distribution = {
                task: count / total_samples
                for task, count in task_counts.items()
            }
        else:
            task_distribution = {}
        
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
        stats = {
            "dataset_size": len(self.dataset),
            "batch_size": self.dataloader_kwargs["batch_size"],
            "num_batches": len(self),
            "sampling_strategy": self.sampling_strategy,
            "num_workers": self.dataloader_kwargs["num_workers"],
            "pin_memory": self.dataloader_kwargs["pin_memory"],
            "drop_last": self.dataloader_kwargs["drop_last"],
            "dataset_statistics": self.dataset.get_task_statistics()
        }
        # 分布式信息
        stats["distributed"] = self._distributed
        if self._distributed:
            stats["world_size"] = self._world_size
            stats["rank"] = self._rank
            stats["local_rank"] = self._local_rank
            if self.sampler is not None:
                stats["local_samples"] = len(self.sampler)
        return stats
