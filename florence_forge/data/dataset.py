"""FlorenceForge多任务数据集模块

提供多任务数据集的加载、处理和采样功能
"""

import numbers
import random
import logging
import torch
from collections import defaultdict, OrderedDict
from typing import Optional, Dict, Any, List, Union, Tuple
from pathlib import Path
from torch.utils.data import Dataset

from ..core.tasks import validate_task_name
from ..core.config import DataConfig
from .collate import Florence2Collator
from .dataset_types import TaskSample
from .dataset_sample_cache import DatasetSampleCache
from . import dataset_io
from . import dataset_encoding
# 图像缓存层已抽出到 image_cache.py；此处重新导出以保持历史导入路径
# （`florence_forge.data.dataset._load_image_cached`）与单测 patch 目标不变。
from .image_cache import (
    _load_image_cached,
)

logger = logging.getLogger(__name__)

# 历史导入路径兼容（实现见 dataset_types.py）
__all__ = ["MultiTaskDataset", "TaskSample"]


class MultiTaskDataset(Dataset):
    """多任务数据集

    支持多种任务类型的数据加载和处理。
    可通过 backend 参数与 VLM 后端解耦，支持任意 VLM 架构。
    """

    def __init__(
        self,
        data_configs: List[Dict[str, Any]],
        image_base_path: str = "",
        config: Optional[DataConfig] = None,
        processor=None,
        backend=None,
        lazy_load: bool = False
    ):
        """初始化多任务数据集

        Args:
            data_configs: 数据配置列表，每个配置包含任务类型和数据路径
            image_base_path: 图像文件基础路径
            config: 数据配置
            processor: 数据处理器（向后兼容，如提供 backend 则优先使用 backend）
            backend: VLM 后端实例，用于获取任务 prompt 和编码（推荐）
            lazy_load: 是否启用延迟加载（只扫描索引，按需读取样本）
        """
        self.data_configs = data_configs
        self.image_base_path = Path(image_base_path)
        self.config = config or DataConfig()
        self.processor = processor
        self.backend = backend  # 新增：VLM 后端
        self.lazy_load = lazy_load

        self.samples: List[TaskSample] = []
        self.task_weights: Dict[str, float] = {}
        self.task_indices: Dict[str, List[int]] = defaultdict(list)

        # 内置 collate_fn，供评估器等外部使用者直接引用
        self.collate_fn = Florence2Collator(pad_token_id=self._get_pad_token_id())

        # 延迟加载索引: List[(data_path, line_number, task_type, weight)]
        self._sample_index: List[Tuple[str, int, str, float]] = []
        # 文件偏移量缓存: {idx: (data_path, byte_offset, line_number, task_type, weight)}
        # 用 dict 实现 O(1) 随机访问，避免每次重新扫描文件
        self._sample_offset_cache: Dict[int, Tuple[str, int, int, str, float]] = {}

        self._validate_configs()

        if self.lazy_load:
            self._scan_all_tasks()
        else:
            self._load_all_tasks()

        self._calculate_task_weights()
        self._build_task_indices()
        self._init_sample_cache()
        self._init_augmentation()

        if not self.lazy_load and self.use_cache and self.processor is not None:
            self.preprocess_and_cache()

        if self.lazy_load:
            logger.info(
                "数据集初始化完成（延迟加载模式），总样本数: %d",
                len(self._sample_index),
            )
        else:
            logger.info("数据集初始化完成，总样本数: %d", len(self.samples))

    def _init_sample_cache(self) -> None:
        """初始化/重建样本缓存（内存 LRU + 磁盘）。"""
        self._sample_cache = DatasetSampleCache(
            use_cache=getattr(self.config, "use_cache", False),
            cache_dir=getattr(self.config, "cache_dir", None),
            cache_max_size=getattr(self.config, "cache_max_size", 10000),
            lazy_load=self.lazy_load,
            sample_index=self._sample_index,
            samples=self.samples,
        )

    def _init_augmentation(self) -> None:
        """根据 ``DataConfig`` 初始化增强器实例。

        缓存模式或 ``use_augmentation=False`` 时全部置 ``None`` 以保证确定性。
        """
        self._image_aug = None
        self._text_aug = None
        self._bbox_aug = None

        if not getattr(self.config, "use_augmentation", False):
            return
        if self.use_cache:
            return

        prob = getattr(self.config, "augmentation_prob", 0.5)

        if getattr(self.config, "augment_image", True):
            from .augmentation import ImageAugmentation
            self._image_aug = ImageAugmentation(probability=prob)
        if getattr(self.config, "augment_text", False):
            from .augmentation import TextAugmentation
            self._text_aug = TextAugmentation(probability=prob)
        if getattr(self.config, "augment_bbox", True):
            from .augmentation import BBoxAugmentation
            self._bbox_aug = BBoxAugmentation(probability=prob)

    def _maybe_augment_bboxes(self, sample: "TaskSample") -> "TaskSample":
        """对样本中的归一化 bbox 执行增强（原地副本，不污染原始样本）。"""
        if getattr(self, "_bbox_aug", None) is None:
            return sample
        bboxes = sample.metadata.get("bboxes")
        if not bboxes:
            return sample
        augmented = self._bbox_aug.apply_augmentations(
            [dict(bb) for bb in bboxes]
        )
        new_metadata = dict(sample.metadata)
        new_metadata["bboxes"] = augmented
        return TaskSample(
            task_type=sample.task_type,
            image_path=sample.image_path,
            prefix=sample.prefix,
            suffix=sample.suffix,
            weight=sample.weight,
            metadata=new_metadata,
        )

    def _maybe_augment_text(self, sample: "TaskSample") -> "TaskSample":
        """对 suffix 执行文本增强（副本，不污染 ``self.samples``）。"""
        if getattr(self, "_text_aug", None) is None:
            return sample
        augmented_suffix = self._text_aug.apply_augmentations(sample.suffix)
        if augmented_suffix == sample.suffix:
            return sample
        return TaskSample(
            task_type=sample.task_type,
            image_path=sample.image_path,
            prefix=sample.prefix,
            suffix=augmented_suffix,
            weight=sample.weight,
            metadata=dict(sample.metadata),
        )

    def _maybe_augment_image(self, image):
        """对 PIL 图像执行增强。"""
        if getattr(self, "_image_aug", None) is None:
            return image
        return self._image_aug.apply_augmentations(image)

    def _apply_sample_augmentations(self, sample: "TaskSample") -> "TaskSample":
        """按 bbox → text 顺序对样本元数据执行增强。"""
        sample = self._maybe_augment_bboxes(sample)
        return self._maybe_augment_text(sample)

    @property
    def use_cache(self) -> bool:
        return self._sample_cache.use_cache

    @use_cache.setter
    def use_cache(self, value: bool) -> None:
        self._sample_cache.use_cache = value

    @property
    def cache_dir(self) -> Optional[str]:
        return str(self._sample_cache.cache_dir) if self._sample_cache.cache_dir else None

    @cache_dir.setter
    def cache_dir(self, value: Optional[str]) -> None:
        self._sample_cache.cache_dir = Path(value) if value else None

    def _validate_configs(self) -> None:
        """验证数据配置"""
        for i, config in enumerate(self.data_configs):
            if "task_type" not in config:
                raise ValueError(f"配置 {i} 缺少 task_type 字段")
            if "data_path" not in config:
                raise ValueError(f"配置 {i} 缺少 data_path 字段")
            
            task_type = config["task_type"]
            if not validate_task_name(task_type):
                raise ValueError(f"未知任务类型: {task_type}")
            
            data_path = Path(config["data_path"])
            if not data_path.exists():
                raise FileNotFoundError(f"数据文件不存在: {data_path}")

    def _get_pad_token_id(self) -> int:
        """返回 tokenizer 的 pad token id；processor 不可用时回退到 0。"""
        tokenizer = getattr(self.processor, "tokenizer", None) if self.processor is not None else None
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None and self.processor is not None:
            pad_token_id = getattr(self.processor, "pad_token_id", None)
        if isinstance(pad_token_id, numbers.Integral):
            return int(pad_token_id)
        return 0
    
    def _load_all_tasks(self) -> None:
        """加载所有任务数据"""
        task_counts = defaultdict(int)
        
        for config in self.data_configs:
            task_type = config["task_type"]
            data_path = config["data_path"]
            weight = config.get("weight", 1.0)
            
            logger.info(f"正在加载任务: {task_type}, 路径: {data_path}")
            
            samples_loaded = dataset_io.load_jsonl_task(
                self.samples,
                task_type=task_type,
                data_path=data_path,
                image_base_path=self.image_base_path,
                weight=weight,
                max_samples=getattr(self.config, "max_samples_per_task", None),
            )
            task_counts[task_type] += samples_loaded
        
        logger.info(f"数据加载完成，各任务样本数: {dict(task_counts)}")
    
    def _scan_all_tasks(self) -> None:
        """扫描所有任务数据文件，建立索引并预构建 byte offset 缓存。"""
        task_counts = defaultdict(int)

        for config in self.data_configs:
            task_type = config["task_type"]
            data_path = config["data_path"]
            weight = config.get("weight", 1.0)
            max_samples = getattr(self.config, "max_samples_per_task", None)

            logger.info("正在扫描任务索引: %s, 路径: %s", task_type, data_path)
            loaded = dataset_io.scan_jsonl_task(
                self._sample_index,
                self._sample_offset_cache,
                task_type=task_type,
                data_path=data_path,
                weight=weight,
                max_samples=max_samples,
            )
            task_counts[task_type] += loaded

        logger.info(
            "索引扫描完成，各任务样本数: %s，已构建 offset 缓存",
            dict(task_counts),
        )

    def _load_sample_by_index(self, idx: int) -> TaskSample:
        return dataset_io.load_jsonl_sample_by_index(
            self._sample_index,
            self._sample_offset_cache,
            self.image_base_path,
            idx,
        )

    def _calculate_task_weights(self) -> None:
        """计算任务权重以实现平衡采样"""
        if not getattr(self.config, 'use_balanced_sampling', False):
            return

        task_counts = defaultdict(int)
        if self.lazy_load:
            for _, _, task_type, _ in self._sample_index:
                task_counts[task_type] += 1
        else:
            for sample in self.samples:
                task_counts[sample.task_type] += 1

        if not task_counts:
            return

        max_count = max(task_counts.values())
        for task_type, count in task_counts.items():
            self.task_weights[task_type] = max_count / count

        logger.info(f"任务权重: {self.task_weights}")

    def _build_task_indices(self) -> None:
        """构建任务索引映射"""
        if self.lazy_load:
            for idx, (_, _, task_type, _) in enumerate(self._sample_index):
                self.task_indices[task_type].append(idx)
        else:
            for idx, sample in enumerate(self.samples):
                self.task_indices[sample.task_type].append(idx)
    
    def __len__(self) -> int:
        """返回数据集大小"""
        if self.lazy_load:
            return len(self._sample_index)
        return len(self.samples)

    def _get_sample(self, idx: int) -> TaskSample:
        """获取样本（支持延迟加载和预加载）

        Args:
            idx: 样本索引

        Returns:
            TaskSample 对象
        """
        if self.lazy_load:
            return self._load_sample_by_index(idx)
        return self.samples[idx]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取单个样本

        Args:
            idx: 样本索引

        Returns:
            处理后的样本数据，包含正确的labels用于训练监督
        """
        cached_memory = self._sample_cache.get_memory(idx)
        if cached_memory is not None:
            return cached_memory

        # 2. 如果 processor 不可用（如在 DataLoader 子进程中），尝试从磁盘缓存加载
        if self.processor is None and self._sample_cache.cache_dir is not None:
            cache_path = self._sample_cache.resolve_disk_path(idx)
            if cache_path.exists():
                try:
                    cached = self._sample_cache.load_disk(
                        idx,
                        cache_path,
                        processor=None,
                        get_sample=self._get_sample,
                        load_image=_load_image_cached,
                    )
                    self._sample_cache.put_memory(idx, cached)
                    return cached
                except Exception as e:
                    logger.warning(f"子进程加载磁盘缓存失败 {cache_path}: {e}")
            # 无法加载缓存，返回原始格式（需要上层处理）
            return dataset_encoding.unencoded_sample_dict(self._get_sample(idx))

        sample = self._apply_sample_augmentations(self._get_sample(idx))
        image = self._maybe_augment_image(_load_image_cached(sample.image_path))

        if self.processor is not None:
            result = dataset_encoding.encode_training_sample(
                sample=sample,
                image=image,
                processor=self.processor,
                backend=self.backend,
            )
        else:
            prompt, answer = dataset_encoding.build_prompt_and_answer(
                sample, backend=self.backend
            )
            result = dataset_encoding.raw_image_result(image, prompt, answer, sample)

        if self.use_cache:
            self._sample_cache.put_memory(idx, result)
        if self._sample_cache.cache_dir is not None:
            try:
                cache_path = self._sample_cache.resolve_disk_path(idx)
                self._sample_cache.save_disk(result, cache_path)
            except Exception as e:
                logger.warning(f"保存缓存失败 {idx}: {e}")

        return result

    def _get_task_prompt(self, task_type: str) -> str:
        return dataset_encoding.get_task_prompt(task_type, self.backend)

    def _build_prompt_and_answer(self, sample: TaskSample) -> Tuple[str, str]:
        return dataset_encoding.build_prompt_and_answer(sample, backend=self.backend)

    def _default_prepare_labels(
        self,
        encoded_prompt: Dict[str, torch.Tensor],
        encoded_full: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return dataset_encoding.default_prepare_labels(encoded_prompt, encoded_full)

    def preprocess_and_cache(self, max_workers: int = 4) -> None:
        """预编码所有样本并缓存到内存（可选持久化到磁盘）

        在训练前一次性将所有样本通过 processor 编码为张量，
        后续 __getitem__ 直接从缓存读取，避免重复的图像加载与 tokenization。
        
        支持线程池并行编码，避免 processor/backend 在 spawn 多进程下不可 pickle 的问题。

        Args:
            max_workers: 并行编码的 worker 数量（>1 时使用多进程并行）
        """
        if self.processor is None:
            logger.warning("Processor 未设置，无法预编码缓存")
            return

        if self.lazy_load:
            logger.warning("延迟加载模式下不支持预编码缓存，请在非延迟加载模式下使用")
            return

        logger.info(f"开始预编码缓存，样本数: {len(self)}，并行度: {max_workers} ...")
        
        # 方案 1：线程池并行（有界提交 future，避免大数据集创建海量任务对象）
        if max_workers > 1:
            self._parallel_preprocess(max_workers)
        else:
            # 方案 2：单进程顺序处理
            self._sequential_preprocess()
    
    def _sequential_preprocess(self) -> None:
        """单进程顺序预编码（内部方法）"""
        cache_hits = 0
        cache_misses = 0

        for idx in range(len(self)):
            if self._sample_cache.cache_dir is not None:
                cache_path = self._sample_cache.resolve_disk_path(idx)
                if cache_path.exists():
                    try:
                        cached = self._sample_cache.load_disk(
                            idx,
                            cache_path,
                            processor=self.processor,
                            get_sample=self._get_sample,
                            load_image=_load_image_cached,
                        )
                        self._sample_cache.put_memory(idx, cached)
                        cache_hits += 1
                        continue
                    except Exception as e:
                        logger.warning(f"加载缓存失败 {cache_path}: {e}")

            try:
                encoded = self.__getitem__(idx)
                cache_misses += 1
                if self._sample_cache.cache_dir is not None:
                    cache_path = self._sample_cache.resolve_disk_path(idx)
                    self._sample_cache.save_disk(encoded, cache_path)
            except Exception as e:
                logger.warning(f"预编码样本 {idx} 失败: {e}")
                continue

        logger.info(
            "预编码完成: 内存缓存 %d 条, 磁盘命中 %d 条, 新编码 %d 条",
            self._sample_cache.memory_size,
            cache_hits,
            cache_misses,
        )
    
    def _parallel_preprocess(self, num_workers: int) -> None:
        """并行预编码（内部方法）

        使用线程池避免 processor/backend 在 spawn 多进程下不可 pickle 的问题。
        """
        from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
        from tqdm import tqdm
        
        cache_hits = 0
        cache_misses = 0

        def encode_idx(idx: int) -> tuple[int, Dict[str, Any], bool]:
            if self._sample_cache.cache_dir is not None:
                cache_path = self._sample_cache.resolve_disk_path(idx)
                if cache_path.exists():
                    return (
                        idx,
                        self._sample_cache.load_disk(
                            idx,
                            cache_path,
                            processor=self.processor,
                            get_sample=self._get_sample,
                            load_image=_load_image_cached,
                        ),
                        True,
                    )
            return idx, self.__getitem__(idx), False

        total_samples = len(self)
        max_pending = max(1, num_workers * 4)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            pending = set()
            next_idx = 0

            def submit_until_full() -> None:
                nonlocal next_idx
                while next_idx < total_samples and len(pending) < max_pending:
                    pending.add(executor.submit(encode_idx, next_idx))
                    next_idx += 1

            submit_until_full()
            with tqdm(total=total_samples, desc="并行预编码") as progress:
                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        try:
                            idx, encoded, from_disk = future.result()
                        except Exception as e:
                            logger.warning(f"预编码样本失败: {e}")
                            progress.update(1)
                            continue

                        self._sample_cache.put_memory(idx, encoded)
                        if from_disk:
                            cache_hits += 1
                        else:
                            cache_misses += 1

                        if self._sample_cache.cache_dir is not None:
                            cache_path = self._sample_cache.resolve_disk_path(idx)
                            if not cache_path.exists():
                                self._sample_cache.save_disk(encoded, cache_path)
                        progress.update(1)

                    submit_until_full()
        
        logger.info(
            "预编码完成（并行）: 内存缓存 %d 条, 磁盘命中 %d 条, 新编码 %d 条",
            self._sample_cache.memory_size,
            cache_hits,
            cache_misses,
        )

    def clear_cache(self) -> None:
        """清除内存与磁盘缓存"""
        self._sample_cache.clear()

    # 向后兼容（测试与旧代码仍可能调用）
    @property
    def _cache_index(self) -> OrderedDict:
        return self._sample_cache._memory

    @property
    def _cache_max_size(self) -> int:
        return self._sample_cache.cache_max_size

    def _cache_put(self, idx: int, data: Dict[str, Any]) -> None:
        self._sample_cache.put_memory(idx, data)

    def _get_cache_path(self, idx: int) -> Path:
        return self._sample_cache.resolve_disk_path(idx)

    def _load_cached_sample(self, idx: int, cache_path: Path) -> Dict[str, Any]:
        return self._sample_cache.load_disk(
            idx,
            cache_path,
            processor=self.processor,
            get_sample=self._get_sample,
            load_image=_load_image_cached,
        )

    def _save_cached_sample(self, data: Dict[str, Any], cache_path: Path) -> None:
        self._sample_cache.save_disk(data, cache_path)

    # ------------------------------------------------------------------
    # 多进程序列化支持（用于 DataLoader num_workers > 0）
    # ------------------------------------------------------------------

    def __getstate__(self) -> Dict[str, Any]:
        """序列化状态（排除不可序列化的内存缓存、processor 和 backend）"""
        state = self.__dict__.copy()
        state.pop("_sample_cache", None)
        state["processor"] = None
        state["backend"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """反序列化状态"""
        self.__dict__.update(state)
        self._init_sample_cache()
        self._sample_cache.reset_memory_for_fork()

    def get_task_samples(self, task_type: str) -> List[int]:
        """获取指定任务的样本索引
        
        Args:
            task_type: 任务类型
            
        Returns:
            样本索引列表
        """
        return self.task_indices.get(task_type, [])
    
    def get_task_statistics(self) -> Dict[str, Any]:
        """获取任务统计信息
        
        Returns:
            统计信息字典
        """
        task_counts = defaultdict(int)
        if self.lazy_load:
            for _, _, task_type, _ in self._sample_index:
                task_counts[task_type] += 1
            total_samples = len(self._sample_index)
        else:
            for sample in self.samples:
                task_counts[sample.task_type] += 1
            total_samples = len(self.samples)

        return {
            "total_samples": total_samples,
            "task_counts": dict(task_counts),
            "task_weights": self.task_weights,
            "num_tasks": len(task_counts)
        }
    
    def sample_by_task(self, task_type: str, num_samples: int) -> List[int]:
        """按任务类型采样
        
        Args:
            task_type: 任务类型
            num_samples: 采样数量
            
        Returns:
            采样的索引列表
        """
        task_indices = self.get_task_samples(task_type)
        if not task_indices:
            return []
            
        if num_samples >= len(task_indices):
            return task_indices.copy()
            
        return random.sample(task_indices, num_samples)
    
    def create_task_subset(
        self, task_type: str, max_samples: Optional[int] = None
    ) -> 'MultiTaskDataset':
        """创建特定任务的子集（供评估器使用）

        Args:
            task_type: 任务类型
            max_samples: 最大样本数限制

        Returns:
            仅包含指定任务样本的子数据集
        """
        indices = self.task_indices.get(task_type, [])
        if max_samples is not None and len(indices) > max_samples:
            indices = indices[:max_samples]
        return self.create_subset(indices)

    @classmethod
    def from_existing(
        cls,
        source: "MultiTaskDataset",
        *,
        samples: List[TaskSample],
        sample_index: List[Tuple[str, int, str, float]],
        offset_cache: Dict[int, Tuple[str, int, int, str, float]],
    ) -> "MultiTaskDataset":
        """从已有数据集构造一个共享配置的新实例（子集/视图）。

        这是所有绕过 ``__init__`` 的构造路径的单一事实源。它显式拷贝
        ``source`` 的配置级属性，并复用与 ``__init__`` 相同的后置初始化
        （``_build_task_indices`` / ``_init_sample_cache`` / ``_init_augmentation``），
        因此当 ``__init__`` 新增属性时只需在此处同步一次，避免子集实例出现
        属性缺失或初始化不一致。
        """
        subset = cls.__new__(cls)
        # 配置级属性（与源共享，不重新加载数据）
        subset.data_configs = source.data_configs
        subset.image_base_path = source.image_base_path
        subset.config = source.config
        subset.processor = source.processor
        subset.backend = source.backend
        subset.lazy_load = source.lazy_load
        subset.task_weights = source.task_weights.copy()
        subset.collate_fn = source.collate_fn
        # 样本数据
        subset.samples = samples
        subset._sample_index = sample_index
        subset._sample_offset_cache = offset_cache
        subset.task_indices = defaultdict(list)
        # 复用 __init__ 的后置初始化步骤，保证一致性
        subset._build_task_indices()
        subset._init_sample_cache()
        subset._sample_cache.use_cache = source.use_cache
        subset._sample_cache.cache_dir = source._sample_cache.cache_dir
        subset._init_augmentation()
        return subset

    def create_subset(self, indices: List[int]) -> 'MultiTaskDataset':
        """创建子集

        Args:
            indices: 样本索引列表

        Returns:
            子数据集
        """
        if self.lazy_load:
            subset_index = [self._sample_index[i] for i in indices]
            subset_offset_cache = {
                new_idx: self._sample_offset_cache[old_idx]
                for new_idx, old_idx in enumerate(indices)
                if old_idx in self._sample_offset_cache
            }
            subset_samples: List[TaskSample] = []
        else:
            subset_samples = [self.samples[i] for i in indices]
            subset_index = []
            subset_offset_cache = {}

        return MultiTaskDataset.from_existing(
            self,
            samples=subset_samples,
            sample_index=subset_index,
            offset_cache=subset_offset_cache,
        )

    @classmethod
    def from_hf_dataset(
        cls,
        hf_dataset,
        task_type: str,
        image_column: str = "image",
        text_column: str = "text",
        config: Optional[DataConfig] = None,
        image_base_path: str = "",
        processor=None,
        backend=None,
        weight: float = 1.0,
    ) -> "MultiTaskDataset":
        """从 HuggingFace Dataset 风格对象创建 MultiTaskDataset。

        支持 ``datasets.Dataset``、list[dict] 等可迭代样本源。图片列可以是
        文件路径、PIL Image，或包含 ``path``/``bytes`` 的字典。
        """
        if not validate_task_name(task_type):
            raise ValueError(f"未知任务类型: {task_type}")

        config = config or DataConfig()
        dataset = cls.__new__(cls)
        dataset.data_configs = [{
            "task_type": task_type,
            "data_path": "<hf_dataset>",
            "weight": weight,
        }]
        dataset.image_base_path = Path(image_base_path)
        dataset.config = config
        dataset.processor = processor
        dataset.backend = backend
        dataset.lazy_load = False

        dataset.samples = []
        dataset.task_weights = {}
        dataset.task_indices = defaultdict(list)
        dataset.collate_fn = Florence2Collator(pad_token_id=dataset._get_pad_token_id())
        dataset._sample_index = []
        dataset._sample_offset_cache = {}

        max_samples = getattr(config, "max_samples_per_task", None)
        for idx, row in enumerate(hf_dataset):
            if max_samples is not None and idx >= max_samples:
                break
            if not isinstance(row, dict):
                raise TypeError(f"HF dataset row {idx} must be a dict-like object")
            if image_column not in row:
                raise KeyError(f"HF dataset row {idx} missing image column '{image_column}'")

            suffix = row.get("suffix", row.get("answer", row.get(text_column, "")))
            prefix = row.get("prefix", row.get("prompt", ""))
            image_path = dataset_io.materialize_hf_image(
                row[image_column],
                idx=idx,
                config=config,
                image_base_path=dataset.image_base_path,
            )
            metadata = {
                "source": "hf_dataset",
                "source_index": idx,
            }
            excluded = {image_column, text_column, "prefix", "prompt", "suffix", "answer"}
            metadata.update({
                key: dataset_io.metadata_safe_value(value)
                for key, value in row.items()
                if key not in excluded
            })

            dataset.samples.append(TaskSample(
                task_type=task_type,
                image_path=str(image_path),
                prefix=str(prefix or ""),
                suffix=str(suffix or ""),
                weight=weight,
                metadata=metadata,
            ))

        dataset._calculate_task_weights()
        dataset._build_task_indices()
        dataset._init_sample_cache()

        if dataset.use_cache and dataset.processor is not None:
            dataset.preprocess_and_cache()

        logger.info(f"HF dataset 已加载为 MultiTaskDataset，样本数: {len(dataset.samples)}")
        return dataset

    _materialize_hf_image = staticmethod(dataset_io.materialize_hf_image)
    _save_hf_image = staticmethod(dataset_io.save_hf_image)
    _metadata_safe_value = staticmethod(dataset_io.metadata_safe_value)

    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """保存数据集到文件
        
        Args:
            file_path: 文件路径
        """
        file_path = Path(file_path)
        if self.lazy_load:
            samples_data = [self._get_sample(idx).to_dict() for idx in range(len(self))]
        else:
            samples_data = [sample.to_dict() for sample in self.samples]
        dataset_io.persist_dataset_json(
            file_path,
            data_configs=self.data_configs,
            image_base_path=self.image_base_path,
            config=self.config,
            samples_data=samples_data,
            task_weights=self.task_weights,
        )
    
    @classmethod
    def load_from_file(
        cls,
        file_path: Union[str, Path],
        processor=None
    ) -> 'MultiTaskDataset':
        """从文件加载数据集
        
        Args:
            file_path: 文件路径
            processor: 数据处理器
            
        Returns:
            数据集实例
        """
        data = dataset_io.restore_dataset_json(Path(file_path))
        dataset = cls.__new__(cls)
        dataset.data_configs = data["data_configs"]
        dataset.image_base_path = Path(data["image_base_path"])
        dataset.config = DataConfig(**data.get("config", {}))
        dataset.processor = processor
        dataset.backend = None
        dataset.lazy_load = False
        dataset.samples = [TaskSample.from_dict(s) for s in data["samples"]]
        dataset.task_weights = data["task_weights"]

        dataset._sample_index = []
        dataset._sample_offset_cache = {}
        dataset.collate_fn = Florence2Collator(pad_token_id=dataset._get_pad_token_id())
        dataset.task_indices = defaultdict(list)
        dataset._build_task_indices()
        dataset._init_sample_cache()
        
        logger.info(f"数据集已从文件加载: {file_path}")
        return dataset
