"""多任务数据集样本缓存（内存 LRU + 磁盘）。"""

from __future__ import annotations

import hashlib
import logging
import shutil
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from ..utils.torch_serialization import safe_torch_load_cpu
from .dataset_types import TaskSample

logger = logging.getLogger(__name__)


class DatasetSampleCache:
    """管理 ``MultiTaskDataset`` 的内存与磁盘样本缓存。"""

    def __init__(
        self,
        *,
        use_cache: bool,
        cache_dir: Optional[str],
        cache_max_size: int,
        lazy_load: bool,
        sample_index: List[Tuple[str, int, str, float]],
        samples: List[TaskSample],
    ) -> None:
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_max_size = cache_max_size
        self.lazy_load = lazy_load
        self.sample_index = sample_index
        self.samples = samples
        self._memory: OrderedDict[int, Dict[str, Any]] = OrderedDict()
        self._lock = threading.RLock()

    def get_memory(self, idx: int) -> Optional[Dict[str, Any]]:
        if not self.use_cache:
            return None
        with self._lock:
            if idx not in self._memory:
                return None
            self._memory.move_to_end(idx)
            return self._memory[idx]

    def put_memory(self, idx: int, data: Dict[str, Any]) -> None:
        if not self.use_cache:
            return
        with self._lock:
            if idx in self._memory:
                self._memory.move_to_end(idx)
                self._memory[idx] = data
                return
            if len(self._memory) >= self.cache_max_size:
                self._memory.popitem(last=False)
            self._memory[idx] = data

    def clear(self) -> None:
        with self._lock:
            self._memory.clear()
        if self.cache_dir is not None and self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            logger.info("已清除磁盘缓存: %s", self.cache_dir)

    def reset_memory_for_fork(self) -> None:
        """DataLoader 子进程反序列化后重建空内存缓存。"""
        with self._lock:
            self._memory = OrderedDict()

    def resolve_disk_path(self, idx: int) -> Path:
        if self.cache_dir is None:
            raise ValueError("cache_dir 未设置")
        if self.lazy_load and idx < len(self.sample_index):
            source_file = self.sample_index[idx][0]
        elif idx < len(self.samples):
            sample = self.samples[idx]
            source_file = sample.metadata.get("source_file") or sample.image_path
        else:
            source_file = str(idx)
        source_hash = hashlib.sha256(str(source_file).encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / source_hash / f"sample_{idx}.pt"

    def load_disk(
        self,
        idx: int,
        cache_path: Path,
        *,
        processor,
        get_sample: Callable[[int], TaskSample],
        load_image: Callable[[str], Any],
    ) -> Dict[str, Any]:
        cached = safe_torch_load_cpu(cache_path, context="Dataset cache")

        if "pixel_values" not in cached and processor is None:
            raise RuntimeError(
                "磁盘缓存不包含 pixel_values，且当前进程没有 processor，无法恢复图像张量。"
                "请使用 num_workers=0 在线编码，或在主进程中预热缓存后再读取。"
            )

        if "pixel_values" not in cached and processor is not None:
            sample = get_sample(idx)
            image = load_image(sample.image_path)
            image_inputs = processor(images=image, return_tensors="pt")
            pixel_values = image_inputs.get("pixel_values")
            if pixel_values is not None:
                cached["pixel_values"] = pixel_values.squeeze(0)
            else:
                raise RuntimeError(
                    f"processor 未为缓存样本 {idx} 返回 pixel_values，无法恢复磁盘缓存"
                )

        return cached

    def save_disk(self, data: Dict[str, Any], cache_path: Path) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_data = {k: v for k, v in data.items() if k != "pixel_values"}
        torch.save(cache_data, cache_path)

    @property
    def memory_size(self) -> int:
        with self._lock:
            return len(self._memory)
