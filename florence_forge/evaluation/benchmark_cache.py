"""Incremental cache helpers for benchmark evaluation."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

from ..utils.torch_serialization import safe_torch_load_cpu

logger = logging.getLogger(__name__)


def load_benchmark_artifact_cpu(path: Union[str, Path]) -> Any:
    """Load a benchmark torch artifact on CPU using the safe loader."""
    return safe_torch_load_cpu(path, context="Benchmark artifact")


class BenchmarkCache:
    """Read/write benchmark incremental cache files.

    New cache entries use ``torch.save`` with a small metadata envelope. Legacy
    pickle caches are ignored unless explicitly enabled in the benchmark config.
    """

    def __init__(
        self,
        cache_dir: Union[str, Path],
        config: Optional[Dict[str, Any]] = None,
        enable_incremental: bool = True,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or {}
        self.enable_incremental = enable_incremental

    @staticmethod
    def make_key(dataset_name: str, task_type: str, model_config: Dict[str, Any]) -> str:
        """Generate a stable cache key from dataset/task/model config."""
        config_str = json.dumps(model_config, sort_keys=True)
        cache_content = f"{dataset_name}_{task_type}_{config_str}"
        return hashlib.md5(cache_content.encode()).hexdigest()

    def save_results(self, cache_key: str, results: Dict[str, Any]) -> None:
        """Save benchmark results to the incremental cache."""
        if not self.enable_incremental:
            return

        cache_file = self.cache_dir / f"{cache_key}.pt"
        try:
            cache_data = {
                "results": results,
                "cache_version": 1,
                "timestamp": datetime.now().isoformat(),
            }
            torch.save(cache_data, cache_file)
            logger.info("保存缓存结果: %s", cache_key)
        except Exception as e:
            logger.warning("保存缓存失败: %s", e)

    def load_results(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Load benchmark results from cache when available."""
        if not self.enable_incremental:
            return None

        cache_file_pt = self.cache_dir / f"{cache_key}.pt"
        cache_file_pkl = self.cache_dir / f"{cache_key}.pkl"

        if cache_file_pt.exists():
            try:
                cached_data = load_benchmark_artifact_cpu(cache_file_pt)
                return cached_data.get("results", cached_data)
            except Exception as e:
                logger.warning("加载缓存失败: %s", e)

        if cache_file_pkl.exists():
            if not self.config.get("allow_legacy_pickle_cache", False):
                logger.warning(
                    "忽略旧版 pickle benchmark 缓存 %s；如确认文件可信，可设置 "
                    "allow_legacy_pickle_cache=True 后手动迁移。",
                    cache_file_pkl,
                )
                return None

            try:
                import pickle

                with open(cache_file_pkl, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning("加载旧版缓存失败: %s", e)
        return None
