"""数据集分布剖析器。

生成样本长度、图像尺寸、任务类别分布等统计报告，
用于训练前数据质量可观测与类别不平衡预警。
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union

from PIL import Image

if TYPE_CHECKING:
    from .dataset import MultiTaskDataset

logger = logging.getLogger(__name__)


def _histogram(values: List[float], bins: int = 10) -> Dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "bins": []}
    vmin = min(values)
    vmax = max(values)
    if vmin == vmax:
        return {
            "count": len(values),
            "min": vmin,
            "max": vmax,
            "mean": vmin,
            "bins": [{"lo": vmin, "hi": vmax, "count": len(values)}],
        }
    width = (vmax - vmin) / bins
    counts = [0] * bins
    for v in values:
        idx = min(int((v - vmin) / width), bins - 1)
        counts[idx] += 1
    return {
        "count": len(values),
        "min": vmin,
        "max": vmax,
        "mean": sum(values) / len(values),
        "bins": [
            {"lo": vmin + i * width, "hi": vmin + (i + 1) * width, "count": counts[i]}
            for i in range(bins)
        ],
    }


class DataProfiler:
    """数据集分布剖析器。"""

    def __init__(self, imbalance_threshold: float = 10.0) -> None:
        """初始化剖析器。

        Args:
            imbalance_threshold: 最大/最小任务样本数比值超过此阈值时发出警告。
        """
        self.imbalance_threshold = imbalance_threshold

    def profile_dataset(self, dataset: "MultiTaskDataset") -> Dict[str, Any]:
        """剖析 ``MultiTaskDataset`` 的样本分布。"""
        prefix_lengths: List[int] = []
        suffix_lengths: List[int] = []
        widths: List[int] = []
        heights: List[int] = []
        task_counts: Counter[str] = Counter()
        duplicate_keys: Counter[str] = Counter()
        image_errors = 0

        for sample in dataset.samples:
            task_counts[sample.task_type] += 1
            prefix_lengths.append(len(sample.prefix or ""))
            suffix_lengths.append(len(sample.suffix or ""))
            dup_key = f"{sample.image_path}|{sample.prefix}|{sample.suffix}"
            duplicate_keys[dup_key] += 1

            try:
                with Image.open(sample.image_path) as img:
                    w, h = img.size
                    widths.append(w)
                    heights.append(h)
            except Exception:
                image_errors += 1

        return self._build_report(
            total_samples=len(dataset.samples),
            task_counts=task_counts,
            prefix_lengths=prefix_lengths,
            suffix_lengths=suffix_lengths,
            widths=widths,
            heights=heights,
            duplicate_keys=duplicate_keys,
            image_errors=image_errors,
        )

    def profile_jsonl(
        self,
        data_path: Union[str, Path],
        image_base_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """剖析 JSONL 训练文件（无需构建完整 ``MultiTaskDataset``）。"""
        data_path = Path(data_path)
        base = Path(image_base_path) if image_base_path else data_path.parent

        prefix_lengths: List[int] = []
        suffix_lengths: List[int] = []
        widths: List[int] = []
        heights: List[int] = []
        task_counts: Counter[str] = Counter()
        duplicate_keys: Counter[str] = Counter()
        image_errors = 0

        with data_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                task_type = record.get("task_type", "UNKNOWN")
                task_counts[task_type] += 1
                prefix = record.get("prefix", "")
                suffix = record.get("suffix", "")
                prefix_lengths.append(len(prefix))
                suffix_lengths.append(len(suffix))
                dup_key = f"{record.get('image', '')}|{prefix}|{suffix}"
                duplicate_keys[dup_key] += 1

                image_name = record.get("image")
                if not image_name:
                    continue
                image_path = base / image_name
                try:
                    with Image.open(image_path) as img:
                        w, h = img.size
                        widths.append(w)
                        heights.append(h)
                except Exception:
                    image_errors += 1

        return self._build_report(
            total_samples=sum(task_counts.values()),
            task_counts=task_counts,
            prefix_lengths=prefix_lengths,
            suffix_lengths=suffix_lengths,
            widths=widths,
            heights=heights,
            duplicate_keys=duplicate_keys,
            image_errors=image_errors,
        )

    def _build_report(
        self,
        *,
        total_samples: int,
        task_counts: Counter[str],
        prefix_lengths: List[int],
        suffix_lengths: List[int],
        widths: List[int],
        heights: List[int],
        duplicate_keys: Counter[str],
        image_errors: int,
    ) -> Dict[str, Any]:
        counts = list(task_counts.values()) or [0]
        max_count = max(counts)
        min_count = min(counts) if counts else 0
        imbalance_ratio = (max_count / min_count) if min_count > 0 else float("inf")

        aspect_ratios = [
            w / h for w, h in zip(widths, heights, strict=False) if h > 0
        ]
        duplicate_samples = sum(c - 1 for c in duplicate_keys.values() if c > 1)

        warnings: List[str] = []
        if imbalance_ratio > self.imbalance_threshold:
            warnings.append(
                f"任务样本不平衡比 {imbalance_ratio:.1f} 超过阈值 "
                f"{self.imbalance_threshold:.1f}"
            )
        if duplicate_samples > 0:
            warnings.append(f"检测到 {duplicate_samples} 条重复样本")
        if image_errors > 0:
            warnings.append(f"{image_errors} 张图像无法读取尺寸")

        for warning in warnings:
            logger.warning("DataProfiler: %s", warning)

        return {
            "total_samples": total_samples,
            "task_distribution": dict(task_counts),
            "imbalance_ratio": imbalance_ratio,
            "prefix_length": _histogram([float(v) for v in prefix_lengths]),
            "suffix_length": _histogram([float(v) for v in suffix_lengths]),
            "image_width": _histogram([float(v) for v in widths]),
            "image_height": _histogram([float(v) for v in heights]),
            "aspect_ratio": _histogram(aspect_ratios),
            "duplicate_samples": duplicate_samples,
            "image_read_errors": image_errors,
            "warnings": warnings,
        }
