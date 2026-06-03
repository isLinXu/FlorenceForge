"""多任务数据集 I/O：JSONL 加载/索引、HF 图像物化、持久化。"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from ..core.config import DataConfig
from .dataset_types import TaskSample

logger = logging.getLogger(__name__)


def extra_metadata_from_record(data: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in data.items() if k not in {"image", "prefix", "suffix"}}


def task_sample_from_jsonl_record(
    data: Dict[str, Any],
    *,
    task_type: str,
    image_base_path: Path,
    data_path: str,
    line_number: int,
    weight: float,
) -> TaskSample:
    image_path = image_base_path / data["image"]
    metadata = {"source_file": data_path, "line_number": line_number}
    metadata.update(extra_metadata_from_record(data))
    return TaskSample(
        task_type=task_type,
        image_path=str(image_path),
        prefix=data["prefix"],
        suffix=data["suffix"],
        weight=weight,
        metadata=metadata,
    )


def load_jsonl_task(
    samples: List[TaskSample],
    *,
    task_type: str,
    data_path: str,
    image_base_path: Path,
    weight: float,
    max_samples: Optional[int] = None,
) -> int:
    """将单个 JSONL 任务文件加载到 ``samples`` 列表。"""
    samples_loaded = 0
    try:
        with open(data_path, "r", encoding="utf-8") as handle:
            for line_num, line in enumerate(handle, 1):
                if max_samples and samples_loaded >= max_samples:
                    break
                try:
                    data = json.loads(line.strip())
                    samples.append(
                        task_sample_from_jsonl_record(
                            data,
                            task_type=task_type,
                            image_base_path=image_base_path,
                            data_path=data_path,
                            line_number=line_num,
                            weight=weight,
                        )
                    )
                    samples_loaded += 1
                except json.JSONDecodeError as exc:
                    logger.warning("解析JSON失败 %s:%s: %s", data_path, line_num, exc)
                except KeyError as exc:
                    logger.warning("缺少必要字段 %s:%s: %s", data_path, line_num, exc)
    except Exception as exc:
        logger.error("加载任务数据失败 %s: %s", task_type, exc)
        raise
    return samples_loaded


def scan_jsonl_task(
    sample_index: List[Tuple[str, int, str, float]],
    offset_cache: Dict[int, int],
    *,
    task_type: str,
    data_path: str,
    weight: float,
    max_samples: Optional[int] = None,
) -> int:
    """扫描 JSONL 并填充 ``sample_index`` 与 byte offset 缓存。"""
    task_count = 0
    try:
        with open(data_path, "rb") as handle:
            line_num = 0
            offset = 0
            for line_bytes in handle:
                line_num += 1
                if max_samples and task_count >= max_samples:
                    break
                if not line_bytes.strip():
                    offset += len(line_bytes)
                    continue
                try:
                    json.loads(line_bytes.strip().decode("utf-8"))
                    sample_index.append((data_path, line_num, task_type, weight))
                    idx = len(sample_index) - 1
                    offset_cache[idx] = offset
                    task_count += 1
                except (json.JSONDecodeError, UnicodeDecodeError):
                    logger.warning("解析JSON失败 %s:%s", data_path, line_num)
                offset += len(line_bytes)
    except Exception as exc:
        logger.error("扫描任务数据失败 %s: %s", task_type, exc)
        raise
    return task_count


def load_jsonl_sample_by_index(
    sample_index: List[Tuple[str, int, str, float]],
    offset_cache: Dict[int, int],
    image_base_path: Path,
    idx: int,
) -> TaskSample:
    data_path, line_number, task_type, weight = sample_index[idx]
    offset = offset_cache.get(idx)
    if offset is not None:
        with open(data_path, "rb") as handle:
            handle.seek(offset)
            line_bytes = handle.readline()
            data = json.loads(line_bytes.strip().decode("utf-8"))
    else:
        logger.warning("_sample_offset_cache 未命中 idx=%s，降级为线性扫描", idx)
        with open(data_path, "r", encoding="utf-8") as handle:
            for current_line_num, line in enumerate(handle, 1):
                if current_line_num == line_number:
                    data = json.loads(line.strip())
                    break
            else:
                raise IndexError(f"行号 {line_number} 在 {data_path} 中不存在")
    return task_sample_from_jsonl_record(
        data,
        task_type=task_type,
        image_base_path=image_base_path,
        data_path=data_path,
        line_number=line_number,
        weight=weight,
    )


def metadata_safe_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [metadata_safe_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): metadata_safe_value(item) for key, item in value.items()}
    return str(value)


def save_hf_image(
    image: Image.Image,
    idx: int,
    config: DataConfig,
    image_base_path: Path,
) -> Path:
    cache_dir = getattr(config, "cache_dir", None)
    if cache_dir:
        image_dir = Path(cache_dir) / "hf_images"
    elif str(image_base_path):
        image_dir = image_base_path / "hf_images"
    else:
        image_dir = Path("hf_dataset_images")
    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / f"sample_{idx}.png"
    image.save(image_path)
    return image_path


def materialize_hf_image(
    image_value: Any,
    idx: int,
    config: DataConfig,
    image_base_path: Path,
) -> Path:
    if isinstance(image_value, (str, os.PathLike)):
        image_path = Path(image_value)
        return image_path if image_path.is_absolute() else image_base_path / image_path

    if isinstance(image_value, dict):
        path_value = image_value.get("path")
        if path_value:
            image_path = Path(path_value)
            return image_path if image_path.is_absolute() else image_base_path / image_path
        if image_value.get("bytes") is not None:
            image = Image.open(BytesIO(image_value["bytes"])).convert("RGB")
            return save_hf_image(image, idx, config, image_base_path)

    if isinstance(image_value, Image.Image):
        return save_hf_image(image_value.convert("RGB"), idx, config, image_base_path)

    raise TypeError(
        "HF image column must contain a path, PIL Image, or dict with 'path'/'bytes'"
    )


def persist_dataset_json(
    file_path: Path,
    *,
    data_configs: List[Dict[str, Any]],
    image_base_path: Path,
    config: DataConfig,
    samples_data: List[Dict[str, Any]],
    task_weights: Dict[str, float],
) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "data_configs": data_configs,
        "image_base_path": str(image_base_path),
        "config": getattr(config, "__dict__", {}),
        "samples": samples_data,
        "task_weights": task_weights,
    }
    with open(file_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    logger.info("数据集已保存到: %s", file_path)


def restore_dataset_json(file_path: Path) -> Dict[str, Any]:
    with open(file_path, "r", encoding="utf-8") as handle:
        return json.load(handle)
