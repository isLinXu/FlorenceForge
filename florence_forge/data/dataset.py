"""FlorenceForge多任务数据集模块

提供多任务数据集的加载、处理和采样功能
"""

import json
import hashlib
import os
import numbers
import random
import logging
import threading
import torch
from io import BytesIO
from collections import defaultdict, OrderedDict
from typing import Optional, Dict, Any, List, Union, Tuple
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image

from ..core.tasks import FLORENCE2_TASKS, validate_task_name
from ..core.config import DataConfig
from ..utils.torch_serialization import safe_torch_load_cpu
from .collate import Florence2Collator
# 图像缓存层已抽出到 image_cache.py；此处重新导出以保持历史导入路径
# （`florence_forge.data.dataset._load_image_cached`）与单测 patch 目标不变。
from .image_cache import (
    _ImagePayloadCacheInfo,
    _IMAGE_PAYLOAD_CACHE_DEFAULT_MAX_BYTES,
    _load_image_payload_cached,
    _image_payload_cache_clear,
    _image_payload_cache_info,
    _image_payload_cache_current_bytes,
    _set_image_payload_cache_max_bytes,
    _load_image_cached,
)

logger = logging.getLogger(__name__)


class TaskSample:
    """任务样本数据结构"""
    
    def __init__(
        self,
        task_type: str,
        image_path: str,
        prefix: str,
        suffix: str,
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.task_type = task_type
        self.image_path = image_path
        self.prefix = prefix
        self.suffix = suffix
        self.weight = weight
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "task_type": self.task_type,
            "image_path": self.image_path,
            "prefix": self.prefix,
            "suffix": self.suffix,
            "weight": self.weight,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskSample':
        """从字典创建样本"""
        return cls(
            task_type=data["task_type"],
            image_path=data["image_path"],
            prefix=data["prefix"],
            suffix=data["suffix"],
            weight=data.get("weight", 1.0),
            metadata=data.get("metadata", {})
        )

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

        # 缓存配置（使用 OrderedDict 实现 O(1) LRU）
        self.use_cache = getattr(self.config, 'use_cache', False)
        self.cache_dir = getattr(self.config, 'cache_dir', None)
        self._cache_index: OrderedDict[int, Dict[str, Any]] = OrderedDict()
        self._cache_max_size: int = getattr(self.config, 'cache_max_size', 10000)
        self._cache_lock = threading.RLock()

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
            self._calculate_task_weights()
            self._build_task_indices()
            logger.info(f"数据集初始化完成（延迟加载模式），总样本数: {len(self._sample_index)}")
        else:
            self._load_all_tasks()
            self._calculate_task_weights()
            self._build_task_indices()

            # 如果启用缓存，执行预编码
            if self.use_cache and self.processor is not None:
                self.preprocess_and_cache()

            logger.info(f"数据集初始化完成，总样本数: {len(self.samples)}")
    
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
            
            samples_loaded = self._load_task_data(task_type, data_path, weight)
            task_counts[task_type] += samples_loaded
        
        logger.info(f"数据加载完成，各任务样本数: {dict(task_counts)}")
    
    def _load_task_data(self, task_type: str, data_path: str, weight: float) -> int:
        """加载单个任务的数据
        
        Args:
            task_type: 任务类型
            data_path: 数据文件路径
            weight: 任务权重
            
        Returns:
            加载的样本数量
        """
        samples_loaded = 0
        max_samples = getattr(self.config, 'max_samples_per_task', None)
        
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if max_samples and samples_loaded >= max_samples:
                        break
                    
                    try:
                        data = json.loads(line.strip())
                        
                        # 构建图像路径
                        image_path = self.image_base_path / data["image"]
                        
                        metadata = {
                            "source_file": data_path,
                            "line_number": line_num
                        }
                        metadata.update({
                            k: v for k, v in data.items()
                            if k not in {"image", "prefix", "suffix"}
                        })

                        # 创建样本
                        sample = TaskSample(
                            task_type=task_type,
                            image_path=str(image_path),
                            prefix=data["prefix"],
                            suffix=data["suffix"],
                            weight=weight,
                            metadata=metadata
                        )
                        
                        self.samples.append(sample)
                        samples_loaded += 1
                        
                    except json.JSONDecodeError as e:
                        logger.warning(f"解析JSON失败 {data_path}:{line_num}: {e}")
                        continue
                    except KeyError as e:
                        logger.warning(f"缺少必要字段 {data_path}:{line_num}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"加载任务数据失败 {task_type}: {e}")
            raise

        return samples_loaded

    def _scan_all_tasks(self) -> None:
        """扫描所有任务数据文件，建立索引并预构建 byte offset 缓存

        用二进制模式打开文件，精确记录每行的 byte offset，
        使 _load_sample_by_index 可以通过 seek 实现 O(1) 随机访问。
        """
        task_counts = defaultdict(int)

        for config in self.data_configs:
            task_type = config["task_type"]
            data_path = config["data_path"]
            weight = config.get("weight", 1.0)
            max_samples = getattr(self.config, 'max_samples_per_task', None)

            logger.info(f"正在扫描任务索引: {task_type}, 路径: {data_path}")

            try:
                # 用二进制模式打开，精确记录每行起始 byte offset
                with open(data_path, 'rb') as f_bin:
                    line_num = 0
                    offset = 0  # 当前行起始 offset
                    for line_bytes in f_bin:
                        line_num += 1
                        if max_samples and task_counts[task_type] >= max_samples:
                            break

                        line = line_bytes.strip()
                        if not line:
                            offset += len(line_bytes)
                            continue

                        # 只验证 JSON 格式，不解析具体内容
                        try:
                            json.loads(line.decode('utf-8'))
                            self._sample_index.append((data_path, line_num, task_type, weight))
                            # 记录 idx -> byte_offset 的映射
                            idx = len(self._sample_index) - 1
                            self._sample_offset_cache[idx] = offset
                            task_counts[task_type] += 1
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            logger.warning(f"解析JSON失败 {data_path}:{line_num}")
                        offset += len(line_bytes)
            except Exception as e:
                logger.error(f"扫描任务数据失败 {task_type}: {e}")
                raise

        logger.info(f"索引扫描完成，各任务样本数: {dict(task_counts)}，已构建 offset 缓存")




    def _load_sample_by_index(self, idx: int) -> TaskSample:
        """根据索引按需加载单个样本（延迟加载模式，O(1) 访问）

        使用预构建的 byte offset 缓存，通过 f.seek() 直接跳转到目标行，
        避免每次重新扫描文件（原实现 O(n) 导致训练 O(n²)）。

        Args:
            idx: 样本索引

        Returns:
            TaskSample 对象
        """
        data_path, line_number, task_type, weight = self._sample_index[idx]

        # 使用预缓存的 byte offset 直接跳转
        offset = self._sample_offset_cache.get(idx)
        if offset is not None:
            with open(data_path, 'rb') as f_bin:
                f_bin.seek(offset)
                line_bytes = f_bin.readline()
                data = json.loads(line_bytes.strip().decode('utf-8'))
        else:
            # 降级：逐行扫描（理论上不会走到这里）
            logger.warning(f"_sample_offset_cache 未命中 idx={idx}，降级为线性扫描")
            with open(data_path, 'r', encoding='utf-8') as f:
                for current_line_num, line in enumerate(f, 1):
                    if current_line_num == line_number:
                        data = json.loads(line.strip())
                        break

        image_path = self.image_base_path / data["image"]

        metadata = {
            "source_file": data_path,
            "line_number": line_number
        }
        metadata.update({
            k: v for k, v in data.items()
            if k not in {"image", "prefix", "suffix"}
        })

        return TaskSample(
            task_type=task_type,
            image_path=str(image_path),
            prefix=data["prefix"],
            suffix=data["suffix"],
            weight=weight,
            metadata=metadata
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
        # 1. 如果内存缓存命中，直接返回
        if self.use_cache:
            with self._cache_lock:
                if idx in self._cache_index:
                    self._cache_index.move_to_end(idx)
                    return self._cache_index[idx]

        # 2. 如果 processor 不可用（如在 DataLoader 子进程中），尝试从磁盘缓存加载
        if self.processor is None and self.cache_dir is not None:
            cache_path = self._get_cache_path(idx)
            if cache_path.exists():
                try:
                    cached = self._load_cached_sample(idx, cache_path)
                    # 存入内存缓存以便下次快速访问（使用 LRU 淘汰）
                    self._cache_put(idx, cached)
                    return cached
                except Exception as e:
                    logger.warning(f"子进程加载磁盘缓存失败 {cache_path}: {e}")
            # 无法加载缓存，返回原始格式（需要上层处理）
            sample = self._get_sample(idx)
            return {
                "image_path": sample.image_path,
                "prompt": sample.prefix,
                "answer": sample.suffix,
                "task_type": sample.task_type,
                "weight": sample.weight,
                "metadata": sample.metadata,
                "_needs_encoding": True,  # 标记需要编码
            }

        sample = self._get_sample(idx)

        # 加载图像（使用 LRU 缓存避免重复 IO）
        image = _load_image_cached(sample.image_path)

        # 构建提示和答案
        prompt, answer = self._build_prompt_and_answer(sample)

        # 如果有处理器，进行预处理
        if self.processor is not None:
            # 获取任务 token（优先使用 backend，回退到 FLORENCE2_TASKS）
            prompt_text = prompt or sample.prefix or self._get_task_prompt(sample.task_type)

            # 优先走 backend.encode_with_task —— 让各后端自己决定如何拼接 prompt/answer
            # 这是关键修复：Florence-2 的 processor 严格要求 task token 独占 text，
            # 不能直接 prompt+answer 拼接。各后端在 encode_with_task 里自行处理拼接细节。
            backend_encode_succeeded = False
            if self.backend is not None and hasattr(self.backend, "encode_with_task"):
                try:
                    backend_encoded = self.backend.encode_with_task(
                        images=[image],
                        task_name=sample.task_type,
                        text_input=answer,
                        return_tensors="pt",
                    )

                    full_processed = {
                        k: (v.squeeze(0) if isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[0] == 1 else v)
                        for k, v in backend_encoded.items()
                    }

                    # 使用 backend.prepare_labels（如果可用）构建监督信号
                    if hasattr(self.backend, "prepare_labels"):
                        try:
                            labels = self.backend.prepare_labels({}, backend_encoded)
                            if isinstance(labels, torch.Tensor) and labels.dim() > 0 and labels.shape[0] == 1:
                                labels = labels.squeeze(0)
                        except Exception as exc:
                            logger.debug(f"backend.prepare_labels 失败，回退到默认: {exc}")
                            labels = full_processed["input_ids"].clone()
                    else:
                        labels = full_processed["input_ids"].clone()

                    attention_mask = full_processed.get("attention_mask")
                    result = {
                        "input_ids": full_processed["input_ids"],
                        "pixel_values": full_processed["pixel_values"],
                        "labels": labels,
                        "prompt": prompt,
                        "answer": answer,
                        "task_type": sample.task_type,
                        "weight": sample.weight,
                        "metadata": sample.metadata,
                    }
                    if attention_mask is not None:
                        result["attention_mask"] = attention_mask
                    for extra_tensor_key in ("token_type_ids", "position_ids", "mm_token_type_ids"):
                        if extra_tensor_key in full_processed:
                            result[extra_tensor_key] = full_processed[extra_tensor_key]
                    backend_encode_succeeded = True
                except AssertionError:
                    # 后端不能处理（例如 base 实现遇到 Florence-2 断言），回退到旧路径
                    backend_encode_succeeded = False
                except Exception as exc:
                    logger.debug(f"backend.encode_with_task 失败，回退到 processor 拼接: {exc}")
                    backend_encode_succeeded = False

            if backend_encode_succeeded:
                if self.use_cache:
                    self._cache_put(idx, result)
                if self.cache_dir is not None:
                    try:
                        cache_path = self._get_cache_path(idx)
                        self._save_cached_sample(result, cache_path)
                    except Exception as e:
                        logger.warning(f"保存缓存失败 {idx}: {e}")
                return result

            # 回退路径：processor 拼接（图像仅编码一次）
            # 1. 完整编码：prompt + answer + image（图像只经过 processor 一次）
            full_text = prompt_text + answer
            full_processed = self.processor(
                text=full_text,
                images=image,
                return_tensors="pt"
            )
            # 移除批次维度
            full_processed = {
                k: v.squeeze(0) if hasattr(v, 'squeeze') else v
                for k, v in full_processed.items()
            }

            # 2. 用纯文本 tokenizer 对 prompt_text 和 full_text 分别 tokenize，
            #    通过 answer token 数量倒推 prompt 在 full_ids 中的边界。
            #    这样完全不需要第二次传入 images，彻底消除图像双重编码。
            tokenizer = (
                getattr(self.processor, 'tokenizer', None)
                or getattr(self.processor, 'text_processor', None)
            )

            prompt_id_len = None
            if tokenizer is not None:
                try:
                    answer_token_len = len(tokenizer(
                        answer,
                        return_tensors="pt",
                        add_special_tokens=False,
                    )["input_ids"][0])
                    full_ids_1d = full_processed["input_ids"]
                    if full_ids_1d.dim() == 2:
                        full_ids_1d = full_ids_1d.squeeze(0)
                    # answer 占据 full_ids 末尾 answer_token_len 个 token
                    prompt_id_len = max(0, full_ids_1d.shape[0] - answer_token_len)
                except Exception:
                    prompt_id_len = None

            # 构建 prompt_processed（仅保留 input_ids 供 prepare_labels 使用）
            full_ids_1d = full_processed["input_ids"]
            if full_ids_1d.dim() == 2:
                full_ids_1d = full_ids_1d.squeeze(0)

            if prompt_id_len is not None:
                prompt_input_ids = full_ids_1d[:prompt_id_len]
            else:
                # 最终降级：无法分离 prompt，全序列监督
                prompt_input_ids = full_ids_1d.clone()

            prompt_processed = {
                "input_ids": prompt_input_ids,
                "pixel_values": full_processed.get("pixel_values"),  # 引用已有张量，不重新编码
            }

            # 构建 labels：优先使用 backend 的 prepare_labels，回退到默认逻辑
            if self.backend is not None and hasattr(self.backend, 'prepare_labels'):
                try:
                    labels = self.backend.prepare_labels(prompt_processed, full_processed)
                except Exception:
                    labels = self._default_prepare_labels(prompt_processed, full_processed)
            else:
                labels = self._default_prepare_labels(prompt_processed, full_processed)

            result = {
                "input_ids": full_processed['input_ids'],
                "attention_mask": full_processed['attention_mask'],
                "pixel_values": full_processed['pixel_values'],
                "labels": labels,
                "prompt": prompt,
                "answer": answer,
                "task_type": sample.task_type,
                "weight": sample.weight,
                "metadata": sample.metadata
            }
            for extra_tensor_key in ("token_type_ids", "position_ids", "mm_token_type_ids"):
                if extra_tensor_key in full_processed:
                    result[extra_tensor_key] = full_processed[extra_tensor_key]
        else:
            # 如果没有处理器，保留原始格式
            result = {
                "image": image,
                "prompt": prompt,
                "answer": answer,
                "task_type": sample.task_type,
                "weight": sample.weight,
                "metadata": sample.metadata
            }

        # 编码后保存到缓存（支持渐进式加载，带 LRU 淘汰）
        if self.use_cache:
            self._cache_put(idx, result)
        if self.cache_dir is not None:
            try:
                cache_path = self._get_cache_path(idx)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                # 排除 pixel_values（大张量，可从图像重新计算）以节省磁盘空间
                cache_data = {k: v for k, v in result.items() if k != 'pixel_values'}
                torch.save(cache_data, cache_path)
            except Exception as e:
                logger.warning(f"保存缓存失败 {idx}: {e}")

        return result
            
    def _get_task_prompt(self, task_type: str) -> str:
        """获取任务 prompt（优先使用 backend，回退到 FLORENCE2_TASKS）

        Args:
            task_type: 任务类型名称

        Returns:
            任务 prompt 字符串
        """
        if self.backend is not None:
            try:
                return self.backend.get_task_prompt(task_type)
            except Exception:
                pass
        # 向后兼容
        task_config = FLORENCE2_TASKS.get(task_type, {})
        return task_config.get("prompt", "")

    def _build_prompt_and_answer(self, sample: TaskSample) -> Tuple[str, str]:
        """构建提示和答案

        Args:
            sample: 任务样本

        Returns:
            (提示, 答案) 元组
        """
        task_prompt = self._get_task_prompt(sample.task_type)

        prefix = sample.prefix or ""
        if task_prompt and prefix:
            prompt = prefix if prefix.startswith(task_prompt) else f"{task_prompt}{prefix}"
        else:
            prompt = prefix or task_prompt

        for extra_key in ("text_input", "region"):
            extra_value = sample.metadata.get(extra_key)
            if extra_value is None:
                continue
            if not isinstance(extra_value, str):
                extra_value = json.dumps(extra_value, ensure_ascii=False)
            if extra_value and extra_value not in prompt:
                prompt = f"{prompt}{extra_value}"

        # 答案就是suffix
        answer = sample.suffix

        return prompt, answer

    def _default_prepare_labels(
        self,
        encoded_prompt: Dict[str, torch.Tensor],
        encoded_full: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """默认 labels 构建（prompt 部分忽略，answer 部分监督）

        Args:
            encoded_prompt: 仅编码 prompt 的结果
            encoded_full: 编码 prompt + answer 的结果

        Returns:
            labels 张量
        """
        prompt_ids = encoded_prompt.get("input_ids")
        full_ids = encoded_full["input_ids"]

        if prompt_ids is None:
            logger.warning("prompt 编码未返回 input_ids，回退为仅监督完整序列")
            return full_ids.clone()

        # 处理可能的 batch 维度
        if prompt_ids.dim() == 2:
            prompt_ids = prompt_ids.squeeze(0)
        if full_ids.dim() == 2:
            full_ids = full_ids.squeeze(0)

        prompt_length = len(prompt_ids)
        full_length = len(full_ids)

        labels = torch.full_like(full_ids, -100)
        if full_length > prompt_length:
            labels[prompt_length:] = full_ids[prompt_length:]
        return labels

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
            # 检查磁盘缓存
            if self.cache_dir is not None:
                cache_path = self._get_cache_path(idx)
                if cache_path.exists():
                    try:
                        cached = self._load_cached_sample(idx, cache_path)
                        self._cache_put(idx, cached)
                        cache_hits += 1
                        continue
                    except Exception as e:
                        logger.warning(f"加载缓存失败 {cache_path}: {e}")

            # 执行编码
            try:
                encoded = self.__getitem__(idx)
                if idx in self._cache_index:
                    cache_hits += 1
                    continue

                self._cache_put(idx, encoded)
                cache_misses += 1

                # 保存到磁盘
                if self.cache_dir is not None:
                    cache_path = self._get_cache_path(idx)
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    self._save_cached_sample(encoded, cache_path)

            except Exception as e:
                logger.warning(f"预编码样本 {idx} 失败: {e}")
                continue

        logger.info(
            f"预编码完成: 内存缓存 {len(self._cache_index)} 条, "
            f"磁盘命中 {cache_hits} 条, 新编码 {cache_misses} 条"
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
            if self.cache_dir is not None:
                cache_path = self._get_cache_path(idx)
                if cache_path.exists():
                    return idx, self._load_cached_sample(idx, cache_path), True
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

                        self._cache_put(idx, encoded)
                        if from_disk:
                            cache_hits += 1
                        else:
                            cache_misses += 1

                        if self.cache_dir is not None:
                            cache_path = self._get_cache_path(idx)
                            if not cache_path.exists():
                                self._save_cached_sample(encoded, cache_path)
                        progress.update(1)

                    submit_until_full()
        
        logger.info(
            f"预编码完成（并行）: 内存缓存 {len(self._cache_index)} 条, "
            f"磁盘命中 {cache_hits} 条, 新编码 {cache_misses} 条"
        )

    def _get_cache_path(self, idx: int) -> Path:
        """获取样本的磁盘缓存路径（使用 idx 直接计算，避免触发 lazy I/O）"""
        if self.cache_dir is None:
            raise ValueError("cache_dir 未设置")
        cache_root = Path(self.cache_dir)
        # 按数据文件分目录，避免单个目录文件过多
        # 使用 _sample_index 中的 data_path 哈希，避免 _get_sample() 触发 I/O
        if self.lazy_load and idx < len(self._sample_index):
            source_file = self._sample_index[idx][0]
        elif idx < len(self.samples):
            sample = self.samples[idx]
            source_file = sample.metadata.get("source_file") or sample.image_path
        else:
            source_file = str(idx)
        source_hash = hashlib.sha256(str(source_file).encode("utf-8")).hexdigest()[:16]
        return cache_root / source_hash / f"sample_{idx}.pt"

    def _load_cached_sample(self, idx: int, cache_path: Path) -> Dict[str, Any]:
        """加载磁盘缓存，并在需要时补回未持久化的图像张量。"""
        cached = safe_torch_load_cpu(cache_path, context="Dataset cache")

        if 'pixel_values' not in cached and self.processor is None:
            raise RuntimeError(
                "磁盘缓存不包含 pixel_values，且当前进程没有 processor，无法恢复图像张量。"
                "请使用 num_workers=0 在线编码，或在主进程中预热缓存后再读取。"
            )

        if 'pixel_values' not in cached and self.processor is not None:
            sample = self._get_sample(idx)
            image = _load_image_cached(sample.image_path)
            image_inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = image_inputs.get("pixel_values")
            if pixel_values is not None:
                cached["pixel_values"] = pixel_values.squeeze(0)
            else:
                raise RuntimeError(f"processor 未为缓存样本 {idx} 返回 pixel_values，无法恢复磁盘缓存")

        return cached

    def _save_cached_sample(self, data: Dict[str, Any], cache_path: Path) -> None:
        """保存磁盘缓存，避免持久化大体积 pixel_values。"""
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_data = {k: v for k, v in data.items() if k != 'pixel_values'}
        torch.save(cache_data, cache_path)

    def _cache_put(self, idx: int, data: Dict[str, Any]) -> None:
        """将数据放入缓存，支持 O(1) LRU 淘汰（使用 OrderedDict）"""
        with self._cache_lock:
            if idx in self._cache_index:
                # 更新访问顺序：移到末尾（最新访问）
                self._cache_index.move_to_end(idx)
                self._cache_index[idx] = data
                return

            # 检查是否需要淘汰
            if len(self._cache_index) >= self._cache_max_size:
                # LRU 淘汰：移除最久未访问的条目（OrderedDict 首个元素）
                self._cache_index.popitem(last=False)

            # 插入新数据
            self._cache_index[idx] = data

    def clear_cache(self) -> None:
        """清除内存与磁盘缓存"""
        with self._cache_lock:
            self._cache_index.clear()
        if self.cache_dir is not None:
            import shutil
            cache_root = Path(self.cache_dir)
            if cache_root.exists():
                shutil.rmtree(cache_root)
                logger.info(f"已清除磁盘缓存: {cache_root}")

    # ------------------------------------------------------------------
    # 多进程序列化支持（用于 DataLoader num_workers > 0）
    # ------------------------------------------------------------------

    def __getstate__(self) -> Dict[str, Any]:
        """序列化状态（排除不可序列化的内存缓存、processor 和 backend）"""
        state = self.__dict__.copy()
        # 内存缓存包含大量张量，不序列化到子进程
        state['_cache_index'] = OrderedDict()
        state.pop('_cache_lock', None)
        # processor 通常包含不可序列化的对象（如 tokenizer 的 C++ 扩展）
        state['processor'] = None
        # backend 包含 torch.nn.Module 和 tokenizer，不可序列化
        state['backend'] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """反序列化状态"""
        self.__dict__.update(state)
        # 重新初始化空的内存缓存
        self._cache_index = OrderedDict()
        self._cache_lock = threading.RLock()
        # 注意：processor 和 backend 需要在主进程中重新设置
        # 如果启用了磁盘缓存，子进程会在 __getitem__ 时按需加载

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
            subset_samples = []
        else:
            subset_samples = [self.samples[i] for i in indices]
            subset_index = []
            subset_offset_cache = {}

        # 创建新的数据集实例
        subset = MultiTaskDataset.__new__(MultiTaskDataset)
        subset.data_configs = self.data_configs
        subset.image_base_path = self.image_base_path
        subset.config = self.config
        subset.processor = self.processor
        subset.backend = self.backend
        subset.lazy_load = self.lazy_load
        subset.samples = subset_samples
        subset._sample_index = subset_index
        subset._sample_offset_cache = subset_offset_cache
        subset.task_weights = self.task_weights.copy()
        subset.collate_fn = self.collate_fn
        subset._cache_lock = threading.RLock()

        # 重新构建任务索引
        subset.task_indices = defaultdict(list)
        subset._build_task_indices()

        # 缓存配置
        subset.use_cache = self.use_cache
        subset.cache_dir = self.cache_dir
        subset._cache_index = OrderedDict()
        subset._cache_max_size = self._cache_max_size

        return subset

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
        dataset.use_cache = getattr(config, "use_cache", False)
        dataset.cache_dir = getattr(config, "cache_dir", None)
        dataset._cache_index = OrderedDict()
        dataset._cache_max_size = getattr(config, "cache_max_size", 10000)
        dataset._cache_lock = threading.RLock()
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
            image_path = cls._materialize_hf_image(
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
                key: cls._metadata_safe_value(value)
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

        if dataset.use_cache and dataset.processor is not None:
            dataset.preprocess_and_cache()

        logger.info(f"HF dataset 已加载为 MultiTaskDataset，样本数: {len(dataset.samples)}")
        return dataset

    @staticmethod
    def _materialize_hf_image(
        image_value: Any,
        idx: int,
        config: DataConfig,
        image_base_path: Path,
    ) -> Path:
        """将 HF image 字段规范化为本地图片路径。"""
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
                return MultiTaskDataset._save_hf_image(image, idx, config, image_base_path)

        if isinstance(image_value, Image.Image):
            return MultiTaskDataset._save_hf_image(image_value.convert("RGB"), idx, config, image_base_path)

        raise TypeError(
            "HF image column must contain a path, PIL Image, or dict with 'path'/'bytes'"
        )

    @staticmethod
    def _save_hf_image(
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

    @staticmethod
    def _metadata_safe_value(value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, (list, tuple)):
            return [MultiTaskDataset._metadata_safe_value(item) for item in value]
        if isinstance(value, dict):
            return {
                str(key): MultiTaskDataset._metadata_safe_value(item)
                for key, item in value.items()
            }
        return str(value)
    
    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """保存数据集到文件
        
        Args:
            file_path: 文件路径
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.lazy_load:
            # 延迟加载模式下需要按需加载样本后序列化
            samples_data = []
            for idx in range(len(self)):
                sample = self._get_sample(idx)
                samples_data.append(sample.to_dict())
        else:
            samples_data = [sample.to_dict() for sample in self.samples]

        data = {
            "data_configs": self.data_configs,
            "image_base_path": str(self.image_base_path),
            "config": getattr(self.config, '__dict__', {}),
            "samples": samples_data,
            "task_weights": self.task_weights
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        logger.info(f"数据集已保存到: {file_path}")
    
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
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 创建新实例
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
        dataset.use_cache = getattr(dataset.config, 'use_cache', False)
        dataset.cache_dir = getattr(dataset.config, 'cache_dir', None)
        dataset._cache_index = OrderedDict()
        dataset._cache_max_size = getattr(dataset.config, 'cache_max_size', 10000)
        dataset._cache_lock = threading.RLock()
        dataset.collate_fn = Florence2Collator(pad_token_id=dataset._get_pad_token_id())

        # 重新构建任务索引
        dataset.task_indices = defaultdict(list)
        dataset._build_task_indices()
        
        logger.info(f"数据集已从文件加载: {file_path}")
        return dataset
