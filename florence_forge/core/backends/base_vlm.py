"""
BaseVLMBackend — 所有 VLM 后端的统一基类

提供通用的加载、编码、生成、保存、信息查询等实现。
子类只需要补充模型/处理器加载和任务 prompt 映射。
"""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Protocol, Type, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BackendConfig(Protocol):
    """结构化描述 VLM 后端实际消费的配置字段。

    后端仍然通过 ``getattr`` 保持向后兼容；该 Protocol 主要服务于静态分析、
    IDE 补全和第三方后端实现者的类型契约。
    """

    model_name: str
    revision: Optional[str]
    trust_remote_code: bool
    torch_dtype: str
    device: str
    device_map: str
    attn_implementation: Optional[str]
    use_fp16: bool
    use_bf16: bool


def _check_flash_attn_availability() -> bool:
    """检查 flash-attn 是否可用。"""
    try:
        import flash_attn  # noqa: F401
        return True
    except Exception:
        return False


def _patch_transformers_import_check() -> None:
    """兼容部分 transformers 远程代码对 flash-attn 的严格导入检查。"""
    try:
        from transformers.utils import import_utils
    except Exception:
        return

    original = getattr(import_utils, "is_flash_attn_2_available", None)
    if callable(original):
        import_utils.is_flash_attn_2_available = lambda: False


def _patch_transformers_config_defaults() -> None:
    """为部分远程配置代码补齐旧版 transformers 缺失的默认字段。"""
    try:
        from transformers import PretrainedConfig
    except Exception:
        return

    if not hasattr(PretrainedConfig, "forced_bos_token_id"):
        PretrainedConfig.forced_bos_token_id = None


def _is_cpu_fallback_candidate(exc: Exception) -> bool:
    """仅对设备/精度相关错误启用 CPU 回退，避免掩盖配置或网络异常。"""
    message = str(exc).lower()
    fallback_signals = (
        "cuda",
        "mps",
        "out of memory",
        "oom",
        "device-side",
        "device type",
        "not implemented for",
        "not supported on this device",
        "bfloat16",
        "float16",
        "half",
        "attention",
    )
    return any(signal in message for signal in fallback_signals)


class BaseVLMBackend(ABC, nn.Module):
    """VLM 后端抽象基类。"""

    ARCHITECTURE_TYPE: str = "encoder_decoder"
    BACKEND_NAME: str = "base"

    _backends: ClassVar[Dict[str, Type["BaseVLMBackend"]]] = {}

    def __init__(self, config: BackendConfig):
        super().__init__()
        self.config = config
        self._model: Optional[nn.Module] = None
        self._processor: Any = None
        self._device = "cpu"
        self._dtype = torch.float32
        self._task_prompts: Dict[str, str] = {}
        self._supports_tasks: List[str] = []
        self.is_peft_model = False
        self._cached_param_count: Optional[int] = None

    @property
    def model(self) -> nn.Module:
        if self._model is None:
            raise RuntimeError("模型未加载，请先调用 backend.load()")
        return self._model

    @property
    def processor(self) -> Any:
        if self._processor is None:
            raise RuntimeError("Processor 未加载，请先调用 backend.load()")
        return self._processor

    @property
    def backend_name(self) -> str:
        return self.BACKEND_NAME

    @property
    def device(self) -> torch.device:
        return torch.device(self._device)

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def task_prompt(self) -> Dict[str, str]:
        return self._task_prompts

    @abstractmethod
    def load_model(self) -> None:
        pass

    @abstractmethod
    def load_processor(self) -> None:
        pass

    @abstractmethod
    def get_task_prompt(self, task_name: str) -> str:
        pass

    @abstractmethod
    def supports_task(self, task_name: str) -> bool:
        pass

    def load(self) -> "BaseVLMBackend":
        """显式加载模型和处理器。"""
        if self._model is None:
            self.load_model()
        if self._processor is None:
            self.load_processor()
        return self

    def _get_optimal_device(self) -> str:
        configured = getattr(self.config, "device", "auto")
        if configured and configured != "auto":
            return configured
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _get_optimal_dtype(self, device: str) -> torch.dtype:
        configured = getattr(self.config, "torch_dtype", "auto")
        if configured == "float16":
            return torch.float16
        if configured == "bfloat16":
            return torch.bfloat16
        if configured == "float32":
            return torch.float32

        if device.startswith("cuda"):
            if getattr(self.config, "use_bf16", False) and getattr(torch.cuda, "is_bf16_supported", lambda: False)():
                return torch.bfloat16
            if getattr(self.config, "use_fp16", False):
                return torch.float16
        return torch.float32

    def _build_model_kwargs(
        self,
        device: str,
        torch_dtype: torch.dtype,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "trust_remote_code": getattr(self.config, "trust_remote_code", True),
            "torch_dtype": torch_dtype,
        }

        # 可选地 pin HuggingFace revision，降低供应链风险
        revision = getattr(self.config, "revision", None)
        if revision:
            kwargs["revision"] = revision

        device_map = getattr(self.config, "device_map", "auto")
        if device in ("cpu", "mps") or not device_map or str(device_map).lower() in (
            "none",
            "null",
        ):
            kwargs["device_map"] = None
            kwargs["low_cpu_mem_usage"] = False
        else:
            kwargs["device_map"] = device_map

        attn_impl = getattr(self.config, "attn_implementation", None)
        if attn_impl:
            kwargs["attn_implementation"] = attn_impl

        if extra_kwargs:
            kwargs.update(extra_kwargs)
        return kwargs

    def _load_with_cpu_fallback(self, load_fn, model_name: str, model_kwargs: Dict[str, Any]) -> None:
        """尝试按目标设备加载，失败时回退到 CPU。"""
        try:
            model = load_fn(model_name, **model_kwargs)
            self._model = model
            target_device = self._get_optimal_device()
            if model_kwargs.get("device_map") is None and hasattr(model, "to"):
                model = model.to(target_device)
                self._model = model
            self._device = target_device
            self._dtype = model_kwargs.get("torch_dtype", torch.float32)
        except Exception as exc:
            if not _is_cpu_fallback_candidate(exc):
                raise
            logger.warning(f"按目标设备加载失败，回退到 CPU: {exc}")
            fallback_kwargs = dict(model_kwargs)
            fallback_kwargs["device_map"] = None
            fallback_kwargs["torch_dtype"] = torch.float32
            model = load_fn(model_name, **fallback_kwargs)
            if hasattr(model, "to"):
                model = model.to("cpu")
            self._model = model
            self._device = "cpu"
            self._dtype = torch.float32

    def _load_processor_base(self, processor_cls: Any, model_name: str) -> None:
        if processor_cls is None:
            raise RuntimeError("处理器类不可用，请检查 transformers 依赖")
        processor_kwargs: Dict[str, Any] = {
            "trust_remote_code": getattr(self.config, "trust_remote_code", True),
        }
        revision = getattr(self.config, "revision", None)
        if revision:
            processor_kwargs["revision"] = revision
        self._processor = processor_cls.from_pretrained(model_name, **processor_kwargs)

    def encode(
        self,
        images: List[Any],
        text: Union[str, List[str]],
        return_tensors: str = "pt",
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        self.load()
        if isinstance(text, str):
            text = [text]
        inputs = self.processor(text=text, images=images, return_tensors=return_tensors, **kwargs)
        return {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }

    def encode_with_task(
        self,
        images: List[Any],
        task_name: str,
        text_input: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        prompt = self.get_task_prompt(task_name)
        text = f"{prompt} {text_input}".strip() if text_input else prompt
        return self.encode(images=images, text=text, **kwargs)

    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        self.load()
        input_ids = input_ids.to(self.device)
        pixel_values = pixel_values.to(self.device)
        generate_kwargs = dict(kwargs)
        if self.BACKEND_NAME == "florence-2":
            generate_kwargs.setdefault("use_cache", False)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
            generate_kwargs["attention_mask"] = attention_mask
        # 设备一致性兜底：将 kwargs 中任意张量（如 decoder_input_ids、
        # decoder_attention_mask 等）同步到模型设备，避免 CPU tensor + CUDA
        # model 组合在 generate 内部触发崩溃。
        generate_kwargs = self._move_tensors_to_device(generate_kwargs)
        return self.model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            **generate_kwargs,
        )

    def _move_tensors_to_device(self, values: Dict[str, Any]) -> Dict[str, Any]:
        """递归地将字典中的张量移动到当前模型设备。

        仅处理 ``torch.Tensor`` 以及由张量组成的 list/tuple，其他类型原样保留。
        用于保证传入 ``generate`` / ``forward`` 的所有张量与模型设备一致。
        """
        moved: Dict[str, Any] = {}
        for key, value in values.items():
            moved[key] = self._move_value_to_device(value)
        return moved

    def _move_value_to_device(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(self.device)
        if isinstance(value, (list, tuple)):
            converted = [self._move_value_to_device(item) for item in value]
            return type(value)(converted)
        return value

    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> List[str]:
        self.load()
        if not hasattr(self.processor, "batch_decode"):
            raise RuntimeError("当前 processor 不支持 batch_decode")
        return self.processor.batch_decode(token_ids, skip_special_tokens=skip_special_tokens)

    def prepare_labels(
        self,
        encoded_prompt: Dict[str, torch.Tensor],
        encoded_full: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """构建训练 labels，prompt 部分置为 -100，仅监督 answer token。"""
        prompt_ids = encoded_prompt.get("input_ids")
        full_ids = encoded_full["input_ids"]

        if prompt_ids is None:
            logger.warning("prompt 编码未返回 input_ids，回退为监督完整序列")
            return full_ids.clone()

        if prompt_ids.dim() == 2:
            prompt_ids = prompt_ids.squeeze(0)
        if full_ids.dim() == 2:
            full_ids = full_ids.squeeze(0)

        labels = torch.full_like(full_ids, -100)
        prompt_length = min(prompt_ids.shape[-1], full_ids.shape[-1])
        if full_ids.shape[-1] > prompt_length:
            labels[prompt_length:] = full_ids[prompt_length:]
        return labels

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        self.load()
        input_ids = input_ids.to(self.device)
        pixel_values = pixel_values.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
        # 同步 kwargs 中任意张量到模型设备，保持设备一致性。
        kwargs = self._move_tensors_to_device(kwargs)
        return self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    def save_pretrained(self, save_directory: Union[str, Path]) -> None:
        self.load()
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(save_directory)
        if hasattr(self.processor, "save_pretrained"):
            self.processor.save_pretrained(save_directory)

    def load_pretrained(self, model_path: Union[str, Path], **kwargs) -> None:
        self.config.model_name = str(model_path)
        self._model = None
        self._processor = None
        self.load()

    def _get_extra_model_info(self) -> Dict[str, Any]:
        return {}

    def get_supported_tasks(self) -> List[str]:
        return list(self._supports_tasks)

    def set_task_prompt(self, task_name: str, prompt: str) -> None:
        self._task_prompts[task_name] = prompt
        if task_name not in self._supports_tasks:
            self._supports_tasks.append(task_name)

    def get_model_info(self) -> Dict[str, Any]:
        self.load()
        total_parameters = sum(p.numel() for p in self.model.parameters())
        trainable_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        trainable_ratio = (trainable_parameters / total_parameters) if total_parameters else 0.0
        info = {
            "model_name": getattr(self.config, "model_name", "unknown"),
            "backend": self.backend_name,
            "architecture_type": self.ARCHITECTURE_TYPE,
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "trainable_ratio": trainable_ratio,
            "device": str(self.device),
            "dtype": str(self.dtype).replace("torch.", ""),
            "is_peft_model": self.is_peft_model,
        }
        info.update(self._get_extra_model_info())
        return info


class VLMBackendRegistry:
    """VLM 后端注册表。"""

    _backends: ClassVar[Dict[str, Type["BaseVLMBackend"]]] = {}

    @staticmethod
    def _normalize_name(name: str) -> str:
        key = name.strip().lower()
        if not key:
            raise ValueError("后端名称不能为空")
        return key

    @classmethod
    def register(cls, name: str, backend_class: Type["BaseVLMBackend"]) -> None:
        if not isinstance(backend_class, type) or not issubclass(backend_class, BaseVLMBackend):
            raise TypeError(f"后端类必须继承自 BaseVLMBackend，得到 {backend_class}")
        cls._backends[cls._normalize_name(name)] = backend_class

    @classmethod
    def get_backend_class(cls, name: str) -> Type["BaseVLMBackend"]:
        """返回已注册后端类，作为解析器等模块的公共查询 API。"""
        key = cls._normalize_name(name)
        if key not in cls._backends:
            available = ", ".join(cls.list_backends())
            raise ValueError(f"未知后端: {key}。可用后端: {available}")
        return cls._backends[key]

    @classmethod
    def create(cls, name: str, config: BackendConfig) -> BaseVLMBackend:
        return cls.get_backend_class(name)(config)

    @classmethod
    def list_backends(cls) -> List[str]:
        return sorted(cls._backends.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        return cls._normalize_name(name) in cls._backends


_registry = VLMBackendRegistry()


def create_backend(name: str, config: BackendConfig) -> BaseVLMBackend:
    return _registry.create(name, config)
