"""通用 HuggingFace VLM 后端

支持任何 transformers 库中的标准视觉语言模型。
通过 transformers 的 Auto API 自动推断模型类型，无需为每个新模型编写专门的后端。

公共逻辑（generate, decode, forward, save, load, get_model_info, _compile_model）
已在 BaseVLMBackend 中实现，此文件仅保留 GenericHF 特有的自动检测和加载逻辑。
"""

import os
import torch
import logging
from typing import Optional, Dict, Any, List, Union
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    from transformers import (
        AutoProcessor,
        AutoTokenizer,
        AutoImageProcessor,
        AutoModelForImageTextToText,
        AutoModelForCausalLM,
        AutoModel,
    )
    AutoModelForVision2Seq = AutoModelForImageTextToText  # 向后兼容
except ImportError:
    try:
        from transformers import AutoModelForVision2Seq
    except ImportError:
        AutoModelForVision2Seq = None
    AutoModelForImageTextToText = None
    AutoProcessor = None
    AutoTokenizer = None
    AutoImageProcessor = None
    AutoModelForCausalLM = None
    AutoModel = None

from .base_vlm import BaseVLMBackend, VLMBackendRegistry

logger = logging.getLogger(__name__)


# 通用任务 prompt 映射（自然语言风格，适用于大多数 VLM）
GENERIC_TASK_PROMPTS = {
    "CAPTION": "Describe this image.",
    "DETAILED_CAPTION": "Describe this image in detail.",
    "MORE_DETAILED_CAPTION": "Provide a comprehensive description of this image.",
    "OD": "Detect all objects in this image.",
    "DENSE_REGION_CAPTION": "Detect all objects and describe each region.",
    "OPEN_VOCABULARY_DETECTION": "Detect",
    "REGION_PROPOSAL": "Find all interesting regions in this image.",
    "REGION_TO_CATEGORY": "What category is this region?",
    "REGION_TO_DESCRIPTION": "Describe this region.",
    "OCR": "Read all text in this image.",
    "OCR_WITH_REGION": "Read all text in this image with locations.",
    "VQA": "Answer:",
    "REGION_TO_SEGMENTATION": "Segment this region.",
    "REFERRING_EXPRESSION_SEGMENTATION": "Segment the referred object.",
}

# 已知模型到架构类型的映射
KNOWN_MODEL_ARCHITECTURES = {
    "microsoft/florence-2": "vision2seq",
    "microsoft/florence-2-base": "vision2seq",
    "microsoft/florence-2-large": "vision2seq",
    "google/paligemma": "causal_lm",
    "llava-hf/llava": "causal_lm",
    "Qwen/Qwen-VL": "causal_lm",
    "OpenGVLab/InternVL": "causal_lm",
    "Salesforce/instructblip": "vision2seq",
    "Salesforce/blip": "vision2seq",
}


def _guess_architecture_type(model_name: str) -> str:
    """根据模型名称猜测架构类型"""
    model_name_lower = model_name.lower()
    for prefix, arch in KNOWN_MODEL_ARCHITECTURES.items():
        if model_name_lower.startswith(prefix.lower()):
            return arch
    if "paligemma" in model_name_lower or "llava" in model_name_lower or "qwen-vl" in model_name_lower:
        return "causal_lm"
    if "florence" in model_name_lower or "blip" in model_name_lower:
        return "vision2seq"
    return "auto"


class GenericHFBackend(BaseVLMBackend):
    """通用 HuggingFace VLM 后端

    通过 transformers Auto API 自动加载任意视觉语言模型，
    无需为每个新模型编写专门的后端类。

    公共接口继承自 BaseVLMBackend。
    """

    ARCHITECTURE_TYPE = "auto"  # 在加载时自动确定
    BACKEND_NAME = "generic-hf"

    def __init__(self, config: Any):
        """初始化"""
        super().__init__(config)

        # 任务配置
        custom_prompts = getattr(config, "task_prompts", None)
        self._task_prompts = {**GENERIC_TASK_PROMPTS, **(custom_prompts or {})}
        custom_tasks = getattr(config, "supported_tasks", None)
        self._supports_tasks = list(custom_tasks or self._task_prompts.keys())

        # 架构类型（自动推断或显式指定）
        self._architecture_type = getattr(
            config, "architecture_type",
            _guess_architecture_type(config.model_name)
        )

        # 用于分拆加载场景的 tokenizer 和 image_processor
        self._tokenizer = None
        self._image_processor = None

        logger.info(f"GenericHFBackend 已初始化（模型未加载，请调用 backend.load()）")

    # ------------------------------------------------------------------
    # 1. 模型加载（GenericHF 特有的自动检测逻辑）
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """加载模型（自动检测类型）"""
        if AutoModel is None:
            raise RuntimeError("transformers 库未安装")

        device = self._get_optimal_device()
        torch_dtype = self._get_optimal_dtype(device)
        model_kwargs = self._build_model_kwargs(device, torch_dtype)

        self._load_with_cpu_fallback(
            load_fn=self._auto_load_model_fn,
            model_name=self.config.model_name,
            model_kwargs=model_kwargs
        )

    def _auto_load_model_fn(self, model_name: str, **kwargs):
        """作为 load_fn 传给 _load_with_cpu_fallback 的自动检测加载函数"""
        arch = self._architecture_type

        if arch == "vision2seq" and AutoModelForVision2Seq is not None:
            logger.info(f"使用 AutoModelForVision2Seq 加载 {model_name}")
            return AutoModelForVision2Seq.from_pretrained(model_name, **kwargs)

        if arch == "causal_lm" and AutoModelForCausalLM is not None:
            logger.info(f"使用 AutoModelForCausalLM 加载 {model_name}")
            return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

        # auto 模式：依次尝试
        errors = []
        if AutoModelForVision2Seq is not None:
            try:
                model = AutoModelForVision2Seq.from_pretrained(model_name, **kwargs)
                self._architecture_type = "vision2seq"
                logger.info(f"自动检测: 使用 AutoModelForVision2Seq")
                return model
            except Exception as e:
                errors.append(f"Vision2Seq: {e}")

        if AutoModelForCausalLM is not None:
            try:
                model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
                self._architecture_type = "causal_lm"
                logger.info(f"自动检测: 使用 AutoModelForCausalLM")
                return model
            except Exception as e:
                errors.append(f"CausalLM: {e}")

        raise RuntimeError(
            f"无法自动加载模型 {model_name}。尝试过的方法:\n" +
            "\n".join(errors)
        )

    def load_processor(self) -> None:
        """加载处理器（AutoProcessor 或 Tokenizer+ImageProcessor）"""
        trust_remote_code = getattr(self.config, "trust_remote_code", True)
        model_name = self.config.model_name

        # 尝试 AutoProcessor
        if AutoProcessor is not None:
            try:
                self._processor = AutoProcessor.from_pretrained(
                    model_name, trust_remote_code=trust_remote_code
                )
                logger.info("Processor 加载成功 (AutoProcessor)")
                return
            except Exception as e:
                logger.warning(f"AutoProcessor 加载失败: {e}")

        # 回退：分别加载
        if AutoTokenizer is not None:
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    model_name, trust_remote_code=trust_remote_code, use_fast=True
                )
                logger.info("Tokenizer 加载成功")
            except Exception as e:
                logger.warning(f"Tokenizer 加载失败: {e}")

        if AutoImageProcessor is not None:
            try:
                self._image_processor = AutoImageProcessor.from_pretrained(
                    model_name, trust_remote_code=trust_remote_code
                )
                logger.info("ImageProcessor 加载成功")
            except Exception as e:
                logger.warning(f"ImageProcessor 加载失败: {e}")

        if self._processor is None and self._tokenizer is None:
            logger.error("所有 processor 加载方式均失败")

    # ------------------------------------------------------------------
    # 2. 编码覆盖（支持分拆的 tokenizer + image_processor）
    # ------------------------------------------------------------------

    def encode(
        self,
        images: List[Any],
        text: Union[str, List[str]],
        return_tensors: str = "pt",
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """编码图像和文本（覆盖基类以支持分拆的 tokenizer + image_processor）"""
        if isinstance(text, str):
            text = [text]
        try:
            from PIL import Image as PILImage
            if isinstance(images, PILImage.Image):
                images = [images]
        except ImportError:
            pass

        if self._processor is not None:
            inputs = self._processor(
                text=text, images=images, return_tensors=return_tensors, **kwargs
            )
        elif self._tokenizer is not None and self._image_processor is not None:
            text_inputs = self._tokenizer(
                text, return_tensors=return_tensors, padding=True, truncation=True, **kwargs
            )
            image_inputs = self._image_processor(images, return_tensors=return_tensors, **kwargs)
            inputs = {**text_inputs, **image_inputs}
        else:
            raise RuntimeError("Processor 未加载，无法进行编码")

        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> List[str]:
        """解码 token ids（覆盖基类以支持 tokenizer 回退）"""
        if self._processor is not None and hasattr(self._processor, "batch_decode"):
            return self._processor.batch_decode(token_ids, skip_special_tokens=skip_special_tokens)
        elif self._tokenizer is not None:
            return self._tokenizer.batch_decode(token_ids, skip_special_tokens=skip_special_tokens)
        else:
            raise RuntimeError("Processor 未加载，无法解码")

    # ------------------------------------------------------------------
    # 3. 保存覆盖（支持 tokenizer 回退）
    # ------------------------------------------------------------------

    def save_pretrained(self, save_directory: Union[str, Path]) -> None:
        """保存模型和 processor（覆盖基类以支持 tokenizer 回退）"""
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(save_directory)
        if self._processor is not None:
            self._processor.save_pretrained(save_directory)
        elif self._tokenizer is not None:
            self._tokenizer.save_pretrained(save_directory)
        logger.info(f"模型已保存到: {save_directory}")

    # ------------------------------------------------------------------
    # 4. 任务相关
    # ------------------------------------------------------------------

    def get_task_prompt(self, task_name: str) -> str:
        """获取任务 prompt"""
        return self._task_prompts.get(task_name, task_name)

    def supports_task(self, task_name: str) -> bool:
        """检查是否支持任务"""
        return task_name in self._supports_tasks

    def _get_extra_model_info(self) -> Dict[str, Any]:
        return {
            "architecture_type": self._architecture_type,
            "is_peft_model": self.is_peft_model,
        }


# 注册后端
try:
    VLMBackendRegistry.register("generic-hf", GenericHFBackend)
    VLMBackendRegistry.register("auto", GenericHFBackend)
    VLMBackendRegistry.register("hf", GenericHFBackend)
    logger.info("GenericHFBackend 已注册到 VLMBackendRegistry")
except Exception as e:
    logger.warning(f"GenericHFBackend 注册失败: {e}")
