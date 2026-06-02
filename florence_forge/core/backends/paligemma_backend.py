"""PaliGemma VLM 后端实现

支持 google/paligemma-3b-pt-224 等 PaliGemma 系列模型，
实现框架与 PaliGemma 的解耦。

公共逻辑（encode, generate, decode, forward, save, load, get_model_info,
_compile_model）已在 BaseVLMBackend 中实现，此文件仅保留 PaliGemma 特有的
模型加载和任务映射逻辑。
"""

import os
import torch
import logging
import re
from typing import Optional, Dict, Any, List, Union
from pathlib import Path

try:
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
except ImportError:
    AutoProcessor = None
    PaliGemmaForConditionalGeneration = None

from .base_vlm import BaseVLMBackend, VLMBackendRegistry

logger = logging.getLogger(__name__)


# PaliGemma 任务 prompt 映射（自然语言风格）
PALIGEMMA_TASK_PROMPTS = {
    "CAPTION": "caption",
    "DETAILED_CAPTION": "caption",
    "MORE_DETAILED_CAPTION": "caption",
    "OD": "detect",
    "DENSE_CAPTION": "caption",
    "REGION_PROPOSAL": "detect",
    "OCR": "ocr",
    "OCR_WITH_REGION": "ocr",
    "VQA": "answer",
}

PALIGEMMA_DISPLAY_SPECIAL_TOKENS = (
    "<image>",
    "<bos>",
    "<eos>",
    "<pad>",
)


class PaliGemmaBackend(BaseVLMBackend):
    """PaliGemma 模型后端

    封装 PaliGemma 的模型加载、processor 调用、生成与前向传播逻辑。
    PaliGemma 使用自然语言任务提示，与 Florence-2 的特殊 token 风格不同。

    公共接口继承自 BaseVLMBackend。
    """

    ARCHITECTURE_TYPE = "decoder_only"
    BACKEND_NAME = "paligemma"

    def __init__(self, config: Any):
        """初始化"""
        super().__init__(config)
        logger.info(f"PaliGemmaBackend 已初始化（模型未加载，请调用 backend.load()）")

    # ------------------------------------------------------------------
    # 1. 模型加载（PaliGemma 特有逻辑）
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """加载 PaliGemma 基础模型"""
        if PaliGemmaForConditionalGeneration is None:
            raise RuntimeError(
                "transformers 库未安装或不支持 PaliGemma，"
                "请安装: pip install transformers>=4.40.0"
            )

        device = self._get_optimal_device()
        torch_dtype = self._get_optimal_dtype(device)

        model_kwargs = self._build_model_kwargs(device, torch_dtype)
        # PaliGemma 不使用 flash_attention_2
        model_kwargs.pop("attn_implementation", None)

        self._load_with_cpu_fallback(
            load_fn=PaliGemmaForConditionalGeneration.from_pretrained,
            model_name=self.config.model_name,
            model_kwargs=model_kwargs
        )

    def load_processor(self) -> None:
        """加载 PaliGemma Processor"""
        self._load_processor_base(AutoProcessor, self.config.model_name)

    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> List[str]:
        """解码 PaliGemma 输出并清理显示用 special tokens。

        FlorenceForge 的图片级 ``generate`` 为兼容 Florence-2 会以
        ``skip_special_tokens=False`` 调后端解码。PaliGemma 的 prompt 会包含大量
        ``<image>`` token；如果直接展示，会污染用户可见输出。这里仅移除
        PaliGemma 包装 token，保留 ``<loc_...>`` 等任务结构化 token。
        """
        self.load()
        if not hasattr(self.processor, "batch_decode"):
            raise RuntimeError("当前 processor 不支持 batch_decode")
        texts = self.processor.batch_decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
        )
        return [self._clean_decoded_text(text) for text in texts]

    def encode_with_task(
        self,
        images: List[Any],
        task_name: str,
        text_input: Optional[Union[str, List[str]]] = None,
        return_tensors: str = "pt",
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """PaliGemma 训练编码。

        PaliGemmaProcessor 支持 ``suffix=`` 作为 fine-tuning target，并会
        返回已屏蔽 prefix 的 ``labels``。这比手工拼接 prompt/answer 更可靠，
        也会在支持的 transformers 版本中返回 ``token_type_ids``。
        """
        self.load()

        if not images:
            raise ValueError("encode_with_task 需要至少一张图像")
        if not isinstance(images, list):
            images = [images]

        prompt = self.get_task_prompt(task_name)
        if "<image>" not in prompt:
            prompt = f"<image>{prompt}"
        text = [prompt] * len(images)

        processor_kwargs = dict(kwargs)
        processor_kwargs["return_tensors"] = return_tensors
        if text_input is not None:
            suffix = text_input
            if isinstance(suffix, str):
                suffix = [suffix] * len(images)
            processor_kwargs["suffix"] = suffix

        inputs = self.processor(text=text, images=images, **processor_kwargs)
        return {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }

    def prepare_labels(
        self,
        encoded_prompt: Dict[str, torch.Tensor],
        encoded_full: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """使用 PaliGemmaProcessor 生成的 labels；缺失时回退到基类逻辑。"""
        labels = encoded_full.get("labels")
        if labels is not None:
            return labels
        return super().prepare_labels(encoded_prompt, encoded_full)

    @staticmethod
    def _clean_decoded_text(text: str) -> str:
        """移除 PaliGemma 仅用于包装输入/序列边界的 special tokens。"""
        if not text:
            return ""
        for token in PALIGEMMA_DISPLAY_SPECIAL_TOKENS:
            text = text.replace(token, " ")
        return re.sub(r"\s+", " ", text).strip()

    # ------------------------------------------------------------------
    # 2. 任务相关（PaliGemma 特有映射）
    # ------------------------------------------------------------------

    def get_task_prompt(self, task_name: str) -> str:
        """获取 PaliGemma 任务 prompt

        PaliGemma 使用自然语言前缀而非特殊 token。
        """
        return PALIGEMMA_TASK_PROMPTS.get(task_name, task_name.lower())

    def supports_task(self, task_name: str) -> bool:
        """检查是否支持任务"""
        return task_name in PALIGEMMA_TASK_PROMPTS

    def _get_extra_model_info(self) -> Dict[str, Any]:
        return {"is_peft_model": self.is_peft_model}


# 注册后端（仅当依赖可用时）
try:
    if PaliGemmaForConditionalGeneration is not None:
        VLMBackendRegistry.register("paligemma", PaliGemmaBackend)
        VLMBackendRegistry.register("paligemma-3b", PaliGemmaBackend)
        logger.info("PaliGemmaBackend 已注册到 VLMBackendRegistry")
except Exception as e:
    logger.warning(f"PaliGemmaBackend 注册失败: {e}")
