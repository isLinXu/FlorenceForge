"""PaliGemma VLM 后端实现

支持 google/paligemma-3b-pt-224 等 PaliGemma 系列模型，
实现框架与 PaliGemma 的解耦。

公共逻辑（encode, generate, decode, forward, save, load, get_model_info,
_compile_model）已在 BaseVLMBackend 中实现，此文件仅保留 PaliGemma 特有的
模型加载和任务映射逻辑。
"""

import logging
from typing import Dict, Any

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
        logger.info("PaliGemmaBackend 已初始化（模型未加载，请调用 backend.load()）")

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
