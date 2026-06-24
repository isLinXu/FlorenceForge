"""腾讯优图 YouTu-VL VLM 后端实现

支持腾讯优图实验室开源的 Youtu-VL / Youtu-VL-4B-Instruct 系列模型。

公共逻辑（encode, generate, decode, forward, save, load, get_model_info,
_compile_model）已在 BaseVLMBackend 中实现，此文件仅保留 YouTu-VL 特有的
模型加载、任务映射和输出解析逻辑。
"""

import re
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

try:
    from transformers import AutoProcessor, AutoModelForImageTextToText, AutoTokenizer
    AutoModelForVision2Seq = AutoModelForImageTextToText  # 向后兼容
except ImportError:
    try:
        from transformers import AutoModelForVision2Seq
    except ImportError:
        AutoModelForVision2Seq = None
    AutoModelForImageTextToText = None
    AutoProcessor = None
    AutoTokenizer = None

from .base_vlm import BaseVLMBackend, VLMBackendRegistry

logger = logging.getLogger(__name__)


# YouTu-VL 任务 prompt 映射
YOUTUVL_TASK_PROMPTS = {
    "CAPTION": "Describe the image in detail.",
    "DETAILED_CAPTION": "Provide a comprehensive description of the image.",
    "OD": "Detect all objects in the image and provide their locations.",
    "DENSE_REGION_CAPTION": "Detect all objects and describe each region in detail.",
    "REGION_PROPOSAL": "Propose all interesting regions in the image.",
    "VISUAL_GROUNDING": "Locate the object described by the text in the image.",
    "VQA": "Answer the question based on the image.",
    "OCR": "Read all text present in the image.",
    "OCR_WITH_REGION": "Read all text in the image and provide their locations.",
    "REGION_TO_SEGMENTATION": "Segment the specified region in the image.",
    "REFERRING_EXPRESSION_SEGMENTATION": "Segment the object referred to in the expression.",
    "SEGMENTATION": "Segment all objects in the image.",
    "POSE_ESTIMATION": "Estimate the pose of all persons in the image.",
    "GUI": "Understand the GUI interface and answer the question.",
    "DOCUMENT": "Extract information from the document image.",
}


class YouTuVLBackend(BaseVLMBackend):
    """腾讯优图 YouTu-VL 模型后端

    封装 YouTu-VL 的模型加载、processor 调用、生成与前向传播逻辑。
    公共接口继承自 BaseVLMBackend。
    """

    ARCHITECTURE_TYPE = "encoder_decoder"
    BACKEND_NAME = "youtuvl"

    def __init__(self, config: Any):
        """初始化"""
        super().__init__(config)

        # 加载自定义任务 prompt
        custom_prompts = getattr(config, "task_prompts", None)
        self._task_prompts = {**YOUTUVL_TASK_PROMPTS, **(custom_prompts or {})}
        custom_tasks = getattr(config, "supported_tasks", None)
        self._supports_tasks = list(custom_tasks or self._task_prompts.keys())

        logger.info("YouTuVLBackend 已初始化（模型未加载，请调用 backend.load()）")

    # ------------------------------------------------------------------
    # 1. 模型加载（YouTu-VL 特有逻辑）
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """加载 YouTu-VL 基础模型"""
        device = self._get_optimal_device()
        torch_dtype = self._get_optimal_dtype(device)
        model_kwargs = self._build_model_kwargs(device, torch_dtype)

        # 选择加载函数
        if AutoModelForVision2Seq is not None:
            load_fn = AutoModelForVision2Seq.from_pretrained
        else:
            try:
                from transformers import AutoModel
                load_fn = AutoModel.from_pretrained
            except ImportError:
                raise RuntimeError("transformers 库未安装")

        self._load_with_cpu_fallback(
            load_fn=load_fn,
            model_name=self.config.model_name,
            model_kwargs=model_kwargs
        )

    def load_processor(self) -> None:
        """加载 YouTu-VL Processor"""
        self._load_processor_base(AutoProcessor, self.config.model_name)

        # 如果 AutoProcessor 加载失败，尝试回退到 Tokenizer
        if self._processor is None and AutoTokenizer is not None:
            try:
                self._processor = AutoTokenizer.from_pretrained(
                    self.config.model_name,
                    trust_remote_code=True,
                    use_fast=True
                )
                logger.info("YouTu-VL Tokenizer 加载成功（作为 processor 回退）")
            except Exception as e2:
                logger.error(f"Tokenizer 加载也失败: {e2}")

    # ------------------------------------------------------------------
    # 2. 任务相关（YouTu-VL 特有映射）
    # ------------------------------------------------------------------

    def get_task_prompt(self, task_name: str) -> str:
        """获取 YouTu-VL 任务 prompt"""
        return self._task_prompts.get(task_name, task_name)

    def supports_task(self, task_name: str) -> bool:
        """检查是否支持任务"""
        return task_name in self._supports_tasks

    def _get_extra_model_info(self) -> Dict[str, Any]:
        return {
            "architecture_type": "encoder_decoder",
            "is_peft_model": self.is_peft_model,
        }

    # ------------------------------------------------------------------
    # 3. YouTu-VL 特有功能
    # ------------------------------------------------------------------

    def generate_with_task(
        self,
        image: Any,
        task_name: str,
        text_input: Optional[str] = None,
        max_new_tokens: int = 1024,
        **kwargs
    ) -> str:
        """使用任务名称进行推理（便捷方法）"""
        from PIL import Image as PILImage

        if isinstance(image, (str, Path)):
            with PILImage.open(image) as img:
                image = img.convert("RGB")

        task_prompt = self.get_task_prompt(task_name)
        prompt = f"{task_prompt} {text_input}" if text_input else task_prompt

        inputs = self.encode(images=[image], text=[prompt])
        generated_ids = self.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=max_new_tokens,
            **kwargs
        )
        texts = self.decode(generated_ids, skip_special_tokens=False)
        return texts[0] if texts else ""

    @staticmethod
    def parse_detection_output(text: str) -> List[Dict[str, Any]]:
        """解析 YouTu-VL 的检测输出文本"""
        results = []
        pattern = re.compile(
            r"<ref>(.*?)</ref>.*?<box>.*?<x_min>(\d+)</x_min>.*?<y_min>(\d+)</y_min>.*?<x_max>(\d+)</x_max>.*?<y_max>(\d+)</y_max>.*?</box>",
            re.DOTALL
        )
        for match in pattern.finditer(text):
            results.append({
                "label": match.group(1).strip(),
                "bbox": [int(match.group(2)), int(match.group(3)), int(match.group(4)), int(match.group(5))],
            })
        return results


# 注册后端
try:
    VLMBackendRegistry.register("youtuvl", YouTuVLBackend)
    VLMBackendRegistry.register("youtu-vl", YouTuVLBackend)
    VLMBackendRegistry.register("tencent-youtuvl", YouTuVLBackend)
    logger.info("YouTuVLBackend 已注册到 VLMBackendRegistry")
except Exception as e:
    logger.warning(f"YouTuVLBackend 注册失败: {e}")
