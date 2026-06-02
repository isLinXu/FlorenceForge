"""Florence-2 VLM 后端实现

将原有的 Florence-2 模型加载、编码、生成逻辑封装为 BaseVLMBackend 的实现，
实现框架与 Florence-2 的解耦。

注意：公共逻辑（encode, generate, decode, forward, save_pretrained, load_pretrained,
get_model_info, _compile_model）已在 BaseVLMBackend 中实现，此文件仅保留
Florence-2 特有的模型加载和任务映射逻辑。
"""

import os
import torch
import logging
from typing import Optional, Dict, Any, List, Union
from pathlib import Path

try:
    from transformers import AutoProcessor, AutoModelForCausalLM
except ImportError:
    AutoProcessor = None
    AutoModelForCausalLM = None

from .base_vlm import (
    BaseVLMBackend,
    VLMBackendRegistry,
    _check_flash_attn_availability,
    _patch_transformers_config_defaults,
    _patch_transformers_import_check,
)

# 核心任务注册表为必需依赖；缺失则属于安装包损坏，直接抛错而非静默回退。
from ..tasks import FLORENCE2_TASKS, get_task_config

logger = logging.getLogger(__name__)


def _ensure_language_model_generation_mixin(model: torch.nn.Module) -> bool:
    """Restore generation support for older Florence-2 remote code.

    Transformers 4.50+ removed GenerationMixin from PreTrainedModel. Some
    Florence-2 remote-code checkpoints still define generate() on the wrapper
    model but delegate to ``language_model.generate()``, while the nested
    language model only implements ``prepare_inputs_for_generation``. Patch the
    nested instance at runtime so local checkpoints remain usable.
    """
    language_model = getattr(model, "language_model", None)
    if language_model is None:
        return False

    if callable(getattr(language_model, "generate", None)):
        _ensure_generation_config(language_model)
        return False

    if not callable(getattr(language_model, "prepare_inputs_for_generation", None)):
        return False

    try:
        from transformers.generation import GenerationConfig, GenerationMixin
    except Exception as exc:
        logger.warning("无法导入 GenerationMixin，Florence-2 推理生成可能不可用: %s", exc)
        return False

    original_cls = language_model.__class__
    if not issubclass(original_cls, GenerationMixin):
        patched_cls = type(
            f"{original_cls.__name__}WithGenerationMixin",
            (original_cls, GenerationMixin),
            {
                "__module__": original_cls.__module__,
                "__doc__": original_cls.__doc__,
            },
        )
        language_model.__class__ = patched_cls

    if getattr(language_model, "generation_config", None) is None:
        language_model.generation_config = GenerationConfig.from_model_config(language_model.config)

    logger.info(
        "已为旧版 Florence-2 language_model 动态补齐 GenerationMixin，"
        "以兼容 transformers>=4.50 的 generate() 行为"
    )
    return True


def _ensure_generation_config(language_model: torch.nn.Module) -> None:
    if getattr(language_model, "generation_config", None) is not None:
        return
    config = getattr(language_model, "config", None)
    if config is None:
        return
    try:
        from transformers.generation import GenerationConfig
    except Exception:
        return
    language_model.generation_config = GenerationConfig.from_model_config(config)


class Florence2Backend(BaseVLMBackend):
    """Florence-2 模型后端

    封装 Florence-2 的模型加载、processor 调用、生成与前向传播逻辑。
    公共接口（encode, generate, decode, forward, save, load, get_model_info）
    继承自 BaseVLMBackend。
    """

    ARCHITECTURE_TYPE = "encoder_decoder"
    BACKEND_NAME = "florence-2"

    def __init__(self, config: Any):
        """初始化

        Args:
            config: ModelConfig 配置对象，需包含 model_name, trust_remote_code 等字段
        """
        super().__init__(config)

        # 如果配置中启用 LoRA，在 model.py 层处理，backend 保持透明
        if getattr(config, 'use_lora', False):
            logger.info("Florence2Backend 初始化完成，LoRA 将在上层注入")
        else:
            logger.info("Florence2Backend 已初始化（模型未加载，请调用 backend.load()）")

    # ------------------------------------------------------------------
    # 1. 模型加载（Florence-2 特有逻辑）
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """加载 Florence-2 基础模型"""
        if AutoModelForCausalLM is None:
            raise RuntimeError("transformers 库未安装，无法加载 Florence-2 模型")

        _patch_transformers_config_defaults()

        device = self._get_optimal_device()
        torch_dtype = self._get_optimal_dtype(device)

        # flash_attn 检测
        flash_attn_available = _check_flash_attn_availability()
        attn_impl = "flash_attention_2" if (flash_attn_available and device != "cpu") else "eager"
        if attn_impl == "eager" and device != "cpu":
            _patch_transformers_import_check()

        model_kwargs = self._build_model_kwargs(device, torch_dtype, {
            "attn_implementation": attn_impl,
        })

        self._load_with_cpu_fallback(
            load_fn=AutoModelForCausalLM.from_pretrained,
            model_name=self.config.model_name,
            model_kwargs=model_kwargs
        )
        if self._model is not None:
            _ensure_language_model_generation_mixin(self._model)

    def load_processor(self) -> None:
        """加载 Florence-2 Processor"""
        self._load_processor_base(AutoProcessor, self.config.model_name)

    # ------------------------------------------------------------------
    # 2. 任务相关（Florence-2 特有映射）
    # ------------------------------------------------------------------

    def get_task_prompt(self, task_name: str) -> str:
        """获取任务 prompt"""
        try:
            config = get_task_config(task_name)
            return config.get("prompt", f"<{task_name}>")
        except KeyError:
            return f"<{task_name}>"

    def supports_task(self, task_name: str) -> bool:
        """检查是否支持任务"""
        return task_name in FLORENCE2_TASKS

    def _get_extra_model_info(self) -> Dict[str, Any]:
        """添加 Florence-2 特有的模型信息"""
        return {"is_peft_model": self.is_peft_model}

    # ------------------------------------------------------------------
    # 3. 训练编码（Florence-2 特有：task token 必须独占 text）
    # ------------------------------------------------------------------

    def encode_with_task(
        self,
        images: List[Any],
        task_name: str,
        text_input: Optional[str] = None,
        return_tensors: str = "pt",
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Florence-2 专用：task token 必须作为 text 唯一内容送入 processor。

        Microsoft 官方 processing_florence2.py L147 硬断言：
            assert _text == task_token, "Task token <X> should be the only token in the text."

        因此训练时不能简单拼接 "<CAPTION> A dummy ..."，必须：
          1. processor(text=task_token, images=...) 取得视觉编码 + task prompt token
          2. tokenizer(answer) 取得答案 token（不再走 processor，避免图像重复编码）
          3. 在数据集层把两段 token 拼接，并用 prepare_labels 屏蔽前半段
        """
        self.load()

        if not images:
            raise ValueError("encode_with_task 需要至少一张图像")
        if not isinstance(images, list):
            images = [images]

        prompt = self.get_task_prompt(task_name)

        # 1) 用 processor 编码 "纯 task token + image"
        prompt_encoded = self.processor(
            text=[prompt] * len(images),
            images=images,
            return_tensors=return_tensors,
        )

        # 2) 如果给了答案文本，单独用 tokenizer 编码并拼接 token ids
        if text_input:
            tokenizer = getattr(self.processor, "tokenizer", None)
            if tokenizer is None:
                raise RuntimeError("Florence-2 processor 未暴露 tokenizer，无法拼接 answer token")

            # add_special_tokens=False 避免重复插入 <s>/</s>
            answer_encoded = tokenizer(
                [text_input] * len(images),
                add_special_tokens=False,
                return_tensors=return_tensors,
                padding=False,
            )

            prompt_ids = prompt_encoded["input_ids"]
            prompt_mask = prompt_encoded.get("attention_mask")
            answer_ids = answer_encoded["input_ids"]
            answer_mask = answer_encoded.get("attention_mask")

            # 兼容 prompt 末尾的 </s>：先剥离，再拼 answer，最后补 </s>
            eos_id = getattr(tokenizer, "eos_token_id", None)
            new_input_ids = []
            new_attn_mask = []
            for i in range(prompt_ids.shape[0]):
                p_ids = prompt_ids[i]
                a_ids = answer_ids[i] if answer_ids.dim() == 2 else answer_ids
                if eos_id is not None and p_ids.numel() > 0 and int(p_ids[-1]) == int(eos_id):
                    p_ids = p_ids[:-1]
                merged = torch.cat([p_ids, a_ids], dim=0)
                if eos_id is not None:
                    merged = torch.cat([merged, torch.tensor([eos_id], dtype=merged.dtype)], dim=0)
                new_input_ids.append(merged)

                if prompt_mask is not None:
                    p_mask = prompt_mask[i]
                    a_mask = answer_mask[i] if answer_mask is not None and answer_mask.dim() == 2 else answer_mask
                    if eos_id is not None and p_mask.numel() > 0:
                        p_mask = p_mask[:-1]
                    parts = [p_mask, a_mask if a_mask is not None else torch.ones_like(answer_ids[i])]
                    if eos_id is not None:
                        parts.append(torch.tensor([1], dtype=p_mask.dtype))
                    new_attn_mask.append(torch.cat(parts, dim=0))

            # batch 内 token 长度可能不同，用 tokenizer.pad_token_id 右 padding
            pad_id = getattr(tokenizer, "pad_token_id", None) or 1
            max_len = max(x.shape[0] for x in new_input_ids)
            padded_ids = torch.full((len(new_input_ids), max_len), pad_id, dtype=new_input_ids[0].dtype)
            padded_mask = torch.zeros((len(new_input_ids), max_len), dtype=torch.long)
            for i, ids in enumerate(new_input_ids):
                padded_ids[i, : ids.shape[0]] = ids
                if new_attn_mask:
                    padded_mask[i, : new_attn_mask[i].shape[0]] = new_attn_mask[i]
                else:
                    padded_mask[i, : ids.shape[0]] = 1

            out: Dict[str, torch.Tensor] = {
                "input_ids": padded_ids,
                "attention_mask": padded_mask,
                "pixel_values": prompt_encoded["pixel_values"],
                # 记录 prompt token 长度（去掉末尾 </s> 之后），供 prepare_labels 使用
                "prompt_lengths": torch.tensor(
                    [
                        (prompt_ids[i].shape[0] - (1 if eos_id is not None and int(prompt_ids[i][-1]) == int(eos_id) else 0))
                        for i in range(prompt_ids.shape[0])
                    ],
                    dtype=torch.long,
                ),
            }
        else:
            # 没有 answer，直接返回 prompt 编码
            out = {k: v for k, v in prompt_encoded.items()}

        # 设备迁移
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in out.items()
        }

    def prepare_labels(
        self,
        encoded_prompt: Dict[str, torch.Tensor],
        encoded_full: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Florence-2 训练 labels：prompt 段置 -100，answer 段保留 token id。

        优先使用 encode_with_task 注入的 prompt_lengths；否则回退到 base 实现。
        """
        prompt_lengths = encoded_full.get("prompt_lengths")
        full_ids = encoded_full["input_ids"]
        if full_ids.dim() == 1:
            full_ids = full_ids.unsqueeze(0)

        if prompt_lengths is None:
            return super().prepare_labels(encoded_prompt, encoded_full)

        labels = full_ids.clone()
        if prompt_lengths.dim() == 0:
            prompt_lengths = prompt_lengths.unsqueeze(0)
        for i in range(labels.shape[0]):
            p_len = int(prompt_lengths[i].item())
            labels[i, :p_len] = -100
        return labels.squeeze(0) if labels.shape[0] == 1 else labels


# 注册后端
VLMBackendRegistry.register("florence-2", Florence2Backend)
VLMBackendRegistry.register("florence2", Florence2Backend)
