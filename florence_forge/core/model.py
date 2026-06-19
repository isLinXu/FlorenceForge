"""FlorenceForge核心模型模块

封装Florence-2模型的加载、配置和推理功能
"""

import logging
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Union

from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel

logger = logging.getLogger(__name__)

# 从 backends 模块导入公共工具函数（消除 model.py 与后端间的重复）
from .backends.base_vlm import _check_flash_attn_availability, _patch_transformers_import_check


from .config import ModelConfig, LoRAConfig, TrainingConfig
from .tasks import FLORENCE2_TASKS, get_task_config, get_task_config_typed
from .backends import VLMBackendRegistry, BaseVLMBackend


class Florence2MultiTaskModel(nn.Module):
    """Florence-2多任务模型封装（基于 VLM 后端抽象）

    提供统一的模型接口，支持LoRA微调和多任务推理。
    通过 backend 属性与底层 VLM 解耦，默认使用 Florence-2 后端。
    """

    def __init__(self, config: ModelConfig):
        """初始化模型

        Args:
            config: 模型配置，新增可选字段 backend_name（默认 "florence-2"）
        """
        super().__init__()
        self.config = config
        self._backend: Optional[BaseVLMBackend] = None
        self.is_peft_model = False

        # 初始化 VLM 后端
        self._init_backend()

        # LoRA 需要底层模型已加载，延迟到 load() 后注入。

    def _init_backend(self) -> None:
        """初始化 VLM 后端（延迟加载模式，不自动加载模型）"""
        backend_name = getattr(self.config, 'backend_name', 'florence-2')

        if VLMBackendRegistry.is_registered(backend_name):
            logger.info(f"使用 VLM 后端: {backend_name}")
            self._backend = VLMBackendRegistry.create(backend_name, self.config)
            # 注册为子模块，确保 PyTorch 参数遍历、设备迁移、状态保存/加载正常工作
            if isinstance(self._backend, nn.Module):
                self.add_module('_backend', self._backend)
            logger.info(f"后端已初始化（延迟加载模式），请显式调用 model.load() 加载模型")
        else:
            # 后端未注册，抛出明确错误而非静默回退
            available = VLMBackendRegistry.list_backends()
            raise ValueError(
                f"VLM 后端 '{backend_name}' 未注册。"
                f"可用后端: {available}。"
                f"请在 config 中设置正确的 backend_name，或检查 backends/ 模块是否正确导入。"
            )

    def load(self) -> 'Florence2MultiTaskModel':
        """显式加载模型和处理器

        Returns:
            自身实例（支持链式调用）
        """
        if self._backend is None:
            raise RuntimeError("后端未初始化，请先创建模型实例")
        self._backend.load()
        if getattr(self.config, 'use_lora', False) and not self.is_peft_model:
            self._setup_lora()
        return self

    # ------------------------------------------------------------------
    # 属性代理：保持向后兼容
    # ------------------------------------------------------------------

    @property
    def model(self) -> nn.Module:
        """底层模型实例（通过后端访问）"""
        if self._backend is None:
            raise RuntimeError("后端未初始化，请检查配置")
        return self._backend.model

    @model.setter
    def model(self, value):
        if self._backend is None:
            raise RuntimeError("后端未初始化，请检查配置")
        self._backend._model = value

    @property
    def processor(self):
        """处理器实例（通过后端访问）"""
        if self._backend is None:
            raise RuntimeError("后端未初始化，请检查配置")
        return self._backend.processor

    @processor.setter
    def processor(self, value):
        if self._backend is None:
            raise RuntimeError("后端未初始化，请检查配置")
        self._backend._processor = value

    # ------------------------------------------------------------------
    # 属性透传：兼容 peft>=0.6 对 generation 接口的强依赖
    # peft.PeftModel.__init__ 会访问 base_model.prepare_inputs_for_generation
    # 等 HF 标准生成接口，必须把这些请求转发到底层 HF 模型
    # ------------------------------------------------------------------

    # 允许透传到底层 HF 模型的属性白名单
    # 仅覆盖 generation / config 相关接口，避免无意中遮蔽 nn.Module 的标准属性
    _BACKEND_PROXY_ATTRS = frozenset({
        "prepare_inputs_for_generation",
        "_prepare_encoder_decoder_kwargs_for_generation",
        "_prepare_decoder_input_ids_for_generation",
        "_reorder_cache",
        "can_generate",
        "generation_config",
        "_supports_cache_class",
        "_supports_static_cache",
        "_supports_quantized_cache",
        "warnings_issued",
        "main_input_name",
        "base_model_prefix",
    })

    def __getattr__(self, name: str):
        # nn.Module.__getattr__ 已经处理了 _modules / _parameters / _buffers，
        # 这里仅在前者都没找到时介入。注意：直接访问 self._backend 会再次触发
        # __getattr__，必须通过 nn.Module 的内部存储获取以避免无限递归。
        if name in type(self)._BACKEND_PROXY_ATTRS:
            try:
                backend = super().__getattr__("_backend")
            except AttributeError:
                raise AttributeError(name)
            if backend is None or backend._model is None:
                raise AttributeError(name)
            return getattr(backend._model, name)
        return super().__getattr__(name)

    # ------------------------------------------------------------------
    # 前向传播 / 生成（委托给后端）
    # ------------------------------------------------------------------

    def forward(self, input_ids: torch.Tensor, pixel_values: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """前向传播（委托给后端）"""
        if self._backend is None:
            raise RuntimeError("后端未初始化，请检查配置")
        return self._backend.forward(input_ids, pixel_values, attention_mask, labels)

    def generate(
        self,
        images: Optional[Union[Image.Image, List[Image.Image]]] = None,
        task_prompt: Optional[str] = None,
        text_input: Optional[str] = None,
        input_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 1024,
        num_beams: int = 3,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        **kwargs
    ) -> Union[str, List[str], torch.Tensor]:
        """生成结果。

        支持两种调用方式：
        1. 图片级接口：generate(images=..., task_prompt=...)，返回解码后的文本。
        2. 张量级接口：generate(input_ids=..., pixel_values=...)，返回生成 token ids。
        """
        if self._backend is None:
            raise RuntimeError("后端未初始化，请检查配置")

        tensor_mode = input_ids is not None or pixel_values is not None
        if tensor_mode:
            if input_ids is None or pixel_values is None:
                raise ValueError("张量级 generate 需要同时提供 input_ids 和 pixel_values")
            target_device = self._backend.device
            input_ids = input_ids.to(target_device)
            pixel_values = pixel_values.to(target_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(target_device)
            return self._backend.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                **kwargs,
            )

        if images is None:
            raise ValueError("图片级 generate 需要提供 images")
        if task_prompt is None:
            raise ValueError("图片级 generate 需要提供 task_prompt")

        if isinstance(images, Image.Image):
            images = [images]
            single_image = True
        else:
            single_image = False

        prompt = f"{task_prompt}{text_input}" if text_input else task_prompt

        # 编码
        inputs = self._backend.encode(text=[prompt] * len(images), images=images)

        # 生成
        generated_ids = self._backend.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        )

        # 解码
        generated_text = self._backend.decode(generated_ids, skip_special_tokens=False)

        # 清理输出（移除输入 prompt）
        cleaned = []
        for text in generated_text:
            if prompt in text:
                text = text.replace(prompt, "").strip()
            cleaned.append(text)

        return cleaned[0] if single_image else cleaned

    def predict_task(
        self,
        images: Union[Image.Image, List[Image.Image]],
        task_name: str,
        text_input: Optional[str] = None,
        **kwargs
    ) -> Union[str, List[str]]:
        """执行特定任务的预测"""
        task_config = get_task_config_typed(task_name)
        generation_kwargs = {
            "max_new_tokens": task_config.max_new_tokens,
            "num_beams": task_config.num_beams
        }
        generation_kwargs.update(kwargs)
        return self.generate(
            images=images,
            task_prompt=task_config.prompt,
            text_input=text_input,
            **generation_kwargs
        )

    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> List[str]:
        """解码 token ids（委托给后端）。"""
        if self._backend is None:
            raise RuntimeError("后端未初始化，请检查配置")
        return self._backend.decode(token_ids, skip_special_tokens=skip_special_tokens)

    # ------------------------------------------------------------------
    # 保存 / 加载 / 信息
    # ------------------------------------------------------------------

    def print_trainable_parameters(self) -> None:
        """打印可训练参数信息"""
        info = self.get_model_info()
        logger.info(
            f"可训练参数: {info['trainable_parameters']:,} || "
            f"总参数: {info['total_parameters']:,} || "
            f"可训练比例: {100 * info['trainable_ratio']:.2f}%"
        )

    # ------------------------------------------------------------------
    # LoRA 设置
    # ------------------------------------------------------------------

    def _setup_lora(self) -> None:
        """设置LoRA配置"""
        target_model = self.model  # 通过 property 获取底层模型
        if not isinstance(target_model, nn.Module):
            logger.warning("底层模型不是 torch.nn.Module，跳过 LoRA 注入")
            return

        lora_config = LoraConfig(
            r=self.config.lora_config.r,
            lora_alpha=self.config.lora_config.lora_alpha,
            target_modules=self.config.lora_config.target_modules,
            lora_dropout=self.config.lora_config.lora_dropout,
            bias=self.config.lora_config.bias,
            task_type=self.config.lora_config.task_type,
            modules_to_save=self.config.lora_config.modules_to_save or None,
        )

        try:
            new_model = get_peft_model(target_model, lora_config)
            if self._backend is None:
                raise RuntimeError("后端未初始化，无法设置 LoRA")
            self._backend._model = new_model
            self._backend.is_peft_model = True
            self.is_peft_model = True
            logger.info("LoRA配置设置成功")
            self.print_trainable_parameters()
        except Exception as e:
            # 使用 exception() 记录完整 traceback
            logger.exception(f"LoRA配置设置失败: {e}")
            raise

    # ------------------------------------------------------------------
    # 模型保存 / 加载 / 信息
    # ------------------------------------------------------------------

    def save_pretrained(self, save_directory: str) -> None:
        """保存模型

        Args:
            save_directory: 保存目录
        """
        if self._backend is None:
            raise RuntimeError("后端未初始化，请检查配置")

        logger.info(f"正在保存模型到: {save_directory}")
        self._backend.save_pretrained(save_directory)
        logger.info("模型保存完成")
    
    @classmethod
    def load_pretrained(
        cls,
        model_path: str,
        config: Optional[ModelConfig] = None,
        is_peft_model: bool = False
    ) -> 'Florence2MultiTaskModel':
        """从检查点加载模型

        Args:
            model_path: 检查点路径
            config: 模型配置（可选，会从检查点自动推断）
            is_peft_model: 是否为 PEFT 模型

        Returns:
            加载好的模型实例
        """
        if config is None:
            config = ModelConfig()
            config.model_name = model_path

        # 创建模型实例（延迟加载模式，不会自动加载模型）
        model_instance = cls(config)

        if model_instance._backend is None:
            raise RuntimeError("后端未初始化，请检查配置")

        # 通过后端加载检查点
        logger.info(f"正在从检查点加载模型: {model_path}")
        model_instance._backend.load_pretrained(model_path, is_peft_model=is_peft_model)
        model_instance.is_peft_model = is_peft_model

        logger.info(f"模型加载完成，使用设备: {model_instance._backend.device}")
        return model_instance
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息

        Returns:
            模型信息字典
        """
        if self._backend is None:
            raise RuntimeError("后端未初始化，请检查配置")
        return self._backend.get_model_info()
    
    def to(self, device: Union[str, torch.device]) -> 'Florence2MultiTaskModel':
        """移动模型到指定设备

        Args:
            device: 目标设备

        Returns:
            自身实例
        """
        if self._backend is None:
            raise RuntimeError("后端未初始化，请检查配置")
        if self._backend.model is None:
            raise RuntimeError("模型未加载，请先调用 backend.load() 或确保后端已加载模型")
        # 移动模型并同步后端设备状态
        self.model = self.model.to(device)
        self._backend._device = str(device) if isinstance(device, torch.device) else device
        return self

    def train(self, mode: bool = True) -> 'Florence2MultiTaskModel':
        """设置训练模式

        Args:
            mode: 是否为训练模式

        Returns:
            自身实例
        """
        if self._backend is None:
            raise RuntimeError("后端未初始化，请检查配置")
        if self._backend.model is None:
            raise RuntimeError("模型未加载，请先调用 backend.load() 或确保后端已加载模型")
        self.model.train(mode)
        return self

    def eval(self) -> 'Florence2MultiTaskModel':
        """设置评估模式

        Returns:
            自身实例
        """
        if self._backend is None:
            raise RuntimeError("后端未初始化，请检查配置")
        if self._backend.model is None:
            raise RuntimeError("模型未加载，请先调用 backend.load() 或确保后端已加载模型")
        self.model.eval()
        return self
