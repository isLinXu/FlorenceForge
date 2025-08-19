"""FlorenceForge核心模型模块

封装Florence-2模型的加载、配置和推理功能
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Union
try:
    from PIL import Image
except ImportError:
    # 如果PIL不可用，创建占位符
    class Image:
        class Image:
            pass
try:
    from transformers import AutoProcessor, AutoModelForCausalLM
except ImportError:
    # 兼容性处理：如果AutoProcessor不可用，使用替代方案
    try:
        from transformers import AutoModelForCausalLM
        AutoProcessor = None
    except ImportError:
        # 最后的兼容性处理
        AutoProcessor = None
        AutoModelForCausalLM = None

try:
    from peft import LoraConfig, get_peft_model, PeftModel
except ImportError:
    # PEFT可选依赖
    LoraConfig = None
    get_peft_model = None
    PeftModel = None

import logging

def _check_flash_attn_availability() -> bool:
    """检测 flash_attn 是否可用
    
    Returns:
        bool: True 如果 flash_attn 可用，False 否则
    """
    try:
        import flash_attn
        return True
    except ImportError:
        return False

def _patch_transformers_import_check():
    """临时修补 transformers 的导入检查以绕过 flash_attn 依赖"""
    try:
        from transformers import dynamic_module_utils
        original_check_imports = dynamic_module_utils.check_imports
        
        def patched_check_imports(filename):
            """修补后的导入检查函数，忽略 flash_attn 依赖"""
            try:
                return original_check_imports(filename)
            except ImportError as e:
                if 'flash_attn' in str(e):
                    logger.warning(f"忽略 flash_attn 导入错误: {e}")
                    return []
                else:
                    raise e
        
        dynamic_module_utils.check_imports = patched_check_imports
        return True
    except Exception as e:
        logger.warning(f"无法修补 transformers 导入检查: {e}")
        return False

try:
    from .config import ModelConfig, LoRAConfig, TrainingConfig
except ImportError:
    # 如果config模块不存在，创建占位符类
    class ModelConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    class LoRAConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    class TrainingConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

try:
    from .tasks import FLORENCE2_TASKS, get_task_config
except ImportError:
    # 如果tasks模块不存在，创建默认配置
    FLORENCE2_TASKS = {}
    def get_task_config(task_name):
        """获取指定任务的配置信息
        
        Args:
            task_name (str): 任务名称，如'caption', 'detection', 'segmentation'等
            
        Returns:
            dict: 包含任务配置信息的字典，至少包含'task'字段
            
        Note:
            这是一个兼容性函数，当FLORENCE2_TASKS不可用时提供基本的任务配置
        """
        return {"task": task_name}

logger = logging.getLogger(__name__)

class Florence2MultiTaskModel(nn.Module):
    """Florence-2多任务模型封装
    
    提供统一的模型接口，支持LoRA微调和多任务推理
    """
    
    def __init__(self, config: ModelConfig):
        """初始化模型
        
        Args:
            config: 模型配置
        """
        super().__init__()
        self.config = config
        self.model = None
        self.processor = None
        self.is_peft_model = False
        
        self._load_model()
        self._load_processor()
        
        if config.use_lora:
            self._setup_lora()
    
    def _load_model(self) -> None:
        """加载基础模型"""
        logger.info(f"正在加载模型: {self.config.model_name}")
        
        # 强制禁用MPS，使用CPU
        import os
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        
        # 动态检测 flash_attn 可用性
        flash_attn_available = _check_flash_attn_availability()
        if flash_attn_available:
            attn_implementation = "flash_attention_2"
            logger.info("检测到 flash_attn 可用，使用 flash_attention_2")
        else:
            attn_implementation = "eager"
            logger.info("flash_attn 不可用，回退到 eager 模式")
            # 修补 transformers 的导入检查以绕过 flash_attn 依赖
            _patch_transformers_import_check()
        
        model_kwargs = {
            "trust_remote_code": self.config.trust_remote_code,
            "device_map": "cpu",  # 强制使用CPU
            "torch_dtype": torch.float32,  # 使用float32避免精度问题
            "attn_implementation": attn_implementation  # 动态设置注意力实现
        }
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                **model_kwargs
            )
            # 确保模型在CPU上
            self.model = self.model.to("cpu")
            logger.info("模型加载成功，使用CPU设备")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def _load_processor(self) -> None:
        """加载处理器"""
        logger.info(f"正在加载处理器: {self.config.model_name}")
        
        if AutoProcessor is None:
            logger.warning("AutoProcessor不可用，跳过处理器加载")
            self.processor = None
            return
        
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.config.model_name,
                trust_remote_code=self.config.trust_remote_code
            )
            logger.info("处理器加载成功")
        except Exception as e:
            logger.error(f"处理器加载失败: {e}")
            self.processor = None
            logger.warning("处理器加载失败，继续使用None")
    
    def _setup_lora(self) -> None:
        """设置LoRA配置"""
        logger.info("正在设置LoRA配置")
        
        if LoraConfig is None or get_peft_model is None:
            logger.warning("PEFT库不可用，跳过LoRA设置")
            return
        
        lora_config = LoraConfig(
            r=self.config.lora_config.r,
            lora_alpha=self.config.lora_config.lora_alpha,
            target_modules=self.config.lora_config.target_modules,
            lora_dropout=self.config.lora_config.lora_dropout,
            bias=self.config.lora_config.bias,
            task_type=self.config.lora_config.task_type
        )
        
        try:
            self.model = get_peft_model(self.model, lora_config)
            self.is_peft_model = True
            logger.info("LoRA配置设置成功")
            
            # 打印可训练参数信息
            self.print_trainable_parameters()
        except Exception as e:
            logger.error(f"LoRA配置设置失败: {e}")
            raise
    
    def print_trainable_parameters(self) -> None:
        """打印可训练参数信息"""
        if not self.is_peft_model:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        else:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(
            f"可训练参数: {trainable_params:,} || "
            f"总参数: {total_params:,} || "
            f"可训练比例: {100 * trainable_params / total_params:.2f}%"
        )
    
    def forward(self, input_ids: torch.Tensor, pixel_values: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None, 
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """前向传播
        
        Args:
            input_ids: 输入token IDs
            pixel_values: 图像像素值
            attention_mask: 注意力掩码
            labels: 标签（训练时使用）
            
        Returns:
            模型输出
        """
        return self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def generate(
        self,
        images: Union[Image.Image, List[Image.Image]],
        task_prompt: str,
        text_input: Optional[str] = None,
        max_new_tokens: int = 1024,
        num_beams: int = 3,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        **kwargs
    ) -> Union[str, List[str]]:
        """生成文本
        
        Args:
            images: 输入图像
            task_prompt: 任务提示
            text_input: 文本输入（某些任务需要）
            max_new_tokens: 最大新token数
            num_beams: beam search数量
            do_sample: 是否采样
            temperature: 采样温度
            top_p: top-p采样参数
            **kwargs: 其他生成参数
            
        Returns:
            生成的文本
        """
        # 确保输入是列表格式
        if isinstance(images, Image.Image):
            images = [images]
            single_image = True
        else:
            single_image = False
        
        # 构建提示
        if text_input:
            prompt = f"{task_prompt}{text_input}"
        else:
            prompt = task_prompt
        
        # 处理输入
        inputs = self.processor(
            text=prompt,
            images=images,
            return_tensors="pt"
        )
        
        # 移动到模型设备
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 生成
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                **kwargs
            )
        
        # 解码
        generated_text = self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=False
        )
        
        # 清理输出
        cleaned_text = []
        for text in generated_text:
            # 移除输入提示部分
            if prompt in text:
                text = text.replace(prompt, "").strip()
            cleaned_text.append(text)
        
        return cleaned_text[0] if single_image else cleaned_text
    
    def predict_task(
        self,
        images: Union[Image.Image, List[Image.Image]],
        task_name: str,
        text_input: Optional[str] = None,
        **kwargs
    ) -> Union[str, List[str]]:
        """执行特定任务的预测
        
        Args:
            images: 输入图像
            task_name: 任务名称
            text_input: 文本输入（某些任务需要）
            **kwargs: 其他生成参数
            
        Returns:
            预测结果
        """
        # 获取任务配置
        task_config = get_task_config(task_name)
        
        # 设置生成参数
        generation_kwargs = {
            "max_new_tokens": task_config.get("max_new_tokens", 1024),
            "num_beams": task_config.get("num_beams", 3)
        }
        generation_kwargs.update(kwargs)
        
        # 执行生成
        return self.generate(
            images=images,
            task_prompt=task_config["prompt"],
            text_input=text_input,
            **generation_kwargs
        )
    
    def save_pretrained(self, save_directory: str) -> None:
        """保存模型
        
        Args:
            save_directory: 保存目录
        """
        logger.info(f"正在保存模型到: {save_directory}")
        
        if self.is_peft_model:
            self.model.save_pretrained(save_directory)
        else:
            self.model.save_pretrained(save_directory)
            
        # 保存处理器
        self.processor.save_pretrained(save_directory)
        
        logger.info("模型保存完成")
    
    @classmethod
    def load_pretrained(
        cls, 
        model_path: str, 
        config: Optional[ModelConfig] = None,
        is_peft_model: bool = False
    ) -> 'Florence2MultiTaskModel':
        """加载预训练模型
        
        Args:
            model_path: 模型路径
            config: 模型配置
            is_peft_model: 是否为PEFT模型
            
        Returns:
            模型实例
        """
        if config is None:
            config = ModelConfig()
            config.model_name = model_path
        
        # 创建模型实例
        model_instance = cls(config)
        
        if is_peft_model:
            # 加载PEFT模型
            model_instance.model = PeftModel.from_pretrained(
                model_instance.model, 
                model_path
            )
            model_instance.is_peft_model = True
        
        return model_instance
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息
        
        Returns:
            模型信息字典
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "model_name": self.config.model_name,
            "is_peft_model": self.is_peft_model,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_ratio": trainable_params / total_params,
            "device": str(next(self.model.parameters()).device),
            "dtype": str(next(self.model.parameters()).dtype)
        }
    
    def to(self, device: Union[str, torch.device]) -> 'Florence2MultiTaskModel':
        """移动模型到指定设备
        
        Args:
            device: 目标设备
            
        Returns:
            自身实例
        """
        self.model = self.model.to(device)
        return self
    
    def train(self, mode: bool = True) -> 'Florence2MultiTaskModel':
        """设置训练模式
        
        Args:
            mode: 是否为训练模式
            
        Returns:
            自身实例
        """
        self.model.train(mode)
        return self
    
    def eval(self) -> 'Florence2MultiTaskModel':
        """设置评估模式
        
        Returns:
            自身实例
        """
        self.model.eval()
        return self