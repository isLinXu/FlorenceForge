"""模型量化模块

支持多种现代量化方法，大幅降低模型显存占用和推理延迟：
- bitsandbytes 4-bit/8-bit 量化（最实用，训练时即可使用）
- GPTQ 4-bit 量化（通过 optimum/auto-gptq，推理专用）
- AWQ 4-bit 量化（通过 autoawq，推理专用）
- PyTorch 动态量化（CPU 推理场景）

量化方法对比：
┌─────────────┬──────────┬──────────┬─────────────┬──────────────┐
│ 方法        │ 精度损失 │ 显存节省 │ 训练时可用  │ 推理加速     │
├─────────────┼──────────┼──────────┼─────────────┼──────────────┤
│ bnb-4bit    │ 中等     │ ~75%     │ ✅ QLoRA    │ ~2x          │
│ bnb-8bit    │ 较小     │ ~50%     │ ✅          │ ~1.5x        │
│ gptq-4bit   │ 中等     │ ~75%     │ ❌          │ ~2-3x        │
│ awq-4bit    │ 较小     │ ~75%     │ ❌          │ ~2-4x        │
│ dynamic-int8│ 较大     │ ~50%     │ ❌          │ ~2x (CPU)    │
└─────────────┴──────────┴──────────┴─────────────┴──────────────┘

使用方式：
    # 训练时使用 bitsandbytes 4-bit 量化（QLoRA）
    from florence_forge.optimization.quantization import ModelQuantizer
    quantizer = ModelQuantizer(method="bnb-4bit")
    model, processor = quantizer.load_quantized_model("microsoft/florence-2-base")

    # 推理时使用 GPTQ 量化
    quantizer = ModelQuantizer(method="gptq-4bit")
    model, processor = quantizer.load_quantized_model("model-gptq")

    # CLI 集成
    florence-forge train --quantization bnb-4bit ...
"""

import logging
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class QuantizationConfig:
    """量化配置

    Args:
        method: 量化方法，支持 "bnb-4bit", "bnb-8bit", "gptq-4bit", "awq-4bit", "dynamic-int8"
        compute_dtype: 计算时使用的数据类型（4-bit 量化时通常用 bf16/fp16）
        double_quant: 是否使用双重量化（仅 bnb-4bit，进一步节省 ~0.4 bits/param）
        quant_type: 量化类型（仅 bnb-4bit，"nf4" 或 "fp4"）
        trust_remote_code: 是否信任远程代码
        device_map: 设备映射策略
    """

    SUPPORTED_METHODS = {"bnb-4bit", "bnb-8bit", "gptq-4bit", "awq-4bit", "dynamic-int8"}

    def __init__(
        self,
        method: str = "bnb-4bit",
        compute_dtype: torch.dtype = torch.bfloat16,
        double_quant: bool = True,
        quant_type: str = "nf4",
        trust_remote_code: bool = True,
        device_map: Optional[str] = None,
    ):
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"不支持的量化方法: {method}。"
                f"支持的方法: {', '.join(sorted(self.SUPPORTED_METHODS))}"
            )
        self.method = method
        self.compute_dtype = compute_dtype
        self.double_quant = double_quant
        self.quant_type = quant_type
        self.trust_remote_code = trust_remote_code
        self.device_map = device_map


class ModelQuantizer:
    """模型量化器

    统一的量化接口，支持多种量化后端。
    """

    def __init__(self, config: Optional[QuantizationConfig] = None):
        self.config = config or QuantizationConfig()
        self._check_dependencies()

    def _check_dependencies(self) -> None:
        """检查量化方法所需的依赖是否已安装"""
        method = self.config.method

        if method in ("bnb-4bit", "bnb-8bit"):
            try:
                import bitsandbytes  # noqa: F401
                logger.info(f"bitsandbytes 可用，支持 {method} 量化")
            except ImportError:
                raise ImportError(
                    f"使用 {method} 量化需要安装 bitsandbytes: pip install bitsandbytes"
                )

        elif method == "gptq-4bit":
            try:
                import auto_gptq  # noqa: F401
                logger.info("auto-gptq 可用，支持 GPTQ 4-bit 量化")
            except ImportError:
                try:
                    from optimum.gptq import GPTQQuantizer  # noqa: F401
                    logger.info("optimum GPTQ 可用")
                except ImportError:
                    raise ImportError(
                        "使用 GPTQ 4-bit 量化需要安装: pip install auto-gptq optimum"
                    )

        elif method == "awq-4bit":
            try:
                import autoawq  # noqa: F401
                logger.info("autoawq 可用，支持 AWQ 4-bit 量化")
            except ImportError:
                try:
                    import awq  # noqa: F401
                    logger.info("awq 可用")
                except ImportError:
                    raise ImportError(
                        "使用 AWQ 4-bit 量化需要安装: pip install autoawq"
                    )

        elif method == "dynamic-int8":
            # PyTorch 内置支持，无需额外依赖
            logger.info("使用 PyTorch 动态量化（无需额外依赖）")

    def load_quantized_model(
        self,
        model_name_or_path: str,
        **kwargs
    ) -> Tuple[nn.Module, Any]:
        """加载已量化的模型或以量化方式加载模型

        Args:
            model_name_or_path: 模型名称或路径
            **kwargs: 额外的加载参数

        Returns:
            (model, processor) 元组
        """
        method = self.config.method

        if method in ("bnb-4bit", "bnb-8bit"):
            return self._load_with_bitsandbytes(model_name_or_path, **kwargs)
        elif method == "gptq-4bit":
            return self._load_gptq_model(model_name_or_path, **kwargs)
        elif method == "awq-4bit":
            return self._load_awq_model(model_name_or_path, **kwargs)
        elif method == "dynamic-int8":
            return self._load_and_dynamic_quantize(model_name_or_path, **kwargs)
        else:
            raise ValueError(f"未实现的量化方法: {method}")

    def _load_with_bitsandbytes(
        self,
        model_name_or_path: str,
        **kwargs
    ) -> Tuple[nn.Module, Any]:
        """使用 bitsandbytes 加载量化模型

        bitsandbytes 支持：
        - 8-bit 量化：load_in_8bit=True
        - 4-bit 量化：load_in_4bit=True（QLoRA 推荐）
        """
        from transformers import AutoModelForCausalLM, AutoProcessor

        method = self.config.method
        device_map = self.config.device_map or ("auto" if torch.cuda.is_available() else "cpu")

        model_kwargs = {
            "trust_remote_code": self.config.trust_remote_code,
            "device_map": device_map,
        }

        if method == "bnb-4bit":
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.config.quant_type,
                bnb_4bit_compute_dtype=self.config.compute_dtype,
                bnb_4bit_use_double_quant=self.config.double_quant,
            )
            model_kwargs["quantization_config"] = bnb_config
            logger.info(
                f"使用 bitsandbytes 4-bit 量化加载模型 "
                f"(quant_type={self.config.quant_type}, "
                f"compute_dtype={self.config.compute_dtype}, "
                f"double_quant={self.config.double_quant})"
            )
        elif method == "bnb-8bit":
            model_kwargs["load_in_8bit"] = True
            logger.info("使用 bitsandbytes 8-bit 量化加载模型")

        model_kwargs.update(kwargs)

        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)

        # 加载处理器
        processor = None
        try:
            processor = AutoProcessor.from_pretrained(
                model_name_or_path,
                trust_remote_code=self.config.trust_remote_code
            )
        except Exception as e:
            logger.warning(f"Processor 加载失败: {e}")

        # 打印量化信息
        self._log_quantization_info(model, method)

        return model, processor

    def _load_gptq_model(
        self,
        model_name_or_path: str,
        **kwargs
    ) -> Tuple[nn.Module, Any]:
        """加载 GPTQ 量化模型"""
        from transformers import AutoModelForCausalLM, AutoProcessor

        device_map = self.config.device_map or ("auto" if torch.cuda.is_available() else "cpu")

        model_kwargs = {
            "trust_remote_code": self.config.trust_remote_code,
            "device_map": device_map,
        }
        model_kwargs.update(kwargs)

        # GPTQ 模型通常已经预量化，直接加载即可
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)

        processor = None
        try:
            processor = AutoProcessor.from_pretrained(
                model_name_or_path,
                trust_remote_code=self.config.trust_remote_code
            )
        except Exception as e:
            logger.warning(f"Processor 加载失败: {e}")

        self._log_quantization_info(model, "gptq-4bit")
        return model, processor

    def _load_awq_model(
        self,
        model_name_or_path: str,
        **kwargs
    ) -> Tuple[nn.Module, Any]:
        """加载 AWQ 量化模型"""
        from transformers import AutoModelForCausalLM, AutoProcessor

        device_map = self.config.device_map or ("auto" if torch.cuda.is_available() else "cpu")

        model_kwargs = {
            "trust_remote_code": self.config.trust_remote_code,
            "device_map": device_map,
        }
        model_kwargs.update(kwargs)

        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)

        processor = None
        try:
            processor = AutoProcessor.from_pretrained(
                model_name_or_path,
                trust_remote_code=self.config.trust_remote_code
            )
        except Exception as e:
            logger.warning(f"Processor 加载失败: {e}")

        self._log_quantization_info(model, "awq-4bit")
        return model, processor

    def _load_and_dynamic_quantize(
        self,
        model_name_or_path: str,
        **kwargs
    ) -> Tuple[nn.Module, Any]:
        """加载模型并应用 PyTorch 动态量化（适用于 CPU 推理）

        动态量化将权重转为 int8，激活值在推理时动态量化。
        适合 CPU 推理加速，对 GPU 没有明显收益。
        """
        from transformers import AutoModelForCausalLM, AutoProcessor

        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=self.config.trust_remote_code,
            **kwargs
        )

        # 应用动态量化
        model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},  # 只量化 Linear 层
            dtype=torch.qint8
        )

        processor = None
        try:
            processor = AutoProcessor.from_pretrained(
                model_name_or_path,
                trust_remote_code=self.config.trust_remote_code
            )
        except Exception as e:
            logger.warning(f"Processor 加载失败: {e}")

        self._log_quantization_info(model, "dynamic-int8")
        return model, processor

    def _log_quantization_info(self, model: nn.Module, method: str) -> None:
        """记录量化后的模型信息"""
        info = self.get_model_size(model)
        logger.info(
            f"量化模型信息 [{method}]: "
            f"参数大小 {info['param_size_mb']:.1f} MB, "
            f"缓冲区大小 {info['buffer_size_mb']:.1f} MB, "
            f"总大小 {info['total_size_mb']:.1f} MB"
        )

    @staticmethod
    def get_model_size(model: nn.Module) -> Dict[str, float]:
        """获取模型大小信息（字节级别精确统计）"""
        param_size = 0
        buffer_size = 0

        for param in model.parameters():
            param_size += param.nelement() * param.element_size()

        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        total_size = param_size + buffer_size

        return {
            'param_size_mb': param_size / 1024 / 1024,
            'buffer_size_mb': buffer_size / 1024 / 1024,
            'total_size_mb': total_size / 1024 / 1024
        }

    @staticmethod
    def get_compression_ratio(original_size_mb: float, quantized_size_mb: float) -> float:
        """计算压缩比

        Args:
            original_size_mb: 原始模型大小（MB）
            quantized_size_mb: 量化后模型大小（MB）

        Returns:
            压缩比（如 4.0 表示量化后是原始的 1/4）
        """
        if quantized_size_mb <= 0:
            return float('inf')
        return original_size_mb / quantized_size_mb

    @staticmethod
    def get_available_methods() -> Dict[str, Dict[str, Any]]:
        """获取当前环境中可用的量化方法

        Returns:
            方法名到详情的映射
        """
        methods = {}

        # bitsandbytes
        try:
            import bitsandbytes  # noqa: F401
            methods["bnb-4bit"] = {
                "available": True,
                "description": "bitsandbytes 4-bit 量化（QLoRA 推荐）",
                "training_supported": True,
            }
            methods["bnb-8bit"] = {
                "available": True,
                "description": "bitsandbytes 8-bit 量化",
                "training_supported": True,
            }
        except ImportError:
            methods["bnb-4bit"] = {"available": False, "description": "需要 bitsandbytes"}
            methods["bnb-8bit"] = {"available": False, "description": "需要 bitsandbytes"}

        # GPTQ
        try:
            import auto_gptq  # noqa: F401
            methods["gptq-4bit"] = {
                "available": True,
                "description": "GPTQ 4-bit 量化（推理专用）",
                "training_supported": False,
            }
        except ImportError:
            try:
                from optimum.gptq import GPTQQuantizer  # noqa: F401
                methods["gptq-4bit"] = {
                    "available": True,
                    "description": "GPTQ 4-bit 量化 via optimum（推理专用）",
                    "training_supported": False,
                }
            except ImportError:
                methods["gptq-4bit"] = {"available": False, "description": "需要 auto-gptq 或 optimum"}

        # AWQ
        try:
            import autoawq  # noqa: F401
            methods["awq-4bit"] = {
                "available": True,
                "description": "AWQ 4-bit 量化（推理专用，低精度损失）",
                "training_supported": False,
            }
        except ImportError:
            try:
                import awq  # noqa: F401
                methods["awq-4bit"] = {
                    "available": True,
                    "description": "AWQ 4-bit 量化（推理专用）",
                    "training_supported": False,
                }
            except ImportError:
                methods["awq-4bit"] = {"available": False, "description": "需要 autoawq"}

        # PyTorch 动态量化
        methods["dynamic-int8"] = {
            "available": True,
            "description": "PyTorch 动态 int8 量化（CPU 推理专用）",
            "training_supported": False,
        }

        return methods
