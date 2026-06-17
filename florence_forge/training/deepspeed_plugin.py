#!/usr/bin/env python3
"""DeepSpeed 分布式训练插件

职责：
- 自动检测 DeepSpeed 可用性（deepspeed >= 0.9.0 + CUDA + NCCL）
- 配置 ZeRO 优化阶段（1/2/3）
- 配置 offload 策略（optimizer/param）
- 与 Accelerate 桥接
"""

import logging
from typing import Optional, Dict, Any, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..core.config import TrainingConfig, DistributedConfig

logger = logging.getLogger(__name__)

__all__ = ["DeepSpeedPlugin"]


class DeepSpeedPlugin:
    """DeepSpeed 插件

    特性：
    - 自动配置 ZeRO 阶段 (1/2/3)
    - 自动配置 offload (optimizer/param)
    - 与 Accelerate 无缝集成
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}
        self._deepspeed_available = self._check_deepspeed_availability()

    def _check_deepspeed_availability(self) -> bool:
        """检测 DeepSpeed 是否可用。"""
        try:
            import deepspeed

            version = tuple(int(x) for x in deepspeed.__version__.split(".")[:2])
            return (
                version >= (0, 9)
                and torch.cuda.is_available()
                and torch.distributed.is_nccl_available()
            )
        except Exception:
            return False

    @property
    def is_available(self) -> bool:
        return self._deepspeed_available

    def get_config(self) -> Dict[str, Any]:
        return dict(self._config)

    def configure_deepspeed(self, config: "TrainingConfig") -> Dict[str, Any]:
        """根据训练配置生成 DeepSpeed 策略字典。"""
        dist_config = config.distributed_settings
        return {
            "zero_stage": dist_config.deepspeed_stage,
            "config_file": dist_config.deepspeed_config_file,
            "offload_optimizer": dist_config.deepspeed_offload_optimizer,
            "offload_param": dist_config.deepspeed_offload_param,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "gradient_clipping": config.max_grad_norm,
        }

    def build_accelerate_plugin(
        self,
        dist_config: "DistributedConfig",
        training_config: Optional["TrainingConfig"] = None,
    ):
        """构建 Accelerate DeepSpeedPlugin 实例。"""
        if not self.is_available:
            logger.warning("DeepSpeed 不可用（需要 deepspeed、CUDA 与 NCCL）")
            return None

        try:
            from accelerate.utils import DeepSpeedPlugin as AccelerateDeepSpeedPlugin

            if dist_config.deepspeed_config_file:
                plugin = AccelerateDeepSpeedPlugin(
                    hf_ds_config=dist_config.deepspeed_config_file,
                    zero_stage=dist_config.deepspeed_stage,
                )
                logger.info("DeepSpeed 插件已配置：%s", dist_config.deepspeed_config_file)
                return plugin

            plugin_kwargs: Dict[str, Any] = {
                "zero_stage": dist_config.deepspeed_stage,
            }
            if training_config is not None:
                plugin_kwargs["gradient_accumulation_steps"] = (
                    training_config.gradient_accumulation_steps
                )
                if training_config.max_grad_norm > 0:
                    plugin_kwargs["gradient_clipping"] = training_config.max_grad_norm

            if dist_config.deepspeed_offload_optimizer:
                plugin_kwargs["offload_optimizer_device"] = "cpu"
            if dist_config.deepspeed_offload_param:
                plugin_kwargs["offload_param_device"] = "cpu"

            plugin = AccelerateDeepSpeedPlugin(**plugin_kwargs)
            logger.info("DeepSpeed 插件已配置：ZeRO stage %s", dist_config.deepspeed_stage)
            return plugin
        except ImportError as exc:
            logger.error("DeepSpeed 不可用：%s", exc)
            return None
        except Exception as exc:
            logger.warning("DeepSpeed 插件构建失败：%s", exc)
            return None
