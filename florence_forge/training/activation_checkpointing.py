"""激活值重计算 / Gradient Checkpointing（v2 训练栈共享实现）。

将历史 v1 训练器中的多档策略逻辑集中到本模块，供 v2 ``MultiTaskTrainer``、
v2 ``GradientCheckpointOptimizer`` 及未来拆分后的训练组件复用，避免双栈语义漂移。
"""

from __future__ import annotations

import logging
import re
from typing import Optional, Union, List, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from ..core.config import ModelConfig, TrainingConfig

logger = logging.getLogger(__name__)


class ActivationCheckpointingApplier:
    """按 ``ModelConfig`` 启用 full / selective / auto 激活值重计算。"""

    def __init__(
        self,
        model: nn.Module,
        model_config: "ModelConfig",
        *,
        device: Union[str, int] = "auto",
        force_enable: bool = False,
    ) -> None:
        self.model = model
        self.model_config = model_config
        self.device = device
        self._force_enable = force_enable

    @classmethod
    def from_training_config(cls, model: nn.Module, config: "TrainingConfig") -> "ActivationCheckpointingApplier":
        """从完整训练配置构造（合并顶层与 model_settings 的开关）。"""
        return cls(
            model,
            config.model_settings,
            device=getattr(config, "device", "auto"),
            force_enable=getattr(config, "gradient_checkpointing", False),
        )

    def should_apply(self) -> bool:
        """是否应尝试启用激活值重计算。"""
        if self._force_enable or self.model_config.gradient_checkpointing:
            return True
        strategy = self.model_config.activation_checkpointing_strategy
        return strategy not in (None, "", "none")

    def apply(self) -> None:
        """启用 Gradient Checkpointing；失败时记录警告并继续训练。"""
        if not self.should_apply():
            return

        model_config = self.model_config
        strategy = model_config.activation_checkpointing_strategy

        if strategy == "none" and (
            model_config.gradient_checkpointing or self._force_enable
        ):
            strategy = "full"
            logger.info(
                "gradient_checkpointing=True 已映射到 activation_checkpointing_strategy='full'"
            )

        if strategy == "none":
            logger.info("激活值重计算已禁用 (strategy='none')")
            return

        try:
            if strategy == "auto":
                strategy = self._auto_select_checkpoint_strategy()
                logger.info("🔄 自动选择重计算策略: %s", strategy)

            if strategy == "full":
                self._apply_full_gradient_checkpointing()
            elif strategy == "selective":
                self._apply_selective_gradient_checkpointing()
            else:
                logger.warning("未知的重计算策略: %s，跳过", strategy)
                return

            self._disable_kv_cache_for_training()

        except Exception as exc:
            logger.warning("⚠️ Gradient Checkpointing 启用失败: %s", exc)
            logger.warning("   将继续训练，但显存占用可能较高")

    def _auto_select_checkpoint_strategy(self) -> str:
        """基于参数量与可用显存选择 full / selective / none。"""
        total_params = 0
        try:
            total_params = sum(p.numel() for p in self.model.parameters())
        except Exception:
            pass

        params_in_billions = total_params / 1e9

        available_vram_gb = float("inf")
        if torch.cuda.is_available():
            try:
                device_ref = self.device if self.device not in (None, "auto") else 0
                if isinstance(device_ref, str) and device_ref.startswith("cuda:"):
                    device_ref = int(device_ref.split(":")[-1])
                device_props = torch.cuda.get_device_properties(device_ref)
                total_vram = device_props.total_memory / (1024**3)
                allocated_vram = torch.cuda.memory_allocated(device_ref) / (1024**3)
                available_vram_gb = total_vram - allocated_vram
            except Exception:
                pass

        logger.info(
            "自动策略检测: 模型参数量=%.2fB, 可用显存=%.1fGB",
            params_in_billions,
            available_vram_gb,
        )

        if params_in_billions < 1.0:
            logger.info("模型较小 (<1B)，无需激活值重计算")
            return "none"
        if params_in_billions >= 7.0 or available_vram_gb < 10.0:
            logger.info("大模型或显存紧张，使用 selective 策略")
            if self.model_config.checkpoint_every_n_layers is None:
                self.model_config.checkpoint_every_n_layers = 2
            return "selective"
        logger.info("中等模型且显存充足，使用 full 策略")
        return "full"

    def _apply_full_gradient_checkpointing(self) -> None:
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
            logger.info("✅ Gradient Checkpointing 已启用（HF 原生接口，full 模式）")
        elif hasattr(self.model, "model") and hasattr(
            self.model.model, "gradient_checkpointing_enable"
        ):
            self.model.model.gradient_checkpointing_enable()
            logger.info("✅ Gradient Checkpointing 已启用（通过 backend 代理，full 模式）")
        else:
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()
            logger.info("✅ Gradient Checkpointing 已启用（PyTorch 原生模式，full 模式）")

    def _apply_selective_gradient_checkpointing(self) -> None:
        target_layers = self.model_config.checkpoint_target_layers
        every_n = self.model_config.checkpoint_every_n_layers

        checkpointed_count = 0
        matched_modules: List[str] = []

        for name, module in self.model.named_modules():
            should_checkpoint = False

            if target_layers is not None:
                if isinstance(target_layers, list):
                    should_checkpoint = any(
                        name == pattern or name.endswith(pattern)
                        for pattern in target_layers
                    )
                elif isinstance(target_layers, str):
                    pattern = target_layers.replace("*", "")
                    should_checkpoint = name.startswith(pattern) or pattern in name
            elif every_n is not None:
                layer_idx = extract_layer_index(name)
                if layer_idx is not None and layer_idx % every_n == 0:
                    should_checkpoint = True

            if should_checkpoint:
                if hasattr(module, "gradient_checkpointing"):
                    module.gradient_checkpointing = True
                    checkpointed_count += 1
                    matched_modules.append(name)
                elif isinstance(module, torch.nn.TransformerEncoderLayer) or isinstance(
                    module, torch.nn.TransformerDecoderLayer
                ):
                    module.checkpoint = True
                    checkpointed_count += 1
                    matched_modules.append(name)

        if checkpointed_count > 0:
            logger.info(
                "✅ 选择性 Gradient Checkpointing 已启用: %d 个模块",
                checkpointed_count,
            )
            if len(matched_modules) <= 10:
                for mod_name in matched_modules:
                    logger.info("   - %s", mod_name)
            else:
                logger.info("   前 10 个: %s...", ", ".join(matched_modules[:10]))
        else:
            logger.warning("未匹配到任何可 checkpoint 的模块，回退到 full 模式")
            self._apply_full_gradient_checkpointing()

    def _disable_kv_cache_for_training(self) -> None:
        if hasattr(self.model, "config") and hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False
            logger.info(
                "   已将 model.config.use_cache 设为 False（与 checkpointing 兼容）"
            )
        for _, module in self.model.named_modules():
            if hasattr(module, "config") and hasattr(module.config, "use_cache"):
                module.config.use_cache = False


def extract_layer_index(module_name: str) -> Optional[int]:
    """从模块名称中提取层索引（如 encoder.layers.3 -> 3）。"""
    patterns = [
        r"layers\.(\d+)",
        r"layer\.(\d+)",
        r"blocks\.(\d+)",
        r"block\.(\d+)",
        r"h\.(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, module_name)
        if match:
            return int(match.group(1))
    return None
