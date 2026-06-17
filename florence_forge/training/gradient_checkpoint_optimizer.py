"""梯度检查点优化器

提供智能梯度检查点策略选择和应用，支持 4 档激活重计算策略
"""
import logging
from typing import Optional, List
from enum import Enum, auto
import re

import torch.nn as nn

logger = logging.getLogger(__name__)


class ActivationRecomputePolicy(Enum):
    """激活值重计算策略

    off: 不重计算，直接前向传播（默认）
    low: 少量重计算（选择性激活）
    medium: 中等重计算
    high: 全量重计算
    """
    off = auto()
    low = auto()
    medium = auto()
    high = auto()


class GradientCheckpointOptimizer:
    """梯度检查点优化器
    
    自动选择和应用最优的梯度检查点策略，支持 4 档激活重计算：
    - off: 不重计算（默认）
    - low: 选择性重计算
    - medium: 中等重计算
    - high: 全量重计算
    """
    
    def __init__(self, model: nn.Module, config, policy: Optional[ActivationRecomputePolicy] = None):
        """初始化优化器
        
        Args:
            model: 训练模型
            config: 训练配置
            policy: 激活重计算策略（默认 None，从 config 读取）
        """
        self.model = model
        self.config = config
        self.policy = policy or ActivationRecomputePolicy.off
    
    def enable_gradient_checkpointing(self) -> None:
        """启用梯度检查点

        策略由 ``ActivationRecomputePolicy``（4 档）或
        ``model_settings.activation_checkpointing_strategy`` 决定。
        """
        strategy = self._resolve_strategy()
        if strategy in (None, "none"):
            logger.info("激活值重计算已禁用")
            return

        logger.info("启用梯度检查点：策略=%s（policy=%s）", strategy, self.policy.name)

        if strategy == "full":
            self._apply_full_checkpointing()
        elif strategy == "selective":
            self._apply_selective_checkpointing()
        else:
            logger.warning("未知的梯度检查点策略：%s", strategy)

    def _resolve_strategy(self) -> Optional[str]:
        """将 4 档 policy 映射为 full / selective / none。"""
        policy_map = {
            ActivationRecomputePolicy.off: "none",
            ActivationRecomputePolicy.low: "selective",
            ActivationRecomputePolicy.medium: "selective",
            ActivationRecomputePolicy.high: "full",
        }
        if self.policy != ActivationRecomputePolicy.off:
            return policy_map[self.policy]

        model_settings = getattr(self.config, "model_settings", None)
        if model_settings is not None:
            strategy = getattr(model_settings, "activation_checkpointing_strategy", "none")
            if strategy == "none" and getattr(model_settings, "gradient_checkpointing", False):
                return "full"
            if strategy != "none":
                if strategy == "auto":
                    return self._auto_select_strategy()
                return strategy

        if getattr(self.config, "gradient_checkpointing", False):
            return self._auto_select_strategy()
        return "none"
    
    def _auto_select_strategy(self) -> str:
        """自动选择梯度检查点策略
        
        Returns:
            策略名称 ('full' 或 'selective')
        """
        if hasattr(self.config, 'checkpoint_strategy'):
            if self.config.checkpoint_strategy != 'auto':
                return self.config.checkpoint_strategy
        
        # 自动选择逻辑
        # 1. 如果模型有原生 gradient_checkpointing_enable 方法，使用 full
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            logger.info("✅ 检测到原生梯度检查点支持，使用 full 策略")
            return "full"
        
        # 2. 否则使用 selective（适用于自定义架构）
        logger.info("🔍 未检测到原生支持，使用 selective 策略")
        return "selective"
    
    def _apply_full_checkpointing(self) -> None:
        """应用全量梯度检查点"""
        try:
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                logger.info("✅ 全量梯度检查点已启用")
            else:
                logger.warning("⚠️  模型不支持 gradient_checkpointing_enable")
        except Exception as e:
            logger.error(f"❌ 启用梯度检查点失败：{e}")
    
    def _apply_selective_checkpointing(self) -> None:
        """应用选择性梯度检查点

        根据配置的层名模式或间隔选择部分层启用检查点。
        low 策略使用更大间隔以节省计算。
        """
        model_settings = getattr(self.config, "model_settings", None)
        checkpoint_layers = getattr(self.config, "checkpoint_layers", None)
        checkpoint_interval = getattr(self.config, "checkpoint_interval", None)

        if model_settings is not None:
            checkpoint_layers = checkpoint_layers or getattr(
                model_settings, "checkpoint_target_layers", None
            )
            checkpoint_interval = checkpoint_interval or getattr(
                model_settings, "checkpoint_every_n_layers", None
            )

        if checkpoint_interval is None:
            checkpoint_interval = 3 if self.policy == ActivationRecomputePolicy.low else 1
        
        if checkpoint_layers:
            # 按层名模式选择
            self._checkpoint_by_pattern(checkpoint_layers)
        else:
            # 按间隔选择
            self._checkpoint_by_interval(checkpoint_interval)
    
    def _checkpoint_by_pattern(self, layer_patterns: List[str]) -> None:
        """按层名模式启用检查点
        
        Args:
            layer_patterns: 层名模式列表（支持正则表达式）
        """
        enabled_count = 0
        
        for name, module in self.model.named_modules():
            for pattern in layer_patterns:
                if re.search(pattern, name):
                    if hasattr(module, 'gradient_checkpointing'):
                        module.gradient_checkpointing = True
                        enabled_count += 1
                    break
        
        logger.info(f"✅ 选择性梯度检查点已启用（{enabled_count} 个模块）")
    
    def _checkpoint_by_interval(self, interval: int) -> None:
        """按间隔启用检查点
        
        Args:
            interval: 检查点间隔（每 N 层启用一次）
        """
        enabled_count = 0
        
        # 提取编码器/解码器层
        encoder_layers = self._get_encoder_layers()
        
        for idx, (name, module) in enumerate(encoder_layers):
            if idx % interval == 0:
                if hasattr(module, 'gradient_checkpointing'):
                    module.gradient_checkpointing = True
                    enabled_count += 1
        
        logger.info(f"✅ 选择性梯度检查点已启用（间隔={interval}，{enabled_count} 个层）")
    
    def _get_encoder_layers(self) -> List:
        """获取编码器/解码器层列表
        
        Returns:
            (name, module) 元组列表
        """
        layers = []
        
        # 常见的层命名模式
        patterns = [
            r'encoder\.layer\.\d+',
            r'decoder\.layer\.\d+',
            r'layers\.\d+',
            r'blocks\.\d+'
        ]
        
        for name, module in self.model.named_modules():
            for pattern in patterns:
                if re.match(pattern, name):
                    layers.append((name, module))
                    break
        
        return layers
    
    def disable_kv_cache_for_training(self) -> None:
        """禁用 KV 缓存（训练时不需要）"""
        if hasattr(self.model, 'config'):
            model_config = self.model.config
            if hasattr(model_config, 'use_cache'):
                model_config.use_cache = False
                logger.info("✅ 已禁用 KV 缓存（训练模式）")
