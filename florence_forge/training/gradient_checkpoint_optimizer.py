"""梯度检查点优化器

提供智能梯度检查点策略选择和应用
"""
import logging
from typing import Optional, List
import re

import torch.nn as nn

logger = logging.getLogger(__name__)


class GradientCheckpointOptimizer:
    """梯度检查点优化器
    
    自动选择和应用最优的梯度检查点策略
    """
    
    def __init__(self, model: nn.Module, config):
        """初始化优化器
        
        Args:
            model: 训练模型
            config: 训练配置
        """
        self.model = model
        self.config = config
    
    def enable_gradient_checkpointing(self) -> None:
        """启用梯度检查点
        
        自动选择策略：full（全量） 或 selective（选择性）
        """
        if not self.config.gradient_checkpointing:
            return
        
        # 自动选择策略
        strategy = self._auto_select_strategy()
        
        logger.info(f"🔄 启用梯度检查点：策略={strategy}")
        
        if strategy == "full":
            self._apply_full_checkpointing()
        elif strategy == "selective":
            self._apply_selective_checkpointing()
        else:
            logger.warning(f"未知的梯度检查点策略：{strategy}")
    
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
        
        根据配置的层名模式或间隔选择部分层启用检查点
        """
        checkpoint_layers = getattr(self.config, 'checkpoint_layers', None)
        checkpoint_interval = getattr(self.config, 'checkpoint_interval', 1)
        
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
