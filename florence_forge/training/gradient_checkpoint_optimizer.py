"""梯度检查点优化器（v2 训练栈入口，委托共享实现）。"""

from __future__ import annotations

import logging
from typing import Union

import torch.nn as nn

from .activation_checkpointing import ActivationCheckpointingApplier

logger = logging.getLogger(__name__)


class GradientCheckpointOptimizer:
    """v2 训练栈的梯度检查点门面，与 v1 ``MultiTaskTrainer`` 语义对齐。"""

    def __init__(self, model: nn.Module, config) -> None:
        self.model = model
        self.config = config
        self._applier = ActivationCheckpointingApplier.from_training_config(model, config)

    def enable_gradient_checkpointing(self) -> None:
        """启用与 v1 一致的 full / selective / auto 多档策略。"""
        self._applier.apply()

    def disable_kv_cache_for_training(self) -> None:
        """训练时禁用 KV cache（与 checkpointing 兼容）。"""
        self._applier._disable_kv_cache_for_training()
