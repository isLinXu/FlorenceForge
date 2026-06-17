"""激活值重计算策略测试（GradientCheckpointOptimizer）。"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock

from florence_forge.core.config import ModelConfig, TrainingConfig
from florence_forge.training.gradient_checkpoint_optimizer import (
    ActivationRecomputePolicy,
    GradientCheckpointOptimizer,
)


class MockTransformerLayer(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)
        self.gradient_checkpointing = False

    def forward(self, x):
        return self.linear(x)


class MockModel(nn.Module):
    def __init__(self, num_layers=6, d_model=64):
        super().__init__()
        self.layers = nn.ModuleList(
            [MockTransformerLayer(d_model) for _ in range(num_layers)]
        )
        self.config = MagicMock()
        self.config.use_cache = True

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TestGradientCheckpointOptimizer:
    def test_policy_off_disables_checkpointing(self):
        config = TrainingConfig()
        config.model_settings.activation_checkpointing_strategy = "none"
        optimizer = GradientCheckpointOptimizer(
            nn.Linear(4, 4), config, policy=ActivationRecomputePolicy.off
        )
        assert optimizer._resolve_strategy() == "none"

    def test_policy_high_maps_to_full(self):
        config = TrainingConfig()
        model = MagicMock()
        model.gradient_checkpointing_enable = MagicMock()
        optimizer = GradientCheckpointOptimizer(
            model, config, policy=ActivationRecomputePolicy.high
        )
        assert optimizer._resolve_strategy() == "full"

    def test_selective_checkpoint_by_interval(self):
        model = MockModel(num_layers=6)
        config = TrainingConfig()
        config.model_settings.activation_checkpointing_strategy = "selective"
        config.model_settings.checkpoint_every_n_layers = 2

        optimizer = GradientCheckpointOptimizer(
            model, config, policy=ActivationRecomputePolicy.low
        )
        optimizer._apply_selective_checkpointing()

        assert model.layers[0].gradient_checkpointing is True
        assert model.layers[2].gradient_checkpointing is True
        assert model.layers[4].gradient_checkpointing is True

    def test_disable_kv_cache(self):
        model = MockModel(num_layers=2)
        config = TrainingConfig()
        optimizer = GradientCheckpointOptimizer(model, config)
        optimizer.disable_kv_cache_for_training()
        assert model.config.use_cache is False


class TestActivationCheckpointingConfig:
    def test_config_defaults(self):
        config = ModelConfig()
        assert config.activation_checkpointing_strategy == "none"

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="activation_checkpointing_strategy"):
            ModelConfig(activation_checkpointing_strategy="invalid")
