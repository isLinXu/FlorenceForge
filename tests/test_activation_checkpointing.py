"""激活值重计算共享模块测试（v1/v2 语义对齐）。"""

import torch
import torch.nn as nn

from florence_forge.core.config import ModelConfig, TrainingConfig
from florence_forge.training.activation_checkpointing import (
    ActivationCheckpointingApplier,
    extract_layer_index,
)


class _StubHFModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = type("Cfg", (), {"use_cache": True})()
        self._checkpointing = False

    def gradient_checkpointing_enable(self) -> None:
        self._checkpointing = True

    def parameters(self, recurse: bool = True):
        return iter([torch.nn.Parameter(torch.zeros(2_000_000_000))])


class _CheckpointBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.gradient_checkpointing = False


class _LayeredModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(_CheckpointBlock() for _ in range(6))


def test_extract_layer_index_patterns():
    assert extract_layer_index("encoder.layers.3.self_attn") == 3
    assert extract_layer_index("blocks.12.mlp") == 12
    assert extract_layer_index("head") is None


def test_should_apply_respects_strategy_and_top_level_flag():
    model = _StubHFModel()
    cfg = TrainingConfig(
        model_settings=ModelConfig(
            gradient_checkpointing=False,
            activation_checkpointing_strategy="selective",
        )
    )
    assert ActivationCheckpointingApplier.from_training_config(model, cfg).should_apply()

    cfg2 = TrainingConfig(gradient_checkpointing=True)
    assert ActivationCheckpointingApplier.from_training_config(model, cfg2).should_apply()

    cfg3 = TrainingConfig(
        model_settings=ModelConfig(
            gradient_checkpointing=False,
            activation_checkpointing_strategy="none",
        )
    )
    assert not ActivationCheckpointingApplier.from_training_config(model, cfg3).should_apply()


def test_full_mode_enables_hf_hook_and_disables_cache():
    model = _StubHFModel()
    model_config = ModelConfig(
        gradient_checkpointing=True,
        activation_checkpointing_strategy="full",
    )
    ActivationCheckpointingApplier(model, model_config).apply()
    assert model._checkpointing is True
    assert model.config.use_cache is False


def test_selective_every_n_layers():
    model = _LayeredModel()
    model_config = ModelConfig(
        activation_checkpointing_strategy="selective",
        checkpoint_every_n_layers=2,
    )
    ActivationCheckpointingApplier(model, model_config).apply()
    enabled = [
        name
        for name, module in model.named_modules()
        if getattr(module, "gradient_checkpointing", False)
    ]
    assert any(name.startswith("layers.0") for name in enabled)
    assert any(name.startswith("layers.2") for name in enabled)
    assert not any(name.startswith("layers.1") for name in enabled)
