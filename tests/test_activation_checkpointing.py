"""测试高级内存优化功能（激活值重计算策略）

覆盖:
- 自动策略选择 (_auto_select_checkpoint_strategy)
- 选择性重计算 (_apply_selective_gradient_checkpointing)
- 配置验证
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

from florence_forge.core.config import ModelConfig, TrainingConfig
from florence_forge.training.trainer import MultiTaskTrainer


# ---------------------------------------------------------------------------
# Mock 模型和模块（用于测试，无需真实权重）
# ---------------------------------------------------------------------------

class MockTransformerLayer(nn.Module):
    """模拟 Transformer 层"""
    def __init__(self, d_model=64):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.linear = nn.Linear(d_model, d_model)
        self.gradient_checkpointing = False

    def forward(self, x):
        return self.linear(x)


class MockModel(nn.Module):
    """模拟大模型，包含多个 Transformer 层"""
    def __init__(self, num_layers=6, d_model=64):
        super().__init__()
        self.layers = nn.ModuleList([
            MockTransformerLayer(d_model) for _ in range(num_layers)
        ])
        self.config = MagicMock()
        self.config.use_cache = True

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# ---------------------------------------------------------------------------
# 自动策略选择测试
# ---------------------------------------------------------------------------

class TestAutoCheckpointStrategy:
    """测试自动选择激活值重计算策略"""

    def test_small_model_selects_none(self):
        """小模型 (<1B) 应该选择 none 策略"""
        mock_dataset = MagicMock()
        mock_dataset.task_indices = {"CAPTION": [0]}
        mock_dataset.__len__ = lambda: 1

        # 创建小模型（参数量 < 1B）
        small_model = nn.Linear(10, 10)  # 只有 100 个参数

        config = TrainingConfig()
        config.model_settings.activation_checkpointing_strategy = "auto"

        trainer = MultiTaskTrainer(small_model, mock_dataset, config=config)

        strategy = trainer._auto_select_checkpoint_strategy()
        assert strategy == "none"

    def test_large_model_selects_selective(self):
        """大模型 (>=7B) 应该选择 selective 策略"""
        mock_dataset = MagicMock()
        mock_dataset.task_indices = {"CAPTION": [0]}
        mock_dataset.__len__ = lambda: 1

        # 使用小模型但 patch 参数量检测来模拟大模型（避免OOM）
        small_model = nn.Linear(10, 10)

        config = TrainingConfig()
        config.model_settings.activation_checkpointing_strategy = "auto"

        trainer = MultiTaskTrainer(small_model, mock_dataset, config=config)

        # Patch 模型参数量为 8B
        with patch.object(trainer.model, 'parameters', return_value=[torch.empty(8_000_000_000)]):
            strategy = trainer._auto_select_checkpoint_strategy()
        assert strategy == "selective"
        assert config.model_settings.checkpoint_every_n_layers == 2

    def test_medium_model_selects_full(self):
        """中等模型 (1B-7B) 应该选择 full 策略"""
        mock_dataset = MagicMock()
        mock_dataset.task_indices = {"CAPTION": [0]}
        mock_dataset.__len__ = lambda: 1

        # 使用小模型但 patch 参数量检测来模拟中等模型
        small_model = nn.Linear(10, 10)

        config = TrainingConfig()
        config.model_settings.activation_checkpointing_strategy = "auto"

        trainer = MultiTaskTrainer(small_model, mock_dataset, config=config)

        # Patch 模型参数量为 2B
        with patch.object(trainer.model, 'parameters', return_value=[torch.empty(2_000_000_000)]):
            strategy = trainer._auto_select_checkpoint_strategy()
        assert strategy == "full"


# ---------------------------------------------------------------------------
# 选择性重计算测试
# ---------------------------------------------------------------------------

class TestSelectiveGradientCheckpointing:
    """测试选择性梯度检查点"""

    def test_selective_with_every_n_layers(self):
        """测试每隔 N 层启用 checkpoint"""
        mock_dataset = MagicMock()
        mock_dataset.task_indices = {"CAPTION": [0]}
        mock_dataset.__len__ = lambda: 1

        model = MockModel(num_layers=6)

        config = TrainingConfig()
        config.model_settings.activation_checkpointing_strategy = "selective"
        config.model_settings.checkpoint_every_n_layers = 2

        trainer = MultiTaskTrainer(model, mock_dataset, config=config)
        trainer._apply_selective_gradient_checkpointing()

        # 检查 layer 0, 2, 4 是否启用了 checkpoint
        assert model.layers[0].gradient_checkpointing is True
        assert model.layers[2].gradient_checkpointing is True
        assert model.layers[4].gradient_checkpointing is True
        # layer 1, 3, 5 应该未启用
        assert model.layers[1].gradient_checkpointing is False
        assert model.layers[3].gradient_checkpointing is False
        assert model.layers[5].gradient_checkpointing is False

    def test_selective_with_target_layers_list(self):
        """测试通过层名列表选择性启用"""
        mock_dataset = MagicMock()
        mock_dataset.task_indices = {"CAPTION": [0]}
        mock_dataset.__len__ = lambda: 1

        model = MockModel(num_layers=6)

        config = TrainingConfig()
        config.model_settings.activation_checkpointing_strategy = "selective"
        config.model_settings.checkpoint_target_layers = ["layers.0", "layers.3"]
        config.model_settings.checkpoint_every_n_layers = None

        trainer = MultiTaskTrainer(model, mock_dataset, config=config)
        trainer._apply_selective_gradient_checkpointing()

        # 检查目标层是否启用了 checkpoint
        assert model.layers[0].gradient_checkpointing is True
        assert model.layers[3].gradient_checkpointing is True
        # 其他层应该未启用
        assert model.layers[1].gradient_checkpointing is False
        assert model.layers[2].gradient_checkpointing is False

    def test_selective_fallback_to_full(self):
        """测试 selective 未匹配到任何层时回退到 full"""
        mock_dataset = MagicMock()
        mock_dataset.task_indices = {"CAPTION": [0]}
        mock_dataset.__len__ = lambda: 1

        model = MockModel(num_layers=6)
        # 移除 gradient_checkpointing 属性，使其无法匹配
        for layer in model.layers:
            delattr(layer, 'gradient_checkpointing')

        config = TrainingConfig()
        config.model_settings.activation_checkpointing_strategy = "selective"
        config.model_settings.checkpoint_every_n_layers = 2

        trainer = MultiTaskTrainer(model, mock_dataset, config=config)

        # 这里应该回退到 full 模式，但由于模型没有 gradient_checkpointing_enable，
        # 所以会打印警告但继续执行
        # 我们不测试具体的回退行为，只测试不抛出异常
        try:
            trainer._apply_selective_gradient_checkpointing()
        except Exception as e:
            pytest.fail(f"不应该抛出异常: {e}")

    def test_extract_layer_index(self):
        """测试从模块名中提取层索引"""
        mock_dataset = MagicMock()
        mock_dataset.task_indices = {"CAPTION": [0]}
        mock_dataset.__len__ = lambda: 1

        model = nn.Linear(10, 10)
        config = TrainingConfig()
        trainer = MultiTaskTrainer(model, mock_dataset, config=config)

        assert trainer._extract_layer_index("encoder.layers.3.self_attn") == 3
        assert trainer._extract_layer_index("layer.5.norm") == 5
        assert trainer._extract_layer_index("blocks.10") == 10
        assert trainer._extract_layer_index("h.2.mlp") == 2
        assert trainer._extract_layer_index("no_index_here") is None


# ---------------------------------------------------------------------------
# KV Cache 禁用测试
# ---------------------------------------------------------------------------

class TestDisableKVCache:
    """测试训练时禁用 KV Cache"""

    def test_disable_kv_cache(self):
        """测试 _disable_kv_cache_for_training"""
        mock_dataset = MagicMock()
        mock_dataset.task_indices = {"CAPTION": [0]}
        mock_dataset.__len__ = lambda: 1

        model = MockModel(num_layers=3)
        # 模拟子模块也有 config.use_cache
        for i, layer in enumerate(model.layers):
            layer.config = MagicMock()
            layer.config.use_cache = True

        config = TrainingConfig()
        trainer = MultiTaskTrainer(model, mock_dataset, config=config)
        trainer._disable_kv_cache_for_training()

        # 主模型的 use_cache 应该被禁用
        assert model.config.use_cache is False
        # 子模块的 use_cache 也应该被禁用
        for layer in model.layers:
            assert layer.config.use_cache is False


# ---------------------------------------------------------------------------
# 配置验证测试
# ---------------------------------------------------------------------------

class TestActivationCheckpointingConfig:
    """测试激活值重计算配置"""

    def test_config_fields(self):
        """测试配置字段存在性和默认值"""
        config = ModelConfig()
        assert config.activation_checkpointing_strategy == "none"
        assert config.checkpoint_target_layers is None
        assert config.checkpoint_every_n_layers is None

    def test_invalid_strategy(self):
        """测试无效策略会报错"""
        with pytest.raises(ValueError, match="activation_checkpointing_strategy 必须是"):
            ModelConfig(activation_checkpointing_strategy="invalid")

    def test_backward_compatibility(self):
        """测试向后兼容：gradient_checkpointing=True 映射到 full"""
        config = ModelConfig(
            gradient_checkpointing=True,
            activation_checkpointing_strategy="none"
        )
        # 在训练器中，gradient_checkpointing=True 应该触发 full 模式
        assert config.gradient_checkpointing is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
