"""Tests for MoECallback and multitask_trainer MoE diagnostics integration."""

from unittest.mock import MagicMock

import torch

from florence_forge.core.callbacks import (
    CallbackManager,
    MoECallback,
    create_default_callbacks,
)
from florence_forge.training.moe.moe_adapter import MoETrainingAdapter
from florence_forge.training.moe.moe_config import MoEConfig
from florence_forge.training.moe.moe_layer import MoELayer


class _FakeConfig:
    """Minimal config stand-in for create_default_callbacks."""

    def __init__(self, **kwargs):
        self.logging_steps = 10
        self.save_steps = 1000
        self.save_total_limit = 3
        self.early_stopping_patience = 0
        self.metric_for_best_model = "eval_loss"
        self.early_stopping_threshold = 0.0
        self.greater_is_better = False
        self.use_tensorboard = False
        self.use_moe = kwargs.get("use_moe", False)


def test_moe_callback_is_added_when_use_moe_is_true():
    """create_default_callbacks 应在 use_moe=True 时包含 MoECallback。"""
    config = _FakeConfig(use_moe=True)
    callbacks = create_default_callbacks(config)
    assert any(isinstance(cb, MoECallback) for cb in callbacks)


def test_moe_callback_is_not_added_when_use_moe_is_false():
    """create_default_callbacks 不应在 use_moe=False 时添加 MoECallback。"""
    config = _FakeConfig(use_moe=False)
    callbacks = create_default_callbacks(config)
    assert not any(isinstance(cb, MoECallback) for cb in callbacks)


def test_moe_callback_on_epoch_end_with_moe_adapter():
    """MoECallback.on_epoch_end 应正确收集并注入 MoE 指标到 logs。"""
    # 构造一个真实的 MoE layer 和 adapter
    moe_layer = MoELayer(num_experts=4, d_model=8, d_state=8, top_k=2, capacity_factor=None)
    x = torch.randn(2, 3, 8)
    _ = moe_layer(x)  # 触发前向，填充 last_gate_weights / _routing_sums

    config = MoEConfig(num_experts=4, d_model=8, d_state=8, top_k=2)
    adapter = MoETrainingAdapter(config)
    adapter._moe_layers = [moe_layer]

    # mock trainer with training_loop._moe_adapter
    trainer = MagicMock()
    trainer.training_loop._moe_adapter = adapter

    callback = MoECallback(log_frequency=1)
    logs = {}
    callback.on_epoch_end(trainer, epoch=0, logs=logs)

    # 验证 logs 被注入了 MoE 指标
    assert "moe_gini" in logs
    assert "moe_overflow_tokens" in logs
    assert "moe_num_layers" in logs
    assert "moe_num_experts" in logs
    assert logs["moe_num_layers"] == 1
    assert logs["moe_num_experts"] == 4


def test_moe_callback_on_epoch_end_without_moe_adapter():
    """没有 MoE adapter 时，on_epoch_end 不应抛出异常且不修改 logs。"""
    class FakeTrainingLoop:
        _moe_adapter = None

    class FakeTrainer:
        training_loop = FakeTrainingLoop()

    trainer = FakeTrainer()
    callback = MoECallback()
    logs = {"loss": 1.23}
    callback.on_epoch_end(trainer, epoch=0, logs=logs)
    # logs 应保持原样
    assert logs == {"loss": 1.23}


def test_moe_callback_on_epoch_end_with_uninjected_adapter():
    """MoE adapter 存在但未注入层时，不应修改 logs。"""
    config = MoEConfig(num_experts=4, d_model=8, d_state=8, top_k=2)
    adapter = MoETrainingAdapter(config)
    # adapter 存在但 _moe_layers 为空（is_injected() == False）
    assert not adapter.is_injected()

    class FakeTrainingLoop:
        _moe_adapter = adapter

    class FakeTrainer:
        training_loop = FakeTrainingLoop()

    trainer = FakeTrainer()
    callback = MoECallback()
    logs = {"loss": 1.23}
    callback.on_epoch_end(trainer, epoch=0, logs=logs)
    assert logs == {"loss": 1.23}


def test_moe_callback_on_train_end_with_moe_adapter():
    """MoECallback.on_train_end 应正确记录最终 MoE 统计。"""
    moe_layer = MoELayer(num_experts=4, d_model=8, d_state=8, top_k=2, capacity_factor=None)
    x = torch.randn(2, 3, 8)
    _ = moe_layer(x)

    config = MoEConfig(num_experts=4, d_model=8, d_state=8, top_k=2)
    adapter = MoETrainingAdapter(config)
    adapter._moe_layers = [moe_layer]

    trainer = MagicMock()
    trainer.training_loop._moe_adapter = adapter

    callback = MoECallback()
    # 不应抛出异常
    callback.on_train_end(trainer, config=None)


def test_moe_callback_on_train_end_without_adapter():
    """没有 MoE adapter 时，on_train_end 不应抛出异常。"""
    class FakeTrainingLoop:
        _moe_adapter = None

    class FakeTrainer:
        training_loop = FakeTrainingLoop()

    trainer = FakeTrainer()
    callback = MoECallback()
    callback.on_train_end(trainer, config=None)


def test_moe_callback_through_callback_manager():
    """MoECallback 应能通过 CallbackManager 正确触发。"""
    moe_layer = MoELayer(num_experts=4, d_model=8, d_state=8, top_k=2, capacity_factor=None)
    x = torch.randn(2, 3, 8)
    _ = moe_layer(x)

    config = MoEConfig(num_experts=4, d_model=8, d_state=8, top_k=2)
    adapter = MoETrainingAdapter(config)
    adapter._moe_layers = [moe_layer]

    trainer = MagicMock()
    trainer.training_loop._moe_adapter = adapter

    callback = MoECallback()
    manager = CallbackManager([callback])

    logs = {}
    manager.on_epoch_end(trainer, epoch=0, logs=logs)

    assert "moe_gini" in logs
    assert "moe_overflow_tokens" in logs


def test_moe_callback_fallback_to_trainer_attribute():
    """当 training_loop 没有 _moe_adapter 时，应回退到 trainer.moe_adapter。"""
    moe_layer = MoELayer(num_experts=4, d_model=8, d_state=8, top_k=2, capacity_factor=None)
    x = torch.randn(2, 3, 8)
    _ = moe_layer(x)

    config = MoEConfig(num_experts=4, d_model=8, d_state=8, top_k=2)
    adapter = MoETrainingAdapter(config)
    adapter._moe_layers = [moe_layer]

    class FakeTrainer:
        training_loop = None
        moe_adapter = adapter

    trainer = FakeTrainer()
    callback = MoECallback()
    logs = {}
    callback.on_epoch_end(trainer, epoch=0, logs=logs)

    assert "moe_gini" in logs
    assert "moe_overflow_tokens" in logs
