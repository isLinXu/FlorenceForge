"""v2 训练栈分布式插件与多 GPU 冒烟测试。"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

from florence_forge.core.config import DistributedConfig, TrainingConfig
from florence_forge.training.device_config import DeviceConfigurator


def _has_accelerate() -> bool:
    try:
        import accelerate  # noqa: F401

        return True
    except ImportError:
        return False


class TestMpsMixedPrecision:
    def test_mps_uses_fp32_when_only_bf16_enabled(self):
        config = TrainingConfig(device="mps", use_bf16=True, use_fp16=False)
        dc = DeviceConfigurator(config)
        dc.device_type = "mps"
        assert dc.determine_mixed_precision() == "no"

    def test_mps_uses_fp16_only_when_use_fp16_explicit(self):
        config = TrainingConfig(device="mps", use_bf16=False, use_fp16=True)
        dc = DeviceConfigurator(config)
        dc.device_type = "mps"
        assert dc.determine_mixed_precision() == "fp16"

    def test_mps_fp32_when_amp_flags_disabled(self):
        config = TrainingConfig(device="mps", use_bf16=False, use_fp16=False)
        dc = DeviceConfigurator(config)
        dc.device_type = "mps"
        assert dc.determine_mixed_precision() == "no"


class TestDeviceConfiguratorDistributed:
    def test_build_fsdp_plugin_when_accelerate_available(self):
        if not _has_accelerate():
            pytest.skip("accelerate 未安装")
        if not torch.cuda.is_available():
            pytest.skip("CUDA 不可用")

        config = TrainingConfig(device="cuda")
        config.distributed_settings = DistributedConfig(
            enabled=True,
            strategy="fsdp",
            fsdp_sharding_strategy="FULL_SHARD",
        )
        plugin = DeviceConfigurator(config).build_distributed_plugin(
            config.distributed_settings
        )
        assert plugin is not None

    def test_build_deepspeed_plugin_requires_config_file(self):
        if not _has_accelerate():
            pytest.skip("accelerate 未安装")

        config = TrainingConfig()
        config.distributed_settings = DistributedConfig(
            enabled=True,
            strategy="deepspeed",
            deepspeed_config_file=None,
        )
        plugin = DeviceConfigurator(config).build_distributed_plugin(
            config.distributed_settings
        )
        assert plugin is None

    def test_disabled_strategy_returns_none(self):
        config = TrainingConfig()
        config.distributed_settings = DistributedConfig(enabled=False, strategy="none")
        assert (
            DeviceConfigurator(config).build_distributed_plugin(
                config.distributed_settings
            )
            is None
        )


class TestTrainerV2AcceleratorWiring:
    def test_create_accelerator_attaches_fsdp_plugin(self):
        if not _has_accelerate():
            pytest.skip("accelerate 未安装")

        from florence_forge.training.trainer_refactored import MultiTaskTrainer

        config = TrainingConfig(device="cpu", num_epochs=1)
        config.distributed_settings = DistributedConfig(
            enabled=True,
            strategy="fsdp",
            fsdp_sharding_strategy="FULL_SHARD",
        )
        model = nn.Linear(4, 2)
        train_dataset = MagicMock()
        val_dataset = None

        fake_plugin = object()
        with patch.object(
            DeviceConfigurator,
            "build_distributed_plugin",
            return_value=fake_plugin,
        ):
            with patch(
                "florence_forge.training.trainer_refactored.Accelerator"
            ) as accelerator_cls:
                trainer = MultiTaskTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    config=config,
                )
                trainer._create_accelerator()

        _, kwargs = accelerator_cls.call_args
        assert kwargs.get("fsdp_plugin") is fake_plugin


def test_plot_backend_defaults_to_no_show(monkeypatch):
    from florence_forge.evaluation.plot_backend import should_show_plots

    monkeypatch.delenv("FLORENCE_FORGE_SHOW_PLOTS", raising=False)
    assert should_show_plots() is False
    monkeypatch.setenv("FLORENCE_FORGE_SHOW_PLOTS", "1")
    assert should_show_plots() is True


def test_training_loop_runs_single_optimizer_step_cpu():
    """v2 训练循环：单 batch 前向 + 反向 + optimizer.step（CPU，无真实模型）。"""
    from florence_forge.training.training_loop import TrainingLoop

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(3, 1)

        def forward(self, input_ids=None, labels=None, **kwargs):
            x = input_ids.float()
            loss = self.proj(x).mean()
            return type("Out", (), {"loss": loss})()

    config = TrainingConfig(device="cpu", num_epochs=1, batch_size=1)
    config.max_grad_norm = 0.0
    config.gradient_accumulation_steps = 1
    config.logging_steps = 999

    model = TinyModel()
    loop = TrainingLoop(model=model, config=config, accelerator=None)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "labels": torch.tensor([[1, 2, 3]]),
    }

    class OneBatchLoader:
        def __iter__(self):
            return iter([batch])

        def __len__(self):
            return 1

    metrics = loop.train_epoch(OneBatchLoader(), optimizer, None, epoch=0)
    assert loop.global_step == 1
    assert "loss" in metrics


@pytest.mark.skipif(not _has_accelerate(), reason="accelerate 未安装")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA 不可用")
def test_accelerate_cuda_training_step_smoke():
    """Accelerate 在 CUDA 上完成一步 backward + optimizer.step。"""
    from accelerate import Accelerator

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 2)

        def forward(self, x):
            return self.fc(x)

    accelerator = Accelerator()
    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model, optimizer = accelerator.prepare(model, optimizer)

    inputs = torch.randn(2, 4, device=accelerator.device)
    with accelerator.accumulate(model):
        loss = model(inputs).sum()
        accelerator.backward(loss)
        if accelerator.sync_gradients:
            optimizer.step()
            optimizer.zero_grad()

    assert accelerator.device.type == "cuda"


@pytest.mark.skipif(not _has_accelerate(), reason="accelerate 未安装")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA 不可用")
def test_multitask_trainer_v2_setup_and_one_training_step_cuda():
    """v2 MultiTaskTrainer：setup_training + TrainingLoop 单步（CUDA）。"""
    from florence_forge.training.trainer_refactored import MultiTaskTrainer

    class TinyModel(nn.Module):
        def forward(self, input_ids=None, labels=None, **kwargs):
            loss = input_ids.float().mean()
            return type("Out", (), {"loss": loss})()

    batch = {
        "input_ids": torch.randint(0, 10, (2, 8)),
        "attention_mask": torch.ones(2, 8),
        "labels": torch.randint(0, 10, (2, 8)),
        "task_type": "CAPTION",
    }

    class OneBatchLoader:
        def __iter__(self):
            return iter([batch])

        def __len__(self):
            return 1

    train_dataset = MagicMock()
    train_dataset.__len__ = MagicMock(return_value=1)

    config = TrainingConfig(device="cuda", num_epochs=1, batch_size=2)
    config.max_grad_norm = 0.0
    config.gradient_accumulation_steps = 1
    config.logging_steps = 999
    config.use_callbacks = False
    config.use_lora = False

    trainer = MultiTaskTrainer(
        model=TinyModel(),
        train_dataset=train_dataset,
        val_dataset=None,
        config=config,
    )
    trainer.setup_training()
    trainer.train_dataloader = OneBatchLoader()

    metrics = trainer.training_loop.train_epoch(
        train_dataloader=trainer.train_dataloader,
        optimizer=trainer.optimizer,
        lr_scheduler=trainer.lr_scheduler,
        epoch=0,
    )
    assert trainer.training_loop.global_step == 1
    assert "loss" in metrics


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="需要至少 2 张 CUDA GPU",
)
def test_multi_gpu_cuda_tensor_placement_smoke():
    """多 GPU 环境最小冒烟：张量可在 cuda:0 / cuda:1 上创建。"""
    device0 = torch.device("cuda:0")
    device1 = torch.device("cuda:1")
    x = torch.ones(2, 2, device=device0)
    y = torch.ones(2, 2, device=device1)
    assert x.device.type == "cuda"
    assert y.device.index == 1
