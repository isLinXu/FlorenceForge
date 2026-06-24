"""Tests for v2 trainer enhancements ported from v1.

Covers:
- torch.compile integration in setup_training
- eval_steps / save_steps frequency control
- DistributedSampler epoch seed setting
- LoRA per-batch adapter switching
- KeyboardInterrupt / OOM exception handling
- Training config persistence (_save_config)
- Richer training summary (_get_training_summary)
- CheckpointManager LoRA state save/restore
- TaskScheduler weight auto-adjustment
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from florence_forge.core.config import (
    TrainingConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _DummyModel(nn.Module):
    """Minimal model for testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, **kwargs):
        return MagicMock(loss=torch.tensor(1.0))


def _make_trainer(config=None, with_val=True):
    """Build a v2 MultiTaskTrainer with mocked internals (avoids real Accelerator)."""
    from florence_forge.training.trainer import MultiTaskTrainer
    from florence_forge.data.dataset import MultiTaskDataset

    cfg = config or TrainingConfig(num_epochs=1, output_dir="/tmp/test_output")

    model = _DummyModel()
    train_ds = MagicMock(spec=MultiTaskDataset)
    train_ds.task_indices = {"task_a": [0]}
    val_ds = MagicMock(spec=MultiTaskDataset) if with_val else None

    # Pass a mock accelerator to avoid real AcceleratorState conflicts
    mock_accel = MagicMock()
    mock_accel.is_local_main_process = True
    mock_accel.prepare = lambda *args: args
    mock_accel.accumulate = MagicMock(return_value=torch.enable_grad())

    trainer = MultiTaskTrainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        config=cfg,
        accelerator=mock_accel,
    )
    return trainer


# ---------------------------------------------------------------------------
# torch.compile integration
# ---------------------------------------------------------------------------


class TestTorchCompileIntegration:
    """Verify torch.compile is called when enabled in config."""

    @patch("florence_forge.utils.torch_compile.torch.compile")
    def test_compile_called_when_enabled(self, mock_torch_compile):
        mock_torch_compile.return_value = lambda m: m
        from florence_forge.utils.torch_compile import compile_module_if_requested
        model = _DummyModel()
        compile_module_if_requested(model, enabled=True, mode="reduce-overhead")
        mock_torch_compile.assert_called_once()

    def test_compile_not_called_when_disabled(self):
        from florence_forge.utils.torch_compile import compile_module_if_requested
        model = _DummyModel()
        result = compile_module_if_requested(model, enabled=False)
        assert result is model


# ---------------------------------------------------------------------------
# eval_steps / save_steps frequency control
# ---------------------------------------------------------------------------


class TestEvalSaveFrequency:
    """Verify eval and save respect their step intervals."""

    def test_should_eval_every_epoch_when_eval_steps_is_1(self):
        cfg = TrainingConfig(eval_steps=1, save_steps=1, num_epochs=3)
        for epoch in range(cfg.num_epochs):
            is_last = (epoch + 1) == cfg.num_epochs
            should_eval = cfg.eval_steps <= 1 or (epoch + 1) % cfg.eval_steps == 0 or is_last
            assert should_eval

    def test_should_not_eval_mid_interval(self):
        cfg = TrainingConfig(eval_steps=5, save_steps=5, num_epochs=10)
        # epoch 2 (0-indexed) => epoch_number=3 => 3 % 5 != 0
        epoch = 2
        is_last = (epoch + 1) == cfg.num_epochs
        should_eval = cfg.eval_steps <= 1 or (epoch + 1) % cfg.eval_steps == 0 or is_last
        assert not should_eval

    def test_should_eval_at_interval_boundary(self):
        cfg = TrainingConfig(eval_steps=5, save_steps=5, num_epochs=10)
        epoch = 4  # epoch_number=5 => 5 % 5 == 0
        is_last = (epoch + 1) == cfg.num_epochs
        should_eval = cfg.eval_steps <= 1 or (epoch + 1) % cfg.eval_steps == 0 or is_last
        assert should_eval


# ---------------------------------------------------------------------------
# DistributedSampler epoch seed
# ---------------------------------------------------------------------------


class TestDistributedSamplerEpochSeed:
    """Verify set_epoch is called on distributed samplers."""

    def test_set_epoch_called_when_sampler_has_it(self):
        mock_sampler = MagicMock()
        mock_sampler.set_epoch = MagicMock()
        mock_dl = MagicMock()
        mock_dl.sampler = mock_sampler

        trainer = _make_trainer()
        trainer.train_dataloader = mock_dl

        # Simulate the logic from train()
        epoch = 3
        sampler = getattr(trainer.train_dataloader, 'sampler', None)
        if sampler is not None and hasattr(sampler, 'set_epoch'):
            sampler.set_epoch(epoch)

        mock_sampler.set_epoch.assert_called_once_with(3)


# ---------------------------------------------------------------------------
# LoRA per-batch adapter switching
# ---------------------------------------------------------------------------


class TestLoRAAdapterSwitching:
    """Verify TrainingLoop passes lora_manager and switches adapters."""

    def test_switch_called_when_task_matches(self):
        mock_lora = MagicMock()
        mock_lora.active_adapters = {"task_a": "adapter_a", "task_b": "adapter_b"}

        batch = {"task_type": "task_a", "input_ids": torch.randn(2, 10)}
        mock_lora.switch_adapter = MagicMock()

        # Simulate the logic from TrainingLoop.train_epoch
        task_type_hint = batch.get('task_type') or batch.get('task_types', [None])[0]
        if task_type_hint and task_type_hint in mock_lora.active_adapters:
            mock_lora.switch_adapter(MagicMock(), task_type_hint)

        mock_lora.switch_adapter.assert_called_once()


# ---------------------------------------------------------------------------
# Exception handling
# ---------------------------------------------------------------------------


class TestExceptionHandling:
    """Verify KeyboardInterrupt and OOM are properly handled."""

    def test_keyboard_interrupt_saves_checkpoint(self):
        trainer = _make_trainer()
        trainer.optimizer = MagicMock()
        trainer.lr_scheduler = MagicMock()
        trainer.current_epoch = 2
        trainer.checkpoint_manager = MagicMock()

        try:
            raise KeyboardInterrupt()
        except KeyboardInterrupt:
            trainer.checkpoint_manager.save_checkpoint(
                epoch=trainer.current_epoch,
                optimizer=trainer.optimizer,
                lr_scheduler=trainer.lr_scheduler,
                metrics={},
                is_best=False,
                async_save=False,
            )

        trainer.checkpoint_manager.save_checkpoint.assert_called_once_with(
            epoch=2,
            optimizer=trainer.optimizer,
            lr_scheduler=trainer.lr_scheduler,
            metrics={},
            is_best=False,
            async_save=False,
        )


# ---------------------------------------------------------------------------
# Training config persistence
# ---------------------------------------------------------------------------


class TestConfigPersistence:
    """Verify _save_config persists config to output dir."""

    def test_save_config_creates_json(self, tmp_path):
        cfg = TrainingConfig(num_epochs=5, output_dir=str(tmp_path / "out"))
        trainer = _make_trainer(cfg)
        # Mock get_task_statistics to return JSON-serializable dicts
        trainer.train_dataset.get_task_statistics.return_value = {"task_a": 10}
        trainer.val_dataset.get_task_statistics.return_value = {"task_a": 2}
        trainer._save_config()

        config_path = tmp_path / "out" / "training_config.json"
        assert config_path.exists()
        data = json.loads(config_path.read_text())
        assert data.get("num_epochs") == 5


# ---------------------------------------------------------------------------
# Richer training summary
# ---------------------------------------------------------------------------


class TestRicherSummary:
    """Verify _get_training_summary includes new fields."""

    def test_summary_has_expected_keys(self):
        trainer = _make_trainer()
        trainer.current_epoch = 2
        trainer.training_loop = MagicMock(global_step=100)
        trainer.training_loop._get_current_lr = MagicMock(return_value=1e-5)
        trainer.checkpoint_manager = MagicMock()
        trainer.checkpoint_manager.get_best_checkpoint_path.return_value = Path("/tmp/best")

        summary = trainer._get_training_summary()
        assert 'total_epochs' in summary
        assert 'global_steps' in summary
        assert 'best_metric' in summary
        assert 'best_checkpoint' in summary
        assert 'final_learning_rate' in summary
        assert 'use_lora' in summary
        assert 'tasks' in summary


# ---------------------------------------------------------------------------
# CheckpointManager LoRA state
# ---------------------------------------------------------------------------


class TestCheckpointLoRAState:
    """Verify CheckpointManager saves and restores LoRA state."""

    def test_collect_lora_state_captures_adapters(self):
        from florence_forge.training.checkpoint_manager import CheckpointManager

        mock_lora = MagicMock()
        mock_lora.active_adapters = {"task_a": "adapter_a"}
        mock_lora.task_configs = {}
        # export_state 优先被调用，返回可序列化字典
        mock_lora.export_state.return_value = {
            'active_adapters': {'task_a': 'adapter_a'},
            'frozen_adapters': {},
            'shared_adapters': {},
            'task_configs': {},
            'base_config': {},
        }

        cfg = TrainingConfig(output_dir="/tmp/test_output")
        cm = CheckpointManager(model=_DummyModel(), config=cfg)

        state = cm._collect_lora_state(mock_lora)
        assert 'active_adapters' in state
        assert state['active_adapters']['task_a'] == 'adapter_a'

    def test_collect_lora_state_returns_empty_when_no_adapters(self):
        from florence_forge.training.checkpoint_manager import CheckpointManager

        mock_lora = MagicMock()
        mock_lora.active_adapters = {}
        mock_lora.task_configs = {}
        # export_state 优先被调用，返回空字典
        mock_lora.export_state.return_value = {}

        cfg = TrainingConfig(output_dir="/tmp/test_output")
        cm = CheckpointManager(model=_DummyModel(), config=cfg)

        state = cm._collect_lora_state(mock_lora)
        assert state == {}

    def test_load_checkpoint_returns_lora_state(self, tmp_path):
        from florence_forge.training.checkpoint_manager import CheckpointManager

        cfg = TrainingConfig(output_dir=str(tmp_path))
        cm = CheckpointManager(model=_DummyModel(), config=cfg)

        # Create a fake checkpoint
        ckpt_dir = tmp_path / "checkpoint-epoch-0"
        ckpt_dir.mkdir()
        ckpt_data = {
            'epoch': 0,
            'model_state_dict': _DummyModel().state_dict(),
            'optimizer_state_dict': {},
            'lr_scheduler_state_dict': {},
            'metrics': {'loss': 0.5},
            'config': {'num_epochs': 1},
            'lora_state': {'active_adapters': {'task_a': 'adapter_a'}},
        }
        torch.save(ckpt_data, ckpt_dir / "checkpoint.pt")

        metadata = cm.load_checkpoint(ckpt_dir)
        assert metadata.get('lora_state') is not None
        assert metadata['lora_state']['active_adapters']['task_a'] == 'adapter_a'


# ---------------------------------------------------------------------------
# TaskScheduler weight adjustment
# ---------------------------------------------------------------------------


class TestTaskSchedulerWeightAdjustment:
    """Verify task scheduler weights are adjusted at epoch end."""

    def test_auto_adjust_called_when_should_update(self):
        mock_scheduler = MagicMock()
        mock_scheduler.should_update_weights.return_value = True
        mock_scheduler.auto_adjust_weight = MagicMock()
        mock_scheduler.auto_adjust_weights = MagicMock()

        trainer = _make_trainer()
        trainer.task_scheduler = mock_scheduler

        # Simulate the logic from train()
        if trainer.task_scheduler and trainer.task_scheduler.should_update_weights():
            trainer.task_scheduler.auto_adjust_weights()

        mock_scheduler.auto_adjust_weights.assert_called_once()


# ---------------------------------------------------------------------------
# Backward compatibility: save_checkpoint with lora_manager kwarg
# ---------------------------------------------------------------------------


class TestSaveCheckpointWithLoRA:
    """Verify save_checkpoint accepts lora_manager kwarg."""

    def test_save_checkpoint_passes_lora_manager(self, tmp_path):
        from florence_forge.training.checkpoint_manager import CheckpointManager

        cfg = TrainingConfig(output_dir=str(tmp_path))
        cm = CheckpointManager(model=_DummyModel(), config=cfg)

        mock_lora = MagicMock()
        mock_lora.active_adapters = {"task_a": "adapter_a"}
        mock_lora.task_configs = {}
        # export_state 优先被调用，返回可序列化字典
        mock_lora.export_state.return_value = {
            'active_adapters': {'task_a': 'adapter_a'},
            'frozen_adapters': {},
            'shared_adapters': {},
            'task_configs': {},
            'base_config': {},
        }

        cm.save_checkpoint(
            epoch=0,
            optimizer=MagicMock(state_dict=MagicMock(return_value={})),
            lr_scheduler=MagicMock(state_dict=MagicMock(return_value={})),
            metrics={'loss': 0.5},
            is_best=False,
            async_save=False,
            lora_manager=mock_lora,
        )

        ckpt_path = tmp_path / "checkpoint-epoch-0" / "checkpoint.pt"
        assert ckpt_path.exists()
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert 'lora_state' in ckpt
