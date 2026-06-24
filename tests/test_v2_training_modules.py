"""v2 训练栈新增模块单元测试。"""


import torch.nn as nn

from florence_forge.core.config import TrainingConfig
from florence_forge.training.async_checkpoint import AsyncCheckpointSaver
from florence_forge.training.deepspeed_plugin import DeepSpeedPlugin
from florence_forge.training.fsdp_plugin import FSDPPlugin
from florence_forge.training.gradient_checkpoint_optimizer import (
    ActivationRecomputePolicy,
    GradientCheckpointOptimizer,
)


class TestFSDPPlugin:
    def test_is_available_property(self):
        plugin = FSDPPlugin()
        assert isinstance(plugin.is_available, bool)

    def test_configure_fsdp_returns_strategy(self):
        config = TrainingConfig()
        config.distributed_settings.fsdp_sharding_strategy = "FULL_SHARD"
        result = FSDPPlugin().configure_fsdp(config)
        assert result["sharding_strategy"] == "FULL_SHARD"


class TestDeepSpeedPlugin:
    def test_configure_deepspeed_returns_zero_stage(self):
        config = TrainingConfig()
        config.distributed_settings.deepspeed_stage = 2
        result = DeepSpeedPlugin().configure_deepspeed(config)
        assert result["zero_stage"] == 2


class TestActivationRecomputePolicy:
    def test_policy_maps_to_full(self):
        model = nn.Linear(4, 2)
        config = TrainingConfig()
        config.model_settings.gradient_checkpointing = True
        config.model_settings.activation_checkpointing_strategy = "full"

        optimizer = GradientCheckpointOptimizer(
            model=model,
            config=config,
            policy=ActivationRecomputePolicy.high,
        )
        assert optimizer._resolve_strategy() == "full"

    def test_policy_off_disables_checkpointing(self):
        model = nn.Linear(4, 2)
        config = TrainingConfig()
        config.model_settings.activation_checkpointing_strategy = "none"

        optimizer = GradientCheckpointOptimizer(
            model=model,
            config=config,
            policy=ActivationRecomputePolicy.off,
        )
        assert optimizer._resolve_strategy() == "none"


class TestAsyncCheckpointSaver:
    def test_sync_save_writes_checkpoint(self, tmp_path):
        saver = AsyncCheckpointSaver(
            checkpoint_dir=str(tmp_path),
            async_save=False,
            compression="none",
        )
        state = {"epoch": 1, "loss": 0.5}
        ckpt_dir = tmp_path / "checkpoint-epoch-1"
        saver.save(state, str(ckpt_dir))
        assert (ckpt_dir / "checkpoint.pt").exists()

    def test_gzip_compression(self, tmp_path):
        saver = AsyncCheckpointSaver(
            checkpoint_dir=str(tmp_path),
            async_save=False,
            compression="gzip",
        )
        state = {"epoch": 2}
        ckpt_dir = tmp_path / "checkpoint-epoch-2"
        saver.save(state, str(ckpt_dir))
        assert (ckpt_dir / "checkpoint.pt").exists()

    def test_cleanup_removes_old_checkpoints(self, tmp_path):
        saver = AsyncCheckpointSaver(
            checkpoint_dir=str(tmp_path),
            max_checkpoints=2,
            async_save=False,
        )
        for epoch in range(4):
            ckpt_dir = tmp_path / f"checkpoint-epoch-{epoch}"
            saver.save({"epoch": epoch}, str(ckpt_dir))

        remaining = list(tmp_path.glob("checkpoint-epoch-*"))
        assert len(remaining) <= 2

    def test_shutdown_waits_for_pending(self, tmp_path):
        saver = AsyncCheckpointSaver(
            checkpoint_dir=str(tmp_path),
            async_save=True,
        )
        saver.save({"epoch": 0}, str(tmp_path / "checkpoint-epoch-0"))
        saver.shutdown(wait=True)
        assert (tmp_path / "checkpoint-epoch-0" / "checkpoint.pt").exists()
