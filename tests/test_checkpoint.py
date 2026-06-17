"""检查点管理器与模型 I/O 工具测试。"""

import torch
import torch.nn as nn
from pathlib import Path

from florence_forge.core.config import TrainingConfig
from florence_forge.training.checkpoint_manager import (
    CheckpointManager,
    load_model_only,
    save_model_only,
)


class TestCheckpointManagerV2:
    def test_save_and_load_checkpoint(self, tmp_path):
        config = TrainingConfig(output_dir=str(tmp_path))
        model = nn.Linear(4, 2)
        manager = CheckpointManager(model=model, config=config)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

        manager.save_checkpoint(
            epoch=0,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            metrics={"loss": 0.5},
            is_best=True,
            async_save=False,
        )

        ckpt_dir = tmp_path / "checkpoint-epoch-0"
        assert (ckpt_dir / "checkpoint.pt").exists()
        assert (ckpt_dir / "BEST_MODEL").exists()

        new_model = nn.Linear(4, 2)
        new_manager = CheckpointManager(model=new_model, config=config)
        metadata = new_manager.load_checkpoint(ckpt_dir)
        assert metadata["epoch"] == 0
        assert metadata["metrics"]["loss"] == 0.5


class TestModelOnlyIO:
    def test_save_and_load_model_only(self, tmp_path):
        model = nn.Linear(3, 3)
        path = tmp_path / "model.pt"
        save_model_only(model, path, metadata={"tag": "test"})
        assert path.exists()

        new_model = nn.Linear(3, 3)
        meta = load_model_only(new_model, path)
        assert meta.get("tag") == "test"
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2)
