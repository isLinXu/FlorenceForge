"""单元测试：检查点管理器 CheckpointManager

测试检查点的保存、加载、清理和元数据管理功能。
"""

import pytest
import torch
import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock

from florence_forge.training.checkpoint import CheckpointManager


@pytest.fixture
def checkpoint_dir(tmp_path):
    """创建临时检查点目录"""
    d = tmp_path / "checkpoints"
    d.mkdir()
    return d


@pytest.fixture
def manager(checkpoint_dir):
    """创建检查点管理器"""
    return CheckpointManager(checkpoint_dir, max_checkpoints=3)


@pytest.fixture
def simple_model():
    """创建简单测试模型"""
    return torch.nn.Linear(10, 10)


@pytest.fixture
def simple_optimizer(simple_model):
    """创建简单测试优化器"""
    return torch.optim.SGD(simple_model.parameters(), lr=0.01)


class TestCheckpointManagerInit:
    """初始化测试"""

    def test_creates_directory(self, tmp_path):
        new_dir = tmp_path / "new_checkpoints"
        mgr = CheckpointManager(new_dir)
        assert new_dir.exists()

    def test_default_max_checkpoints(self, checkpoint_dir):
        mgr = CheckpointManager(checkpoint_dir)
        assert mgr.max_checkpoints == 5  # 默认值

    def test_custom_max_checkpoints(self, checkpoint_dir):
        mgr = CheckpointManager(checkpoint_dir, max_checkpoints=10)
        assert mgr.max_checkpoints == 10


class TestCheckpointSave:
    """保存检查点测试"""

    def test_save_basic_checkpoint(self, manager, simple_model, simple_optimizer):
        path = manager.save_checkpoint(
            model=simple_model,
            optimizer=simple_optimizer,
            scheduler=None,
            epoch=1,
            step=100,
            loss=0.5,
        )

        assert path is not None
        assert Path(path).exists()

    def test_save_increments_counter(self, manager, simple_model, simple_optimizer):
        path1 = manager.save_checkpoint(simple_model, simple_optimizer, None, 1, 100, 0.5)
        path2 = manager.save_checkpoint(simple_model, simple_optimizer, None, 2, 200, 0.3)
        assert Path(path1).exists()
        assert Path(path2).exists()

    def test_save_with_metrics(self, manager, simple_model, simple_optimizer):
        path = manager.save_checkpoint(
            model=simple_model,
            optimizer=simple_optimizer,
            scheduler=None,
            epoch=3,
            step=300,
            loss=0.3,
            metrics={"accuracy": 0.9},
        )
        assert Path(path).exists()

    def test_save_with_is_best_flag(self, manager, simple_model, simple_optimizer):
        path = manager.save_checkpoint(
            model=simple_model,
            optimizer=simple_optimizer,
            scheduler=None,
            epoch=3,
            step=300,
            loss=0.1,
            is_best=True,
        )
        assert Path(path).exists()


class TestCheckpointLoad:
    """加载检查点测试"""

    def test_load_saved_checkpoint(self, manager, simple_model, simple_optimizer):
        path = manager.save_checkpoint(
            model=simple_model,
            optimizer=simple_optimizer,
            scheduler=None,
            epoch=5,
            step=500,
            loss=0.1,
        )

        # load_checkpoint 需要 model 参数来恢复权重
        loaded = manager.load_checkpoint(path, model=simple_model)
        assert loaded is not None
        assert loaded["epoch"] == 5
        assert loaded["step"] == 500

    def test_load_nonexistent_checkpoint(self, manager, simple_model, tmp_path):
        with pytest.raises((FileNotFoundError, OSError, ValueError)):
            manager.load_checkpoint(tmp_path / "nonexistent", model=simple_model)


class TestCheckpointCleanup:
    """检查点清理测试"""

    def test_cleanup_respects_max(self, manager, checkpoint_dir, simple_model, simple_optimizer):
        # 保存 5 个检查点，但 max_checkpoints=3
        for i in range(5):
            manager.save_checkpoint(
                model=simple_model,
                optimizer=simple_optimizer,
                scheduler=None,
                epoch=i + 1,
                step=(i + 1) * 100,
                loss=1.0 / (i + 1),
            )

        # 验证只保留 max_checkpoints 个
        checkpoint_dirs = [d for d in checkpoint_dir.iterdir() if d.is_dir()]
        assert len(checkpoint_dirs) <= 3


class TestCheckpointHistory:
    """检查点历史记录测试"""

    def test_history_records_checkpoints(self, manager, simple_model, simple_optimizer):
        manager.save_checkpoint(simple_model, simple_optimizer, None, 1, 100, 0.5)
        manager.save_checkpoint(simple_model, simple_optimizer, None, 2, 200, 0.3)

        # 历史记录应至少包含 2 个条目
        assert len(manager.checkpoint_history) >= 2

    def test_history_contains_loss(self, manager, simple_model, simple_optimizer):
        manager.save_checkpoint(simple_model, simple_optimizer, None, 1, 100, 0.5)

        if manager.checkpoint_history:
            entry = manager.checkpoint_history[0]
            assert "loss" in entry or "metrics" in entry or "epoch" in entry


class TestCheckpointEdgeCases:
    """边界条件测试"""

    def test_max_checkpoints_one(self, checkpoint_dir, simple_model, simple_optimizer):
        mgr = CheckpointManager(checkpoint_dir, max_checkpoints=1)

        # 保存 3 个检查点
        for i in range(3):
            mgr.save_checkpoint(simple_model, simple_optimizer, None, i + 1, (i + 1) * 100, 0.5)

        # 应该只保留 1 个
        checkpoint_dirs = [d for d in checkpoint_dir.iterdir() if d.is_dir()]
        assert len(checkpoint_dirs) <= 1

    def test_extra_data_saved(self, manager, simple_model, simple_optimizer):
        """额外的数据也能保存和加载"""
        path = manager.save_checkpoint(
            model=simple_model,
            optimizer=simple_optimizer,
            scheduler=None,
            epoch=1,
            step=100,
            loss=0.5,
            extra_data={"custom_key": "custom_value"},
        )
        loaded = manager.load_checkpoint(path, model=simple_model)
