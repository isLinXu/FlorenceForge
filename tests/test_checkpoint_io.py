"""单元测试：检查点底层共享原语 _checkpoint_io

覆盖 atomic_torch_save / load_checkpoint_file / prune_checkpoints。
"""

import pytest
import torch

from florence_forge.training._checkpoint_io import (
    atomic_torch_save,
    load_checkpoint_file,
    prune_checkpoints,
)


class TestAtomicTorchSave:
    def test_roundtrip(self, tmp_path):
        payload = {"a": torch.tensor([1.0, 2.0]), "epoch": 3}
        path = tmp_path / "nested" / "ckpt.pt"

        returned = atomic_torch_save(payload, path)

        assert returned == path
        assert path.exists()
        loaded = load_checkpoint_file(path)
        assert loaded["epoch"] == 3
        assert torch.allclose(loaded["a"], payload["a"])

    def test_no_tmp_file_left_behind(self, tmp_path):
        path = tmp_path / "ckpt.pt"
        atomic_torch_save({"x": torch.tensor(1.0)}, path)

        leftovers = list(tmp_path.glob("*.tmp-*"))
        assert leftovers == []

    def test_failure_cleans_up_tmp(self, tmp_path, monkeypatch):
        path = tmp_path / "ckpt.pt"

        def boom(*args, **kwargs):
            raise RuntimeError("save failed")

        monkeypatch.setattr(torch, "save", boom)

        with pytest.raises(RuntimeError, match="save failed"):
            atomic_torch_save({"x": 1}, path)

        assert not path.exists()
        assert list(tmp_path.glob("*.tmp-*")) == []

    def test_overwrites_existing_atomically(self, tmp_path):
        path = tmp_path / "ckpt.pt"
        atomic_torch_save({"v": 1}, path)
        atomic_torch_save({"v": 2}, path)

        assert load_checkpoint_file(path)["v"] == 2


class TestLoadCheckpointFile:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises((FileNotFoundError, OSError, ValueError, RuntimeError)):
            load_checkpoint_file(tmp_path / "nope.pt")

    def test_map_location_none_ok(self, tmp_path):
        path = tmp_path / "ckpt.pt"
        atomic_torch_save({"t": torch.tensor([3.0])}, path)
        loaded = load_checkpoint_file(path, map_location=None)
        assert torch.allclose(loaded["t"], torch.tensor([3.0]))


class TestPruneCheckpoints:
    def test_keeps_most_recent_n(self):
        entries = [("a", 1), ("b", 2), ("c", 3), ("d", 4)]
        removed = []

        result = prune_checkpoints(
            entries,
            keep=2,
            sort_key=lambda e: e[1],
            is_protected=lambda e: False,
            remove=removed.append,
        )

        # 保留最新的 c(3)/d(4)，删除 a(1)/b(2)
        assert set(result) == {("a", 1), ("b", 2)}
        assert set(removed) == {("a", 1), ("b", 2)}

    def test_protected_entries_are_kept(self):
        entries = [("a", 1), ("best", 2), ("c", 3), ("d", 4)]
        removed = []

        prune_checkpoints(
            entries,
            keep=1,
            sort_key=lambda e: e[1],
            is_protected=lambda e: e[0] == "best",
            remove=removed.append,
        )

        assert ("best", 2) not in removed
        assert ("a", 1) in removed

    def test_keep_zero_or_negative_is_noop(self):
        removed = []
        result = prune_checkpoints(
            [("a", 1), ("b", 2)],
            keep=0,
            sort_key=lambda e: e[1],
            is_protected=lambda e: False,
            remove=removed.append,
        )
        assert result == []
        assert removed == []

    def test_remove_failure_does_not_raise(self):
        def boom(_entry):
            raise OSError("cannot delete")

        # 删除失败应被吞掉并记录，不抛出
        result = prune_checkpoints(
            [("a", 1), ("b", 2), ("c", 3)],
            keep=1,
            sort_key=lambda e: e[1],
            is_protected=lambda e: False,
            remove=boom,
        )
        assert result == []
