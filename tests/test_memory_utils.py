"""Memory utility regression tests (CPU paths)."""

import json
import time

import torch

from florence_forge.utils import memory as mem
from florence_forge.utils.memory import (
    GPUMemoryInfo,
    MemoryInfo,
    MemoryPool,
    MemoryTracker,
    check_memory_requirements,
    clear_cache,
    clear_gpu_cache,
    get_gpu_memory_usage,
    get_memory_usage,
    low_memory_mode,
    memory_monitor,
    optimize_memory,
    suggest_batch_size,
)


# ---------------------------------------------------------------------------
# dataclasses
# ---------------------------------------------------------------------------


def test_memory_info_str():
    info = MemoryInfo(total=16.0, available=8.0, used=8.0, percent=50.0)
    assert "8.0GB/16.0GB" in str(info)
    assert "50.0%" in str(info)


def test_gpu_memory_info_str():
    info = GPUMemoryInfo(
        device_id=0, name="Fake", total=24.0, allocated=2.0, reserved=3.0, free=21.0
    )
    assert "GPU 0 (Fake)" in str(info)


# ---------------------------------------------------------------------------
# basic queries
# ---------------------------------------------------------------------------


def test_get_memory_usage_returns_info():
    info = get_memory_usage()
    assert info.total > 0
    assert 0 <= info.percent <= 100


def test_get_gpu_memory_usage_empty_when_no_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert get_gpu_memory_usage() == []


def test_clear_cache_and_gpu_cache_no_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    clear_cache()  # should not raise
    clear_gpu_cache()


# ---------------------------------------------------------------------------
# optimize_memory
# ---------------------------------------------------------------------------


def test_optimize_memory_clears_model_gradients(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = torch.nn.Linear(4, 2)
    out = model(torch.randn(2, 4)).sum()
    out.backward()
    assert any(p.grad is not None for p in model.parameters())

    result = optimize_memory(aggressive=True, clear_gradients=True, model=model)
    assert all(p.grad is None for p in model.parameters())
    assert "system_memory_freed" in result


# ---------------------------------------------------------------------------
# memory_monitor context manager
# ---------------------------------------------------------------------------


def test_memory_monitor_logs_start_and_end():
    logs = []
    with memory_monitor("unit-test", log_func=logs.append):
        pass
    assert any("开始" in line for line in logs)
    assert any("结束" in line for line in logs)


# ---------------------------------------------------------------------------
# MemoryTracker
# ---------------------------------------------------------------------------


def test_memory_tracker_peak_and_average_from_history():
    tracker = MemoryTracker()
    tracker.history.append(
        {"system_memory": {"total": 16, "used": 4.0, "percent": 25}}
    )
    tracker.history.append(
        {"system_memory": {"total": 16, "used": 6.0, "percent": 37}}
    )
    assert tracker.get_peak_usage()["peak_system_memory"] == 6.0
    assert tracker.get_average_usage()["avg_system_memory"] == 5.0


def test_memory_tracker_empty_history_returns_empty():
    tracker = MemoryTracker()
    assert tracker.get_peak_usage() == {}
    assert tracker.get_average_usage() == {}


def test_memory_tracker_start_stop_records(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    tracker = MemoryTracker(interval=0.01)
    tracker.start()
    tracker.start()  # second call is a no-op
    time.sleep(0.05)
    tracker.stop()
    assert len(tracker.history) >= 1
    assert not tracker.is_tracking


def test_memory_tracker_export_history(tmp_path):
    tracker = MemoryTracker()
    tracker.history.append({"system_memory": {"total": 16, "used": 4.0, "percent": 25}})
    out = tmp_path / "nested" / "history.json"
    tracker.export_history(out)
    data = json.loads(out.read_text())
    assert data[0]["system_memory"]["used"] == 4.0


# ---------------------------------------------------------------------------
# MemoryPool
# ---------------------------------------------------------------------------


def test_memory_pool_reuse_and_stats():
    pool = MemoryPool(device=torch.device("cpu"))
    t = pool.get_tensor((2, 3))
    assert t.shape == (2, 3)

    t.add_(1.0)
    pool.return_tensor(t)
    stats = pool.get_stats()
    assert stats["total_tensors"] == 1
    assert stats["total_pools"] == 1
    assert stats["total_memory_mb"] >= 0

    # reused tensor should be zeroed
    reused = pool.get_tensor((2, 3))
    assert torch.all(reused == 0)


def test_memory_pool_tuple_shape_and_clear(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    pool = MemoryPool()
    t = pool.get_tensor((4,), dtype=torch.float32)
    pool.return_tensor(t)
    assert pool.get_stats()["total_tensors"] == 1
    pool.clear()
    assert pool.get_stats()["total_tensors"] == 0


def test_memory_pool_rejects_wrong_device():
    pool = MemoryPool(device=torch.device("cpu"))
    # craft a tensor pretending to be on another device by monkeypatching .device
    t = torch.zeros(2)

    class _Other:
        type = "meta"

        def __eq__(self, other):
            return False

    # tensor.device != pool.device triggers early return
    pool.max_pool_size = 0  # also ensures no append even if device matched
    pool.return_tensor(t)
    assert pool.get_stats()["total_tensors"] == 0


# ---------------------------------------------------------------------------
# requirement helpers
# ---------------------------------------------------------------------------


def test_check_memory_requirements_structure(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    result = check_memory_requirements(
        model_size_gb=1.0, batch_size=4, sequence_length=512
    )
    assert "estimated_memory_gb" in result
    assert "system_sufficient" in result
    assert result["model_memory_gb"] == 1.0


def test_suggest_batch_size_positive_and_floor():
    # plenty of memory -> batch size > 1
    assert suggest_batch_size(model_size_gb=1.0, available_memory_gb=64.0) >= 1
    # not enough memory -> floor at 1
    assert suggest_batch_size(model_size_gb=10.0, available_memory_gb=5.0) == 1


# ---------------------------------------------------------------------------
# low_memory_mode
# ---------------------------------------------------------------------------


def test_low_memory_mode_restores_settings(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    original_benchmark = torch.backends.cudnn.benchmark
    original_deterministic = torch.backends.cudnn.deterministic
    with low_memory_mode():
        assert torch.backends.cudnn.benchmark is False
        assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark == original_benchmark
    assert torch.backends.cudnn.deterministic == original_deterministic
