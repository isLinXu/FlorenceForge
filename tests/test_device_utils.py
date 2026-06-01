"""Device utility regression tests."""

from collections import namedtuple

import pytest
import torch

import florence_forge.utils.device as device_mod
from florence_forge.utils.device import (
    DeviceInfo,
    DeviceManager,
    check_device_compatibility,
    get_cpu_info,
    get_cuda_info,
    get_device_info,
    get_mps_info,
    get_optimal_device,
    move_to_device,
    optimize_device_settings,
    set_device,
    setup_device,
)


class _FakeProps:
    def __init__(self, name="Fake GPU", total_memory_gb=24.0, major=8, minor=0):
        self.name = name
        self.total_memory = int(total_memory_gb * (1024**3))
        self.major = major
        self.minor = minor


def _patch_cuda(monkeypatch, *, count=1, props=None, allocated_gb=2.0):
    props = props or _FakeProps()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: props)
    monkeypatch.setattr(
        torch.cuda, "memory_allocated", lambda i=0: int(allocated_gb * (1024**3))
    )


def _patch_no_accelerators(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if hasattr(torch.backends, "mps"):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)


def test_move_to_device_preserves_namedtuple_type_and_fields():
    Batch = namedtuple("Batch", ["input_ids", "metadata"])
    batch = Batch(
        input_ids=torch.tensor([1, 2]),
        metadata={"labels": torch.tensor([3, 4]), "id": "sample-1"},
    )

    moved = move_to_device(batch, torch.device("cpu"))

    assert isinstance(moved, Batch)
    assert moved.input_ids.device.type == "cpu"
    assert moved.metadata["labels"].device.type == "cpu"
    assert moved.metadata["id"] == "sample-1"


def test_move_to_device_preserves_plain_tuple_and_list_containers():
    nested = ([torch.tensor([1])], (torch.tensor([2]), "keep"))

    moved = move_to_device(nested, torch.device("cpu"))

    assert isinstance(moved, tuple)
    assert isinstance(moved[0], list)
    assert isinstance(moved[1], tuple)
    assert moved[0][0].device.type == "cpu"
    assert moved[1][0].device.type == "cpu"
    assert moved[1][1] == "keep"


def test_set_device_falls_back_to_cpu_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    device = set_device("cuda:0", verbose=False)

    assert device == torch.device("cpu")


def test_device_info_formats_cpu_and_cuda():
    assert str(DeviceInfo(device_type="cpu", name="Test CPU")) == "CPU: Test CPU"
    assert (
        str(
            DeviceInfo(
                device_type="cuda",
                device_id=1,
                name="Test GPU",
                memory_total=24.0,
                memory_available=12.0,
            )
        )
        == "CUDA 1: Test GPU (12.0GB/24.0GB)"
    )


def test_device_info_formats_mps_and_other():
    assert str(DeviceInfo(device_type="mps", name="M3")) == "MPS: M3"
    assert str(DeviceInfo(device_type="xpu", name="Intel")) == "xpu: Intel"


# ---------------------------------------------------------------------------
# get_cpu_info
# ---------------------------------------------------------------------------


def test_get_cpu_info_returns_available_device():
    info = get_cpu_info()
    assert info.device_type == "cpu"
    assert info.is_available is True
    assert info.memory_total >= 0.0


# ---------------------------------------------------------------------------
# get_cuda_info
# ---------------------------------------------------------------------------


def test_get_cuda_info_empty_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert get_cuda_info() == []


def test_get_cuda_info_reports_devices(monkeypatch):
    _patch_cuda(monkeypatch, count=2, props=_FakeProps(major=7, minor=5), allocated_gb=4.0)
    devices = get_cuda_info()
    assert len(devices) == 2
    first = devices[0]
    assert first.device_type == "cuda"
    assert first.compute_capability == "7.5"
    assert first.memory_total == pytest.approx(24.0)
    assert first.memory_available == pytest.approx(20.0)


def test_get_cuda_info_handles_property_errors(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    def boom(i):
        raise RuntimeError("driver error")

    monkeypatch.setattr(torch.cuda, "get_device_properties", boom)
    assert get_cuda_info() == []


# ---------------------------------------------------------------------------
# get_mps_info
# ---------------------------------------------------------------------------


def test_get_mps_info_none_when_unavailable(monkeypatch):
    if hasattr(torch.backends, "mps"):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert get_mps_info() is None


# ---------------------------------------------------------------------------
# get_device_info
# ---------------------------------------------------------------------------


def test_get_device_info_aggregates(monkeypatch):
    _patch_no_accelerators(monkeypatch)
    info = get_device_info()
    assert "cpu" in info
    assert info["cuda_devices"] == []
    assert info["pytorch_version"] == torch.__version__
    assert info["cuda_version"] is None


# ---------------------------------------------------------------------------
# get_optimal_device
# ---------------------------------------------------------------------------


def test_get_optimal_device_explicit_id_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert get_optimal_device(device_id=3) == torch.device("cpu")


def test_get_optimal_device_picks_largest_memory_gpu(monkeypatch):
    big = DeviceInfo(device_type="cuda", device_id=1, memory_available=20.0)
    small = DeviceInfo(device_type="cuda", device_id=0, memory_available=3.0)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(device_mod, "get_cuda_info", lambda: [small, big])

    assert get_optimal_device(prefer_gpu=True, min_memory_gb=2.0) == torch.device("cuda:1")


def test_get_optimal_device_falls_back_to_cpu_without_gpu(monkeypatch):
    _patch_no_accelerators(monkeypatch)
    assert get_optimal_device(prefer_gpu=True) == torch.device("cpu")


# ---------------------------------------------------------------------------
# set_device
# ---------------------------------------------------------------------------


def test_set_device_int_resolves_to_cuda(monkeypatch):
    _patch_cuda(monkeypatch, count=2)
    assert set_device(1, verbose=False) == torch.device("cuda:1")


def test_set_device_int_falls_back_when_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert set_device(2, verbose=False) == torch.device("cpu")


def test_set_device_clamps_out_of_range_cuda_index(monkeypatch):
    _patch_cuda(monkeypatch, count=1)
    assert set_device("cuda:5", verbose=False) == torch.device("cuda:0")


def test_set_device_mps_falls_back_when_unavailable(monkeypatch):
    if hasattr(torch.backends, "mps"):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert set_device("mps", verbose=False) == torch.device("cpu")


# ---------------------------------------------------------------------------
# check_device_compatibility
# ---------------------------------------------------------------------------


def test_check_device_compatibility_cpu_only(monkeypatch):
    fake_info = {
        "cpu": DeviceInfo(device_type="cpu", name="CPU", memory_available=64.0),
        "cuda_devices": [],
        "mps": None,
    }
    monkeypatch.setattr(device_mod, "get_device_info", lambda: fake_info)

    result = check_device_compatibility(required_memory_gb=4.0)
    assert "cpu" in result["compatible_devices"]


def test_check_device_compatibility_flags_insufficient_cpu_and_old_gpu(monkeypatch):
    fake_info = {
        "cpu": DeviceInfo(device_type="cpu", name="CPU", memory_available=1.0),
        "cuda_devices": [
            DeviceInfo(
                device_type="cuda",
                device_id=0,
                memory_available=8.0,
                compute_capability="5.2",
            )
        ],
        "mps": None,
    }
    monkeypatch.setattr(device_mod, "get_device_info", lambda: fake_info)

    result = check_device_compatibility(required_memory_gb=4.0)
    assert "cuda:0" in result["compatible_devices"]
    assert any("架构较老" in w for w in result["warnings"])


def test_check_device_compatibility_no_devices(monkeypatch):
    fake_info = {
        "cpu": DeviceInfo(device_type="cpu", name="CPU", memory_available=1.0),
        "cuda_devices": [],
        "mps": None,
    }
    monkeypatch.setattr(device_mod, "get_device_info", lambda: fake_info)

    result = check_device_compatibility(required_memory_gb=4.0)
    assert result["compatible_devices"] == []
    assert any("没有找到兼容" in w for w in result["warnings"])


def test_check_device_compatibility_includes_mps(monkeypatch):
    fake_info = {
        "cpu": DeviceInfo(device_type="cpu", name="CPU", memory_available=64.0),
        "cuda_devices": [],
        "mps": DeviceInfo(device_type="mps", name="M3", is_available=True),
    }
    monkeypatch.setattr(device_mod, "get_device_info", lambda: fake_info)

    result = check_device_compatibility(required_memory_gb=4.0)
    assert "mps" in result["compatible_devices"]


# ---------------------------------------------------------------------------
# optimize_device_settings
# ---------------------------------------------------------------------------


def test_optimize_device_settings_cpu():
    settings = optimize_device_settings(torch.device("cpu"), enable_amp=True)
    assert settings["amp_enabled"] is False
    assert any("CPU线程" in opt for opt in settings["optimizations"])


def test_optimize_device_settings_cuda_with_tensor_cores(monkeypatch):
    _patch_cuda(monkeypatch, props=_FakeProps(major=8))
    settings = optimize_device_settings(torch.device("cuda:0"), enable_amp=True)
    assert settings["amp_enabled"] is True


def test_optimize_device_settings_cuda_old_gpu_disables_amp(monkeypatch):
    _patch_cuda(monkeypatch, props=_FakeProps(major=6))
    settings = optimize_device_settings(torch.device("cuda:0"), enable_amp=True)
    assert settings["amp_enabled"] is False


# ---------------------------------------------------------------------------
# setup_device
# ---------------------------------------------------------------------------


def test_setup_device_auto_selects_cpu(monkeypatch):
    _patch_no_accelerators(monkeypatch)
    device, settings = setup_device(device=None, verbose=False)
    assert device == torch.device("cpu")
    assert settings["device"] == "cpu"


def test_setup_device_with_explicit_device(monkeypatch):
    _patch_no_accelerators(monkeypatch)
    device, settings = setup_device(device="cpu", verbose=True)
    assert device == torch.device("cpu")


# ---------------------------------------------------------------------------
# DeviceManager
# ---------------------------------------------------------------------------


def test_device_manager_cpu_lifecycle(monkeypatch):
    _patch_no_accelerators(monkeypatch)
    manager = DeviceManager(auto_select=True, prefer_gpu=True)
    assert manager.device == torch.device("cpu")

    tensor = torch.tensor([1, 2, 3])
    moved = manager.move_to_device(tensor)
    assert moved.device.type == "cpu"

    mem = manager.get_memory_info()
    assert "total_gb" in mem

    manager.clear_cache()  # no-op on cpu, should not raise

    summary = manager.get_device_summary()
    assert "当前设备" in summary


def test_device_manager_no_auto_select_uses_cpu(monkeypatch):
    _patch_no_accelerators(monkeypatch)
    manager = DeviceManager(auto_select=False)
    assert manager.device == torch.device("cpu")


# ---------------------------------------------------------------------------
# additional branch coverage
# ---------------------------------------------------------------------------


def test_set_device_verbose_cpu_logs(monkeypatch):
    _patch_no_accelerators(monkeypatch)
    device = set_device("cpu", verbose=True)
    assert device == torch.device("cpu")


def test_get_optimal_device_explicit_valid_id(monkeypatch):
    _patch_cuda(monkeypatch, count=4)
    assert get_optimal_device(device_id=2) == torch.device("cuda:2")


def test_get_optimal_device_mps_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if hasattr(torch.backends, "mps"):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
        assert get_optimal_device(prefer_gpu=True) == torch.device("mps")


def test_move_to_device_handles_module():
    module = torch.nn.Linear(2, 2)
    moved = move_to_device(module, torch.device("cpu"))
    assert isinstance(moved, torch.nn.Module)


def test_move_to_device_passthrough_non_tensor():
    assert move_to_device("plain-string", torch.device("cpu")) == "plain-string"


def test_check_device_compatibility_recommends_tensor_cores(monkeypatch):
    fake_info = {
        "cpu": DeviceInfo(device_type="cpu", name="CPU", memory_available=64.0),
        "cuda_devices": [
            DeviceInfo(
                device_type="cuda",
                device_id=0,
                memory_available=16.0,
                compute_capability="8.0",
            )
        ],
        "mps": None,
    }
    monkeypatch.setattr(device_mod, "get_device_info", lambda: fake_info)
    result = check_device_compatibility(required_memory_gb=4.0)
    assert any("Tensor Cores" in r for r in result["recommendations"])
    assert result["recommendations"][0].startswith("推荐使用CUDA")


def test_check_device_compatibility_pascal_and_insufficient_memory(monkeypatch):
    fake_info = {
        "cpu": DeviceInfo(device_type="cpu", name="CPU", memory_available=64.0),
        "cuda_devices": [
            DeviceInfo(
                device_type="cuda",
                device_id=0,
                memory_available=16.0,
                compute_capability="6.1",
            ),
            DeviceInfo(
                device_type="cuda",
                device_id=1,
                memory_available=1.0,
                compute_capability="8.0",
            ),
        ],
        "mps": None,
    }
    monkeypatch.setattr(device_mod, "get_device_info", lambda: fake_info)
    result = check_device_compatibility(required_memory_gb=4.0)
    assert any("较老架构" in r for r in result["recommendations"])
    assert any("内存不足" in w for w in result["warnings"])


def test_optimize_device_settings_mps_and_compile():
    settings = optimize_device_settings(
        torch.device("mps"), enable_amp=True, enable_compile=True
    )
    assert settings["amp_enabled"] is True
    assert settings["compile_enabled"] is True
    assert any("MPS" in opt for opt in settings["optimizations"])


def test_device_manager_cuda_memory_and_summary(monkeypatch):
    _patch_cuda(monkeypatch, props=_FakeProps(name="A100", major=8), allocated_gb=4.0)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda i=0: int(6 * (1024**3)))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    manager = DeviceManager(auto_select=True, prefer_gpu=True, min_memory_gb=2.0)
    assert manager.device.type == "cuda"

    mem = manager.get_memory_info()
    assert mem["total_gb"] == pytest.approx(24.0)
    assert mem["reserved_gb"] == pytest.approx(6.0)

    manager.clear_cache()
    summary = manager.get_device_summary()
    assert "A100" in summary


def test_device_manager_set_device_updates_settings(monkeypatch):
    _patch_no_accelerators(monkeypatch)
    manager = DeviceManager(auto_select=False)
    manager.set_device("cpu")
    assert manager.device == torch.device("cpu")
    assert "optimizations" in manager.settings


def test_get_cpu_info_falls_back_on_error(monkeypatch):
    monkeypatch.setattr(device_mod.platform, "system", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    info = get_cpu_info()
    assert info.device_type == "cpu"
    assert info.is_available is True
