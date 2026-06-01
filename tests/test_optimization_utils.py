"""Model/memory optimization utility regression tests.

Also guards against the missing-``typing`` import regression that previously
made ``florence_forge.utils.optimization`` impossible to import.
"""

import pytest
import torch
import torch.nn as nn

from florence_forge.utils.optimization import (
    MemoryOptimizer,
    ModelOptimizer,
    create_model_optimizer,
    quick_prune,
    quick_quantize,
)


def _tiny_model() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))


def _quantization_supported() -> bool:
    """Dynamic quantization needs a usable qengine (absent on some platforms)."""
    try:
        engines = torch.backends.quantized.supported_engines
    except Exception:  # pragma: no cover - defensive
        return False
    usable = [e for e in engines if e != "none"]
    if not usable:
        return False
    try:
        torch.backends.quantized.engine = usable[0]
    except Exception:  # pragma: no cover - defensive
        return False
    try:
        torch.quantization.quantize_dynamic(
            nn.Linear(2, 2), {nn.Linear}, dtype=torch.qint8
        )(torch.randn(1, 2))
        return True
    except Exception:
        return False


requires_quantization = pytest.mark.skipif(
    not _quantization_supported(),
    reason="No usable quantization engine on this platform",
)


# ---------------------------------------------------------------------------
# import regression guard
# ---------------------------------------------------------------------------


def test_module_is_importable():
    import florence_forge.utils.optimization as opt

    assert hasattr(opt, "ModelOptimizer")
    assert hasattr(opt, "MemoryOptimizer")


# ---------------------------------------------------------------------------
# ModelOptimizer basics
# ---------------------------------------------------------------------------


def test_get_model_size_and_param_counts():
    optimizer = ModelOptimizer(_tiny_model())
    assert optimizer.get_model_size() > 0.0
    assert optimizer.count_parameters() == 4 * 8 + 8 + 8 * 2 + 2
    assert optimizer.count_trainable_parameters() == optimizer.count_parameters()


def test_save_and_restore_original_state():
    model = _tiny_model()
    optimizer = ModelOptimizer(model)
    optimizer.save_original_state()
    assert optimizer.original_state is not None

    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)

    optimizer.restore_original_state()
    # restored weights should match the saved snapshot
    saved = optimizer.original_state["state_dict"]
    for name, param in model.state_dict().items():
        assert torch.allclose(param, saved[name])


def test_restore_without_save_is_noop():
    optimizer = ModelOptimizer(_tiny_model())
    optimizer.restore_original_state()  # should warn, not raise
    assert optimizer.original_state is None


def test_optimize_for_inference_disables_grad():
    model = _tiny_model()
    optimizer = ModelOptimizer(model)
    result = optimizer.optimize_for_inference()
    assert all(not p.requires_grad for p in result.parameters())
    assert optimizer.get_optimization_summary()["optimization_history"]


@requires_quantization
def test_quantize_dynamic_records_history():
    optimizer = ModelOptimizer(_tiny_model())
    quantized = optimizer.quantize_model("dynamic")
    assert isinstance(quantized, nn.Module)
    history = optimizer.get_optimization_summary()["optimization_history"]
    assert history[-1]["type"] == "quantization"


def test_quantize_invalid_type_raises():
    optimizer = ModelOptimizer(_tiny_model())
    with pytest.raises(ValueError, match="Unsupported quantization type"):
        optimizer.quantize_model("invalid")


def test_prune_model_unstructured():
    optimizer = ModelOptimizer(_tiny_model())
    pruned = optimizer.prune_model(pruning_ratio=0.5, structured=False)
    assert isinstance(pruned, nn.Module)
    history = optimizer.get_optimization_summary()["optimization_history"]
    assert history[-1]["type"] == "pruning"


def test_benchmark_model_small_runs():
    optimizer = ModelOptimizer(_tiny_model())
    results = optimizer.benchmark_model(
        input_shape=(1, 4), num_runs=2, warmup_runs=1, device=torch.device("cpu")
    )
    assert results["throughput_fps"] > 0
    assert results["parameter_count"] == optimizer.count_parameters()


def test_optimization_summary_reports_state_flag():
    optimizer = ModelOptimizer(_tiny_model())
    summary = optimizer.get_optimization_summary()
    assert summary["original_state_available"] is False
    optimizer.save_original_state()
    assert optimizer.get_optimization_summary()["original_state_available"] is True


# ---------------------------------------------------------------------------
# module-level helpers
# ---------------------------------------------------------------------------


def test_create_model_optimizer_saves_state():
    optimizer = create_model_optimizer(_tiny_model())
    assert optimizer.original_state is not None


def test_quick_prune():
    assert isinstance(quick_prune(_tiny_model(), 0.3), nn.Module)


@requires_quantization
def test_quick_quantize():
    assert isinstance(quick_quantize(_tiny_model(), "dynamic"), nn.Module)


# ---------------------------------------------------------------------------
# MemoryOptimizer
# ---------------------------------------------------------------------------


def test_memory_optimizer_clear_cache_and_usage():
    MemoryOptimizer.clear_cache()  # should not raise on CPU
    usage = MemoryOptimizer.get_memory_usage()
    assert "system_memory_mb" in usage
    assert usage["system_memory_percent"] >= 0


def test_optimize_batch_size_breaks_on_memory_limit():
    model = _tiny_model()
    # max_memory_mb=0 forces a break on the very first iteration
    bs = MemoryOptimizer.optimize_batch_size(
        model, input_shape=(4,), max_memory_mb=0.0, start_batch_size=1
    )
    assert bs == 1
