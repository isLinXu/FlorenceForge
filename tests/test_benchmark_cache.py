"""Benchmark 缓存安全回归测试。"""

import pickle
from collections import deque
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from florence_forge.evaluation.benchmark import BenchmarkEvaluator
from florence_forge.evaluation.benchmark_cache import BenchmarkCache


def _bare_benchmark_evaluator(tmp_path, allow_pickle=False):
    evaluator = BenchmarkEvaluator.__new__(BenchmarkEvaluator)
    evaluator.enable_incremental = True
    evaluator.cache_dir = tmp_path
    evaluator.config = {"allow_legacy_pickle_cache": allow_pickle}
    return evaluator


def test_load_cached_results_prefers_safe_pt_cache(tmp_path):
    evaluator = _bare_benchmark_evaluator(tmp_path)
    cache_file = tmp_path / "abc.pt"
    torch.save({"results": {"ok": True}}, cache_file)

    assert evaluator._load_cached_results("abc") == {"ok": True}


def test_load_cached_results_ignores_pickle_by_default(tmp_path):
    evaluator = _bare_benchmark_evaluator(tmp_path)
    cache_file = tmp_path / "abc.pkl"
    with open(cache_file, "wb") as f:
        pickle.dump({"unsafe": True}, f)

    assert evaluator._load_cached_results("abc") is None


def test_benchmark_cache_helper_roundtrip_and_stable_key(tmp_path):
    cache = BenchmarkCache(tmp_path, config={}, enable_incremental=True)

    key = BenchmarkCache.make_key("dataset", "CAPTION", {"model": "mock", "rank": 0})
    assert key == BenchmarkCache.make_key("dataset", "CAPTION", {"rank": 0, "model": "mock"})

    cache.save_results(key, {"score": 0.9})

    assert cache.load_results(key) == {"score": 0.9}


def test_monitoring_snapshot_converts_deque_to_list():
    evaluator = BenchmarkEvaluator.__new__(BenchmarkEvaluator)
    evaluator.monitoring_data = {
        "start_time": 1.0,
        "current_progress": 0.5,
        "estimated_time": 10.0,
        "resource_usage": deque([{"cpu_percent": 10.0}], maxlen=1),
        "performance_metrics": deque([{"latency": 0.1}], maxlen=1),
    }

    snapshot = evaluator._monitoring_data_snapshot()

    assert snapshot["resource_usage"] == [{"cpu_percent": 10.0}]
    assert snapshot["performance_metrics"] == [{"latency": 0.1}]
    assert isinstance(snapshot["resource_usage"], list)


def test_setup_model_skips_ddp_when_world_size_is_one():
    evaluator = BenchmarkEvaluator.__new__(BenchmarkEvaluator)
    evaluator.model = nn.Linear(2, 2)
    evaluator.device = torch.device("cpu")
    evaluator.enable_distributed = True
    evaluator.world_size = 1
    evaluator.rank = 0
    evaluator.config = {}

    evaluator._setup_model()

    assert isinstance(evaluator.model, nn.Linear)


def test_setup_model_requires_rendezvous_for_multi_process_ddp(monkeypatch):
    evaluator = BenchmarkEvaluator.__new__(BenchmarkEvaluator)
    evaluator.model = nn.Linear(2, 2)
    evaluator.device = torch.device("cpu")
    evaluator.enable_distributed = True
    evaluator.world_size = 2
    evaluator.rank = 0
    evaluator.config = {}

    monkeypatch.delenv("MASTER_ADDR", raising=False)
    monkeypatch.delenv("MASTER_PORT", raising=False)

    with patch("florence_forge.evaluation.benchmark.dist.is_initialized", return_value=False):
        with pytest.raises(ValueError, match="MASTER_ADDR"):
            evaluator._setup_model()


def test_parallel_benchmark_uses_torch_spawn_and_collects_results(tmp_path, monkeypatch):
    evaluator = BenchmarkEvaluator.__new__(BenchmarkEvaluator)
    evaluator.model = nn.Linear(1, 1)
    evaluator.config = {"batch_size": 1, "num_workers": 0}
    evaluator.cache_dir = tmp_path / "cache"
    evaluator.cache_dir.mkdir()
    evaluator.enable_incremental = False
    evaluator._compute_overall_summary = MagicMock(return_value={"datasets": 2})
    evaluator._compute_task_performance = MagicMock(return_value={"mixed": {"count": 2}})
    evaluator._save_benchmark_results = MagicMock()

    spawn_call = {}

    def fake_spawn(fn, args, nprocs, join):
        payload = args[0]
        spawn_call["fn"] = fn
        spawn_call["nprocs"] = nprocs
        spawn_call["join"] = join
        spawn_call["payload"] = payload

        assert next(payload["model_template"].parameters()).device.type == "cpu"
        for rank in range(nprocs):
            dataset_name = payload["assignments"][rank][0][0]
            result_file = tmp_path / "out" / ".parallel_results" / f"worker_{rank}.pt"
            result_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save({dataset_name: {"rank": rank}}, result_file)

    monkeypatch.setattr("florence_forge.evaluation.benchmark_parallel.torch_mp.spawn", fake_spawn)

    results = evaluator._run_parallel_benchmark(
        datasets={"dataset_a": object(), "dataset_b": object()},
        output_path=tmp_path / "out",
        compare_baseline=None,
        save_detailed=False,
        num_gpus=2,
    )

    assert spawn_call["fn"].__name__ == "benchmark_parallel_worker"
    assert spawn_call["nprocs"] == 2
    assert spawn_call["join"] is True
    assert results["dataset_results"] == {
        "dataset_a": {"rank": 0},
        "dataset_b": {"rank": 1},
    }
    evaluator._save_benchmark_results.assert_called_once()
