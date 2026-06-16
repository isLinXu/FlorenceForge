"""Parallel benchmark execution helpers."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.multiprocessing as torch_mp

from ..data.dataset import MultiTaskDataset
from .benchmark_cache import BenchmarkCache, load_benchmark_artifact_cpu
from .evaluator import MultiTaskEvaluator

logger = logging.getLogger(__name__)


@dataclass
class ParallelBenchmarkRun:
    """Collected dataset results from a parallel benchmark run."""

    dataset_results: Dict[str, Any]
    worker_count: int


def parallel_result_file(result_dir: Union[str, Path], rank: int) -> Path:
    """Return the temporary result file path for a parallel worker rank."""
    return Path(result_dir) / f"worker_{rank}.pt"


def benchmark_parallel_worker(rank: int, payload: Dict[str, Any]) -> None:
    """Spawn worker for multi-GPU benchmark evaluation.

    The parent process passes a CPU model template and per-rank dataset
    assignment. Each spawned process creates its own CUDA context and moves its
    local model copy to the assigned GPU.
    """
    result_file = parallel_result_file(payload["result_dir"], rank)
    result_file.parent.mkdir(parents=True, exist_ok=True)

    assignments = payload["assignments"]
    dataset_items = assignments[rank]
    if not dataset_items:
        torch.save({}, result_file)
        return

    gpu_id = payload["gpu_ids"][rank]
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)
    else:
        logger.warning("CUDA 不可用，parallel benchmark worker %s 回退到 CPU", rank)
        device = torch.device("cpu")

    model_copy = copy.deepcopy(payload["model_template"])
    model_copy = model_copy.to(device)
    model_copy.eval()

    evaluator = MultiTaskEvaluator(model_copy, device=device)
    cache_helper = BenchmarkCache(
        cache_dir=payload["cache_dir"],
        config=payload["config"],
        enable_incremental=payload["enable_incremental"],
    )

    output_path = Path(payload["output_path"])
    results = {}
    for dataset_name, dataset in dataset_items:
        logger.info("GPU %s 评估数据集: %s", gpu_id, dataset_name)
        cache_key = BenchmarkCache.make_key(
            dataset_name,
            "mixed",
            {"model_name": getattr(model_copy, "model_name", "Florence2"), "gpu_id": gpu_id},
        )

        cached_result = cache_helper.load_results(cache_key)
        if cached_result:
            results[dataset_name] = cached_result
            logger.info("GPU %s 使用缓存结果: %s", gpu_id, dataset_name)
            continue

        dataset_result = evaluator.evaluate_dataset(
            dataset,
            batch_size=payload["config"].get("batch_size", 8),
            num_workers=payload["config"].get("num_workers", 4),
            max_samples_per_task=payload["config"].get("max_samples_per_task"),
            save_predictions=payload["config"].get("save_predictions", False),
            output_dir=output_path / dataset_name,
        )

        cache_helper.save_results(cache_key, dataset_result)
        results[dataset_name] = dataset_result

    torch.save(results, result_file)


def resolve_parallel_worker_count(
    dataset_count: int,
    num_gpus: Optional[int] = None,
) -> int:
    """Resolve worker count for parallel benchmark execution."""
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    if num_gpus <= 0 and dataset_count:
        raise ValueError("并行 benchmark 需要至少 1 个 GPU")

    available_gpus = torch.cuda.device_count()
    if available_gpus > 0 and num_gpus > available_gpus:
        logger.warning("请求使用 %s 个 GPU，但当前仅检测到 %s 个，将自动收敛", num_gpus, available_gpus)
        num_gpus = available_gpus

    if not dataset_count:
        logger.warning("没有可评估的数据集，返回空 benchmark 结果")
        return 0

    return min(num_gpus, dataset_count)


def run_parallel_dataset_evaluation(
    *,
    datasets: Dict[str, MultiTaskDataset],
    output_path: Path,
    cache_dir: Union[str, Path],
    config: Dict[str, Any],
    enable_incremental: bool,
    model_template_factory: Callable[[], nn.Module],
    num_gpus: Optional[int] = None,
) -> ParallelBenchmarkRun:
    """Run dataset evaluation in spawned workers and collect results."""
    dataset_items = list(datasets.items())
    worker_count = resolve_parallel_worker_count(len(dataset_items), num_gpus)
    logger.info("开始多GPU并行benchmark评估，使用%s个GPU", worker_count)

    dataset_results: Dict[str, Any] = {}
    if worker_count <= 0:
        return ParallelBenchmarkRun(dataset_results=dataset_results, worker_count=worker_count)

    assignments: List[List[Tuple[str, MultiTaskDataset]]] = [
        [] for _ in range(worker_count)
    ]
    for idx, item in enumerate(dataset_items):
        assignments[idx % worker_count].append(item)

    result_dir = output_path / ".parallel_results"
    result_dir.mkdir(parents=True, exist_ok=True)
    for stale_file in result_dir.glob("worker_*.pt"):
        try:
            stale_file.unlink()
        except OSError:
            logger.debug("无法清理旧 parallel benchmark 结果文件: %s", stale_file, exc_info=True)

    payload = {
        "assignments": assignments,
        "cache_dir": str(cache_dir),
        "config": config,
        "enable_incremental": enable_incremental,
        "gpu_ids": list(range(worker_count)),
        "model_template": model_template_factory(),
        "output_path": str(output_path),
        "result_dir": str(result_dir),
    }

    torch_mp.spawn(
        benchmark_parallel_worker,
        args=(payload,),
        nprocs=worker_count,
        join=True,
    )

    for rank in range(worker_count):
        result_file = parallel_result_file(result_dir, rank)
        if not result_file.exists():
            raise RuntimeError(f"并行 benchmark worker {rank} 未写出结果文件: {result_file}")
        gpu_results = load_benchmark_artifact_cpu(result_file)
        dataset_results.update(gpu_results)
        try:
            result_file.unlink()
        except OSError:
            logger.debug("无法删除 parallel benchmark 临时结果文件: %s", result_file, exc_info=True)
    try:
        result_dir.rmdir()
    except OSError:
        pass

    return ParallelBenchmarkRun(dataset_results=dataset_results, worker_count=worker_count)
