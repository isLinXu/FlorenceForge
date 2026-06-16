"""Benchmark result statistics and analysis helpers."""

from __future__ import annotations

import numbers
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np


def _numeric_values(values: List[Any]) -> List[float]:
    return [float(value) for value in values if isinstance(value, numbers.Real)]


def _stats(values: List[Any]) -> Dict[str, float]:
    numeric = _numeric_values(values)
    if not numeric:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(numeric)),
        "std": float(np.std(numeric)),
        "min": float(np.min(numeric)),
        "max": float(np.max(numeric)),
    }


def enhance_results_for_report(
    results: Dict[str, Any],
    *,
    include_monitoring: bool,
    include_recommendations: bool,
) -> Dict[str, Any]:
    """Add derived analysis sections used by benchmark reports."""
    enhanced = results.copy()
    enhanced["performance_analysis"] = analyze_performance_trends(results)

    if include_monitoring and "monitoring_data" in results:
        enhanced["resource_analysis"] = analyze_resource_usage(
            results["monitoring_data"]
        )

    if include_recommendations:
        enhanced["optimization_recommendations"] = (
            generate_optimization_recommendations(results)
        )

    enhanced["statistical_summary"] = compute_statistical_summary(results)
    return enhanced


def analyze_performance_trends(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze relative task performance from aggregated metrics."""
    analysis = {
        "best_performing_tasks": [],
        "worst_performing_tasks": [],
        "performance_variance": {},
        "task_difficulty_ranking": [],
    }

    task_scores = []
    for task_type, perf in results.get("task_performance", {}).items():
        scores = []
        for stats in perf.get("average_metrics", {}).values():
            if isinstance(stats, dict) and "mean" in stats:
                scores.append(float(stats["mean"]))
            elif isinstance(stats, numbers.Real):
                scores.append(float(stats))

        if scores:
            task_scores.append(
                (task_type, float(np.mean(scores)), float(np.std(scores)))
            )

    task_scores.sort(key=lambda item: item[1], reverse=True)
    if task_scores:
        analysis["best_performing_tasks"] = task_scores[:3]
        analysis["worst_performing_tasks"] = task_scores[-3:]
        analysis["task_difficulty_ranking"] = [
            (task, score) for task, score, _ in task_scores
        ]

    return analysis


def analyze_resource_usage(monitoring_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze CPU, memory, and optional GPU monitoring samples."""
    resource_usage = monitoring_data.get("resource_usage", [])
    if not resource_usage:
        return {"error": "无监控数据"}

    analysis = {
        "cpu_stats": {},
        "memory_stats": {},
        "gpu_stats": {},
        "resource_bottlenecks": [],
        "efficiency_score": 0.0,
    }

    analysis["cpu_stats"] = _stats(
        [data.get("cpu_percent", 0.0) for data in resource_usage]
    )
    analysis["memory_stats"] = _stats(
        [data.get("memory_percent", 0.0) for data in resource_usage]
    )

    gpu_loads = []
    gpu_memory = []
    for sample in resource_usage:
        for gpu in sample.get("gpu_info", []) or []:
            gpu_loads.append(float(gpu.get("load", 0.0)))
            memory_total = float(gpu.get("memory_total", 0.0) or 0.0)
            if memory_total > 0:
                gpu_memory.append(
                    float(gpu.get("memory_used", 0.0)) / memory_total * 100
                )

    if gpu_loads:
        analysis["gpu_stats"] = {
            "load_mean": float(np.mean(gpu_loads)),
            "load_max": float(np.max(gpu_loads)),
            "memory_mean": float(np.mean(gpu_memory)) if gpu_memory else 0.0,
            "memory_max": float(np.max(gpu_memory)) if gpu_memory else 0.0,
        }

    if analysis["cpu_stats"]["mean"] > 80:
        analysis["resource_bottlenecks"].append("CPU使用率过高")
    if analysis["memory_stats"]["mean"] > 85:
        analysis["resource_bottlenecks"].append("内存使用率过高")
    if analysis["gpu_stats"] and analysis["gpu_stats"].get("memory_mean", 0) > 90:
        analysis["resource_bottlenecks"].append("GPU内存使用率过高")

    cpu_efficiency = min(analysis["cpu_stats"]["mean"] / 100, 1.0)
    memory_efficiency = min(analysis["memory_stats"]["mean"] / 100, 1.0)
    gpu_efficiency = 1.0
    if analysis["gpu_stats"]:
        gpu_efficiency = min(analysis["gpu_stats"]["load_mean"] / 100, 1.0)

    analysis["efficiency_score"] = float(
        (cpu_efficiency + memory_efficiency + gpu_efficiency) / 3
    )
    return analysis


def generate_optimization_recommendations(results: Dict[str, Any]) -> List[str]:
    """Generate benchmark optimization recommendations from results."""
    recommendations = []

    task_performance = results.get("task_performance", {})
    for task_type, perf in task_performance.items():
        avg_metrics = perf.get("average_metrics", {})

        if "bleu" in str(task_type).lower():
            bleu_scores = [
                stats.get("mean", 0)
                for metric, stats in avg_metrics.items()
                if "bleu" in metric.lower() and isinstance(stats, dict)
            ]
            if bleu_scores and max(bleu_scores) < 0.3:
                recommendations.append(
                    f"考虑为{task_type}任务增加训练数据或调整模型参数以提高BLEU分数"
                )

        if "detection" in str(task_type).lower():
            map_scores = [
                stats.get("mean", 0)
                for metric, stats in avg_metrics.items()
                if "map" in metric.lower() and isinstance(stats, dict)
            ]
            if map_scores and max(map_scores) < 0.5:
                recommendations.append(
                    f"考虑为{task_type}任务调整检测阈值或增强数据增广"
                )

    if "monitoring_data" in results:
        resource_analysis = analyze_resource_usage(results["monitoring_data"])
        if resource_analysis.get("efficiency_score", 1.0) < 0.6:
            recommendations.append("系统资源利用率较低，考虑增加批处理大小或并行度")
        if "CPU使用率过高" in resource_analysis.get("resource_bottlenecks", []):
            recommendations.append(
                "CPU使用率过高，考虑减少数据预处理的复杂度或使用更多GPU"
            )
        if "GPU内存使用率过高" in resource_analysis.get("resource_bottlenecks", []):
            recommendations.append(
                "GPU内存使用率过高，考虑减少批处理大小或使用梯度累积"
            )

    total_time = results.get("benchmark_info", {}).get("total_evaluation_time", 0)
    if total_time > 3600:
        recommendations.append("评估时间较长，考虑使用增量评估或多GPU并行评估")

    return recommendations


def compute_statistical_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    """Compute high-level benchmark summary statistics."""
    summary = {
        "total_metrics_computed": 0,
        "average_performance_score": 0.0,
        "performance_consistency": 0.0,
        "evaluation_efficiency": 0.0,
    }

    total_metrics = 0
    all_scores = []
    for perf in results.get("task_performance", {}).values():
        avg_metrics = perf.get("average_metrics", {})
        total_metrics += len(avg_metrics)
        for stats in avg_metrics.values():
            if isinstance(stats, dict) and "mean" in stats:
                all_scores.append(float(stats["mean"]))
            elif isinstance(stats, numbers.Real):
                all_scores.append(float(stats))

    summary["total_metrics_computed"] = total_metrics
    if all_scores:
        mean_score = float(np.mean(all_scores))
        std_score = float(np.std(all_scores))
        summary["average_performance_score"] = mean_score
        summary["performance_consistency"] = (
            1.0 - (std_score / mean_score) if mean_score else 0.0
        )

    total_time = float(
        results.get("benchmark_info", {}).get("total_evaluation_time", 1) or 1
    )
    total_samples = float(
        results.get("overall_summary", {}).get("total_samples", 1) or 0
    )
    summary["evaluation_efficiency"] = total_samples / total_time
    return summary


def compute_overall_summary(
    dataset_results: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate dataset-level benchmark results."""
    summary = {
        "total_datasets": len(dataset_results),
        "total_samples": 0,
        "total_evaluation_time": 0.0,
        "average_metrics": {},
        "best_performance": {},
        "worst_performance": {},
    }

    all_metrics = defaultdict(list)
    for dataset_name, result in dataset_results.items():
        dataset_info = result.get("dataset_info", {})
        summary["total_samples"] += int(dataset_info.get("total_samples", 0) or 0)
        summary["total_evaluation_time"] += float(
            result.get("evaluation_time", 0.0) or 0.0
        )

        for metric_name, value in result.get("overall_metrics", {}).items():
            if isinstance(value, numbers.Real):
                all_metrics[metric_name].append((dataset_name, float(value)))

    for metric_name, values in all_metrics.items():
        if not values:
            continue
        metric_values = [value for _, value in values]
        summary["average_metrics"][metric_name] = float(np.mean(metric_values))

        best_idx = int(np.argmax(metric_values))
        worst_idx = int(np.argmin(metric_values))
        summary["best_performance"][metric_name] = {
            "dataset": values[best_idx][0],
            "value": values[best_idx][1],
        }
        summary["worst_performance"][metric_name] = {
            "dataset": values[worst_idx][0],
            "value": values[worst_idx][1],
        }

    return summary


def compute_task_performance(
    dataset_results: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate per-task benchmark metrics across datasets."""
    task_stats = defaultdict(
        lambda: {
            "datasets": [],
            "total_samples": 0,
            "metrics": defaultdict(list),
        }
    )

    for dataset_name, result in dataset_results.items():
        for task_type, task_data in result.get("metrics", {}).items():
            task_stats[task_type]["datasets"].append(dataset_name)
            task_stats[task_type]["total_samples"] += int(
                task_data.get("sample_count", 0) or 0
            )
            for metric_name, value in task_data.get("metrics", {}).items():
                if isinstance(value, numbers.Real):
                    task_stats[task_type]["metrics"][metric_name].append(float(value))

    task_performance = {}
    for task_type, stats in task_stats.items():
        task_performance[task_type] = {
            "participating_datasets": len(stats["datasets"]),
            "total_samples": stats["total_samples"],
            "average_metrics": {},
        }
        for metric_name, values in stats["metrics"].items():
            if values:
                task_performance[task_type]["average_metrics"][metric_name] = _stats(
                    values
                )

    return task_performance


def compare_with_baseline(
    current_results: Dict[str, Any],
    baseline_results: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare current benchmark results with a baseline result payload."""
    comparison = {
        "overall_improvement": {},
        "task_improvements": {},
        "dataset_improvements": {},
    }

    current_overall = current_results.get("overall_summary", {}).get(
        "average_metrics", {}
    )
    baseline_overall = baseline_results.get("overall_summary", {}).get(
        "average_metrics", {}
    )
    for metric_name in current_overall:
        if metric_name not in baseline_overall:
            continue
        current_val = float(current_overall[metric_name])
        baseline_val = float(baseline_overall[metric_name])
        improvement = current_val - baseline_val
        comparison["overall_improvement"][metric_name] = {
            "absolute": improvement,
            "relative": improvement / baseline_val if baseline_val != 0 else 0,
            "current": current_val,
            "baseline": baseline_val,
        }

    current_tasks = current_results.get("task_performance", {})
    baseline_tasks = baseline_results.get("task_performance", {})
    for task_type in current_tasks:
        if task_type in baseline_tasks:
            comparison["task_improvements"][task_type] = compare_task_metrics(
                current_tasks[task_type].get("average_metrics", {}),
                baseline_tasks[task_type].get("average_metrics", {}),
            )

    return comparison


def compare_task_metrics(
    current_metrics: Dict[str, Dict[str, float]],
    baseline_metrics: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """Compare task-level mean metrics."""
    comparison = {}
    for metric_name in current_metrics:
        if metric_name not in baseline_metrics:
            continue
        current_mean = float(current_metrics[metric_name].get("mean", 0))
        baseline_mean = float(baseline_metrics[metric_name].get("mean", 0))
        improvement = current_mean - baseline_mean
        comparison[metric_name] = {
            "absolute": improvement,
            "relative": improvement / baseline_mean if baseline_mean != 0 else 0,
            "current": current_mean,
            "baseline": baseline_mean,
        }

    return comparison
