"""Benchmark statistics helper regression tests."""

import json

import numpy as np
import pytest

from florence_forge.evaluation.benchmark_statistics import (
    analyze_resource_usage,
    compare_with_baseline,
    compute_overall_summary,
    compute_statistical_summary,
    compute_task_performance,
)


def test_compute_overall_summary_returns_json_serializable_values():
    summary = compute_overall_summary(
        {
            "coco-mini": {
                "dataset_info": {"total_samples": 10},
                "evaluation_time": 2.5,
                "overall_metrics": {"accuracy": np.float64(0.8), "count": np.int64(2)},
            },
            "ocr-mini": {
                "dataset_info": {"total_samples": 5},
                "evaluation_time": 1.5,
                "overall_metrics": {"accuracy": 0.6, "count": np.int64(4)},
            },
        }
    )

    assert summary["total_datasets"] == 2
    assert summary["total_samples"] == 15
    assert summary["average_metrics"]["accuracy"] == 0.7
    assert summary["average_metrics"]["count"] == 3.0
    assert summary["best_performance"]["accuracy"] == {
        "dataset": "coco-mini",
        "value": 0.8,
    }
    json.dumps(summary)


def test_compute_task_performance_aggregates_metrics_across_datasets():
    performance = compute_task_performance(
        {
            "dataset-a": {
                "metrics": {
                    "CAPTION": {
                        "sample_count": 3,
                        "metrics": {"bleu_4": 0.2, "rouge": 0.4},
                    }
                }
            },
            "dataset-b": {
                "metrics": {
                    "CAPTION": {
                        "sample_count": 2,
                        "metrics": {"bleu_4": 0.4, "rouge": 0.6},
                    }
                }
            },
        }
    )

    caption = performance["CAPTION"]
    assert caption["participating_datasets"] == 2
    assert caption["total_samples"] == 5
    assert caption["average_metrics"]["bleu_4"]["mean"] == pytest.approx(0.3)
    assert caption["average_metrics"]["rouge"]["max"] == 0.6
    json.dumps(performance)


def test_statistical_summary_handles_zero_mean_scores():
    summary = compute_statistical_summary(
        {
            "benchmark_info": {"total_evaluation_time": 2.0},
            "overall_summary": {"total_samples": 4},
            "task_performance": {
                "CAPTION": {
                    "average_metrics": {
                        "bleu_4": {"mean": 0.0},
                        "rouge": {"mean": 0.0},
                    }
                }
            },
        }
    )

    assert summary["total_metrics_computed"] == 2
    assert summary["average_performance_score"] == 0.0
    assert summary["performance_consistency"] == 0.0
    assert summary["evaluation_efficiency"] == 2.0


def test_analyze_resource_usage_handles_zero_gpu_memory_total():
    analysis = analyze_resource_usage(
        {
            "resource_usage": [
                {
                    "cpu_percent": 90.0,
                    "memory_percent": 50.0,
                    "gpu_info": [
                        {
                            "load": 75.0,
                            "memory_used": 100.0,
                            "memory_total": 0.0,
                        }
                    ],
                }
            ]
        }
    )

    assert analysis["cpu_stats"]["mean"] == 90.0
    assert analysis["gpu_stats"]["load_mean"] == 75.0
    assert analysis["gpu_stats"]["memory_mean"] == 0.0
    assert "CPU使用率过高" in analysis["resource_bottlenecks"]


def test_compare_with_baseline_computes_overall_and_task_improvements():
    current = {
        "overall_summary": {"average_metrics": {"accuracy": 0.75}},
        "task_performance": {"CAPTION": {"average_metrics": {"bleu_4": {"mean": 0.3}}}},
    }
    baseline = {
        "overall_summary": {"average_metrics": {"accuracy": 0.5}},
        "task_performance": {"CAPTION": {"average_metrics": {"bleu_4": {"mean": 0.2}}}},
    }

    comparison = compare_with_baseline(current, baseline)

    assert comparison["overall_improvement"]["accuracy"]["absolute"] == 0.25
    assert comparison["overall_improvement"]["accuracy"]["relative"] == 0.5
    assert comparison["task_improvements"]["CAPTION"]["bleu_4"][
        "absolute"
    ] == pytest.approx(0.1)
