"""评估结果打分、排名与指标统计（无绘图依赖）。"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np


def calculate_performance_score(metrics: Dict[str, float]) -> float:
    key_metrics = {
        "accuracy": 0.3,
        "f1": 0.25,
        "bleu": 0.2,
        "precision": 0.15,
        "recall": 0.15,
        "rouge1_f1": 0.2,
        "mean_iou": 0.25,
        "character_accuracy": 0.3,
        "word_accuracy": 0.25,
    }
    weighted_score = 0.0
    total_weight = 0.0
    for metric_name, weight in key_metrics.items():
        if metric_name in metrics:
            value = metrics[metric_name]
            if isinstance(value, (int, float)) and 0 <= value <= 1:
                weighted_score += value * weight
                total_weight += weight
    if total_weight > 0:
        return weighted_score / total_weight
    valid_values = [
        value
        for value in metrics.values()
        if isinstance(value, (int, float)) and 0 <= value <= 1
    ]
    return float(np.mean(valid_values)) if valid_values else 0.0


def calculate_difficulty_score(metrics: Dict[str, float]) -> float:
    performance_score = calculate_performance_score(metrics)
    return max(0.0, min(1.0, 1.0 - performance_score))


def extract_key_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    key_metric_names = [
        "accuracy",
        "f1",
        "bleu",
        "precision",
        "recall",
        "rouge1_f1",
        "mean_iou",
        "character_accuracy",
        "word_accuracy",
    ]
    return {name: metrics[name] for name in key_metric_names if name in metrics}


def rank_tasks_by_performance(
    task_metrics: Dict[str, Dict],
) -> List[Tuple[str, float]]:
    task_scores = [
        (task_type, calculate_performance_score(task_data["metrics"]))
        for task_type, task_data in task_metrics.items()
    ]
    task_scores.sort(key=lambda item: item[1], reverse=True)
    return task_scores


def analyze_metric_distribution(
    task_metrics: Dict[str, Dict],
) -> Dict[str, Dict[str, float]]:
    metric_values: Dict[str, List[float]] = defaultdict(list)
    for task_data in task_metrics.values():
        for metric_name, value in task_data["metrics"].items():
            if isinstance(value, (int, float)):
                metric_values[metric_name].append(value)
    distribution = {}
    for metric_name, values in metric_values.items():
        if values:
            distribution[metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "median": float(np.median(values)),
            }
    return distribution


def categorize_task_performance(task_metrics: Dict[str, Dict]) -> Dict[str, List[str]]:
    categories = {"优秀": [], "良好": [], "一般": [], "需要改进": []}
    for task_type, task_data in task_metrics.items():
        score = calculate_performance_score(task_data["metrics"])
        if score >= 0.8:
            categories["优秀"].append(task_type)
        elif score >= 0.6:
            categories["良好"].append(task_type)
        elif score >= 0.4:
            categories["一般"].append(task_type)
        else:
            categories["需要改进"].append(task_type)
    return categories


def analyze_metric_correlations(task_metrics: Dict[str, Dict]) -> Dict[str, float]:
    metric_data: Dict[str, List[float]] = defaultdict(list)
    for task_data in task_metrics.values():
        for metric_name, value in task_data["metrics"].items():
            if isinstance(value, (int, float)):
                metric_data[metric_name].append(value)
    correlations = {}
    metric_names = list(metric_data.keys())
    for index, metric1 in enumerate(metric_names):
        for metric2 in metric_names[index + 1 :]:
            if len(metric_data[metric1]) == len(metric_data[metric2]):
                corr = np.corrcoef(metric_data[metric1], metric_data[metric2])[0, 1]
                correlations[f"{metric1}_vs_{metric2}"] = float(corr)
    return correlations


def analyze_task_errors(predictions: List[Dict], task_type: str) -> Dict[str, Any]:
    errors: Dict[str, Any] = {
        "total_samples": len(predictions),
        "error_types": defaultdict(int),
        "common_mistakes": [],
    }
    for pred_data in predictions:
        prediction = pred_data.get("prediction", "")
        reference = pred_data.get("reference", "")
        if prediction == reference:
            continue
        if len(prediction) < len(reference) * 0.5:
            errors["error_types"]["too_short"] += 1
        elif len(prediction) > len(reference) * 1.5:
            errors["error_types"]["too_long"] += 1
        if not prediction.strip():
            errors["error_types"]["empty_prediction"] += 1
        if task_type in ["object_detection", "phrase_grounding"]:
            if "<loc_" not in prediction:
                errors["error_types"]["missing_location"] += 1
    return errors
