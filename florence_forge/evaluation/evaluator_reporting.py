"""Evaluator reporting helpers — bad-case export, result normalization, persistence."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def iter_result_items(
    results: Union[List[Dict[str, Any]], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Normalize evaluation results to a per-sample dict list."""
    if isinstance(results, list):
        return [item for item in results if isinstance(item, dict)]

    if not isinstance(results, dict):
        return []

    predictions = results.get("predictions")
    references = results.get("references")
    if isinstance(predictions, list) and isinstance(references, list):
        task_type = results.get("task_type")
        return [
            {
                "sample_id": idx,
                "task_type": task_type,
                "prediction": pred,
                "reference": ref,
            }
            for idx, (pred, ref) in enumerate(zip(predictions, references))
        ]

    items = results.get("items") or results.get("samples") or results.get("bad_cases")
    if isinstance(items, list):
        return [item for item in items if isinstance(item, dict)]
    return []


def infer_case_score(item: Dict[str, Any]) -> Optional[float]:
    """Infer a scalar score for bad-case filtering."""
    for key in ("score", "metric", "accuracy", "f1", "exact_match"):
        value = item.get(key)
        if isinstance(value, (int, float, bool)):
            return float(value)

    metrics = item.get("metrics")
    if isinstance(metrics, dict):
        for key in ("score", "accuracy", "f1", "exact_match", "bleu", "rouge1_f1"):
            value = metrics.get(key)
            if isinstance(value, (int, float, bool)):
                return float(value)

    prediction = item.get("prediction")
    reference = item.get("reference")
    if prediction is not None and reference is not None:
        return float(str(prediction).strip() == str(reference).strip())
    return None


def export_bad_cases_to_jsonl(
    results: Union[List[Dict[str, Any]], Dict[str, Any]],
    *,
    threshold: float = 0.5,
    output_dir: Union[str, Path] = "bad_cases",
    filename: str = "bad_cases.jsonl",
) -> Path:
    """Export low-scoring samples as JSONL for relabeling."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    bad_case_path = output_path / filename

    bad_cases = []
    for sample_id, item in enumerate(iter_result_items(results)):
        score = infer_case_score(item)
        is_bad = bool(item.get("is_bad", False)) or (
            score is not None and score <= threshold
        )
        if not is_bad:
            continue
        bad_cases.append({
            "sample_id": item.get("sample_id", sample_id),
            "task_type": item.get("task_type"),
            "prediction": item.get("prediction"),
            "reference": item.get("reference"),
            "score": score,
            "threshold": threshold,
            "metadata": item.get("metadata", {}),
        })

    with open(bad_case_path, "w", encoding="utf-8") as f:
        for item in bad_cases:
            f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")

    logger.info("已导出 %d 个 bad case 到: %s", len(bad_cases), bad_case_path)
    return bad_case_path


def save_evaluation_results(
    results: Dict[str, Any],
    predictions: Dict[str, List[Dict]],
    output_dir: Union[str, Path],
) -> None:
    """Persist evaluation metrics and per-task predictions."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    for task_type, task_predictions in predictions.items():
        task_file = output_dir / f"predictions_{task_type}.json"
        with open(task_file, "w", encoding="utf-8") as f:
            json.dump(task_predictions, f, indent=2, ensure_ascii=False)

    logger.info("评估结果已保存到: %s", output_dir)


def compute_overall_metrics(task_metrics: Dict[str, Dict]) -> Dict[str, float]:
    """Compute weighted averages across task-level metric dicts."""
    overall_metrics: Dict[str, float] = {}
    all_metric_names: set[str] = set()
    for task_data in task_metrics.values():
        all_metric_names.update(task_data["metrics"].keys())

    total_samples = sum(task_data["sample_count"] for task_data in task_metrics.values())
    for metric_name in all_metric_names:
        weighted_sum = 0.0
        valid_tasks = 0
        for _task_type, task_data in task_metrics.items():
            if metric_name in task_data["metrics"]:
                metric_value = task_data["metrics"][metric_name]
                sample_count = task_data["sample_count"]
                if isinstance(metric_value, (int, float)):
                    weighted_sum += metric_value * (sample_count / total_samples)
                    valid_tasks += 1
        if valid_tasks > 0:
            overall_metrics[f"avg_{metric_name}"] = weighted_sum

    overall_metrics["num_tasks"] = len(task_metrics)
    overall_metrics["total_samples"] = total_samples
    return overall_metrics
