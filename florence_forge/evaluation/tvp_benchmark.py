"""TVP benchmark evaluation helpers for FlorenceForge."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from .tvp_metrics import TVPCompositeMetric

logger = logging.getLogger(__name__)

_BASE_TASK_TO_METRIC = {
    "counting": "counting",
    "od": "counting",
    "od_vp": "counting",
    "phrase_grounding_vp": "phrase_grounding_vp",
    "spatial": "spatial",
    "maze": "maze",
    "path": "path",
}


def load_tvp_jsonl_records(path: str | Path) -> List[Dict[str, Any]]:
    """Load JSONL records for TVP benchmark evaluation."""
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def resolve_tvp_task_type(record: Dict[str, Any]) -> str:
    """Resolve the TVP metric task type from a dataset record."""
    explicit = str(record.get("vp_task_type") or record.get("task_type") or "").strip()
    if explicit:
        return explicit.lower()
    base_task = str(record.get("base_task") or "").strip().lower()
    return _BASE_TASK_TO_METRIC.get(base_task, base_task or "counting")


def build_tvp_metric_kwargs(record: Dict[str, Any]) -> Dict[str, Any]:
    """Build keyword arguments for :class:`TVPCompositeMetric`."""
    kwargs: Dict[str, Any] = {}

    if "count" in record:
        kwargs["gt_count"] = record["count"]
    if "gt_boxes" in record:
        kwargs["gt_boxes"] = record["gt_boxes"]
    if "gt_points" in record:
        kwargs["gt_points"] = [tuple(point) for point in record["gt_points"]]
    if "points" in record and "gt_points" not in record:
        kwargs["gt_points"] = [tuple(point) for point in record["points"]]
    if "solvable" in record:
        kwargs["solvable"] = record["solvable"]
    if "grid" in record:
        kwargs["grid"] = record["grid"]
    if "grid_height" in record:
        kwargs["grid_height"] = record["grid_height"]
    if "grid_width" in record:
        kwargs["grid_width"] = record["grid_width"]
    if "solution_points" in record:
        kwargs["gt_path"] = [tuple(point) for point in record["solution_points"]]
    if "answer" in record:
        kwargs["gt_answer"] = record["answer"]
    if "end_label" in record:
        kwargs["gt_label"] = record["end_label"]
    return kwargs


def evaluate_tvp_predictions(
    records: Sequence[Dict[str, Any]],
    predictions: Sequence[str],
    *,
    metric: Optional[TVPCompositeMetric] = None,
) -> Dict[str, Any]:
    """Evaluate model predictions against TVP JSONL records."""
    if len(records) != len(predictions):
        raise ValueError(
            f"records and predictions must have the same length: "
            f"{len(records)} != {len(predictions)}"
        )

    metric = metric or TVPCompositeMetric()
    per_sample: List[Dict[str, Any]] = []
    grouped_scores: Dict[str, List[float]] = defaultdict(list)
    grouped_metric_sums: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    grouped_metric_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for index, (record, prediction) in enumerate(zip(records, predictions)):
        task_type = resolve_tvp_task_type(record)
        metric_kwargs = build_tvp_metric_kwargs(record)
        scores = metric.compute(prediction, task_type=task_type, **metric_kwargs)
        sample_result = {
            "index": index,
            "task_type": task_type,
            "composite": scores.get("composite", 0.0),
            "metrics": scores,
        }
        per_sample.append(sample_result)
        grouped_scores[task_type].append(scores.get("composite", 0.0))
        for metric_name, metric_value in scores.items():
            if isinstance(metric_value, (int, float)):
                grouped_metric_sums[task_type][metric_name] += float(metric_value)
                grouped_metric_counts[task_type][metric_name] += 1

    task_metrics: Dict[str, Any] = {}
    for task_type, composites in grouped_scores.items():
        task_metrics[task_type] = {
            "sample_count": len(composites),
            "composite_mean": sum(composites) / max(len(composites), 1),
            "metrics_mean": {
                metric_name: grouped_metric_sums[task_type][metric_name]
                / max(grouped_metric_counts[task_type][metric_name], 1)
                for metric_name in grouped_metric_sums[task_type]
            },
        }

    overall_composites = [
        sample["composite"] for sample in per_sample
    ]
    return {
        "overall_metrics": {
            "composite_mean": sum(overall_composites) / max(len(overall_composites), 1),
            "sample_count": len(per_sample),
        },
        "task_metrics": task_metrics,
        "samples": per_sample,
    }


def run_tvp_benchmark(
    model: Any,
    data_path: str | Path,
    *,
    max_samples: Optional[int] = None,
    predict_fn: Optional[Callable[[Any, Dict[str, Any]], str]] = None,
) -> Dict[str, Any]:
    """Run end-to-end TVP benchmark inference + metric aggregation."""
    records = load_tvp_jsonl_records(data_path)
    if max_samples is not None:
        records = records[:max_samples]

    predictions: List[str] = []
    if predict_fn is None:
        predict_fn = _default_tvp_predict

    for record in records:
        predictions.append(predict_fn(model, record))

    results = evaluate_tvp_predictions(records, predictions)
    results["evaluation_info"] = {
        "data_path": str(Path(data_path).absolute()),
        "sample_count": len(records),
        "benchmark": "tvp",
    }
    return results


def _default_tvp_predict(model: Any, record: Dict[str, Any]) -> str:
    """Generate one prediction for a TVP record using a FlorenceForge model."""
    from PIL import Image

    image_path = Path(str(record.get("image", "")))
    if not image_path.exists():
        raise FileNotFoundError(f"TVP benchmark image not found: {image_path}")

    prefix = str(record.get("prefix", "<OD>"))
    text_input = record.get("text_input")
    task_name = str(record.get("vp_task_type") or record.get("task_type") or "OD_VP")

    image = Image.open(image_path).convert("RGB")
    if hasattr(model, "predict_task"):
        result = model.predict_task(
            image,
            task_name=task_name,
            text_input=text_input,
        )
        if isinstance(result, dict):
            return str(result.get("generated_text") or result.get("text") or result)
        return str(result)

    if hasattr(model, "generate"):
        generated = model.generate(image=image, task_prompt=prefix, text_input=text_input)
        if isinstance(generated, list) and generated:
            return str(generated[0])
        return str(generated)

    raise TypeError(
        f"Model {type(model).__name__} does not implement predict_task() or generate()"
    )
