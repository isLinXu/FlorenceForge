"""VP core helpers — unified parsing + aggregation layer.

Merged from ``vp_parsing.py`` and ``vp_aggregation.py`` (P1-1 evaluation layer
refactoring).  Provides a single source of truth for all visual-primitive
quality-evaluation helpers: parsing, indexing, field resolution, metric
computation, comparison, and summarization.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from .structured_vp_decoder import (
    StructuredVisualPrimitiveDecoder,
    resolve_structured_vp_filter_caps,
)
from .visual_primitive_parser import VisualPrimitiveParser


# ── Prediction parsing ────────────────────────────────────────────────────


def parse_prediction(
    text: str,
    *,
    parser: VisualPrimitiveParser,
    decoder: StructuredVisualPrimitiveDecoder,
    config: Any,  # VPDetectionQualityConfig – avoided circular import
    record: Mapping[str, Any],
    task_prompt: Any,
    allowed_labels: Optional[Union[Sequence[str], str]],
) -> Tuple[List[Dict[str, Any]], str, int, int, int]:
    """Parse a VP prediction using either the raw parser or structured decoder."""
    if not config.use_structured_decoder:
        return parser.parse_detections(text), "visual_primitive_raw", 0, 0, 0

    filter_caps = resolve_structured_vp_filter_caps(
        policy=config.filter_policy,
        task_prompt=task_prompt,
        max_boxes_per_label=config.max_boxes_per_label,
        max_total_boxes=resolve_record_positive_int_field(
            record,
            config.max_total_boxes_field,
            fallback=config.max_total_boxes,
        ),
        nms_iou_threshold=config.nms_iou_threshold,
        allowed_labels=allowed_labels,
    )
    decoded = decoder.decode(
        text,
        max_boxes_per_label=filter_caps["max_boxes_per_label"],
        max_total_boxes=filter_caps["max_total_boxes"],
        nms_iou_threshold=filter_caps["nms_iou_threshold"],
        allowed_labels=filter_caps["allowed_labels"],
        allowed_label_match_mode=config.allowed_label_match_mode or config.label_match_mode,
        repair_malformed_tail=config.repair_malformed_tail,
    )
    return (
        decoded.detections,
        decoded.source,
        decoded.raw_detection_count,
        decoded.filtered_detection_count,
        decoded.repaired_tail_detection_count,
    )


# ── Record indexing ──────────────────────────────────────────────────────


def summary_records(summary: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    """Extract valid records from a summary dict."""
    records = summary.get("records", [])
    if not isinstance(records, Sequence):
        return []
    return [record for record in records if isinstance(record, Mapping)]


def index_quality_records(records: Any) -> Dict[str, Mapping[str, Any]]:
    """Index quality records by a composite key."""
    if not isinstance(records, Sequence):
        return {}

    indexed: Dict[str, Mapping[str, Any]] = {}
    for position, record in enumerate(records):
        if not isinstance(record, Mapping):
            continue
        key = quality_record_key(record)
        if key in indexed:
            key = f"{key}#pos={position}"
        indexed[key] = record
    return indexed


def quality_record_key(record: Mapping[str, Any]) -> str:
    """Build a composite key for a quality record."""
    image = record.get("image")
    image_key = Path(str(image)).name if image else ""
    label_value = record.get("allowed_labels")
    if isinstance(label_value, Sequence) and not isinstance(label_value, (str, bytes)):
        label_key = ",".join(str(item).strip() for item in label_value)
    else:
        label_key = str(label_value or "")
    return "|".join([
        str(record.get("index", "")),
        image_key,
        label_key,
        str(record.get("query_box_count", record.get("gt_box_count", ""))),
    ])


# ── Record field resolution ──────────────────────────────────────────────


def record_text(record: Mapping[str, Any], field: str) -> str:
    """Extract text from a record, with fallback fields."""
    value = record.get(field)
    if isinstance(value, str):
        return value
    fallback_fields = ("raw_prediction", "prediction", "structured_prediction", "target")
    for fallback in fallback_fields:
        value = record.get(fallback)
        if isinstance(value, str):
            return value
    return ""


def resolve_record_allowed_labels(
    *,
    config: Any,  # VPDetectionQualityConfig – avoided circular import
    record: Mapping[str, Any],
    reference_detections: Sequence[Mapping[str, Any]],
) -> Optional[Union[Sequence[str], str]]:
    """Resolve the effective allowed labels for a record."""
    if config.allowed_labels is not None:
        return config.allowed_labels
    if not config.allowed_labels_field:
        return None

    for field in allowed_label_field_candidates(config.allowed_labels_field):
        normalized_field = field.strip().lower()
        if normalized_field in {"target_labels", "reference_labels", "gt_labels"}:
            labels = [
                str(detection.get("label", "")).strip()
                for detection in reference_detections
                if str(detection.get("label", "")).strip()
            ]
            return labels or None
        value = record_field_value(record, field)
        if value not in (None, ""):
            return value
    return None


def allowed_label_field_candidates(value: str) -> List[str]:
    """Split a comma/pipe/semicolon-separated field spec into candidates."""
    return [
        item.strip()
        for item in str(value or "").replace("|", ",").replace(";", ",").split(",")
        if item.strip()
    ]


def record_field_value(record: Mapping[str, Any], field: str) -> Any:
    """Resolve a dot-separated field path from a nested record."""
    current: Any = record
    for part in str(field).split("."):
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current


def resolve_record_positive_int_field(
    record: Mapping[str, Any],
    field_spec: Optional[str],
    *,
    fallback: Optional[int] = None,
) -> Optional[int]:
    """Resolve a positive integer cap from a per-record field specification."""
    if field_spec:
        for field in allowed_label_field_candidates(field_spec):
            value = record_field_value(record, field)
            try:
                if value is not None and str(value).strip() != "":
                    parsed = int(value)
                    if parsed >= 1:
                        return parsed
            except (TypeError, ValueError):
                continue
    if fallback is None:
        return None
    parsed_fallback = int(fallback)
    return parsed_fallback if parsed_fallback >= 1 else None


# ── Bad-case diagnostics ────────────────────────────────────────────────


def bad_case_reasons(record: Mapping[str, Any]) -> List[str]:
    """Identify reasons why a record is a bad case."""
    reasons: List[str] = []
    if int(record.get("pred_box_count", 0) or 0) == 0 and int(record.get("gt_box_count", 0) or 0) > 0:
        reasons.append("no_prediction")
    if int(record.get("false_positives", 0) or 0) > 0:
        reasons.append("false_positive")
    if int(record.get("false_negatives", 0) or 0) > 0:
        reasons.append("false_negative")
    if bool(record.get("overgenerated")):
        reasons.append("overgenerated")
    if bool(record.get("undergenerated")):
        reasons.append("undergenerated")
    return reasons


# ── Box count utilities ─────────────────────────────────────────────────


def record_query_box_count(record: Mapping[str, Any], fallback: int) -> int:
    """Resolve the query box count from a record."""
    for key in ("query_box_count", "curriculum_query_box_count", "gt_box_count"):
        value = record.get(key)
        try:
            if value is not None:
                return max(0, int(value))
        except (TypeError, ValueError):
            pass
    return max(0, int(fallback))


def box_count_bucket(box_count: Any) -> str:
    """Classify a box count into a bucket: single, medium, or dense."""
    try:
        parsed = int(box_count)
    except (TypeError, ValueError):
        parsed = 0
    if parsed <= 1:
        return "single"
    if parsed <= 3:
        return "medium"
    return "dense"


# ── Generic utilities ───────────────────────────────────────────────────


def normalize_label(value: Any) -> str:
    """Normalize a label string."""
    return " ".join(str(value or "").strip().lower().split())


def is_box(value: Any) -> bool:
    """Check if a value looks like a bounding box [x1, y1, x2, y2]."""
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 4
        and all(isinstance(coord, (int, float)) for coord in value)
    )


# ════════════════════════════════════════════════════════════════════════
#  Aggregation helpers
# ════════════════════════════════════════════════════════════════════════


# ── Core metric computation ─────────────────────────────────────────────


def aggregate_counts(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """Compute precision, recall, and F1 from raw counts."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def record_f1(tp: int, fp: int, fn: int) -> float:
    """Compute F1 score from raw counts."""
    return aggregate_counts(tp, fp, fn)["f1"]


def int_record_metric(record: Mapping[str, Any], key: str) -> int:
    """Safely extract an integer metric from a record."""
    try:
        return int(record.get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0


def safe_policy_label(value: str) -> str:
    """Sanitize a policy name for use in file paths and identifiers."""
    cleaned = "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "_"
        for ch in str(value or "").strip()
    )
    return cleaned or "policy"


# ── Statistical helpers ─────────────────────────────────────────────────


def mean(values: Iterable[float]) -> float:
    """Compute the mean of a sequence of floats, returning 0.0 for empty."""
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def ratio(records: Sequence[Mapping[str, Any]], key: str) -> float:
    """Compute the ratio of records where *key* is truthy."""
    if not records:
        return 0.0
    return sum(1 for record in records if bool(record.get(key))) / len(records)


# ── Quality report brief ────────────────────────────────────────────────


def quality_report_brief(report: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract a compact summary from a full quality report."""
    return {
        "num_samples": int(report.get("num_samples", 0) or 0),
        "precision": float(report.get("precision", 0.0) or 0.0),
        "recall": float(report.get("recall", 0.0) or 0.0),
        "f1": float(report.get("f1", 0.0) or 0.0),
        "true_positives": int(report.get("true_positives", 0) or 0),
        "false_positives": int(report.get("false_positives", 0) or 0),
        "false_negatives": int(report.get("false_negatives", 0) or 0),
        "avg_pred_boxes": float(report.get("avg_pred_boxes", 0.0) or 0.0),
        "avg_gt_boxes": float(report.get("avg_gt_boxes", 0.0) or 0.0),
    }


# ── Record comparison ───────────────────────────────────────────────────


def compare_quality_record(
    key: str,
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
    *,
    bucket: str,
) -> Dict[str, Any]:
    """Compare a candidate quality record against a baseline."""
    candidate_tp = int_record_metric(candidate, "true_positives")
    candidate_fp = int_record_metric(candidate, "false_positives")
    candidate_fn = int_record_metric(candidate, "false_negatives")
    baseline_tp = int_record_metric(baseline, "true_positives")
    baseline_fp = int_record_metric(baseline, "false_positives")
    baseline_fn = int_record_metric(baseline, "false_negatives")
    candidate_pred = int_record_metric(candidate, "pred_box_count")
    baseline_pred = int_record_metric(baseline, "pred_box_count")
    delta_tp = candidate_tp - baseline_tp
    delta_fp = candidate_fp - baseline_fp
    delta_fn = candidate_fn - baseline_fn
    delta_pred = candidate_pred - baseline_pred
    baseline_f1 = record_f1(baseline_tp, baseline_fp, baseline_fn)
    candidate_f1 = record_f1(candidate_tp, candidate_fp, candidate_fn)
    row = {
        "record_key": key,
        "index": candidate.get("index", baseline.get("index")),
        "image": candidate.get("image", baseline.get("image")),
        "allowed_labels": candidate.get("allowed_labels", baseline.get("allowed_labels")),
        "box_count_bucket": bucket,
        "gt_box_count": candidate.get("gt_box_count", baseline.get("gt_box_count")),
        "query_box_count": candidate.get("query_box_count", baseline.get("query_box_count")),
        "baseline_pred_box_count": baseline_pred,
        "candidate_pred_box_count": candidate_pred,
        "delta_pred_box_count": delta_pred,
        "baseline_raw_detection_count": int_record_metric(baseline, "raw_detection_count"),
        "candidate_raw_detection_count": int_record_metric(candidate, "raw_detection_count"),
        "delta_raw_detection_count": (
            int_record_metric(candidate, "raw_detection_count")
            - int_record_metric(baseline, "raw_detection_count")
        ),
        "baseline_filtered_detection_count": int_record_metric(baseline, "filtered_detection_count"),
        "candidate_filtered_detection_count": int_record_metric(candidate, "filtered_detection_count"),
        "delta_filtered_detection_count": (
            int_record_metric(candidate, "filtered_detection_count")
            - int_record_metric(baseline, "filtered_detection_count")
        ),
        "baseline_true_positives": baseline_tp,
        "candidate_true_positives": candidate_tp,
        "delta_true_positives": delta_tp,
        "baseline_false_positives": baseline_fp,
        "candidate_false_positives": candidate_fp,
        "delta_false_positives": delta_fp,
        "baseline_false_negatives": baseline_fn,
        "candidate_false_negatives": candidate_fn,
        "delta_false_negatives": delta_fn,
        "baseline_f1": baseline_f1,
        "candidate_f1": candidate_f1,
        "delta_f1": candidate_f1 - baseline_f1,
        "baseline_mean_matched_iou": float(baseline.get("mean_matched_iou", 0.0) or 0.0),
        "candidate_mean_matched_iou": float(candidate.get("mean_matched_iou", 0.0) or 0.0),
        "baseline_undergenerated": bool(baseline.get("undergenerated")),
        "candidate_undergenerated": bool(candidate.get("undergenerated")),
        "baseline_overgenerated": bool(baseline.get("overgenerated")),
        "candidate_overgenerated": bool(candidate.get("overgenerated")),
    }
    row["delta_mean_matched_iou"] = (
        row["candidate_mean_matched_iou"] - row["baseline_mean_matched_iou"]
    )
    row["undergeneration_fixed"] = (
        bool(row["baseline_undergenerated"]) and not bool(row["candidate_undergenerated"])
    )
    row["undergeneration_introduced"] = (
        not bool(row["baseline_undergenerated"]) and bool(row["candidate_undergenerated"])
    )
    row["outcome"] = quality_record_delta_outcome(row)
    return row


def quality_record_delta_outcome(row: Mapping[str, Any]) -> str:
    """Classify a comparison row by its delta pattern."""
    delta_tp = int(row.get("delta_true_positives", 0) or 0)
    delta_fp = int(row.get("delta_false_positives", 0) or 0)
    delta_pred = int(row.get("delta_pred_box_count", 0) or 0)
    if delta_tp > 0 and delta_fp <= 0:
        return "strict_improvement"
    if delta_tp > 0:
        return "recall_improvement_with_fp_cost"
    if delta_tp < 0:
        return "recall_regression"
    if delta_fp < 0:
        return "precision_improvement"
    if delta_fp > 0:
        return "precision_regression"
    if delta_pred > 0:
        return "box_count_increase_only"
    if delta_pred < 0:
        return "box_count_decrease_only"
    return "unchanged"


def summarize_quality_record_comparison(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-record comparison rows into a summary with delta metrics."""
    if not rows:
        return {"num_records": 0}
    outcome_counts = Counter(str(row.get("outcome", "unknown")) for row in rows)
    delta_tp = sum(int(row.get("delta_true_positives", 0) or 0) for row in rows)
    delta_fp = sum(int(row.get("delta_false_positives", 0) or 0) for row in rows)
    delta_fn = sum(int(row.get("delta_false_negatives", 0) or 0) for row in rows)
    delta_pred = sum(int(row.get("delta_pred_box_count", 0) or 0) for row in rows)
    baseline_tp = sum(int(row.get("baseline_true_positives", 0) or 0) for row in rows)
    baseline_fp = sum(int(row.get("baseline_false_positives", 0) or 0) for row in rows)
    baseline_fn = sum(int(row.get("baseline_false_negatives", 0) or 0) for row in rows)
    candidate_tp = sum(int(row.get("candidate_true_positives", 0) or 0) for row in rows)
    candidate_fp = sum(int(row.get("candidate_false_positives", 0) or 0) for row in rows)
    candidate_fn = sum(int(row.get("candidate_false_negatives", 0) or 0) for row in rows)
    baseline_metrics = aggregate_counts(baseline_tp, baseline_fp, baseline_fn)
    candidate_metrics = aggregate_counts(candidate_tp, candidate_fp, candidate_fn)
    return {
        "baseline_compared_summary": {
            **baseline_metrics,
            "true_positives": baseline_tp,
            "false_positives": baseline_fp,
            "false_negatives": baseline_fn,
            "avg_pred_boxes": mean(
                float(row.get("baseline_pred_box_count", 0) or 0) for row in rows
            ),
        },
        "candidate_compared_summary": {
            **candidate_metrics,
            "true_positives": candidate_tp,
            "false_positives": candidate_fp,
            "false_negatives": candidate_fn,
            "avg_pred_boxes": mean(
                float(row.get("candidate_pred_box_count", 0) or 0) for row in rows
            ),
        },
        "delta": {
            "precision": candidate_metrics["precision"] - baseline_metrics["precision"],
            "recall": candidate_metrics["recall"] - baseline_metrics["recall"],
            "f1": candidate_metrics["f1"] - baseline_metrics["f1"],
            "true_positives": delta_tp,
            "false_positives": delta_fp,
            "false_negatives": delta_fn,
            "pred_box_count": delta_pred,
            "avg_pred_boxes": mean(
                float(row.get("delta_pred_box_count", 0) or 0) for row in rows
            ),
            "raw_detection_count": sum(
                int(row.get("delta_raw_detection_count", 0) or 0) for row in rows
            ),
            "filtered_detection_count": sum(
                int(row.get("delta_filtered_detection_count", 0) or 0) for row in rows
            ),
            "mean_matched_iou": mean(
                float(row.get("delta_mean_matched_iou", 0.0) or 0.0) for row in rows
            ),
        },
        "outcome_counts": dict(outcome_counts),
        "tp_improved_records": sum(
            1 for row in rows if int(row.get("delta_true_positives", 0) or 0) > 0
        ),
        "tp_regressed_records": sum(
            1 for row in rows if int(row.get("delta_true_positives", 0) or 0) < 0
        ),
        "fp_increased_records": sum(
            1 for row in rows if int(row.get("delta_false_positives", 0) or 0) > 0
        ),
        "fp_decreased_records": sum(
            1 for row in rows if int(row.get("delta_false_positives", 0) or 0) < 0
        ),
        "undergeneration_fixed_records": sum(
            1 for row in rows if bool(row.get("undergeneration_fixed"))
        ),
        "undergeneration_introduced_records": sum(
            1 for row in rows if bool(row.get("undergeneration_introduced"))
        ),
        "num_records": len(rows),
    }


def summarize_quality_record_comparison_buckets(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Aggregate comparison rows by box-count bucket."""
    summary: Dict[str, Dict[str, Any]] = {}
    for bucket in ("single", "medium", "dense"):
        bucket_rows = [
            row for row in rows
            if str(row.get("box_count_bucket", "")) == bucket
        ]
        if not bucket_rows:
            continue
        summary[bucket] = {
            "num_records": len(bucket_rows),
            "delta_true_positives": sum(
                int(row.get("delta_true_positives", 0) or 0) for row in bucket_rows
            ),
            "delta_false_positives": sum(
                int(row.get("delta_false_positives", 0) or 0) for row in bucket_rows
            ),
            "delta_false_negatives": sum(
                int(row.get("delta_false_negatives", 0) or 0) for row in bucket_rows
            ),
            "delta_pred_box_count": sum(
                int(row.get("delta_pred_box_count", 0) or 0) for row in bucket_rows
            ),
            "avg_delta_pred_box_count": mean(
                float(row.get("delta_pred_box_count", 0) or 0) for row in bucket_rows
            ),
            "tp_improved_records": sum(
                1 for row in bucket_rows
                if int(row.get("delta_true_positives", 0) or 0) > 0
            ),
            "tp_regressed_records": sum(
                1 for row in bucket_rows
                if int(row.get("delta_true_positives", 0) or 0) < 0
            ),
        }
    return summary


def top_record_deltas(
    rows: Sequence[Mapping[str, Any]],
    *,
    reverse: bool = True,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """Return the top-N records by delta F1, filtering unchanged rows."""
    changed = [
        dict(row) for row in rows
        if str(row.get("outcome")) != "unchanged"
    ]
    return sorted(
        changed,
        key=lambda row: (
            float(row.get("delta_f1", 0.0) or 0.0),
            int(row.get("delta_true_positives", 0) or 0),
            -int(row.get("delta_false_positives", 0) or 0),
        ),
        reverse=reverse,
    )[:limit]


# ── Target count gap ────────────────────────────────────────────────────


def target_count_gap_row(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Compute target-count gap metrics for a single record."""
    tp = int_record_metric(record, "true_positives")
    fp = int_record_metric(record, "false_positives")
    fn = int_record_metric(record, "false_negatives")
    pred_count = int_record_metric(record, "pred_box_count")
    gt_count = int_record_metric(record, "gt_box_count")
    query_count = int_record_metric(record, "query_box_count") or gt_count
    target_count = query_count if query_count > 0 else gt_count
    target_deficit = max(0, target_count - pred_count)
    target_overage = max(0, pred_count - target_count)
    recoverable_tp = min(fn, target_deficit)
    current_metrics = aggregate_counts(tp, fp, fn)
    oracle_metrics = aggregate_counts(tp + recoverable_tp, fp, fn - recoverable_tp)
    bucket = str(record.get("box_count_bucket") or box_count_bucket(target_count))
    return {
        "index": record.get("index"),
        "image": record.get("image"),
        "allowed_labels": record.get("allowed_labels"),
        "box_count_bucket": bucket,
        "pred_box_count": pred_count,
        "gt_box_count": gt_count,
        "query_box_count": query_count,
        "target_box_count": target_count,
        "target_box_deficit": target_deficit,
        "target_box_overage": target_overage,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "current_f1": current_metrics["f1"],
        "oracle_true_positives": tp + recoverable_tp,
        "oracle_false_positives": fp,
        "oracle_false_negatives": fn - recoverable_tp,
        "oracle_f1": oracle_metrics["f1"],
        "oracle_delta_f1": oracle_metrics["f1"] - current_metrics["f1"],
        "oracle_recoverable_true_positives": recoverable_tp,
        "unrecoverable_false_negatives": fn - recoverable_tp,
        "blocked_by_no_count_slots": fn > 0 and target_deficit == 0,
    }


def summarize_target_count_gap_buckets(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Aggregate target-count gap rows by bucket."""
    summary: Dict[str, Dict[str, Any]] = {}
    for bucket in ("single", "medium", "dense"):
        bucket_rows = [
            row for row in rows
            if str(row.get("box_count_bucket", box_count_bucket(row.get("target_box_count")))) == bucket
        ]
        summary[bucket] = summarize_target_count_gap_rows(bucket_rows)
    return summary


def summarize_target_count_gap_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregate target-count gap metrics across rows."""
    current_tp = sum(int(row.get("true_positives", 0) or 0) for row in rows)
    current_fp = sum(int(row.get("false_positives", 0) or 0) for row in rows)
    current_fn = sum(int(row.get("false_negatives", 0) or 0) for row in rows)
    recoverable_tp = sum(int(row.get("oracle_recoverable_true_positives", 0) or 0) for row in rows)
    current_metrics = aggregate_counts(current_tp, current_fp, current_fn)
    oracle_metrics = aggregate_counts(
        current_tp + recoverable_tp,
        current_fp,
        current_fn - recoverable_tp,
    )
    return {
        "num_records": len(rows),
        "current_precision": current_metrics["precision"],
        "current_recall": current_metrics["recall"],
        "current_f1": current_metrics["f1"],
        "oracle_precision": oracle_metrics["precision"],
        "oracle_recall": oracle_metrics["recall"],
        "oracle_f1": oracle_metrics["f1"],
        "oracle_f1_delta": oracle_metrics["f1"] - current_metrics["f1"],
        "false_negatives": current_fn,
        "recoverable_false_negatives": recoverable_tp,
        "unrecoverable_false_negatives": current_fn - recoverable_tp,
        "recall_gap_closure_ratio": recoverable_tp / current_fn if current_fn else 0.0,
        "target_box_deficit": sum(int(row.get("target_box_deficit", 0) or 0) for row in rows),
        "target_box_overage": sum(int(row.get("target_box_overage", 0) or 0) for row in rows),
        "records_with_deficit": sum(
            1 for row in rows
            if int(row.get("target_box_deficit", 0) or 0) > 0
        ),
        "records_blocked_by_no_count_slots": sum(
            1 for row in rows
            if bool(row.get("blocked_by_no_count_slots"))
        ),
    }


# ── Box count bucket summarization ─────────────────────────────────────


def summarize_box_count_buckets(records: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Aggregate quality records by box-count bucket."""
    summary: Dict[str, Dict[str, Any]] = {}
    for bucket in ("single", "medium", "dense"):
        bucket_records = [
            record for record in records
            if str(record.get("box_count_bucket", box_count_bucket(record.get("gt_box_count")))) == bucket
        ]
        summary[bucket] = summarize_bucket_records(bucket_records)
    return summary


def summarize_bucket_records(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregate metrics for a set of records."""
    total_tp = sum(int(record.get("true_positives", 0) or 0) for record in records)
    total_fp = sum(int(record.get("false_positives", 0) or 0) for record in records)
    total_fn = sum(int(record.get("false_negatives", 0) or 0) for record in records)
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "num_samples": len(records),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "avg_pred_boxes": mean(float(record.get("pred_box_count", 0) or 0) for record in records),
        "avg_gt_boxes": mean(float(record.get("gt_box_count", 0) or 0) for record in records),
        "box_count_exact_match_ratio": ratio(records, "box_count_exact_match"),
        "box_count_overgeneration_ratio": ratio(records, "overgenerated"),
        "box_count_undergeneration_ratio": ratio(records, "undergenerated"),
    }
