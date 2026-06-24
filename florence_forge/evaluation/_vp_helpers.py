"""Private helpers for :mod:`vp_detection_quality`.

These functions are internal implementation details and should not be imported
directly.  They are re-exported from ``vp_detection_quality`` for backward
compatibility with any code that may have imported them.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from .structured_vp_decoder import (
    StructuredVisualPrimitiveDecoder,
    resolve_structured_vp_filter_caps,
)
from .visual_primitive_parser import VisualPrimitiveParser


# ── Parsing ────────────────────────────────────────────────────────────────


def _parse_prediction(
    text: str,
    *,
    parser: VisualPrimitiveParser,
    decoder: StructuredVisualPrimitiveDecoder,
    config: Any,  # VPDetectionQualityConfig – avoided circular import
    record: Mapping[str, Any],
    task_prompt: Any,
    allowed_labels: Optional[Union[Sequence[str], str]],
) -> Tuple[List[Dict[str, Any]], str, int, int, int]:
    if not config.use_structured_decoder:
        return parser.parse_detections(text), "visual_primitive_raw", 0, 0, 0

    filter_caps = resolve_structured_vp_filter_caps(
        policy=config.filter_policy,
        task_prompt=task_prompt,
        max_boxes_per_label=config.max_boxes_per_label,
        max_total_boxes=_resolve_record_positive_int_field(
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


# ── Record indexing ────────────────────────────────────────────────────────


def _summary_records(summary: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    records = summary.get("records", [])
    if not isinstance(records, Sequence):
        return []
    return [record for record in records if isinstance(record, Mapping)]


def _index_quality_records(records: Any) -> Dict[str, Mapping[str, Any]]:
    if not isinstance(records, Sequence):
        return {}

    indexed: Dict[str, Mapping[str, Any]] = {}
    for position, record in enumerate(records):
        if not isinstance(record, Mapping):
            continue
        key = _quality_record_key(record)
        if key in indexed:
            key = f"{key}#pos={position}"
        indexed[key] = record
    return indexed


def _quality_record_key(record: Mapping[str, Any]) -> str:
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


# ── Record comparison ──────────────────────────────────────────────────────


def _compare_quality_record(
    key: str,
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
    *,
    bucket: str,
) -> Dict[str, Any]:
    candidate_tp = _int_record_metric(candidate, "true_positives")
    candidate_fp = _int_record_metric(candidate, "false_positives")
    candidate_fn = _int_record_metric(candidate, "false_negatives")
    baseline_tp = _int_record_metric(baseline, "true_positives")
    baseline_fp = _int_record_metric(baseline, "false_positives")
    baseline_fn = _int_record_metric(baseline, "false_negatives")
    candidate_pred = _int_record_metric(candidate, "pred_box_count")
    baseline_pred = _int_record_metric(baseline, "pred_box_count")
    delta_tp = candidate_tp - baseline_tp
    delta_fp = candidate_fp - baseline_fp
    delta_fn = candidate_fn - baseline_fn
    delta_pred = candidate_pred - baseline_pred
    baseline_f1 = _record_f1(baseline_tp, baseline_fp, baseline_fn)
    candidate_f1 = _record_f1(candidate_tp, candidate_fp, candidate_fn)
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
        "baseline_raw_detection_count": _int_record_metric(baseline, "raw_detection_count"),
        "candidate_raw_detection_count": _int_record_metric(candidate, "raw_detection_count"),
        "delta_raw_detection_count": (
            _int_record_metric(candidate, "raw_detection_count")
            - _int_record_metric(baseline, "raw_detection_count")
        ),
        "baseline_filtered_detection_count": _int_record_metric(baseline, "filtered_detection_count"),
        "candidate_filtered_detection_count": _int_record_metric(candidate, "filtered_detection_count"),
        "delta_filtered_detection_count": (
            _int_record_metric(candidate, "filtered_detection_count")
            - _int_record_metric(baseline, "filtered_detection_count")
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
    row["outcome"] = _quality_record_delta_outcome(row)
    return row


def _quality_record_delta_outcome(row: Mapping[str, Any]) -> str:
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


def _summarize_quality_record_comparison(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
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
    baseline_metrics = _aggregate_counts(baseline_tp, baseline_fp, baseline_fn)
    candidate_metrics = _aggregate_counts(candidate_tp, candidate_fp, candidate_fn)
    return {
        "baseline_compared_summary": {
            **baseline_metrics,
            "true_positives": baseline_tp,
            "false_positives": baseline_fp,
            "false_negatives": baseline_fn,
            "avg_pred_boxes": _mean(
                float(row.get("baseline_pred_box_count", 0) or 0) for row in rows
            ),
        },
        "candidate_compared_summary": {
            **candidate_metrics,
            "true_positives": candidate_tp,
            "false_positives": candidate_fp,
            "false_negatives": candidate_fn,
            "avg_pred_boxes": _mean(
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
            "avg_pred_boxes": _mean(
                float(row.get("delta_pred_box_count", 0) or 0) for row in rows
            ),
            "raw_detection_count": sum(
                int(row.get("delta_raw_detection_count", 0) or 0) for row in rows
            ),
            "filtered_detection_count": sum(
                int(row.get("delta_filtered_detection_count", 0) or 0) for row in rows
            ),
            "mean_matched_iou": _mean(
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
        "fp_reduced_records": sum(
            1 for row in rows if int(row.get("delta_false_positives", 0) or 0) < 0
        ),
        "fp_increased_records": sum(
            1 for row in rows if int(row.get("delta_false_positives", 0) or 0) > 0
        ),
        "pred_count_increased_records": sum(
            1 for row in rows if int(row.get("delta_pred_box_count", 0) or 0) > 0
        ),
        "pred_count_decreased_records": sum(
            1 for row in rows if int(row.get("delta_pred_box_count", 0) or 0) < 0
        ),
        "undergeneration_fixed_records": sum(
            1 for row in rows if bool(row.get("undergeneration_fixed"))
        ),
        "undergeneration_introduced_records": sum(
            1 for row in rows if bool(row.get("undergeneration_introduced"))
        ),
        "bucket_summary": _summarize_quality_record_comparison_buckets(rows),
    }


def _summarize_quality_record_comparison_buckets(
    rows: Sequence[Mapping[str, Any]]
) -> Dict[str, Dict[str, Any]]:
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
            "avg_delta_pred_box_count": _mean(
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
            "fp_increased_records": sum(
                1 for row in bucket_rows
                if int(row.get("delta_false_positives", 0) or 0) > 0
            ),
            "fp_reduced_records": sum(
                1 for row in bucket_rows
                if int(row.get("delta_false_positives", 0) or 0) < 0
            ),
        }
    return summary


def _top_record_deltas(
    rows: Sequence[Mapping[str, Any]],
    *,
    reverse: bool,
    limit: int = 5,
) -> List[Dict[str, Any]]:
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


# ── Metric aggregation ────────────────────────────────────────────────────


def _quality_report_brief(report: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "num_samples": report.get("num_samples"),
        "precision": float(report.get("precision", 0.0) or 0.0),
        "recall": float(report.get("recall", 0.0) or 0.0),
        "f1": float(report.get("f1", 0.0) or 0.0),
        "true_positives": int(report.get("true_positives", 0) or 0),
        "false_positives": int(report.get("false_positives", 0) or 0),
        "false_negatives": int(report.get("false_negatives", 0) or 0),
        "avg_pred_boxes": float(report.get("avg_pred_boxes", 0.0) or 0.0),
        "avg_gt_boxes": float(report.get("avg_gt_boxes", 0.0) or 0.0),
    }


def _aggregate_counts(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _record_f1(tp: int, fp: int, fn: int) -> float:
    return _aggregate_counts(tp, fp, fn)["f1"]


def _int_record_metric(record: Mapping[str, Any], key: str) -> int:
    try:
        return int(record.get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0


def _safe_policy_label(value: str) -> str:
    cleaned = "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "_"
        for ch in str(value or "").strip()
    )
    return cleaned or "policy"


# ── Record field resolution ───────────────────────────────────────────────


def _record_text(record: Mapping[str, Any], field: str) -> str:
    value = record.get(field)
    if isinstance(value, str):
        return value
    fallback_fields = ("raw_prediction", "prediction", "structured_prediction", "target")
    for fallback in fallback_fields:
        value = record.get(fallback)
        if isinstance(value, str):
            return value
    return ""


def _resolve_record_allowed_labels(
    *,
    config: Any,  # VPDetectionQualityConfig – avoided circular import
    record: Mapping[str, Any],
    reference_detections: Sequence[Mapping[str, Any]],
) -> Optional[Union[Sequence[str], str]]:
    if config.allowed_labels is not None:
        return config.allowed_labels
    if not config.allowed_labels_field:
        return None

    for field in _allowed_label_field_candidates(config.allowed_labels_field):
        normalized_field = field.strip().lower()
        if normalized_field in {"target_labels", "reference_labels", "gt_labels"}:
            labels = [
                str(detection.get("label", "")).strip()
                for detection in reference_detections
                if str(detection.get("label", "")).strip()
            ]
            return labels or None
        value = _record_field_value(record, field)
        if value not in (None, ""):
            return value
    return None


def _allowed_label_field_candidates(value: str) -> List[str]:
    return [
        item.strip()
        for item in str(value or "").replace("|", ",").replace(";", ",").split(",")
        if item.strip()
    ]


def _record_field_value(record: Mapping[str, Any], field: str) -> Any:
    current: Any = record
    for part in str(field).split("."):
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current


def _resolve_record_positive_int_field(
    record: Mapping[str, Any],
    field_spec: Optional[str],
    *,
    fallback: Optional[int] = None,
) -> Optional[int]:
    """Resolve a positive integer cap from a per-record field specification."""

    if field_spec:
        for field in _allowed_label_field_candidates(field_spec):
            value = _record_field_value(record, field)
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


# ── Bad-case diagnostics ──────────────────────────────────────────────────


def _bad_case_reasons(record: Mapping[str, Any]) -> List[str]:
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


# ── Box count utilities ───────────────────────────────────────────────────


def _record_query_box_count(record: Mapping[str, Any], fallback: int) -> int:
    for key in ("query_box_count", "curriculum_query_box_count", "gt_box_count"):
        value = record.get(key)
        try:
            if value is not None:
                return max(0, int(value))
        except (TypeError, ValueError):
            pass
    return max(0, int(fallback))


def _box_count_bucket(box_count: Any) -> str:
    try:
        parsed = int(box_count)
    except (TypeError, ValueError):
        parsed = 0
    if parsed <= 1:
        return "single"
    if parsed <= 3:
        return "medium"
    return "dense"


# ── Target count gap ──────────────────────────────────────────────────────


def _target_count_gap_row(record: Mapping[str, Any]) -> Dict[str, Any]:
    tp = _int_record_metric(record, "true_positives")
    fp = _int_record_metric(record, "false_positives")
    fn = _int_record_metric(record, "false_negatives")
    pred_count = _int_record_metric(record, "pred_box_count")
    gt_count = _int_record_metric(record, "gt_box_count")
    query_count = _int_record_metric(record, "query_box_count") or gt_count
    target_count = query_count if query_count > 0 else gt_count
    target_deficit = max(0, target_count - pred_count)
    target_overage = max(0, pred_count - target_count)
    recoverable_tp = min(fn, target_deficit)
    current_metrics = _aggregate_counts(tp, fp, fn)
    oracle_metrics = _aggregate_counts(tp + recoverable_tp, fp, fn - recoverable_tp)
    bucket = str(record.get("box_count_bucket") or _box_count_bucket(target_count))
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


def _summarize_target_count_gap_buckets(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}
    for bucket in ("single", "medium", "dense"):
        bucket_rows = [
            row for row in rows
            if str(row.get("box_count_bucket", _box_count_bucket(row.get("target_box_count")))) == bucket
        ]
        summary[bucket] = _summarize_target_count_gap_rows(bucket_rows)
    return summary


def _summarize_target_count_gap_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    current_tp = sum(int(row.get("true_positives", 0) or 0) for row in rows)
    current_fp = sum(int(row.get("false_positives", 0) or 0) for row in rows)
    current_fn = sum(int(row.get("false_negatives", 0) or 0) for row in rows)
    recoverable_tp = sum(int(row.get("oracle_recoverable_true_positives", 0) or 0) for row in rows)
    current_metrics = _aggregate_counts(current_tp, current_fp, current_fn)
    oracle_metrics = _aggregate_counts(
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


def _summarize_box_count_buckets(records: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}
    for bucket in ("single", "medium", "dense"):
        bucket_records = [
            record for record in records
            if str(record.get("box_count_bucket", _box_count_bucket(record.get("gt_box_count")))) == bucket
        ]
        summary[bucket] = _summarize_bucket_records(bucket_records)
    return summary


def _summarize_bucket_records(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
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
        "avg_pred_boxes": _mean(float(record.get("pred_box_count", 0) or 0) for record in records),
        "avg_gt_boxes": _mean(float(record.get("gt_box_count", 0) or 0) for record in records),
        "box_count_exact_match_ratio": _ratio(records, "box_count_exact_match"),
        "box_count_overgeneration_ratio": _ratio(records, "overgenerated"),
        "box_count_undergeneration_ratio": _ratio(records, "undergenerated"),
    }


# ── Generic utilities ─────────────────────────────────────────────────────


def _normalize_label(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _is_box(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 4
        and all(isinstance(coord, (int, float)) for coord in value)
    )


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _ratio(records: Sequence[Mapping[str, Any]], key: str) -> float:
    if not records:
        return 0.0
    return sum(1 for record in records if bool(record.get(key))) / len(records)


# ── Policy comparison ─────────────────────────────────────────────────────


def _coerce_named_reports(
    reports: Union[Mapping[str, Mapping[str, Any]], Sequence[Tuple[str, Mapping[str, Any]]]],
) -> List[Tuple[str, Mapping[str, Any]]]:
    if isinstance(reports, Mapping):
        return [
            (str(name), report)
            for name, report in reports.items()
            if isinstance(report, Mapping)
        ]
    items: List[Tuple[str, Mapping[str, Any]]] = []
    for item in reports:
        if not isinstance(item, Sequence) or len(item) != 2:
            raise TypeError("quality reports must be named (name, report) pairs")
        name, report = item
        if not isinstance(report, Mapping):
            raise TypeError(f"quality report for {name!r} must be a mapping")
        items.append((str(name), report))
    return items


def _quality_report_to_comparison_row(
    name: str,
    report: Mapping[str, Any],
    *,
    focus_bucket: Optional[str] = None,
) -> Dict[str, Any]:
    config = dict(report.get("config", {}) or {})
    policy_kind = _infer_policy_kind(name, config)
    num_samples = int(report.get("num_samples", 0) or 0)
    single_target_samples = int(report.get("single_target_samples", 0) or 0)
    single_target_coverage = single_target_samples / num_samples if num_samples else 0.0
    focus_bucket = _normalize_focus_bucket(focus_bucket)
    focus_metrics = _focus_bucket_metrics(report, focus_bucket)
    constraints, caveats = _policy_constraints_and_caveats(
        name=name,
        policy_kind=policy_kind,
        config=config,
        num_samples=num_samples,
        single_target_coverage=single_target_coverage,
    )
    row = {
        "policy": name,
        "policy_kind": policy_kind,
        "num_samples": num_samples,
        "precision": float(report.get("precision", 0.0) or 0.0),
        "recall": float(report.get("recall", 0.0) or 0.0),
        "f1": float(report.get("f1", 0.0) or 0.0),
        "mean_matched_iou": float(report.get("mean_matched_iou", 0.0) or 0.0),
        "avg_pred_boxes": float(report.get("avg_pred_boxes", 0.0) or 0.0),
        "avg_gt_boxes": float(report.get("avg_gt_boxes", 0.0) or 0.0),
        "box_count_exact_match_ratio": float(report.get("box_count_exact_match_ratio", 0.0) or 0.0),
        "box_count_overgeneration_ratio": float(report.get("box_count_overgeneration_ratio", 0.0) or 0.0),
        "box_count_undergeneration_ratio": float(report.get("box_count_undergeneration_ratio", 0.0) or 0.0),
        "single_target_samples": single_target_samples,
        "single_target_coverage": single_target_coverage,
        "single_target_exact_hit_ratio": float(report.get("single_target_exact_hit_ratio", 0.0) or 0.0),
        "true_positives": int(report.get("true_positives", 0) or 0),
        "false_positives": int(report.get("false_positives", 0) or 0),
        "false_negatives": int(report.get("false_negatives", 0) or 0),
        "bad_case_count": int(report.get("bad_case_count", 0) or 0),
        "prediction_source_counts": dict(report.get("prediction_source_counts", {}) or {}),
        "source_report_path": report.get("quality_json_path") or report.get("source_report_path"),
        "source_summary_path": report.get("source_summary_path"),
        "config": config,
        "constraints": constraints,
        "caveats": caveats,
        "requires_allowed_labels": bool(_allowed_labels_from_config(config)),
        "unsafe_single_target_constraint": (
            policy_kind == "single_target"
            and bool(num_samples)
            and single_target_coverage < 1.0
        ),
    }
    if focus_bucket:
        row.update({
            "focus_bucket": focus_bucket,
            "focus_num_samples": int(focus_metrics.get("num_samples", 0) or 0),
            "focus_precision": float(focus_metrics.get("precision", 0.0) or 0.0),
            "focus_recall": float(focus_metrics.get("recall", 0.0) or 0.0),
            "focus_f1": float(focus_metrics.get("f1", 0.0) or 0.0),
            "focus_avg_pred_boxes": float(focus_metrics.get("avg_pred_boxes", 0.0) or 0.0),
            "focus_avg_gt_boxes": float(focus_metrics.get("avg_gt_boxes", 0.0) or 0.0),
            "focus_box_count_exact_match_ratio": float(
                focus_metrics.get("box_count_exact_match_ratio", 0.0) or 0.0
            ),
            "focus_box_count_overgeneration_ratio": float(
                focus_metrics.get("box_count_overgeneration_ratio", 0.0) or 0.0
            ),
            "focus_box_count_undergeneration_ratio": float(
                focus_metrics.get("box_count_undergeneration_ratio", 0.0) or 0.0
            ),
            "focus_true_positives": int(focus_metrics.get("true_positives", 0) or 0),
            "focus_false_positives": int(focus_metrics.get("false_positives", 0) or 0),
            "focus_false_negatives": int(focus_metrics.get("false_negatives", 0) or 0),
        })
    return row


def _infer_policy_kind(name: str, config: Mapping[str, Any]) -> str:
    filter_policy = str(config.get("filter_policy") or "").strip().lower()
    lowered_name = name.strip().lower()
    if _allowed_labels_from_config(config):
        return "allowed_labels"
    if filter_policy == "single-target" or int(config.get("max_total_boxes") or 0) == 1:
        return "single_target"
    if filter_policy == "nms":
        return "nms"
    if filter_policy == "auto":
        return "auto"
    if "single" in lowered_name or "total1" in lowered_name:
        return "single_target"
    if "allowed" in lowered_name or "label" in lowered_name:
        return "allowed_labels"
    if "nms" in lowered_name:
        return "nms"
    return "none"


def _allowed_labels_from_config(config: Mapping[str, Any]) -> Any:
    return (
        config.get("allowed_labels")
        or config.get("structured_vp_allowed_labels")
        or config.get("allowed_labels_field")
    )


def _policy_constraints_and_caveats(
    *,
    name: str,
    policy_kind: str,
    config: Mapping[str, Any],
    num_samples: int,
    single_target_coverage: float,
) -> Tuple[List[str], List[str]]:
    constraints: List[str] = []
    caveats: List[str] = []
    allowed_labels = _allowed_labels_from_config(config)
    allowed_labels_field = config.get("allowed_labels_field")
    if allowed_labels:
        if allowed_labels_field:
            constraints.append(f"allowed_labels_field={allowed_labels_field}")
            caveats.append(
                f"`{name}` depends on per-record allowed labels from `{allowed_labels_field}`; "
                "use it for query/category-constrained runs, not as proof of unconstrained OD capability."
            )
        else:
            constraints.append("explicit_allowed_labels")
            caveats.append(
                f"`{name}` depends on an explicit allowed-label list; use it for query/category-constrained "
                "runs, not as proof of unconstrained OD capability."
            )
    if policy_kind == "single_target":
        constraints.append("max_total_boxes=1")
        if num_samples and single_target_coverage < 1.0:
            caveats.append(
                f"`{name}` forces one box but only {single_target_coverage:.1%} of samples are single-target."
            )
        else:
            caveats.append(
                f"`{name}` is appropriate only when the task contract is single-target grounding/detection."
            )
    if policy_kind == "nms":
        threshold = config.get("nms_iou_threshold")
        constraints.append(f"nms_iou={threshold}" if threshold is not None else "nms")
    if bool(config.get("repair_malformed_tail")):
        constraints.append("repair_malformed_tail")
    if str(config.get("label_match_mode") or "strict") != "strict":
        constraints.append(f"label_match_mode={config.get('label_match_mode')}")
        caveats.append(
            f"`{name}` uses phrase-contained label matching; reserve it for query/category-constrained "
            "diagnostics where labels may be specialized phrases."
        )
    if str(config.get("allowed_label_match_mode") or "strict") != "strict":
        constraints.append(f"allowed_label_match_mode={config.get('allowed_label_match_mode')}")
    return constraints, caveats


def _normalize_focus_bucket(value: Any) -> Optional[str]:
    if value in (None, "", "overall", "none"):
        return None
    bucket = str(value).strip().lower()
    if bucket not in {"single", "medium", "dense"}:
        raise ValueError("focus_bucket must be one of: single, medium, dense")
    return bucket


def _focus_bucket_metrics(report: Mapping[str, Any], focus_bucket: Optional[str]) -> Mapping[str, Any]:
    if not focus_bucket:
        return {}
    bucket_summary = dict(report.get("box_count_bucket_summary", {}) or {})
    return dict(bucket_summary.get(focus_bucket, {}) or {})


def _row_metric(row: Mapping[str, Any], key: str, *, focus_bucket: Optional[str] = None) -> float:
    if focus_bucket:
        return float(row.get(f"focus_{key}", 0.0) or 0.0)
    return float(row.get(key, 0.0) or 0.0)


def _policy_rank_key(
    row: Mapping[str, Any],
    *,
    focus_bucket: Optional[str] = None,
) -> Tuple[float, float, float, float, float, float, float, float]:
    avg_pred = _row_metric(row, "avg_pred_boxes", focus_bucket=focus_bucket)
    avg_gt = _row_metric(row, "avg_gt_boxes", focus_bucket=focus_bucket)
    drift = abs(avg_pred - avg_gt)
    return (
        _row_metric(row, "f1", focus_bucket=focus_bucket),
        -_row_metric(row, "box_count_overgeneration_ratio", focus_bucket=focus_bucket),
        _row_metric(row, "box_count_exact_match_ratio", focus_bucket=focus_bucket),
        -drift,
        float(row.get("mean_matched_iou", 0.0) or 0.0),
        _row_metric(row, "precision", focus_bucket=focus_bucket),
        _row_metric(row, "recall", focus_bucket=focus_bucket),
        -_row_metric(row, "false_positives", focus_bucket=focus_bucket),
    )


def _metric_gap(
    rows: Sequence[Mapping[str, Any]],
    key: str,
    *,
    focus_bucket: Optional[str] = None,
) -> float:
    if len(rows) < 2:
        return 0.0
    values = sorted(
        (_row_metric(row, key, focus_bucket=focus_bucket) for row in rows),
        reverse=True,
    )
    return values[0] - values[1]


def load_vp_quality_summary(path: str | Path) -> Dict[str, Any]:
    """Load an inference visualization summary JSON object."""

    data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object at {path}")
    return data
