"""Quality evaluation helpers for visual primitive detection outputs."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from .structured_vp_decoder import labels_match
from .vp_parsing import (
    allowed_label_field_candidates,
    bad_case_reasons,
    box_count_bucket,
    is_box,
    normalize_label,
    parse_prediction,
    quality_record_key,
    record_field_value,
    record_query_box_count,
    record_text,
    resolve_record_allowed_labels,
    resolve_record_positive_int_field,
    summary_records,
)
from .vp_aggregation import (
    aggregate_counts,
    compare_quality_record,
    int_record_metric,
    mean,
    quality_report_brief,
    ratio,
    safe_policy_label,
    summarize_box_count_buckets,
    summarize_bucket_records,
    summarize_quality_record_comparison,
    summarize_quality_record_comparison_buckets,
    summarize_target_count_gap_buckets,
    summarize_target_count_gap_rows,
    target_count_gap_row,
    top_record_deltas,
)
from ._vp_helpers import (
    _allowed_labels_from_config,
    _coerce_named_reports,
    _focus_bucket_metrics,
    _infer_policy_kind,
    _metric_gap,
    _normalize_focus_bucket,
    _policy_constraints_and_caveats,
    _policy_rank_key,
    _quality_report_to_comparison_row,
    _row_metric,
    load_vp_quality_summary,
)


@dataclass(frozen=True)
class VPDetectionQualityConfig:
    """Configuration for VP detection quality evaluation."""

    iou_threshold: float = 0.5
    use_structured_decoder: bool = True
    prediction_field: str = "raw_prediction"
    reference_field: str = "target"
    box_format: str = "loc_tokens"
    marker_style: str = "plain"
    filter_policy: str = "none"
    max_boxes_per_label: Optional[int] = None
    max_total_boxes: Optional[int] = None
    max_total_boxes_field: Optional[str] = None
    nms_iou_threshold: Optional[float] = None
    allowed_labels: Optional[Union[Sequence[str], str]] = None
    allowed_labels_field: Optional[str] = None
    label_match_mode: str = "strict"
    allowed_label_match_mode: Optional[str] = None
    repair_malformed_tail: bool = False
    max_bad_cases: int = 20


def evaluate_vp_detection_quality(
    predictions: Sequence[str],
    references: Sequence[str],
    *,
    config: Optional[VPDetectionQualityConfig] = None,
    metadata: Sequence[Mapping[str, Any]] = (),
) -> Dict[str, Any]:
    """Evaluate VP detection quality from prediction/reference text pairs."""

    config = config or VPDetectionQualityConfig()
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions and references must have the same length: "
            f"{len(predictions)} != {len(references)}"
        )
    from .visual_primitive_parser import VisualPrimitiveParser
    from .structured_vp_decoder import StructuredVisualPrimitiveDecoder

    parser = VisualPrimitiveParser()
    decoder = StructuredVisualPrimitiveDecoder(
        box_format=config.box_format,
        marker_style=config.marker_style,
    )

    records: List[Dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    for index, (prediction, reference) in enumerate(zip(predictions, references)):
        meta = metadata[index] if index < len(metadata) else {}
        task_prompt = meta.get("prefix") or meta.get("task_prompt")
        parsed_reference = parser.parse_detections(reference)
        effective_allowed_labels = resolve_record_allowed_labels(
            config=config,
            record=meta,
            reference_detections=parsed_reference,
        )
        (
            parsed_prediction,
            source,
            raw_detection_count,
            filtered_detection_count,
            repaired_tail_detection_count,
        ) = parse_prediction(
            prediction,
            parser=parser,
            decoder=decoder,
            config=config,
            record=meta,
            task_prompt=task_prompt,
            allowed_labels=effective_allowed_labels,
        )
        match_result = match_vp_detections(
            parsed_prediction,
            parsed_reference,
            iou_threshold=config.iou_threshold,
            label_match_mode=config.label_match_mode,
        )
        source_counts[source] += 1
        record = {
            "index": meta.get("index", index),
            "image": meta.get("image"),
            "prediction_source": source,
            "pred_box_count": len(parsed_prediction),
            "gt_box_count": len(parsed_reference),
            "query_box_count": record_query_box_count(meta, len(parsed_reference)),
            "raw_detection_count": raw_detection_count,
            "filtered_detection_count": filtered_detection_count,
            "repaired_tail_detection_count": repaired_tail_detection_count,
            "allowed_labels": effective_allowed_labels,
            **match_result,
        }
        record["box_count_bucket"] = box_count_bucket(record["query_box_count"])
        record["box_count_exact_match"] = record["pred_box_count"] == record["gt_box_count"]
        record["overgenerated"] = record["pred_box_count"] > record["gt_box_count"]
        record["undergenerated"] = record["pred_box_count"] < record["gt_box_count"]
        record["single_target_hit"] = (
            record["gt_box_count"] == 1 and record["true_positives"] >= 1
        )
        record["single_target_exact_hit"] = (
            record["gt_box_count"] == 1
            and record["true_positives"] == 1
            and record["false_positives"] == 0
            and record["false_negatives"] == 0
        )
        record["bad_case_reasons"] = bad_case_reasons(record)
        records.append(record)

    return summarize_vp_quality_records(
        records,
        source_counts=source_counts,
        config=config,
    )


def evaluate_vp_summary(
    summary: Mapping[str, Any],
    *,
    config: Optional[VPDetectionQualityConfig] = None,
) -> Dict[str, Any]:
    """Evaluate a saved ``vp_inference_visualization_summary.json`` object."""

    config = config or VPDetectionQualityConfig()
    from .vp_parsing import summary_records as _summary_records

    records = _summary_records(summary)
    predictions = [record_text(record, config.prediction_field) for record in records]
    references = [record_text(record, config.reference_field) for record in records]
    result = evaluate_vp_detection_quality(
        predictions,
        references,
        config=config,
        metadata=records,
    )
    result.update({
        "source_summary_path": summary.get("summary_path"),
        "model_path": summary.get("model_path"),
        "adapter_dir": summary.get("adapter_dir"),
        "data_path": summary.get("data_path"),
    })
    return result


def match_vp_detections(
    predictions: Sequence[Mapping[str, Any]],
    references: Sequence[Mapping[str, Any]],
    *,
    iou_threshold: float = 0.5,
    label_match_mode: str = "strict",
) -> Dict[str, Any]:
    """Greedily match predictions to references by normalized label and IoU."""

    candidate_matches: List[Tuple[float, int, int]] = []
    for pred_index, prediction in enumerate(predictions):
        for ref_index, reference in enumerate(references):
            if not labels_match(
                prediction.get("label"),
                reference.get("label"),
                mode=label_match_mode,
            ):
                continue
            iou = compute_bbox_iou(prediction.get("bbox"), reference.get("bbox"))
            if iou >= iou_threshold:
                candidate_matches.append((iou, pred_index, ref_index))

    candidate_matches.sort(reverse=True)
    matched_predictions = set()
    matched_references = set()
    matches: List[Dict[str, Any]] = []
    for iou, pred_index, ref_index in candidate_matches:
        if pred_index in matched_predictions or ref_index in matched_references:
            continue
        matched_predictions.add(pred_index)
        matched_references.add(ref_index)
        matches.append({
            "pred_index": pred_index,
            "ref_index": ref_index,
            "label": str(predictions[pred_index].get("label", "")),
            "iou": iou,
        })

    true_positives = len(matches)
    false_positives = len(predictions) - true_positives
    false_negatives = len(references) - true_positives
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "matched_ious": [match["iou"] for match in matches],
        "mean_matched_iou": mean(match["iou"] for match in matches),
        "matches": matches,
    }


def summarize_vp_quality_records(
    records: Sequence[Mapping[str, Any]],
    *,
    source_counts: Optional[Counter[str]] = None,
    config: Optional[VPDetectionQualityConfig] = None,
) -> Dict[str, Any]:
    """Aggregate per-record VP quality rows."""

    config = config or VPDetectionQualityConfig()
    source_counts = source_counts or Counter(str(record.get("prediction_source", "unknown")) for record in records)
    total_tp = sum(int(record.get("true_positives", 0) or 0) for record in records)
    total_fp = sum(int(record.get("false_positives", 0) or 0) for record in records)
    total_fn = sum(int(record.get("false_negatives", 0) or 0) for record in records)
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    num_samples = len(records)
    single_target_records = [
        record for record in records
        if int(record.get("gt_box_count", 0) or 0) == 1
    ]

    bad_cases = [
        {
            "index": record.get("index"),
            "image": record.get("image"),
            "reasons": record.get("bad_case_reasons", []),
            "pred_box_count": record.get("pred_box_count", 0),
            "gt_box_count": record.get("gt_box_count", 0),
            "true_positives": record.get("true_positives", 0),
            "false_positives": record.get("false_positives", 0),
            "false_negatives": record.get("false_negatives", 0),
            "mean_matched_iou": record.get("mean_matched_iou", 0.0),
        }
        for record in records
        if record.get("bad_case_reasons")
    ][: max(0, int(config.max_bad_cases))]

    source_total = sum(source_counts.values()) or 1
    return {
        "num_samples": num_samples,
        "config": {
            "iou_threshold": config.iou_threshold,
            "use_structured_decoder": config.use_structured_decoder,
            "prediction_field": config.prediction_field,
            "reference_field": config.reference_field,
            "box_format": config.box_format,
            "marker_style": config.marker_style,
            "filter_policy": config.filter_policy,
            "max_boxes_per_label": config.max_boxes_per_label,
            "max_total_boxes": config.max_total_boxes,
            "max_total_boxes_field": config.max_total_boxes_field,
            "nms_iou_threshold": config.nms_iou_threshold,
            "allowed_labels": config.allowed_labels,
            "allowed_labels_field": config.allowed_labels_field,
            "label_match_mode": config.label_match_mode,
            "allowed_label_match_mode": config.allowed_label_match_mode,
            "repair_malformed_tail": config.repair_malformed_tail,
        },
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
        "repaired_tail_detection_count": sum(
            int(record.get("repaired_tail_detection_count", 0) or 0)
            for record in records
        ),
        "repaired_tail_record_ratio": (
            sum(
                1 for record in records
                if int(record.get("repaired_tail_detection_count", 0) or 0) > 0
            ) / len(records)
            if records else 0.0
        ),
        "avg_repaired_tail_detection_count": mean(
            float(record.get("repaired_tail_detection_count", 0) or 0)
            for record in records
        ),
        "mean_matched_iou": mean(
            iou
            for record in records
            for iou in record.get("matched_ious", [])
        ),
        "single_target_samples": len(single_target_records),
        "single_target_hit_ratio": ratio(single_target_records, "single_target_hit"),
        "single_target_exact_hit_ratio": ratio(single_target_records, "single_target_exact_hit"),
        "box_count_bucket_summary": summarize_box_count_buckets(records),
        "prediction_source_counts": dict(source_counts),
        "prediction_source_ratios": {
            source: count / source_total
            for source, count in source_counts.items()
        },
        "bad_case_count": len(bad_cases),
        "bad_cases": bad_cases,
        "records": list(records),
    }


def compute_bbox_iou(box1: Any, box2: Any) -> float:
    """Compute IoU for two ``[x1, y1, x2, y2]`` boxes."""

    if not is_box(box1) or not is_box(box2):
        return 0.0
    x1_1, y1_1, x2_1, y2_1 = [float(value) for value in box1]
    x1_2, y1_2, x2_2, y2_2 = [float(value) for value in box2]
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    area1 = max(0.0, (x2_1 - x1_1)) * max(0.0, (y2_1 - y1_1))
    area2 = max(0.0, (x2_2 - x1_2)) * max(0.0, (y2_2 - y1_2))
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def render_vp_detection_quality_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact Markdown VP detection quality report."""

    lines = [
        "# VP Detection Quality",
        "",
        f"- Samples: `{report.get('num_samples', 0)}`",
        f"- Precision: `{float(report.get('precision', 0.0) or 0.0):.4f}`",
        f"- Recall: `{float(report.get('recall', 0.0) or 0.0):.4f}`",
        f"- F1: `{float(report.get('f1', 0.0) or 0.0):.4f}`",
        f"- Mean matched IoU: `{float(report.get('mean_matched_iou', 0.0) or 0.0):.4f}`",
        f"- Avg pred boxes: `{float(report.get('avg_pred_boxes', 0.0) or 0.0):.4f}`",
        f"- Avg GT boxes: `{float(report.get('avg_gt_boxes', 0.0) or 0.0):.4f}`",
        f"- Box count exact match: `{float(report.get('box_count_exact_match_ratio', 0.0) or 0.0):.4f}`",
        f"- Box count overgeneration: `{float(report.get('box_count_overgeneration_ratio', 0.0) or 0.0):.4f}`",
        f"- Single-target exact hit: `{float(report.get('single_target_exact_hit_ratio', 0.0) or 0.0):.4f}`",
        f"- Repaired tail detections: `{int(report.get('repaired_tail_detection_count', 0) or 0)}`",
        f"- Repaired tail record ratio: `{float(report.get('repaired_tail_record_ratio', 0.0) or 0.0):.4f}`",
    ]
    bucket_summary = dict(report.get("box_count_bucket_summary", {}) or {})
    if bucket_summary:
        lines.extend([
            "",
            "## Box Count Buckets",
            "",
            "| bucket | samples | precision | recall | f1 | avg pred | avg GT | FP | FN |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for bucket in ("single", "medium", "dense"):
            row = dict(bucket_summary.get(bucket, {}) or {})
            if not row:
                continue
            lines.append(
                f"| `{bucket}` "
                f"| {int(row.get('num_samples', 0) or 0)} "
                f"| {float(row.get('precision', 0.0) or 0.0):.4f} "
                f"| {float(row.get('recall', 0.0) or 0.0):.4f} "
                f"| {float(row.get('f1', 0.0) or 0.0):.4f} "
                f"| {float(row.get('avg_pred_boxes', 0.0) or 0.0):.2f} "
                f"| {float(row.get('avg_gt_boxes', 0.0) or 0.0):.2f} "
                f"| {int(row.get('false_positives', 0) or 0)} "
                f"| {int(row.get('false_negatives', 0) or 0)} |"
            )

    lines.extend([
        "",
        "## Prediction Sources",
        "",
    ])
    for source, count in dict(report.get("prediction_source_counts", {}) or {}).items():
        ratio = dict(report.get("prediction_source_ratios", {}) or {}).get(source, 0.0)
        lines.append(f"- `{source}`: `{count}` (`{float(ratio):.4f}`)")

    bad_cases = list(report.get("bad_cases", []) or [])
    if bad_cases:
        lines.extend(["", "## Bad Cases", ""])
        for case in bad_cases:
            reasons = ", ".join(str(reason) for reason in case.get("reasons", []))
            lines.append(
                f"- index `{case.get('index')}`: {reasons}; "
                f"pred={case.get('pred_box_count')}, gt={case.get('gt_box_count')}, "
                f"tp={case.get('true_positives')}, fp={case.get('false_positives')}, fn={case.get('false_negatives')}"
            )

    return "\n".join(lines) + "\n"


def compare_vp_quality_reports(
    reports: Union[Mapping[str, Mapping[str, Any]], Sequence[Tuple[str, Mapping[str, Any]]]],
    *,
    focus_bucket: Optional[str] = None,
) -> Dict[str, Any]:
    """Compare multiple VP quality reports and rank their post-filter policies."""

    focus_bucket = _normalize_focus_bucket(focus_bucket)
    items = _coerce_named_reports(reports)
    rows = [
        _quality_report_to_comparison_row(name, report, focus_bucket=focus_bucket)
        for name, report in items
    ]
    ranked_rows = sorted(
        rows,
        key=lambda row: _policy_rank_key(row, focus_bucket=focus_bucket),
        reverse=True,
    )
    for rank, row in enumerate(ranked_rows, start=1):
        row["rank"] = rank

    rows_by_policy = {row["policy"]: row for row in ranked_rows}
    ordered_rows = [rows_by_policy[row["policy"]] for row in rows]
    recommendation = recommend_vp_policy(ranked_rows, focus_bucket=focus_bucket)
    return {
        "num_reports": len(rows),
        "focus_bucket": focus_bucket,
        "recommended_policy": recommendation.get("policy"),
        "recommendation": recommendation,
        "rows": ordered_rows,
        "ranked_rows": ranked_rows,
    }


def compare_vp_quality_record_reports(
    candidate_report: Mapping[str, Any],
    baseline_report: Mapping[str, Any],
    *,
    candidate_name: str = "candidate",
    baseline_name: str = "baseline",
    focus_bucket: Optional[str] = None,
) -> Dict[str, Any]:
    """Compare two VP quality reports at the per-record level."""

    focus_bucket = _normalize_focus_bucket(focus_bucket)
    from .vp_parsing import index_quality_records

    candidate_records = index_quality_records(candidate_report.get("records", []))
    baseline_records = index_quality_records(baseline_report.get("records", []))
    common_keys = [
        key for key in baseline_records
        if key in candidate_records
    ]

    rows = []
    for key in common_keys:
        baseline = baseline_records[key]
        candidate = candidate_records[key]
        bucket = str(
            candidate.get("box_count_bucket")
            or baseline.get("box_count_bucket")
            or box_count_bucket(candidate.get("gt_box_count", baseline.get("gt_box_count")))
        )
        if focus_bucket and bucket != focus_bucket:
            continue
        rows.append(compare_quality_record(key, candidate, baseline, bucket=bucket))

    row_summary = summarize_quality_record_comparison(rows)
    result = {
        "candidate_name": safe_policy_label(candidate_name),
        "baseline_name": safe_policy_label(baseline_name),
        "focus_bucket": focus_bucket,
        "candidate_report_path": candidate_report.get("source_report_path"),
        "baseline_report_path": baseline_report.get("source_report_path"),
        "candidate_num_records": len(candidate_records),
        "baseline_num_records": len(baseline_records),
        "matched_records": len(common_keys),
        "compared_records": len(rows),
        "missing_candidate_records": [
            key for key in baseline_records
            if key not in candidate_records
        ],
        "extra_candidate_records": [
            key for key in candidate_records
            if key not in baseline_records
        ],
        "candidate_summary": quality_report_brief(candidate_report),
        "baseline_summary": quality_report_brief(baseline_report),
        **row_summary,
        "rows": rows,
    }
    result["top_improvements"] = top_record_deltas(rows, reverse=True)
    result["top_regressions"] = top_record_deltas(rows, reverse=False)
    result["bucket_summary"] = summarize_quality_record_comparison_buckets(rows)
    return result


def recommend_vp_policy(
    comparison_or_rows: Union[Mapping[str, Any], Sequence[Mapping[str, Any]]],
    *,
    focus_bucket: Optional[str] = None,
) -> Dict[str, Any]:
    """Recommend a VP policy from comparison rows with deployment caveats."""

    focus_bucket = _normalize_focus_bucket(focus_bucket)
    if isinstance(comparison_or_rows, Mapping):
        focus_bucket = _normalize_focus_bucket(
            focus_bucket or comparison_or_rows.get("focus_bucket")
        )
        rows = list(comparison_or_rows.get("ranked_rows") or comparison_or_rows.get("rows") or [])
    else:
        rows = list(comparison_or_rows)
    ranked_rows = sorted(
        rows,
        key=lambda row: _policy_rank_key(row, focus_bucket=focus_bucket),
        reverse=True,
    )
    if not ranked_rows:
        return {
            "policy": None,
            "reason": "no_reports",
            "confidence": "none",
            "caveats": ["No VP quality reports were provided."],
        }

    best = ranked_rows[0]
    general_candidates = [
        row for row in ranked_rows
        if not row.get("requires_allowed_labels")
        and row.get("policy_kind") != "single_target"
    ]
    general = general_candidates[0] if general_candidates else None
    caveats = list(best.get("caveats", []) or [])
    if general and general.get("policy") != best.get("policy"):
        caveats.append(
            f"For unconstrained multi-target detection, prefer `{general.get('policy')}` "
            f"(F1={float(general.get('f1', 0.0) or 0.0):.4f})."
        )

    sample_key = "focus_num_samples" if focus_bucket else "num_samples"
    min_samples = min(int(row.get(sample_key, 0) or 0) for row in ranked_rows)
    confidence = "exploratory" if min_samples < 10 else "moderate"
    if min_samples >= 50 and _metric_gap(ranked_rows, "f1", focus_bucket=focus_bucket) >= 0.03:
        confidence = "strong"

    metric_label = f"{focus_bucket} bucket" if focus_bucket else "overall"
    best_f1 = _row_metric(best, "f1", focus_bucket=focus_bucket)
    best_overgen = _row_metric(best, "box_count_overgeneration_ratio", focus_bucket=focus_bucket)
    return {
        "policy": best.get("policy"),
        "policy_kind": best.get("policy_kind"),
        "focus_bucket": focus_bucket,
        "confidence": confidence,
        "reason": (
            f"Best ranked policy by {metric_label} F1, overgeneration, "
            f"box-count exactness, and mean IoU: `{best.get('policy')}` "
            f"(F1={best_f1:.4f}, overgeneration={best_overgen:.4f})."
        ),
        "general_detection_policy": general.get("policy") if general else None,
        "general_detection_reason": (
            "Best policy without explicit allow-list or single-target assumptions."
            if general else "No unconstrained general-detection candidate was provided."
        ),
        "caveats": caveats,
    }


def render_vp_policy_comparison_markdown(comparison: Mapping[str, Any]) -> str:
    """Render a Markdown table comparing VP post-filter policy quality."""

    recommendation = dict(comparison.get("recommendation", {}) or {})
    focus_bucket = _normalize_focus_bucket(comparison.get("focus_bucket") or recommendation.get("focus_bucket"))
    rows = list(comparison.get("ranked_rows") or comparison.get("rows") or [])
    lines = [
        "# VP Policy Comparison",
        "",
        f"- Reports: `{comparison.get('num_reports', len(rows))}`",
        f"- Recommended policy: `{recommendation.get('policy')}`",
        f"- Confidence: `{recommendation.get('confidence', 'unknown')}`",
    ]
    if recommendation.get("general_detection_policy"):
        lines.append(
            f"- General detection fallback: `{recommendation.get('general_detection_policy')}`"
        )
    if recommendation.get("reason"):
        lines.append(f"- Reason: {recommendation['reason']}")
    if focus_bucket:
        lines.append(f"- Focus bucket: `{focus_bucket}`")

    if focus_bucket:
        lines.extend([
            "",
            "| rank | policy | kind | overall f1 | focus precision | focus recall | focus f1 | focus avg pred | focus avg GT | focus FP | focus FN | constraints |",
            "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ])
    else:
        lines.extend([
            "",
            "| rank | policy | kind | precision | recall | f1 | mean IoU | avg pred | avg GT | overgen | exact | FP | FN | constraints |",
            "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ])
    for row in rows:
        constraints = ", ".join(str(item) for item in row.get("constraints", []) or []) or "-"
        if focus_bucket:
            lines.append(
                f"| {int(row.get('rank', 0) or 0)} "
                f"| `{row.get('policy')}` "
                f"| `{row.get('policy_kind')}` "
                f"| {float(row.get('f1', 0.0) or 0.0):.4f} "
                f"| {float(row.get('focus_precision', 0.0) or 0.0):.4f} "
                f"| {float(row.get('focus_recall', 0.0) or 0.0):.4f} "
                f"| {float(row.get('focus_f1', 0.0) or 0.0):.4f} "
                f"| {float(row.get('focus_avg_pred_boxes', 0.0) or 0.0):.2f} "
                f"| {float(row.get('focus_avg_gt_boxes', 0.0) or 0.0):.2f} "
                f"| {int(row.get('focus_false_positives', 0) or 0)} "
                f"| {int(row.get('focus_false_negatives', 0) or 0)} "
                f"| {constraints} |"
            )
        else:
            lines.append(
                f"| {int(row.get('rank', 0) or 0)} "
                f"| `{row.get('policy')}` "
                f"| `{row.get('policy_kind')}` "
                f"| {float(row.get('precision', 0.0) or 0.0):.4f} "
                f"| {float(row.get('recall', 0.0) or 0.0):.4f} "
                f"| {float(row.get('f1', 0.0) or 0.0):.4f} "
                f"| {float(row.get('mean_matched_iou', 0.0) or 0.0):.4f} "
                f"| {float(row.get('avg_pred_boxes', 0.0) or 0.0):.2f} "
                f"| {float(row.get('avg_gt_boxes', 0.0) or 0.0):.2f} "
                f"| {float(row.get('box_count_overgeneration_ratio', 0.0) or 0.0):.4f} "
                f"| {float(row.get('box_count_exact_match_ratio', 0.0) or 0.0):.4f} "
                f"| {int(row.get('false_positives', 0) or 0)} "
                f"| {int(row.get('false_negatives', 0) or 0)} "
                f"| {constraints} |"
            )

    caveats = list(recommendation.get("caveats", []) or [])
    if caveats:
        lines.extend(["", "## Caveats", ""])
        for caveat in caveats:
            lines.append(f"- {caveat}")

    return "\n".join(lines) + "\n"


def render_vp_record_comparison_markdown(comparison: Mapping[str, Any]) -> str:
    """Render a Markdown report for per-record adapter-vs-baseline deltas."""

    candidate_name = comparison.get("candidate_name", "candidate")
    baseline_name = comparison.get("baseline_name", "baseline")
    delta = dict(comparison.get("delta", {}) or {})
    outcome_counts = dict(comparison.get("outcome_counts", {}) or {})
    lines = [
        "# VP Record Comparison",
        "",
        f"- Candidate: `{candidate_name}`",
        f"- Baseline: `{baseline_name}`",
        f"- Compared records: `{int(comparison.get('compared_records', 0) or 0)}`",
        f"- Focus bucket: `{comparison.get('focus_bucket') or 'all'}`",
        f"- Delta TP / FP / FN: `{int(delta.get('true_positives', 0) or 0)}` / "
        f"`{int(delta.get('false_positives', 0) or 0)}` / "
        f"`{int(delta.get('false_negatives', 0) or 0)}`",
        f"- Delta avg pred boxes: `{float(delta.get('avg_pred_boxes', 0.0) or 0.0):.4f}`",
        f"- Delta F1: `{float(delta.get('f1', 0.0) or 0.0):.4f}`",
        "",
        "## Outcome Counts",
        "",
    ]
    if outcome_counts:
        for outcome, count in sorted(outcome_counts.items()):
            lines.append(f"- `{outcome}`: `{count}`")
    else:
        lines.append("- `none`: `0`")

    bucket_summary = dict(comparison.get("bucket_summary", {}) or {})
    if bucket_summary:
        lines.extend([
            "",
            "## Buckets",
            "",
            "| bucket | records | delta TP | delta FP | delta FN | delta pred | improved TP | regressed TP |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for bucket in ("single", "medium", "dense"):
            row = dict(bucket_summary.get(bucket, {}) or {})
            if not row:
                continue
            lines.append(
                f"| `{bucket}` "
                f"| {int(row.get('num_records', 0) or 0)} "
                f"| {int(row.get('delta_true_positives', 0) or 0)} "
                f"| {int(row.get('delta_false_positives', 0) or 0)} "
                f"| {int(row.get('delta_false_negatives', 0) or 0)} "
                f"| {float(row.get('delta_pred_box_count', 0.0) or 0.0):.2f} "
                f"| {int(row.get('tp_improved_records', 0) or 0)} "
                f"| {int(row.get('tp_regressed_records', 0) or 0)} |"
            )

    changed_rows = [
        row for row in list(comparison.get("rows", []) or [])
        if str(row.get("outcome")) != "unchanged"
    ][:20]
    if changed_rows:
        lines.extend([
            "",
            "## Changed Records",
            "",
            "| index | label | bucket | outcome | base pred/tp/fp/fn | cand pred/tp/fp/fn | delta TP/FP/FN |",
            "| ---: | --- | --- | --- | --- | --- | --- |",
        ])
        for row in changed_rows:
            lines.append(
                f"| {row.get('index')} "
                f"| `{row.get('allowed_labels')}` "
                f"| `{row.get('box_count_bucket')}` "
                f"| `{row.get('outcome')}` "
                f"| {row.get('baseline_pred_box_count')}/{row.get('baseline_true_positives')}/"
                f"{row.get('baseline_false_positives')}/{row.get('baseline_false_negatives')} "
                f"| {row.get('candidate_pred_box_count')}/{row.get('candidate_true_positives')}/"
                f"{row.get('candidate_false_positives')}/{row.get('candidate_false_negatives')} "
                f"| {row.get('delta_true_positives')}/{row.get('delta_false_positives')}/"
                f"{row.get('delta_false_negatives')} |"
            )

    return "\n".join(lines) + "\n"


def analyze_vp_target_count_gap(
    report: Mapping[str, Any],
    *,
    focus_bucket: Optional[str] = None,
    max_rows: int = 20,
) -> Dict[str, Any]:
    """Estimate the upper bound of filling missing boxes up to target count."""

    focus_bucket = _normalize_focus_bucket(focus_bucket)
    records = [
        record for record in list(report.get("records", []) or [])
        if isinstance(record, Mapping)
    ]
    if focus_bucket:
        records = [
            record for record in records
            if str(record.get("box_count_bucket", box_count_bucket(record.get("gt_box_count")))) == focus_bucket
        ]

    rows = [target_count_gap_row(record) for record in records]
    current_tp = sum(int(row.get("true_positives", 0) or 0) for row in rows)
    current_fp = sum(int(row.get("false_positives", 0) or 0) for row in rows)
    current_fn = sum(int(row.get("false_negatives", 0) or 0) for row in rows)
    recoverable_tp = sum(int(row.get("oracle_recoverable_true_positives", 0) or 0) for row in rows)
    oracle_tp = current_tp + recoverable_tp
    oracle_fp = current_fp
    oracle_fn = current_fn - recoverable_tp
    current_metrics = aggregate_counts(current_tp, current_fp, current_fn)
    oracle_metrics = aggregate_counts(oracle_tp, oracle_fp, oracle_fn)
    bucket_summary = summarize_target_count_gap_buckets(rows)
    ranked_rows = sorted(
        rows,
        key=lambda row: (
            float(row.get("oracle_delta_f1", 0.0) or 0.0),
            int(row.get("oracle_recoverable_true_positives", 0) or 0),
            int(row.get("target_box_deficit", 0) or 0),
        ),
        reverse=True,
    )[: max(0, int(max_rows))]

    return {
        "num_records": len(rows),
        "focus_bucket": focus_bucket,
        "source_report_path": report.get("quality_json_path") or report.get("source_report_path"),
        "current": {
            **current_metrics,
            "true_positives": current_tp,
            "false_positives": current_fp,
            "false_negatives": current_fn,
            "avg_pred_boxes": mean(float(row.get("pred_box_count", 0) or 0) for row in rows),
            "avg_target_boxes": mean(float(row.get("target_box_count", 0) or 0) for row in rows),
        },
        "oracle_count_fill": {
            **oracle_metrics,
            "true_positives": oracle_tp,
            "false_positives": oracle_fp,
            "false_negatives": oracle_fn,
            "recovered_true_positives": recoverable_tp,
            "precision_delta": oracle_metrics["precision"] - current_metrics["precision"],
            "recall_delta": oracle_metrics["recall"] - current_metrics["recall"],
            "f1_delta": oracle_metrics["f1"] - current_metrics["f1"],
        },
        "count_gap": {
            "target_box_deficit": sum(int(row.get("target_box_deficit", 0) or 0) for row in rows),
            "target_box_overage": sum(int(row.get("target_box_overage", 0) or 0) for row in rows),
            "records_with_deficit": sum(1 for row in rows if int(row.get("target_box_deficit", 0) or 0) > 0),
            "records_with_overage": sum(1 for row in rows if int(row.get("target_box_overage", 0) or 0) > 0),
            "records_with_recoverable_gap": sum(
                1 for row in rows
                if int(row.get("oracle_recoverable_true_positives", 0) or 0) > 0
            ),
            "records_blocked_by_no_count_slots": sum(
                1 for row in rows
                if int(row.get("false_negatives", 0) or 0) > 0
                and int(row.get("target_box_deficit", 0) or 0) == 0
            ),
            "false_negatives": current_fn,
            "recoverable_false_negatives": recoverable_tp,
            "unrecoverable_false_negatives": current_fn - recoverable_tp,
            "recall_gap_closure_ratio": recoverable_tp / current_fn if current_fn else 0.0,
        },
        "bucket_summary": bucket_summary,
        "top_gap_records": ranked_rows,
        "rows": rows,
    }


def render_vp_target_count_gap_markdown(analysis: Mapping[str, Any]) -> str:
    """Render target-count gap analysis as Markdown."""

    current = dict(analysis.get("current", {}) or {})
    oracle = dict(analysis.get("oracle_count_fill", {}) or {})
    gap = dict(analysis.get("count_gap", {}) or {})
    lines = [
        "# VP Target-Count Gap Analysis",
        "",
        f"- Records: `{int(analysis.get('num_records', 0) or 0)}`",
        f"- Focus bucket: `{analysis.get('focus_bucket') or 'all'}`",
        f"- Current F1 / recall: `{float(current.get('f1', 0.0) or 0.0):.4f}` / "
        f"`{float(current.get('recall', 0.0) or 0.0):.4f}`",
        f"- Oracle count-fill F1 / recall: `{float(oracle.get('f1', 0.0) or 0.0):.4f}` / "
        f"`{float(oracle.get('recall', 0.0) or 0.0):.4f}`",
        f"- Recoverable FN: `{int(gap.get('recoverable_false_negatives', 0) or 0)}` / "
        f"`{int(gap.get('false_negatives', 0) or 0)}` "
        f"(`{float(gap.get('recall_gap_closure_ratio', 0.0) or 0.0):.4f}`)",
        f"- Target box deficit / overage: `{int(gap.get('target_box_deficit', 0) or 0)}` / "
        f"`{int(gap.get('target_box_overage', 0) or 0)}`",
    ]

    bucket_summary = dict(analysis.get("bucket_summary", {}) or {})
    if bucket_summary:
        lines.extend([
            "",
            "## Buckets",
            "",
            "| bucket | records | current f1 | oracle f1 | recoverable FN | total FN | deficit | no-slot blocked |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for bucket in ("single", "medium", "dense"):
            row = dict(bucket_summary.get(bucket, {}) or {})
            if not row:
                continue
            lines.append(
                f"| `{bucket}` "
                f"| {int(row.get('num_records', 0) or 0)} "
                f"| {float(row.get('current_f1', 0.0) or 0.0):.4f} "
                f"| {float(row.get('oracle_f1', 0.0) or 0.0):.4f} "
                f"| {int(row.get('recoverable_false_negatives', 0) or 0)} "
                f"| {int(row.get('false_negatives', 0) or 0)} "
                f"| {int(row.get('target_box_deficit', 0) or 0)} "
                f"| {int(row.get('records_blocked_by_no_count_slots', 0) or 0)} |"
            )

    top_rows = list(analysis.get("top_gap_records", []) or [])
    if top_rows:
        lines.extend([
            "",
            "## Top Gap Records",
            "",
            "| index | label | bucket | pred/target | TP/FP/FN | recoverable TP | oracle delta F1 |",
            "| ---: | --- | --- | ---: | ---: | ---: | ---: |",
        ])
        for row in top_rows:
            lines.append(
                f"| {row.get('index')} "
                f"| `{row.get('allowed_labels')}` "
                f"| `{row.get('box_count_bucket')}` "
                f"| {row.get('pred_box_count')}/{row.get('target_box_count')} "
                f"| {row.get('true_positives')}/{row.get('false_positives')}/{row.get('false_negatives')} "
                f"| {row.get('oracle_recoverable_true_positives')} "
                f"| {float(row.get('oracle_delta_f1', 0.0) or 0.0):.4f} |"
            )

    return "\n".join(lines) + "\n"
