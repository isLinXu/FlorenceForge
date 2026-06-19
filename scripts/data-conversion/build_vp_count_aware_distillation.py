#!/usr/bin/env python3
"""Build count-aware VP distillation rows from an inference summary."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from florence_forge.evaluation.vp_detection_quality import (  # noqa: E402
    VPDetectionQualityConfig,
    evaluate_vp_summary,
)
from florence_forge.evaluation.vp_parsing import box_count_bucket, summary_records  # noqa: E402


DEFAULT_TEXT_INPUT_TEMPLATE = (
    "{label} | target_count={query_box_count} | "
    "predicted={pred_box_count} | missing={missing_box_count}"
)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    summary_path = Path(args.inference_summary).expanduser()
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    inference_summary = _load_json(summary_path)

    quality = evaluate_vp_summary(
        inference_summary,
        config=VPDetectionQualityConfig(
            prediction_field=args.prediction_field,
            reference_field=args.reference_field,
            box_format=args.structured_vp_box_format,
            marker_style=args.structured_vp_marker_style,
            filter_policy=args.structured_vp_filter_policy,
            max_total_boxes_field=args.structured_vp_max_total_boxes_field,
            nms_iou_threshold=args.structured_vp_nms_iou_threshold,
            allowed_labels=args.structured_vp_allowed_labels,
            allowed_labels_field=args.structured_vp_allowed_labels_field,
            label_match_mode=args.vp_label_match_mode,
            allowed_label_match_mode=args.structured_vp_allowed_label_match_mode,
            repair_malformed_tail=args.structured_vp_repair_malformed_tail,
            iou_threshold=args.quality_iou_threshold,
            max_bad_cases=0,
        ),
    )
    rows, counters = build_count_aware_rows(
        summary_records(inference_summary),
        quality.get("records", []),
        source_summary_path=str(summary_path),
        text_input_template=args.text_input_template,
        focus_bucket=args.focus_bucket,
        min_missing_boxes=args.min_missing_boxes,
        min_false_negatives=args.min_false_negatives,
        require_undergenerated=not args.include_non_undergenerated,
        max_rows=args.max_rows,
        output_task_type=args.output_task_type,
        source_format=args.source_format,
        box_format=args.structured_vp_box_format,
        marker_style=args.structured_vp_marker_style,
        target_mode=args.distillation_target_mode,
    )
    _write_jsonl(output_path, rows)

    summary_path_out = (
        Path(args.summary_output).expanduser()
        if args.summary_output else _default_summary_path(output_path)
    )
    markdown_path = (
        Path(args.markdown_output).expanduser()
        if args.markdown_output else summary_path_out.with_suffix(".md")
    )
    summary = {
        "inference_summary_path": str(summary_path),
        "output_path": str(output_path),
        "summary_path": str(summary_path_out),
        "markdown_path": str(markdown_path),
        "input_records": len(summary_records(inference_summary)),
        "output_rows": len(rows),
        "text_input_template": args.text_input_template,
        "focus_bucket": args.focus_bucket,
        "min_missing_boxes": args.min_missing_boxes,
        "min_false_negatives": args.min_false_negatives,
        "require_undergenerated": not args.include_non_undergenerated,
        "distillation_target_mode": args.distillation_target_mode,
        "quality": _quality_brief(quality),
        **counters,
    }
    summary_path_out.parent.mkdir(parents=True, exist_ok=True)
    summary_path_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_path.write_text(render_markdown(summary), encoding="utf-8")
    return summary


def build_count_aware_rows(
    records: Sequence[Mapping[str, Any]],
    quality_records: Sequence[Mapping[str, Any]],
    *,
    source_summary_path: str,
    text_input_template: str = DEFAULT_TEXT_INPUT_TEMPLATE,
    focus_bucket: Optional[str] = "dense",
    min_missing_boxes: int = 1,
    min_false_negatives: int = 1,
    require_undergenerated: bool = True,
    max_rows: Optional[int] = None,
    output_task_type: str = "OPEN_VOCABULARY_DETECTION",
    source_format: str = "vp_count_aware_distillation",
    box_format: str = "loc_tokens",
    marker_style: str = "plain",
    target_mode: str = "reference",
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    skip_counts: Counter[str] = Counter()
    missing_counts: List[int] = []
    recoverable_counts: List[int] = []
    delta_f1_total = 0.0

    min_missing_boxes = max(0, int(min_missing_boxes))
    min_false_negatives = max(0, int(min_false_negatives))

    for position, record in enumerate(records):
        if not isinstance(record, Mapping):
            skip_counts["invalid_record"] += 1
            continue
        quality = quality_records[position] if position < len(quality_records) else {}
        query_count = _positive_int(
            record.get("query_box_count"),
            record.get("gt_box_count"),
            quality.get("query_box_count"),
            quality.get("gt_box_count"),
        )
        bucket = str(quality.get("box_count_bucket") or box_count_bucket(query_count))
        if focus_bucket and bucket != focus_bucket:
            skip_counts["focus_bucket"] += 1
            continue

        pred_count = _safe_int(quality.get("pred_box_count"), _safe_int(record.get("pred_box_count")))
        gt_count = _positive_int(quality.get("gt_box_count"), record.get("gt_box_count"), query_count)
        missing = max(0, query_count - pred_count)
        false_negatives = _safe_int(quality.get("false_negatives"))
        if require_undergenerated and missing <= 0:
            skip_counts["not_undergenerated"] += 1
            continue
        if missing < min_missing_boxes:
            skip_counts["min_missing_boxes"] += 1
            continue
        if false_negatives < min_false_negatives:
            skip_counts["min_false_negatives"] += 1
            continue

        suffix = _distillation_suffix(record, target_mode)
        if not suffix:
            skip_counts["empty_suffix"] += 1
            continue

        recoverable = min(missing, false_negatives)
        row = _count_aware_row(
            record,
            quality,
            suffix=suffix,
            source_summary_path=source_summary_path,
            position=position,
            query_count=query_count,
            gt_count=gt_count,
            pred_count=pred_count,
            missing=missing,
            recoverable=recoverable,
            bucket=bucket,
            text_input_template=text_input_template,
            output_task_type=output_task_type,
            source_format=source_format,
            box_format=box_format,
            marker_style=marker_style,
            target_mode=target_mode,
        )
        rows.append(row)
        missing_counts.append(missing)
        recoverable_counts.append(recoverable)
        delta_f1_total += float(row.get("distillation_delta_f1", 0.0) or 0.0)
        if max_rows is not None and len(rows) >= max(0, int(max_rows)):
            break

    return rows, {
        "skip_counts": dict(skip_counts),
        "total_missing_boxes_in_output": sum(missing_counts),
        "avg_missing_boxes_in_output": _mean(missing_counts),
        "total_recoverable_fn_in_output": sum(recoverable_counts),
        "avg_recoverable_fn_in_output": _mean(recoverable_counts),
        "avg_delta_f1_estimate": delta_f1_total / len(rows) if rows else 0.0,
        "bucket_counts": _bucket_counts(rows),
        "top_labels": [
            {"label": label, "count": count}
            for label, count in Counter(str(row.get("query_label", "") or "") for row in rows).most_common(20)
            if label
        ],
    }


def render_markdown(summary: Mapping[str, Any]) -> str:
    quality = dict(summary.get("quality", {}) or {})
    lines = [
        "# VP Count-Aware Distillation Dataset",
        "",
        f"- Input records: `{int(summary.get('input_records', 0) or 0)}`",
        f"- Output rows: `{int(summary.get('output_rows', 0) or 0)}`",
        f"- Focus bucket: `{summary.get('focus_bucket')}`",
        f"- Target mode: `{summary.get('distillation_target_mode')}`",
        f"- Min missing boxes: `{int(summary.get('min_missing_boxes', 0) or 0)}`",
        f"- Min false negatives: `{int(summary.get('min_false_negatives', 0) or 0)}`",
        f"- Output missing boxes: `{int(summary.get('total_missing_boxes_in_output', 0) or 0)}`",
        f"- Output recoverable FN estimate: `{int(summary.get('total_recoverable_fn_in_output', 0) or 0)}`",
        f"- Avg missing boxes / row: `{float(summary.get('avg_missing_boxes_in_output', 0.0) or 0.0):.4f}`",
        f"- Source F1 / P / R: `{float(quality.get('f1', 0.0) or 0.0):.4f}` / "
        f"`{float(quality.get('precision', 0.0) or 0.0):.4f}` / "
        f"`{float(quality.get('recall', 0.0) or 0.0):.4f}`",
        "",
        "## Buckets",
        "",
        "| bucket | rows |",
        "| --- | ---: |",
    ]
    for bucket, count in dict(summary.get("bucket_counts", {}) or {}).items():
        lines.append(f"| `{bucket}` | {int(count or 0)} |")
    skip_counts = dict(summary.get("skip_counts", {}) or {})
    if skip_counts:
        lines.extend(["", "## Skips", ""])
        for reason, count in skip_counts.items():
            lines.append(f"- `{reason}`: `{count}`")
    top_labels = list(summary.get("top_labels", []) or [])
    if top_labels:
        lines.extend(["", "## Top Labels", ""])
        for item in top_labels:
            lines.append(f"- `{item.get('label')}`: `{item.get('count')}`")
    return "\n".join(lines) + "\n"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-summary", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--markdown-output", default=None)
    parser.add_argument("--output-task-type", default="OPEN_VOCABULARY_DETECTION")
    parser.add_argument("--source-format", default="vp_count_aware_distillation")
    parser.add_argument("--text-input-template", default=DEFAULT_TEXT_INPUT_TEMPLATE)
    parser.add_argument("--focus-bucket", default="dense", choices=["single", "medium", "dense"])
    parser.add_argument("--min-missing-boxes", type=int, default=1)
    parser.add_argument("--min-false-negatives", type=int, default=1)
    parser.add_argument("--include-non-undergenerated", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--distillation-target-mode", default="reference", choices=["reference", "prediction"])
    parser.add_argument("--prediction-field", default="raw_prediction")
    parser.add_argument("--reference-field", default="target")
    parser.add_argument("--structured-vp-box-format", default="loc_tokens", choices=["loc_tokens", "json"])
    parser.add_argument("--structured-vp-marker-style", default="plain", choices=["plain", "special"])
    parser.add_argument("--structured-vp-filter-policy", default="nms", choices=["none", "auto", "single-target", "nms"])
    parser.add_argument("--structured-vp-max-total-boxes-field", default="query_box_count")
    parser.add_argument("--structured-vp-nms-iou-threshold", type=float, default=0.5)
    parser.add_argument("--structured-vp-allowed-labels", default=None)
    parser.add_argument("--structured-vp-allowed-labels-field", default="query_label")
    parser.add_argument("--structured-vp-allowed-label-match-mode", default="strict", choices=["strict", "contains"])
    parser.add_argument("--structured-vp-repair-malformed-tail", action="store_true")
    parser.add_argument("--vp-label-match-mode", default="strict", choices=["strict", "contains"])
    parser.add_argument("--quality-iou-threshold", type=float, default=0.5)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    print(json.dumps(run(parse_args(argv)), indent=2, ensure_ascii=False))
    return 0


def _count_aware_row(
    record: Mapping[str, Any],
    quality: Mapping[str, Any],
    *,
    suffix: str,
    source_summary_path: str,
    position: int,
    query_count: int,
    gt_count: int,
    pred_count: int,
    missing: int,
    recoverable: int,
    bucket: str,
    text_input_template: str,
    output_task_type: str,
    source_format: str,
    box_format: str,
    marker_style: str,
    target_mode: str,
) -> Dict[str, Any]:
    label = _query_label(record)
    original_text_input = str(record.get("text_input") or label)
    text_input = _format_text_input(
        text_input_template,
        label=label,
        text_input=original_text_input,
        query_box_count=query_count,
        gt_box_count=gt_count,
        pred_box_count=pred_count,
        missing_box_count=missing,
        false_negatives=_safe_int(quality.get("false_negatives")),
    )
    tp = _safe_int(quality.get("true_positives"))
    fp = _safe_int(quality.get("false_positives"))
    fn = _safe_int(quality.get("false_negatives"))
    current_f1 = _f1(tp, fp, fn)
    oracle_f1 = _f1(tp + recoverable, fp, max(0, fn - recoverable))
    return {
        "image": record.get("image"),
        "prefix": record.get("prefix") or f"<{output_task_type}>",
        "suffix": suffix,
        "task_family": "visual_primitive",
        "base_task": record.get("base_task") or "CAPTION_TO_PHRASE_GROUNDING",
        "source_format": source_format,
        "query_label": label,
        "text_input": text_input,
        "query_box_count": query_count,
        "gt_box_count": gt_count,
        "vp_box_format": box_format,
        "vp_marker_style": marker_style,
        "vp_task_type": output_task_type,
        "distillation_source": "target_count_gap",
        "distillation_target_mode": target_mode,
        "distillation_source_index": record.get("index", position),
        "distillation_source_summary_path": source_summary_path,
        "distillation_added_box_count": missing,
        "distillation_deficit_before": missing,
        "distillation_deficit_after": max(0, missing - recoverable),
        "distillation_target_count_reached": recoverable == missing,
        "distillation_primary_pred_box_count": pred_count,
        "distillation_teacher_pred_box_count": query_count,
        "distillation_primary_tp": tp,
        "distillation_primary_fp": fp,
        "distillation_primary_fn": fn,
        "distillation_primary_f1": current_f1,
        "distillation_teacher_tp": tp + recoverable,
        "distillation_teacher_fp": fp,
        "distillation_teacher_fn": max(0, fn - recoverable),
        "distillation_teacher_f1": oracle_f1,
        "distillation_delta_tp": recoverable,
        "distillation_delta_fp": 0,
        "distillation_delta_fn": -recoverable,
        "distillation_delta_f1": oracle_f1 - current_f1,
        "count_aware_source": "vp_detection_quality_gap",
        "count_aware_text_input_template": text_input_template,
        "count_hint_original_text_input": original_text_input,
        "count_hint_text_input": text_input,
        "count_hint_query_box_count": query_count,
        "count_aware_bucket": bucket,
        "count_aware_pred_box_count": pred_count,
        "count_aware_missing_box_count": missing,
        "count_aware_recoverable_fn_estimate": recoverable,
        "count_aware_false_negatives": fn,
        "count_aware_true_positives": tp,
        "count_aware_false_positives": fp,
    }


def _format_text_input(template: str, **values: Any) -> str:
    result = str(template)
    for key, value in values.items():
        result = result.replace("{" + key + "}", str(value))
    result = result.replace("{count}", str(values.get("query_box_count", "")))
    return result


def _query_label(record: Mapping[str, Any]) -> str:
    return str(record.get("query_label") or record.get("text_input") or record.get("allowed_labels") or "").strip()


def _distillation_suffix(record: Mapping[str, Any], target_mode: str) -> str:
    mode = str(target_mode or "reference").strip().lower()
    if mode == "reference":
        return str(record.get("target", "") or "").strip()
    if mode == "prediction":
        return str(record.get("structured_prediction") or record.get("prediction") or "").strip()
    raise ValueError("target_mode must be one of: reference, prediction")


def _quality_brief(report: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "num_samples": report.get("num_samples", 0),
        "precision": report.get("precision", 0.0),
        "recall": report.get("recall", 0.0),
        "f1": report.get("f1", 0.0),
        "true_positives": report.get("true_positives", 0),
        "false_positives": report.get("false_positives", 0),
        "false_negatives": report.get("false_negatives", 0),
        "avg_pred_boxes": report.get("avg_pred_boxes", 0.0),
        "avg_gt_boxes": report.get("avg_gt_boxes", 0.0),
    }


def _bucket_counts(rows: Iterable[Mapping[str, Any]]) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        bucket = str(row.get("count_aware_bucket") or box_count_bucket(_safe_int(row.get("query_box_count"))))
        counts[bucket] += 1
    return {bucket: int(counts.get(bucket, 0)) for bucket in ("single", "medium", "dense")}


def _positive_int(*values: Any) -> int:
    for value in values:
        parsed = _safe_int(value)
        if parsed > 0:
            return parsed
    return 0


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _f1(tp: int, fp: int, fn: int) -> float:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def _mean(values: Sequence[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _load_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object at {path}")
    return data


def _default_summary_path(output_path: Path) -> Path:
    suffix = "".join(output_path.suffixes)
    name = output_path.name[: -len(suffix)] if suffix else output_path.name
    return output_path.with_name(f"{name}_summary.json")


if __name__ == "__main__":
    raise SystemExit(main())
