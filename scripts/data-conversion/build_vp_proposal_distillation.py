#!/usr/bin/env python3
"""Build VP query-grounding distillation JSONL from proposal replay."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = REPO_ROOT / "scripts" / "experiments"
for path in (REPO_ROOT, EXPERIMENTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from florence_forge.evaluation.vp_detection_quality import (  # noqa: E402
    VPDetectionQualityConfig,
    evaluate_vp_summary,
)
from replay_vp_target_count_proposals import build_target_count_proposal_summary  # noqa: E402


def run(args: argparse.Namespace) -> Dict[str, Any]:
    primary_summary_path = Path(args.primary_summary).expanduser()
    proposal_summary_path = Path(args.proposal_summary).expanduser()
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    primary_summary = _load_json(primary_summary_path)
    proposal_summary = _load_json(proposal_summary_path)
    replay_summary = build_target_count_proposal_summary(
        primary_summary,
        proposal_summary,
        primary_summary_path=str(primary_summary_path),
        proposal_summary_path=str(proposal_summary_path),
        primary_policy=args.primary_filter_policy,
        proposal_policy=args.proposal_filter_policy,
        box_format=args.structured_vp_box_format,
        marker_style=args.structured_vp_marker_style,
        allowed_labels=args.structured_vp_allowed_labels,
        allowed_labels_field=args.structured_vp_allowed_labels_field,
        allowed_label_match_mode=args.structured_vp_allowed_label_match_mode,
        primary_nms_iou_threshold=args.primary_nms_iou_threshold,
        proposal_nms_iou_threshold=args.proposal_nms_iou_threshold,
        target_count_field=args.target_count_field,
        duplicate_iou_threshold=args.duplicate_iou_threshold,
        cap_to_target_count=args.cap_to_target_count,
        proposal_selection_policy=args.proposal_selection_policy,
        proposal_min_confidence=args.proposal_min_confidence,
        proposal_allowed_sources=args.proposal_allowed_sources,
        max_proposal_additions_per_record=args.max_proposal_additions_per_record,
    )

    primary_quality = evaluate_vp_summary(
        primary_summary,
        config=VPDetectionQualityConfig(
            prediction_field=args.primary_prediction_field,
            reference_field="target",
            box_format=args.structured_vp_box_format,
            marker_style=args.structured_vp_marker_style,
            filter_policy=args.primary_filter_policy,
            nms_iou_threshold=args.primary_nms_iou_threshold,
            allowed_labels=args.structured_vp_allowed_labels,
            allowed_labels_field=args.structured_vp_allowed_labels_field,
            label_match_mode=args.vp_label_match_mode,
            allowed_label_match_mode=args.structured_vp_allowed_label_match_mode,
            max_bad_cases=0,
            iou_threshold=args.quality_iou_threshold,
        ),
    )
    teacher_quality = evaluate_vp_summary(
        replay_summary,
        config=VPDetectionQualityConfig(
            prediction_field="structured_prediction",
            reference_field="target",
            box_format=args.structured_vp_box_format,
            marker_style=args.structured_vp_marker_style,
            filter_policy="none",
            label_match_mode=args.vp_label_match_mode,
            max_bad_cases=0,
            iou_threshold=args.quality_iou_threshold,
        ),
    )

    rows, counters = build_distillation_rows(
        replay_summary.get("records", []),
        primary_quality.get("records", []),
        teacher_quality.get("records", []),
        primary_summary_path=str(primary_summary_path),
        proposal_summary_path=str(proposal_summary_path),
        min_added_boxes=args.min_added_boxes,
        quality_filter=args.quality_filter,
        target_mode=args.distillation_target_mode,
        require_target_count_reached=args.require_target_count_reached,
        max_rows=args.max_rows,
        output_task_type=args.output_task_type,
        source_format=args.source_format,
        box_format=args.structured_vp_box_format,
        marker_style=args.structured_vp_marker_style,
    )
    _write_jsonl(output_path, rows)

    summary_path = Path(args.summary_output).expanduser() if args.summary_output else _default_summary_path(output_path)
    markdown_path = Path(args.markdown_output).expanduser() if args.markdown_output else summary_path.with_suffix(".md")
    summary = {
        "primary_summary_path": str(primary_summary_path),
        "proposal_summary_path": str(proposal_summary_path),
        "output_path": str(output_path),
        "summary_path": str(summary_path),
        "markdown_path": str(markdown_path),
        "input_records": len(replay_summary.get("records", []) or []),
        "output_rows": len(rows),
        "quality_filter": args.quality_filter,
        "distillation_target_mode": args.distillation_target_mode,
        "min_added_boxes": args.min_added_boxes,
        "require_target_count_reached": bool(args.require_target_count_reached),
        "max_rows": args.max_rows,
        "teacher_config": replay_summary.get("target_count_proposal_config", {}),
        "teacher_fill": replay_summary.get("target_count_proposal_fill", {}),
        "primary_quality": _quality_brief(primary_quality),
        "teacher_quality": _quality_brief(teacher_quality),
        **counters,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_path.write_text(render_markdown(summary), encoding="utf-8")
    return summary


def build_distillation_rows(
    replay_records: Sequence[Mapping[str, Any]],
    primary_quality_records: Sequence[Mapping[str, Any]],
    teacher_quality_records: Sequence[Mapping[str, Any]],
    *,
    primary_summary_path: str,
    proposal_summary_path: str,
    min_added_boxes: int = 1,
    quality_filter: str = "none",
    target_mode: str = "teacher",
    require_target_count_reached: bool = False,
    max_rows: Optional[int] = None,
    output_task_type: str = "OPEN_VOCABULARY_DETECTION",
    source_format: str = "vp_proposal_distillation",
    box_format: str = "loc_tokens",
    marker_style: str = "plain",
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    skip_counts: Counter[str] = Counter()
    added_counts: List[int] = []
    delta_tp_total = 0
    delta_fp_total = 0
    delta_fn_total = 0
    delta_f1_total = 0.0

    for position, replay_record in enumerate(replay_records):
        if not isinstance(replay_record, Mapping):
            skip_counts["invalid_record"] += 1
            continue
        added = _safe_int(replay_record.get("target_count_added_box_count"))
        if added < min_added_boxes:
            skip_counts["min_added_boxes"] += 1
            continue
        if require_target_count_reached and not bool(replay_record.get("target_count_reached")):
            skip_counts["target_count_not_reached"] += 1
            continue
        suffix = _distillation_suffix(replay_record, target_mode)
        if not suffix:
            skip_counts["empty_suffix"] += 1
            continue

        primary_quality = primary_quality_records[position] if position < len(primary_quality_records) else {}
        teacher_quality = teacher_quality_records[position] if position < len(teacher_quality_records) else {}
        if not _passes_quality_filter(primary_quality, teacher_quality, quality_filter):
            skip_counts["quality_filter"] += 1
            continue

        row = _distillation_row(
            replay_record,
            primary_quality,
            teacher_quality,
            suffix=suffix,
            position=position,
            primary_summary_path=primary_summary_path,
            proposal_summary_path=proposal_summary_path,
            output_task_type=output_task_type,
            source_format=source_format,
            box_format=box_format,
            marker_style=marker_style,
            target_mode=target_mode,
        )
        rows.append(row)
        added_counts.append(added)
        delta_tp_total += _safe_int(row.get("distillation_delta_tp"))
        delta_fp_total += _safe_int(row.get("distillation_delta_fp"))
        delta_fn_total += _safe_int(row.get("distillation_delta_fn"))
        delta_f1_total += float(row.get("distillation_delta_f1", 0.0) or 0.0)
        if max_rows is not None and len(rows) >= max(0, int(max_rows)):
            break

    return rows, {
        "skip_counts": dict(skip_counts),
        "total_added_boxes_in_output": sum(added_counts),
        "avg_added_boxes_in_output": _mean(added_counts),
        "delta_tp_total": delta_tp_total,
        "delta_fp_total": delta_fp_total,
        "delta_fn_total": delta_fn_total,
        "avg_delta_f1": delta_f1_total / len(rows) if rows else 0.0,
        "bucket_counts": _bucket_counts(rows),
        "top_labels": [
            {"label": label, "count": count}
            for label, count in Counter(str(row.get("query_label", "") or "") for row in rows).most_common(20)
            if label
        ],
    }


def render_markdown(summary: Mapping[str, Any]) -> str:
    primary = dict(summary.get("primary_quality", {}) or {})
    teacher = dict(summary.get("teacher_quality", {}) or {})
    lines = [
        "# VP Proposal Distillation Dataset",
        "",
        f"- Input records: `{int(summary.get('input_records', 0) or 0)}`",
        f"- Output rows: `{int(summary.get('output_rows', 0) or 0)}`",
        f"- Quality filter: `{summary.get('quality_filter')}`",
        f"- Target mode: `{summary.get('distillation_target_mode')}`",
        f"- Min added boxes: `{int(summary.get('min_added_boxes', 0) or 0)}`",
        f"- Output added boxes: `{int(summary.get('total_added_boxes_in_output', 0) or 0)}`",
        f"- Avg added boxes / row: `{float(summary.get('avg_added_boxes_in_output', 0.0) or 0.0):.4f}`",
        f"- Delta TP / FP / FN: `{int(summary.get('delta_tp_total', 0) or 0)}` / "
        f"`{int(summary.get('delta_fp_total', 0) or 0)}` / "
        f"`{int(summary.get('delta_fn_total', 0) or 0)}`",
        f"- Primary F1 / P / R: `{float(primary.get('f1', 0.0) or 0.0):.4f}` / "
        f"`{float(primary.get('precision', 0.0) or 0.0):.4f}` / "
        f"`{float(primary.get('recall', 0.0) or 0.0):.4f}`",
        f"- Teacher F1 / P / R: `{float(teacher.get('f1', 0.0) or 0.0):.4f}` / "
        f"`{float(teacher.get('precision', 0.0) or 0.0):.4f}` / "
        f"`{float(teacher.get('recall', 0.0) or 0.0):.4f}`",
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
    parser.add_argument("--primary-summary", required=True)
    parser.add_argument("--proposal-summary", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--markdown-output", default=None)
    parser.add_argument("--output-task-type", default="OPEN_VOCABULARY_DETECTION")
    parser.add_argument("--source-format", default="vp_proposal_distillation")
    parser.add_argument("--min-added-boxes", type=int, default=1)
    parser.add_argument("--distillation-target-mode", default="teacher", choices=["teacher", "reference"])
    parser.add_argument("--quality-filter", default="none", choices=["none", "non_regression", "improvement"])
    parser.add_argument("--require-target-count-reached", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--primary-prediction-field", default="raw_prediction")
    parser.add_argument("--primary-filter-policy", default="nms", choices=["none", "auto", "single-target", "nms"])
    parser.add_argument("--proposal-filter-policy", default="none", choices=["none", "auto", "single-target", "nms"])
    parser.add_argument("--primary-nms-iou-threshold", type=float, default=0.5)
    parser.add_argument("--proposal-nms-iou-threshold", type=float, default=0.5)
    parser.add_argument("--duplicate-iou-threshold", type=float, default=0.5)
    parser.add_argument("--target-count-field", default="query_box_count")
    parser.add_argument("--cap-to-target-count", action="store_true")
    parser.add_argument(
        "--proposal-selection-policy",
        default="source_order",
        choices=["source_order", "confidence", "edge_density", "area_small", "area_large"],
    )
    parser.add_argument("--proposal-min-confidence", type=float, default=None)
    parser.add_argument("--proposal-allowed-sources", default=None)
    parser.add_argument("--max-proposal-additions-per-record", type=int, default=None)
    parser.add_argument("--structured-vp-box-format", default="loc_tokens", choices=["loc_tokens", "json"])
    parser.add_argument("--structured-vp-marker-style", default="plain", choices=["plain", "special"])
    parser.add_argument("--structured-vp-allowed-labels", default=None)
    parser.add_argument("--structured-vp-allowed-labels-field", default="query_label")
    parser.add_argument("--structured-vp-allowed-label-match-mode", default="strict", choices=["strict", "contains"])
    parser.add_argument("--vp-label-match-mode", default="strict", choices=["strict", "contains"])
    parser.add_argument("--quality-iou-threshold", type=float, default=0.5)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    print(json.dumps(run(parse_args(argv)), indent=2, ensure_ascii=False))
    return 0


def _distillation_row(
    replay_record: Mapping[str, Any],
    primary_quality: Mapping[str, Any],
    teacher_quality: Mapping[str, Any],
    *,
    suffix: str,
    position: int,
    primary_summary_path: str,
    proposal_summary_path: str,
    output_task_type: str,
    source_format: str,
    box_format: str,
    marker_style: str,
    target_mode: str,
) -> Dict[str, Any]:
    primary_f1 = _record_f1(primary_quality)
    teacher_f1 = _record_f1(teacher_quality)
    primary_tp = _safe_int(primary_quality.get("true_positives"))
    primary_fp = _safe_int(primary_quality.get("false_positives"))
    primary_fn = _safe_int(primary_quality.get("false_negatives"))
    teacher_tp = _safe_int(teacher_quality.get("true_positives"))
    teacher_fp = _safe_int(teacher_quality.get("false_positives"))
    teacher_fn = _safe_int(teacher_quality.get("false_negatives"))
    return {
        "image": replay_record.get("image"),
        "prefix": replay_record.get("prefix") or f"<{output_task_type}>",
        "suffix": suffix,
        "task_family": "visual_primitive",
        "base_task": replay_record.get("prefix") or output_task_type,
        "source_format": source_format,
        "query_label": replay_record.get("query_label") or replay_record.get("text_input"),
        "text_input": replay_record.get("text_input") or replay_record.get("query_label"),
        "query_box_count": replay_record.get("query_box_count", replay_record.get("target_count")),
        "gt_box_count": replay_record.get("gt_box_count"),
        "vp_box_format": box_format,
        "vp_marker_style": marker_style,
        "vp_task_type": output_task_type,
        "distillation_source": "target_count_proposal_replay",
        "distillation_target_mode": target_mode,
        "distillation_source_index": replay_record.get("index", position),
        "distillation_primary_summary_path": primary_summary_path,
        "distillation_proposal_summary_path": proposal_summary_path,
        "distillation_added_box_count": replay_record.get("target_count_added_box_count", 0),
        "distillation_deficit_before": replay_record.get("target_count_deficit_before", 0),
        "distillation_deficit_after": replay_record.get("target_count_deficit_after", 0),
        "distillation_target_count_reached": bool(replay_record.get("target_count_reached")),
        "distillation_primary_pred_box_count": replay_record.get("primary_pred_box_count", 0),
        "distillation_teacher_pred_box_count": replay_record.get("pred_box_count", 0),
        "distillation_primary_tp": primary_tp,
        "distillation_primary_fp": primary_fp,
        "distillation_primary_fn": primary_fn,
        "distillation_primary_f1": primary_f1,
        "distillation_teacher_tp": teacher_tp,
        "distillation_teacher_fp": teacher_fp,
        "distillation_teacher_fn": teacher_fn,
        "distillation_teacher_f1": teacher_f1,
        "distillation_delta_tp": teacher_tp - primary_tp,
        "distillation_delta_fp": teacher_fp - primary_fp,
        "distillation_delta_fn": teacher_fn - primary_fn,
        "distillation_delta_f1": teacher_f1 - primary_f1,
    }


def _passes_quality_filter(
    primary_quality: Mapping[str, Any],
    teacher_quality: Mapping[str, Any],
    mode: str,
) -> bool:
    mode = str(mode or "none").lower()
    if mode == "none":
        return True
    primary_f1 = _record_f1(primary_quality)
    teacher_f1 = _record_f1(teacher_quality)
    primary_tp = _safe_int(primary_quality.get("true_positives"))
    teacher_tp = _safe_int(teacher_quality.get("true_positives"))
    eps = 1e-12
    if mode == "non_regression":
        return teacher_f1 + eps >= primary_f1 and teacher_tp >= primary_tp
    if mode == "improvement":
        return teacher_f1 > primary_f1 + eps or teacher_tp > primary_tp
    raise ValueError("quality_filter must be one of: none, non_regression, improvement")


def _distillation_suffix(record: Mapping[str, Any], target_mode: str) -> str:
    mode = str(target_mode or "teacher").strip().lower()
    if mode == "teacher":
        return str(record.get("structured_prediction", "") or "").strip()
    if mode == "reference":
        return str(record.get("target", "") or "").strip()
    raise ValueError("target_mode must be one of: teacher, reference")


def _record_f1(record: Mapping[str, Any]) -> float:
    tp = _safe_int(record.get("true_positives"))
    fp = _safe_int(record.get("false_positives"))
    fn = _safe_int(record.get("false_negatives"))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


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
        box_count = _safe_int(row.get("query_box_count"))
        if box_count <= 1:
            counts["single"] += 1
        elif box_count <= 3:
            counts["medium"] += 1
        else:
            counts["dense"] += 1
    return {bucket: int(counts.get(bucket, 0)) for bucket in ("single", "medium", "dense")}


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _load_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object at {path}")
    return data


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _mean(values: Sequence[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def _default_summary_path(output_path: Path) -> Path:
    suffix = "".join(output_path.suffixes)
    name = output_path.name[: -len(suffix)] if suffix else output_path.name
    return output_path.with_name(f"{name}_summary.json")


if __name__ == "__main__":
    raise SystemExit(main())
