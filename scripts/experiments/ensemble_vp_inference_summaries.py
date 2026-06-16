#!/usr/bin/env python3
"""Build a GT-free VP candidate ensemble from multiple inference summaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from florence_forge.evaluation.structured_vp_decoder import (
    StructuredVisualPrimitiveDecoder,
    filter_native_detections,
    native_detections_to_vp,
    resolve_structured_vp_filter_caps,
)


def _load_summary(path: Path) -> Dict[str, Any]:
    return json.loads(path.expanduser().read_text(encoding="utf-8"))


def _parse_named_summary(value: str) -> tuple[str, Path]:
    name, sep, path = str(value).partition("=")
    if sep:
        return _safe_label(name), Path(path).expanduser()
    path_obj = Path(value).expanduser()
    return _safe_label(path_obj.parent.name or path_obj.stem), path_obj


def _safe_label(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value).strip())
    return cleaned or "summary"


def _resolve_field_value(row: Mapping[str, Any], field: Optional[str]) -> Any:
    if not field:
        return None
    value: Any = row
    for part in str(field).split("."):
        if not isinstance(value, Mapping):
            return None
        value = value.get(part)
    return value


def _record_key(record: Mapping[str, Any], index: int) -> str:
    return "|".join([
        str(record.get("image", "")),
        str(record.get("prefix", "")),
        str(record.get("text_input", record.get("query_label", ""))),
        str(record.get("index", index)),
    ])


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _merge_records(
    *,
    base_record: Dict[str, Any],
    member_records: Sequence[Dict[str, Any]],
    member_names: Sequence[str],
    decoder: StructuredVisualPrimitiveDecoder,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    allowed_labels = args.structured_vp_allowed_labels or _resolve_field_value(
        base_record,
        args.structured_vp_allowed_labels_field,
    )
    caps = resolve_structured_vp_filter_caps(
        policy=args.structured_vp_filter_policy,
        task_prompt=base_record.get("prefix"),
        max_boxes_per_label=args.structured_vp_max_boxes_per_label,
        max_total_boxes=args.structured_vp_max_total_boxes,
        nms_iou_threshold=args.structured_vp_nms_iou_threshold,
        allowed_labels=allowed_labels,
    )

    raw_detections: List[Dict[str, Any]] = []
    source_counts: Dict[str, int] = {}
    for name, record in zip(member_names, member_records):
        text = str(record.get(args.prediction_field, record.get("raw_prediction", "")) or "")
        decoded = decoder.decode(
            text,
            max_boxes_per_label=None,
            max_total_boxes=None,
            nms_iou_threshold=None,
            allowed_labels=caps["allowed_labels"],
            allowed_label_match_mode=args.structured_vp_allowed_label_match_mode,
        )
        source_counts[name] = len(decoded.detections)
        for detection in decoded.detections:
            item = dict(detection)
            item["ensemble_source"] = name
            raw_detections.append(item)

    merged_detections = filter_native_detections(
        raw_detections,
        max_boxes_per_label=caps["max_boxes_per_label"],
        max_total_boxes=caps["max_total_boxes"],
        nms_iou_threshold=caps["nms_iou_threshold"],
        allowed_labels=caps["allowed_labels"],
        allowed_label_match_mode=args.structured_vp_allowed_label_match_mode,
    )
    merged_text = native_detections_to_vp(
        merged_detections,
        box_format=args.structured_vp_box_format,
        marker_style=args.structured_vp_marker_style,
    )

    merged = dict(base_record)
    merged.update({
        "prediction": merged_text,
        "raw_prediction": merged_text,
        "structured_prediction": merged_text,
        "structured_source": "ensemble",
        "pred_box_count": len(merged_detections),
        "structured_pred_box_count": len(merged_detections),
        "structured_raw_detection_count": len(raw_detections),
        "structured_filtered_detection_count": max(0, len(raw_detections) - len(merged_detections)),
        "used_structured_vp_decoder": False,
        "vp_format_valid": bool(merged_detections),
        "structured_vp_format_valid": bool(merged_detections),
        "ensemble_member_detection_counts": source_counts,
        "ensemble_member_count": len(member_records),
        "structured_vp_filter_policy": args.structured_vp_filter_policy,
        "structured_vp_resolved_max_boxes_per_label": caps["max_boxes_per_label"],
        "structured_vp_resolved_max_total_boxes": caps["max_total_boxes"],
        "structured_vp_resolved_nms_iou_threshold": caps["nms_iou_threshold"],
        "structured_vp_resolved_allowed_labels": caps["allowed_labels"],
        "structured_vp_allowed_label_match_mode": args.structured_vp_allowed_label_match_mode,
    })
    return merged


def run(args: argparse.Namespace) -> Dict[str, Any]:
    named_paths = [_parse_named_summary(value) for value in args.summary]
    if len(named_paths) < 2:
        raise ValueError("At least two --summary values are required for an ensemble diagnostic.")

    summaries = [(name, _load_summary(path), path) for name, path in named_paths]
    base_records = list(summaries[0][1].get("records", []) or [])
    if not base_records:
        raise ValueError(f"Base summary has no records: {summaries[0][2]}")

    member_record_maps: List[Dict[str, Dict[str, Any]]] = []
    for _name, summary, _path in summaries:
        records = list(summary.get("records", []) or [])
        member_record_maps.append({
            _record_key(record, index): record
            for index, record in enumerate(records)
        })

    decoder = StructuredVisualPrimitiveDecoder(
        box_format=args.structured_vp_box_format,
        marker_style=args.structured_vp_marker_style,
    )
    member_names = [name for name, _summary, _path in summaries]
    output_records: List[Dict[str, Any]] = []
    skipped_records = 0

    for index, base_record in enumerate(base_records):
        key = _record_key(base_record, index)
        member_records = [record_map.get(key) for record_map in member_record_maps]
        if any(record is None for record in member_records):
            skipped_records += 1
            continue
        output_records.append(_merge_records(
            base_record=dict(base_record),
            member_records=[dict(record) for record in member_records if record is not None],
            member_names=member_names,
            decoder=decoder,
            args=args,
        ))

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "vp_inference_visualization_summary.json"
    avg_pred_boxes = _mean([float(record.get("pred_box_count", 0) or 0) for record in output_records])
    avg_gt_boxes = _mean([float(record.get("gt_box_count", 0) or 0) for record in output_records])
    overgenerated = sum(
        1 for record in output_records
        if int(record.get("pred_box_count", 0) or 0) > int(record.get("gt_box_count", 0) or 0)
    )
    source_counts: Dict[str, int] = {}
    for record in output_records:
        for name, count in dict(record.get("ensemble_member_detection_counts", {}) or {}).items():
            source_counts[name] = source_counts.get(name, 0) + int(count)

    summary = {
        "output_dir": str(output_dir),
        "summary_paths": {name: str(path) for name, _summary, path in summaries},
        "num_samples": len(output_records),
        "skipped_records": skipped_records,
        "prediction_field": args.prediction_field,
        "structured_vp_mode": "ensemble",
        "structured_vp_box_format": args.structured_vp_box_format,
        "structured_vp_marker_style": args.structured_vp_marker_style,
        "structured_vp_filter_policy": args.structured_vp_filter_policy,
        "structured_vp_max_boxes_per_label": args.structured_vp_max_boxes_per_label,
        "structured_vp_max_total_boxes": args.structured_vp_max_total_boxes,
        "structured_vp_nms_iou_threshold": args.structured_vp_nms_iou_threshold,
        "structured_vp_allowed_labels": args.structured_vp_allowed_labels,
        "structured_vp_allowed_labels_field": args.structured_vp_allowed_labels_field,
        "structured_vp_allowed_label_match_mode": args.structured_vp_allowed_label_match_mode,
        "avg_pred_boxes": avg_pred_boxes,
        "avg_gt_boxes": avg_gt_boxes,
        "box_count_overgeneration_ratio": overgenerated / len(output_records) if output_records else 0.0,
        "ensemble_source_detection_counts": source_counts,
        "records": output_records,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        action="append",
        required=True,
        help="Inference summary path. Use name=path to set a stable member label.",
    )
    parser.add_argument("--output-dir", default=".codex_reports/florence_vp_ensemble")
    parser.add_argument("--prediction-field", default="raw_prediction")
    parser.add_argument("--structured-vp-box-format", default="loc_tokens", choices=["loc_tokens", "json"])
    parser.add_argument("--structured-vp-marker-style", default="plain", choices=["special", "plain"])
    parser.add_argument("--structured-vp-filter-policy", default="nms", choices=["none", "auto", "single-target", "nms"])
    parser.add_argument("--structured-vp-max-boxes-per-label", type=int, default=None)
    parser.add_argument("--structured-vp-max-total-boxes", type=int, default=None)
    parser.add_argument("--structured-vp-nms-iou-threshold", type=float, default=0.5)
    parser.add_argument("--structured-vp-allowed-labels", default=None)
    parser.add_argument("--structured-vp-allowed-labels-field", default="text_input")
    parser.add_argument(
        "--structured-vp-allowed-label-match-mode",
        default="strict",
        choices=["strict", "contains"],
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    print(json.dumps(run(parse_args(argv)), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
