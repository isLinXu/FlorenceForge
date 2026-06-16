#!/usr/bin/env python3
"""Replay structured VP post-filters against saved inference summaries."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from florence_forge.evaluation.structured_vp_decoder import StructuredVisualPrimitiveDecoder
from florence_forge.evaluation.visual_primitive_parser import VisualPrimitiveParser


DEFAULT_FILTER_CONFIGS = (
    "unfiltered:",
    "total1:max_total_boxes=1",
    "per_label1_total2:max_boxes_per_label=1,max_total_boxes=2",
)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    filter_configs = [_parse_filter_config(value) for value in args.filter_config]
    runs: Dict[str, Any] = {}
    for label, summary_path in _parse_summary_args(args.inference_summary):
        source_summary = _load_json(summary_path)
        policy_summaries: Dict[str, Any] = {}
        for config_name, config in filter_configs:
            replayed = replay_summary(
                source_summary,
                max_boxes_per_label=config.get("max_boxes_per_label"),
                max_total_boxes=config.get("max_total_boxes"),
                nms_iou_threshold=config.get("nms_iou_threshold"),
                allowed_labels=config.get("allowed_labels"),
                box_format=args.structured_vp_box_format,
                marker_style=args.structured_vp_marker_style,
            )
            replayed["source_summary_path"] = str(summary_path)
            replayed["filter_policy"] = config_name
            replayed["filter_config"] = config
            policy_path = output_dir / f"{label}_{config_name}_vp_inference_visualization_summary.json"
            replayed["summary_path"] = str(policy_path)
            policy_path.write_text(json.dumps(replayed, indent=2, ensure_ascii=False), encoding="utf-8")
            policy_summaries[config_name] = replayed

        runs[label] = {
            "source_summary_path": str(summary_path),
            "policies": policy_summaries,
        }

    aggregate = {
        "output_dir": str(output_dir),
        "structured_vp_box_format": args.structured_vp_box_format,
        "structured_vp_marker_style": args.structured_vp_marker_style,
        "filter_configs": {name: config for name, config in filter_configs},
        "runs": runs,
    }
    summary_path = output_dir / "filtered_replay_summary.json"
    aggregate["summary_path"] = str(summary_path)
    summary_path.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")
    return aggregate


def replay_summary(
    summary: Mapping[str, Any],
    *,
    max_boxes_per_label: Optional[int],
    max_total_boxes: Optional[int],
    nms_iou_threshold: Optional[float],
    allowed_labels: Any,
    box_format: str = "loc_tokens",
    marker_style: str = "plain",
) -> Dict[str, Any]:
    """Recompute structured VP metrics from saved raw predictions."""

    parser = VisualPrimitiveParser()
    decoder = StructuredVisualPrimitiveDecoder(
        box_format=box_format,
        marker_style=marker_style,
        max_boxes_per_label=max_boxes_per_label,
        max_total_boxes=max_total_boxes,
        nms_iou_threshold=nms_iou_threshold,
        allowed_labels=allowed_labels,
    )
    records: List[Dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    vp_valid_count = 0
    structured_valid_count = 0
    structured_decoder_count = 0
    filtered_detection_total = 0
    raw_detection_total = 0

    for source_record in _as_records(summary):
        raw_prediction = _record_text(source_record)
        target_text = str(source_record.get("target", "") or "")
        gt_count = _gt_box_count(source_record, target_text, parser)
        raw_vp_detections = parser.parse_detections(raw_prediction)
        result = decoder.decode(raw_prediction)
        pred_count = len(result.detections)
        source_counts[result.source] += 1
        vp_valid_count += int(bool(raw_vp_detections))
        structured_valid_count += int(bool(result.detections))
        structured_decoder_count += int(result.used_structured_decoder)
        filtered_detection_total += int(result.filtered_detection_count)
        raw_detection_total += int(result.raw_detection_count)
        records.append({
            "index": source_record.get("index", len(records)),
            "image": source_record.get("image"),
            "raw_prediction": raw_prediction,
            "structured_prediction": result.text,
            "structured_source": result.source,
            "target": target_text,
            "pred_box_count": pred_count,
            "vp_pred_box_count": len(raw_vp_detections),
            "structured_pred_box_count": pred_count,
            "vp_format_valid": bool(raw_vp_detections),
            "structured_vp_format_valid": bool(result.detections),
            "used_structured_vp_decoder": result.used_structured_decoder,
            "structured_raw_detection_count": int(result.raw_detection_count),
            "structured_filtered_detection_count": int(result.filtered_detection_count),
            "gt_box_count": gt_count,
        })

    num_samples = len(records)
    overgenerated_count = sum(
        1 for record in records
        if int(record["pred_box_count"]) > int(record["gt_box_count"])
    )
    avg_pred_boxes = _mean(float(record["pred_box_count"]) for record in records)
    avg_gt_boxes = _mean(float(record["gt_box_count"]) for record in records)

    return {
        "model_path": summary.get("model_path"),
        "adapter_dir": summary.get("adapter_dir"),
        "data_path": summary.get("data_path"),
        "output_dir": summary.get("output_dir"),
        "num_samples": num_samples,
        "structured_vp_decode": True,
        "structured_vp_mode": "replay",
        "structured_vp_box_format": box_format,
        "structured_vp_marker_style": marker_style,
        "structured_vp_max_boxes_per_label": max_boxes_per_label,
        "structured_vp_max_total_boxes": max_total_boxes,
        "structured_vp_nms_iou_threshold": nms_iou_threshold,
        "structured_vp_allowed_labels": allowed_labels,
        "vp_format_valid_ratio": (vp_valid_count / num_samples) if num_samples else 0.0,
        "structured_vp_format_valid_ratio": (
            structured_valid_count / num_samples if num_samples else 0.0
        ),
        "structured_vp_decoder_ratio": (
            structured_decoder_count / num_samples if num_samples else 0.0
        ),
        "structured_source_counts": dict(source_counts),
        "native_fallback_ratio": float(summary.get("native_fallback_ratio", 0.0) or 0.0),
        "avg_pred_boxes": avg_pred_boxes,
        "avg_gt_boxes": avg_gt_boxes,
        "box_count_overgeneration_ratio": (
            overgenerated_count / num_samples if num_samples else 0.0
        ),
        "structured_raw_detection_count": raw_detection_total,
        "structured_filtered_detection_count": filtered_detection_total,
        "records": records,
    }


def _parse_summary_args(values: Sequence[str]) -> List[Tuple[str, Path]]:
    if not values:
        raise ValueError("At least one --inference-summary is required")

    parsed: List[Tuple[str, Path]] = []
    seen: Counter[str] = Counter()
    for value in values:
        label: Optional[str] = None
        path_text = value
        if "=" in value:
            maybe_label, maybe_path = value.split("=", 1)
            if maybe_label and "/" not in maybe_label and "\\" not in maybe_label:
                label = maybe_label
                path_text = maybe_path
        path = Path(path_text).expanduser()
        if label is None:
            label = path.parent.name or path.stem
        label = _safe_label(label)
        seen[label] += 1
        if seen[label] > 1:
            label = f"{label}_{seen[label]}"
        parsed.append((label, path))
    return parsed


def _parse_filter_config(value: str) -> Tuple[str, Dict[str, Any]]:
    name, _, config_text = str(value).partition(":")
    name = _safe_label(name or "custom")
    config: Dict[str, Any] = {}
    if not config_text:
        return name, config
    for item in config_text.split(","):
        item = item.strip()
        if not item:
            continue
        key, sep, raw_value = item.partition("=")
        if not sep:
            raise ValueError(f"Invalid filter config item: {item}")
        key = key.strip()
        if key not in {"max_boxes_per_label", "max_total_boxes", "nms_iou_threshold", "allowed_labels"}:
            raise ValueError(f"Unsupported filter config key: {key}")
        if key == "allowed_labels":
            config[key] = raw_value
            continue
        if key == "nms_iou_threshold":
            float_value = float(raw_value)
            if float_value <= 0.0 or float_value > 1.0:
                raise ValueError(f"{key} must be in (0, 1]")
            config[key] = float_value
            continue
        int_value = int(raw_value)
        if int_value < 1:
            raise ValueError(f"{key} must be >= 1")
        config[key] = int_value
    return name, config


def _safe_label(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value).strip())
    return cleaned or "run"


def _load_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object at {path}")
    return data


def _as_records(summary: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    records = summary.get("records", [])
    if not isinstance(records, Sequence):
        return []
    return [record for record in records if isinstance(record, Mapping)]


def _record_text(record: Mapping[str, Any]) -> str:
    for key in ("raw_prediction", "prediction", "structured_prediction"):
        value = record.get(key)
        if isinstance(value, str):
            return value
    return ""


def _gt_box_count(record: Mapping[str, Any], target_text: str, parser: VisualPrimitiveParser) -> int:
    if target_text:
        return len(parser.parse_detections(target_text))
    return int(record.get("gt_box_count", 0) or 0)


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inference-summary",
        action="append",
        default=[],
        help="Path to vp_inference_visualization_summary.json. Use name=path to set a run label.",
    )
    parser.add_argument("--output-dir", default=".codex_reports/florence_vp_postfilter_replay")
    parser.add_argument("--structured-vp-box-format", default="loc_tokens", choices=["loc_tokens", "json"])
    parser.add_argument("--structured-vp-marker-style", default="plain", choices=["special", "plain"])
    parser.add_argument(
        "--filter-config",
        action="append",
        default=[],
        help=(
            "Named filter policy, e.g. total1:max_total_boxes=1 or "
            "label1_total2:max_boxes_per_label=1,max_total_boxes=2."
        ),
    )
    args = parser.parse_args(argv)
    if not args.filter_config:
        args.filter_config = list(DEFAULT_FILTER_CONFIGS)
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    print(json.dumps(run(parse_args(argv)), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
