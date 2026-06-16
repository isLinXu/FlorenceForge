#!/usr/bin/env python3
"""Replay target-count-aware proposal filling for VP summaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from florence_forge.evaluation.structured_vp_decoder import (
    StructuredVisualPrimitiveDecoder,
    filter_native_detections,
    labels_match,
    native_detections_to_vp,
    normalize_allowed_labels,
    resolve_structured_vp_filter_caps,
)
from florence_forge.evaluation.vp_detection_quality import (
    VPDetectionQualityConfig,
    compute_bbox_iou,
    evaluate_vp_summary,
    render_vp_detection_quality_markdown,
)
from florence_forge.evaluation.visual_primitive_parser import VisualPrimitiveParser


def run(args: argparse.Namespace) -> Dict[str, Any]:
    primary_summary_path = Path(args.primary_summary).expanduser()
    proposal_summary_path = Path(args.proposal_summary).expanduser()
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

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "vp_target_count_proposal_summary.json"
    quality_path = output_dir / "vp_target_count_proposal_quality.json"
    markdown_path = output_dir / "vp_target_count_proposal_replay.md"
    quality_markdown_path = output_dir / "vp_target_count_proposal_quality.md"

    replay_summary["summary_path"] = str(summary_path)
    quality = evaluate_vp_summary(
        replay_summary,
        config=VPDetectionQualityConfig(
            prediction_field="structured_prediction",
            reference_field="target",
            box_format=args.structured_vp_box_format,
            marker_style=args.structured_vp_marker_style,
            filter_policy="none",
            label_match_mode=args.vp_label_match_mode,
            max_bad_cases=args.quality_max_bad_cases,
            iou_threshold=args.quality_iou_threshold,
        ),
    )
    quality["quality_json_path"] = str(quality_path)
    replay_report = {
        "summary_path": str(summary_path),
        "quality_path": str(quality_path),
        "markdown_path": str(markdown_path),
        "quality_markdown_path": str(quality_markdown_path),
        "fill_summary": replay_summary.get("target_count_proposal_fill", {}),
        "quality": {
            "precision": quality.get("precision", 0.0),
            "recall": quality.get("recall", 0.0),
            "f1": quality.get("f1", 0.0),
            "true_positives": quality.get("true_positives", 0),
            "false_positives": quality.get("false_positives", 0),
            "false_negatives": quality.get("false_negatives", 0),
            "avg_pred_boxes": quality.get("avg_pred_boxes", 0.0),
            "avg_gt_boxes": quality.get("avg_gt_boxes", 0.0),
        },
    }
    replay_summary["target_count_proposal_report"] = replay_report

    summary_path.write_text(json.dumps(replay_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    quality_path.write_text(json.dumps(quality, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_path.write_text(render_target_count_proposal_markdown(replay_summary, quality), encoding="utf-8")
    quality_markdown_path.write_text(render_vp_detection_quality_markdown(quality), encoding="utf-8")
    return replay_report


def build_target_count_proposal_summary(
    primary_summary: Mapping[str, Any],
    proposal_summary: Mapping[str, Any],
    *,
    primary_summary_path: Optional[str] = None,
    proposal_summary_path: Optional[str] = None,
    primary_policy: str = "nms",
    proposal_policy: str = "nms",
    box_format: str = "loc_tokens",
    marker_style: str = "plain",
    allowed_labels: Optional[Union[Sequence[str], str]] = None,
    allowed_labels_field: Optional[str] = "query_label",
    allowed_label_match_mode: str = "strict",
    primary_nms_iou_threshold: Optional[float] = 0.5,
    proposal_nms_iou_threshold: Optional[float] = 0.5,
    target_count_field: str = "query_box_count",
    duplicate_iou_threshold: float = 0.5,
    cap_to_target_count: bool = False,
    proposal_selection_policy: str = "source_order",
    proposal_min_confidence: Optional[float] = None,
    proposal_allowed_sources: Optional[str] = None,
    max_proposal_additions_per_record: Optional[int] = None,
) -> Dict[str, Any]:
    """Fill primary detections with proposal detections up to target count."""

    parser = VisualPrimitiveParser()
    decoder = StructuredVisualPrimitiveDecoder(
        box_format=box_format,
        marker_style=marker_style,
    )
    primary_records = _as_records(primary_summary)
    proposal_records_list = _as_records(proposal_summary)
    proposal_records = _index_records(proposal_records_list)
    records: List[Dict[str, Any]] = []
    added_total = 0
    filled_records = 0
    target_reached_records = 0
    target_deficit_before_total = 0
    target_deficit_after_total = 0
    missing_proposal_records = 0

    for position, primary_record in enumerate(primary_records):
        key = _record_key(primary_record, position)
        proposal_record = proposal_records.get(key)
        if proposal_record is None:
            proposal_record = proposal_records_list[position] if position < len(proposal_records_list) else {}
            missing_proposal_records += int(not bool(proposal_record))

        resolved_allowed_labels = _resolve_allowed_labels(
            primary_record,
            explicit_allowed_labels=allowed_labels,
            allowed_labels_field=allowed_labels_field,
        )
        primary_detections, primary_raw_count, primary_filtered_count = _decode_record(
            primary_record,
            decoder=decoder,
            policy=primary_policy,
            nms_iou_threshold=primary_nms_iou_threshold,
            allowed_labels=resolved_allowed_labels,
            allowed_label_match_mode=allowed_label_match_mode,
            box_format=box_format,
            marker_style=marker_style,
        )
        proposal_detections, proposal_raw_count, proposal_filtered_count = _decode_record(
            proposal_record,
            decoder=decoder,
            policy=proposal_policy,
            nms_iou_threshold=proposal_nms_iou_threshold,
            allowed_labels=resolved_allowed_labels,
            allowed_label_match_mode=allowed_label_match_mode,
            box_format=box_format,
            marker_style=marker_style,
        )
        proposal_detections = _prepare_proposal_detections(
            proposal_detections,
            selection_policy=proposal_selection_policy,
            min_confidence=proposal_min_confidence,
            allowed_sources=proposal_allowed_sources,
        )
        target_text = str(primary_record.get("target", "") or proposal_record.get("target", "") or "")
        target_count = _target_count(primary_record, target_text, parser, target_count_field)
        selected = [dict(detection) for detection in primary_detections]
        if cap_to_target_count and target_count > 0:
            selected = selected[:target_count]

        deficit_before = max(0, target_count - len(selected))
        added: List[Dict[str, Any]] = []
        addition_limit = (
            max(0, int(max_proposal_additions_per_record))
            if max_proposal_additions_per_record is not None else None
        )
        if target_count > 0:
            for proposal in proposal_detections:
                if len(selected) >= target_count:
                    break
                if addition_limit is not None and len(added) >= addition_limit:
                    break
                if _is_duplicate_detection(
                    proposal,
                    selected,
                    threshold=duplicate_iou_threshold,
                    label_match_mode=allowed_label_match_mode,
                ):
                    continue
                copied = dict(proposal)
                selected.append(copied)
                added.append(copied)

        deficit_after = max(0, target_count - len(selected))
        added_total += len(added)
        filled_records += int(bool(added))
        target_reached_records += int(target_count > 0 and len(selected) >= target_count)
        target_deficit_before_total += deficit_before
        target_deficit_after_total += deficit_after
        structured_prediction = native_detections_to_vp(
            selected,
            box_format=box_format,
            marker_style=marker_style,
        )
        records.append({
            "index": primary_record.get("index", position),
            "image": primary_record.get("image") or proposal_record.get("image"),
            "prefix": primary_record.get("prefix") or proposal_record.get("prefix"),
            "query_label": primary_record.get("query_label") or proposal_record.get("query_label"),
            "text_input": primary_record.get("text_input") or proposal_record.get("text_input"),
            "query_box_count": primary_record.get("query_box_count", proposal_record.get("query_box_count")),
            "gt_box_count": primary_record.get("gt_box_count", proposal_record.get("gt_box_count")),
            "target": target_text,
            "raw_prediction": str(primary_record.get("raw_prediction", "") or primary_record.get("prediction", "") or ""),
            "proposal_raw_prediction": str(proposal_record.get("raw_prediction", "") or proposal_record.get("prediction", "") or ""),
            "structured_prediction": structured_prediction,
            "target_count": target_count,
            "primary_pred_box_count": len(primary_detections),
            "proposal_pred_box_count": len(proposal_detections),
            "pred_box_count": len(selected),
            "target_count_deficit_before": deficit_before,
            "target_count_deficit_after": deficit_after,
            "target_count_added_box_count": len(added),
            "target_count_reached": target_count > 0 and len(selected) >= target_count,
            "structured_raw_detection_count": primary_raw_count,
            "structured_filtered_detection_count": primary_filtered_count,
            "proposal_raw_detection_count": proposal_raw_count,
            "proposal_filtered_detection_count": proposal_filtered_count,
            "structured_vp_format_valid": bool(selected),
            "used_target_count_proposal_replay": True,
        })

    num_records = len(records)
    return {
        "model_path": primary_summary.get("model_path"),
        "adapter_dir": primary_summary.get("adapter_dir"),
        "data_path": primary_summary.get("data_path"),
        "primary_summary_path": primary_summary_path,
        "proposal_summary_path": proposal_summary_path,
        "num_samples": num_records,
        "structured_vp_decode": True,
        "structured_vp_mode": "target_count_proposal_replay",
        "structured_vp_box_format": box_format,
        "structured_vp_marker_style": marker_style,
        "target_count_proposal_config": {
            "primary_filter_policy": primary_policy,
            "proposal_filter_policy": proposal_policy,
            "allowed_labels": allowed_labels,
            "allowed_labels_field": allowed_labels_field,
            "allowed_label_match_mode": allowed_label_match_mode,
            "primary_nms_iou_threshold": primary_nms_iou_threshold,
            "proposal_nms_iou_threshold": proposal_nms_iou_threshold,
            "target_count_field": target_count_field,
            "duplicate_iou_threshold": duplicate_iou_threshold,
            "cap_to_target_count": cap_to_target_count,
            "proposal_selection_policy": proposal_selection_policy,
            "proposal_min_confidence": proposal_min_confidence,
            "proposal_allowed_sources": proposal_allowed_sources,
            "max_proposal_additions_per_record": max_proposal_additions_per_record,
        },
        "target_count_proposal_fill": {
            "num_records": num_records,
            "filled_records": filled_records,
            "target_reached_records": target_reached_records,
            "missing_proposal_records": missing_proposal_records,
            "added_proposal_boxes": added_total,
            "target_deficit_before": target_deficit_before_total,
            "target_deficit_after": target_deficit_after_total,
            "deficit_closure_ratio": (
                (target_deficit_before_total - target_deficit_after_total) / target_deficit_before_total
                if target_deficit_before_total else 0.0
            ),
        },
        "records": records,
    }


def render_target_count_proposal_markdown(
    summary: Mapping[str, Any],
    quality: Mapping[str, Any],
) -> str:
    fill = dict(summary.get("target_count_proposal_fill", {}) or {})
    lines = [
        "# VP Target-Count Proposal Replay",
        "",
        f"- Records: `{int(fill.get('num_records', 0) or 0)}`",
        f"- Filled records: `{int(fill.get('filled_records', 0) or 0)}`",
        f"- Added proposal boxes: `{int(fill.get('added_proposal_boxes', 0) or 0)}`",
        f"- Deficit before / after: `{int(fill.get('target_deficit_before', 0) or 0)}` / "
        f"`{int(fill.get('target_deficit_after', 0) or 0)}`",
        f"- Deficit closure ratio: `{float(fill.get('deficit_closure_ratio', 0.0) or 0.0):.4f}`",
        f"- Quality F1 / precision / recall: `{float(quality.get('f1', 0.0) or 0.0):.4f}` / "
        f"`{float(quality.get('precision', 0.0) or 0.0):.4f}` / "
        f"`{float(quality.get('recall', 0.0) or 0.0):.4f}`",
        "",
        "## Filled Records",
        "",
        "| index | label | primary/proposal/final/target | added | deficit before/after |",
        "| ---: | --- | ---: | ---: | ---: |",
    ]
    filled_rows = [
        record for record in list(summary.get("records", []) or [])
        if int(record.get("target_count_added_box_count", 0) or 0) > 0
    ][:20]
    if not filled_rows:
        lines.append("| - | - | - | - | - |")
    for record in filled_rows:
        lines.append(
            f"| {record.get('index')} "
            f"| `{record.get('query_label') or record.get('text_input')}` "
            f"| {record.get('primary_pred_box_count')}/{record.get('proposal_pred_box_count')}/"
            f"{record.get('pred_box_count')}/{record.get('target_count')} "
            f"| {record.get('target_count_added_box_count')} "
            f"| {record.get('target_count_deficit_before')}/{record.get('target_count_deficit_after')} |"
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-summary", required=True)
    parser.add_argument("--proposal-summary", required=True)
    parser.add_argument("--output-dir", default=".codex_reports/florence_vp_target_count_proposals")
    parser.add_argument("--primary-filter-policy", default="nms", choices=["none", "auto", "single-target", "nms"])
    parser.add_argument("--proposal-filter-policy", default="nms", choices=["none", "auto", "single-target", "nms"])
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
    parser.add_argument(
        "--structured-vp-allowed-label-match-mode",
        default="strict",
        choices=["strict", "contains"],
    )
    parser.add_argument("--vp-label-match-mode", default="strict", choices=["strict", "contains"])
    parser.add_argument("--quality-iou-threshold", type=float, default=0.5)
    parser.add_argument("--quality-max-bad-cases", type=int, default=20)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    print(json.dumps(run(parse_args(argv)), indent=2, ensure_ascii=False))
    return 0


def _decode_record(
    record: Mapping[str, Any],
    *,
    decoder: StructuredVisualPrimitiveDecoder,
    policy: str,
    nms_iou_threshold: Optional[float],
    allowed_labels: Optional[Union[Sequence[str], str]],
    allowed_label_match_mode: str,
    box_format: str,
    marker_style: str,
) -> Tuple[List[Dict[str, object]], int, int]:
    caps = resolve_structured_vp_filter_caps(
        policy=policy,
        task_prompt=record.get("prefix") or record.get("task_prompt"),
        nms_iou_threshold=nms_iou_threshold,
        allowed_labels=allowed_labels,
    )
    candidate_detections = _normalize_proposal_candidates(record)
    if candidate_detections:
        filtered = filter_native_detections(
            candidate_detections,
            max_boxes_per_label=caps["max_boxes_per_label"],
            max_total_boxes=caps["max_total_boxes"],
            nms_iou_threshold=caps["nms_iou_threshold"],
            allowed_labels=caps["allowed_labels"],
            allowed_label_match_mode=allowed_label_match_mode,
        )
        return filtered, len(candidate_detections), max(0, len(candidate_detections) - len(filtered))

    text = str(record.get("raw_prediction", "") or record.get("prediction", "") or record.get("structured_prediction", "") or "")
    decoded = decoder.decode(
        text,
        box_format=box_format,
        marker_style=marker_style,
        max_boxes_per_label=caps["max_boxes_per_label"],
        max_total_boxes=caps["max_total_boxes"],
        nms_iou_threshold=caps["nms_iou_threshold"],
        allowed_labels=caps["allowed_labels"],
        allowed_label_match_mode=allowed_label_match_mode,
    )
    return [dict(detection) for detection in decoded.detections], decoded.raw_detection_count, decoded.filtered_detection_count


def _prepare_proposal_detections(
    detections: Sequence[Mapping[str, Any]],
    *,
    selection_policy: str,
    min_confidence: Optional[float],
    allowed_sources: Optional[str],
) -> List[Dict[str, Any]]:
    """Apply proposal-only filters and ordering before target-count filling."""

    policy = str(selection_policy or "source_order").strip().lower().replace("-", "_")
    if policy not in {"source_order", "confidence", "edge_density", "area_small", "area_large"}:
        raise ValueError(
            "proposal selection policy must be one of: "
            "source_order, confidence, edge_density, area_small, area_large"
        )
    threshold = float(min_confidence) if min_confidence is not None else None
    source_filter = _parse_allowed_sources(allowed_sources)
    prepared: List[Dict[str, Any]] = []
    for detection in detections:
        item = dict(detection)
        if threshold is not None and _proposal_confidence(item) < threshold:
            continue
        if source_filter and _proposal_source(item).lower() not in source_filter:
            continue
        prepared.append(item)

    if policy == "source_order":
        return prepared
    if policy == "confidence":
        return sorted(prepared, key=lambda item: (_proposal_confidence(item), _proposal_edge_density(item)), reverse=True)
    if policy == "edge_density":
        return sorted(prepared, key=lambda item: (_proposal_edge_density(item), _proposal_confidence(item)), reverse=True)
    if policy == "area_small":
        return sorted(prepared, key=lambda item: (_proposal_area_ratio(item), -_proposal_confidence(item)))
    return sorted(prepared, key=lambda item: (_proposal_area_ratio(item), _proposal_confidence(item)), reverse=True)


def _normalize_proposal_candidates(record: Mapping[str, Any]) -> List[Dict[str, Any]]:
    candidates = record.get("proposal_candidates")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        return []
    default_label = _candidate_default_label(record)
    normalized: List[Dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        bbox = candidate.get("bbox")
        if not isinstance(bbox, Sequence) or isinstance(bbox, (str, bytes)) or len(bbox) != 4:
            continue
        try:
            parsed_bbox = [int(round(float(value))) for value in bbox]
        except (TypeError, ValueError):
            continue
        if parsed_bbox[2] <= parsed_bbox[0] or parsed_bbox[3] <= parsed_bbox[1]:
            continue
        item = dict(candidate)
        item["bbox"] = [max(0, min(999, value)) for value in parsed_bbox]
        item["label"] = str(candidate.get("label") or default_label).strip()
        if "confidence" in item:
            item["confidence"] = _safe_float(item.get("confidence"), default=1.0)
        elif "score" in item:
            item["confidence"] = _safe_float(item.get("score"), default=1.0)
        else:
            item["confidence"] = 1.0
        if "proposal_source" not in item and "source" in item:
            item["proposal_source"] = item.get("source")
        normalized.append(item)
    return normalized


def _is_duplicate_detection(
    candidate: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
    label_match_mode: str,
) -> bool:
    for existing in selected:
        if not labels_match(candidate.get("label"), existing.get("label"), mode=label_match_mode):
            continue
        if compute_bbox_iou(candidate.get("bbox"), existing.get("bbox")) >= threshold:
            return True
    return False


def _target_count(
    record: Mapping[str, Any],
    target_text: str,
    parser: VisualPrimitiveParser,
    field: str,
) -> int:
    for key in (field, "query_box_count", "gt_box_count"):
        value = record.get(key)
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = 0
        if parsed > 0:
            return parsed
    return len(parser.parse_detections(target_text))


def _resolve_allowed_labels(
    record: Mapping[str, Any],
    *,
    explicit_allowed_labels: Optional[Union[Sequence[str], str]],
    allowed_labels_field: Optional[str],
) -> Optional[List[str]]:
    explicit = normalize_allowed_labels(explicit_allowed_labels)
    if explicit:
        return explicit
    if not allowed_labels_field:
        return None
    for field in str(allowed_labels_field).split(","):
        value = record.get(field.strip())
        resolved = normalize_allowed_labels(value if isinstance(value, (str, list, tuple)) else None)
        if resolved:
            return resolved
    return None


def _candidate_default_label(record: Mapping[str, Any]) -> str:
    for key in ("query_label", "text_input", "label"):
        value = str(record.get(key, "") or "").strip()
        if value:
            return value
    return "proposal"


def _parse_allowed_sources(value: Optional[str]) -> set[str]:
    return {
        item.strip().lower()
        for item in str(value or "").split(",")
        if item.strip()
    }


def _proposal_confidence(detection: Mapping[str, Any]) -> float:
    return _safe_float(detection.get("confidence", detection.get("score", 1.0)), default=1.0)


def _proposal_edge_density(detection: Mapping[str, Any]) -> float:
    return _safe_float(detection.get("proposal_edge_density", detection.get("edge_density", 0.0)), default=0.0)


def _proposal_area_ratio(detection: Mapping[str, Any]) -> float:
    if "proposal_area_ratio" in detection:
        return _safe_float(detection.get("proposal_area_ratio"), default=0.0)
    bbox = detection.get("bbox")
    if not isinstance(bbox, Sequence) or len(bbox) != 4:
        return 0.0
    try:
        x1, y1, x2, y2 = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, (x2 - x1) * (y2 - y1)) / (999.0 * 999.0)


def _proposal_source(detection: Mapping[str, Any]) -> str:
    return str(detection.get("proposal_source") or detection.get("source") or "").strip()


def _safe_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _record_key(record: Mapping[str, Any], position: int) -> str:
    index = record.get("index", position)
    image = record.get("image", "")
    label = record.get("query_label") or record.get("text_input") or ""
    return f"{index}|{image}|{label}"


def _index_records(records: Sequence[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    indexed: Dict[str, Mapping[str, Any]] = {}
    for position, record in enumerate(records):
        indexed[_record_key(record, position)] = record
    return indexed


def _as_records(summary: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    records = summary.get("records", [])
    if not isinstance(records, Sequence):
        return []
    return [record for record in records if isinstance(record, Mapping)]


def _load_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object at {path}")
    return data


if __name__ == "__main__":
    raise SystemExit(main())
