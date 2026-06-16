#!/usr/bin/env python3
"""Generate class-agnostic image proposals as a VP inference summary."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PIL import Image

from florence_forge.evaluation.structured_vp_decoder import native_detections_to_vp
from florence_forge.evaluation.vp_detection_quality import compute_bbox_iou
from florence_forge.evaluation.visual_primitive_parser import VisualPrimitiveParser


DEFAULT_METHODS = ("grid", "felzenszwalb", "slic")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    source_summary_path = Path(args.source_summary).expanduser()
    source_summary = _load_json(source_summary_path)
    records = _as_records(source_summary)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    parser = VisualPrimitiveParser()
    methods = _parse_methods(args.methods)

    proposal_records: List[Dict[str, Any]] = []
    coverage_rows: List[Dict[str, Any]] = []
    for index, record in enumerate(records[: args.max_samples if args.max_samples is not None else None]):
        image_path = Path(str(record.get("image", ""))).expanduser()
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        target_text = str(record.get("target", "") or "")
        target_detections = parser.parse_detections(target_text)
        label = _record_label(record, target_detections)
        proposals = generate_image_proposals(
            image,
            methods=methods,
            max_proposals=args.max_proposals_per_record,
            min_area_ratio=args.min_area_ratio,
            max_area_ratio=args.max_area_ratio,
            grid_size_fractions=_parse_float_list(args.grid_size_fractions),
            grid_stride_fraction=args.grid_stride_fraction,
            felzenszwalb_scales=_parse_float_list(args.felzenszwalb_scales),
            slic_segments=_parse_int_list(args.slic_segments),
            rank_policy=args.rank_policy,
            target_detections=target_detections,
        )
        detections = [_proposal_to_detection(proposal, label) for proposal in proposals]
        structured_prediction = native_detections_to_vp(
            detections,
            box_format=args.structured_vp_box_format,
            marker_style=args.structured_vp_marker_style,
        )
        coverage = summarize_proposal_coverage(detections, target_detections)
        coverage_rows.append(coverage)
        proposal_records.append({
            "index": record.get("index", index),
            "image": str(image_path),
            "prefix": record.get("prefix", "<IMAGE_PROPOSALS>"),
            "query_label": label,
            "text_input": record.get("text_input") or label,
            "query_box_count": record.get("query_box_count", len(target_detections)),
            "gt_box_count": record.get("gt_box_count", len(target_detections)),
            "target": target_text,
            "raw_prediction": structured_prediction,
            "structured_prediction": structured_prediction,
            "structured_source": "image_proposals",
            "pred_box_count": len(detections),
            "structured_pred_box_count": len(detections),
            "vp_pred_box_count": len(detections),
            "vp_format_valid": bool(detections),
            "structured_vp_format_valid": bool(detections),
            "used_image_proposal_teacher": True,
            "proposal_methods": list(methods),
            "proposal_rank_policy": args.rank_policy,
            "proposal_candidates": detections,
            "proposal_coverage_iou25": coverage["proposal_gt_recall_iou25"],
            "proposal_coverage_iou50": coverage["proposal_gt_recall_iou50"],
            "proposal_coverage_iou75": coverage["proposal_gt_recall_iou75"],
            "proposal_mean_best_gt_iou": coverage["mean_best_gt_iou"],
        })

    summary = {
        "source_summary_path": str(source_summary_path),
        "model_path": source_summary.get("model_path"),
        "adapter_dir": source_summary.get("adapter_dir"),
        "data_path": source_summary.get("data_path"),
        "output_dir": str(output_dir),
        "num_samples": len(proposal_records),
        "structured_vp_decode": True,
        "structured_vp_mode": "image_proposal_teacher",
        "structured_vp_box_format": args.structured_vp_box_format,
        "structured_vp_marker_style": args.structured_vp_marker_style,
        "proposal_methods": list(methods),
        "proposal_rank_policy": args.rank_policy,
        "max_proposals_per_record": args.max_proposals_per_record,
        "avg_pred_boxes": _mean(row["pred_box_count"] for row in proposal_records),
        "avg_gt_boxes": _mean(row.get("gt_box_count", 0) for row in proposal_records),
        "proposal_gt_recall_iou25": _mean(row["proposal_gt_recall_iou25"] for row in coverage_rows),
        "proposal_gt_recall_iou50": _mean(row["proposal_gt_recall_iou50"] for row in coverage_rows),
        "proposal_gt_recall_iou75": _mean(row["proposal_gt_recall_iou75"] for row in coverage_rows),
        "proposal_mean_best_gt_iou": _mean(row["mean_best_gt_iou"] for row in coverage_rows),
        "records": proposal_records,
    }
    summary_path = output_dir / "vp_image_proposal_summary.json"
    markdown_path = output_dir / "vp_image_proposal_summary.md"
    summary["summary_path"] = str(summary_path)
    summary["markdown_path"] = str(markdown_path)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_path.write_text(render_image_proposal_markdown(summary), encoding="utf-8")
    return {
        "summary_path": str(summary_path),
        "markdown_path": str(markdown_path),
        "num_samples": summary["num_samples"],
        "proposal_methods": summary["proposal_methods"],
        "proposal_rank_policy": summary["proposal_rank_policy"],
        "avg_pred_boxes": summary["avg_pred_boxes"],
        "avg_gt_boxes": summary["avg_gt_boxes"],
        "proposal_gt_recall_iou25": summary["proposal_gt_recall_iou25"],
        "proposal_gt_recall_iou50": summary["proposal_gt_recall_iou50"],
        "proposal_gt_recall_iou75": summary["proposal_gt_recall_iou75"],
        "proposal_mean_best_gt_iou": summary["proposal_mean_best_gt_iou"],
    }


def generate_image_proposals(
    image: Image.Image,
    *,
    methods: Sequence[str] = DEFAULT_METHODS,
    max_proposals: int = 300,
    min_area_ratio: float = 0.0005,
    max_area_ratio: float = 0.95,
    grid_size_fractions: Sequence[float] = (0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 0.7, 1.0),
    grid_stride_fraction: float = 0.5,
    felzenszwalb_scales: Sequence[float] = (80.0, 160.0, 320.0),
    slic_segments: Sequence[int] = (80, 160),
    rank_policy: str = "objectness",
    target_detections: Sequence[Mapping[str, Any]] = (),
) -> List[Dict[str, Any]]:
    """Generate normalized proposal boxes from image-only cues."""

    width, height = image.size
    proposals: List[Dict[str, Any]] = []
    if "grid" in methods:
        proposals.extend(_grid_proposals(width, height, grid_size_fractions, grid_stride_fraction))
    if "felzenszwalb" in methods:
        proposals.extend(_skimage_segment_proposals(image, "felzenszwalb", felzenszwalb_scales, ()))
    if "slic" in methods:
        proposals.extend(_skimage_segment_proposals(image, "slic", (), slic_segments))

    proposals = _dedupe_and_filter_proposals(
        proposals,
        width=width,
        height=height,
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio,
    )
    feature_maps = _proposal_image_feature_maps(image)
    for proposal in proposals:
        proposal.update(_proposal_image_features(proposal, feature_maps))
        proposal["score"] = _proposal_score(proposal, width=width, height=height)
        if rank_policy == "edge_density":
            proposal["score"] = _proposal_edge_density_score(proposal)
        if rank_policy == "oracle_iou":
            proposal["score"] = _proposal_oracle_score(proposal, target_detections)
    proposals.sort(key=lambda item: (float(item.get("score", 0.0)), -float(item.get("area_ratio", 0.0))), reverse=True)
    return proposals[: max(0, int(max_proposals))]


def summarize_proposal_coverage(
    proposals: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    best_ious = []
    for target in targets:
        best_ious.append(max(
            (compute_bbox_iou(proposal.get("bbox"), target.get("bbox")) for proposal in proposals),
            default=0.0,
        ))
    return {
        "gt_count": len(targets),
        "proposal_count": len(proposals),
        "proposal_gt_recall_iou25": _ratio_at(best_ious, 0.25),
        "proposal_gt_recall_iou50": _ratio_at(best_ious, 0.5),
        "proposal_gt_recall_iou75": _ratio_at(best_ious, 0.75),
        "mean_best_gt_iou": _mean(best_ious),
        "best_ious": best_ious,
    }


def render_image_proposal_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# VP Image Proposal Summary",
        "",
        f"- Records: `{int(summary.get('num_samples', 0) or 0)}`",
        f"- Methods: `{','.join(summary.get('proposal_methods', []) or [])}`",
        f"- Rank policy: `{summary.get('proposal_rank_policy')}`",
        f"- Avg proposals / GT: `{float(summary.get('avg_pred_boxes', 0.0) or 0.0):.2f}` / "
        f"`{float(summary.get('avg_gt_boxes', 0.0) or 0.0):.2f}`",
        f"- Proposal GT recall @ IoU 0.25 / 0.50 / 0.75: "
        f"`{float(summary.get('proposal_gt_recall_iou25', 0.0) or 0.0):.4f}` / "
        f"`{float(summary.get('proposal_gt_recall_iou50', 0.0) or 0.0):.4f}` / "
        f"`{float(summary.get('proposal_gt_recall_iou75', 0.0) or 0.0):.4f}`",
        f"- Mean best GT IoU: `{float(summary.get('proposal_mean_best_gt_iou', 0.0) or 0.0):.4f}`",
        "",
        "| index | label | proposals | gt | recall@50 | mean best IoU |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for record in list(summary.get("records", []) or [])[:20]:
        lines.append(
            f"| {record.get('index')} "
            f"| `{record.get('query_label')}` "
            f"| {int(record.get('pred_box_count', 0) or 0)} "
            f"| {int(record.get('gt_box_count', 0) or 0)} "
            f"| {float(record.get('proposal_coverage_iou50', 0.0) or 0.0):.4f} "
            f"| {float(record.get('proposal_mean_best_gt_iou', 0.0) or 0.0):.4f} |"
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-summary", required=True)
    parser.add_argument("--output-dir", default=".codex_reports/florence_vp_image_proposals")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-proposals-per-record", type=int, default=300)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--rank-policy", default="objectness", choices=["objectness", "edge_density", "oracle_iou"])
    parser.add_argument("--structured-vp-box-format", default="loc_tokens", choices=["loc_tokens", "json"])
    parser.add_argument("--structured-vp-marker-style", default="plain", choices=["plain", "special"])
    parser.add_argument("--min-area-ratio", type=float, default=0.0005)
    parser.add_argument("--max-area-ratio", type=float, default=0.95)
    parser.add_argument("--grid-size-fractions", default="0.08,0.12,0.18,0.25,0.35,0.5,0.7,1.0")
    parser.add_argument("--grid-stride-fraction", type=float, default=0.5)
    parser.add_argument("--felzenszwalb-scales", default="80,160,320")
    parser.add_argument("--slic-segments", default="80,160")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    print(json.dumps(run(parse_args(argv)), indent=2, ensure_ascii=False))
    return 0


def _grid_proposals(
    width: int,
    height: int,
    size_fractions: Sequence[float],
    stride_fraction: float,
) -> List[Dict[str, Any]]:
    proposals: List[Dict[str, Any]] = []
    base = float(min(width, height))
    aspect_ratios = (0.5, 0.75, 1.0, 1.333, 2.0)
    for fraction in size_fractions:
        side = max(2, int(round(base * float(fraction))))
        for aspect in aspect_ratios:
            box_w = max(2, min(width, int(round(side * math.sqrt(aspect)))))
            box_h = max(2, min(height, int(round(side / math.sqrt(aspect)))))
            step_x = max(1, int(round(box_w * stride_fraction)))
            step_y = max(1, int(round(box_h * stride_fraction)))
            x_values = _scan_positions(width, box_w, step_x)
            y_values = _scan_positions(height, box_h, step_y)
            for x1 in x_values:
                for y1 in y_values:
                    proposals.append(_proposal_from_pixel_box(x1, y1, x1 + box_w, y1 + box_h, width, height, "grid"))
    proposals.append(_proposal_from_pixel_box(0, 0, width, height, width, height, "grid_full"))
    return proposals


def _skimage_segment_proposals(
    image: Image.Image,
    method: str,
    felzenszwalb_scales: Sequence[float],
    slic_segments: Sequence[int],
) -> List[Dict[str, Any]]:
    try:
        import numpy as np
        from skimage.segmentation import felzenszwalb, slic
    except Exception:
        return []

    width, height = image.size
    array = np.asarray(image)
    label_maps = []
    if method == "felzenszwalb":
        for scale in felzenszwalb_scales:
            label_maps.append(felzenszwalb(array, scale=float(scale), sigma=0.8, min_size=20))
    elif method == "slic":
        for n_segments in slic_segments:
            label_maps.append(slic(array, n_segments=int(n_segments), compactness=10.0, start_label=1))

    proposals: List[Dict[str, Any]] = []
    for label_map in label_maps:
        for label in np.unique(label_map):
            ys, xs = np.where(label_map == label)
            if xs.size == 0 or ys.size == 0:
                continue
            x1 = int(xs.min())
            y1 = int(ys.min())
            x2 = int(xs.max()) + 1
            y2 = int(ys.max()) + 1
            proposals.append(_proposal_from_pixel_box(x1, y1, x2, y2, width, height, method))
    return proposals


def _proposal_from_pixel_box(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    width: int,
    height: int,
    source: str,
) -> Dict[str, Any]:
    x1 = max(0, min(width - 1, int(x1)))
    y1 = max(0, min(height - 1, int(y1)))
    x2 = max(x1 + 1, min(width, int(x2)))
    y2 = max(y1 + 1, min(height, int(y2)))
    bbox = [
        _to_loc(x1, width),
        _to_loc(y1, height),
        _to_loc(x2 - 1, width),
        _to_loc(y2 - 1, height),
    ]
    return {
        "bbox": bbox,
        "pixel_box": [x1, y1, x2, y2],
        "source": source,
        "area_ratio": ((x2 - x1) * (y2 - y1)) / max(1.0, float(width * height)),
    }


def _dedupe_and_filter_proposals(
    proposals: Sequence[Mapping[str, Any]],
    *,
    width: int,
    height: int,
    min_area_ratio: float,
    max_area_ratio: float,
) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for proposal in proposals:
        bbox = tuple(int(value) for value in proposal.get("bbox", []) or [])
        if len(bbox) != 4 or bbox in seen:
            continue
        seen.add(bbox)
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            continue
        area_ratio = _bbox_area_ratio(bbox)
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue
        item = dict(proposal)
        item["area_ratio"] = area_ratio
        deduped.append(item)
    return deduped


def _proposal_score(proposal: Mapping[str, Any], *, width: int, height: int) -> float:
    area = float(proposal.get("area_ratio", 0.0) or 0.0)
    x1, y1, x2, y2 = [float(value) for value in proposal.get("bbox", [0, 0, 0, 0])]
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    aspect = max(box_w / box_h, box_h / box_w)
    area_prior = math.exp(-abs(math.log(max(area, 1e-6) / 0.04)))
    aspect_prior = math.exp(-abs(math.log(max(aspect, 1e-6))))
    source_bonus = 0.08 if str(proposal.get("source", "")).startswith("felzenszwalb") else 0.0
    source_bonus += 0.04 if str(proposal.get("source", "")).startswith("slic") else 0.0
    return area_prior + 0.3 * aspect_prior + source_bonus


def _proposal_edge_density_score(proposal: Mapping[str, Any]) -> float:
    edge_density = float(proposal.get("edge_density", 0.0) or 0.0)
    contrast = float(proposal.get("contrast", 0.0) or 0.0)
    objectness = float(proposal.get("score", 0.0) or 0.0)
    area = float(proposal.get("area_ratio", 0.0) or 0.0)
    area_penalty = max(0.0, math.log(max(area, 1e-6) / 0.16))
    return (2.0 * edge_density) + (0.5 * contrast) + (0.35 * objectness) - (0.15 * area_penalty)


def _proposal_image_feature_maps(image: Image.Image) -> Optional[Mapping[str, Any]]:
    try:
        import numpy as np
        from skimage.color import rgb2gray
        from skimage.filters import sobel
    except Exception:
        return None

    array = np.asarray(image)
    gray = rgb2gray(array)
    return {
        "gray": gray,
        "edges": sobel(gray),
        "np": np,
    }


def _proposal_image_features(proposal: Mapping[str, Any], feature_maps: Optional[Mapping[str, Any]]) -> Dict[str, float]:
    if not feature_maps:
        return {"edge_density": 0.0, "contrast": 0.0}
    pixel_box = proposal.get("pixel_box")
    if not isinstance(pixel_box, Sequence) or len(pixel_box) != 4:
        return {"edge_density": 0.0, "contrast": 0.0}
    x1, y1, x2, y2 = [int(value) for value in pixel_box]
    if x2 <= x1 or y2 <= y1:
        return {"edge_density": 0.0, "contrast": 0.0}
    gray = feature_maps.get("gray")
    edges = feature_maps.get("edges")
    np = feature_maps.get("np")
    if gray is None or edges is None or np is None:
        return {"edge_density": 0.0, "contrast": 0.0}
    gray_crop = gray[y1:y2, x1:x2]
    edge_crop = edges[y1:y2, x1:x2]
    if gray_crop.size == 0 or edge_crop.size == 0:
        return {"edge_density": 0.0, "contrast": 0.0}
    return {
        "edge_density": float(np.mean(edge_crop)),
        "contrast": float(np.std(gray_crop)),
    }


def _proposal_to_detection(proposal: Mapping[str, Any], label: str) -> Dict[str, Any]:
    return {
        "label": label,
        "bbox": [int(value) for value in proposal.get("bbox", [])],
        "confidence": float(proposal.get("score", 0.0) or 0.0),
        "proposal_source": proposal.get("source"),
        "proposal_area_ratio": float(proposal.get("area_ratio", 0.0) or 0.0),
        "proposal_edge_density": float(proposal.get("edge_density", 0.0) or 0.0),
        "proposal_contrast": float(proposal.get("contrast", 0.0) or 0.0),
    }


def _proposal_oracle_score(
    proposal: Mapping[str, Any],
    target_detections: Sequence[Mapping[str, Any]],
) -> float:
    return max(
        (compute_bbox_iou(proposal.get("bbox"), target.get("bbox")) for target in target_detections),
        default=0.0,
    )


def _scan_positions(length: int, window: int, step: int) -> List[int]:
    if window >= length:
        return [0]
    values = list(range(0, max(1, length - window + 1), step))
    last = length - window
    if not values or values[-1] != last:
        values.append(last)
    return values


def _to_loc(value: int, length: int) -> int:
    if length <= 1:
        return 0
    return max(0, min(999, int(round((float(value) / float(length - 1)) * 999.0))))


def _bbox_area_ratio(bbox: Sequence[int]) -> float:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    return max(0.0, (x2 - x1) * (y2 - y1)) / (999.0 * 999.0)


def _record_label(record: Mapping[str, Any], target_detections: Sequence[Mapping[str, Any]]) -> str:
    for key in ("query_label", "text_input", "allowed_labels"):
        value = str(record.get(key, "") or "").strip()
        if value:
            return value
    if target_detections:
        return str(target_detections[0].get("label", "") or "proposal")
    return "proposal"


def _ratio_at(values: Sequence[float], threshold: float) -> float:
    if not values:
        return 0.0
    return sum(1 for value in values if float(value) >= threshold) / len(values)


def _mean(values: Iterable[Any]) -> float:
    parsed = [float(value) for value in values if value is not None]
    return sum(parsed) / len(parsed) if parsed else 0.0


def _parse_methods(value: str) -> Tuple[str, ...]:
    methods = tuple(item.strip().lower() for item in str(value or "").split(",") if item.strip())
    allowed = set(DEFAULT_METHODS)
    unknown = sorted(set(methods) - allowed)
    if unknown:
        raise ValueError(f"Unknown proposal methods: {', '.join(unknown)}")
    return methods or DEFAULT_METHODS


def _parse_float_list(value: str) -> List[float]:
    return [float(item.strip()) for item in str(value or "").split(",") if item.strip()]


def _parse_int_list(value: str) -> List[int]:
    return [int(item.strip()) for item in str(value or "").split(",") if item.strip()]


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
