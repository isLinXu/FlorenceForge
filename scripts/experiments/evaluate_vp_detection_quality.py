#!/usr/bin/env python3
"""Evaluate VP detection quality from saved inference summaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from florence_forge.evaluation.vp_detection_quality import (
    VPDetectionQualityConfig,
    evaluate_vp_summary,
    load_vp_quality_summary,
    render_vp_detection_quality_markdown,
)


def run(args: argparse.Namespace):
    summary = load_vp_quality_summary(args.summary)
    report = evaluate_vp_summary(
        summary,
        config=VPDetectionQualityConfig(
            iou_threshold=args.iou_threshold,
            use_structured_decoder=args.structured_vp_mode != "off",
            prediction_field=args.prediction_field,
            reference_field=args.reference_field,
            box_format=args.structured_vp_box_format,
            marker_style=args.structured_vp_marker_style,
            filter_policy=args.structured_vp_filter_policy,
            max_boxes_per_label=args.structured_vp_max_boxes_per_label,
            max_total_boxes=args.structured_vp_max_total_boxes,
            max_total_boxes_field=args.structured_vp_max_total_boxes_field,
            nms_iou_threshold=args.structured_vp_nms_iou_threshold,
            allowed_labels=args.structured_vp_allowed_labels,
            allowed_labels_field=args.structured_vp_allowed_labels_field,
            label_match_mode=args.vp_label_match_mode,
            allowed_label_match_mode=args.structured_vp_allowed_label_match_mode,
            repair_malformed_tail=args.structured_vp_repair_malformed_tail,
            max_bad_cases=args.max_bad_cases,
        ),
    )
    report["source_summary_path"] = str(Path(args.summary).expanduser())

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "vp_detection_quality.json"
    markdown_path = output_dir / "vp_detection_quality.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_path.write_text(render_vp_detection_quality_markdown(report), encoding="utf-8")
    report["quality_json_path"] = str(json_path)
    report["quality_markdown_path"] = str(markdown_path)
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, help="Path to vp_inference_visualization_summary.json.")
    parser.add_argument("--output-dir", default=".codex_reports/florence_vp_detection_quality")
    parser.add_argument("--prediction-field", default="raw_prediction")
    parser.add_argument("--reference-field", default="target")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--structured-vp-mode", default="on", choices=["off", "on"])
    parser.add_argument("--structured-vp-box-format", default="loc_tokens", choices=["loc_tokens", "json"])
    parser.add_argument("--structured-vp-marker-style", default="plain", choices=["special", "plain"])
    parser.add_argument("--structured-vp-filter-policy", default="none", choices=["none", "auto", "single-target", "nms"])
    parser.add_argument("--structured-vp-max-boxes-per-label", type=int, default=None)
    parser.add_argument("--structured-vp-max-total-boxes", type=int, default=None)
    parser.add_argument(
        "--structured-vp-max-total-boxes-field",
        default=None,
        help="Optional per-record field used as dynamic max_total_boxes, e.g. query_box_count.",
    )
    parser.add_argument("--structured-vp-nms-iou-threshold", type=float, default=None)
    parser.add_argument("--structured-vp-allowed-labels", default=None)
    parser.add_argument(
        "--structured-vp-allowed-label-match-mode",
        default=None,
        choices=["strict", "contains"],
        help=(
            "Optional match mode for allowed-label filtering. Defaults to --vp-label-match-mode "
            "inside evaluation."
        ),
    )
    parser.add_argument(
        "--vp-label-match-mode",
        default="strict",
        choices=["strict", "contains"],
        help="Label matching mode used when matching predictions to references.",
    )
    parser.add_argument(
        "--structured-vp-allowed-labels-field",
        default=None,
        help=(
            "Optional per-record field used as allowed labels, e.g. text_input or query. "
            "Use target_labels/reference_labels only for oracle diagnostics."
        ),
    )
    parser.add_argument(
        "--structured-vp-repair-malformed-tail",
        action="store_true",
        help=(
            "Merge malformed native loc groups after a valid VP box back into "
            "the structured VP output before filtering."
        ),
    )
    parser.add_argument("--max-bad-cases", type=int, default=20)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    print(json.dumps(run(parse_args(argv)), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
