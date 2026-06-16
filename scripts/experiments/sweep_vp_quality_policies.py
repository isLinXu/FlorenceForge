#!/usr/bin/env python3
"""Sweep VP quality policies from one saved inference summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from florence_forge.evaluation.vp_detection_quality import (
    VPDetectionQualityConfig,
    compare_vp_quality_reports,
    evaluate_vp_summary,
    load_vp_quality_summary,
    render_vp_detection_quality_markdown,
    render_vp_policy_comparison_markdown,
)


DEFAULT_POLICY_CONFIGS = (
    "none:filter_policy=none",
    "nms:filter_policy=nms,nms_iou_threshold=0.5",
    "single:filter_policy=single-target",
)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    summary_path = Path(args.summary).expanduser()
    summary = load_vp_quality_summary(summary_path)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    policy_configs = [_parse_policy_config(value) for value in args.policy_config]
    if args.structured_vp_allowed_labels:
        policy_configs.append((
            "static_allowed",
            {"allowed_labels": args.structured_vp_allowed_labels},
        ))
    if args.structured_vp_allowed_labels_field:
        policy_configs.append((
            _safe_label(f"{args.structured_vp_allowed_labels_field}_allowed"),
            {"allowed_labels_field": args.structured_vp_allowed_labels_field},
        ))
        if args.include_phrase_label_policy:
            policy_configs.append((
                _safe_label(f"{args.structured_vp_allowed_labels_field}_phrase_allowed"),
                {
                    "allowed_labels_field": args.structured_vp_allowed_labels_field,
                    "label_match_mode": "contains",
                    "allowed_label_match_mode": "contains",
                },
            ))
    if args.include_target_label_oracle:
        policy_configs.append(("target_oracle", {"allowed_labels_field": "target_labels"}))
    if args.include_repair_policy:
        policy_configs.extend(_repair_policy_configs(policy_configs))

    reports: Dict[str, Dict[str, Any]] = {}
    report_paths: Dict[str, Dict[str, str]] = {}
    for name, overrides in _dedupe_policy_configs(policy_configs):
        report = evaluate_vp_summary(
            summary,
            config=VPDetectionQualityConfig(
                iou_threshold=args.iou_threshold,
                use_structured_decoder=args.structured_vp_mode != "off",
                prediction_field=args.prediction_field,
                reference_field=args.reference_field,
                box_format=args.structured_vp_box_format,
                marker_style=args.structured_vp_marker_style,
                filter_policy=str(overrides.get("filter_policy", args.structured_vp_filter_policy)),
                max_boxes_per_label=overrides.get("max_boxes_per_label", args.structured_vp_max_boxes_per_label),
                max_total_boxes=overrides.get("max_total_boxes", args.structured_vp_max_total_boxes),
                max_total_boxes_field=overrides.get(
                    "max_total_boxes_field",
                    args.structured_vp_max_total_boxes_field,
                ),
                nms_iou_threshold=overrides.get("nms_iou_threshold", args.structured_vp_nms_iou_threshold),
                allowed_labels=overrides.get("allowed_labels"),
                allowed_labels_field=overrides.get("allowed_labels_field"),
                label_match_mode=str(overrides.get("label_match_mode", args.vp_label_match_mode)),
                allowed_label_match_mode=overrides.get(
                    "allowed_label_match_mode",
                    args.structured_vp_allowed_label_match_mode,
                ),
                repair_malformed_tail=_coerce_bool(
                    overrides.get(
                        "repair_malformed_tail",
                        args.structured_vp_repair_malformed_tail,
                    )
                ),
                max_bad_cases=args.max_bad_cases,
            ),
        )
        report["source_summary_path"] = str(summary_path)
        report["sweep_policy"] = name
        policy_dir = output_dir / name
        policy_dir.mkdir(parents=True, exist_ok=True)
        json_path = policy_dir / "vp_detection_quality.json"
        markdown_path = policy_dir / "vp_detection_quality.md"
        report["quality_json_path"] = str(json_path)
        report["quality_markdown_path"] = str(markdown_path)
        json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        markdown_path.write_text(render_vp_detection_quality_markdown(report), encoding="utf-8")
        reports[name] = report
        report_paths[name] = {
            "quality_json_path": str(json_path),
            "quality_markdown_path": str(markdown_path),
        }

    comparison = compare_vp_quality_reports(reports, focus_bucket=args.focus_bucket)
    comparison_path = output_dir / "vp_policy_comparison.json"
    comparison_markdown_path = output_dir / "vp_policy_comparison.md"
    comparison["policy_comparison_json_path"] = str(comparison_path)
    comparison["policy_comparison_markdown_path"] = str(comparison_markdown_path)
    comparison_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")
    comparison_markdown_path.write_text(render_vp_policy_comparison_markdown(comparison), encoding="utf-8")

    aggregate = {
        "summary_path": str(summary_path),
        "output_dir": str(output_dir),
        "num_policies": len(reports),
        "quality_reports": report_paths,
        "recommended_policy": comparison.get("recommended_policy"),
        "policy_comparison_json_path": str(comparison_path),
        "policy_comparison_markdown_path": str(comparison_markdown_path),
        "comparison": comparison,
    }
    aggregate_path = output_dir / "vp_quality_policy_sweep.json"
    aggregate["sweep_json_path"] = str(aggregate_path)
    aggregate_path.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")
    return aggregate


def _parse_policy_config(value: str) -> Tuple[str, Dict[str, Any]]:
    name, _, config_text = str(value).partition(":")
    name = _safe_label(name or "policy")
    config: Dict[str, Any] = {}
    if not config_text:
        return name, config
    for item in config_text.split(","):
        item = item.strip()
        if not item:
            continue
        key, sep, raw_value = item.partition("=")
        if not sep:
            raise ValueError(f"Invalid policy config item: {item}")
        key = key.strip()
        raw_value = raw_value.strip()
        if key == "filter_policy":
            if raw_value not in {"none", "auto", "single-target", "nms"}:
                raise ValueError("filter_policy must be one of: none, auto, single-target, nms")
            config[key] = raw_value
        elif key in {"max_boxes_per_label", "max_total_boxes"}:
            int_value = int(raw_value)
            if int_value < 1:
                raise ValueError(f"{key} must be >= 1")
            config[key] = int_value
        elif key == "nms_iou_threshold":
            float_value = float(raw_value)
            if float_value <= 0.0 or float_value > 1.0:
                raise ValueError("nms_iou_threshold must be in (0, 1]")
            config[key] = float_value
        elif key in {"allowed_labels", "allowed_labels_field", "max_total_boxes_field"}:
            config[key] = raw_value
        elif key in {"label_match_mode", "allowed_label_match_mode"}:
            if raw_value not in {"strict", "contains"}:
                raise ValueError(f"{key} must be one of: strict, contains")
            config[key] = raw_value
        elif key == "repair_malformed_tail":
            config[key] = _coerce_bool(raw_value)
        else:
            raise ValueError(f"Unsupported policy config key: {key}")
    return name, config


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def _dedupe_policy_configs(configs: Sequence[Tuple[str, Dict[str, Any]]]) -> Sequence[Tuple[str, Dict[str, Any]]]:
    seen: Dict[str, int] = {}
    deduped = []
    for name, config in configs:
        seen[name] = seen.get(name, 0) + 1
        final_name = name if seen[name] == 1 else f"{name}_{seen[name]}"
        deduped.append((final_name, config))
    return deduped


def _repair_policy_configs(configs: Sequence[Tuple[str, Dict[str, Any]]]) -> Sequence[Tuple[str, Dict[str, Any]]]:
    repaired = []
    for name, config in configs:
        if _coerce_bool(config.get("repair_malformed_tail", False)):
            continue
        repair_config = dict(config)
        repair_config["repair_malformed_tail"] = True
        repaired.append((f"{name}_repair", repair_config))
    return repaired


def _safe_label(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value).strip())
    return cleaned or "policy"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, help="Path to vp_inference_visualization_summary.json.")
    parser.add_argument("--output-dir", default=".codex_reports/florence_vp_quality_policy_sweep")
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
    parser.add_argument("--structured-vp-allowed-labels-field", default=None)
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
        "--structured-vp-repair-malformed-tail",
        action="store_true",
        help=(
            "Merge malformed native loc groups after a valid VP box back into "
            "the structured VP output before filtering."
        ),
    )
    parser.add_argument(
        "--vp-label-match-mode",
        default="strict",
        choices=["strict", "contains"],
        help="Label matching mode used when matching predictions to references.",
    )
    parser.add_argument(
        "--include-phrase-label-policy",
        action="store_true",
        help=(
            "When per-record allowed labels are provided, also evaluate a phrase-contained policy "
            "for specialized labels such as 'cup' vs 'coffee cup'."
        ),
    )
    parser.add_argument(
        "--include-repair-policy",
        action="store_true",
        help="Also evaluate repair_malformed_tail=true variants for all configured policies.",
    )
    parser.add_argument("--include-target-label-oracle", action="store_true")
    parser.add_argument(
        "--focus-bucket",
        default=None,
        choices=["single", "medium", "dense"],
        help="Rank policy comparison by a box-count bucket instead of overall metrics.",
    )
    parser.add_argument("--max-bad-cases", type=int, default=20)
    parser.add_argument(
        "--policy-config",
        action="append",
        default=[],
        help=(
            "Named policy override, e.g. nms07:filter_policy=nms,nms_iou_threshold=0.7 "
            "or query:allowed_labels_field=text_input."
        ),
    )
    args = parser.parse_args(argv)
    if not args.policy_config:
        args.policy_config = list(DEFAULT_POLICY_CONFIGS)
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    print(json.dumps(run(parse_args(argv)), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
