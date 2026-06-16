#!/usr/bin/env python3
"""Build a Florence-VP readiness report card from saved diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from florence_forge.evaluation.vp_detection_quality import (  # noqa: E402
    analyze_vp_target_count_gap,
    load_vp_quality_summary,
)
from florence_forge.evaluation.vp_report_card import (  # noqa: E402
    VPReportCardThresholds,
    build_vp_report_card,
    render_vp_report_card_markdown,
)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    quality_path = Path(args.quality_report).expanduser()
    quality_report = load_vp_quality_summary(quality_path)
    quality_report.setdefault("source_report_path", str(quality_path))

    policy_sweep = None
    if args.policy_sweep:
        policy_path = Path(args.policy_sweep).expanduser()
        policy_sweep = load_vp_quality_summary(policy_path)
        policy_sweep.setdefault("sweep_json_path", str(policy_path))

    target_count_gap = None
    if args.target_count_gap:
        gap_path = Path(args.target_count_gap).expanduser()
        target_count_gap = load_vp_quality_summary(gap_path)
        target_count_gap.setdefault("target_count_gap_json_path", str(gap_path))
    elif not args.skip_target_count_gap:
        target_count_gap = analyze_vp_target_count_gap(
            quality_report,
            focus_bucket=args.focus_bucket,
            max_rows=args.max_gap_rows,
        )

    thresholds = VPReportCardThresholds(
        min_samples=args.min_samples,
        min_precision=args.min_precision,
        min_recall=args.min_recall,
        min_f1=args.min_f1,
        max_undergeneration_ratio=args.max_undergeneration_ratio,
        max_overgeneration_ratio=args.max_overgeneration_ratio,
        max_repair_record_ratio=args.max_repair_record_ratio,
        min_raw_vp_format_ratio=args.min_raw_vp_format_ratio,
        max_structured_decoder_ratio=args.max_structured_decoder_ratio,
        min_policy_confidence=args.min_policy_confidence,
        high_recoverable_fn_ratio=args.high_recoverable_fn_ratio,
    )
    card = build_vp_report_card(
        quality_report,
        policy_sweep=policy_sweep,
        target_count_gap=target_count_gap,
        thresholds=thresholds,
        focus_bucket=args.focus_bucket,
    )
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "vp_report_card.json"
    markdown_path = output_dir / "vp_report_card.md"
    card.update({
        "quality_report_path": str(quality_path),
        "policy_sweep_path": (
            str(Path(args.policy_sweep).expanduser()) if args.policy_sweep else None
        ),
        "target_count_gap_path": (
            str(Path(args.target_count_gap).expanduser()) if args.target_count_gap else None
        ),
        "report_card_json_path": str(json_path),
        "report_card_markdown_path": str(markdown_path),
    })
    json_path.write_text(json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_path.write_text(render_vp_report_card_markdown(card), encoding="utf-8")
    return card


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quality-report",
        required=True,
        help="Path to vp_detection_quality.json.",
    )
    parser.add_argument(
        "--policy-sweep",
        default=None,
        help="Optional path to vp_quality_policy_sweep.json.",
    )
    parser.add_argument(
        "--target-count-gap",
        default=None,
        help="Optional path to vp_target_count_gap.json.",
    )
    parser.add_argument("--output-dir", default=".codex_reports/florence_vp_report_card")
    parser.add_argument("--focus-bucket", default=None, choices=["single", "medium", "dense"])
    parser.add_argument(
        "--skip-target-count-gap",
        action="store_true",
        help="Do not derive target-count gap diagnostics from the quality report.",
    )
    parser.add_argument("--max-gap-rows", type=int, default=20)
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--min-precision", type=float, default=0.80)
    parser.add_argument("--min-recall", type=float, default=0.70)
    parser.add_argument("--min-f1", type=float, default=0.75)
    parser.add_argument("--max-undergeneration-ratio", type=float, default=0.35)
    parser.add_argument("--max-overgeneration-ratio", type=float, default=0.25)
    parser.add_argument("--max-repair-record-ratio", type=float, default=0.25)
    parser.add_argument("--min-raw-vp-format-ratio", type=float, default=0.95)
    parser.add_argument("--max-structured-decoder-ratio", type=float, default=0.50)
    parser.add_argument(
        "--min-policy-confidence",
        default="moderate",
        choices=["none", "exploratory", "moderate", "strong"],
    )
    parser.add_argument("--high-recoverable-fn-ratio", type=float, default=0.40)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    print(json.dumps(run(parse_args(argv)), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
