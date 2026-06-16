#!/usr/bin/env python3
"""Analyze target-count gap upper bounds from a VP quality report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from florence_forge.evaluation.vp_detection_quality import (
    analyze_vp_target_count_gap,
    load_vp_quality_summary,
    render_vp_target_count_gap_markdown,
)


def run(args: argparse.Namespace) -> Dict[str, object]:
    report_path = Path(args.report).expanduser()
    report = load_vp_quality_summary(report_path)
    report["source_report_path"] = str(report_path)
    analysis = analyze_vp_target_count_gap(
        report,
        focus_bucket=args.focus_bucket,
        max_rows=args.max_rows,
    )

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "vp_target_count_gap.json"
    markdown_path = output_dir / "vp_target_count_gap.md"
    analysis["target_count_gap_json_path"] = str(json_path)
    analysis["target_count_gap_markdown_path"] = str(markdown_path)
    json_path.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_path.write_text(render_vp_target_count_gap_markdown(analysis), encoding="utf-8")
    return analysis


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, help="Path to vp_detection_quality.json.")
    parser.add_argument("--output-dir", default=".codex_reports/florence_vp_target_count_gap")
    parser.add_argument(
        "--focus-bucket",
        default=None,
        choices=["single", "medium", "dense"],
        help="Analyze only one query/GT box-count bucket.",
    )
    parser.add_argument("--max-rows", type=int, default=20)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    print(json.dumps(run(parse_args(argv)), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
