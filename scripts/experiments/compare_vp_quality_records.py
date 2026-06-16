#!/usr/bin/env python3
"""Compare two VP detection quality reports at the per-record level."""

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
    compare_vp_quality_record_reports,
    load_vp_quality_summary,
    render_vp_record_comparison_markdown,
)


def run(args: argparse.Namespace) -> Dict[str, object]:
    candidate_report = _load_report(args.candidate_report)
    baseline_report = _load_report(args.baseline_report)
    comparison = compare_vp_quality_record_reports(
        candidate_report,
        baseline_report,
        candidate_name=args.candidate_name,
        baseline_name=args.baseline_name,
        focus_bucket=args.focus_bucket,
    )

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "vp_record_comparison.json"
    markdown_path = output_dir / "vp_record_comparison.md"
    comparison["record_comparison_json_path"] = str(json_path)
    comparison["record_comparison_markdown_path"] = str(markdown_path)
    json_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_path.write_text(render_vp_record_comparison_markdown(comparison), encoding="utf-8")
    return comparison


def _load_report(path: str) -> Dict[str, object]:
    report_path = Path(path).expanduser()
    report = load_vp_quality_summary(report_path)
    report["source_report_path"] = str(report_path)
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-report", required=True)
    parser.add_argument("--baseline-report", required=True)
    parser.add_argument("--candidate-name", default="adapter")
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--output-dir", default=".codex_reports/florence_vp_record_comparison")
    parser.add_argument(
        "--focus-bucket",
        default=None,
        choices=["single", "medium", "dense"],
        help="Compare only one query/GT box-count bucket.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    print(json.dumps(run(parse_args(argv)), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
