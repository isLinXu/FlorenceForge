#!/usr/bin/env python3
"""Compare VP detection quality reports from multiple post-filter policies."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from florence_forge.evaluation.vp_detection_quality import (
    compare_vp_quality_reports,
    load_vp_quality_summary,
    render_vp_policy_comparison_markdown,
)


def run(args: argparse.Namespace) -> Dict[str, object]:
    reports = {
        name: _load_quality_report(path)
        for name, path in _parse_report_args(args.report)
    }
    comparison = compare_vp_quality_reports(reports, focus_bucket=args.focus_bucket)

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "vp_policy_comparison.json"
    markdown_path = output_dir / "vp_policy_comparison.md"
    comparison["policy_comparison_json_path"] = str(json_path)
    comparison["policy_comparison_markdown_path"] = str(markdown_path)
    json_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_path.write_text(render_vp_policy_comparison_markdown(comparison), encoding="utf-8")
    return comparison


def _parse_report_args(values: Sequence[str]) -> Sequence[Tuple[str, Path]]:
    if not values:
        raise ValueError("At least one --report name=path is required")

    parsed = []
    for value in values:
        name, sep, raw_path = str(value).partition("=")
        if not sep or not name.strip() or not raw_path.strip():
            raise ValueError(f"Invalid --report value {value!r}; expected name=path")
        parsed.append((_safe_label(name), Path(raw_path).expanduser()))
    return parsed


def _load_quality_report(path: Path) -> Dict[str, object]:
    report = load_vp_quality_summary(path)
    report["source_report_path"] = str(path)
    return report


def _safe_label(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value).strip())
    return cleaned or "policy"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        action="append",
        default=[],
        help="Named quality report path, e.g. none=path/to/vp_detection_quality.json.",
    )
    parser.add_argument("--output-dir", default=".codex_reports/florence_vp_policy_comparison")
    parser.add_argument(
        "--focus-bucket",
        default=None,
        choices=["single", "medium", "dense"],
        help="Rank and recommend policies by a box-count bucket instead of overall metrics.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    print(json.dumps(run(parse_args(argv)), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
