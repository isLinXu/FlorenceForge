#!/usr/bin/env python3
"""Compare VP detection quality reports on the same record prefix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from florence_forge.evaluation.vp_detection_quality import (
    VPDetectionQualityConfig,
    load_vp_quality_summary,
    summarize_vp_quality_records,
)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    rows = []
    for report_spec in args.report:
        name, path = _parse_named_path(report_spec)
        report = load_vp_quality_summary(path)
        selected_records = _select_records(
            report,
            max_records=args.max_records,
            focus_bucket=args.focus_bucket,
        )
        summary = summarize_vp_quality_records(
            selected_records,
            config=VPDetectionQualityConfig(max_bad_cases=args.max_bad_cases),
        )
        rows.append(_comparison_row(
            name,
            path=path,
            source_report=report,
            selected_records=selected_records,
            summary=summary,
        ))

    baseline = rows[0] if rows else {}
    for row in rows:
        row["delta_vs_first"] = {
            "precision": _float(row.get("precision")) - _float(baseline.get("precision")),
            "recall": _float(row.get("recall")) - _float(baseline.get("recall")),
            "f1": _float(row.get("f1")) - _float(baseline.get("f1")),
            "true_positives": _int(row.get("true_positives")) - _int(baseline.get("true_positives")),
            "false_positives": _int(row.get("false_positives")) - _int(baseline.get("false_positives")),
            "false_negatives": _int(row.get("false_negatives")) - _int(baseline.get("false_negatives")),
            "avg_pred_boxes": _float(row.get("avg_pred_boxes")) - _float(baseline.get("avg_pred_boxes")),
        }

    comparison = {
        "num_reports": len(rows),
        "max_records": max(0, int(args.max_records)),
        "focus_bucket": args.focus_bucket,
        "reports": rows,
    }

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "vp_quality_prefix_comparison.json"
    markdown_path = output_dir / "vp_quality_prefix_comparison.md"
    comparison["prefix_comparison_json_path"] = str(json_path)
    comparison["prefix_comparison_markdown_path"] = str(markdown_path)
    json_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_path.write_text(render_markdown(comparison), encoding="utf-8")
    return comparison


def render_markdown(comparison: Mapping[str, Any]) -> str:
    rows = list(comparison.get("reports", []) or [])
    lines = [
        "# VP Quality Prefix Comparison",
        "",
        f"- Reports: `{int(comparison.get('num_reports', 0) or 0)}`",
        f"- Max records: `{int(comparison.get('max_records', 0) or 0)}`",
        f"- Focus bucket: `{comparison.get('focus_bucket') or 'all'}`",
        "",
        "| report | samples | precision | recall | f1 | TP/FP/FN | avg pred / GT | undergen | delta F1 | delta TP/FP/FN |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        delta = dict(row.get("delta_vs_first", {}) or {})
        lines.append(
            f"| `{row.get('name')}` "
            f"| {int(row.get('num_samples', 0) or 0)} "
            f"| {float(row.get('precision', 0.0) or 0.0):.4f} "
            f"| {float(row.get('recall', 0.0) or 0.0):.4f} "
            f"| {float(row.get('f1', 0.0) or 0.0):.4f} "
            f"| {int(row.get('true_positives', 0) or 0)}/"
            f"{int(row.get('false_positives', 0) or 0)}/"
            f"{int(row.get('false_negatives', 0) or 0)} "
            f"| {float(row.get('avg_pred_boxes', 0.0) or 0.0):.2f} / "
            f"{float(row.get('avg_gt_boxes', 0.0) or 0.0):.2f} "
            f"| {float(row.get('box_count_undergeneration_ratio', 0.0) or 0.0):.4f} "
            f"| {float(delta.get('f1', 0.0) or 0.0):+.4f} "
            f"| {int(delta.get('true_positives', 0) or 0):+d}/"
            f"{int(delta.get('false_positives', 0) or 0):+d}/"
            f"{int(delta.get('false_negatives', 0) or 0):+d} |"
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        action="append",
        required=True,
        help="Named quality report, e.g. adapter=path/to/vp_detection_quality.json.",
    )
    parser.add_argument("--output-dir", default=".codex_reports/florence_vp_quality_prefix_comparison")
    parser.add_argument("--max-records", type=int, default=8)
    parser.add_argument("--max-bad-cases", type=int, default=20)
    parser.add_argument(
        "--focus-bucket",
        default=None,
        choices=["single", "medium", "dense"],
        help="Filter records by box-count bucket before taking the prefix.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    print(json.dumps(run(parse_args(argv)), indent=2, ensure_ascii=False))
    return 0


def _parse_named_path(spec: str) -> Tuple[str, str]:
    if "=" in spec:
        name, path = spec.split("=", 1)
        return name.strip() or Path(path).stem, path
    return Path(spec).stem, spec


def _select_records(
    report: Mapping[str, Any],
    *,
    max_records: int,
    focus_bucket: Optional[str],
) -> List[Mapping[str, Any]]:
    records = [
        record for record in list(report.get("records", []) or [])
        if isinstance(record, Mapping)
    ]
    if focus_bucket:
        records = [
            record for record in records
            if str(record.get("box_count_bucket", "")) == focus_bucket
        ]
    return records[: max(0, int(max_records))]


def _comparison_row(
    name: str,
    *,
    path: str,
    source_report: Mapping[str, Any],
    selected_records: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "name": name,
        "source_report_path": path,
        "source_num_samples": _int(source_report.get("num_samples")),
        "selected_record_keys": [_record_key(record) for record in selected_records],
        "num_samples": _int(summary.get("num_samples")),
        "precision": _float(summary.get("precision")),
        "recall": _float(summary.get("recall")),
        "f1": _float(summary.get("f1")),
        "true_positives": _int(summary.get("true_positives")),
        "false_positives": _int(summary.get("false_positives")),
        "false_negatives": _int(summary.get("false_negatives")),
        "avg_pred_boxes": _float(summary.get("avg_pred_boxes")),
        "avg_gt_boxes": _float(summary.get("avg_gt_boxes")),
        "box_count_undergeneration_ratio": _float(summary.get("box_count_undergeneration_ratio")),
        "box_count_overgeneration_ratio": _float(summary.get("box_count_overgeneration_ratio")),
        "box_count_bucket_summary": dict(summary.get("box_count_bucket_summary", {}) or {}),
    }


def _record_key(record: Mapping[str, Any]) -> str:
    return "|".join([
        str(record.get("index", "")),
        Path(str(record.get("image", "") or "")).name,
        str(record.get("allowed_labels", "")),
    ])


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    raise SystemExit(main())
