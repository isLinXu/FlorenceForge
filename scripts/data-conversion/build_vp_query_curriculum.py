#!/usr/bin/env python3
"""Build a multi-instance curriculum JSONL from VP query-grounding samples."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from florence_forge.evaluation.visual_primitive_parser import VisualPrimitiveParser


BUCKETS = ("single", "medium", "dense")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    parser = VisualPrimitiveParser()

    rows = _read_jsonl(input_path)
    weights = {
        "single": _validate_non_negative_int(args.single_weight, "single_weight"),
        "medium": _validate_non_negative_int(args.medium_weight, "medium_weight"),
        "dense": _validate_non_negative_int(args.dense_weight, "dense_weight"),
    }

    output_rows: List[Dict[str, Any]] = []
    bucket_counts: Counter[str] = Counter()
    bucket_output_counts: Counter[str] = Counter()
    query_box_counts: List[int] = []
    output_query_box_counts: List[int] = []
    label_counts: Counter[str] = Counter()
    skipped_rows = 0

    for source_index, row in enumerate(rows):
        box_count = _resolve_query_box_count(row, parser)
        if box_count < args.min_query_boxes:
            skipped_rows += 1
            continue
        bucket = _bucket_for_box_count(box_count)
        repeat_total = weights[bucket]
        bucket_counts[bucket] += 1
        query_box_counts.append(box_count)
        label = _query_label(row)
        if label:
            label_counts[label] += 1

        for repeat_index in range(repeat_total):
            item = dict(row)
            item["curriculum_bucket"] = bucket
            item["curriculum_query_box_count"] = box_count
            item["curriculum_repeat_index"] = repeat_index
            item["curriculum_repeat_total"] = repeat_total
            item["curriculum_source_index"] = source_index
            output_rows.append(item)
            bucket_output_counts[bucket] += 1
            output_query_box_counts.append(box_count)

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(output_rows)

    _write_jsonl(output_path, output_rows)
    summary = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "input_rows": len(rows),
        "output_rows": len(output_rows),
        "skipped_rows": skipped_rows,
        "min_query_boxes": args.min_query_boxes,
        "weights": weights,
        "shuffle": bool(args.shuffle),
        "seed": args.seed if args.shuffle else None,
        "bucket_counts": {bucket: int(bucket_counts.get(bucket, 0)) for bucket in BUCKETS},
        "bucket_output_counts": {bucket: int(bucket_output_counts.get(bucket, 0)) for bucket in BUCKETS},
        "avg_query_box_count_input": _mean(query_box_counts),
        "avg_query_box_count_output": _mean(output_query_box_counts),
        "max_query_box_count": max(query_box_counts) if query_box_counts else 0,
        "top_labels": [
            {"label": label, "count": count}
            for label, count in label_counts.most_common(args.max_labels)
        ],
    }

    summary_path = Path(args.summary_output).expanduser() if args.summary_output else _default_summary_path(output_path)
    markdown_path = (
        Path(args.markdown_output).expanduser()
        if args.markdown_output else summary_path.with_suffix(".md")
    )
    summary["summary_path"] = str(summary_path)
    summary["markdown_path"] = str(markdown_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_path.write_text(render_markdown(summary), encoding="utf-8")
    return summary


def render_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# VP Query Curriculum",
        "",
        f"- Input rows: `{summary.get('input_rows', 0)}`",
        f"- Output rows: `{summary.get('output_rows', 0)}`",
        f"- Skipped rows: `{summary.get('skipped_rows', 0)}`",
        f"- Avg query boxes input: `{float(summary.get('avg_query_box_count_input', 0.0) or 0.0):.4f}`",
        f"- Avg query boxes output: `{float(summary.get('avg_query_box_count_output', 0.0) or 0.0):.4f}`",
        f"- Max query boxes: `{summary.get('max_query_box_count', 0)}`",
        "",
        "## Buckets",
        "",
        "| bucket | input | output | weight |",
        "| --- | ---: | ---: | ---: |",
    ]
    bucket_counts = dict(summary.get("bucket_counts", {}) or {})
    bucket_output_counts = dict(summary.get("bucket_output_counts", {}) or {})
    weights = dict(summary.get("weights", {}) or {})
    for bucket in BUCKETS:
        lines.append(
            f"| `{bucket}` | {int(bucket_counts.get(bucket, 0) or 0)} "
            f"| {int(bucket_output_counts.get(bucket, 0) or 0)} "
            f"| {int(weights.get(bucket, 0) or 0)} |"
        )

    top_labels = list(summary.get("top_labels", []) or [])
    if top_labels:
        lines.extend(["", "## Top Labels", ""])
        for item in top_labels:
            lines.append(f"- `{item.get('label')}`: `{item.get('count')}`")

    return "\n".join(lines) + "\n"


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _resolve_query_box_count(row: Mapping[str, Any], parser: VisualPrimitiveParser) -> int:
    for key in ("query_box_count", "curriculum_query_box_count", "count"):
        value = row.get(key)
        try:
            if value is not None:
                return max(0, int(value))
        except (TypeError, ValueError):
            pass
    detections = parser.parse_detections(str(row.get("suffix", "")))
    return len(detections)


def _bucket_for_box_count(box_count: int) -> str:
    if box_count <= 1:
        return "single"
    if box_count <= 3:
        return "medium"
    return "dense"


def _query_label(row: Mapping[str, Any]) -> str:
    for key in ("query_label", "text_input", "count_label"):
        value = str(row.get(key, "") or "").strip()
        if value:
            return value
    return ""


def _mean(values: Sequence[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def _validate_non_negative_int(value: Any, name: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{name} must be >= 0")
    return parsed


def _default_summary_path(output_path: Path) -> Path:
    suffix = "".join(output_path.suffixes)
    if suffix:
        name = output_path.name[: -len(suffix)]
    else:
        name = output_path.name
    return output_path.with_name(f"{name}_summary.json")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input query-grounding JSONL path.")
    parser.add_argument("--output", "-o", required=True, help="Output curriculum JSONL path.")
    parser.add_argument("--summary-output", default=None, help="Optional JSON summary output path.")
    parser.add_argument("--markdown-output", default=None, help="Optional Markdown summary output path.")
    parser.add_argument("--single-weight", type=int, default=1)
    parser.add_argument("--medium-weight", type=int, default=2)
    parser.add_argument("--dense-weight", type=int, default=3)
    parser.add_argument("--min-query-boxes", type=int, default=1)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-labels", type=int, default=12)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    print(json.dumps(run(parse_args(argv)), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
