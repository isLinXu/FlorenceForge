#!/usr/bin/env python3
"""Mix base VP grounding rows with proposal-distillation hard positives."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def run(args: argparse.Namespace) -> Dict[str, Any]:
    base_paths = [Path(path).expanduser() for path in args.base_input]
    distillation_paths = [Path(path).expanduser() for path in args.distillation_input or []]
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base_rows = _read_many_jsonl(base_paths)
    distillation_rows = _read_many_jsonl(distillation_paths)
    if args.max_base_rows is not None:
        base_rows = base_rows[: max(0, int(args.max_base_rows))]
    distillation_rows = _filter_distillation_rows(
        distillation_rows,
        min_delta_tp=args.distillation_min_delta_tp,
        target_mode=args.distillation_target_mode,
    )
    if args.max_distillation_rows is not None:
        distillation_rows = distillation_rows[: max(0, int(args.max_distillation_rows))]

    output_rows, counters = build_mixed_rows(
        base_rows,
        distillation_rows,
        base_repeat=args.base_repeat,
        distillation_repeat=args.distillation_repeat,
        distillation_repeat_order=args.distillation_repeat_order,
        replace_base_on_distillation_key=args.replace_base_on_distillation_key,
        placement=args.placement,
    )
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(output_rows)

    _write_jsonl(output_path, output_rows)
    summary_path = Path(args.summary_output).expanduser() if args.summary_output else _default_summary_path(output_path)
    markdown_path = Path(args.markdown_output).expanduser() if args.markdown_output else summary_path.with_suffix(".md")
    summary = {
        "base_inputs": [str(path) for path in base_paths],
        "distillation_inputs": [str(path) for path in distillation_paths],
        "output_path": str(output_path),
        "summary_path": str(summary_path),
        "markdown_path": str(markdown_path),
        "base_input_rows": len(base_rows),
        "distillation_input_rows": len(distillation_rows),
        "base_repeat": args.base_repeat,
        "distillation_repeat": args.distillation_repeat,
        "distillation_repeat_order": args.distillation_repeat_order,
        "placement": args.placement,
        "replace_base_on_distillation_key": bool(args.replace_base_on_distillation_key),
        "shuffle": bool(args.shuffle),
        "seed": args.seed if args.shuffle else None,
        "distillation_min_delta_tp": args.distillation_min_delta_tp,
        "distillation_target_mode": args.distillation_target_mode,
        **counters,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_path.write_text(render_markdown(summary), encoding="utf-8")
    return summary


def build_mixed_rows(
    base_rows: Sequence[Mapping[str, Any]],
    distillation_rows: Sequence[Mapping[str, Any]],
    *,
    base_repeat: int = 1,
    distillation_repeat: int = 4,
    distillation_repeat_order: str = "grouped",
    replace_base_on_distillation_key: bool = False,
    placement: str = "append",
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    base_repeat = _validate_non_negative_int(base_repeat, "base_repeat")
    distillation_repeat = _validate_non_negative_int(distillation_repeat, "distillation_repeat")
    if placement not in {"append", "prepend", "interleave"}:
        raise ValueError(f"Unsupported placement: {placement}")
    if distillation_repeat_order not in {"grouped", "round_robin"}:
        raise ValueError(f"Unsupported distillation_repeat_order: {distillation_repeat_order}")
    distillation_keys = {_row_key(row) for row in distillation_rows}
    base_output_rows: List[Dict[str, Any]] = []
    skipped_base_replaced = 0

    for source_index, row in enumerate(base_rows):
        if replace_base_on_distillation_key and _row_key(row) in distillation_keys:
            skipped_base_replaced += 1
            continue
        base_output_rows.extend(_repeat_row(
            row,
            group="base",
            repeat_total=base_repeat,
            source_index=source_index,
        ))

    distillation_output_rows = _repeat_distillation_rows(
        distillation_rows,
        repeat_total=distillation_repeat,
        repeat_order=distillation_repeat_order,
    )

    output_rows = _place_rows(base_output_rows, distillation_output_rows, placement=placement)
    group_counts = Counter(str(row.get("mix_group", "")) for row in output_rows)
    distillation_output_rows = int(group_counts.get("distillation", 0))
    output_count = len(output_rows)
    return output_rows, {
        "output_rows": output_count,
        "base_output_rows": int(group_counts.get("base", 0)),
        "distillation_output_rows": distillation_output_rows,
        "distillation_output_ratio": distillation_output_rows / output_count if output_count else 0.0,
        "skipped_base_replaced_rows": skipped_base_replaced,
        "bucket_counts": _bucket_counts(output_rows),
        "bucket_counts_by_group": _bucket_counts_by_group(output_rows),
        "avg_query_box_count": _mean([_query_box_count(row) for row in output_rows]),
        "top_labels": [
            {"label": label, "count": count}
            for label, count in Counter(_query_label(row) for row in output_rows).most_common(20)
            if label
        ],
    }


def render_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# VP Distillation Mix",
        "",
        f"- Base input rows: `{int(summary.get('base_input_rows', 0) or 0)}`",
        f"- Distillation input rows: `{int(summary.get('distillation_input_rows', 0) or 0)}`",
        f"- Output rows: `{int(summary.get('output_rows', 0) or 0)}`",
        f"- Base / distillation output rows: `{int(summary.get('base_output_rows', 0) or 0)}` / "
        f"`{int(summary.get('distillation_output_rows', 0) or 0)}`",
        f"- Distillation output ratio: `{float(summary.get('distillation_output_ratio', 0.0) or 0.0):.4f}`",
        f"- Avg query boxes: `{float(summary.get('avg_query_box_count', 0.0) or 0.0):.4f}`",
        f"- Repeats base / distillation: `{int(summary.get('base_repeat', 0) or 0)}` / "
        f"`{int(summary.get('distillation_repeat', 0) or 0)}`",
        f"- Distillation repeat order: `{summary.get('distillation_repeat_order', 'grouped')}`",
        f"- Placement: `{summary.get('placement', 'append')}`",
        f"- Replaced base rows: `{int(summary.get('skipped_base_replaced_rows', 0) or 0)}`",
        "",
        "## Buckets",
        "",
        "| bucket | rows |",
        "| --- | ---: |",
    ]
    for bucket, count in dict(summary.get("bucket_counts", {}) or {}).items():
        lines.append(f"| `{bucket}` | {int(count or 0)} |")
    top_labels = list(summary.get("top_labels", []) or [])
    if top_labels:
        lines.extend(["", "## Top Labels", ""])
        for item in top_labels:
            lines.append(f"- `{item.get('label')}`: `{item.get('count')}`")
    return "\n".join(lines) + "\n"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-input", action="append", required=True)
    parser.add_argument("--distillation-input", action="append", default=[])
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--markdown-output", default=None)
    parser.add_argument("--base-repeat", type=int, default=1)
    parser.add_argument("--distillation-repeat", type=int, default=4)
    parser.add_argument("--distillation-repeat-order", default="grouped", choices=["grouped", "round_robin"])
    parser.add_argument("--max-base-rows", type=int, default=None)
    parser.add_argument("--max-distillation-rows", type=int, default=None)
    parser.add_argument("--distillation-min-delta-tp", type=int, default=None)
    parser.add_argument("--distillation-target-mode", default=None, choices=["teacher", "reference"])
    parser.add_argument("--replace-base-on-distillation-key", action="store_true")
    parser.add_argument(
        "--placement",
        default="append",
        choices=["append", "prepend", "interleave"],
        help=(
            "Where repeated distillation rows are placed before optional global shuffle. "
            "`prepend` makes short-step probes see hard rows early."
        ),
    )
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    print(json.dumps(run(parse_args(argv)), indent=2, ensure_ascii=False))
    return 0


def _repeat_row(
    row: Mapping[str, Any],
    *,
    group: str,
    repeat_total: int,
    source_index: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for repeat_index in range(repeat_total):
        item = dict(row)
        item["mix_group"] = group
        item["mix_source_index"] = source_index
        item["mix_repeat_index"] = repeat_index
        item["mix_repeat_total"] = repeat_total
        item["mix_key"] = _row_key(row)
        rows.append(item)
    return rows


def _repeat_distillation_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    repeat_total: int,
    repeat_order: str,
) -> List[Dict[str, Any]]:
    if repeat_order == "grouped":
        output_rows: List[Dict[str, Any]] = []
        for source_index, row in enumerate(rows):
            output_rows.extend(_repeat_row(
                row,
                group="distillation",
                repeat_total=repeat_total,
                source_index=source_index,
            ))
        return output_rows
    if repeat_order == "round_robin":
        output_rows = []
        for repeat_index in range(repeat_total):
            for source_index, row in enumerate(rows):
                item = dict(row)
                item["mix_group"] = "distillation"
                item["mix_source_index"] = source_index
                item["mix_repeat_index"] = repeat_index
                item["mix_repeat_total"] = repeat_total
                item["mix_key"] = _row_key(row)
                output_rows.append(item)
        return output_rows
    raise ValueError(f"Unsupported distillation repeat order: {repeat_order}")


def _place_rows(
    base_rows: Sequence[Dict[str, Any]],
    distillation_rows: Sequence[Dict[str, Any]],
    *,
    placement: str,
) -> List[Dict[str, Any]]:
    if placement == "append":
        return list(base_rows) + list(distillation_rows)
    if placement == "prepend":
        return list(distillation_rows) + list(base_rows)
    if placement == "interleave":
        return _interleave_rows(base_rows, distillation_rows)
    raise ValueError(f"Unsupported placement: {placement}")


def _interleave_rows(
    base_rows: Sequence[Dict[str, Any]],
    distillation_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    base_count = len(base_rows)
    distillation_count = len(distillation_rows)
    if not base_count:
        return list(distillation_rows)
    if not distillation_count:
        return list(base_rows)

    rows: List[Dict[str, Any]] = []
    consumed_distillation = 0
    for base_index, base_row in enumerate(base_rows):
        target_distillation = min(
            distillation_count,
            ((base_index + 1) * distillation_count + base_count - 1) // base_count,
        )
        rows.extend(distillation_rows[consumed_distillation:target_distillation])
        consumed_distillation = target_distillation
        rows.append(base_row)
    rows.extend(distillation_rows[consumed_distillation:])
    return rows


def _filter_distillation_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    min_delta_tp: Optional[int],
    target_mode: Optional[str],
) -> List[Mapping[str, Any]]:
    filtered: List[Mapping[str, Any]] = []
    for row in rows:
        if min_delta_tp is not None and _safe_int(row.get("distillation_delta_tp")) < int(min_delta_tp):
            continue
        if target_mode and str(row.get("distillation_target_mode", "")) != target_mode:
            continue
        filtered.append(row)
    return filtered


def _read_many_jsonl(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if isinstance(row, dict):
                    item = dict(row)
                    item.setdefault("mix_original_source_path", str(path))
                    rows.append(item)
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _row_key(row: Mapping[str, Any]) -> str:
    image = Path(str(row.get("image", "") or "")).name
    return "|".join([image, _query_label(row)])


def _query_label(row: Mapping[str, Any]) -> str:
    for key in ("query_label", "text_input", "count_label"):
        value = str(row.get(key, "") or "").strip()
        if value:
            return value
    return ""


def _query_box_count(row: Mapping[str, Any]) -> int:
    for key in ("query_box_count", "curriculum_query_box_count", "gt_box_count", "count"):
        value = row.get(key)
        try:
            if value is not None:
                return max(0, int(value))
        except (TypeError, ValueError):
            pass
    return str(row.get("suffix", "") or "").count("<loc_") // 4


def _bucket_counts(rows: Iterable[Mapping[str, Any]]) -> Dict[str, int]:
    counts: Counter[str] = Counter(_bucket(_query_box_count(row)) for row in rows)
    return {bucket: int(counts.get(bucket, 0)) for bucket in ("single", "medium", "dense")}


def _bucket_counts_by_group(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[str, int]]:
    grouped: Dict[str, Counter[str]] = {}
    for row in rows:
        group = str(row.get("mix_group", "unknown") or "unknown")
        grouped.setdefault(group, Counter())[_bucket(_query_box_count(row))] += 1
    return {
        group: {bucket: int(counts.get(bucket, 0)) for bucket in ("single", "medium", "dense")}
        for group, counts in grouped.items()
    }


def _bucket(box_count: int) -> str:
    if box_count <= 1:
        return "single"
    if box_count <= 3:
        return "medium"
    return "dense"


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _mean(values: Sequence[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def _validate_non_negative_int(value: Any, name: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{name} must be >= 0")
    return parsed


def _default_summary_path(output_path: Path) -> Path:
    suffix = "".join(output_path.suffixes)
    name = output_path.name[: -len(suffix)] if suffix else output_path.name
    return output_path.with_name(f"{name}_summary.json")


if __name__ == "__main__":
    raise SystemExit(main())
