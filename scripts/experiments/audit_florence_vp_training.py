#!/usr/bin/env python3
"""Audit Florence-VP training completeness from existing run summaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from florence_forge.evaluation.vp_training_audit import (
    VPTrainingAuditThresholds,
    build_vp_training_audit,
    load_json,
    render_vp_training_audit_markdown,
)


def run(args: argparse.Namespace):
    training_summary = load_json(args.training_summary)
    inference_summaries = [load_json(path) for path in args.inference_summary]
    baseline_summaries = [load_json(path) for path in args.baseline_summary]
    thresholds = VPTrainingAuditThresholds(
        raw_vp_format_threshold=args.raw_vp_format_threshold,
        structured_vp_format_threshold=args.structured_vp_format_threshold,
        decoder_dependency_threshold=args.decoder_dependency_threshold,
        min_train_rows=args.min_train_rows,
        min_val_rows=args.min_val_rows,
        min_training_steps=args.min_training_steps,
        min_delta_norm=args.min_delta_norm,
        min_inference_samples=args.min_inference_samples,
    )

    audit = build_vp_training_audit(
        training_summary=training_summary,
        inference_summaries=inference_summaries,
        baseline_summaries=baseline_summaries,
        thresholds=thresholds,
    )

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "vp_training_audit.json"
    markdown_path = output_dir / "vp_training_audit.md"
    json_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_path.write_text(render_vp_training_audit_markdown(audit), encoding="utf-8")
    audit["audit_json_path"] = str(json_path)
    audit["audit_markdown_path"] = str(markdown_path)
    return audit


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-summary",
        default=".codex_reports/florence_vp_loc_token_smoke/real_florence_vp_training_smoke_summary.json",
        help="Path to real_florence_vp_training_smoke_summary.json.",
    )
    parser.add_argument(
        "--inference-summary",
        action="append",
        default=[],
        help="Path to vp_inference_visualization_summary.json. Can be passed multiple times.",
    )
    parser.add_argument(
        "--baseline-summary",
        action="append",
        default=[],
        help="Optional base Florence inference summary. Can be passed multiple times.",
    )
    parser.add_argument("--output-dir", default=".codex_reports/florence_vp_training_audit")
    parser.add_argument("--raw-vp-format-threshold", type=float, default=0.95)
    parser.add_argument("--structured-vp-format-threshold", type=float, default=0.95)
    parser.add_argument("--decoder-dependency-threshold", type=float, default=0.50)
    parser.add_argument("--min-train-rows", type=int, default=1)
    parser.add_argument("--min-val-rows", type=int, default=1)
    parser.add_argument("--min-training-steps", type=int, default=1)
    parser.add_argument("--min-delta-norm", type=float, default=0.0)
    parser.add_argument("--min-inference-samples", type=int, default=1)
    args = parser.parse_args(argv)
    if not args.inference_summary:
        args.inference_summary = [
            ".codex_reports/florence_vp_structured_decoder_visualizations/train/vp_inference_visualization_summary.json"
        ]
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    print(json.dumps(run(parse_args(argv)), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
