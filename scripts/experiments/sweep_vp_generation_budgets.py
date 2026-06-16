#!/usr/bin/env python3
"""Sweep Florence-VP generation budgets and evaluate dense detection quality."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def _parse_int_list(value: str) -> List[int]:
    parsed: List[int] = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        parsed.append(int(item))
    if not parsed:
        raise ValueError(f"Expected at least one integer, got {value!r}")
    return parsed


def _parse_optional_int_list(value: str) -> List[Optional[int]]:
    parsed: List[Optional[int]] = []
    for item in str(value).split(","):
        item = item.strip().lower()
        if not item:
            continue
        parsed.append(None if item in {"none", "default"} else int(item))
    if not parsed:
        raise ValueError(f"Expected at least one integer or none, got {value!r}")
    return parsed


def _parse_optional_float_list(value: str) -> List[Optional[float]]:
    parsed: List[Optional[float]] = []
    for item in str(value).split(","):
        item = item.strip().lower()
        if not item:
            continue
        parsed.append(None if item in {"none", "default"} else float(item))
    if not parsed:
        raise ValueError(f"Expected at least one float or none, got {value!r}")
    return parsed


def _label_part(name: str, value: object) -> str:
    if value is None:
        return f"{name}default"
    text = str(value).replace(".", "p").replace("-", "m")
    return f"{name}{text}"


def _run_command(cmd: List[str], cwd: Path) -> None:
    result = subprocess.run(cmd, cwd=cwd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Command failed with exit code "
            f"{result.returncode}: {' '.join(cmd)}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def _build_inference_cmd(
    args: argparse.Namespace,
    *,
    max_new_tokens: int,
    num_beams: int,
    length_penalty: Optional[float],
    repetition_penalty: Optional[float],
    no_repeat_ngram_size: Optional[int],
    output_dir: Path,
) -> List[str]:
    cmd = [
        sys.executable,
        "scripts/infer/visualize_florence_vp_adapter.py",
        "--model-path",
        args.model_path,
        "--manifest-path",
        args.manifest_path,
        "--split",
        args.split,
        "--output-dir",
        str(output_dir),
        "--max-samples",
        str(args.max_samples),
        "--device",
        args.device,
        "--torch-dtype",
        args.torch_dtype,
        "--max-new-tokens",
        str(max_new_tokens),
        "--num-beams",
        str(num_beams),
        "--structured-vp-mode",
        args.structured_vp_mode,
        "--structured-vp-box-format",
        args.structured_vp_box_format,
        "--structured-vp-marker-style",
        args.structured_vp_marker_style,
        "--structured-vp-filter-policy",
        args.structured_vp_filter_policy,
    ]
    if args.adapter_dir:
        cmd.extend(["--adapter-dir", args.adapter_dir])
    if args.data_path:
        cmd.extend(["--data-path", args.data_path])
    if args.data_key:
        cmd.extend(["--data-key", args.data_key])
    if args.visualization_limit is not None:
        cmd.extend(["--visualization-limit", str(args.visualization_limit)])
    if args.text_input_template:
        cmd.extend(["--text-input-template", args.text_input_template])
    if length_penalty is not None:
        cmd.extend(["--length-penalty", str(length_penalty)])
    if repetition_penalty is not None:
        cmd.extend(["--repetition-penalty", str(repetition_penalty)])
    if no_repeat_ngram_size is not None:
        cmd.extend(["--no-repeat-ngram-size", str(no_repeat_ngram_size)])
    if args.early_stopping:
        cmd.append("--early-stopping")
    if args.stop_after_vp_max_total_boxes:
        cmd.append("--stop-after-vp-max-total-boxes")
    if args.continue_underfilled_vp_boxes:
        cmd.append("--continue-underfilled-vp-boxes")
        cmd.extend(["--vp-continuation-max-rounds", str(args.vp_continuation_max_rounds)])
        cmd.extend(["--vp-continuation-max-new-tokens", str(args.vp_continuation_max_new_tokens)])
        cmd.extend(["--vp-continuation-min-missing-boxes", str(args.vp_continuation_min_missing_boxes)])
    if args.min_query_boxes is not None:
        cmd.extend(["--min-query-boxes", str(args.min_query_boxes)])
    if args.max_query_boxes is not None:
        cmd.extend(["--max-query-boxes", str(args.max_query_boxes)])
    if args.structured_vp_max_boxes_per_label is not None:
        cmd.extend(["--structured-vp-max-boxes-per-label", str(args.structured_vp_max_boxes_per_label)])
    if args.structured_vp_max_total_boxes is not None:
        cmd.extend(["--structured-vp-max-total-boxes", str(args.structured_vp_max_total_boxes)])
    if args.structured_vp_max_total_boxes_field:
        cmd.extend(["--structured-vp-max-total-boxes-field", args.structured_vp_max_total_boxes_field])
    if args.structured_vp_nms_iou_threshold is not None:
        cmd.extend(["--structured-vp-nms-iou-threshold", str(args.structured_vp_nms_iou_threshold)])
    if args.structured_vp_allowed_labels:
        cmd.extend(["--structured-vp-allowed-labels", args.structured_vp_allowed_labels])
    if args.structured_vp_allowed_labels_field:
        cmd.extend(["--structured-vp-allowed-labels-field", args.structured_vp_allowed_labels_field])
    if args.structured_vp_allowed_label_match_mode != "strict":
        cmd.extend([
            "--structured-vp-allowed-label-match-mode",
            args.structured_vp_allowed_label_match_mode,
        ])
    if getattr(args, "structured_vp_repair_malformed_tail", False):
        cmd.append("--structured-vp-repair-malformed-tail")
    return cmd


def _build_policy_sweep_cmd(
    args: argparse.Namespace,
    *,
    summary_path: Path,
    output_dir: Path,
) -> List[str]:
    cmd = [
        sys.executable,
        "scripts/experiments/sweep_vp_quality_policies.py",
        "--summary",
        str(summary_path),
        "--output-dir",
        str(output_dir),
        "--structured-vp-mode",
        "on" if args.structured_vp_mode != "off" else "off",
        "--structured-vp-box-format",
        args.structured_vp_box_format,
        "--structured-vp-marker-style",
        args.structured_vp_marker_style,
        "--structured-vp-filter-policy",
        args.structured_vp_filter_policy,
        "--focus-bucket",
        args.focus_bucket,
        "--max-bad-cases",
        str(args.max_bad_cases),
    ]
    if args.structured_vp_max_boxes_per_label is not None:
        cmd.extend(["--structured-vp-max-boxes-per-label", str(args.structured_vp_max_boxes_per_label)])
    if args.structured_vp_max_total_boxes is not None:
        cmd.extend(["--structured-vp-max-total-boxes", str(args.structured_vp_max_total_boxes)])
    if args.structured_vp_max_total_boxes_field:
        cmd.extend(["--structured-vp-max-total-boxes-field", args.structured_vp_max_total_boxes_field])
    if args.structured_vp_nms_iou_threshold is not None:
        cmd.extend(["--structured-vp-nms-iou-threshold", str(args.structured_vp_nms_iou_threshold)])
    if args.structured_vp_allowed_labels:
        cmd.extend(["--structured-vp-allowed-labels", args.structured_vp_allowed_labels])
    if args.structured_vp_allowed_labels_field:
        cmd.extend(["--structured-vp-allowed-labels-field", args.structured_vp_allowed_labels_field])
    if args.include_phrase_label_policy:
        cmd.append("--include-phrase-label-policy")
    if args.include_target_label_oracle:
        cmd.append("--include-target-label-oracle")
    if getattr(args, "include_repair_policy", False):
        cmd.append("--include-repair-policy")
    if args.structured_vp_allowed_label_match_mode != "strict":
        cmd.extend([
            "--structured-vp-allowed-label-match-mode",
            args.structured_vp_allowed_label_match_mode,
        ])
    if getattr(args, "structured_vp_repair_malformed_tail", False):
        cmd.append("--structured-vp-repair-malformed-tail")
    return cmd


def _best_row(policy_sweep: Dict[str, Any]) -> Dict[str, Any]:
    comparison = policy_sweep.get("comparison") or {}
    rows = comparison.get("ranked_rows") or comparison.get("rows") or []
    return dict(rows[0]) if rows else {}


def _score_row(row: Dict[str, Any], focus_bucket: str) -> float:
    if focus_bucket:
        value = row.get("focus_f1")
        if value is not None:
            return float(value)
    return float(row.get("f1") or 0.0)


def _run_budget(
    args: argparse.Namespace,
    max_new_tokens: int,
    num_beams: int,
    length_penalty: Optional[float],
    repetition_penalty: Optional[float],
    no_repeat_ngram_size: Optional[int],
) -> Dict[str, Any]:
    label = "_".join([
        f"tokens{max_new_tokens}",
        f"beams{num_beams}",
        _label_part("lp", length_penalty),
        _label_part("rp", repetition_penalty),
        _label_part("ngram", no_repeat_ngram_size),
    ])
    output_root = Path(args.output_dir).expanduser()
    inference_dir = output_root / label / "inference"
    policy_dir = output_root / label / "policy_sweep"
    inference_dir.mkdir(parents=True, exist_ok=True)
    policy_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parents[2]
    infer_cmd = _build_inference_cmd(
        args,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        length_penalty=length_penalty,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        output_dir=inference_dir,
    )
    _run_command(infer_cmd, repo_root)
    summary_path = inference_dir / "vp_inference_visualization_summary.json"

    sweep_cmd = _build_policy_sweep_cmd(args, summary_path=summary_path, output_dir=policy_dir)
    _run_command(sweep_cmd, repo_root)
    policy_sweep_path = policy_dir / "vp_quality_policy_sweep.json"

    inference_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    policy_sweep = json.loads(policy_sweep_path.read_text(encoding="utf-8"))
    best = _best_row(policy_sweep)

    return {
        "label": label,
        "max_new_tokens": max_new_tokens,
        "num_beams": num_beams,
        "length_penalty": length_penalty,
        "repetition_penalty": repetition_penalty,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "early_stopping": args.early_stopping,
        "stop_after_vp_max_total_boxes": args.stop_after_vp_max_total_boxes,
        "continue_underfilled_vp_boxes": args.continue_underfilled_vp_boxes,
        "structured_vp_repair_malformed_tail": args.structured_vp_repair_malformed_tail,
        "include_repair_policy": args.include_repair_policy,
        "inference_summary_path": str(summary_path),
        "policy_sweep_path": str(policy_sweep_path),
        "recommended_policy": policy_sweep.get("recommended_policy"),
        "score": _score_row(best, args.focus_bucket),
        "best_policy_row": best,
        "num_samples": inference_summary.get("num_samples", 0),
        "filtered_row_count": inference_summary.get("filtered_row_count", 0),
        "avg_pred_boxes": inference_summary.get("avg_pred_boxes", 0.0),
        "avg_gt_boxes": inference_summary.get("avg_gt_boxes", 0.0),
        "generation_budget_hit_ratio": inference_summary.get("generation_budget_hit_ratio", 0.0),
        "generation_budget_near_hit_ratio": inference_summary.get("generation_budget_near_hit_ratio", 0.0),
        "dense_generation_budget_hit_ratio": inference_summary.get("dense_generation_budget_hit_ratio", 0.0),
        "dense_generation_budget_near_hit_ratio": inference_summary.get("dense_generation_budget_near_hit_ratio", 0.0),
        "avg_raw_prediction_token_count": inference_summary.get("avg_raw_prediction_token_count", 0.0),
        "dense_avg_raw_prediction_token_count": inference_summary.get("dense_avg_raw_prediction_token_count", 0.0),
        "max_raw_loc_token_count": inference_summary.get("max_raw_loc_token_count", 0),
        "vp_count_stopping_available_ratio": inference_summary.get("vp_count_stopping_available_ratio", 0.0),
        "vp_count_stopping_targeted_ratio": inference_summary.get("vp_count_stopping_targeted_ratio", 0.0),
        "vp_count_stopping_triggered_ratio": inference_summary.get("vp_count_stopping_triggered_ratio", 0.0),
        "vp_continuation_attempted_ratio": inference_summary.get("vp_continuation_attempted_ratio", 0.0),
        "vp_continuation_applied_ratio": inference_summary.get("vp_continuation_applied_ratio", 0.0),
        "vp_continuation_reached_target_ratio": inference_summary.get("vp_continuation_reached_target_ratio", 0.0),
    }


def _render_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Florence-VP Generation Budget Sweep",
        "",
        f"- focus bucket: `{report.get('focus_bucket')}`",
        f"- recommended budget: `{report.get('recommended_label')}`",
        "",
        "| rank | run | policy | samples | avg pred | avg GT | budget near-hit | dense near-hit | focus recall | focus f1 | focus avg pred | focus FN |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(report.get("ranked_runs", []), start=1):
        best = row.get("best_policy_row") or {}
        lines.append(
            "| "
            f"{rank} | `{row.get('label')}` | `{best.get('policy')}` | "
            f"{int(row.get('num_samples') or 0)} | "
            f"{float(row.get('avg_pred_boxes') or 0.0):.2f} | "
            f"{float(row.get('avg_gt_boxes') or 0.0):.2f} | "
            f"{float(row.get('generation_budget_near_hit_ratio') or 0.0):.4f} | "
            f"{float(row.get('dense_generation_budget_near_hit_ratio') or 0.0):.4f} | "
            f"{float(best.get('focus_recall') or best.get('recall') or 0.0):.4f} | "
            f"{float(best.get('focus_f1') or best.get('f1') or 0.0):.4f} | "
            f"{float(best.get('focus_avg_pred_boxes') or best.get('avg_pred_boxes') or 0.0):.2f} | "
            f"{int(best.get('focus_false_negatives') or best.get('false_negatives') or 0)} |"
        )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    runs: List[Dict[str, Any]] = []
    for max_new_tokens in _parse_int_list(args.max_new_tokens_list):
        for num_beams in _parse_int_list(args.num_beams_list):
            for length_penalty in _parse_optional_float_list(args.length_penalty_list):
                for repetition_penalty in _parse_optional_float_list(args.repetition_penalty_list):
                    for no_repeat_ngram_size in _parse_optional_int_list(args.no_repeat_ngram_size_list):
                        runs.append(_run_budget(
                            args,
                            max_new_tokens,
                            num_beams,
                            length_penalty,
                            repetition_penalty,
                            no_repeat_ngram_size,
                        ))

    ranked_runs = sorted(
        runs,
        key=lambda row: (
            -float(row.get("score") or 0.0),
            int(row["max_new_tokens"]),
            int(row["num_beams"]),
            float(row["length_penalty"] if row["length_penalty"] is not None else 1.0),
        ),
    )
    report = {
        "model_path": args.model_path,
        "adapter_dir": args.adapter_dir,
        "data_path": args.data_path,
        "data_key": args.data_key,
        "output_dir": str(output_dir),
        "focus_bucket": args.focus_bucket,
        "text_input_template": args.text_input_template,
        "early_stopping": args.early_stopping,
        "stop_after_vp_max_total_boxes": args.stop_after_vp_max_total_boxes,
        "continue_underfilled_vp_boxes": args.continue_underfilled_vp_boxes,
        "structured_vp_repair_malformed_tail": args.structured_vp_repair_malformed_tail,
        "include_repair_policy": args.include_repair_policy,
        "vp_continuation_max_rounds": args.vp_continuation_max_rounds,
        "vp_continuation_max_new_tokens": args.vp_continuation_max_new_tokens,
        "vp_continuation_min_missing_boxes": args.vp_continuation_min_missing_boxes,
        "min_query_boxes": args.min_query_boxes,
        "max_query_boxes": args.max_query_boxes,
        "runs": runs,
        "ranked_runs": ranked_runs,
        "recommended_label": ranked_runs[0]["label"] if ranked_runs else None,
        "summary_path": str(output_dir / "vp_generation_budget_sweep.json"),
        "markdown_path": str(output_dir / "vp_generation_budget_sweep.md"),
    }
    Path(report["summary_path"]).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    Path(report["markdown_path"]).write_text(_render_markdown(report), encoding="utf-8")
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-dir", default=None)
    parser.add_argument("--manifest-path", default=".codex_reports/florence_vp_saved_adapter_smoke/vp_real_data_manifest.json")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--data-key", default=None)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--output-dir", default=".codex_reports/florence_vp_generation_budget_sweep")
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--torch-dtype", default="float32", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--min-query-boxes", type=int, default=None)
    parser.add_argument("--max-query-boxes", type=int, default=None)
    parser.add_argument("--max-new-tokens-list", default="64,96,128")
    parser.add_argument("--num-beams-list", default="1")
    parser.add_argument("--length-penalty-list", default="none")
    parser.add_argument("--repetition-penalty-list", default="none")
    parser.add_argument("--no-repeat-ngram-size-list", default="none")
    parser.add_argument("--early-stopping", action="store_true")
    parser.add_argument("--stop-after-vp-max-total-boxes", action="store_true")
    parser.add_argument("--continue-underfilled-vp-boxes", action="store_true")
    parser.add_argument("--vp-continuation-max-rounds", type=int, default=1)
    parser.add_argument("--vp-continuation-max-new-tokens", type=int, default=48)
    parser.add_argument("--vp-continuation-min-missing-boxes", type=int, default=1)
    parser.add_argument(
        "--text-input-template",
        default=None,
        help="Optional generation-only text_input template, e.g. 'all {text_input}'.",
    )
    parser.add_argument("--visualization-limit", type=int, default=0)
    parser.add_argument("--structured-vp-mode", default="auto", choices=["off", "auto", "on"])
    parser.add_argument("--structured-vp-box-format", default="loc_tokens", choices=["loc_tokens", "json"])
    parser.add_argument("--structured-vp-marker-style", default="plain", choices=["special", "plain"])
    parser.add_argument("--structured-vp-filter-policy", default="nms", choices=["none", "auto", "single-target", "nms"])
    parser.add_argument("--structured-vp-max-boxes-per-label", type=int, default=None)
    parser.add_argument("--structured-vp-max-total-boxes", type=int, default=None)
    parser.add_argument("--structured-vp-max-total-boxes-field", default=None)
    parser.add_argument("--structured-vp-nms-iou-threshold", type=float, default=0.5)
    parser.add_argument("--structured-vp-allowed-labels", default=None)
    parser.add_argument("--structured-vp-allowed-labels-field", default="text_input")
    parser.add_argument(
        "--include-phrase-label-policy",
        action="store_true",
        help=(
            "Also evaluate a contains-match allowed-label policy. Useful for prompt diagnostics "
            "that may emit labels such as 'all person' while preserving strict metrics separately."
        ),
    )
    parser.add_argument("--include-target-label-oracle", action="store_true")
    parser.add_argument("--include-repair-policy", action="store_true")
    parser.add_argument(
        "--structured-vp-allowed-label-match-mode",
        default="strict",
        choices=["strict", "contains"],
    )
    parser.add_argument("--structured-vp-repair-malformed-tail", action="store_true")
    parser.add_argument("--focus-bucket", default="dense", choices=["single", "medium", "dense"])
    parser.add_argument("--max-bad-cases", type=int, default=20)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    print(json.dumps(run(parse_args(argv)), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
