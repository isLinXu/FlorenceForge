#!/usr/bin/env python3
"""Run a reproducible Florence-VP train/infer/audit experiment."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_MODEL_PATH = Path.home() / "Downloads" / "Florence2_det_base_ovd-v3-1751283651704-model"
DEFAULT_TRAINING_SUMMARY = Path(".codex_reports/florence_vp_loc_token_smoke/real_florence_vp_training_smoke_summary.json")
DEFAULT_MANIFEST = Path(".codex_reports/florence_vp_loc_token_smoke/vp_real_data_manifest.json")


def build_experiment_plan(args: argparse.Namespace) -> Dict[str, Any]:
    """Build the command plan for a Florence-VP experiment."""

    output_dir = Path(args.output_dir).expanduser()
    training_output_dir = Path(args.training_output_dir).expanduser() if args.training_output_dir else output_dir / "training"
    adapter_output_dir = Path(args.adapter_output_dir).expanduser() if args.adapter_output_dir else output_dir / "adapter_inference"
    baseline_output_dir = Path(args.baseline_output_dir).expanduser() if args.baseline_output_dir else output_dir / "baseline_inference"
    audit_output_dir = Path(args.audit_output_dir).expanduser() if args.audit_output_dir else output_dir / "audit"
    token_probe_output_dir = (
        Path(args.token_probe_output_dir).expanduser()
        if args.token_probe_output_dir else output_dir / "token_probe"
    )
    filter_replay_output_dir = (
        Path(args.filter_replay_output_dir).expanduser()
        if args.filter_replay_output_dir else output_dir / "postfilter_replay"
    )
    quality_output_dir = (
        Path(args.quality_output_dir).expanduser()
        if args.quality_output_dir else output_dir / "quality"
    )
    policy_comparison_output_dir = (
        Path(args.policy_comparison_output_dir).expanduser()
        if args.policy_comparison_output_dir else output_dir / "policy_comparison"
    )
    record_comparison_output_dir = (
        Path(args.record_comparison_output_dir).expanduser()
        if args.record_comparison_output_dir else output_dir / "record_comparison"
    )
    target_count_gap_output_dir = (
        Path(args.target_count_gap_output_dir).expanduser()
        if args.target_count_gap_output_dir else output_dir / "target_count_gap"
    )
    distillation_mix_output_dir = (
        Path(args.distillation_mix_output_dir).expanduser()
        if args.distillation_mix_output_dir else output_dir / "distillation_mix"
    )
    policy_sweep_output_dir = (
        Path(args.policy_sweep_output_dir).expanduser()
        if args.policy_sweep_output_dir else output_dir / "policy_sweep"
    )
    report_card_output_dir = (
        Path(args.report_card_output_dir).expanduser()
        if args.report_card_output_dir else output_dir / "report_card"
    )

    model_path = str(Path(args.model_path).expanduser())
    training_summary_path = Path(args.training_summary).expanduser()
    manifest_path = Path(args.manifest_path).expanduser()
    adapter_dir = args.adapter_dir
    include_grounding = bool(args.include_grounding or args.run_distillation_mix)
    effective_grounding_train_path = args.grounding_train_path
    distillation_mix_base_inputs = list(args.distillation_mix_base_input or [])
    if args.run_distillation_mix and not distillation_mix_base_inputs and args.grounding_train_path:
        distillation_mix_base_inputs = [args.grounding_train_path]
    distillation_mix_output_path: Optional[Path] = None
    distillation_mix_summary_path: Optional[Path] = None
    distillation_mix_markdown_path: Optional[Path] = None

    commands: List[Dict[str, Any]] = []

    if args.run_distillation_mix:
        distillation_mix_output_path = (
            Path(args.distillation_mix_output).expanduser()
            if args.distillation_mix_output
            else distillation_mix_output_dir / "query_ovd_distill_mix.jsonl"
        )
        distillation_mix_summary_path = (
            Path(args.distillation_mix_summary_output).expanduser()
            if args.distillation_mix_summary_output
            else _default_jsonl_summary_path(distillation_mix_output_path)
        )
        distillation_mix_markdown_path = (
            Path(args.distillation_mix_markdown_output).expanduser()
            if args.distillation_mix_markdown_output
            else distillation_mix_summary_path.with_suffix(".md")
        )
        effective_grounding_train_path = str(distillation_mix_output_path)
        commands.append({
            "name": "build_distillation_mix",
            "cmd": _build_distillation_mix_cmd(
                base_inputs=distillation_mix_base_inputs,
                distillation_inputs=args.distillation_mix_input,
                output=str(distillation_mix_output_path),
                summary_output=str(distillation_mix_summary_path),
                markdown_output=str(distillation_mix_markdown_path),
                base_repeat=args.distillation_mix_base_repeat,
                distillation_repeat=args.distillation_mix_repeat,
                distillation_repeat_order=args.distillation_mix_repeat_order,
                max_base_rows=args.distillation_mix_max_base_rows,
                max_distillation_rows=args.distillation_mix_max_rows,
                distillation_min_delta_tp=args.distillation_mix_min_delta_tp,
                distillation_target_mode=args.distillation_mix_target_mode,
                replace_base_on_distillation_key=args.distillation_mix_replace_base_on_key,
                placement=args.distillation_mix_placement,
                shuffle=args.distillation_mix_shuffle,
                seed=args.distillation_mix_seed,
            ),
            "output_path": str(distillation_mix_output_path),
            "summary_path": str(distillation_mix_summary_path),
            "markdown_path": str(distillation_mix_markdown_path),
        })

    if args.run_training:
        train_cmd = [
            sys.executable,
            "scripts/smoke/real_florence_vp_training_smoke.py",
            "--model-path",
            model_path,
            "--output-dir",
            str(training_output_dir),
            "--training-mode",
            "lora",
            "--vp-box-format",
            "loc_tokens",
            "--vp-marker-style",
            args.vp_marker_style,
            "--train-vp-head",
            "--save-adapter",
            "--max-steps",
            str(args.training_steps),
            "--max-train-samples",
            str(args.max_train_samples),
            "--max-val-samples",
            str(args.max_val_samples),
            "--learning-rate",
            str(args.learning_rate),
            "--lora-task-type",
            args.lora_task_type,
            "--training-data-order",
            args.training_data_order,
            "--device",
            args.device,
            "--torch-dtype",
            args.torch_dtype,
        ]
        if args.shuffle_train_data:
            train_cmd.append("--shuffle-train-data")
            train_cmd.extend(["--shuffle-seed", str(args.shuffle_seed)])
        if args.skip_od_training_data:
            train_cmd.append("--skip-od-training-data")
        if args.include_count:
            train_cmd.append("--include-count")
        if include_grounding:
            train_cmd.extend(["--include-grounding", "--grounding-task-type", args.grounding_task_type])
            if effective_grounding_train_path:
                train_cmd.extend(["--grounding-train-path", effective_grounding_train_path])
            if args.grounding_val_path:
                train_cmd.extend(["--grounding-val-path", args.grounding_val_path])
            if args.grounding_selection != "od-split":
                train_cmd.extend(["--grounding-selection", args.grounding_selection])
            if args.grounding_curriculum != "none":
                train_cmd.extend(["--grounding-curriculum", args.grounding_curriculum])
                train_cmd.extend(["--grounding-single-weight", str(args.grounding_single_weight)])
                train_cmd.extend(["--grounding-medium-weight", str(args.grounding_medium_weight)])
                train_cmd.extend(["--grounding-dense-weight", str(args.grounding_dense_weight)])
            if args.grounding_min_query_boxes is not None:
                train_cmd.extend(["--grounding-min-query-boxes", str(args.grounding_min_query_boxes)])
            if args.grounding_max_query_boxes is not None:
                train_cmd.extend(["--grounding-max-query-boxes", str(args.grounding_max_query_boxes)])
            if args.grounding_count_hint_template:
                train_cmd.extend(["--grounding-count-hint-template", args.grounding_count_hint_template])
                train_cmd.extend(["--grounding-count-hint-splits", args.grounding_count_hint_splits])
        commands.append({
            "name": "train_adapter",
            "cmd": train_cmd,
            "output_dir": str(training_output_dir),
        })
        training_summary_path = training_output_dir / "real_florence_vp_training_smoke_summary.json"
        manifest_path = training_output_dir / "vp_real_data_manifest.json"
        adapter_dir = str(training_output_dir / "adapter")

    if adapter_dir is None:
        adapter_dir = _read_adapter_dir_from_summary(training_summary_path)

    adapter_summary_path = adapter_output_dir / "vp_inference_visualization_summary.json"
    if not args.skip_adapter_infer:
        commands.append({
            "name": "infer_adapter",
            "cmd": _build_visualize_cmd(
                model_path=model_path,
                manifest_path=str(manifest_path),
                output_dir=str(adapter_output_dir),
                adapter_dir=adapter_dir,
                data_key=args.manifest_data_key,
                split=args.split,
                max_samples=args.max_samples,
                min_query_boxes=args.inference_min_query_boxes,
                max_query_boxes=args.inference_max_query_boxes,
                device=args.device,
                torch_dtype=args.torch_dtype,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                length_penalty=args.length_penalty,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                early_stopping=args.early_stopping,
                visualization_limit=args.visualization_limit,
                structured_vp_mode=args.structured_vp_mode,
                structured_box_format=args.structured_vp_box_format,
                structured_marker_style=args.structured_vp_marker_style,
                structured_filter_policy=args.structured_vp_filter_policy,
                structured_max_boxes_per_label=args.structured_vp_max_boxes_per_label,
                structured_max_total_boxes=args.structured_vp_max_total_boxes,
                structured_max_total_boxes_field=args.structured_vp_max_total_boxes_field,
                structured_nms_iou_threshold=args.structured_vp_nms_iou_threshold,
                structured_allowed_labels=args.structured_vp_allowed_labels,
                structured_allowed_labels_field=args.structured_vp_allowed_labels_field,
                structured_allowed_label_match_mode=args.structured_vp_allowed_label_match_mode,
                structured_repair_malformed_tail=args.structured_vp_repair_malformed_tail,
                decoder_prefix=args.decoder_prefix,
                stop_after_vp_max_total_boxes=args.stop_after_vp_max_total_boxes,
                continue_underfilled_vp_boxes=args.continue_underfilled_vp_boxes,
                vp_continuation_max_rounds=args.vp_continuation_max_rounds,
                vp_continuation_max_new_tokens=args.vp_continuation_max_new_tokens,
                vp_continuation_min_missing_boxes=args.vp_continuation_min_missing_boxes,
            ),
            "summary_path": str(adapter_summary_path),
        })

    baseline_summary_path = baseline_output_dir / "vp_inference_visualization_summary.json"
    if not args.skip_baseline:
        commands.append({
            "name": "infer_baseline",
            "cmd": _build_visualize_cmd(
                model_path=model_path,
                manifest_path=str(manifest_path),
                output_dir=str(baseline_output_dir),
                adapter_dir=None,
                data_key=args.manifest_data_key,
                split=args.split,
                max_samples=args.max_samples,
                min_query_boxes=args.inference_min_query_boxes,
                max_query_boxes=args.inference_max_query_boxes,
                device=args.device,
                torch_dtype=args.torch_dtype,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                length_penalty=args.length_penalty,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                early_stopping=args.early_stopping,
                visualization_limit=args.visualization_limit,
                structured_vp_mode=args.structured_vp_mode,
                structured_box_format=args.structured_vp_box_format,
                structured_marker_style=args.structured_vp_marker_style,
                structured_filter_policy=args.structured_vp_filter_policy,
                structured_max_boxes_per_label=args.structured_vp_max_boxes_per_label,
                structured_max_total_boxes=args.structured_vp_max_total_boxes,
                structured_max_total_boxes_field=args.structured_vp_max_total_boxes_field,
                structured_nms_iou_threshold=args.structured_vp_nms_iou_threshold,
                structured_allowed_labels=args.structured_vp_allowed_labels,
                structured_allowed_labels_field=args.structured_vp_allowed_labels_field,
                structured_allowed_label_match_mode=args.structured_vp_allowed_label_match_mode,
                structured_repair_malformed_tail=args.structured_vp_repair_malformed_tail,
                decoder_prefix=args.decoder_prefix,
                stop_after_vp_max_total_boxes=args.stop_after_vp_max_total_boxes,
                continue_underfilled_vp_boxes=args.continue_underfilled_vp_boxes,
                vp_continuation_max_rounds=args.vp_continuation_max_rounds,
                vp_continuation_max_new_tokens=args.vp_continuation_max_new_tokens,
                vp_continuation_min_missing_boxes=args.vp_continuation_min_missing_boxes,
            ),
            "summary_path": str(baseline_summary_path),
        })

    audit_cmd = [
        sys.executable,
        "scripts/experiments/audit_florence_vp_training.py",
        "--training-summary",
        str(training_summary_path),
        "--inference-summary",
        str(adapter_summary_path),
        "--output-dir",
        str(audit_output_dir),
        "--min-inference-samples",
        str(args.min_inference_samples),
    ]
    if not args.skip_baseline:
        audit_cmd.extend(["--baseline-summary", str(baseline_summary_path)])
    commands.append({
        "name": "audit",
        "cmd": audit_cmd,
        "summary_path": str(audit_output_dir / "vp_training_audit.json"),
        "markdown_path": str(audit_output_dir / "vp_training_audit.md"),
    })

    filter_replay_summary_path = filter_replay_output_dir / "filtered_replay_summary.json"
    if args.run_filter_replay:
        filter_replay_cmd = [
            sys.executable,
            "scripts/experiments/replay_structured_vp_filters.py",
            "--output-dir",
            str(filter_replay_output_dir),
            "--structured-vp-box-format",
            args.structured_vp_box_format,
            "--structured-vp-marker-style",
            args.structured_vp_marker_style,
            "--inference-summary",
            f"adapter={adapter_summary_path}",
        ]
        if not args.skip_baseline:
            filter_replay_cmd.extend([
                "--inference-summary",
                f"baseline={baseline_summary_path}",
            ])
        for config in args.filter_replay_config:
            filter_replay_cmd.extend(["--filter-config", config])
        commands.append({
            "name": "filter_replay",
            "cmd": filter_replay_cmd,
            "summary_path": str(filter_replay_summary_path),
        })

    quality_summaries: Dict[str, str] = {}
    if args.run_quality_eval:
        adapter_quality_path = quality_output_dir / "adapter" / "vp_detection_quality.json"
        quality_summaries["adapter"] = str(adapter_quality_path)
        commands.append({
            "name": "quality_adapter",
            "cmd": _build_quality_cmd(
                summary_path=str(adapter_summary_path),
                output_dir=str(quality_output_dir / "adapter"),
                structured_box_format=args.structured_vp_box_format,
                structured_marker_style=args.structured_vp_marker_style,
                structured_filter_policy=args.structured_vp_filter_policy,
                structured_max_boxes_per_label=args.structured_vp_max_boxes_per_label,
                structured_max_total_boxes=args.structured_vp_max_total_boxes,
                structured_max_total_boxes_field=args.structured_vp_max_total_boxes_field,
                structured_nms_iou_threshold=args.structured_vp_nms_iou_threshold,
                structured_allowed_labels=args.structured_vp_allowed_labels,
                structured_allowed_labels_field=args.structured_vp_allowed_labels_field,
                structured_allowed_label_match_mode=args.structured_vp_allowed_label_match_mode,
                structured_repair_malformed_tail=args.structured_vp_repair_malformed_tail,
                label_match_mode=args.vp_label_match_mode,
                iou_threshold=args.quality_iou_threshold,
                max_bad_cases=args.quality_max_bad_cases,
            ),
            "summary_path": str(adapter_quality_path),
        })
        if not args.skip_baseline:
            baseline_quality_path = quality_output_dir / "baseline" / "vp_detection_quality.json"
            quality_summaries["baseline"] = str(baseline_quality_path)
            commands.append({
                "name": "quality_baseline",
                "cmd": _build_quality_cmd(
                    summary_path=str(baseline_summary_path),
                    output_dir=str(quality_output_dir / "baseline"),
                    structured_box_format=args.structured_vp_box_format,
                    structured_marker_style=args.structured_vp_marker_style,
                    structured_filter_policy=args.structured_vp_filter_policy,
                    structured_max_boxes_per_label=args.structured_vp_max_boxes_per_label,
                    structured_max_total_boxes=args.structured_vp_max_total_boxes,
                    structured_max_total_boxes_field=args.structured_vp_max_total_boxes_field,
                    structured_nms_iou_threshold=args.structured_vp_nms_iou_threshold,
                    structured_allowed_labels=args.structured_vp_allowed_labels,
                    structured_allowed_labels_field=args.structured_vp_allowed_labels_field,
                    structured_allowed_label_match_mode=args.structured_vp_allowed_label_match_mode,
                    structured_repair_malformed_tail=args.structured_vp_repair_malformed_tail,
                    label_match_mode=args.vp_label_match_mode,
                    iou_threshold=args.quality_iou_threshold,
                    max_bad_cases=args.quality_max_bad_cases,
                ),
                "summary_path": str(baseline_quality_path),
            })

    target_count_gap_summaries: Dict[str, str] = {}
    if args.run_target_count_gap_analysis:
        for name, report_path in quality_summaries.items():
            output_subdir = target_count_gap_output_dir / name
            summary_path = output_subdir / "vp_target_count_gap.json"
            target_count_gap_summaries[name] = str(summary_path)
            commands.append({
                "name": f"target_count_gap_{name}",
                "cmd": _build_target_count_gap_cmd(
                    report=report_path,
                    output_dir=str(output_subdir),
                    focus_bucket=args.target_count_gap_focus_bucket,
                    max_rows=args.target_count_gap_max_rows,
                ),
                "summary_path": str(summary_path),
                "markdown_path": str(output_subdir / "vp_target_count_gap.md"),
            })

    policy_comparison_reports = list(args.policy_comparison_report)
    policy_comparison_summary_path = policy_comparison_output_dir / "vp_policy_comparison.json"
    if args.run_policy_comparison:
        if not policy_comparison_reports:
            policy_comparison_reports = [
                f"{name}={path}"
                for name, path in quality_summaries.items()
            ]
        commands.append({
            "name": "policy_comparison",
            "cmd": _build_policy_comparison_cmd(
                reports=policy_comparison_reports,
                output_dir=str(policy_comparison_output_dir),
                focus_bucket=args.policy_comparison_focus_bucket,
            ),
            "summary_path": str(policy_comparison_summary_path),
            "markdown_path": str(policy_comparison_output_dir / "vp_policy_comparison.md"),
        })

    record_comparison_summary_path = record_comparison_output_dir / "vp_record_comparison.json"
    if args.run_record_comparison:
        commands.append({
            "name": "record_comparison",
            "cmd": _build_record_comparison_cmd(
                candidate_report=quality_summaries["adapter"],
                baseline_report=quality_summaries["baseline"],
                output_dir=str(record_comparison_output_dir),
                candidate_name="adapter",
                baseline_name="baseline",
                focus_bucket=args.record_comparison_focus_bucket,
            ),
            "summary_path": str(record_comparison_summary_path),
            "markdown_path": str(record_comparison_output_dir / "vp_record_comparison.md"),
        })

    policy_sweep_summaries: Dict[str, str] = {}
    if args.run_policy_sweep:
        adapter_sweep_path = policy_sweep_output_dir / "adapter" / "vp_quality_policy_sweep.json"
        policy_sweep_summaries["adapter"] = str(adapter_sweep_path)
        commands.append({
            "name": "policy_sweep_adapter",
            "cmd": _build_policy_sweep_cmd(
                summary_path=str(adapter_summary_path),
                output_dir=str(policy_sweep_output_dir / "adapter"),
                structured_box_format=args.structured_vp_box_format,
                structured_marker_style=args.structured_vp_marker_style,
                structured_filter_policy=args.structured_vp_filter_policy,
                structured_max_boxes_per_label=args.structured_vp_max_boxes_per_label,
                structured_max_total_boxes=args.structured_vp_max_total_boxes,
                structured_max_total_boxes_field=args.structured_vp_max_total_boxes_field,
                structured_nms_iou_threshold=args.structured_vp_nms_iou_threshold,
                structured_allowed_labels=args.structured_vp_allowed_labels,
                structured_allowed_labels_field=args.structured_vp_allowed_labels_field,
                structured_allowed_label_match_mode=args.structured_vp_allowed_label_match_mode,
                structured_repair_malformed_tail=args.structured_vp_repair_malformed_tail,
                label_match_mode=args.vp_label_match_mode,
                policy_configs=args.policy_sweep_config,
                include_target_label_oracle=args.policy_sweep_include_target_label_oracle,
                include_phrase_label_policy=args.policy_sweep_include_phrase_label_policy,
                include_repair_policy=args.policy_sweep_include_repair_policy,
                focus_bucket=args.policy_sweep_focus_bucket,
                iou_threshold=args.quality_iou_threshold,
                max_bad_cases=args.quality_max_bad_cases,
            ),
            "summary_path": str(adapter_sweep_path),
        })
        if not args.skip_baseline:
            baseline_sweep_path = policy_sweep_output_dir / "baseline" / "vp_quality_policy_sweep.json"
            policy_sweep_summaries["baseline"] = str(baseline_sweep_path)
            commands.append({
                "name": "policy_sweep_baseline",
                "cmd": _build_policy_sweep_cmd(
                    summary_path=str(baseline_summary_path),
                    output_dir=str(policy_sweep_output_dir / "baseline"),
                    structured_box_format=args.structured_vp_box_format,
                    structured_marker_style=args.structured_vp_marker_style,
                    structured_filter_policy=args.structured_vp_filter_policy,
                    structured_max_boxes_per_label=args.structured_vp_max_boxes_per_label,
                    structured_max_total_boxes=args.structured_vp_max_total_boxes,
                    structured_max_total_boxes_field=args.structured_vp_max_total_boxes_field,
                    structured_nms_iou_threshold=args.structured_vp_nms_iou_threshold,
                    structured_allowed_labels=args.structured_vp_allowed_labels,
                    structured_allowed_labels_field=args.structured_vp_allowed_labels_field,
                    structured_allowed_label_match_mode=args.structured_vp_allowed_label_match_mode,
                    structured_repair_malformed_tail=args.structured_vp_repair_malformed_tail,
                    label_match_mode=args.vp_label_match_mode,
                    policy_configs=args.policy_sweep_config,
                    include_target_label_oracle=args.policy_sweep_include_target_label_oracle,
                    include_phrase_label_policy=args.policy_sweep_include_phrase_label_policy,
                    include_repair_policy=args.policy_sweep_include_repair_policy,
                    focus_bucket=args.policy_sweep_focus_bucket,
                    iou_threshold=args.quality_iou_threshold,
                    max_bad_cases=args.quality_max_bad_cases,
                ),
                "summary_path": str(baseline_sweep_path),
            })

    report_card_summaries: Dict[str, str] = {}
    if args.run_report_card:
        for name, quality_path in quality_summaries.items():
            output_subdir = report_card_output_dir / name
            report_card_path = output_subdir / "vp_report_card.json"
            report_card_summaries[name] = str(report_card_path)
            commands.append({
                "name": f"report_card_{name}",
                "cmd": _build_report_card_cmd(
                    quality_report=quality_path,
                    output_dir=str(output_subdir),
                    policy_sweep=policy_sweep_summaries.get(name),
                    target_count_gap=target_count_gap_summaries.get(name),
                    focus_bucket=args.report_card_focus_bucket,
                    max_gap_rows=args.report_card_max_gap_rows,
                    min_samples=args.report_card_min_samples,
                    min_precision=args.report_card_min_precision,
                    min_recall=args.report_card_min_recall,
                    min_f1=args.report_card_min_f1,
                    max_undergeneration_ratio=args.report_card_max_undergeneration_ratio,
                    max_overgeneration_ratio=args.report_card_max_overgeneration_ratio,
                    max_repair_record_ratio=args.report_card_max_repair_record_ratio,
                    min_raw_vp_format_ratio=args.report_card_min_raw_vp_format_ratio,
                    max_structured_decoder_ratio=args.report_card_max_structured_decoder_ratio,
                    min_policy_confidence=args.report_card_min_policy_confidence,
                    high_recoverable_fn_ratio=args.report_card_high_recoverable_fn_ratio,
                ),
                "summary_path": str(report_card_path),
                "markdown_path": str(output_subdir / "vp_report_card.md"),
            })

    token_probe_summary_path = token_probe_output_dir / "florence_vp_token_probe.json"
    if args.run_token_probe:
        token_probe_cmd = [
            sys.executable,
            "scripts/experiments/probe_florence_vp_tokens.py",
            "--model-path",
            model_path,
            "--manifest-path",
            str(manifest_path),
            "--task-type",
            args.grounding_task_type if args.include_grounding else "OD_VP",
            "--split",
            args.split,
            "--output-dir",
            str(token_probe_output_dir),
            "--max-samples",
            str(args.token_probe_max_samples),
            "--max-new-tokens",
            str(args.token_probe_max_new_tokens),
            "--device",
            args.device,
            "--torch-dtype",
            args.torch_dtype,
        ]
        if args.include_grounding:
            token_probe_cmd.extend([
                "--manifest-data-key",
                "train_grounding_effective_path,train_grounding_path",
            ])
        token_probe_cmd.extend(["--marker-style", args.vp_marker_style])
        if adapter_dir:
            token_probe_cmd.extend(["--adapter-dir", adapter_dir])
        commands.append({
            "name": "token_probe",
            "cmd": token_probe_cmd,
            "summary_path": str(token_probe_summary_path),
            "markdown_path": str(token_probe_output_dir / "florence_vp_token_probe.md"),
        })

    return {
        "dry_run": bool(args.dry_run),
        "output_dir": str(output_dir),
        "model_path": model_path,
        "training_summary": str(training_summary_path),
        "manifest_path": str(manifest_path),
        "adapter_dir": adapter_dir,
        "cleanup_adapter_after_audit": bool(args.cleanup_adapter_after_audit),
        "query_grounding_preset": args.query_grounding_preset,
        "include_grounding": include_grounding,
        "grounding_task_type": args.grounding_task_type,
        "grounding_train_path": effective_grounding_train_path,
        "grounding_val_path": args.grounding_val_path,
        "grounding_selection": args.grounding_selection,
        "grounding_curriculum": args.grounding_curriculum,
        "grounding_single_weight": args.grounding_single_weight,
        "grounding_medium_weight": args.grounding_medium_weight,
        "grounding_dense_weight": args.grounding_dense_weight,
        "grounding_min_query_boxes": args.grounding_min_query_boxes,
        "grounding_max_query_boxes": args.grounding_max_query_boxes,
        "grounding_count_hint_template": args.grounding_count_hint_template,
        "grounding_count_hint_splits": args.grounding_count_hint_splits,
        "training_data_order": args.training_data_order,
        "skip_od_training_data": bool(args.skip_od_training_data),
        "shuffle_train_data": bool(args.shuffle_train_data),
        "shuffle_seed": args.shuffle_seed if args.shuffle_train_data else None,
        "lora_task_type": args.lora_task_type,
        "max_new_tokens": args.max_new_tokens,
        "num_beams": args.num_beams,
        "inference_min_query_boxes": args.inference_min_query_boxes,
        "inference_max_query_boxes": args.inference_max_query_boxes,
        "length_penalty": args.length_penalty,
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "early_stopping": bool(args.early_stopping),
        "vp_marker_style": args.vp_marker_style,
        "structured_vp_mode": args.structured_vp_mode,
        "structured_vp_marker_style": args.structured_vp_marker_style,
        "structured_vp_filter_policy": args.structured_vp_filter_policy,
        "structured_vp_max_boxes_per_label": args.structured_vp_max_boxes_per_label,
        "structured_vp_max_total_boxes": args.structured_vp_max_total_boxes,
        "structured_vp_max_total_boxes_field": args.structured_vp_max_total_boxes_field,
        "structured_vp_nms_iou_threshold": args.structured_vp_nms_iou_threshold,
        "structured_vp_allowed_labels": args.structured_vp_allowed_labels,
        "structured_vp_allowed_labels_field": args.structured_vp_allowed_labels_field,
        "structured_vp_repair_malformed_tail": bool(args.structured_vp_repair_malformed_tail),
        "decoder_prefix": args.decoder_prefix,
        "stop_after_vp_max_total_boxes": bool(args.stop_after_vp_max_total_boxes),
        "continue_underfilled_vp_boxes": bool(args.continue_underfilled_vp_boxes),
        "vp_continuation_max_rounds": args.vp_continuation_max_rounds,
        "vp_continuation_max_new_tokens": args.vp_continuation_max_new_tokens,
        "vp_continuation_min_missing_boxes": args.vp_continuation_min_missing_boxes,
        "manifest_data_key": args.manifest_data_key,
        "adapter_inference_summary": str(adapter_summary_path),
        "baseline_inference_summary": None if args.skip_baseline else str(baseline_summary_path),
        "audit_summary": str(audit_output_dir / "vp_training_audit.json"),
        "filter_replay_summary": str(filter_replay_summary_path) if args.run_filter_replay else None,
        "quality_summaries": quality_summaries,
        "target_count_gap_summaries": target_count_gap_summaries,
        "target_count_gap_focus_bucket": args.target_count_gap_focus_bucket,
        "policy_comparison_reports": policy_comparison_reports,
        "policy_comparison_focus_bucket": args.policy_comparison_focus_bucket,
        "policy_comparison_summary": str(policy_comparison_summary_path) if args.run_policy_comparison else None,
        "record_comparison_focus_bucket": args.record_comparison_focus_bucket,
        "record_comparison_summary": str(record_comparison_summary_path) if args.run_record_comparison else None,
        "distillation_mix_output": str(distillation_mix_output_path) if distillation_mix_output_path else None,
        "distillation_mix_summary": str(distillation_mix_summary_path) if distillation_mix_summary_path else None,
        "distillation_mix_markdown": str(distillation_mix_markdown_path) if distillation_mix_markdown_path else None,
        "distillation_mix_base_inputs": distillation_mix_base_inputs if args.run_distillation_mix else [],
        "distillation_mix_inputs": list(args.distillation_mix_input or []) if args.run_distillation_mix else [],
        "distillation_mix_repeat": args.distillation_mix_repeat if args.run_distillation_mix else None,
        "distillation_mix_repeat_order": args.distillation_mix_repeat_order if args.run_distillation_mix else None,
        "distillation_mix_placement": args.distillation_mix_placement if args.run_distillation_mix else None,
        "policy_sweep_summaries": policy_sweep_summaries,
        "policy_sweep_focus_bucket": args.policy_sweep_focus_bucket,
        "policy_sweep_include_repair_policy": bool(args.policy_sweep_include_repair_policy),
        "report_card_summaries": report_card_summaries,
        "report_card_focus_bucket": args.report_card_focus_bucket,
        "report_card_min_samples": args.report_card_min_samples,
        "report_card_min_recall": args.report_card_min_recall,
        "report_card_min_f1": args.report_card_min_f1,
        "report_card_min_raw_vp_format_ratio": args.report_card_min_raw_vp_format_ratio,
        "report_card_max_structured_decoder_ratio": args.report_card_max_structured_decoder_ratio,
        "token_probe_summary": str(token_probe_summary_path) if args.run_token_probe else None,
        "commands": commands,
    }


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    plan = build_experiment_plan(args)
    output_dir = Path(plan["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    executed: List[Dict[str, Any]] = []

    for command in plan["commands"]:
        if args.dry_run:
            executed.append({"name": command["name"], "returncode": None, "skipped": True})
            continue
        result = subprocess.run(command["cmd"], cwd=str(REPO_ROOT), check=False)
        executed.append({"name": command["name"], "returncode": result.returncode, "skipped": False})
        if result.returncode != 0:
            plan["ok"] = False
            plan["executed"] = executed
            plan["failed_command"] = command["name"]
            _write_summary(output_dir, plan)
            return plan

    plan["ok"] = all(item["skipped"] or item["returncode"] == 0 for item in executed)
    plan["executed"] = executed
    plan["cleanup"] = _cleanup_generated_adapter(args, plan) if plan["ok"] else {
        "requested": bool(args.cleanup_adapter_after_audit),
        "deleted": False,
        "reason": "experiment_failed",
    }
    _write_summary(output_dir, plan)
    return plan


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--training-summary", default=str(DEFAULT_TRAINING_SUMMARY))
    parser.add_argument("--adapter-dir", default=None)
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-dir", default=".codex_reports/florence_vp_experiment")
    parser.add_argument("--training-output-dir", default=None)
    parser.add_argument("--adapter-output-dir", default=None)
    parser.add_argument("--baseline-output-dir", default=None)
    parser.add_argument("--audit-output-dir", default=None)
    parser.add_argument("--token-probe-output-dir", default=None)
    parser.add_argument("--filter-replay-output-dir", default=None)
    parser.add_argument("--quality-output-dir", default=None)
    parser.add_argument("--policy-comparison-output-dir", default=None)
    parser.add_argument("--record-comparison-output-dir", default=None)
    parser.add_argument("--target-count-gap-output-dir", default=None)
    parser.add_argument("--distillation-mix-output-dir", default=None)
    parser.add_argument("--policy-sweep-output-dir", default=None)
    parser.add_argument("--report-card-output-dir", default=None)
    parser.add_argument("--run-training", action="store_true")
    parser.add_argument("--run-token-probe", action="store_true")
    parser.add_argument("--run-filter-replay", action="store_true")
    parser.add_argument("--run-quality-eval", action="store_true")
    parser.add_argument("--run-policy-comparison", action="store_true")
    parser.add_argument("--run-record-comparison", action="store_true")
    parser.add_argument("--run-target-count-gap-analysis", action="store_true")
    parser.add_argument("--run-distillation-mix", action="store_true")
    parser.add_argument("--run-policy-sweep", action="store_true")
    parser.add_argument("--run-report-card", action="store_true")
    parser.add_argument(
        "--query-grounding-preset",
        default="none",
        choices=["none", "ovd-nms"],
        help=(
            "Shortcut for proven query-grounding settings. `ovd-nms` uses "
            "OPEN_VOCABULARY_DETECTION, val_grounding_path, text_input allow-list diagnostics, "
            "plain VP markers, NMS@0.5, and a small visualization limit unless explicitly overridden."
        ),
    )
    parser.add_argument("--include-count", action="store_true", help="Include COUNT_VP samples during training.")
    parser.add_argument("--include-grounding", action="store_true", help="Include PHRASE_GROUNDING_VP query samples during training data materialization.")
    parser.add_argument(
        "--grounding-train-path",
        default=None,
        help="Optional query-grounding JSONL passed through as the effective training split.",
    )
    parser.add_argument(
        "--grounding-val-path",
        default=None,
        help="Optional query-grounding JSONL exposed as val_grounding_path in the training manifest.",
    )
    parser.add_argument(
        "--grounding-task-type",
        default="PHRASE_GROUNDING_VP",
        choices=["PHRASE_GROUNDING_VP", "OPEN_VOCABULARY_DETECTION"],
    )
    parser.add_argument(
        "--grounding-selection",
        default="od-split",
        choices=["od-split", "shortest-query", "multi-instance"],
        help="Derived grounding row selection mode used by the training smoke.",
    )
    parser.add_argument(
        "--grounding-curriculum",
        default="none",
        choices=["none", "multi-instance"],
        help="Optional training-data curriculum for derived query grounding samples.",
    )
    parser.add_argument("--grounding-single-weight", type=int, default=1)
    parser.add_argument("--grounding-medium-weight", type=int, default=2)
    parser.add_argument("--grounding-dense-weight", type=int, default=3)
    parser.add_argument("--grounding-min-query-boxes", type=int, default=None)
    parser.add_argument("--grounding-max-query-boxes", type=int, default=None)
    parser.add_argument(
        "--grounding-count-hint-template",
        default=None,
        help=(
            "Optional query-grounding train/val text_input template with count placeholders, "
            "e.g. '{label} | count={query_box_count}'. When used with the ovd-nms preset, "
            "the structured allow-list defaults to query_label."
        ),
    )
    parser.add_argument(
        "--grounding-count-hint-splits",
        default="both",
        choices=["both", "train", "val"],
        help=(
            "Which query-grounding splits should receive --grounding-count-hint-template. "
            "Use 'train' to train with count hints while evaluating clean validation prompts."
        ),
    )
    parser.add_argument(
        "--cleanup-adapter-after-audit",
        action="store_true",
        help="Delete the adapter generated by --run-training after inference and audit finish.",
    )
    parser.add_argument("--skip-adapter-infer", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--manifest-data-key", default=None, help="Optional manifest data key for visualization inference, e.g. val_grounding_path.")
    parser.add_argument("--max-samples", type=int, default=2)
    parser.add_argument(
        "--inference-min-query-boxes",
        type=int,
        default=None,
        help="Optional visualization/eval row filter by query/GT box count.",
    )
    parser.add_argument(
        "--inference-max-query-boxes",
        type=int,
        default=None,
        help="Optional visualization/eval row filter by query/GT box count.",
    )
    parser.add_argument("--min-inference-samples", type=int, default=1)
    parser.add_argument("--training-steps", type=int, default=64)
    parser.add_argument("--max-train-samples", type=int, default=16)
    parser.add_argument("--max-val-samples", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument(
        "--lora-task-type",
        default="SEQ_2_SEQ_LM",
        choices=["CAUSAL_LM", "SEQ_2_SEQ_LM"],
        help="PEFT task type passed to the real training smoke.",
    )
    parser.add_argument(
        "--training-data-order",
        default="as-is",
        choices=["as-is", "grounding-first"],
        help="Task order passed to the real training smoke for short runs.",
    )
    parser.add_argument(
        "--skip-od-training-data",
        action="store_true",
        help="Train only the explicitly enabled non-OD tasks, e.g. external grounding rows for overfit probes.",
    )
    parser.add_argument("--shuffle-train-data", action="store_true")
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument(
        "--distillation-mix-base-input",
        action="append",
        default=[],
        help=(
            "Base query-grounding JSONL used by --run-distillation-mix. "
            "If omitted, --grounding-train-path is used as the base input."
        ),
    )
    parser.add_argument(
        "--distillation-mix-input",
        action="append",
        default=[],
        help="Proposal-distillation JSONL passed to scripts/data-conversion/build_vp_distillation_mix.py.",
    )
    parser.add_argument("--distillation-mix-output", default=None)
    parser.add_argument("--distillation-mix-summary-output", default=None)
    parser.add_argument("--distillation-mix-markdown-output", default=None)
    parser.add_argument("--distillation-mix-base-repeat", type=int, default=1)
    parser.add_argument("--distillation-mix-repeat", type=int, default=4)
    parser.add_argument(
        "--distillation-mix-repeat-order",
        default="grouped",
        choices=["grouped", "round_robin"],
    )
    parser.add_argument("--distillation-mix-max-base-rows", type=int, default=None)
    parser.add_argument("--distillation-mix-max-rows", type=int, default=None)
    parser.add_argument("--distillation-mix-min-delta-tp", type=int, default=None)
    parser.add_argument(
        "--distillation-mix-target-mode",
        default=None,
        choices=["teacher", "reference"],
    )
    parser.add_argument("--distillation-mix-replace-base-on-key", action="store_true")
    parser.add_argument(
        "--distillation-mix-placement",
        default="append",
        choices=["append", "prepend", "interleave"],
    )
    parser.add_argument("--distillation-mix-shuffle", action="store_true")
    parser.add_argument("--distillation-mix-seed", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--torch-dtype", default="float32", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--length-penalty", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=None)
    parser.add_argument("--early-stopping", action="store_true")
    parser.add_argument("--visualization-limit", type=int, default=None)
    parser.add_argument("--structured-vp-mode", default="on", choices=["off", "auto", "on"])
    parser.add_argument("--structured-vp-box-format", default="loc_tokens", choices=["loc_tokens", "json"])
    parser.add_argument("--structured-vp-marker-style", default="special", choices=["special", "plain"])
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
        default="strict",
        choices=["strict", "contains"],
    )
    parser.add_argument("--structured-vp-repair-malformed-tail", action="store_true")
    parser.add_argument("--vp-label-match-mode", default="strict", choices=["strict", "contains"])
    parser.add_argument(
        "--filter-replay-config",
        action="append",
        default=[],
        help=(
            "Optional replay policy for scripts/experiments/replay_structured_vp_filters.py, "
            "e.g. total1:max_total_boxes=1. Can be passed multiple times."
        ),
    )
    parser.add_argument("--quality-iou-threshold", type=float, default=0.5)
    parser.add_argument("--quality-max-bad-cases", type=int, default=20)
    parser.add_argument(
        "--policy-comparison-report",
        action="append",
        default=[],
        help=(
            "Named VP quality report for policy comparison, e.g. "
            "none=.codex_reports/run/quality_unfiltered/vp_detection_quality.json. "
            "If omitted, --run-policy-comparison uses reports produced by --run-quality-eval."
        ),
    )
    parser.add_argument(
        "--policy-comparison-focus-bucket",
        default=None,
        choices=["single", "medium", "dense"],
        help="Rank policy comparison by a box-count bucket instead of overall metrics.",
    )
    parser.add_argument(
        "--record-comparison-focus-bucket",
        default=None,
        choices=["single", "medium", "dense"],
        help="Compare adapter/base record deltas only within one box-count bucket.",
    )
    parser.add_argument(
        "--target-count-gap-focus-bucket",
        default=None,
        choices=["single", "medium", "dense"],
        help="Analyze target-count gap upper bounds only within one box-count bucket.",
    )
    parser.add_argument("--target-count-gap-max-rows", type=int, default=20)
    parser.add_argument(
        "--policy-sweep-config",
        action="append",
        default=[],
        help=(
            "Named policy override for scripts/experiments/sweep_vp_quality_policies.py, "
            "e.g. nms07:filter_policy=nms,nms_iou_threshold=0.7."
        ),
    )
    parser.add_argument("--policy-sweep-include-target-label-oracle", action="store_true")
    parser.add_argument("--policy-sweep-include-phrase-label-policy", action="store_true")
    parser.add_argument("--policy-sweep-include-repair-policy", action="store_true")
    parser.add_argument(
        "--policy-sweep-focus-bucket",
        default=None,
        choices=["single", "medium", "dense"],
        help="Rank each policy sweep by a box-count bucket instead of overall metrics.",
    )
    parser.add_argument(
        "--report-card-focus-bucket",
        default=None,
        choices=["single", "medium", "dense"],
        help="Build report-card checks for one box-count bucket instead of all records.",
    )
    parser.add_argument("--report-card-max-gap-rows", type=int, default=20)
    parser.add_argument("--report-card-min-samples", type=int, default=10)
    parser.add_argument("--report-card-min-precision", type=float, default=0.80)
    parser.add_argument("--report-card-min-recall", type=float, default=0.70)
    parser.add_argument("--report-card-min-f1", type=float, default=0.75)
    parser.add_argument("--report-card-max-undergeneration-ratio", type=float, default=0.35)
    parser.add_argument("--report-card-max-overgeneration-ratio", type=float, default=0.25)
    parser.add_argument("--report-card-max-repair-record-ratio", type=float, default=0.25)
    parser.add_argument("--report-card-min-raw-vp-format-ratio", type=float, default=0.95)
    parser.add_argument("--report-card-max-structured-decoder-ratio", type=float, default=0.50)
    parser.add_argument(
        "--report-card-min-policy-confidence",
        default="moderate",
        choices=["none", "exploratory", "moderate", "strong"],
    )
    parser.add_argument("--report-card-high-recoverable-fn-ratio", type=float, default=0.40)
    parser.add_argument("--vp-marker-style", default="special", choices=["special", "plain"])
    parser.add_argument(
        "--decoder-prefix",
        default=None,
        help=(
            "Optional decoder prefix passed to visualization inference for forced-prefix diagnostics. "
            "Use {label} to insert the first GT label, e.g. '<ref>{label}</ref> <box>'."
        ),
    )
    parser.add_argument(
        "--stop-after-vp-max-total-boxes",
        action="store_true",
        help=(
            "Pass generation-time VP count stopping to visualization inference. "
            "The target count is the resolved structured max_total_boxes per row."
        ),
    )
    parser.add_argument(
        "--continue-underfilled-vp-boxes",
        action="store_true",
        help=(
            "Pass min-count continuation diagnostics to visualization inference "
            "for rows whose raw loc-token box count is below the resolved target count."
        ),
    )
    parser.add_argument("--vp-continuation-max-rounds", type=int, default=1)
    parser.add_argument("--vp-continuation-max-new-tokens", type=int, default=48)
    parser.add_argument("--vp-continuation-min-missing-boxes", type=int, default=1)
    parser.add_argument("--token-probe-max-samples", type=int, default=1)
    parser.add_argument("--token-probe-max-new-tokens", type=int, default=8)
    args = parser.parse_args(argv)
    _apply_query_grounding_preset(args, raw_argv)
    if args.grounding_train_path or args.grounding_val_path or args.grounding_selection != "od-split":
        args.include_grounding = True
    if args.run_distillation_mix:
        args.include_grounding = True
        if not args.distillation_mix_base_input and not args.grounding_train_path:
            parser.error("--run-distillation-mix requires --distillation-mix-base-input or --grounding-train-path")
        if not args.distillation_mix_input:
            parser.error("--run-distillation-mix requires --distillation-mix-input")
    if args.grounding_min_query_boxes is not None or args.grounding_max_query_boxes is not None:
        args.include_grounding = True
    if args.grounding_count_hint_template:
        args.include_grounding = True
        if (
            not _option_present(raw_argv, "--structured-vp-allowed-labels-field")
            and not _option_present(raw_argv, "--structured-vp-allowed-labels")
        ):
            args.structured_vp_allowed_labels_field = "query_label"
    if args.run_policy_comparison and not args.run_quality_eval and not args.policy_comparison_report:
        parser.error("--run-policy-comparison requires --run-quality-eval or --policy-comparison-report")
    if args.run_record_comparison and (not args.run_quality_eval or args.skip_baseline):
        parser.error("--run-record-comparison requires --run-quality-eval and baseline inference")
    if args.run_target_count_gap_analysis and not args.run_quality_eval:
        parser.error("--run-target-count-gap-analysis requires --run-quality-eval")
    if args.run_report_card and not args.run_quality_eval:
        parser.error("--run-report-card requires --run-quality-eval")
    return args


def _apply_query_grounding_preset(args: argparse.Namespace, raw_argv: Sequence[str]) -> None:
    preset = str(getattr(args, "query_grounding_preset", "none") or "none")
    if preset == "none":
        return
    if preset != "ovd-nms":
        raise ValueError(f"Unsupported query grounding preset: {preset}")

    if not _option_present(raw_argv, "--include-grounding"):
        args.include_grounding = True
    if not _option_present(raw_argv, "--grounding-task-type"):
        args.grounding_task_type = "OPEN_VOCABULARY_DETECTION"
    if not _option_present(raw_argv, "--grounding-curriculum"):
        args.grounding_curriculum = "multi-instance"
    if not _option_present(raw_argv, "--grounding-selection"):
        args.grounding_selection = "multi-instance"
    if not _option_present(raw_argv, "--training-data-order"):
        args.training_data_order = "grounding-first"
    if not _option_present(raw_argv, "--manifest-data-key"):
        args.manifest_data_key = "val_grounding_path"
    if not _option_present(raw_argv, "--structured-vp-filter-policy"):
        args.structured_vp_filter_policy = "nms"
    if not _option_present(raw_argv, "--structured-vp-nms-iou-threshold"):
        args.structured_vp_nms_iou_threshold = 0.5
    if (
        not _option_present(raw_argv, "--structured-vp-allowed-labels-field")
        and not _option_present(raw_argv, "--structured-vp-allowed-labels")
    ):
        args.structured_vp_allowed_labels_field = (
            "query_label" if args.grounding_count_hint_template else "text_input"
        )
    if not _option_present(raw_argv, "--structured-vp-marker-style"):
        args.structured_vp_marker_style = "plain"
    if not _option_present(raw_argv, "--vp-marker-style"):
        args.vp_marker_style = "plain"
    if not _option_present(raw_argv, "--visualization-limit"):
        args.visualization_limit = 5
    if not _option_present(raw_argv, "--max-new-tokens"):
        args.max_new_tokens = 64


def _option_present(raw_argv: Sequence[str], *names: str) -> bool:
    return any(
        arg == name or arg.startswith(f"{name}=")
        for arg in raw_argv
        for name in names
    )


def _build_visualize_cmd(
    *,
    model_path: str,
    manifest_path: str,
    output_dir: str,
    adapter_dir: Optional[str],
    data_key: Optional[str],
    split: str,
    max_samples: int,
    min_query_boxes: Optional[int],
    max_query_boxes: Optional[int],
    device: str,
    torch_dtype: str,
    max_new_tokens: int,
    num_beams: int,
    length_penalty: Optional[float],
    repetition_penalty: Optional[float],
    no_repeat_ngram_size: Optional[int],
    early_stopping: bool,
    visualization_limit: Optional[int],
    structured_vp_mode: str,
    structured_box_format: str,
    structured_marker_style: str,
    structured_filter_policy: str,
    structured_max_boxes_per_label: Optional[int],
    structured_max_total_boxes: Optional[int],
    structured_max_total_boxes_field: Optional[str],
    structured_nms_iou_threshold: Optional[float],
    structured_allowed_labels: Optional[str],
    structured_allowed_labels_field: Optional[str],
    structured_allowed_label_match_mode: str,
    structured_repair_malformed_tail: bool,
    decoder_prefix: Optional[str],
    stop_after_vp_max_total_boxes: bool,
    continue_underfilled_vp_boxes: bool,
    vp_continuation_max_rounds: int,
    vp_continuation_max_new_tokens: int,
    vp_continuation_min_missing_boxes: int,
) -> List[str]:
    cmd = [
        sys.executable,
        "scripts/infer/visualize_florence_vp_adapter.py",
        "--model-path",
        model_path,
        "--manifest-path",
        manifest_path,
        "--split",
        split,
        "--output-dir",
        output_dir,
        "--max-samples",
        str(max_samples),
        "--device",
        device,
        "--torch-dtype",
        torch_dtype,
        "--max-new-tokens",
        str(max_new_tokens),
        "--num-beams",
        str(num_beams),
        "--structured-vp-mode",
        structured_vp_mode,
        "--structured-vp-box-format",
        structured_box_format,
        "--structured-vp-marker-style",
        structured_marker_style,
        "--structured-vp-filter-policy",
        structured_filter_policy,
    ]
    if adapter_dir:
        cmd.extend(["--adapter-dir", adapter_dir])
    if data_key:
        cmd.extend(["--data-key", data_key])
    if min_query_boxes is not None:
        cmd.extend(["--min-query-boxes", str(min_query_boxes)])
    if max_query_boxes is not None:
        cmd.extend(["--max-query-boxes", str(max_query_boxes)])
    if length_penalty is not None:
        cmd.extend(["--length-penalty", str(length_penalty)])
    if repetition_penalty is not None:
        cmd.extend(["--repetition-penalty", str(repetition_penalty)])
    if no_repeat_ngram_size is not None:
        cmd.extend(["--no-repeat-ngram-size", str(no_repeat_ngram_size)])
    if early_stopping:
        cmd.append("--early-stopping")
    if visualization_limit is not None:
        cmd.extend(["--visualization-limit", str(visualization_limit)])
    if structured_max_boxes_per_label is not None:
        cmd.extend(["--structured-vp-max-boxes-per-label", str(structured_max_boxes_per_label)])
    if structured_max_total_boxes is not None:
        cmd.extend(["--structured-vp-max-total-boxes", str(structured_max_total_boxes)])
    if structured_max_total_boxes_field:
        cmd.extend(["--structured-vp-max-total-boxes-field", structured_max_total_boxes_field])
    if structured_nms_iou_threshold is not None:
        cmd.extend(["--structured-vp-nms-iou-threshold", str(structured_nms_iou_threshold)])
    if structured_allowed_labels:
        cmd.extend(["--structured-vp-allowed-labels", structured_allowed_labels])
    if structured_allowed_labels_field:
        cmd.extend(["--structured-vp-allowed-labels-field", structured_allowed_labels_field])
    if structured_allowed_label_match_mode != "strict":
        cmd.extend(["--structured-vp-allowed-label-match-mode", structured_allowed_label_match_mode])
    if structured_repair_malformed_tail:
        cmd.append("--structured-vp-repair-malformed-tail")
    if decoder_prefix:
        cmd.extend(["--decoder-prefix", decoder_prefix])
    if stop_after_vp_max_total_boxes:
        cmd.append("--stop-after-vp-max-total-boxes")
    if continue_underfilled_vp_boxes:
        cmd.append("--continue-underfilled-vp-boxes")
        cmd.extend(["--vp-continuation-max-rounds", str(vp_continuation_max_rounds)])
        cmd.extend(["--vp-continuation-max-new-tokens", str(vp_continuation_max_new_tokens)])
        cmd.extend(["--vp-continuation-min-missing-boxes", str(vp_continuation_min_missing_boxes)])
    return cmd


def _build_quality_cmd(
    *,
    summary_path: str,
    output_dir: str,
    structured_box_format: str,
    structured_marker_style: str,
    structured_filter_policy: str,
    structured_max_boxes_per_label: Optional[int],
    structured_max_total_boxes: Optional[int],
    structured_max_total_boxes_field: Optional[str],
    structured_nms_iou_threshold: Optional[float],
    structured_allowed_labels: Optional[str],
    structured_allowed_labels_field: Optional[str],
    structured_allowed_label_match_mode: str,
    structured_repair_malformed_tail: bool,
    label_match_mode: str,
    iou_threshold: float,
    max_bad_cases: int,
) -> List[str]:
    cmd = [
        sys.executable,
        "scripts/experiments/evaluate_vp_detection_quality.py",
        "--summary",
        summary_path,
        "--output-dir",
        output_dir,
        "--structured-vp-box-format",
        structured_box_format,
        "--structured-vp-marker-style",
        structured_marker_style,
        "--structured-vp-filter-policy",
        structured_filter_policy,
        "--iou-threshold",
        str(iou_threshold),
        "--max-bad-cases",
        str(max_bad_cases),
    ]
    if structured_max_boxes_per_label is not None:
        cmd.extend(["--structured-vp-max-boxes-per-label", str(structured_max_boxes_per_label)])
    if structured_max_total_boxes is not None:
        cmd.extend(["--structured-vp-max-total-boxes", str(structured_max_total_boxes)])
    if structured_max_total_boxes_field:
        cmd.extend(["--structured-vp-max-total-boxes-field", structured_max_total_boxes_field])
    if structured_nms_iou_threshold is not None:
        cmd.extend(["--structured-vp-nms-iou-threshold", str(structured_nms_iou_threshold)])
    if structured_allowed_labels:
        cmd.extend(["--structured-vp-allowed-labels", structured_allowed_labels])
    if structured_allowed_labels_field:
        cmd.extend(["--structured-vp-allowed-labels-field", structured_allowed_labels_field])
    if structured_allowed_label_match_mode != "strict":
        cmd.extend(["--structured-vp-allowed-label-match-mode", structured_allowed_label_match_mode])
    if structured_repair_malformed_tail:
        cmd.append("--structured-vp-repair-malformed-tail")
    if label_match_mode != "strict":
        cmd.extend(["--vp-label-match-mode", label_match_mode])
    return cmd


def _build_policy_comparison_cmd(
    *,
    reports: Sequence[str],
    output_dir: str,
    focus_bucket: Optional[str],
) -> List[str]:
    cmd = [
        sys.executable,
        "scripts/experiments/compare_vp_quality_policies.py",
        "--output-dir",
        output_dir,
    ]
    if focus_bucket:
        cmd.extend(["--focus-bucket", focus_bucket])
    for report in reports:
        cmd.extend(["--report", report])
    return cmd


def _build_record_comparison_cmd(
    *,
    candidate_report: str,
    baseline_report: str,
    output_dir: str,
    candidate_name: str,
    baseline_name: str,
    focus_bucket: Optional[str],
) -> List[str]:
    cmd = [
        sys.executable,
        "scripts/experiments/compare_vp_quality_records.py",
        "--candidate-report",
        candidate_report,
        "--baseline-report",
        baseline_report,
        "--candidate-name",
        candidate_name,
        "--baseline-name",
        baseline_name,
        "--output-dir",
        output_dir,
    ]
    if focus_bucket:
        cmd.extend(["--focus-bucket", focus_bucket])
    return cmd


def _build_target_count_gap_cmd(
    *,
    report: str,
    output_dir: str,
    focus_bucket: Optional[str],
    max_rows: int,
) -> List[str]:
    cmd = [
        sys.executable,
        "scripts/experiments/analyze_vp_target_count_gap.py",
        "--report",
        report,
        "--output-dir",
        output_dir,
        "--max-rows",
        str(max_rows),
    ]
    if focus_bucket:
        cmd.extend(["--focus-bucket", focus_bucket])
    return cmd


def _build_distillation_mix_cmd(
    *,
    base_inputs: Sequence[str],
    distillation_inputs: Sequence[str],
    output: str,
    summary_output: str,
    markdown_output: str,
    base_repeat: int,
    distillation_repeat: int,
    distillation_repeat_order: str,
    max_base_rows: Optional[int],
    max_distillation_rows: Optional[int],
    distillation_min_delta_tp: Optional[int],
    distillation_target_mode: Optional[str],
    replace_base_on_distillation_key: bool,
    placement: str,
    shuffle: bool,
    seed: int,
) -> List[str]:
    cmd = [
        sys.executable,
        "scripts/data-conversion/build_vp_distillation_mix.py",
        "--output",
        output,
        "--summary-output",
        summary_output,
        "--markdown-output",
        markdown_output,
        "--base-repeat",
        str(base_repeat),
        "--distillation-repeat",
        str(distillation_repeat),
        "--distillation-repeat-order",
        distillation_repeat_order,
        "--placement",
        placement,
    ]
    for path in base_inputs:
        cmd.extend(["--base-input", path])
    for path in distillation_inputs:
        cmd.extend(["--distillation-input", path])
    if max_base_rows is not None:
        cmd.extend(["--max-base-rows", str(max_base_rows)])
    if max_distillation_rows is not None:
        cmd.extend(["--max-distillation-rows", str(max_distillation_rows)])
    if distillation_min_delta_tp is not None:
        cmd.extend(["--distillation-min-delta-tp", str(distillation_min_delta_tp)])
    if distillation_target_mode:
        cmd.extend(["--distillation-target-mode", distillation_target_mode])
    if replace_base_on_distillation_key:
        cmd.append("--replace-base-on-distillation-key")
    if shuffle:
        cmd.append("--shuffle")
        cmd.extend(["--seed", str(seed)])
    return cmd


def _build_report_card_cmd(
    *,
    quality_report: str,
    output_dir: str,
    policy_sweep: Optional[str],
    target_count_gap: Optional[str],
    focus_bucket: Optional[str],
    max_gap_rows: int,
    min_samples: int,
    min_precision: float,
    min_recall: float,
    min_f1: float,
    max_undergeneration_ratio: float,
    max_overgeneration_ratio: float,
    max_repair_record_ratio: float,
    min_raw_vp_format_ratio: float,
    max_structured_decoder_ratio: float,
    min_policy_confidence: str,
    high_recoverable_fn_ratio: float,
) -> List[str]:
    cmd = [
        sys.executable,
        "scripts/experiments/build_vp_report_card.py",
        "--quality-report",
        quality_report,
        "--output-dir",
        output_dir,
        "--max-gap-rows",
        str(max_gap_rows),
        "--min-samples",
        str(min_samples),
        "--min-precision",
        str(min_precision),
        "--min-recall",
        str(min_recall),
        "--min-f1",
        str(min_f1),
        "--max-undergeneration-ratio",
        str(max_undergeneration_ratio),
        "--max-overgeneration-ratio",
        str(max_overgeneration_ratio),
        "--max-repair-record-ratio",
        str(max_repair_record_ratio),
        "--min-raw-vp-format-ratio",
        str(min_raw_vp_format_ratio),
        "--max-structured-decoder-ratio",
        str(max_structured_decoder_ratio),
        "--min-policy-confidence",
        min_policy_confidence,
        "--high-recoverable-fn-ratio",
        str(high_recoverable_fn_ratio),
    ]
    if policy_sweep:
        cmd.extend(["--policy-sweep", policy_sweep])
    if target_count_gap:
        cmd.extend(["--target-count-gap", target_count_gap])
    if focus_bucket:
        cmd.extend(["--focus-bucket", focus_bucket])
    return cmd


def _build_policy_sweep_cmd(
    *,
    summary_path: str,
    output_dir: str,
    structured_box_format: str,
    structured_marker_style: str,
    structured_filter_policy: str,
    structured_max_boxes_per_label: Optional[int],
    structured_max_total_boxes: Optional[int],
    structured_max_total_boxes_field: Optional[str],
    structured_nms_iou_threshold: Optional[float],
    structured_allowed_labels: Optional[str],
    structured_allowed_labels_field: Optional[str],
    structured_allowed_label_match_mode: str,
    structured_repair_malformed_tail: bool,
    label_match_mode: str,
    policy_configs: Sequence[str],
    include_target_label_oracle: bool,
    include_phrase_label_policy: bool,
    include_repair_policy: bool,
    focus_bucket: Optional[str],
    iou_threshold: float,
    max_bad_cases: int,
) -> List[str]:
    cmd = [
        sys.executable,
        "scripts/experiments/sweep_vp_quality_policies.py",
        "--summary",
        summary_path,
        "--output-dir",
        output_dir,
        "--structured-vp-box-format",
        structured_box_format,
        "--structured-vp-marker-style",
        structured_marker_style,
        "--structured-vp-filter-policy",
        structured_filter_policy,
        "--iou-threshold",
        str(iou_threshold),
        "--max-bad-cases",
        str(max_bad_cases),
    ]
    if structured_max_boxes_per_label is not None:
        cmd.extend(["--structured-vp-max-boxes-per-label", str(structured_max_boxes_per_label)])
    if structured_max_total_boxes is not None:
        cmd.extend(["--structured-vp-max-total-boxes", str(structured_max_total_boxes)])
    if structured_max_total_boxes_field:
        cmd.extend(["--structured-vp-max-total-boxes-field", structured_max_total_boxes_field])
    if structured_nms_iou_threshold is not None:
        cmd.extend(["--structured-vp-nms-iou-threshold", str(structured_nms_iou_threshold)])
    if structured_allowed_labels:
        cmd.extend(["--structured-vp-allowed-labels", structured_allowed_labels])
    if structured_allowed_labels_field:
        cmd.extend(["--structured-vp-allowed-labels-field", structured_allowed_labels_field])
    if structured_allowed_label_match_mode != "strict":
        cmd.extend(["--structured-vp-allowed-label-match-mode", structured_allowed_label_match_mode])
    if structured_repair_malformed_tail:
        cmd.append("--structured-vp-repair-malformed-tail")
    if label_match_mode != "strict":
        cmd.extend(["--vp-label-match-mode", label_match_mode])
    for policy_config in policy_configs:
        cmd.extend(["--policy-config", policy_config])
    if include_target_label_oracle:
        cmd.append("--include-target-label-oracle")
    if include_phrase_label_policy:
        cmd.append("--include-phrase-label-policy")
    if include_repair_policy:
        cmd.append("--include-repair-policy")
    if focus_bucket:
        cmd.extend(["--focus-bucket", focus_bucket])
    return cmd


def _default_jsonl_summary_path(output_path: Path) -> Path:
    suffix = "".join(output_path.suffixes)
    name = output_path.name[: -len(suffix)] if suffix else output_path.name
    return output_path.with_name(f"{name}_summary.json")


def _read_adapter_dir_from_summary(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    adapter_dir = data.get("adapter_dir")
    return str(adapter_dir) if adapter_dir else None


def _write_summary(output_dir: Path, summary: Dict[str, Any]) -> None:
    summary_path = output_dir / "experiment_summary.json"
    summary["experiment_summary_path"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def _cleanup_generated_adapter(args: argparse.Namespace, plan: Dict[str, Any]) -> Dict[str, Any]:
    requested = bool(args.cleanup_adapter_after_audit)
    if not requested:
        return {"requested": False, "deleted": False, "reason": "not_requested"}
    if not args.run_training:
        return {"requested": True, "deleted": False, "reason": "not_generated_by_this_run"}

    adapter_dir = plan.get("adapter_dir")
    if not adapter_dir:
        return {"requested": True, "deleted": False, "reason": "adapter_dir_missing"}

    adapter_path = Path(adapter_dir).expanduser()
    expected_path = Path(plan["output_dir"]).expanduser() / "training" / "adapter"
    if adapter_path.resolve() != expected_path.resolve():
        return {
            "requested": True,
            "deleted": False,
            "reason": "adapter_path_not_experiment_owned",
            "adapter_dir": str(adapter_path),
        }
    if not adapter_path.exists():
        return {
            "requested": True,
            "deleted": False,
            "reason": "adapter_dir_not_found",
            "adapter_dir": str(adapter_path),
        }

    shutil.rmtree(adapter_path)
    return {
        "requested": True,
        "deleted": True,
        "reason": "deleted",
        "adapter_dir": str(adapter_path),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    summary = run_experiment(parse_args(argv))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary.get("ok", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
