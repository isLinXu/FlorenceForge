"""Audit helpers for Florence-VP training completeness."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


@dataclass(frozen=True)
class VPTrainingAuditThresholds:
    """Thresholds used by the Florence-VP training audit."""

    raw_vp_format_threshold: float = 0.95
    structured_vp_format_threshold: float = 0.95
    decoder_dependency_threshold: float = 0.50
    min_train_rows: int = 1
    min_val_rows: int = 1
    min_training_steps: int = 1
    min_delta_norm: float = 0.0
    min_inference_samples: int = 1
    max_box_count_overgeneration_ratio: float = 0.10


def load_json(path: str | Path) -> Dict[str, Any]:
    """Load a JSON object from disk."""

    value = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object at {path}")
    return value


def build_vp_training_audit(
    *,
    training_summary: Mapping[str, Any],
    inference_summaries: Sequence[Mapping[str, Any]],
    baseline_summaries: Sequence[Mapping[str, Any]] = (),
    thresholds: Optional[VPTrainingAuditThresholds] = None,
) -> Dict[str, Any]:
    """Build a structured audit report for a Florence-VP training run."""

    thresholds = thresholds or VPTrainingAuditThresholds()
    inference_metrics = [_summarize_inference(summary) for summary in inference_summaries]
    baseline_metrics = [_summarize_inference(summary) for summary in baseline_summaries]

    aggregate_inference = _aggregate_inference_metrics(inference_metrics)
    aggregate_baseline = _aggregate_inference_metrics(baseline_metrics)
    data_summary = _summarize_data(training_summary.get("data", {}))
    train_summary = _summarize_training(training_summary)
    gates = _build_gates(
        data_summary=data_summary,
        train_summary=train_summary,
        aggregate_inference=aggregate_inference,
        baseline_metrics=baseline_metrics,
        thresholds=thresholds,
    )
    status = _classify_status(gates)
    recommendations = _build_recommendations(gates, train_summary, aggregate_inference)

    return {
        "status": status,
        "thresholds": thresholds.__dict__,
        "data": data_summary,
        "training": train_summary,
        "inference": {
            "runs": inference_metrics,
            "aggregate": aggregate_inference,
        },
        "baseline": {
            "runs": baseline_metrics,
            "aggregate": aggregate_baseline,
            "deltas": _compute_baseline_deltas(aggregate_inference, aggregate_baseline),
        },
        "gates": gates,
        "recommendations": recommendations,
    }


def render_vp_training_audit_markdown(audit: Mapping[str, Any]) -> str:
    """Render a compact Markdown report for a VP training audit."""

    gates = audit.get("gates", {})
    training = audit.get("training", {})
    inference = audit.get("inference", {}).get("aggregate", {})
    baseline = audit.get("baseline", {})
    recommendations = audit.get("recommendations", [])

    lines = [
        "# Florence-VP Training Audit",
        "",
        f"- Status: `{audit.get('status', 'unknown')}`",
        f"- Training mode: `{training.get('training_mode', 'unknown')}`",
        f"- LoRA task type: `{training.get('lora_task_type', 'unknown')}`",
        f"- VP box format: `{training.get('vp_box_format', 'unknown')}`",
        f"- VP marker style: `{training.get('vp_marker_style', 'unknown')}`",
        f"- Steps executed: `{training.get('steps_executed', 0)}`",
        f"- Final loss: `{training.get('final_loss', 'n/a')}`",
        "",
        "## Inference Metrics",
        "",
        f"- raw `vp_format_valid_ratio`: `{inference.get('vp_format_valid_ratio', 0.0):.4f}`",
        f"- `structured_vp_format_valid_ratio`: `{inference.get('structured_vp_format_valid_ratio', 0.0):.4f}`",
        f"- `structured_vp_decoder_ratio`: `{inference.get('structured_vp_decoder_ratio', 0.0):.4f}`",
        f"- `avg_pred_boxes`: `{inference.get('avg_pred_boxes', 0.0):.4f}`",
        f"- `avg_gt_boxes`: `{inference.get('avg_gt_boxes', 0.0):.4f}`",
        f"- `box_count_overgeneration_ratio`: `{inference.get('box_count_overgeneration_ratio', 0.0):.4f}`",
        f"- samples: `{inference.get('num_samples', 0)}`",
        "",
        "## Gates",
        "",
    ]

    for name, gate in gates.items():
        icon = "PASS" if gate.get("passed") else "FAIL"
        lines.append(f"- {icon} `{name}`: {gate.get('message', '')}")

    if baseline.get("runs"):
        lines.extend(["", "## Baseline Deltas", ""])
        for key, value in baseline.get("deltas", {}).items():
            lines.append(f"- `{key}`: `{value:.4f}`")

    if recommendations:
        lines.extend(["", "## Recommendations", ""])
        for item in recommendations:
            lines.append(f"- {item}")

    return "\n".join(lines) + "\n"


def _summarize_data(data: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "train_od_rows": int(data.get("train_od_rows", 0) or 0),
        "val_od_rows": int(data.get("val_od_rows", 0) or 0),
        "train_count_rows": int(data.get("train_count_rows", 0) or 0),
        "val_count_rows": int(data.get("val_count_rows", 0) or 0),
        "train_grounding_rows": int(data.get("train_grounding_rows", 0) or 0),
        "train_grounding_effective_rows": int(data.get("train_grounding_effective_rows", 0) or 0),
        "val_grounding_rows": int(data.get("val_grounding_rows", 0) or 0),
        "skip_od_training_data": bool(data.get("skip_od_training_data", False)),
        "manifest_path": data.get("manifest_path"),
        "train_od_path": data.get("train_od_path"),
        "val_od_path": data.get("val_od_path"),
    }


def _summarize_training(summary: Mapping[str, Any]) -> Dict[str, Any]:
    modules_to_save = _as_string_list(summary.get("lora_modules_to_save"))
    return {
        "ok": bool(summary.get("ok", False)),
        "training_mode": summary.get("training_mode"),
        "lora_task_type": summary.get("lora_task_type"),
        "vp_box_format": summary.get("vp_box_format"),
        "vp_marker_style": summary.get("vp_marker_style", "special"),
        "max_steps": int(summary.get("max_steps", 0) or 0),
        "steps_executed": int(summary.get("steps_executed", 0) or 0),
        "final_loss": _optional_float(summary.get("final_loss")),
        "trainable_param_delta_norm": _optional_float(summary.get("trainable_param_delta_norm")),
        "trainable_parameter_ratio": _optional_float(summary.get("trainable_parameter_ratio")),
        "lora_modules_to_save": modules_to_save,
        "vp_head_trainable": _has_vp_head_modules(modules_to_save),
        "adapter_dir": summary.get("adapter_dir"),
        "summary_path": summary.get("summary_path"),
    }


def _summarize_inference(summary: Mapping[str, Any]) -> Dict[str, Any]:
    box_metrics = _summarize_box_counts(summary)
    return {
        "summary_path": summary.get("summary_path"),
        "adapter_dir": summary.get("adapter_dir"),
        "output_dir": summary.get("output_dir"),
        "num_samples": int(summary.get("num_samples", 0) or 0),
        "vp_format_valid_ratio": float(summary.get("vp_format_valid_ratio", 0.0) or 0.0),
        "structured_vp_format_valid_ratio": float(
            summary.get("structured_vp_format_valid_ratio", 0.0) or 0.0
        ),
        "structured_vp_decoder_ratio": float(summary.get("structured_vp_decoder_ratio", 0.0) or 0.0),
        "native_fallback_ratio": float(summary.get("native_fallback_ratio", 0.0) or 0.0),
        "structured_source_counts": dict(summary.get("structured_source_counts", {}) or {}),
        **box_metrics,
    }


def _summarize_box_counts(summary: Mapping[str, Any]) -> Dict[str, float]:
    if "avg_pred_boxes" in summary or "box_count_overgeneration_ratio" in summary:
        return {
            "avg_pred_boxes": float(summary.get("avg_pred_boxes", 0.0) or 0.0),
            "avg_gt_boxes": float(summary.get("avg_gt_boxes", 0.0) or 0.0),
            "box_count_overgeneration_ratio": float(
                summary.get("box_count_overgeneration_ratio", 0.0) or 0.0
            ),
        }

    records = summary.get("records", [])
    if not isinstance(records, Sequence) or not records:
        return {
            "avg_pred_boxes": 0.0,
            "avg_gt_boxes": 0.0,
            "box_count_overgeneration_ratio": 0.0,
        }

    pred_counts = [int(record.get("pred_box_count", 0) or 0) for record in records if isinstance(record, Mapping)]
    gt_counts = [int(record.get("gt_box_count", 0) or 0) for record in records if isinstance(record, Mapping)]
    paired = list(zip(pred_counts, gt_counts))
    return {
        "avg_pred_boxes": _mean(float(value) for value in pred_counts),
        "avg_gt_boxes": _mean(float(value) for value in gt_counts),
        "box_count_overgeneration_ratio": (
            sum(1 for pred_count, gt_count in paired if pred_count > gt_count) / len(paired)
            if paired else 0.0
        ),
    }


def _aggregate_inference_metrics(runs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not runs:
        return {
            "num_runs": 0,
            "num_samples": 0,
            "vp_format_valid_ratio": 0.0,
            "structured_vp_format_valid_ratio": 0.0,
            "structured_vp_decoder_ratio": 0.0,
            "native_fallback_ratio": 0.0,
        }

    total_samples = sum(int(run.get("num_samples", 0) or 0) for run in runs)

    def weighted_mean(key: str) -> float:
        if total_samples <= 0:
            return _mean(float(run.get(key, 0.0) or 0.0) for run in runs)
        return sum(
            float(run.get(key, 0.0) or 0.0) * int(run.get("num_samples", 0) or 0)
            for run in runs
        ) / total_samples

    source_counts: Dict[str, int] = {}
    for run in runs:
        for source, count in dict(run.get("structured_source_counts", {}) or {}).items():
            source_counts[str(source)] = source_counts.get(str(source), 0) + int(count)

    return {
        "num_runs": len(runs),
        "num_samples": total_samples,
        "vp_format_valid_ratio": weighted_mean("vp_format_valid_ratio"),
        "structured_vp_format_valid_ratio": weighted_mean("structured_vp_format_valid_ratio"),
        "structured_vp_decoder_ratio": weighted_mean("structured_vp_decoder_ratio"),
        "native_fallback_ratio": weighted_mean("native_fallback_ratio"),
        "avg_pred_boxes": weighted_mean("avg_pred_boxes"),
        "avg_gt_boxes": weighted_mean("avg_gt_boxes"),
        "box_count_overgeneration_ratio": weighted_mean("box_count_overgeneration_ratio"),
        "structured_source_counts": source_counts,
    }


def _build_gates(
    *,
    data_summary: Mapping[str, Any],
    train_summary: Mapping[str, Any],
    aggregate_inference: Mapping[str, Any],
    baseline_metrics: Sequence[Mapping[str, Any]],
    thresholds: VPTrainingAuditThresholds,
) -> Dict[str, Dict[str, Any]]:
    raw_ratio = float(aggregate_inference.get("vp_format_valid_ratio", 0.0) or 0.0)
    structured_ratio = float(aggregate_inference.get("structured_vp_format_valid_ratio", 0.0) or 0.0)
    decoder_ratio = float(aggregate_inference.get("structured_vp_decoder_ratio", 0.0) or 0.0)
    overgeneration_ratio = float(aggregate_inference.get("box_count_overgeneration_ratio", 0.0) or 0.0)
    train_delta = float(train_summary.get("trainable_param_delta_norm", 0.0) or 0.0)
    train_rows = max(
        int(data_summary.get("train_od_rows", 0) or 0),
        int(data_summary.get("train_grounding_effective_rows", 0) or 0),
        int(data_summary.get("train_grounding_rows", 0) or 0),
    )
    val_rows = max(
        int(data_summary.get("val_od_rows", 0) or 0),
        int(data_summary.get("val_grounding_rows", 0) or 0),
    )

    return {
        "data_ready": _gate(
            train_rows >= thresholds.min_train_rows
            and val_rows >= thresholds.min_val_rows,
            (
                f"train_od_rows={data_summary.get('train_od_rows', 0)}, "
                f"val_od_rows={data_summary.get('val_od_rows', 0)}, "
                f"train_grounding_effective_rows={data_summary.get('train_grounding_effective_rows', 0)}, "
                f"val_grounding_rows={data_summary.get('val_grounding_rows', 0)}"
            ),
        ),
        "training_smoke_passed": _gate(
            bool(train_summary.get("ok"))
            and int(train_summary.get("steps_executed", 0) or 0) >= thresholds.min_training_steps
            and train_delta > thresholds.min_delta_norm,
            (
                f"ok={train_summary.get('ok')}, steps={train_summary.get('steps_executed')}, "
                f"delta_norm={train_delta:.6f}"
            ),
        ),
        "loc_token_format": _gate(
            train_summary.get("vp_box_format") == "loc_tokens",
            f"vp_box_format={train_summary.get('vp_box_format')}",
        ),
        "vp_head_trainable": _gate(
            bool(train_summary.get("vp_head_trainable")),
            f"lora_modules_to_save={train_summary.get('lora_modules_to_save')}",
        ),
        "inference_available": _gate(
            int(aggregate_inference.get("num_samples", 0) or 0) >= thresholds.min_inference_samples,
            f"num_samples={aggregate_inference.get('num_samples', 0)}",
        ),
        "raw_vp_internalized": _gate(
            raw_ratio >= thresholds.raw_vp_format_threshold,
            f"vp_format_valid_ratio={raw_ratio:.4f}",
        ),
        "structured_vp_usable": _gate(
            structured_ratio >= thresholds.structured_vp_format_threshold,
            f"structured_vp_format_valid_ratio={structured_ratio:.4f}",
        ),
        "decoder_dependency_low": _gate(
            decoder_ratio < thresholds.decoder_dependency_threshold,
            f"structured_vp_decoder_ratio={decoder_ratio:.4f}",
        ),
        "baseline_present": _gate(
            bool(baseline_metrics),
            f"baseline_runs={len(baseline_metrics)}",
        ),
        "box_count_not_overgenerated": _gate(
            overgeneration_ratio <= thresholds.max_box_count_overgeneration_ratio,
            f"box_count_overgeneration_ratio={overgeneration_ratio:.4f}",
        ),
    }


def _gate(passed: bool, message: str) -> Dict[str, Any]:
    return {"passed": bool(passed), "message": message}


def _classify_status(gates: Mapping[str, Mapping[str, Any]]) -> str:
    if not gates.get("training_smoke_passed", {}).get("passed"):
        return "blocked_training_smoke"
    if not gates.get("inference_available", {}).get("passed"):
        return "blocked_inference_missing"
    if gates.get("raw_vp_internalized", {}).get("passed") and gates.get("baseline_present", {}).get("passed"):
        return "candidate_training_complete"
    if gates.get("structured_vp_usable", {}).get("passed"):
        return "engineering_mvp_ready_needs_wrapper_training"
    return "incomplete_needs_training"


def _build_recommendations(
    gates: Mapping[str, Mapping[str, Any]],
    train_summary: Mapping[str, Any],
    aggregate_inference: Mapping[str, Any],
) -> List[str]:
    recommendations: List[str] = []
    if not gates["vp_head_trainable"]["passed"]:
        recommendations.append("Enable `--train-vp-head` or save/train `lm_head` and shared embeddings for wrapper tokens.")
    if not gates["loc_token_format"]["passed"]:
        recommendations.append("Use `--vp-box-format loc_tokens` as the default Florence-VP format.")
    if gates["structured_vp_usable"]["passed"] and not gates["raw_vp_internalized"]["passed"]:
        recommendations.append("Scale SFT focused on VP wrapper internalization; structured decoding is usable but still doing the wrapping.")
    if not gates["baseline_present"]["passed"]:
        recommendations.append("Add a base Florence inference summary so adapter-vs-baseline deltas can be audited.")
    if gates["decoder_dependency_low"]["passed"] is False:
        recommendations.append("Track decoder dependency as a risk: high `structured_vp_decoder_ratio` means the model has not learned the wrapper yet.")
    if int(train_summary.get("steps_executed", 0) or 0) < 100:
        recommendations.append("Run a longer LoRA schedule before claiming training completeness; current evidence is smoke-scale.")
    if int(aggregate_inference.get("num_samples", 0) or 0) < 50:
        recommendations.append("Increase inference audit samples; current visualization evidence is too small for a training claim.")
    if float(aggregate_inference.get("box_count_overgeneration_ratio", 0.0) or 0.0) > 0:
        recommendations.append("Inspect overgenerated native loc sequences; structured VP can wrap them, but box count drift still needs constrained decoding or post-filtering.")
    return recommendations


def _compute_baseline_deltas(
    aggregate_inference: Mapping[str, Any],
    aggregate_baseline: Mapping[str, Any],
) -> Dict[str, float]:
    if not aggregate_baseline.get("num_runs"):
        return {}
    keys = (
        "vp_format_valid_ratio",
        "structured_vp_format_valid_ratio",
        "structured_vp_decoder_ratio",
        "native_fallback_ratio",
        "avg_pred_boxes",
        "avg_gt_boxes",
        "box_count_overgeneration_ratio",
    )
    return {
        key: float(aggregate_inference.get(key, 0.0) or 0.0)
        - float(aggregate_baseline.get(key, 0.0) or 0.0)
        for key in keys
    }


def _has_vp_head_modules(modules: Sequence[str]) -> bool:
    normalized = {module.lower() for module in modules}
    has_lm_head = "lm_head" in normalized
    has_shared = any("shared" in module or "embed" in module for module in normalized)
    return has_lm_head and has_shared


def _as_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return [str(item) for item in value]
    return [str(value)]


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0
