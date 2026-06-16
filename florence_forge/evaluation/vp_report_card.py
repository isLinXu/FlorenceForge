"""Report-card diagnostics for Florence-VP experiment readiness."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence


_STRUCTURED_SOURCE_NAMES = {
    "florence_native",
    "unparsed",
    "visual_primitive",
    "visual_primitive_repaired_tail",
}


@dataclass(frozen=True)
class VPReportCardThresholds:
    """Thresholds used to judge whether a VP run is ready to promote."""

    min_samples: int = 10
    min_precision: float = 0.80
    min_recall: float = 0.70
    min_f1: float = 0.75
    max_undergeneration_ratio: float = 0.35
    max_overgeneration_ratio: float = 0.25
    max_repair_record_ratio: float = 0.25
    min_raw_vp_format_ratio: float = 0.95
    max_structured_decoder_ratio: float = 0.50
    min_policy_confidence: str = "moderate"
    high_recoverable_fn_ratio: float = 0.40


def build_vp_report_card(
    quality_report: Mapping[str, Any],
    *,
    policy_sweep: Optional[Mapping[str, Any]] = None,
    target_count_gap: Optional[Mapping[str, Any]] = None,
    thresholds: Optional[VPReportCardThresholds] = None,
    focus_bucket: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a compact readiness card from saved VP diagnostics."""

    thresholds = thresholds or VPReportCardThresholds()
    quality_summary = _quality_summary(quality_report, focus_bucket=focus_bucket)
    policy_summary = _policy_summary(policy_sweep)
    gap_summary = _target_count_gap_summary(target_count_gap)
    checks = _build_checks(
        quality_summary,
        policy_summary=policy_summary,
        gap_summary=gap_summary,
        thresholds=thresholds,
    )
    status = _overall_status(checks)
    next_actions = _next_actions(
        quality_summary,
        policy_summary=policy_summary,
        gap_summary=gap_summary,
        checks=checks,
        thresholds=thresholds,
    )
    return {
        "status": status,
        "readiness": _readiness_label(status),
        "focus_bucket": focus_bucket,
        "thresholds": _thresholds_to_dict(thresholds),
        "quality_summary": quality_summary,
        "policy_summary": policy_summary,
        "target_count_gap_summary": gap_summary,
        "checks": checks,
        "next_actions": next_actions,
    }


def render_vp_report_card_markdown(card: Mapping[str, Any]) -> str:
    """Render a Florence-VP report card as Markdown."""

    quality = dict(card.get("quality_summary", {}) or {})
    policy = dict(card.get("policy_summary", {}) or {})
    gap = dict(card.get("target_count_gap_summary", {}) or {})
    lines = [
        "# Florence-VP Report Card",
        "",
        f"- Status: `{card.get('status', 'unknown')}`",
        f"- Readiness: `{card.get('readiness', 'unknown')}`",
        f"- Focus bucket: `{card.get('focus_bucket') or 'all'}`",
        f"- Samples: `{int(quality.get('num_samples', 0) or 0)}`",
        f"- Precision / recall / F1: "
        f"`{float(quality.get('precision', 0.0) or 0.0):.4f}` / "
        f"`{float(quality.get('recall', 0.0) or 0.0):.4f}` / "
        f"`{float(quality.get('f1', 0.0) or 0.0):.4f}`",
        f"- Avg pred / GT boxes: "
        f"`{float(quality.get('avg_pred_boxes', 0.0) or 0.0):.2f}` / "
        f"`{float(quality.get('avg_gt_boxes', 0.0) or 0.0):.2f}`",
        f"- Undergeneration / overgeneration: "
        f"`{float(quality.get('undergeneration_ratio', 0.0) or 0.0):.4f}` / "
        f"`{float(quality.get('overgeneration_ratio', 0.0) or 0.0):.4f}`",
        f"- Repair record ratio: "
        f"`{float(quality.get('repair_record_ratio', 0.0) or 0.0):.4f}`",
        f"- Raw VP format ratio: "
        f"`{_format_optional_ratio(quality.get('raw_vp_format_ratio'))}`",
        f"- Structured decoder dependency: "
        f"`{_format_optional_ratio(quality.get('structured_decoder_ratio'))}`",
        f"- Unparsed prediction ratio: "
        f"`{_format_optional_ratio(quality.get('unparsed_prediction_ratio'))}`",
    ]

    if policy:
        lines.extend([
            "",
            "## Policy",
            "",
            f"- Recommended policy: `{policy.get('recommended_policy')}`",
            f"- Confidence: `{policy.get('confidence', 'unknown')}`",
            f"- General detection policy: `{policy.get('general_detection_policy')}`",
        ])
        repair_lift = dict(policy.get("repair_lift", {}) or {})
        if repair_lift:
            lines.append(
                f"- Best repair lift: `{repair_lift.get('policy')}` "
                f"(delta F1 `{float(repair_lift.get('delta_f1', 0.0) or 0.0):.4f}`)"
            )

    if gap:
        lines.extend([
            "",
            "## Target-Count Gap",
            "",
            f"- Current recall / F1: "
            f"`{float(gap.get('current_recall', 0.0) or 0.0):.4f}` / "
            f"`{float(gap.get('current_f1', 0.0) or 0.0):.4f}`",
            f"- Oracle recall / F1: "
            f"`{float(gap.get('oracle_recall', 0.0) or 0.0):.4f}` / "
            f"`{float(gap.get('oracle_f1', 0.0) or 0.0):.4f}`",
            f"- Recoverable FN: `{int(gap.get('recoverable_false_negatives', 0) or 0)}` / "
            f"`{int(gap.get('false_negatives', 0) or 0)}` "
            f"(`{float(gap.get('recall_gap_closure_ratio', 0.0) or 0.0):.4f}`)",
        ])

    lines.extend([
        "",
        "## Checks",
        "",
        "| check | status | value | threshold | reason |",
        "| --- | --- | ---: | --- | --- |",
    ])
    for check in list(card.get("checks", []) or []):
        lines.append(
            f"| `{check.get('name')}` "
            f"| `{check.get('status')}` "
            f"| {check.get('value')} "
            f"| {check.get('threshold')} "
            f"| {check.get('reason')} |"
        )

    actions = list(card.get("next_actions", []) or [])
    if actions:
        lines.extend(["", "## Next Actions", ""])
        for action in actions:
            lines.append(f"- {action}")

    return "\n".join(lines) + "\n"


def _quality_summary(report: Mapping[str, Any], *, focus_bucket: Optional[str]) -> Dict[str, Any]:
    bucket_metrics = _focus_bucket_metrics(report, focus_bucket)
    source = bucket_metrics or report
    prediction_source_counts = dict(report.get("prediction_source_counts", {}) or {})
    source_diagnostics = _prediction_source_diagnostics(report, prediction_source_counts)
    return {
        "source_report_path": report.get("quality_json_path") or report.get("source_report_path"),
        "num_samples": _int(source.get("num_samples")),
        "precision": _float(source.get("precision")),
        "recall": _float(source.get("recall")),
        "f1": _float(source.get("f1")),
        "mean_matched_iou": _float(report.get("mean_matched_iou")),
        "true_positives": _int(source.get("true_positives")),
        "false_positives": _int(source.get("false_positives")),
        "false_negatives": _int(source.get("false_negatives")),
        "avg_pred_boxes": _float(source.get("avg_pred_boxes")),
        "avg_gt_boxes": _float(source.get("avg_gt_boxes")),
        "box_count_exact_match_ratio": _float(source.get("box_count_exact_match_ratio")),
        "undergeneration_ratio": _float(source.get("box_count_undergeneration_ratio")),
        "overgeneration_ratio": _float(source.get("box_count_overgeneration_ratio")),
        "repair_detection_count": _int(report.get("repaired_tail_detection_count")),
        "repair_record_ratio": _float(report.get("repaired_tail_record_ratio")),
        "avg_repair_detection_count": _float(report.get("avg_repaired_tail_detection_count")),
        "prediction_source_counts": prediction_source_counts,
        **source_diagnostics,
    }


def _prediction_source_diagnostics(
    report: Mapping[str, Any],
    prediction_source_counts: Mapping[str, Any],
) -> Dict[str, Any]:
    prediction_source_ratios = dict(report.get("prediction_source_ratios", {}) or {})
    raw_ratio = _optional_float(report.get("vp_format_valid_ratio"))
    if raw_ratio is None:
        raw_ratio = _source_ratio(
            prediction_source_counts,
            prediction_source_ratios,
            ("visual_primitive",),
        )

    decoder_ratio = _optional_float(report.get("structured_vp_decoder_ratio"))
    if decoder_ratio is None:
        decoder_ratio = _source_ratio(
            prediction_source_counts,
            prediction_source_ratios,
            ("florence_native",),
        )

    repaired_ratio = _source_ratio(
        prediction_source_counts,
        prediction_source_ratios,
        ("visual_primitive_repaired_tail",),
    )
    unparsed_ratio = _source_ratio(
        prediction_source_counts,
        prediction_source_ratios,
        ("unparsed",),
    )
    return {
        "raw_vp_format_ratio": raw_ratio,
        "structured_decoder_ratio": decoder_ratio,
        "repaired_tail_source_ratio": repaired_ratio,
        "unparsed_prediction_ratio": unparsed_ratio,
    }


def _source_ratio(
    source_counts: Mapping[str, Any],
    source_ratios: Mapping[str, Any],
    source_names: Sequence[str],
) -> Optional[float]:
    total = sum(_int(count) for count in source_counts.values())
    if total > 0:
        if not any(source in source_counts for source in _STRUCTURED_SOURCE_NAMES):
            return None
        return sum(_int(source_counts.get(name)) for name in source_names) / total

    ratio_values = [
        _optional_float(source_ratios.get(name))
        for name in source_names
        if name in source_ratios
    ]
    ratio_values = [value for value in ratio_values if value is not None]
    if ratio_values:
        return sum(ratio_values)
    return None


def _policy_summary(policy_sweep: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not policy_sweep:
        return {}
    comparison = dict(policy_sweep.get("comparison", {}) or {})
    recommendation = dict(comparison.get("recommendation", {}) or {})
    rows = list(comparison.get("ranked_rows") or comparison.get("rows") or [])
    top_policies = [
        {
            "rank": _int(row.get("rank")),
            "policy": row.get("policy"),
            "policy_kind": row.get("policy_kind"),
            "precision": _float(row.get("precision")),
            "recall": _float(row.get("recall")),
            "f1": _float(row.get("f1")),
            "constraints": list(row.get("constraints", []) or []),
        }
        for row in rows[:5]
        if isinstance(row, Mapping)
    ]
    repair_lift = _best_repair_lift(rows)
    return {
        "source_sweep_path": policy_sweep.get("sweep_json_path"),
        "recommended_policy": (
            policy_sweep.get("recommended_policy") or recommendation.get("policy")
        ),
        "confidence": recommendation.get("confidence"),
        "general_detection_policy": recommendation.get("general_detection_policy"),
        "reason": recommendation.get("reason"),
        "top_policies": top_policies,
        "repair_lift": repair_lift,
    }


def _target_count_gap_summary(gap: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not gap:
        return {}
    current = dict(gap.get("current", {}) or {})
    oracle = dict(gap.get("oracle_count_fill", {}) or {})
    count_gap = dict(gap.get("count_gap", {}) or {})
    return {
        "source_gap_path": gap.get("target_count_gap_json_path") or gap.get("source_report_path"),
        "num_records": _int(gap.get("num_records")),
        "focus_bucket": gap.get("focus_bucket"),
        "current_precision": _float(current.get("precision")),
        "current_recall": _float(current.get("recall")),
        "current_f1": _float(current.get("f1")),
        "oracle_precision": _float(oracle.get("precision")),
        "oracle_recall": _float(oracle.get("recall")),
        "oracle_f1": _float(oracle.get("f1")),
        "oracle_f1_delta": _float(oracle.get("f1_delta")),
        "recoverable_false_negatives": _int(count_gap.get("recoverable_false_negatives")),
        "unrecoverable_false_negatives": _int(count_gap.get("unrecoverable_false_negatives")),
        "false_negatives": _int(count_gap.get("false_negatives")),
        "recall_gap_closure_ratio": _float(count_gap.get("recall_gap_closure_ratio")),
        "target_box_deficit": _int(count_gap.get("target_box_deficit")),
        "target_box_overage": _int(count_gap.get("target_box_overage")),
        "records_with_deficit": _int(count_gap.get("records_with_deficit")),
    }


def _build_checks(
    quality: Mapping[str, Any],
    *,
    policy_summary: Mapping[str, Any],
    gap_summary: Mapping[str, Any],
    thresholds: VPReportCardThresholds,
) -> List[Dict[str, Any]]:
    checks = [
        _check(
            "sample_size",
            "pass" if _int(quality.get("num_samples")) >= thresholds.min_samples else "warn",
            _int(quality.get("num_samples")),
            f">= {thresholds.min_samples}",
            "sample count supports threshold decisions",
            "sample count is still exploratory",
        ),
        _metric_min_check("precision", quality.get("precision"), thresholds.min_precision),
        _metric_min_check("recall", quality.get("recall"), thresholds.min_recall),
        _metric_min_check("f1", quality.get("f1"), thresholds.min_f1),
        _metric_max_check(
            "undergeneration_ratio",
            quality.get("undergeneration_ratio"),
            thresholds.max_undergeneration_ratio,
        ),
        _metric_max_check(
            "overgeneration_ratio",
            quality.get("overgeneration_ratio"),
            thresholds.max_overgeneration_ratio,
        ),
        _check(
            "repair_dependency",
            (
                "pass"
                if _float(quality.get("repair_record_ratio"))
                <= thresholds.max_repair_record_ratio
                else "warn"
            ),
            _round(_float(quality.get("repair_record_ratio"))),
            f"<= {thresholds.max_repair_record_ratio:.4f}",
            "malformed-tail repair is not dominating results",
            "quality depends on malformed-tail repair for many records",
        ),
        _optional_metric_min_check(
            "raw_vp_internalization",
            quality.get("raw_vp_format_ratio"),
            thresholds.min_raw_vp_format_ratio,
            "model emits VP wrappers without structured decoding",
            "raw VP wrapper format is not internalized yet",
        ),
        _optional_metric_max_check(
            "structured_decoder_dependency",
            quality.get("structured_decoder_ratio"),
            thresholds.max_structured_decoder_ratio,
            "structured decoder is a fallback, not the main path",
            "structured decoder is still doing too much wrapper work",
        ),
    ]
    if policy_summary:
        confidence = str(policy_summary.get("confidence") or "none")
        checks.append(
            _check(
                "policy_confidence",
                (
                    "pass"
                    if _confidence_rank(confidence)
                    >= _confidence_rank(thresholds.min_policy_confidence)
                    else "warn"
                ),
                confidence,
                f">= {thresholds.min_policy_confidence}",
                "policy recommendation has enough support",
                "policy recommendation is still exploratory",
            )
        )
    else:
        checks.append(_skip_check("policy_confidence", "no policy sweep provided"))

    if gap_summary:
        closure = _float(gap_summary.get("recall_gap_closure_ratio"))
        checks.append(
            _check(
                "target_count_gap",
                "warn" if closure >= thresholds.high_recoverable_fn_ratio else "pass",
                _round(closure),
                f"< {thresholds.high_recoverable_fn_ratio:.4f}",
                "missing boxes are not mostly explainable by target-count deficit",
                "target-count/proposal filling could recover many false negatives",
            )
        )
    else:
        checks.append(
            _skip_check("target_count_gap", "no target-count gap analysis provided")
        )
    return checks


def _next_actions(
    quality: Mapping[str, Any],
    *,
    policy_summary: Mapping[str, Any],
    gap_summary: Mapping[str, Any],
    checks: Sequence[Mapping[str, Any]],
    thresholds: VPReportCardThresholds,
) -> List[str]:
    actions: List[str] = []
    failed_or_warned = {
        str(check.get("name")): str(check.get("status")) for check in checks
    }
    if failed_or_warned.get("sample_size") == "warn":
        actions.append(
            "Run the same report card on a held-out dense split with at least the "
            "minimum sample count."
        )
    if (
        failed_or_warned.get("recall") == "fail"
        or failed_or_warned.get("undergeneration_ratio") == "fail"
    ):
        actions.append(
            "Prioritize count-conditioned dense decoding and wrapper-internalization "
            "training to reduce false negatives."
        )
    if (
        failed_or_warned.get("precision") == "fail"
        or failed_or_warned.get("overgeneration_ratio") == "fail"
    ):
        actions.append(
            "Tune label constraints, NMS, and negative/no-object samples before "
            "broadening the adapter."
        )
    if failed_or_warned.get("repair_dependency") == "warn":
        actions.append(
            "Keep malformed-tail repair as an explicit ablation and validate "
            "repair-on/off on held-out data."
        )
    if failed_or_warned.get("raw_vp_internalization") == "fail":
        actions.append(
            "Scale wrapper-focused SFT until raw VP format validity clears the "
            "report-card threshold."
        )
    if failed_or_warned.get("structured_decoder_dependency") == "fail":
        actions.append(
            "Keep structured decoding as a fallback path and report raw-vs-structured "
            "metrics separately."
        )
    if failed_or_warned.get("policy_confidence") == "warn":
        actions.append(
            "Repeat the policy sweep on more records before treating the recommended "
            "policy as default."
        )
    if failed_or_warned.get("target_count_gap") == "warn":
        actions.append(
            "Use target-count gap rows to drive proposal distillation or continuation "
            "training for recoverable false negatives."
        )

    if policy_summary and str(policy_summary.get("recommended_policy") or "").endswith("_repair"):
        actions.append(
            "Treat the repair policy as a diagnostic win, then train the model to "
            "emit the repaired boxes natively."
        )
    if gap_summary and _int(gap_summary.get("unrecoverable_false_negatives")) > 0:
        actions.append(
            "Inspect unrecoverable false negatives; they need visual proposal/data "
            "coverage, not only count conditioning."
        )
    if not actions and _overall_status(checks) == "pass":
        actions.append(
            "Promote this configuration to adapter-vs-baseline regression and freeze "
            "thresholds for future runs."
        )
    elif not actions:
        actions.append(
            "Review bad cases in the source VP quality report before starting another "
            "training run."
        )
    return _dedupe(actions)


def _metric_min_check(name: str, value: Any, threshold: float) -> Dict[str, Any]:
    parsed = _float(value)
    return _check(
        name,
        "pass" if parsed >= threshold else "fail",
        _round(parsed),
        f">= {threshold:.4f}",
        f"{name} meets promotion threshold",
        f"{name} is below promotion threshold",
    )


def _metric_max_check(name: str, value: Any, threshold: float) -> Dict[str, Any]:
    parsed = _float(value)
    return _check(
        name,
        "pass" if parsed <= threshold else "fail",
        _round(parsed),
        f"<= {threshold:.4f}",
        f"{name} is within tolerance",
        f"{name} exceeds tolerance",
    )


def _optional_metric_min_check(
    name: str,
    value: Any,
    threshold: float,
    pass_reason: str,
    fail_reason: str,
) -> Dict[str, Any]:
    parsed = _optional_float(value)
    if parsed is None:
        return _skip_check(name, "no prediction source diagnostics provided")
    return _check(
        name,
        "pass" if parsed >= threshold else "fail",
        _round(parsed),
        f">= {threshold:.4f}",
        pass_reason,
        fail_reason,
    )


def _optional_metric_max_check(
    name: str,
    value: Any,
    threshold: float,
    pass_reason: str,
    fail_reason: str,
) -> Dict[str, Any]:
    parsed = _optional_float(value)
    if parsed is None:
        return _skip_check(name, "no prediction source diagnostics provided")
    return _check(
        name,
        "pass" if parsed <= threshold else "fail",
        _round(parsed),
        f"<= {threshold:.4f}",
        pass_reason,
        fail_reason,
    )


def _check(
    name: str,
    status: str,
    value: Any,
    threshold: str,
    pass_reason: str,
    fail_reason: str,
) -> Dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "value": value,
        "threshold": threshold,
        "reason": pass_reason if status == "pass" else fail_reason,
    }


def _skip_check(name: str, reason: str) -> Dict[str, Any]:
    return {
        "name": name,
        "status": "skip",
        "value": None,
        "threshold": "-",
        "reason": reason,
    }


def _overall_status(checks: Sequence[Mapping[str, Any]]) -> str:
    statuses = {str(check.get("status")) for check in checks}
    if "fail" in statuses:
        return "fail"
    if "warn" in statuses:
        return "warn"
    return "pass"


def _readiness_label(status: str) -> str:
    if status == "pass":
        return "mvp_ready"
    if status == "warn":
        return "experimental"
    return "needs_work"


def _thresholds_to_dict(thresholds: VPReportCardThresholds) -> Dict[str, Any]:
    return {
        "min_samples": thresholds.min_samples,
        "min_precision": thresholds.min_precision,
        "min_recall": thresholds.min_recall,
        "min_f1": thresholds.min_f1,
        "max_undergeneration_ratio": thresholds.max_undergeneration_ratio,
        "max_overgeneration_ratio": thresholds.max_overgeneration_ratio,
        "max_repair_record_ratio": thresholds.max_repair_record_ratio,
        "min_raw_vp_format_ratio": thresholds.min_raw_vp_format_ratio,
        "max_structured_decoder_ratio": thresholds.max_structured_decoder_ratio,
        "min_policy_confidence": thresholds.min_policy_confidence,
        "high_recoverable_fn_ratio": thresholds.high_recoverable_fn_ratio,
    }


def _focus_bucket_metrics(report: Mapping[str, Any], focus_bucket: Optional[str]) -> Dict[str, Any]:
    if not focus_bucket:
        return {}
    bucket_summary = dict(report.get("box_count_bucket_summary", {}) or {})
    return dict(bucket_summary.get(focus_bucket, {}) or {})


def _best_repair_lift(rows: Sequence[Any]) -> Dict[str, Any]:
    by_policy = {
        str(row.get("policy")): row
        for row in rows
        if isinstance(row, Mapping) and row.get("policy")
    }
    best: Dict[str, Any] = {}
    for name, row in by_policy.items():
        if not name.endswith("_repair"):
            continue
        base_name = name[:-7]
        base = by_policy.get(base_name)
        if not isinstance(base, Mapping):
            continue
        delta_f1 = _float(row.get("f1")) - _float(base.get("f1"))
        delta_recall = _float(row.get("recall")) - _float(base.get("recall"))
        candidate = {
            "policy": name,
            "base_policy": base_name,
            "delta_f1": delta_f1,
            "delta_recall": delta_recall,
            "f1": _float(row.get("f1")),
            "base_f1": _float(base.get("f1")),
        }
        if not best or delta_f1 > _float(best.get("delta_f1")):
            best = candidate
    if best:
        best["delta_f1"] = _round(best["delta_f1"])
        best["delta_recall"] = _round(best["delta_recall"])
    return best


def _confidence_rank(value: str) -> int:
    ranks = {
        "none": 0,
        "exploratory": 1,
        "moderate": 2,
        "strong": 3,
    }
    return ranks.get(str(value or "").strip().lower(), 0)


def _dedupe(values: Sequence[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _round(value: Any) -> float:
    return round(_float(value), 6)


def _format_optional_ratio(value: Any) -> str:
    parsed = _optional_float(value)
    return "n/a" if parsed is None else f"{parsed:.4f}"
