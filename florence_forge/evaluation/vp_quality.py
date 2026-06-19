"""Unified VP quality evaluation — detection quality, policy comparison, and report cards.

This module consolidates the public API previously split across
``vp_detection_quality`` and ``vp_report_card``.  Both legacy modules
are retained as thin re-export shims for backward compatibility.
"""

from __future__ import annotations

# ── Detection quality (from vp_detection_quality) ────────────────────────
from .vp_detection_quality import (  # noqa: F401
    VPDetectionQualityConfig,
    analyze_vp_target_count_gap,
    compare_vp_quality_record_reports,
    compare_vp_quality_reports,
    compute_bbox_iou,
    evaluate_vp_detection_quality,
    evaluate_vp_summary,
    match_vp_detections,
    recommend_vp_policy,
    render_vp_detection_quality_markdown,
    render_vp_policy_comparison_markdown,
    render_vp_record_comparison_markdown,
    render_vp_target_count_gap_markdown,
    summarize_vp_quality_records,
)

# ── Report card (from vp_report_card) ───────────────────────────────────
from .vp_report_card import (  # noqa: F401
    VPReportCardThresholds,
    build_vp_report_card,
    render_vp_report_card_markdown,
)

__all__ = [
    # Detection quality
    "VPDetectionQualityConfig",
    "analyze_vp_target_count_gap",
    "compare_vp_quality_record_reports",
    "compare_vp_quality_reports",
    "compute_bbox_iou",
    "evaluate_vp_detection_quality",
    "evaluate_vp_summary",
    "match_vp_detections",
    "recommend_vp_policy",
    "render_vp_detection_quality_markdown",
    "render_vp_policy_comparison_markdown",
    "render_vp_record_comparison_markdown",
    "render_vp_target_count_gap_markdown",
    "summarize_vp_quality_records",
    # Report card
    "VPReportCardThresholds",
    "build_vp_report_card",
    "render_vp_report_card_markdown",
]
