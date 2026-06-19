"""VP aggregation helpers — backward-compatible re-export facade.

All implementations have been merged into ``vp_core.py``.  This module re-exports
the aggregation helpers to preserve existing import paths.
"""

from __future__ import annotations

from .vp_core import (
    aggregate_counts,
    compare_quality_record,
    int_record_metric,
    mean,
    quality_record_delta_outcome,
    quality_report_brief,
    ratio,
    record_f1,
    safe_policy_label,
    summarize_box_count_buckets,
    summarize_bucket_records,
    summarize_quality_record_comparison,
    summarize_quality_record_comparison_buckets,
    summarize_target_count_gap_buckets,
    summarize_target_count_gap_rows,
    target_count_gap_row,
    top_record_deltas,
)

__all__ = [
    "aggregate_counts",
    "compare_quality_record",
    "int_record_metric",
    "mean",
    "quality_record_delta_outcome",
    "quality_report_brief",
    "ratio",
    "record_f1",
    "safe_policy_label",
    "summarize_box_count_buckets",
    "summarize_bucket_records",
    "summarize_quality_record_comparison",
    "summarize_quality_record_comparison_buckets",
    "summarize_target_count_gap_buckets",
    "summarize_target_count_gap_rows",
    "target_count_gap_row",
    "top_record_deltas",
]
