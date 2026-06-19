"""VP parsing helpers — backward-compatible re-export facade.

All implementations have been merged into ``vp_core.py``.  This module re-exports
the parsing helpers to preserve existing import paths.
"""

from __future__ import annotations

from .vp_core import (
    allowed_label_field_candidates,
    bad_case_reasons,
    box_count_bucket,
    index_quality_records,
    is_box,
    normalize_label,
    parse_prediction,
    quality_record_key,
    record_field_value,
    record_query_box_count,
    record_text,
    resolve_record_allowed_labels,
    resolve_record_positive_int_field,
    summary_records,
)

__all__ = [
    "allowed_label_field_candidates",
    "bad_case_reasons",
    "box_count_bucket",
    "index_quality_records",
    "is_box",
    "normalize_label",
    "parse_prediction",
    "quality_record_key",
    "record_field_value",
    "record_query_box_count",
    "record_text",
    "resolve_record_allowed_labels",
    "resolve_record_positive_int_field",
    "summary_records",
]
