"""Task-type → metric calculator routing."""

from __future__ import annotations

from .base import MetricCalculator
from .calculators import (
    CaptionMetrics,
    DetectionMetrics,
    OCRMetrics,
    SegmentationMetrics,
)
from .vp import VisualPrimitiveDetectionMetrics


def get_metric_calculator(task_type: str) -> MetricCalculator:
    """根据任务类型返回对应的指标计算器。"""
    task_type_lower = task_type.lower()

    detection_aliases = {
        "od",
        "open_vocabulary_detection",
        "region_proposal",
        "phrase_grounding",
        "caption_to_phrase_grounding",
    }
    segmentation_aliases = {
        "region_to_segmentation",
        "referring_expression_segmentation",
        "seg",
        "segmentation",
    }

    if "caption" in task_type_lower or "description" in task_type_lower:
        return CaptionMetrics()
    if "_vp" in task_type_lower or "visual_primitive" in task_type_lower:
        return VisualPrimitiveDetectionMetrics()
    if (
        "detection" in task_type_lower
        or "object" in task_type_lower
        or task_type_lower in detection_aliases
    ):
        return DetectionMetrics()
    if "ocr" in task_type_lower or "text" in task_type_lower:
        return OCRMetrics()
    if (
        "segmentation" in task_type_lower
        or "segment" in task_type_lower
        or task_type_lower in segmentation_aliases
    ):
        return SegmentationMetrics()
    return MetricCalculator(task_type)
