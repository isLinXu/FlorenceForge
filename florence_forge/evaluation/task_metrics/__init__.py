"""Task-level metric calculators — split from the legacy metrics monolith."""

from .base import MetricCalculator
from .calculators import (
    CaptionMetrics,
    DetectionMetrics,
    OCRMetrics,
    SegmentationMetrics,
)
from .registry import get_metric_calculator
from .vp import VisualPrimitiveDetectionMetrics
from ._deps import COCO_AVAILABLE, CV2_AVAILABLE, ROUGE_AVAILABLE

__all__ = [
    "MetricCalculator",
    "CaptionMetrics",
    "DetectionMetrics",
    "OCRMetrics",
    "SegmentationMetrics",
    "VisualPrimitiveDetectionMetrics",
    "get_metric_calculator",
    "COCO_AVAILABLE",
    "CV2_AVAILABLE",
    "ROUGE_AVAILABLE",
]
