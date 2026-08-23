"""FlorenceForge指标计算模块 — 兼容层，实现已拆分至 task_metrics/。"""

from .task_metrics import (
    COCO_AVAILABLE,
    CV2_AVAILABLE,
    ROUGE_AVAILABLE,
    CaptionMetrics,
    DetectionMetrics,
    MetricCalculator,
    OCRMetrics,
    SegmentationMetrics,
    VisualPrimitiveDetectionMetrics,
    get_metric_calculator,
)

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
