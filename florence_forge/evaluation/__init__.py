"""FlorenceForge评估模块

提供模型评估、指标计算和结果分析功能
"""

from .evaluator import MultiTaskEvaluator
from .metrics import (
    MetricCalculator,
    CaptionMetrics,
    DetectionMetrics,
    OCRMetrics,
    SegmentationMetrics
)
from .analyzer import ResultAnalyzer

__all__ = [
    'MultiTaskEvaluator',
    'MetricCalculator',
    'CaptionMetrics',
    'DetectionMetrics', 
    'OCRMetrics',
    'SegmentationMetrics',
    'ResultAnalyzer'
]