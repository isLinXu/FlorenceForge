"""FlorenceForge 评估模块导出入口。"""

from importlib import import_module

__all__ = [
    "MultiTaskEvaluator",
    "BenchmarkEvaluator",
    "MetricCalculator",
    "CaptionMetrics",
    "DetectionMetrics",
    "OCRMetrics",
    "SegmentationMetrics",
    "ResultAnalyzer",
    "SemanticMetricsCalculator",
    "MultiModalMetricsCalculator",
    "RobustnessMetricsCalculator",
    "EfficiencyMetricsCalculator",
]

_LAZY_EXPORTS = {
    "MultiTaskEvaluator": ("florence_forge.evaluation.evaluator", "MultiTaskEvaluator"),
    "BenchmarkEvaluator": ("florence_forge.evaluation.benchmark", "BenchmarkEvaluator"),
    "MetricCalculator": ("florence_forge.evaluation.metrics", "MetricCalculator"),
    "CaptionMetrics": ("florence_forge.evaluation.metrics", "CaptionMetrics"),
    "DetectionMetrics": ("florence_forge.evaluation.metrics", "DetectionMetrics"),
    "OCRMetrics": ("florence_forge.evaluation.metrics", "OCRMetrics"),
    "SegmentationMetrics": ("florence_forge.evaluation.metrics", "SegmentationMetrics"),
    "ResultAnalyzer": ("florence_forge.evaluation.analyzer", "ResultAnalyzer"),
    "SemanticMetricsCalculator": ("florence_forge.evaluation.advanced_metrics", "SemanticMetricsCalculator"),
    "MultiModalMetricsCalculator": ("florence_forge.evaluation.advanced_metrics", "MultiModalMetricsCalculator"),
    "RobustnessMetricsCalculator": ("florence_forge.evaluation.advanced_metrics", "RobustnessMetricsCalculator"),
    "EfficiencyMetricsCalculator": ("florence_forge.evaluation.advanced_metrics", "EfficiencyMetricsCalculator"),
}


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        module = import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
