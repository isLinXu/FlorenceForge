"""FlorenceForge 高级评估指标导出入口。"""

from importlib import import_module

__all__ = [
    "SemanticMetricsCalculator",
    "MultiModalMetricsCalculator",
    "RobustnessMetricsCalculator",
    "EfficiencyMetricsCalculator",
]

_LAZY_EXPORTS = {
    "SemanticMetricsCalculator": (
        "florence_forge.evaluation.advanced_metrics.semantic_metrics_calculator",
        "SemanticMetricsCalculator",
    ),
    "MultiModalMetricsCalculator": (
        "florence_forge.evaluation.advanced_metrics.multimodal_metrics_calculator",
        "MultiModalMetricsCalculator",
    ),
    "RobustnessMetricsCalculator": (
        "florence_forge.evaluation.advanced_metrics.robustness_metrics_calculator",
        "RobustnessMetricsCalculator",
    ),
    "EfficiencyMetricsCalculator": (
        "florence_forge.evaluation.advanced_metrics.efficiency_metrics_calculator",
        "EfficiencyMetricsCalculator",
    ),
}


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        module = import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
