"""Lazy wrappers for benchmark-only advanced metric calculators."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Optional


class LazyMetricCalculator:
    """Proxy that instantiates a metric calculator on first real use."""

    def __init__(
        self,
        module_name: str,
        class_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self._module_name = module_name
        self._class_name = class_name
        self._args = args
        self._kwargs = kwargs
        self._instance: Optional[Any] = None

    @property
    def is_loaded(self) -> bool:
        return self._instance is not None

    def load(self) -> Any:
        """Instantiate and return the wrapped calculator."""
        if self._instance is None:
            module = import_module(self._module_name)
            calculator_cls = getattr(module, self._class_name)
            self._instance = calculator_cls(*self._args, **self._kwargs)
        return self._instance

    def __getattr__(self, name: str) -> Any:
        return getattr(self.load(), name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.load()(*args, **kwargs)

    def __repr__(self) -> str:
        state = "loaded" if self.is_loaded else "pending"
        return f"<LazyMetricCalculator {self._class_name} {state}>"


def make_default_advanced_metric_calculators() -> dict[str, LazyMetricCalculator]:
    """Create lazy proxies for BenchmarkEvaluator advanced metric attributes."""
    base = "florence_forge.evaluation.advanced_metrics"
    return {
        "semantic_calculator": LazyMetricCalculator(
            f"{base}.semantic_metrics_calculator",
            "SemanticMetricsCalculator",
        ),
        "multimodal_calculator": LazyMetricCalculator(
            f"{base}.multimodal_metrics_calculator",
            "MultiModalMetricsCalculator",
        ),
        "robustness_calculator": LazyMetricCalculator(
            f"{base}.robustness_metrics_calculator",
            "RobustnessMetricsCalculator",
        ),
        "efficiency_calculator": LazyMetricCalculator(
            f"{base}.efficiency_metrics_calculator",
            "EfficiencyMetricsCalculator",
        ),
    }
