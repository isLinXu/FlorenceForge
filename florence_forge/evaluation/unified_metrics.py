"""unified_metrics.py — 统一的指标计算模块

合并基础指标与高级指标计算，使用 availability registry 避免静默降级。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from .advanced_metrics_registry import (
    advanced_metrics_status_report,
    get_available_advanced_calculators,
    unavailable_advanced_metrics,
)

logger = logging.getLogger(__name__)


class UnifiedMetrics:
    """统一指标计算类 — 基础 + 高级（Semantic/Efficiency/Robustness）。"""

    def __init__(self) -> None:
        self._basic_metrics = None
        self._advanced_registry: Dict[str, Any] | None = None
        self._availability_logged = False

    @property
    def basic_metrics(self):
        if self._basic_metrics is None:
            from .metrics import MetricCalculator
            self._basic_metrics = MetricCalculator("unified")
        return self._basic_metrics

    @property
    def advanced_metrics(self) -> Dict[str, Any]:
        if self._advanced_registry is None:
            self._advanced_registry = get_available_advanced_calculators()
            if not self._availability_logged:
                missing = unavailable_advanced_metrics()
                if missing:
                    logger.warning(
                        "高级评估指标不可用（将跳过，不会返回假分数）: %s",
                        ", ".join(missing),
                    )
                self._availability_logged = True
        return self._advanced_registry

    def availability_report(self) -> Dict[str, Any]:
        """Return structured availability status for doctor / benchmark logs."""
        return advanced_metrics_status_report()

    def compute(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, Any]:
        self.basic_metrics.predictions = predictions
        self.basic_metrics.references = references
        result: Dict[str, Any] = dict(self.basic_metrics.compute())

        for name, calculator_cls in self.advanced_metrics.items():
            try:
                calculator = calculator_cls()
                advanced_result = calculator.compute(predictions, references)
                if advanced_result:
                    result.update(advanced_result)
            except Exception as e:
                logger.warning("高级指标 %s 计算失败: %s", name, e)

        if unavailable_advanced_metrics():
            result["_advanced_metrics_availability"] = advanced_metrics_status_report()

        return result


__all__ = ["UnifiedMetrics"]
