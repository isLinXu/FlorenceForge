"""unified_metrics.py — 统一的指标计算模块

合并 _metrics.py 和 _metrics_calculator.py 为一个模块。
原有的两个文件分别实现了不同方面的指标计算逻辑：
- _metrics.py 专注于基础指标的存储和累积
- _metrics_calculator.py 专注于高级指标的批量计算

本模块提供统一的接口，消除冗余，提高可维护性。
"""

import logging

logger = logging.getLogger(__name__)

class UnifiedMetrics:
    """统一指标计算类

    合并了基础指标存储和高级指标计算的功能。
    使用延迟加载避免循环导入。
    """

    def __init__(self):
        self._basic_metrics = None
        self._advanced_metrics = None

    @property
    def basic_metrics(self):
        """延迟加载基础指标计算器"""
        if self._basic_metrics is None:
            from .metrics import MetricCalculator
            self._basic_metrics = MetricCalculator("unified")
        return self._basic_metrics

    @property
    def advanced_metrics(self):
        """延迟加载高级指标计算器"""
        if self._advanced_metrics is None:
            try:
                from .advanced_metrics import (
                    SemanticMetricsCalculator,
                    EfficiencyMetricsCalculator,
                    RobustnessMetricsCalculator,
                )
                self._advanced_metrics = {
                    'semantic': SemanticMetricsCalculator,
                    'efficiency': EfficiencyMetricsCalculator,
                    'robustness': RobustnessMetricsCalculator,
                }
            except ImportError:
                self._advanced_metrics = {}
        return self._advanced_metrics

    def compute(self, predictions, references):
        """计算统一指标

        Args:
            predictions: 预测结果列表
            references: 参考结果列表

        Returns:
            包含所有指标的字典
        """
        # 基础指标
        self.basic_metrics.predictions = predictions
        self.basic_metrics.references = references
        result = self.basic_metrics.compute()

        # 高级指标（延迟加载）
        for name, calculator_cls in self.advanced_metrics.items():
            try:
                calculator = calculator_cls()
                advanced_result = calculator.compute(predictions, references)
                result.update(advanced_result)
            except Exception as e:
                logger.warning(f"高级指标 {name} 计算失败: {e}")

        return result


# 公开API
__all__ = ['UnifiedMetrics']
