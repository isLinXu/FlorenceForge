"""FlorenceForge 结果分析器模块

提供评估结果的深度分析和可视化功能
"""

from __future__ import annotations

from .analyzer_base import ResultAnalyzerBase
from .analyzer_performance import PerformanceMixin
from .analyzer_plots import PlotMixin
from .analyzer_errors import ErrorMixin
from .analyzer_diagnosis import DiagnosisMixin


class ResultAnalyzer(ResultAnalyzerBase, PerformanceMixin, PlotMixin, ErrorMixin, DiagnosisMixin):
    """结果分析器

    提供评估结果的深度分析和可视化功能
    """
    pass
