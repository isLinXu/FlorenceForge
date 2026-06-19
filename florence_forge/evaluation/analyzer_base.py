"""FlorenceForge 结果分析器 — 基础层

提供 ResultAnalyzer 的初始化、结果加载，以及各 Mixin 共享的通用辅助方法。
"""

from __future__ import annotations

import json
import logging
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ..utils.optional_dependencies import missing_dependency_message

logger = logging.getLogger(__name__)


# ── 全局依赖加载器（各 Mixin 按需调用）───────────────────────────────────


def _load_plotting_dependencies():
    """按需加载绘图相关依赖。"""
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
    except ImportError as exc:
        raise ImportError(
            missing_dependency_message(
                "结果分析可视化功能",
                "matplotlib, seaborn, pandas",
            )
        ) from exc
    return plt, sns, pd


def _load_clustering_dependencies():
    """按需加载聚类分析相关依赖。"""
    try:
        from sklearn.cluster import DBSCAN, KMeans
        from sklearn.metrics import silhouette_score
    except ImportError as exc:
        raise ImportError(
            missing_dependency_message(
                "错误聚类分析功能",
                "scikit-learn",
            )
        ) from exc
    return KMeans, DBSCAN, silhouette_score


def _load_psutil():
    """按需加载系统资源监控依赖。"""
    try:
        import psutil
    except ImportError:
        return None
    return psutil


# ── 基础类 ────────────────────────────────────────────────────────────────


class ResultAnalyzerBase:
    """结果分析器基础类

    提供评估结果存储、缓存管理和各分析层共享的通用辅助方法。
    """

    def __init__(self, evaluation_results: Optional[Dict[str, Any]] = None):
        """初始化结果分析器

        Args:
            evaluation_results: 评估结果字典
        """
        self.evaluation_results = evaluation_results or {}
        self.analysis_cache = {}

    def load_results(self, results_path: Union[str, Path]) -> None:
        """从文件加载评估结果

        Args:
            results_path: 结果文件路径
        """
        results_path = Path(results_path)

        if results_path.suffix == '.json':
            with open(results_path, 'r', encoding='utf-8') as f:
                self.evaluation_results = json.load(f)
        else:
            raise ValueError(f"不支持的文件格式: {results_path.suffix}")

        self.analysis_cache = {}
        logger.info(f"评估结果已从 {results_path} 加载")

    # ── 通用性能辅助 ────────────────────────────────────────────────────

    def _rank_tasks_by_performance(
        self, task_metrics: Dict[str, Dict]
    ) -> List[Tuple[str, float]]:
        """按性能对任务排名"""
        task_scores = []
        for task_type, task_data in task_metrics.items():
            metrics = task_data['metrics']
            score = self._calculate_performance_score(metrics)
            task_scores.append((task_type, score))
        task_scores.sort(key=lambda x: x[1], reverse=True)
        return task_scores

    def _analyze_metric_distribution(
        self, task_metrics: Dict[str, Dict]
    ) -> Dict[str, Dict[str, float]]:
        """分析指标分布"""
        metric_values = defaultdict(list)
        for task_data in task_metrics.values():
            metrics = task_data['metrics']
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    metric_values[metric_name].append(value)

        distribution = {}
        for metric_name, values in metric_values.items():
            if values:
                distribution[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values)
                }
        return distribution

    def _categorize_task_performance(
        self, task_metrics: Dict[str, Dict]
    ) -> Dict[str, List[str]]:
        """将任务按性能分类"""
        categories = {
            '优秀': [],
            '良好': [],
            '一般': [],
            '需改进': []
        }
        for task_type, task_data in task_metrics.items():
            metrics = task_data['metrics']
            score = self._calculate_performance_score(metrics)
            if score >= 0.9:
                categories['优秀'].append(task_type)
            elif score >= 0.7:
                categories['良好'].append(task_type)
            elif score >= 0.5:
                categories['一般'].append(task_type)
            else:
                categories['需改进'].append(task_type)
        return categories

    def _analyze_metric_correlations(
        self, task_metrics: Dict[str, Dict]
    ) -> Dict[str, float]:
        """分析指标相关性"""
        metric_values = defaultdict(list)
        for task_data in task_metrics.values():
            metrics = task_data['metrics']
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    metric_values[metric_name].append(value)

        correlations = {}
        metric_names = list(metric_values.keys())
        for i, metric1 in enumerate(metric_names):
            for metric2 in metric_names[i + 1:]:
                values1 = metric_values[metric1]
                values2 = metric_values[metric2]
                if len(values1) > 1 and len(values2) > 1:
                    correlation = np.corrcoef(values1, values2)[0, 1]
                    correlations[f"{metric1}_vs_{metric2}"] = correlation
        return correlations

    def _calculate_performance_score(self, metrics: Dict[str, float]) -> float:
        """计算综合性能分数"""
        key_metrics = self._extract_key_metrics(metrics)
        if not key_metrics:
            return 0.0
        weights = {
            'accuracy': 0.3,
            'f1': 0.3,
            'precision': 0.2,
            'recall': 0.2
        }
        score = 0.0
        total_weight = 0.0
        for metric, weight in weights.items():
            if metric in key_metrics:
                score += key_metrics[metric] * weight
                total_weight += weight
        if total_weight > 0:
            score /= total_weight
        return score

    def _calculate_difficulty_score(self, metrics: Dict[str, float]) -> float:
        """计算综合难度分数"""
        key_metrics = self._extract_key_metrics(metrics)
        if not key_metrics:
            return 0.5
        performance_score = self._calculate_performance_score(metrics)
        difficulty = 1.0 - performance_score
        return max(0.0, min(1.0, difficulty))

    def _extract_key_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """提取关键指标"""
        key_names = ['accuracy', 'f1', 'precision', 'recall', 'iou', 'ap']
        return {name: metrics[name] for name in key_names if name in metrics}

    def _generate_recommendations(self) -> List[str]:
        """生成改进建议"""
        recommendations = []
        task_metrics = self.evaluation_results.get('task_metrics', {})
        if not task_metrics:
            return recommendations

        categories = self._categorize_task_performance(task_metrics)
        if categories['需改进']:
            recommendations.append(
                f"以下任务性能较低，建议优化: {', '.join(categories['需改进'])}"
            )
        if categories['优秀']:
            recommendations.append(
                f"以下任务表现优秀，可作为参考: {', '.join(categories['优秀'])}"
            )

        correlation_analysis = self._analyze_metric_correlations(task_metrics)
        negative_correlations = [
            pair for pair, corr in correlation_analysis.items() if corr < -0.3
        ]
        if negative_correlations:
            recommendations.append(
                f"发现负相关指标对，可能存在冲突: {', '.join(negative_correlations)}"
            )

        return recommendations

    def _get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        import platform
        info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'processor': platform.processor(),
        }
        psutil = _load_psutil()
        if psutil:
            info['cpu_count'] = psutil.cpu_count()
            info['memory_gb'] = psutil.virtual_memory().total / (1024 ** 3)
        return info

    # ── 通用错误分析辅助（被 ErrorMixin 和 DiagnosisMixin 共享）──────────

    def _classify_error_type(
        self, prediction: str, reference: str, task_type: str
    ) -> str:
        """分类错误类型"""
        if not prediction.strip():
            return 'empty_prediction'
        if not reference.strip():
            return 'empty_reference'
        pred_lower = prediction.lower().strip()
        ref_lower = reference.lower().strip()
        if pred_lower == ref_lower:
            return 'correct'
        if task_type in ['OD', 'REGION_PROPOSAL']:
            return 'detection_error'
        if task_type in ['CAPTION', 'DETAILED_CAPTION']:
            return 'description_error'
        if task_type in ['OCR', 'OCR_WITH_REGION']:
            return 'ocr_error'
        return 'general_error'

    def _extract_error_features(
        self, prediction: str, reference: str, task_type: str
    ) -> List[float]:
        """提取错误特征向量"""
        features = []
        pred_len = len(prediction.split()) if prediction.strip() else 0
        ref_len = len(reference.split()) if reference.strip() else 0
        features.append(pred_len / max(ref_len, 1))
        features.append(1.0 if pred_len == 0 else 0.0)
        features.append(1.0 if ref_len == 0 else 0.0)
        pred_set = set(prediction.lower().split()) if prediction.strip() else set()
        ref_set = set(reference.lower().split()) if reference.strip() else set()
        overlap = len(pred_set & ref_set) / max(len(pred_set | ref_set), 1)
        features.append(overlap)
        features.append(1.0 - overlap)
        return features
