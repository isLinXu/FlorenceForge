"""ResultAnalyzer 可选依赖的惰性加载。"""

from __future__ import annotations

from ..utils.optional_dependencies import missing_dependency_message


def load_plotting_dependencies():
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


def load_clustering_dependencies():
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


def clustering_available() -> bool:
    try:
        load_clustering_dependencies()
        return True
    except ImportError:
        return False


def load_psutil():
    try:
        import psutil
    except ImportError:
        return None
    return psutil
