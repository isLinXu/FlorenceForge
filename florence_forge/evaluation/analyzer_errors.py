from __future__ import annotations

"""FlorenceForge 结果分析器 — 错误模式分析 Mixin

提供错误样本聚类、错误模式识别和改进建议生成。
"""

from collections import Counter, defaultdict
from typing import Any, Dict, List

import numpy as np

from .analyzer_base import logger, _load_clustering_dependencies

try:
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.metrics import silhouette_score
    CLUSTERING_AVAILABLE = True
except ImportError:
    CLUSTERING_AVAILABLE = False


class ErrorMixin:
    """错误模式分析 Mixin"""

    def analyze_error_patterns(self, clustering_method: str = 'kmeans', n_clusters: int = 5) -> Dict[str, Any]:
        """错误模式聚类分析

        Args:
            clustering_method: 聚类方法 ('kmeans' 或 'dbscan')
            n_clusters: 聚类数量（仅用于kmeans）

        Returns:
            错误模式分析结果
        """
        if not CLUSTERING_AVAILABLE:
            logger.warning("聚类分析库不可用，跳过错误模式分析")
            return {}

        # 收集错误样本
        error_samples = []
        error_features = []

        task_metrics = self.evaluation_results.get('task_metrics', {})

        for task_type, task_data in task_metrics.items():
            predictions = task_data.get('predictions', [])

            for pred_data in predictions:
                prediction = pred_data.get('prediction', '')
                reference = pred_data.get('reference', '')

                # 识别错误样本
                if prediction != reference:
                    error_info = {
                        'task_type': task_type,
                        'prediction': prediction,
                        'reference': reference,
                        'error_type': self._classify_error_type(prediction, reference, task_type)
                    }
                    error_samples.append(error_info)

                    # 提取特征用于聚类
                    features = self._extract_error_features(prediction, reference, task_type)
                    error_features.append(features)

        if not error_samples:
            return {'message': '没有发现错误样本'}

        # 执行聚类分析
        cluster_results = self._perform_error_clustering(
            error_features, clustering_method, n_clusters
        )

        # 分析聚类结果
        pattern_analysis = self._analyze_error_clusters(
            error_samples, cluster_results['labels']
        )

        return {
            'total_errors': len(error_samples),
            'clustering_method': clustering_method,
            'n_clusters': cluster_results.get('n_clusters', n_clusters),
            'silhouette_score': cluster_results.get('silhouette_score'),
            'error_patterns': pattern_analysis,
            'cluster_distribution': Counter(cluster_results['labels']),
            'recommendations': self._generate_error_pattern_recommendations(pattern_analysis)
        }

    def _perform_error_clustering(self, features: List[List[float]], method: str, n_clusters: int) -> Dict[str, Any]:
        """执行错误聚类"""
        if not features:
            return {'labels': [], 'n_clusters': 0}

        try:
            KMeans, DBSCAN, silhouette_score = _load_clustering_dependencies()
        except ImportError as exc:
            logger.warning(str(exc))
            return {
                'labels': [0] * len(features),
                'n_clusters': 0,
                'silhouette_score': None,
                'error': str(exc),
            }

        features_array = np.array(features)

        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = clusterer.fit_predict(features_array)

            # 计算轮廓系数
            if len(set(labels)) > 1:
                sil_score = silhouette_score(features_array, labels)
            else:
                sil_score = 0.0

            return {
                'labels': labels.tolist(),
                'n_clusters': n_clusters,
                'silhouette_score': sil_score
            }

        elif method == 'dbscan':
            clusterer = DBSCAN(eps=0.5, min_samples=3)
            labels = clusterer.fit_predict(features_array)

            n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)

            if n_clusters_found > 1:
                sil_score = silhouette_score(features_array, labels)
            else:
                sil_score = 0.0

            return {
                'labels': labels.tolist(),
                'n_clusters': n_clusters_found,
                'silhouette_score': sil_score
            }

        else:
            raise ValueError(f"不支持的聚类方法: {method}")

    def _analyze_error_clusters(self, error_samples: List[Dict], labels: List[int]) -> Dict[str, Any]:
        """分析错误聚类结果"""
        clusters = defaultdict(list)

        for sample, label in zip(error_samples, labels):
            clusters[label].append(sample)

        pattern_analysis = {}

        for cluster_id, samples in clusters.items():
            if cluster_id == -1:  # DBSCAN噪声点
                continue

            # 分析聚类特征
            error_types = [s['error_type'] for s in samples]
            task_types = [s['task_type'] for s in samples]

            pattern_analysis[f'cluster_{cluster_id}'] = {
                'size': len(samples),
                'dominant_error_type': Counter(error_types).most_common(1)[0][0],
                'error_type_distribution': dict(Counter(error_types)),
                'task_distribution': dict(Counter(task_types)),
                'sample_examples': samples[:3]  # 前3个样本作为示例
            }

        return pattern_analysis

    def _generate_error_pattern_recommendations(self, pattern_analysis: Dict[str, Any]) -> List[str]:
        """生成错误模式改进建议"""
        recommendations = []

        for cluster_id, cluster_info in pattern_analysis.items():
            dominant_error = cluster_info['dominant_error_type']
            cluster_size = cluster_info['size']

            if dominant_error == 'empty_prediction' and cluster_size > 5:
                recommendations.append(
                    f"发现{cluster_size}个空预测错误，建议检查模型输入预处理和解码逻辑"
                )

            elif dominant_error == 'missing_location' and cluster_size > 3:
                recommendations.append(
                    f"发现{cluster_size}个位置标记缺失错误，建议加强位置检测训练"
                )

            elif dominant_error in ['too_short', 'too_long'] and cluster_size > 5:
                recommendations.append(
                    f"发现{cluster_size}个长度异常错误，建议调整生成长度控制参数"
                )

        return recommendations
