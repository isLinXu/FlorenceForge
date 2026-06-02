"""FlorenceForge结果分析器模块

提供评估结果的深度分析和可视化功能
"""

import json
import logging
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import time
from datetime import datetime

from ..utils.optional_dependencies import missing_dependency_message

logger = logging.getLogger(__name__)


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

class ResultAnalyzer:
    """结果分析器
    
    提供评估结果的深度分析和可视化功能
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
        
        # 清空缓存
        self.analysis_cache = {}
        
        logger.info(f"评估结果已从 {results_path} 加载")
    
    def analyze_task_performance(self) -> Dict[str, Any]:
        """分析任务性能
        
        Returns:
            任务性能分析结果
        """
        if 'task_performance' in self.analysis_cache:
            return self.analysis_cache['task_performance']
        
        task_metrics = self.evaluation_results.get('task_metrics', {})
        
        if not task_metrics:
            return {}
        
        analysis = {
            'task_ranking': self._rank_tasks_by_performance(task_metrics),
            'metric_distribution': self._analyze_metric_distribution(task_metrics),
            'performance_categories': self._categorize_task_performance(task_metrics),
            'correlation_analysis': self._analyze_metric_correlations(task_metrics)
        }
        
        self.analysis_cache['task_performance'] = analysis
        return analysis
    
    def analyze_error_patterns(
        self,
        predictions_dir: Optional[Union[str,
        Path]] = None
    ) -> Dict[str, Any]:
        """分析错误模式
        
        Args:
            predictions_dir: 预测结果目录
            
        Returns:
            错误模式分析结果
        """
        if 'error_patterns' in self.analysis_cache:
            return self.analysis_cache['error_patterns']
        
        analysis = {
            'common_errors': {},
            'error_frequency': {},
            'task_specific_errors': {}
        }
        
        # 如果提供了预测结果目录，分析具体错误
        if predictions_dir:
            predictions_dir = Path(predictions_dir)
            
            for pred_file in predictions_dir.glob('predictions_*.json'):
                task_type = pred_file.stem.replace('predictions_', '')
                
                with open(pred_file, 'r', encoding='utf-8') as f:
                    predictions = json.load(f)
                
                task_errors = self._analyze_task_errors(predictions, task_type)
                analysis['task_specific_errors'][task_type] = task_errors
        
        self.analysis_cache['error_patterns'] = analysis
        return analysis
    
    def analyze_sample_difficulty(self) -> Dict[str, Any]:
        """分析样本难度
        
        Returns:
            样本难度分析结果
        """
        if 'sample_difficulty' in self.analysis_cache:
            return self.analysis_cache['sample_difficulty']
        
        task_metrics = self.evaluation_results.get('task_metrics', {})
        
        analysis = {
            'difficulty_distribution': {},
            'challenging_tasks': [],
            'easy_tasks': []
        }
        
        # 基于性能指标推断难度
        for task_type, task_data in task_metrics.items():
            metrics = task_data['metrics']
            sample_count = task_data['sample_count']
            
            # 计算综合难度分数
            difficulty_score = self._calculate_difficulty_score(metrics)
            
            analysis['difficulty_distribution'][task_type] = {
                'difficulty_score': difficulty_score,
                'sample_count': sample_count,
                'key_metrics': self._extract_key_metrics(metrics)
            }
            
            # 分类任务难度
            if difficulty_score > 0.7:
                analysis['challenging_tasks'].append(task_type)
            elif difficulty_score < 0.3:
                analysis['easy_tasks'].append(task_type)
        
        self.analysis_cache['sample_difficulty'] = analysis
        return analysis
    
    def generate_performance_report(self, output_path: Optional[Union[str, Path]] = None) -> str:
        """生成性能报告
        
        Args:
            output_path: 输出文件路径
            
        Returns:
            报告内容
        """
        report_lines = []
        
        # 报告标题
        report_lines.append("# Florence2 多任务模型性能报告\n")
        
        # 总体性能
        overall_metrics = self.evaluation_results.get('overall_metrics', {})
        if overall_metrics:
            report_lines.append("## 总体性能")
            report_lines.append("")
            
            for metric_name, value in overall_metrics.items():
                if isinstance(value, (int, float)):
                    report_lines.append(f"- **{metric_name}**: {value:.4f}")
                else:
                    report_lines.append(f"- **{metric_name}**: {value}")
            
            report_lines.append("")
        
        # 任务性能分析
        task_analysis = self.analyze_task_performance()
        if task_analysis:
            report_lines.append("## 任务性能分析")
            report_lines.append("")
            
            # 任务排名
            if 'task_ranking' in task_analysis:
                report_lines.append("### 任务性能排名")
                report_lines.append("")
                
                for i, (task_type, score) in enumerate(task_analysis['task_ranking'][:10], 1):
                    report_lines.append(f"{i}. **{task_type}**: {score:.4f}")
                
                report_lines.append("")
            
            # 性能分类
            if 'performance_categories' in task_analysis:
                categories = task_analysis['performance_categories']
                
                report_lines.append("### 性能分类")
                report_lines.append("")
                
                for category, tasks in categories.items():
                    if tasks:
                        report_lines.append(f"**{category}**: {', '.join(tasks)}")
                
                report_lines.append("")
        
        # 样本难度分析
        difficulty_analysis = self.analyze_sample_difficulty()
        if difficulty_analysis:
            report_lines.append("## 样本难度分析")
            report_lines.append("")
            
            if difficulty_analysis['challenging_tasks']:
                report_lines.append(f"**挑战性任务**: {', '.join(difficulty_analysis['challenging_tasks'])}")
            
            if difficulty_analysis['easy_tasks']:
                report_lines.append(f"**简单任务**: {', '.join(difficulty_analysis['easy_tasks'])}")
            
            report_lines.append("")
        
        # 评估信息
        eval_info = self.evaluation_results.get('evaluation_info', {})
        if eval_info:
            report_lines.append("## 评估信息")
            report_lines.append("")
            
            for key, value in eval_info.items():
                report_lines.append(f"- **{key}**: {value}")
            
            report_lines.append("")
        
        # 建议和改进方向
        recommendations = self._generate_recommendations()
        if recommendations:
            report_lines.append("## 建议和改进方向")
            report_lines.append("")
            
            for rec in recommendations:
                report_lines.append(f"- {rec}")
            
            report_lines.append("")
        
        report_content = "\n".join(report_lines)
        
        # 保存报告
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            logger.info(f"性能报告已保存到: {output_path}")
        
        return report_content
    
    def plot_task_performance(self, output_dir: Optional[Union[str, Path]] = None) -> None:
        """绘制任务性能图表
        
        Args:
            output_dir: 输出目录
        """
        try:
            plt, sns, _ = _load_plotting_dependencies()
        except ImportError as exc:
            logger.warning(str(exc))
            return

        plt.style.use('default')
        sns.set_palette("husl")
        
        task_metrics = self.evaluation_results.get('task_metrics', {})
        if not task_metrics:
            logger.warning("没有任务指标数据")
            return
        
        # 准备数据
        task_names = []
        performance_scores = []
        sample_counts = []
        
        for task_type, task_data in task_metrics.items():
            metrics = task_data['metrics']
            score = self._calculate_performance_score(metrics)
            
            task_names.append(task_type)
            performance_scores.append(score)
            sample_counts.append(task_data['sample_count'])
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('任务性能分析', fontsize=16)
        
        # 1. 性能分数条形图
        axes[0, 0].bar(range(len(task_names)), performance_scores)
        axes[0, 0].set_title('任务性能分数')
        axes[0, 0].set_xlabel('任务')
        axes[0, 0].set_ylabel('性能分数')
        axes[0, 0].set_xticks(range(len(task_names)))
        axes[0, 0].set_xticklabels(task_names, rotation=45, ha='right')
        
        # 2. 样本数量分布
        axes[0, 1].bar(range(len(task_names)), sample_counts, color='orange')
        axes[0, 1].set_title('样本数量分布')
        axes[0, 1].set_xlabel('任务')
        axes[0, 1].set_ylabel('样本数量')
        axes[0, 1].set_xticks(range(len(task_names)))
        axes[0, 1].set_xticklabels(task_names, rotation=45, ha='right')
        
        # 3. 性能vs样本数量散点图
        axes[1, 0].scatter(sample_counts, performance_scores, alpha=0.7)
        axes[1, 0].set_title('性能 vs 样本数量')
        axes[1, 0].set_xlabel('样本数量')
        axes[1, 0].set_ylabel('性能分数')
        
        # 添加任务标签
        for i, task in enumerate(task_names):
            axes[1, 0].annotate(task, (sample_counts[i], performance_scores[i]), 
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 4. 性能分数分布直方图
        axes[1, 1].hist(performance_scores, bins=10, alpha=0.7, color='green')
        axes[1, 1].set_title('性能分数分布')
        axes[1, 1].set_xlabel('性能分数')
        axes[1, 1].set_ylabel('任务数量')
        
        plt.tight_layout()
        
        # 保存图表
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            plt.savefig(output_dir / 'task_performance.png', dpi=300, bbox_inches='tight')
            logger.info(f"性能图表已保存到: {output_dir / 'task_performance.png'}")
        
        plt.show()
    
    def plot_metric_comparison(
        self,
        metric_names: List[str],
        output_dir: Optional[Union[str,
        Path]] = None
    ) -> None:
        """绘制指标比较图
        
        Args:
            metric_names: 要比较的指标名称列表
            output_dir: 输出目录
        """
        try:
            plt, sns, pd = _load_plotting_dependencies()
        except ImportError as exc:
            logger.warning(str(exc))
            return
        
        task_metrics = self.evaluation_results.get('task_metrics', {})
        if not task_metrics:
            logger.warning("没有任务指标数据")
            return
        
        # 准备数据
        data = []
        for task_type, task_data in task_metrics.items():
            metrics = task_data['metrics']
            
            for metric_name in metric_names:
                if metric_name in metrics:
                    data.append({
                        'task': task_type,
                        'metric': metric_name,
                        'value': metrics[metric_name]
                    })
        
        if not data:
            logger.warning("没有找到指定的指标数据")
            return
        
        # 创建DataFrame
        df = pd.DataFrame(data)
        
        # 创建图表
        plt.figure(figsize=(12, 8))
        
        # 使用seaborn绘制分组条形图
        sns.barplot(data=df, x='task', y='value', hue='metric')
        
        plt.title('任务指标比较')
        plt.xlabel('任务')
        plt.ylabel('指标值')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='指标')
        
        plt.tight_layout()
        
        # 保存图表
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            plt.savefig(output_dir / 'metric_comparison.png', dpi=300, bbox_inches='tight')
            logger.info(f"指标比较图已保存到: {output_dir / 'metric_comparison.png'}")
        
        plt.show()
    
    def _rank_tasks_by_performance(self, task_metrics: Dict[str, Dict]) -> List[Tuple[str, float]]:
        """按性能对任务排名"""
        task_scores = []
        
        for task_type, task_data in task_metrics.items():
            metrics = task_data['metrics']
            score = self._calculate_performance_score(metrics)
            task_scores.append((task_type, score))
        
        # 按分数降序排列
        task_scores.sort(key=lambda x: x[1], reverse=True)
        
        return task_scores
    
    def _analyze_metric_distribution(
        self,
        task_metrics: Dict[str,
        Dict]
    ) -> Dict[str, Dict[str, float]]:
        """分析指标分布"""
        metric_values = defaultdict(list)
        
        # 收集所有指标值
        for task_data in task_metrics.values():
            metrics = task_data['metrics']
            
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    metric_values[metric_name].append(value)
        
        # 计算统计信息
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
    
    def _categorize_task_performance(self, task_metrics: Dict[str, Dict]) -> Dict[str, List[str]]:
        """将任务按性能分类"""
        categories = {
            '优秀': [],
            '良好': [],
            '一般': [],
            '需要改进': []
        }
        
        for task_type, task_data in task_metrics.items():
            metrics = task_data['metrics']
            score = self._calculate_performance_score(metrics)
            
            if score >= 0.8:
                categories['优秀'].append(task_type)
            elif score >= 0.6:
                categories['良好'].append(task_type)
            elif score >= 0.4:
                categories['一般'].append(task_type)
            else:
                categories['需要改进'].append(task_type)
        
        return categories
    
    def _analyze_metric_correlations(self, task_metrics: Dict[str, Dict]) -> Dict[str, float]:
        """分析指标相关性"""
        # 收集指标数据
        metric_data = defaultdict(list)
        
        for task_data in task_metrics.values():
            metrics = task_data['metrics']
            
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    metric_data[metric_name].append(value)
        
        # 计算相关性（简化版本）
        correlations = {}
        metric_names = list(metric_data.keys())
        
        for i, metric1 in enumerate(metric_names):
            for metric2 in metric_names[i+1:]:
                if len(metric_data[metric1]) == len(metric_data[metric2]):
                    corr = np.corrcoef(metric_data[metric1], metric_data[metric2])[0, 1]
                    correlations[f'{metric1}_vs_{metric2}'] = corr
        
        return correlations
    
    def _analyze_task_errors(self, predictions: List[Dict], task_type: str) -> Dict[str, Any]:
        """分析任务特定错误"""
        errors = {
            'total_samples': len(predictions),
            'error_types': defaultdict(int),
            'common_mistakes': []
        }
        
        for pred_data in predictions:
            prediction = pred_data.get('prediction', '')
            reference = pred_data.get('reference', '')
            
            # 简单的错误分析
            if prediction != reference:
                # 长度差异
                if len(prediction) < len(reference) * 0.5:
                    errors['error_types']['too_short'] += 1
                elif len(prediction) > len(reference) * 1.5:
                    errors['error_types']['too_long'] += 1
                
                # 空预测
                if not prediction.strip():
                    errors['error_types']['empty_prediction'] += 1
                
                # 格式错误（针对特定任务）
                if task_type in ['object_detection', 'phrase_grounding']:
                    if '<loc_' not in prediction:
                        errors['error_types']['missing_location'] += 1
        
        return errors
    
    def _calculate_difficulty_score(self, metrics: Dict[str, float]) -> float:
        """计算难度分数（0-1，1表示最难）"""
        # 基于性能指标推断难度
        performance_score = self._calculate_performance_score(metrics)
        
        # 难度与性能成反比
        difficulty_score = 1.0 - performance_score
        
        return max(0.0, min(1.0, difficulty_score))
    
    def _calculate_performance_score(self, metrics: Dict[str, float]) -> float:
        """计算综合性能分数"""
        # 定义关键指标权重
        key_metrics = {
            'accuracy': 0.3,
            'f1': 0.25,
            'bleu': 0.2,
            'precision': 0.15,
            'recall': 0.15,
            'rouge1_f1': 0.2,
            'mean_iou': 0.25,
            'character_accuracy': 0.3,
            'word_accuracy': 0.25
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for metric_name, weight in key_metrics.items():
            if metric_name in metrics:
                value = metrics[metric_name]
                if isinstance(value, (int, float)) and 0 <= value <= 1:
                    weighted_score += value * weight
                    total_weight += weight
        
        if total_weight > 0:
            return weighted_score / total_weight
        else:
            # 如果没有关键指标，使用所有可用指标的平均值
            valid_values = []
            for value in metrics.values():
                if isinstance(value, (int, float)) and 0 <= value <= 1:
                    valid_values.append(value)
            
            return np.mean(valid_values) if valid_values else 0.0
    
    def _extract_key_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """提取关键指标"""
        key_metric_names = [
            'accuracy', 'f1', 'bleu', 'precision', 'recall',
            'rouge1_f1', 'mean_iou', 'character_accuracy', 'word_accuracy'
        ]
        
        key_metrics = {}
        for metric_name in key_metric_names:
            if metric_name in metrics:
                key_metrics[metric_name] = metrics[metric_name]
        
        return key_metrics
    
    def _generate_recommendations(self) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        # 基于任务性能分析生成建议
        task_analysis = self.analyze_task_performance()
        
        if 'performance_categories' in task_analysis:
            categories = task_analysis['performance_categories']
            
            if categories.get('需要改进'):
                recommendations.append(
                    f"重点关注以下需要改进的任务: {', '.join(categories['需要改进'][:3])}"
                )
            
            if len(categories.get('优秀', [])) < len(categories.get('需要改进', [])):
                recommendations.append("考虑增加训练数据或调整模型架构以提升整体性能")
        
        # 基于样本难度分析生成建议
        difficulty_analysis = self.analyze_sample_difficulty()
        
        if difficulty_analysis.get('challenging_tasks'):
            recommendations.append(
                f"对于挑战性任务 {', '.join(difficulty_analysis['challenging_tasks'][:2])}，"
                "建议增加训练时间或使用更复杂的模型结构"
            )
        
        # 基于评估信息生成建议
        eval_info = self.evaluation_results.get('evaluation_info', {})
        task_counts = eval_info.get('task_sample_counts', {})
        
        if task_counts:
            min_samples = min(task_counts.values())
            max_samples = max(task_counts.values())
            
            if max_samples > min_samples * 3:
                recommendations.append("数据分布不均衡，建议平衡各任务的训练样本数量")
        
        return recommendations
    
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
    
    def diagnose_performance_bottlenecks(self) -> Dict[str, Any]:
        """性能瓶颈诊断
        
        Returns:
            性能瓶颈诊断结果
        """
        diagnosis = {
            'timestamp': datetime.now().isoformat(),
            'system_info': self._get_system_info(),
            'task_bottlenecks': {},
            'metric_bottlenecks': {},
            'resource_bottlenecks': {},
            'recommendations': []
        }
        
        task_metrics = self.evaluation_results.get('task_metrics', {})
        
        # 分析任务级别瓶颈
        task_performance = []
        for task_type, task_data in task_metrics.items():
            metrics = task_data['metrics']
            score = self._calculate_performance_score(metrics)
            sample_count = task_data.get('sample_count', 0)
            
            task_performance.append({
                'task': task_type,
                'score': score,
                'sample_count': sample_count,
                'metrics': metrics
            })
        
        # 识别性能最差的任务
        task_performance.sort(key=lambda x: x['score'])
        worst_tasks = task_performance[:3]
        
        diagnosis['task_bottlenecks'] = {
            'worst_performing_tasks': [
                {
                    'task': task['task'],
                    'score': task['score'],
                    'issues': self._identify_task_issues(task)
                }
                for task in worst_tasks
            ]
        }
        
        # 分析指标级别瓶颈
        metric_analysis = self._analyze_metric_bottlenecks(task_metrics)
        diagnosis['metric_bottlenecks'] = metric_analysis
        
        # 分析资源瓶颈
        resource_analysis = self._analyze_resource_bottlenecks()
        diagnosis['resource_bottlenecks'] = resource_analysis
        
        # 生成改进建议
        diagnosis['recommendations'] = self._generate_bottleneck_recommendations(
            diagnosis
        )
        
        return diagnosis
    
    def assess_data_quality(self) -> Dict[str, Any]:
        """数据质量评估
        
        Returns:
            数据质量评估结果
        """
        quality_assessment = {
            'timestamp': datetime.now().isoformat(),
            'overall_quality_score': 0.0,
            'task_quality': {},
            'data_distribution': {},
            'quality_issues': [],
            'recommendations': []
        }
        
        task_metrics = self.evaluation_results.get('task_metrics', {})
        
        task_quality_scores = []
        
        for task_type, task_data in task_metrics.items():
            predictions = task_data.get('predictions', [])
            sample_count = task_data.get('sample_count', 0)
            
            # 评估任务数据质量
            task_quality = self._assess_task_data_quality(
                predictions, task_type, sample_count
            )
            
            quality_assessment['task_quality'][task_type] = task_quality
            task_quality_scores.append(task_quality['quality_score'])
        
        # 计算整体质量分数
        if task_quality_scores:
            quality_assessment['overall_quality_score'] = np.mean(task_quality_scores)
        
        # 分析数据分布
        distribution_analysis = self._analyze_data_distribution(task_metrics)
        quality_assessment['data_distribution'] = distribution_analysis
        
        # 识别质量问题
        quality_issues = self._identify_quality_issues(
            quality_assessment['task_quality'],
            distribution_analysis
        )
        quality_assessment['quality_issues'] = quality_issues
        
        # 生成改进建议
        quality_assessment['recommendations'] = self._generate_quality_recommendations(
            quality_assessment
        )
        
        return quality_assessment
    
    def analyze_model_behavior(self) -> Dict[str, Any]:
        """模型行为分析
        
        Returns:
            模型行为分析结果
        """
        behavior_analysis = {
            'timestamp': datetime.now().isoformat(),
            'prediction_patterns': {},
            'confidence_analysis': {},
            'bias_detection': {},
            'consistency_analysis': {},
            'behavioral_insights': []
        }
        
        task_metrics = self.evaluation_results.get('task_metrics', {})
        
        # 分析预测模式
        prediction_patterns = self._analyze_prediction_patterns(task_metrics)
        behavior_analysis['prediction_patterns'] = prediction_patterns
        
        # 分析置信度分布
        confidence_analysis = self._analyze_confidence_distribution(task_metrics)
        behavior_analysis['confidence_analysis'] = confidence_analysis
        
        # 检测偏见
        bias_detection = self._detect_model_bias(task_metrics)
        behavior_analysis['bias_detection'] = bias_detection
        
        # 一致性分析
        consistency_analysis = self._analyze_prediction_consistency(task_metrics)
        behavior_analysis['consistency_analysis'] = consistency_analysis
        
        # 生成行为洞察
        behavioral_insights = self._generate_behavioral_insights(behavior_analysis)
        behavior_analysis['behavioral_insights'] = behavioral_insights
        
        return behavior_analysis
    
    def _classify_error_type(self, prediction: str, reference: str, task_type: str) -> str:
        """分类错误类型"""
        if not prediction.strip():
            return 'empty_prediction'
        
        if len(prediction) < len(reference) * 0.3:
            return 'too_short'
        
        if len(prediction) > len(reference) * 2.0:
            return 'too_long'
        
        if task_type in ['object_detection', 'phrase_grounding']:
            if '<loc_' not in prediction and '<loc_' in reference:
                return 'missing_location'
            if prediction.count('<loc_') != reference.count('<loc_'):
                return 'location_count_mismatch'
        
        if task_type == 'ocr':
            # 计算字符级别差异
            char_diff = abs(len(prediction) - len(reference)) / max(len(reference), 1)
            if char_diff > 0.5:
                return 'character_mismatch'
        
        return 'content_mismatch'
    
    def _extract_error_features(self, prediction: str, reference: str, task_type: str) -> List[float]:
        """提取错误特征用于聚类"""
        features = []
        
        # 长度特征
        features.append(len(prediction))
        features.append(len(reference))
        features.append(abs(len(prediction) - len(reference)))
        features.append(len(prediction) / max(len(reference), 1))
        
        # 内容特征
        features.append(1 if prediction.strip() == '' else 0)  # 空预测
        features.append(prediction.count(' '))  # 空格数量
        features.append(prediction.count('\n'))  # 换行数量
        
        # 任务特定特征
        if task_type in ['object_detection', 'phrase_grounding']:
            features.append(prediction.count('<loc_'))  # 位置标记数量
            features.append(reference.count('<loc_'))
        else:
            features.extend([0, 0])
        
        # 字符级别特征
        common_chars = set(prediction.lower()) & set(reference.lower())
        features.append(len(common_chars))
        features.append(len(common_chars) / max(len(set(reference.lower())), 1))
        
        return features
    
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
    
    def _get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        system_info = {
            'timestamp': datetime.now().isoformat()
        }

        psutil = _load_psutil()
        if psutil:
            try:
                system_info.update({
                    'cpu_percent': psutil.cpu_percent(interval=1),
                    'memory_percent': psutil.virtual_memory().percent,
                    'disk_usage': psutil.disk_usage('/').percent,
                    'cpu_count': psutil.cpu_count(),
                    'memory_total_gb': psutil.virtual_memory().total / (1024**3)
                })
            except Exception as e:
                logger.warning(f"获取系统信息失败: {e}")
        
        return system_info
    
    def _identify_task_issues(self, task_info: Dict[str, Any]) -> List[str]:
        """识别任务问题"""
        issues = []
        
        score = task_info['score']
        metrics = task_info['metrics']
        sample_count = task_info['sample_count']
        
        if score < 0.3:
            issues.append('性能严重不足')
        
        if sample_count < 50:
            issues.append('训练样本不足')
        
        # 检查特定指标
        if 'accuracy' in metrics and metrics['accuracy'] < 0.5:
            issues.append('准确率过低')
        
        if 'f1' in metrics and metrics['f1'] < 0.4:
            issues.append('F1分数过低')
        
        if 'bleu' in metrics and metrics['bleu'] < 0.2:
            issues.append('BLEU分数过低')
        
        return issues
    
    def _analyze_metric_bottlenecks(self, task_metrics: Dict[str, Dict]) -> Dict[str, Any]:
        """分析指标瓶颈"""
        metric_scores = defaultdict(list)
        
        # 收集所有指标分数
        for task_data in task_metrics.values():
            metrics = task_data['metrics']
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    metric_scores[metric_name].append(value)
        
        # 分析每个指标的表现
        bottlenecks = {}
        for metric_name, scores in metric_scores.items():
            if scores:
                avg_score = np.mean(scores)
                min_score = np.min(scores)
                std_score = np.std(scores)
                
                bottlenecks[metric_name] = {
                    'average': avg_score,
                    'minimum': min_score,
                    'std_deviation': std_score,
                    'is_bottleneck': avg_score < 0.5 or min_score < 0.2
                }
        
        return bottlenecks
    
    def _analyze_resource_bottlenecks(self) -> Dict[str, Any]:
        """分析资源瓶颈"""
        resource_analysis = {}

        psutil = _load_psutil()
        if psutil:
            try:
                # CPU分析
                cpu_percent = psutil.cpu_percent(interval=1)
                resource_analysis['cpu'] = {
                    'usage_percent': cpu_percent,
                    'is_bottleneck': cpu_percent > 80
                }
                
                # 内存分析
                memory = psutil.virtual_memory()
                resource_analysis['memory'] = {
                    'usage_percent': memory.percent,
                    'available_gb': memory.available / (1024**3),
                    'is_bottleneck': memory.percent > 85
                }
                
                # 磁盘分析
                disk = psutil.disk_usage('/')
                resource_analysis['disk'] = {
                    'usage_percent': disk.percent,
                    'free_gb': disk.free / (1024**3),
                    'is_bottleneck': disk.percent > 90
                }
                
            except Exception as e:
                logger.warning(f"资源分析失败: {e}")
                resource_analysis['error'] = str(e)
        
        return resource_analysis
    
    def _generate_bottleneck_recommendations(self, diagnosis: Dict[str, Any]) -> List[str]:
        """生成瓶颈改进建议"""
        recommendations = []
        
        # 任务瓶颈建议
        task_bottlenecks = diagnosis.get('task_bottlenecks', {})
        worst_tasks = task_bottlenecks.get('worst_performing_tasks', [])
        
        for task_info in worst_tasks:
            task_name = task_info['task']
            issues = task_info['issues']
            
            if '性能严重不足' in issues:
                recommendations.append(f"任务{task_name}性能严重不足，建议重新设计模型架构")
            
            if '训练样本不足' in issues:
                recommendations.append(f"任务{task_name}训练样本不足，建议增加数据或使用数据增强")
        
        # 指标瓶颈建议
        metric_bottlenecks = diagnosis.get('metric_bottlenecks', {})
        for metric_name, metric_info in metric_bottlenecks.items():
            if metric_info.get('is_bottleneck'):
                recommendations.append(f"指标{metric_name}表现不佳，建议针对性优化")
        
        # 资源瓶颈建议
        resource_bottlenecks = diagnosis.get('resource_bottlenecks', {})
        
        if resource_bottlenecks.get('cpu', {}).get('is_bottleneck'):
            recommendations.append("CPU使用率过高，建议优化计算逻辑或增加计算资源")
        
        if resource_bottlenecks.get('memory', {}).get('is_bottleneck'):
            recommendations.append("内存使用率过高，建议优化内存管理或增加内存")
        
        return recommendations
