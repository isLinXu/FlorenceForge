from __future__ import annotations

"""FlorenceForge 结果分析器 — 诊断与瓶颈分析 Mixin

提供性能瓶颈诊断、数据质量评估、模型行为分析以及底层诊断辅助方法。
"""

from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List

import numpy as np

from .analyzer_base import logger, _load_psutil


class DiagnosisMixin:
    """诊断与瓶颈分析 Mixin"""

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
