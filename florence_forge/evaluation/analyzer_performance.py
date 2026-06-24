from __future__ import annotations

"""FlorenceForge 结果分析器 — 性能分析 Mixin

提供任务性能分析、样本难度分析和性能报告生成。
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

from .analyzer_base import logger


class PerformanceMixin:
    """性能分析 Mixin"""

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
