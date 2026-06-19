from __future__ import annotations

"""FlorenceForge 结果分析器 — 可视化 Mixin

提供任务性能图表和指标比较图绘制。
"""

from pathlib import Path
from typing import List, Optional, Union

from .analyzer_base import logger, _load_plotting_dependencies


class PlotMixin:
    """可视化分析 Mixin"""

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
