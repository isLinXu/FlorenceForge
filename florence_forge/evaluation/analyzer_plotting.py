"""评估结果绘图（依赖 matplotlib / seaborn / pandas）。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

from . import analyzer_deps
from . import analyzer_scoring
from florence_forge.utils.plot_backend import finalize_matplotlib_figure

logger = logging.getLogger(__name__)


def plot_task_performance(
    evaluation_results: Dict,
    output_dir: Optional[Union[str, Path]] = None,
) -> None:
    try:
        plt, sns, _ = analyzer_deps.load_plotting_dependencies()
    except ImportError as exc:
        logger.warning("%s", exc)
        return

    plt.style.use("default")
    sns.set_palette("husl")

    task_metrics = evaluation_results.get("task_metrics", {})
    if not task_metrics:
        logger.warning("没有任务指标数据")
        return

    task_names = []
    performance_scores = []
    sample_counts = []
    for task_type, task_data in task_metrics.items():
        task_names.append(task_type)
        performance_scores.append(
            analyzer_scoring.calculate_performance_score(task_data["metrics"])
        )
        sample_counts.append(task_data["sample_count"])

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("任务性能分析", fontsize=16)
    axes[0, 0].bar(range(len(task_names)), performance_scores)
    axes[0, 0].set_title("任务性能分数")
    axes[0, 0].set_xticks(range(len(task_names)))
    axes[0, 0].set_xticklabels(task_names, rotation=45, ha="right")
    axes[0, 1].bar(range(len(task_names)), sample_counts, color="orange")
    axes[0, 1].set_title("样本数量分布")
    axes[0, 1].set_xticks(range(len(task_names)))
    axes[0, 1].set_xticklabels(task_names, rotation=45, ha="right")
    axes[1, 0].scatter(sample_counts, performance_scores, alpha=0.7)
    for index, task in enumerate(task_names):
        axes[1, 0].annotate(
            task,
            (sample_counts[index], performance_scores[index]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    axes[1, 1].hist(performance_scores, bins=10, alpha=0.7, color="green")
    plt.tight_layout()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "task_performance.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        logger.info("性能图表已保存到: %s", path)
    finalize_matplotlib_figure()


def plot_metric_comparison(
    evaluation_results: Dict,
    metric_names: List[str],
    output_dir: Optional[Union[str, Path]] = None,
) -> None:
    try:
        plt, sns, pd = analyzer_deps.load_plotting_dependencies()
    except ImportError as exc:
        logger.warning("%s", exc)
        return

    task_metrics = evaluation_results.get("task_metrics", {})
    if not task_metrics:
        logger.warning("没有任务指标数据")
        return

    rows = []
    for task_type, task_data in task_metrics.items():
        metrics = task_data["metrics"]
        for metric_name in metric_names:
            if metric_name in metrics:
                rows.append(
                    {"task": task_type, "metric": metric_name, "value": metrics[metric_name]}
                )
    if not rows:
        logger.warning("没有找到指定的指标数据")
        return

    df = pd.DataFrame(rows)
    plt.figure(figsize=(12, 8))
    sns.barplot(data=df, x="task", y="value", hue="metric")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "metric_comparison.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        logger.info("指标比较图已保存到: %s", path)
    finalize_matplotlib_figure()
