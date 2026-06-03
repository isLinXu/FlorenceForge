"""Benchmark result persistence and report generation helpers."""

from __future__ import annotations

import json
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Dict, Union

from .benchmark_pdf_report import generate_pdf_report as _generate_pdf_report


def save_benchmark_results(
    results: Dict[str, Any],
    output_dir: Union[str, Path],
    save_detailed: bool,
) -> None:
    """Save benchmark results, summary, and optional per-dataset details."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    summary = {
        "benchmark_info": results["benchmark_info"],
        "overall_summary": results["overall_summary"],
        "task_performance": results["task_performance"],
    }
    with open(output_path / "benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if save_detailed:
        detailed_dir = output_path / "detailed_results"
        detailed_dir.mkdir(exist_ok=True)
        for dataset_name, dataset_result in results.get("dataset_results", {}).items():
            with open(
                detailed_dir / f"{dataset_name}_detailed.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(dataset_result, f, indent=2, ensure_ascii=False)


def generate_benchmark_report(
    results: Dict[str, Any],
    output_file: Union[str, Path],
    format: str = "markdown",
    *,
    enable_distributed: bool = False,
    enable_incremental: bool = False,
) -> None:
    """Generate a benchmark report in the requested format."""
    output_path = Path(output_file)
    if format == "markdown":
        generate_markdown_report(results, output_path)
    elif format == "html":
        generate_html_report(results, output_path)
    elif format == "json":
        generate_json_report(
            results,
            output_path,
            enable_distributed=enable_distributed,
            enable_incremental=enable_incremental,
        )
    elif format == "pdf":
        generate_pdf_report(
            results,
            output_path,
            enable_distributed=enable_distributed,
            enable_incremental=enable_incremental,
        )
    else:
        raise ValueError(f"不支持的报告格式: {format}")


def generate_markdown_report(results: Dict[str, Any], output_file: Union[str, Path]) -> None:
    """Generate a Markdown benchmark report."""
    output_path = Path(output_file)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Benchmark评估报告\n\n")

        benchmark_info = results.get("benchmark_info", {})
        f.write("## 基本信息\n\n")
        f.write(f"- 评估时间: {benchmark_info.get('timestamp', 'N/A')}\n")
        f.write(f"- 总耗时: {benchmark_info.get('total_evaluation_time', 0):.2f}秒\n")
        f.write(f"- 数据集数量: {len(results.get('task_performance', {}))}\n\n")

        if "statistical_summary" in results:
            stats = results["statistical_summary"]
            f.write("## 统计摘要\n\n")
            f.write(f"- 计算指标总数: {stats.get('total_metrics_computed', 0)}\n")
            f.write(f"- 平均性能分数: {stats.get('average_performance_score', 0):.4f}\n")
            f.write(f"- 性能一致性: {stats.get('performance_consistency', 0):.4f}\n")
            f.write(f"- 评估效率: {stats.get('evaluation_efficiency', 0):.2f} 样本/秒\n\n")

        overall = results.get("overall_summary", {})
        f.write("## 总体摘要\n\n")
        f.write(f"- 总样本数: {overall.get('total_samples', 0)}\n")
        f.write(f"- 平均准确率: {overall.get('average_accuracy', 0):.4f}\n")
        f.write(f"- 平均F1分数: {overall.get('average_f1', 0):.4f}\n\n")

        if "performance_analysis" in results:
            perf_analysis = results["performance_analysis"]
            f.write("## 性能分析\n\n")

            best_tasks = perf_analysis.get("best_performing_tasks", [])
            if best_tasks:
                f.write("### 最佳表现任务\n\n")
                for i, (task, score, std) in enumerate(best_tasks, 1):
                    f.write(f"{i}. {task}: {score:.4f} (+/-{std:.4f})\n")
                f.write("\n")

            worst_tasks = perf_analysis.get("worst_performing_tasks", [])
            if worst_tasks:
                f.write("### 需要改进的任务\n\n")
                for i, (task, score, std) in enumerate(worst_tasks, 1):
                    f.write(f"{i}. {task}: {score:.4f} (+/-{std:.4f})\n")
                f.write("\n")

        if "resource_analysis" in results:
            resource_analysis = results["resource_analysis"]
            f.write("## 资源使用分析\n\n")

            cpu_stats = resource_analysis.get("cpu_stats", {})
            if cpu_stats:
                f.write("### CPU使用情况\n\n")
                f.write(f"- 平均使用率: {cpu_stats.get('mean', 0):.2f}%\n")
                f.write(f"- 最高使用率: {cpu_stats.get('max', 0):.2f}%\n")
                f.write(f"- 使用率标准差: {cpu_stats.get('std', 0):.2f}%\n\n")

            memory_stats = resource_analysis.get("memory_stats", {})
            if memory_stats:
                f.write("### 内存使用情况\n\n")
                f.write(f"- 平均使用率: {memory_stats.get('mean', 0):.2f}%\n")
                f.write(f"- 最高使用率: {memory_stats.get('max', 0):.2f}%\n")
                f.write(f"- 使用率标准差: {memory_stats.get('std', 0):.2f}%\n\n")

            gpu_stats = resource_analysis.get("gpu_stats", {})
            if gpu_stats:
                f.write("### GPU使用情况\n\n")
                f.write(f"- 平均负载: {gpu_stats.get('load_mean', 0):.2f}%\n")
                f.write(f"- 最高负载: {gpu_stats.get('load_max', 0):.2f}%\n")
                f.write(f"- 平均显存使用: {gpu_stats.get('memory_mean', 0):.2f}%\n")
                f.write(f"- 最高显存使用: {gpu_stats.get('memory_max', 0):.2f}%\n\n")

            bottlenecks = resource_analysis.get("resource_bottlenecks", [])
            if bottlenecks:
                f.write("### 资源瓶颈\n\n")
                for bottleneck in bottlenecks:
                    f.write(f"- {bottleneck}\n")
                f.write("\n")

            efficiency = resource_analysis.get("efficiency_score", 0)
            f.write(f"### 整体效率分数: {efficiency:.4f}\n\n")

        recommendations = results.get("optimization_recommendations", [])
        if recommendations:
            f.write("## 优化建议\n\n")
            for i, rec in enumerate(recommendations, 1):
                f.write(f"{i}. {rec}\n")
            f.write("\n")

        f.write("## 任务性能详情\n\n")
        for task_type, perf in results.get("task_performance", {}).items():
            f.write(f"### {task_type}\n\n")
            f.write(f"- 样本数: {perf.get('sample_count', 0)}\n")

            for metric_name, stats in perf.get("average_metrics", {}).items():
                if isinstance(stats, dict):
                    f.write(
                        f"- {metric_name}: {stats.get('mean', 0):.4f} "
                        f"(+/-{stats.get('std', 0):.4f})\n"
                    )
                elif isinstance(stats, (int, float)):
                    f.write(f"- {metric_name}: {stats:.4f}\n")
                else:
                    f.write(f"- {metric_name}: {stats}\n")
            f.write("\n")


def generate_html_report(results: Dict[str, Any], output_file: Union[str, Path]) -> None:
    """Generate an HTML benchmark report."""
    output_path = Path(output_file)
    benchmark_info = results.get("benchmark_info", {})
    overall = results.get("overall_summary", {})
    task_performance = results.get("task_performance", {})

    statistical_summary = _render_statistical_summary_html(results)
    performance_analysis = _render_performance_analysis_html(results)
    resource_analysis = _render_resource_analysis_html(results)
    optimization_recommendations = _render_recommendations_html(results)
    task_rows = _render_task_rows_html(task_performance)

    style = """
        body { font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }
        .section { margin: 30px 0; padding: 20px; border-left: 4px solid #667eea; background-color: #f8f9fa; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }
        .stat-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); text-align: center; }
        .stat-value { font-size: 2em; font-weight: bold; color: #667eea; }
        .stat-label { color: #666; margin-top: 5px; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        th, td { padding: 15px; text-align: left; border-bottom: 1px solid #eee; }
        th { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; font-weight: 600; }
        tr:hover { background-color: #f8f9fa; }
        .recommendation { background: #e8f5e8; border-left: 4px solid #28a745; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .bottleneck { background: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .progress-bar { width: 100%; height: 20px; background-color: #e0e0e0; border-radius: 10px; overflow: hidden; }
        .progress-fill { height: 100%; background: linear-gradient(90deg, #28a745, #20c997); transition: width 0.3s ease; }
    """
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Benchmark评估报告</title>
    <meta charset="UTF-8">
    <style>{style}</style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Benchmark评估报告</h1>
            <p><strong>评估时间:</strong> {escape(str(benchmark_info.get('timestamp', 'N/A')))}</p>
            <p><strong>总耗时:</strong> {benchmark_info.get('total_evaluation_time', 0):.2f}秒</p>
        </div>

        {statistical_summary}

        <div class="section">
            <h2>总体摘要</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{overall.get('total_samples', 0)}</div>
                    <div class="stat-label">总样本数</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{overall.get('average_accuracy', 0):.4f}</div>
                    <div class="stat-label">平均准确率</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{overall.get('average_f1', 0):.4f}</div>
                    <div class="stat-label">平均F1分数</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{len(task_performance)}</div>
                    <div class="stat-label">数据集数量</div>
                </div>
            </div>
        </div>

        {performance_analysis}
        {resource_analysis}
        {optimization_recommendations}

        <div class="section">
            <h2>任务性能详情</h2>
            <table>
                <tr>
                    <th>任务类型</th>
                    <th>样本数</th>
                    <th>主要指标</th>
                    <th>性能评分</th>
                </tr>
                {task_rows}
            </table>
        </div>
    </div>
</body>
</html>
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)


def generate_json_report(
    results: Dict[str, Any],
    output_file: Union[str, Path],
    *,
    enable_distributed: bool = False,
    enable_incremental: bool = False,
) -> None:
    """Generate a JSON benchmark report."""
    json_data = {
        "benchmark_info": results.get("benchmark_info", {}),
        "overall_summary": results.get("overall_summary", {}),
        "task_performance": results.get("task_performance", {}),
        "statistical_summary": results.get("statistical_summary", {}),
        "performance_analysis": results.get("performance_analysis", {}),
        "resource_analysis": results.get("resource_analysis", {}),
        "optimization_recommendations": results.get("optimization_recommendations", []),
        "monitoring_data": results.get("monitoring_data", {}),
        "metadata": {
            "report_generated_at": datetime.now().isoformat(),
            "report_version": "1.0",
            "florence_forge_version": "1.0.0",
            "total_datasets_evaluated": len(results.get("task_performance", {})),
            "evaluation_mode": "distributed" if enable_distributed else "single_gpu",
            "cache_enabled": enable_incremental,
        },
    }

    with open(Path(output_file), "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)


def generate_pdf_report(
    results: Dict[str, Any],
    output_file: Union[str, Path],
    *,
    enable_distributed: bool = False,
    enable_incremental: bool = False,
) -> None:
    """Generate a PDF benchmark report when reportlab is installed."""
    _generate_pdf_report(
        results,
        output_file,
        enable_distributed=enable_distributed,
        enable_incremental=enable_incremental,
    )


def _render_statistical_summary_html(results: Dict[str, Any]) -> str:
    if "statistical_summary" not in results:
        return ""
    stats = results["statistical_summary"]
    return f"""
        <div class="section">
            <h2>统计摘要</h2>
            <div class="stats-grid">
                <div class="stat-card"><div class="stat-value">{stats.get('total_metrics_computed', 0)}</div><div class="stat-label">计算指标总数</div></div>
                <div class="stat-card"><div class="stat-value">{stats.get('average_performance_score', 0):.4f}</div><div class="stat-label">平均性能分数</div></div>
                <div class="stat-card"><div class="stat-value">{stats.get('performance_consistency', 0):.4f}</div><div class="stat-label">性能一致性</div></div>
                <div class="stat-card"><div class="stat-value">{stats.get('evaluation_efficiency', 0):.2f}</div><div class="stat-label">评估效率 (样本/秒)</div></div>
            </div>
        </div>
    """


def _render_performance_analysis_html(results: Dict[str, Any]) -> str:
    if "performance_analysis" not in results:
        return ""

    perf_analysis = results["performance_analysis"]
    best_html = _task_score_list_html(perf_analysis.get("best_performing_tasks", []))
    worst_html = _task_score_list_html(perf_analysis.get("worst_performing_tasks", []))
    if not best_html and not worst_html:
        return ""
    return f"""
        <div class="section">
            <h2>性能分析</h2>
            {'<h3>最佳表现任务</h3>' + best_html if best_html else ''}
            {'<h3>需要改进的任务</h3>' + worst_html if worst_html else ''}
        </div>
    """


def _render_resource_analysis_html(results: Dict[str, Any]) -> str:
    if "resource_analysis" not in results:
        return ""

    res_analysis = results["resource_analysis"]
    cards = ""
    for label, stats_key, value_key in (
        ("平均CPU使用率", "cpu_stats", "mean"),
        ("平均内存使用率", "memory_stats", "mean"),
        ("平均GPU负载", "gpu_stats", "load_mean"),
    ):
        stats = res_analysis.get(stats_key, {})
        if stats:
            value = stats.get(value_key, 0)
            cards += f"""
                <div class="stat-card">
                    <div class="stat-value">{value:.1f}%</div>
                    <div class="stat-label">{label}</div>
                    <div class="progress-bar"><div class="progress-fill" style="width: {value}%"></div></div>
                </div>
            """

    cards += f"""
        <div class="stat-card">
            <div class="stat-value">{res_analysis.get('efficiency_score', 0):.3f}</div>
            <div class="stat-label">整体效率分数</div>
        </div>
    """

    bottleneck_html = ""
    bottlenecks = res_analysis.get("resource_bottlenecks", [])
    if bottlenecks:
        bottleneck_html = "<h3>资源瓶颈</h3>" + "".join(
            f'<div class="bottleneck">{escape(str(bottleneck))}</div>'
            for bottleneck in bottlenecks
        )

    return f"""
        <div class="section">
            <h2>资源使用分析</h2>
            <div class="stats-grid">{cards}</div>
            {bottleneck_html}
        </div>
    """


def _render_recommendations_html(results: Dict[str, Any]) -> str:
    recommendations = results.get("optimization_recommendations", [])
    if not recommendations:
        return ""
    rec_html = "".join(
        f'<div class="recommendation">{i}. {escape(str(rec))}</div>'
        for i, rec in enumerate(recommendations, 1)
    )
    return f"""
        <div class="section">
            <h2>优化建议</h2>
            {rec_html}
        </div>
    """


def _render_task_rows_html(task_performance: Dict[str, Any]) -> str:
    rows = ""
    for task_type, perf in task_performance.items():
        avg_metrics = perf.get("average_metrics", {})
        main_metrics = []
        performance_score = 0.0
        for metric_name, stats in avg_metrics.items():
            if isinstance(stats, dict) and "mean" in stats:
                main_metrics.append(f"{escape(str(metric_name))}: {stats['mean']:.4f}")
                performance_score += stats["mean"]
            elif isinstance(stats, (int, float)):
                main_metrics.append(f"{escape(str(metric_name))}: {stats:.4f}")
                performance_score += stats
        if avg_metrics:
            performance_score /= len(avg_metrics)
        rows += f"""
            <tr>
                <td><strong>{escape(str(task_type))}</strong></td>
                <td>{perf.get('sample_count', 0):,}</td>
                <td>{'<br>'.join(main_metrics[:3])}</td>
                <td>{performance_score:.4f}</td>
            </tr>
        """
    return rows


def _task_score_list_html(tasks: Any) -> str:
    if not tasks:
        return ""
    items = "".join(
        f"<li>{escape(str(task))}: {score:.4f} (+/-{std:.4f})</li>"
        for task, score, std in tasks
    )
    return f"<ul>{items}</ul>"
