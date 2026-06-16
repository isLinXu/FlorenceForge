"""PDF report generation for benchmark results."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Union

logger = logging.getLogger(__name__)


def generate_pdf_report(
    results: Dict[str, Any],
    output_file: Union[str, Path],
    *,
    enable_distributed: bool = False,
    enable_incremental: bool = False,
) -> None:
    """Generate a PDF benchmark report when reportlab is installed."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            PageBreak,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError:
        logger.warning("无法导入reportlab库，PDF报告生成功能不可用")
        logger.info("请安装reportlab: pip install reportlab")
        return

    doc = SimpleDocTemplate(str(output_file), pagesize=A4)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Heading1"],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.darkblue,
    )
    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading2"],
        fontSize=16,
        spaceAfter=12,
        textColor=colors.darkblue,
    )

    story = [Paragraph("Benchmark评估报告", title_style), Spacer(1, 20)]
    benchmark_info = results.get("benchmark_info", {})
    story.append(
        _styled_table(
            [
                ["评估时间", benchmark_info.get("timestamp", "N/A")],
                ["总耗时", f"{benchmark_info.get('total_evaluation_time', 0):.2f}秒"],
                ["评估模式", "分布式" if enable_distributed else "单GPU"],
                ["缓存启用", "是" if enable_incremental else "否"],
            ],
            col_widths=[2 * inch, 3 * inch],
            header_color=colors.lightblue,
            body_color=colors.beige,
            colors=colors,
            Table=Table,
            TableStyle=TableStyle,
        )
    )
    story.append(Spacer(1, 30))

    story.append(Paragraph("总体摘要", heading_style))
    overall = results.get("overall_summary", {})
    story.append(
        _styled_table(
            [
                ["指标", "数值"],
                ["总样本数", f"{overall.get('total_samples', 0):,}"],
                ["平均准确率", f"{overall.get('average_accuracy', 0):.4f}"],
                ["平均F1分数", f"{overall.get('average_f1', 0):.4f}"],
                ["数据集数量", str(len(results.get("task_performance", {})))],
            ],
            col_widths=[2.5 * inch, 2.5 * inch],
            header_color=colors.darkblue,
            body_color=colors.lightgrey,
            colors=colors,
            Table=Table,
            TableStyle=TableStyle,
        )
    )
    story.append(Spacer(1, 20))

    if "statistical_summary" in results:
        stats = results["statistical_summary"]
        story.append(Paragraph("统计摘要", heading_style))
        story.append(
            _styled_table(
                [
                    ["统计项", "数值"],
                    ["计算指标总数", str(stats.get("total_metrics_computed", 0))],
                    ["平均性能分数", f"{stats.get('average_performance_score', 0):.4f}"],
                    ["性能一致性", f"{stats.get('performance_consistency', 0):.4f}"],
                    ["评估效率 (样本/秒)", f"{stats.get('evaluation_efficiency', 0):.2f}"],
                ],
                col_widths=[3 * inch, 2 * inch],
                header_color=colors.darkgreen,
                body_color=colors.lightgreen,
                colors=colors,
                Table=Table,
                TableStyle=TableStyle,
            )
        )
        story.append(Spacer(1, 20))

    if "performance_analysis" in results:
        story.append(Paragraph("性能分析", heading_style))
        perf_analysis = results["performance_analysis"]
        for title, key in (
            ("最佳表现任务:", "best_performing_tasks"),
            ("需要改进的任务:", "worst_performing_tasks"),
        ):
            tasks = perf_analysis.get(key, [])
            if tasks:
                story.append(Paragraph(title, styles["Heading3"]))
                for task, score, std in tasks[:5]:
                    story.append(Paragraph(f"- {task}: {score:.4f} (+/-{std:.4f})", styles["Normal"]))
                story.append(Spacer(1, 10))

    if "resource_analysis" in results:
        resource_data = _resource_pdf_rows(results["resource_analysis"])
        if len(resource_data) > 1:
            story.append(Paragraph("资源使用分析", heading_style))
            story.append(
                _styled_table(
                    resource_data,
                    col_widths=[1.5 * inch, 1.5 * inch, 1.5 * inch],
                    header_color=colors.orange,
                    body_color=colors.lightyellow,
                    colors=colors,
                    Table=Table,
                    TableStyle=TableStyle,
                )
            )
            story.append(Spacer(1, 20))

    recommendations = results.get("optimization_recommendations", [])
    if recommendations:
        story.append(Paragraph("优化建议", heading_style))
        for i, rec in enumerate(recommendations[:10], 1):
            story.append(Paragraph(f"{i}. {rec}", styles["Normal"]))
            story.append(Spacer(1, 5))
        story.append(Spacer(1, 20))

    story.append(PageBreak())
    story.append(Paragraph("任务性能详情", heading_style))
    task_data = _task_pdf_rows(results.get("task_performance", {}))
    if len(task_data) > 1:
        story.append(
            _styled_table(
                task_data,
                col_widths=[1.5 * inch, 1 * inch, 1 * inch, 1 * inch, 1 * inch],
                header_color=colors.purple,
                body_color=colors.lavender,
                colors=colors,
                Table=Table,
                TableStyle=TableStyle,
                body_font_size=9,
                header_font_size=10,
            )
        )

    doc.build(story)
    logger.info("PDF报告已生成: %s", output_file)


def _styled_table(
    data: Any,
    *,
    col_widths: Any,
    header_color: Any,
    body_color: Any,
    colors: Any,
    Table: Any,
    TableStyle: Any,
    header_font_size: int = 12,
    body_font_size: int = 12,
) -> Any:
    table = Table(data, colWidths=col_widths)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), header_color),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), header_font_size),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                ("BACKGROUND", (0, 1), (-1, -1), body_color),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ("FONTSIZE", (0, 1), (-1, -1), body_font_size),
            ]
        )
    )
    return table


def _resource_pdf_rows(resource_analysis: Dict[str, Any]) -> list:
    rows = [["资源类型", "平均使用率", "峰值使用率"]]
    cpu_stats = resource_analysis.get("cpu_stats", {})
    if cpu_stats:
        rows.append(["CPU", f"{cpu_stats.get('mean', 0):.1f}%", f"{cpu_stats.get('max', 0):.1f}%"])

    memory_stats = resource_analysis.get("memory_stats", {})
    if memory_stats:
        rows.append(
            ["内存", f"{memory_stats.get('mean', 0):.1f}%", f"{memory_stats.get('max', 0):.1f}%"]
        )

    gpu_stats = resource_analysis.get("gpu_stats", {})
    if gpu_stats:
        rows.append(["GPU", f"{gpu_stats.get('load_mean', 0):.1f}%", f"{gpu_stats.get('load_max', 0):.1f}%"])
    return rows


def _task_pdf_rows(task_performance: Dict[str, Any]) -> list:
    rows = [["任务类型", "样本数", "平均准确率", "平均F1分数", "性能评分"]]
    for task_type, perf in task_performance.items():
        avg_metrics = perf.get("average_metrics", {})
        accuracy = (
            avg_metrics.get("accuracy", {}).get("mean", 0)
            if isinstance(avg_metrics.get("accuracy"), dict)
            else avg_metrics.get("accuracy", 0)
        )
        f1 = (
            avg_metrics.get("f1", {}).get("mean", 0)
            if isinstance(avg_metrics.get("f1"), dict)
            else avg_metrics.get("f1", 0)
        )

        performance_score = 0.0
        metric_count = 0
        for stats in avg_metrics.values():
            if isinstance(stats, dict) and "mean" in stats:
                performance_score += stats["mean"]
                metric_count += 1
            elif isinstance(stats, (int, float)):
                performance_score += stats
                metric_count += 1
        if metric_count:
            performance_score /= metric_count

        rows.append(
            [
                task_type,
                f"{perf.get('sample_count', 0):,}",
                f"{accuracy:.4f}",
                f"{f1:.4f}",
                f"{performance_score:.4f}",
            ]
        )
    return rows
