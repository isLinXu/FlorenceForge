"""Benchmark report writer regression tests."""

import json

import pytest

from florence_forge.evaluation.benchmark import BenchmarkEvaluator
from florence_forge.evaluation.benchmark_reports import (
    generate_benchmark_report,
    generate_html_report,
    generate_json_report,
    generate_markdown_report,
    save_benchmark_results,
)
from florence_forge.evaluation.benchmark_pdf_report import (
    _resource_pdf_rows,
    _task_pdf_rows,
    generate_pdf_report,
)


def _sample_results():
    return {
        "benchmark_info": {
            "timestamp": "2026-05-26T12:00:00",
            "total_evaluation_time": 12.5,
        },
        "overall_summary": {
            "total_samples": 3,
            "average_accuracy": 0.8,
            "average_f1": 0.75,
        },
        "task_performance": {
            "CAPTION": {
                "sample_count": 3,
                "average_metrics": {
                    "accuracy": {"mean": 0.8, "std": 0.1},
                    "f1": 0.75,
                },
            }
        },
        "dataset_results": {
            "coco-mini": {
                "predictions": ["a caption"],
                "metrics": {"accuracy": 0.8},
            }
        },
        "statistical_summary": {
            "total_metrics_computed": 2,
            "average_performance_score": 0.775,
            "performance_consistency": 0.9,
            "evaluation_efficiency": 4.2,
        },
        "performance_analysis": {
            "best_performing_tasks": [("CAPTION", 0.8, 0.1)],
            "worst_performing_tasks": [],
        },
        "resource_analysis": {
            "cpu_stats": {"mean": 10.0, "max": 20.0, "std": 2.0},
            "memory_stats": {"mean": 30.0, "max": 40.0, "std": 3.0},
            "resource_bottlenecks": ["CPU peak"],
            "efficiency_score": 0.82,
        },
        "optimization_recommendations": ["Increase batch size"],
        "monitoring_data": {"samples_per_second": 4.2},
    }


def test_save_benchmark_results_writes_summary_and_detailed_files(tmp_path):
    results = _sample_results()

    save_benchmark_results(results, tmp_path, save_detailed=True)

    full = json.loads((tmp_path / "benchmark_results.json").read_text(encoding="utf-8"))
    summary = json.loads((tmp_path / "benchmark_summary.json").read_text(encoding="utf-8"))
    detail = json.loads(
        (tmp_path / "detailed_results" / "coco-mini_detailed.json").read_text(
            encoding="utf-8"
        )
    )

    assert full["overall_summary"]["total_samples"] == 3
    assert set(summary) == {"benchmark_info", "overall_summary", "task_performance"}
    assert detail["metrics"]["accuracy"] == 0.8


def test_markdown_and_html_reports_are_written(tmp_path):
    results = _sample_results()
    markdown_path = tmp_path / "report.md"
    html_path = tmp_path / "report.html"

    generate_markdown_report(results, markdown_path)
    generate_html_report(results, html_path)

    markdown = markdown_path.read_text(encoding="utf-8")
    html = html_path.read_text(encoding="utf-8")

    assert "# Benchmark评估报告" in markdown
    assert "CAPTION" in markdown
    assert "<style>" in html
    assert "Benchmark评估报告" in html
    assert "CAPTION" in html


def test_json_report_includes_evaluator_mode_metadata(tmp_path):
    results = _sample_results()
    report_path = tmp_path / "report.json"

    generate_json_report(
        results,
        report_path,
        enable_distributed=True,
        enable_incremental=False,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["metadata"]["evaluation_mode"] == "distributed"
    assert report["metadata"]["cache_enabled"] is False
    assert report["metadata"]["total_datasets_evaluated"] == 1


def test_benchmark_evaluator_report_proxy_passes_runtime_flags(tmp_path):
    evaluator = BenchmarkEvaluator.__new__(BenchmarkEvaluator)
    evaluator.enable_distributed = False
    evaluator.enable_incremental = True
    report_path = tmp_path / "proxy.json"

    evaluator._generate_json_report(_sample_results(), report_path)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["metadata"]["evaluation_mode"] == "single_gpu"
    assert report["metadata"]["cache_enabled"] is True


def test_report_dispatcher_rejects_unknown_format(tmp_path):
    with pytest.raises(ValueError, match="不支持的报告格式"):
        generate_benchmark_report(_sample_results(), tmp_path / "report.txt", format="txt")


def test_pdf_report_is_rendered_when_reportlab_is_installed(tmp_path):
    pytest.importorskip("reportlab")
    results = _sample_results()
    results["resource_analysis"]["gpu_stats"] = {"load_mean": 50.0, "load_max": 70.0}
    results["performance_analysis"]["worst_performing_tasks"] = [("OCR", 0.4, 0.2)]
    output_path = tmp_path / "report.pdf"

    generate_pdf_report(
        results,
        output_path,
        enable_distributed=True,
        enable_incremental=True,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_pdf_table_rows_handle_scalar_and_structured_metrics():
    resource_rows = _resource_pdf_rows(
        {
            "cpu_stats": {"mean": 12.5, "max": 40.0},
            "memory_stats": {"mean": 22.0, "max": 45.0},
            "gpu_stats": {"load_mean": 31.0, "load_max": 55.0},
        }
    )
    task_rows = _task_pdf_rows(
        {
            "CAPTION": {
                "sample_count": 1200,
                "average_metrics": {
                    "accuracy": {"mean": 0.8},
                    "f1": 0.7,
                },
            }
        }
    )

    assert resource_rows[1:] == [
        ["CPU", "12.5%", "40.0%"],
        ["内存", "22.0%", "45.0%"],
        ["GPU", "31.0%", "55.0%"],
    ]
    assert task_rows[1] == ["CAPTION", "1,200", "0.8000", "0.7000", "0.7500"]
