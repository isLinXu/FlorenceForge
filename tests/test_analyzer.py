"""ResultAnalyzer 与 scoring 模块冒烟测试。"""

import json

import pytest

from florence_forge.evaluation.analyzer import ResultAnalyzer
from florence_forge.evaluation import analyzer_scoring


@pytest.fixture
def sample_results():
    return {
        "overall_metrics": {"accuracy": 0.9},
        "task_metrics": {
            "CAPTION": {
                "metrics": {"accuracy": 0.9, "f1": 0.85},
                "sample_count": 100,
            },
            "OD": {
                "metrics": {"accuracy": 0.4, "f1": 0.35},
                "sample_count": 50,
            },
        },
        "evaluation_info": {"task_sample_counts": {"CAPTION": 100, "OD": 50}},
    }


def test_calculate_performance_score():
    score = analyzer_scoring.calculate_performance_score({"accuracy": 0.8, "f1": 0.7})
    assert 0.0 <= score <= 1.0


def test_analyze_task_performance_ranking(sample_results):
    analyzer = ResultAnalyzer(sample_results)
    analysis = analyzer.analyze_task_performance()
    assert analysis["task_ranking"][0][0] == "CAPTION"
    assert "需要改进" in analysis["performance_categories"]
    assert "OD" in analysis["performance_categories"]["需要改进"]


def test_generate_performance_report(tmp_path, sample_results):
    analyzer = ResultAnalyzer(sample_results)
    report = analyzer.generate_performance_report(tmp_path / "report.md")
    assert "CAPTION" in report
    assert (tmp_path / "report.md").exists()


def test_load_results_json(tmp_path, sample_results):
    path = tmp_path / "results.json"
    path.write_text(json.dumps(sample_results), encoding="utf-8")
    analyzer = ResultAnalyzer()
    analyzer.load_results(path)
    assert "CAPTION" in analyzer.evaluation_results["task_metrics"]


def test_diagnose_performance_bottlenecks(sample_results):
    from florence_forge.evaluation.analyzer_diagnostics import (
        diagnose_performance_bottlenecks,
    )

    diagnosis = diagnose_performance_bottlenecks(sample_results)
    assert "task_bottlenecks" in diagnosis
    assert diagnosis["task_bottlenecks"]["worst_performing_tasks"][0]["task"] == "OD"


def test_cluster_error_patterns_with_mismatches(sample_results):
    sample_results["task_metrics"]["CAPTION"]["predictions"] = [
        {"prediction": "wrong", "reference": "right answer here"}
    ]
    analyzer = ResultAnalyzer(sample_results)
    result = analyzer.cluster_error_patterns(n_clusters=1)
    assert result.get("total_errors", 0) >= 1 or "message" in result


def test_assess_data_quality(sample_results):
    analyzer = ResultAnalyzer(sample_results)
    quality = analyzer.assess_data_quality()
    assert quality["overall_quality_score"] > 0
    assert "CAPTION" in quality["task_quality"]
