"""Phase-1 roadmap regression tests."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit]


class TestAdvancedMetricsRegistry:
    def test_probe_returns_all_families(self):
        from florence_forge.evaluation.advanced_metrics_registry import (
            ADVANCED_METRIC_SPECS,
            probe_all_advanced_metrics,
        )

        statuses = probe_all_advanced_metrics()
        assert len(statuses) == len(ADVANCED_METRIC_SPECS)
        for spec in ADVANCED_METRIC_SPECS:
            assert spec.name in statuses

    def test_status_report_structure(self):
        from florence_forge.evaluation.advanced_metrics_registry import (
            advanced_metrics_status_report,
        )

        report = advanced_metrics_status_report()
        assert "available" in report
        assert "unavailable" in report
        assert report["total"] >= 4


class TestUnifiedMetricsNoSilentDegrade:
    def test_compute_does_not_fake_advanced_scores(self):
        from florence_forge.evaluation.unified_metrics import UnifiedMetrics

        um = UnifiedMetrics()
        result = um.compute(["hello"], ["hello"])
        assert "num_samples" in result or result  # basic metrics present
        # Must not inject zero-valued semantic/efficiency keys when unavailable
        for fake_key in ("semantic_score", "efficiency_score", "robustness_score"):
            if fake_key in result:
                assert result[fake_key] != 0.0 or um.advanced_metrics


class TestTaskMetricsSplit:
    def test_shim_imports_match_task_metrics(self):
        from florence_forge.evaluation import metrics as shim
        from florence_forge.evaluation.task_metrics import DetectionMetrics

        assert shim.DetectionMetrics is DetectionMetrics

    def test_map_still_computed(self):
        from florence_forge.evaluation.metrics import DetectionMetrics

        calc = DetectionMetrics()
        score = calc._compute_map(
            [[{"label": "cat", "bbox": [0, 0, 10, 10], "confidence": 0.9}]],
            [[{"label": "cat", "bbox": [0, 0, 10, 10]}]],
        )
        assert score > 0.0


class TestRewardModelsSplit:
    def test_shim_exports_format_rm(self):
        from florence_forge.training import reward_models as shim
        from florence_forge.training.rewards import FormatRewardModel

        assert shim.FormatRewardModel is FormatRewardModel

    def test_format_rm_scores_valid_box(self):
        from florence_forge.training.rewards import FormatRewardModel

        rm = FormatRewardModel()
        text = "<|box|>[10,20,30,40]<|/box|>"
        assert rm(text) > 0.5


class TestEvaluatorReportingSplit:
    def test_infer_case_score_exact_match(self):
        from florence_forge.evaluation.evaluator_reporting import infer_case_score

        assert infer_case_score({"prediction": "a", "reference": "a"}) == 1.0
        assert infer_case_score({"prediction": "a", "reference": "b"}) == 0.0

    def test_compute_overall_metrics_weighted(self):
        from florence_forge.evaluation.evaluator_reporting import compute_overall_metrics

        overall = compute_overall_metrics({
            "t1": {"metrics": {"f1": 0.8}, "sample_count": 2},
            "t2": {"metrics": {"f1": 0.4}, "sample_count": 2},
        })
        assert overall["avg_f1"] == pytest.approx(0.6)
        assert overall["total_samples"] == 4


class TestConsoleHelper:
    def test_cli_print_does_not_raise(self, capsys):
        from florence_forge.utils.console import cli_print

        cli_print("phase1 smoke")
        captured = capsys.readouterr()
        assert "phase1 smoke" in captured.out
