"""Benchmark advanced metric lazy-loading regression tests."""

import sys
import types
from unittest.mock import Mock, patch

from florence_forge.evaluation.benchmark import BenchmarkEvaluator
from florence_forge.evaluation.benchmark_lazy_metrics import LazyMetricCalculator


def test_lazy_metric_calculator_loads_underlying_class_on_first_use():
    module_name = "_florence_forge_lazy_metric_dummy"
    module = types.ModuleType(module_name)
    events = []

    class DummyCalculator:
        def __init__(self, value):
            events.append(("init", value))
            self.value = value

        def compute(self):
            return {"value": self.value}

    module.DummyCalculator = DummyCalculator
    sys.modules[module_name] = module
    try:
        calculator = LazyMetricCalculator(module_name, "DummyCalculator", 7)

        assert calculator.is_loaded is False
        assert events == []

        assert calculator.compute() == {"value": 7}
        assert calculator.is_loaded is True
        assert events == [("init", 7)]
    finally:
        sys.modules.pop(module_name, None)


def test_benchmark_evaluator_does_not_import_advanced_metrics_during_init(tmp_path):
    mock_model = Mock()
    mock_model.to = Mock(return_value=mock_model)
    mock_model.eval = Mock(return_value=mock_model)

    with patch(
        "florence_forge.evaluation.benchmark_lazy_metrics.import_module",
        side_effect=AssertionError("advanced metrics should stay lazy"),
    ):
        evaluator = BenchmarkEvaluator(
            model=mock_model,
            config={
                "cache_dir": str(tmp_path / "cache"),
                "enable_monitoring": False,
            },
        )

        assert evaluator.semantic_calculator is not None
        assert evaluator.multimodal_calculator is not None
        assert evaluator.robustness_calculator is not None
        assert evaluator.efficiency_calculator is not None
        assert evaluator.semantic_calculator.is_loaded is False


def test_benchmark_evaluator_can_eagerly_load_advanced_metrics_when_configured(
    tmp_path,
    monkeypatch,
):
    mock_model = Mock()
    mock_model.to = Mock(return_value=mock_model)
    mock_model.eval = Mock(return_value=mock_model)

    class DummyLazyMetricCalculator:
        loaded = []

        def __init__(self, module_name, class_name):
            self.module_name = module_name
            self.class_name = class_name

        def load(self):
            self.loaded.append(self.class_name)
            return self

    def fake_calculators():
        return {
            "semantic_calculator": DummyLazyMetricCalculator("m", "Semantic"),
            "multimodal_calculator": DummyLazyMetricCalculator("m", "MultiModal"),
        }

    monkeypatch.setattr(
        "florence_forge.evaluation.benchmark.make_default_advanced_metric_calculators",
        fake_calculators,
    )

    BenchmarkEvaluator(
        model=mock_model,
        config={
            "cache_dir": str(tmp_path / "cache"),
            "enable_monitoring": False,
            "lazy_advanced_metrics": False,
        },
    )

    assert DummyLazyMetricCalculator.loaded == ["Semantic", "MultiModal"]
