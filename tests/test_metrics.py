"""评估指标单元测试。"""

import pytest

from florence_forge.evaluation.metrics import CaptionMetrics, DetectionMetrics, get_metric_calculator


def test_get_metric_calculator_routes_od_to_detection():
    calculator = get_metric_calculator("OD")
    assert isinstance(calculator, DetectionMetrics)


def test_detection_map_uses_label_and_confidence_fields():
    calculator = DetectionMetrics()
    predictions = [[
        {"label": "cat", "bbox": [0, 0, 10, 10], "confidence": 0.95},
    ]]
    references = [[
        {"label": "cat", "bbox": [0, 0, 10, 10]},
    ]]

    score = calculator._compute_map(predictions, references)
    assert score > 0.0


def test_detection_map_does_not_match_boxes_across_images():
    calculator = DetectionMetrics()
    predictions = [
        [{"label": "cat", "bbox": [0, 0, 10, 10], "confidence": 0.95}],
        [],
    ]
    references = [
        [],
        [{"label": "cat", "bbox": [0, 0, 10, 10]}],
    ]

    score = calculator._compute_map(predictions, references)

    assert score == 0.0


def test_detection_compute_reports_map_without_coco_dependency_gate():
    calculator = DetectionMetrics()
    calculator.add_batch(
        ['[{"label": "cat", "bbox": [0, 0, 10, 10], "confidence": 0.95}]'],
        ['[{"label": "cat", "bbox": [0, 0, 10, 10]}]'],
    )

    metrics = calculator.compute()

    assert "mAP" in metrics
    assert metrics["mAP"] > 0.0


def test_caption_bleu_falls_back_when_nltk_punkt_is_missing(monkeypatch):
    pytest.importorskip("nltk.tokenize")

    def raise_lookup_error(_text):
        raise LookupError("punkt missing")

    monkeypatch.setattr("nltk.tokenize.word_tokenize", raise_lookup_error)

    calculator = CaptionMetrics()
    calculator.add_batch(["a red image"], ["a red image"])

    metrics = calculator.compute()

    assert "bleu" in metrics
    assert metrics["bleu"] > 0.0
