import json
from types import SimpleNamespace

from PIL import Image

from florence_forge.core.tasks import get_task_config, validate_task_name
from florence_forge.core.visual_primitives import (
    VISUAL_PRIMITIVE_PLAIN_TOKENS,
    VISUAL_PRIMITIVE_SPECIAL_TOKENS,
    format_ref_box,
    format_ref_box_loc_tokens,
    get_visual_primitive_tokens,
    normalize_bbox,
    validate_normalized_bbox,
)
from florence_forge.core.backends.florence2_backend import Florence2Backend
from florence_forge.data.dataset import MultiTaskDataset
from florence_forge.evaluation.metrics import (
    DetectionMetrics,
    VisualPrimitiveDetectionMetrics,
    get_metric_calculator,
)
from florence_forge.evaluation.visual_primitive_parser import VisualPrimitiveParser


def test_visual_primitive_bbox_normalization_and_formatting():
    bbox = normalize_bbox([10, 20, 30, 40], (100, 200), input_format="xyxy")

    assert bbox == [100, 100, 300, 200]
    assert validate_normalized_bbox(bbox)
    assert format_ref_box("cat", [bbox]) == "<|ref|>cat<|/ref|><|box|>[[100,100,300,200]]<|/box|>"
    assert (
        format_ref_box_loc_tokens("cat", [bbox])
        == "<|ref|>cat<|/ref|><|box|><loc_100><loc_100><loc_300><loc_200><|/box|>"
    )
    assert get_visual_primitive_tokens("plain") == VISUAL_PRIMITIVE_PLAIN_TOKENS
    assert format_ref_box("cat", [bbox], marker_style="plain") == "<ref>cat</ref> <box>[[100,100,300,200]]</box>"
    assert (
        format_ref_box_loc_tokens("cat", [bbox], marker_style="plain")
        == "<ref>cat</ref> <box><loc_100><loc_100><loc_300><loc_200></box>"
    )


def test_visual_primitive_tasks_are_registered():
    assert validate_task_name("OD_VP")
    assert validate_task_name("COUNT_VP")
    assert validate_task_name("PHRASE_GROUNDING_VP")
    assert get_task_config("OD_VP")["prompt"] == "<OD>"
    assert get_task_config("OD_VP")["is_visual_primitive"] is True


def test_multitask_dataset_accepts_visual_primitive_task(tmp_path):
    image_path = tmp_path / "image.jpg"
    Image.new("RGB", (16, 16), color="white").save(image_path)
    data_path = tmp_path / "vp.jsonl"
    row = {
        "image": image_path.name,
        "prefix": "<OD>",
        "suffix": "<|ref|>cat<|/ref|><|box|>[[0,0,999,999]]<|/box|>",
    }
    data_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    dataset = MultiTaskDataset(
        data_configs=[{"task_type": "OD_VP", "data_path": str(data_path)}],
        image_base_path=str(tmp_path),
    )

    assert len(dataset) == 1
    assert dataset[0]["task_type"] == "OD_VP"
    assert dataset[0]["prompt"] == "<OD>"


def test_visual_primitive_parser_extracts_ref_boxes():
    text = (
        "2. Object grounding\n"
        "<|ref|>red cat<|/ref|>\n"
        "<|box|>[[10,20,30,40],[50,60,70,80]]<|/box|>"
    )

    detections = VisualPrimitiveParser().parse_detections(text)

    assert detections == [
        {"label": "red cat", "bbox": [10, 20, 30, 40], "confidence": 1.0},
        {"label": "red cat", "bbox": [50, 60, 70, 80], "confidence": 1.0},
    ]


def test_visual_primitive_parser_extracts_loc_token_boxes():
    text = "<|ref|>cat<|/ref|><|box|><loc_10><loc_20><loc_30><loc_40><|/box|>"

    detections = VisualPrimitiveParser().parse_detections(text)

    assert detections == [
        {"label": "cat", "bbox": [10, 20, 30, 40], "confidence": 1.0}
    ]


def test_visual_primitive_parser_extracts_plain_marker_boxes():
    text = "<ref>cat</ref><box><loc_10><loc_20><loc_30><loc_40></box>"

    detections = VisualPrimitiveParser().parse_detections(text)

    assert detections == [
        {"label": "cat", "bbox": [10, 20, 30, 40], "confidence": 1.0}
    ]


def test_detection_metrics_parse_visual_primitive_output():
    calculator = DetectionMetrics()
    prediction = "<|ref|>cat<|/ref|><|box|>[[0,0,100,100]]<|/box|>"
    reference = json.dumps([
        {"label": "cat", "bbox": [0, 0, 100, 100], "confidence": 1.0}
    ])
    calculator.add_batch([prediction], [reference])

    metrics = calculator.compute()

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["mAP"] > 0.0
    assert isinstance(get_metric_calculator("OD_VP"), DetectionMetrics)


def test_visual_primitive_detection_metrics_add_format_quality_fields():
    calculator = get_metric_calculator("OD_VP")
    assert isinstance(calculator, VisualPrimitiveDetectionMetrics)
    prediction = "<|ref|>cat<|/ref|><|box|>[[0,0,100,100]]<|/box|>"
    reference = "<|ref|>cat<|/ref|><|box|>[[0,0,100,100]]<|/box|>"
    calculator.add_batch([prediction], [reference])

    metrics = calculator.compute()

    assert metrics["vp_format_valid_ratio"] == 1.0
    assert metrics["vp_coordinate_valid_ratio"] == 1.0
    assert metrics["vp_ref_coverage_ratio"] == 1.0
    assert metrics["vp_avg_pred_boxes"] == 1.0
    assert metrics["vp_box_count_exact_match"] == 1.0


def test_visual_primitive_detection_metrics_accept_loc_token_boxes():
    calculator = get_metric_calculator("OD_VP")
    prediction = "<|ref|>cat<|/ref|><|box|><loc_0><loc_0><loc_100><loc_100><|/box|>"
    calculator.add_batch([prediction], [prediction])

    metrics = calculator.compute()

    assert metrics["vp_format_valid_ratio"] == 1.0
    assert metrics["vp_coordinate_valid_ratio"] == 1.0
    assert metrics["vp_box_count_exact_match"] == 1.0


def test_visual_primitive_detection_metrics_accept_plain_marker_boxes():
    calculator = get_metric_calculator("OD_VP")
    prediction = "<ref>cat</ref><box><loc_0><loc_0><loc_100><loc_100></box>"
    calculator.add_batch([prediction], [prediction])

    metrics = calculator.compute()

    assert metrics["vp_format_valid_ratio"] == 1.0
    assert metrics["vp_coordinate_valid_ratio"] == 1.0
    assert metrics["vp_box_count_exact_match"] == 1.0


def test_visual_primitive_detection_metrics_report_structured_native_output():
    calculator = get_metric_calculator("OD_VP")
    prediction = "</s><s>zebra<loc_0><loc_51><loc_690><loc_938></s>"
    reference = "<|ref|>zebra<|/ref|><|box|><loc_0><loc_51><loc_690><loc_938><|/box|>"
    calculator.add_batch([prediction], [reference])

    metrics = calculator.compute()

    assert metrics["vp_format_valid_ratio"] == 0.0
    assert metrics["structured_vp_format_valid_ratio"] == 1.0
    assert metrics["structured_vp_decoder_ratio"] == 1.0
    assert metrics["structured_vp_source_florence_native_ratio"] == 1.0
    assert metrics["structured_vp_box_count_exact_match"] == 1.0
    assert metrics["structured_precision"] == 1.0
    assert metrics["structured_recall"] == 1.0


def test_florence_backend_adds_visual_primitive_tokens_and_resizes_embeddings():
    class DummyTokenizer:
        def __init__(self):
            self.vocab = {"<OD>": 0}

        def get_vocab(self):
            return dict(self.vocab)

        def add_tokens(self, tokens, special_tokens=False):
            for token in tokens:
                self.vocab[token] = len(self.vocab)
            self.special_tokens = special_tokens
            return len(tokens)

        def __len__(self):
            return len(self.vocab)

    class DummyModel:
        def resize_token_embeddings(self, size):
            self.resized_to = size

    tokenizer = DummyTokenizer()
    backend = Florence2Backend(SimpleNamespace(enable_visual_primitives=True))
    backend._processor = SimpleNamespace(tokenizer=tokenizer)
    backend._model = DummyModel()

    backend._maybe_add_visual_primitive_tokens()

    for token in VISUAL_PRIMITIVE_SPECIAL_TOKENS:
        assert token in tokenizer.vocab
    assert tokenizer.special_tokens is True
    assert backend._model.resized_to == len(tokenizer)
