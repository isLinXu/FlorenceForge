from florence_forge.evaluation import (
    FlorenceNativeDetectionParser,
    StructuredVisualPrimitiveDecoder,
    labels_match,
    normalize_allowed_labels,
    native_detections_to_vp,
    resolve_structured_vp_filter_caps,
)
from florence_forge.evaluation.structured_vp_decoder import filter_native_detections


def test_florence_native_detection_parser_extracts_label_loc_groups():
    text = "</s><s>cat<loc_0><loc_78><loc_564><loc_774>footwear<loc_515><loc_156><loc_729><loc_560></s>"

    detections = FlorenceNativeDetectionParser().parse(text)

    assert detections == [
        {"label": "cat", "bbox": [0, 78, 564, 774], "confidence": 1.0},
        {"label": "footwear", "bbox": [515, 156, 729, 560], "confidence": 1.0},
    ]


def test_florence_native_parser_prefers_partial_plain_ref_prefix_label():
    text = (
        "</s><ref>cat</ref> <box> cat<loc_0><loc_78><loc_564><loc_774>"
        "footwear<loc_515><loc_156><loc_729><loc_560></s>"
    )

    detections = FlorenceNativeDetectionParser().parse(text)

    assert detections == [
        {"label": "cat", "bbox": [0, 78, 564, 774], "confidence": 1.0},
        {"label": "footwear", "bbox": [515, 156, 729, 560], "confidence": 1.0},
    ]


def test_florence_native_parser_prefers_partial_special_ref_prefix_label():
    text = "<|ref|>zebra<|/ref|><|box|>zoo<loc_0><loc_52><loc_689><loc_940></s>"

    detections = FlorenceNativeDetectionParser().parse(text)

    assert detections == [
        {"label": "zebra", "bbox": [0, 52, 689, 940], "confidence": 1.0}
    ]


def test_native_detections_to_vp_formats_loc_tokens():
    text = native_detections_to_vp([
        {"label": "cat", "bbox": [0, 78, 564, 774]},
        {"label": "footwear", "bbox": [515, 156, 729, 560]},
    ])

    assert "<|ref|>cat<|/ref|><|box|><loc_0><loc_78><loc_564><loc_774><|/box|>" in text
    assert "<|ref|>footwear<|/ref|><|box|><loc_515><loc_156><loc_729><loc_560><|/box|>" in text


def test_structured_decoder_wraps_florence_native_output_as_vp():
    text = "</s><s>cat<loc_0><loc_78><loc_564><loc_774>footwear<loc_515><loc_156><loc_729><loc_560></s>"

    result = StructuredVisualPrimitiveDecoder().decode(text)

    assert result.source == "florence_native"
    assert result.used_structured_decoder is True
    assert result.detections == [
        {"label": "cat", "bbox": [0, 78, 564, 774], "confidence": 1.0},
        {"label": "footwear", "bbox": [515, 156, 729, 560], "confidence": 1.0},
    ]
    assert "<|ref|>cat<|/ref|>" in result.text
    assert "<|ref|>footwear<|/ref|>" in result.text
    assert result.raw_detection_count == 2
    assert result.filtered_detection_count == 0


def test_filter_native_detections_caps_boxes_per_label_in_order():
    detections = [
        {"label": "dog", "bbox": [1, 1, 10, 10]},
        {"label": "footwear", "bbox": [20, 20, 30, 30]},
        {"label": "footwear", "bbox": [40, 40, 50, 50]},
        {"label": "dog", "bbox": [60, 60, 70, 70]},
    ]

    filtered = filter_native_detections(detections, max_boxes_per_label=1)

    assert filtered == [
        {"label": "dog", "bbox": [1, 1, 10, 10]},
        {"label": "footwear", "bbox": [20, 20, 30, 30]},
    ]


def test_structured_decoder_can_filter_overgenerated_native_boxes():
    text = (
        "dog<loc_335><loc_84><loc_888><loc_622>"
        "footwear<loc_20><loc_606><loc_463><loc_998>"
        "<loc_212><loc_309><loc_541><loc_632>"
        "<loc_0><loc_514><loc_169><loc_998>"
    )

    result = StructuredVisualPrimitiveDecoder(
        marker_style="plain",
        max_boxes_per_label=1,
        max_total_boxes=2,
    ).decode(text)

    assert result.source == "florence_native"
    assert result.raw_detection_count == 4
    assert result.filtered_detection_count == 2
    assert result.detections == [
        {"label": "dog", "bbox": [335, 84, 888, 622], "confidence": 1.0},
        {"label": "footwear", "bbox": [20, 606, 463, 998], "confidence": 1.0},
    ]
    assert result.text == (
        "<ref>dog</ref> <box><loc_335><loc_84><loc_888><loc_622></box>\n"
        "<ref>footwear</ref> <box><loc_20><loc_606><loc_463><loc_998></box>"
    )


def test_filter_native_detections_can_apply_per_label_nms_in_order():
    detections = [
        {"label": "cat", "bbox": [0, 0, 100, 100]},
        {"label": "cat", "bbox": [5, 5, 105, 105]},
        {"label": "dog", "bbox": [5, 5, 105, 105]},
        {"label": "cat", "bbox": [300, 300, 400, 400]},
    ]

    filtered = filter_native_detections(detections, nms_iou_threshold=0.5)

    assert filtered == [
        {"label": "cat", "bbox": [0, 0, 100, 100]},
        {"label": "dog", "bbox": [5, 5, 105, 105]},
        {"label": "cat", "bbox": [300, 300, 400, 400]},
    ]


def test_structured_decoder_can_filter_duplicate_native_boxes_with_nms():
    text = (
        "cat<loc_0><loc_0><loc_100><loc_100>"
        "<loc_5><loc_5><loc_105><loc_105>"
        "dog<loc_5><loc_5><loc_105><loc_105>"
    )

    result = StructuredVisualPrimitiveDecoder(
        marker_style="plain",
        nms_iou_threshold=0.5,
    ).decode(text)

    assert result.raw_detection_count == 3
    assert result.filtered_detection_count == 1
    assert result.detections == [
        {"label": "cat", "bbox": [0, 0, 100, 100], "confidence": 1.0},
        {"label": "dog", "bbox": [5, 5, 105, 105], "confidence": 1.0},
    ]


def test_structured_decoder_can_filter_by_allowed_labels():
    text = (
        "cat<loc_0><loc_0><loc_100><loc_100>"
        "footwear<loc_200><loc_200><loc_300><loc_300>"
        "dog<loc_400><loc_400><loc_500><loc_500>"
    )

    result = StructuredVisualPrimitiveDecoder(
        marker_style="plain",
        allowed_labels="cat, dog",
    ).decode(text)

    assert normalize_allowed_labels(" cat ; dog\ncat ") == ["cat", "dog"]
    assert result.raw_detection_count == 3
    assert result.filtered_detection_count == 1
    assert result.detections == [
        {"label": "cat", "bbox": [0, 0, 100, 100], "confidence": 1.0},
        {"label": "dog", "bbox": [400, 400, 500, 500], "confidence": 1.0},
    ]


def test_structured_decoder_can_use_phrase_contained_allowed_label_matching():
    text = (
        "coffee cup<loc_0><loc_0><loc_100><loc_100>"
        "business sign<loc_200><loc_200><loc_300><loc_300>"
    )

    result = StructuredVisualPrimitiveDecoder(
        marker_style="plain",
        allowed_labels="cup,bus",
        allowed_label_match_mode="contains",
    ).decode(text)

    assert labels_match("coffee cup", "cup", mode="contains") is True
    assert labels_match("business sign", "bus", mode="contains") is False
    assert labels_match("coffee cup", "cup", mode="strict") is False
    assert result.raw_detection_count == 2
    assert result.filtered_detection_count == 1
    assert result.detections == [
        {"label": "coffee cup", "bbox": [0, 0, 100, 100], "confidence": 1.0},
    ]


def test_resolve_structured_vp_filter_caps_supports_auto_single_target_policy():
    assert resolve_structured_vp_filter_caps(
        policy="auto",
        task_prompt="<OD_VP>",
    ) == {
        "max_boxes_per_label": None,
        "max_total_boxes": 1,
        "nms_iou_threshold": None,
        "allowed_labels": None,
    }
    assert resolve_structured_vp_filter_caps(
        policy="auto",
        task_prompt="<OD>",
    ) == {
        "max_boxes_per_label": None,
        "max_total_boxes": None,
        "nms_iou_threshold": None,
        "allowed_labels": None,
    }
    assert resolve_structured_vp_filter_caps(
        policy="single-target",
        task_prompt="<OD>",
        max_total_boxes=2,
    ) == {
        "max_boxes_per_label": None,
        "max_total_boxes": 2,
        "nms_iou_threshold": None,
        "allowed_labels": None,
    }
    assert resolve_structured_vp_filter_caps(
        policy="nms",
        task_prompt="<OD>",
    ) == {
        "max_boxes_per_label": None,
        "max_total_boxes": None,
        "nms_iou_threshold": 0.5,
        "allowed_labels": None,
    }
    assert resolve_structured_vp_filter_caps(
        policy="none",
        allowed_labels="Cat, dog",
    ) == {
        "max_boxes_per_label": None,
        "max_total_boxes": None,
        "nms_iou_threshold": None,
        "allowed_labels": ["cat", "dog"],
    }


def test_structured_decoder_preserves_already_valid_vp_text():
    text = "<|ref|>cat<|/ref|><|box|><loc_0><loc_78><loc_564><loc_774><|/box|>"

    result = StructuredVisualPrimitiveDecoder().decode(text)

    assert result.source == "visual_primitive"
    assert result.text == text
    assert result.detections == [
        {"label": "cat", "bbox": [0, 78, 564, 774], "confidence": 1.0}
    ]


def test_structured_decoder_can_filter_already_valid_vp_text():
    text = (
        "<ref>cat</ref> <box><loc_0><loc_0><loc_100><loc_100></box>\n"
        "<ref>footwear</ref> <box><loc_200><loc_200><loc_300><loc_300></box>"
    )

    result = StructuredVisualPrimitiveDecoder(
        marker_style="plain",
        allowed_labels="cat",
    ).decode(text)

    assert result.source == "visual_primitive"
    assert result.raw_detection_count == 2
    assert result.filtered_detection_count == 1
    assert result.text == "<ref>cat</ref> <box><loc_0><loc_0><loc_100><loc_100></box>"
    assert result.detections == [
        {"label": "cat", "bbox": [0, 0, 100, 100], "confidence": 1.0}
    ]


def test_structured_decoder_can_repair_malformed_vp_tail_loc_groups():
    text = (
        "<ref>person</ref> <box><loc_282><loc_517><loc_445><loc_840>person</box> "
        "<<loc_282><loc_517><loc_446><loc_840>person"
        "<loc_327><loc_487><loc_366><loc_661>"
    )

    default_result = StructuredVisualPrimitiveDecoder(marker_style="plain").decode(text)
    repaired = StructuredVisualPrimitiveDecoder(
        marker_style="plain",
        nms_iou_threshold=0.5,
        allowed_labels="person",
        repair_malformed_tail=True,
    ).decode(text)

    assert default_result.source == "visual_primitive"
    assert len(default_result.detections) == 1
    assert repaired.source == "visual_primitive_repaired_tail"
    assert repaired.used_tail_repair is True
    assert repaired.raw_detection_count == 3
    assert repaired.filtered_detection_count == 1
    assert repaired.repaired_tail_detection_count == 2
    assert repaired.detections == [
        {"label": "person", "bbox": [282, 517, 445, 840], "confidence": 1.0},
        {"label": "person", "bbox": [327, 487, 366, 661], "confidence": 1.0},
    ]
    assert repaired.text == (
        "<ref>person</ref> <box>"
        "<loc_282><loc_517><loc_445><loc_840>"
        "<loc_327><loc_487><loc_366><loc_661>"
        "</box>"
    )


def test_structured_decoder_can_emit_json_vp_boxes():
    result = StructuredVisualPrimitiveDecoder(box_format="json").decode(
        "cat<loc_0><loc_78><loc_564><loc_774>"
    )

    assert result.source == "florence_native"
    assert result.text == "<|ref|>cat<|/ref|><|box|>[[0,78,564,774]]<|/box|>"


def test_structured_decoder_can_emit_plain_marker_loc_boxes():
    result = StructuredVisualPrimitiveDecoder(marker_style="plain").decode(
        "cat<loc_0><loc_78><loc_564><loc_774>"
    )

    assert result.source == "florence_native"
    assert result.text == "<ref>cat</ref> <box><loc_0><loc_78><loc_564><loc_774></box>"
    assert result.detections == [
        {"label": "cat", "bbox": [0, 78, 564, 774], "confidence": 1.0}
    ]
