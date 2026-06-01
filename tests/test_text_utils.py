"""Text utility regression tests."""

import pytest

from florence_forge.utils.text import (
    TextProcessor,
    calculate_text_similarity,
    clean_text,
    convert_coordinates,
    extract_caption_keywords,
    extract_coordinates,
    extract_labels_and_coordinates,
    format_detection_result,
    parse_ocr_result,
    tokenize_text,
    validate_florence_format,
)


# ---------------------------------------------------------------------------
# clean_text
# ---------------------------------------------------------------------------


def test_clean_text_empty_returns_empty():
    assert clean_text("") == ""


def test_clean_text_collapses_whitespace_and_lowercases():
    out = clean_text("  Hello   World  ", lowercase=True)
    assert out == "hello world"


def test_clean_text_removes_special_chars():
    out = clean_text("hi @#$ there!", remove_special_chars=True)
    assert "@" not in out and "#" not in out
    assert "there!" in out


# ---------------------------------------------------------------------------
# tokenize_text
# ---------------------------------------------------------------------------


def test_tokenize_empty():
    assert tokenize_text("") == []


@pytest.mark.parametrize(
    "method,expected",
    [
        ("simple", ["Hello", "world", "123"]),
        ("whitespace", ["Hello,", "world", "123!"]),
    ],
)
def test_tokenize_methods(method, expected):
    assert tokenize_text("Hello, world 123!", method=method) == expected


def test_tokenize_regex_keeps_punctuation():
    assert tokenize_text("a, b.", method="regex") == ["a", ",", "b", "."]


def test_tokenize_preserve_case_false():
    assert tokenize_text("HeLLo", preserve_case=False) == ["hello"]


def test_tokenize_invalid_method_raises():
    with pytest.raises(ValueError, match="不支持的分词方法"):
        tokenize_text("x", method="bogus")


# ---------------------------------------------------------------------------
# extract_coordinates
# ---------------------------------------------------------------------------


def test_extract_coordinates_florence():
    coords = extract_coordinates("a<loc_100><loc_200><loc_300><loc_400>")
    assert coords == [(0.1, 0.2, 0.3, 0.4)]


def test_extract_coordinates_bbox():
    coords = extract_coordinates("[10, 20, 30, 40]", format_type="bbox")
    assert coords == [(10.0, 20.0, 30.0, 40.0)]


def test_extract_coordinates_center_size():
    coords = extract_coordinates("(50, 50, 20, 10)", format_type="center_size")
    assert coords == [(40.0, 45.0, 60.0, 55.0)]


def test_extract_coordinates_unknown_format_returns_empty():
    assert extract_coordinates("whatever", format_type="nope") == []


# ---------------------------------------------------------------------------
# extract_labels_and_coordinates
# ---------------------------------------------------------------------------


def test_extract_labels_florence():
    text = "person<loc_100><loc_200><loc_300><loc_400>"
    result = extract_labels_and_coordinates(text)
    assert result[0]["label"] == "person"
    assert result[0]["bbox"] == [0.1, 0.2, 0.3, 0.4]
    assert result[0]["confidence"] == 1.0


def test_extract_labels_json_list():
    result = extract_labels_and_coordinates('[{"label": "x"}]', format_type="json")
    assert result == [{"label": "x"}]


def test_extract_labels_json_objects_key():
    result = extract_labels_and_coordinates(
        '{"objects": [{"label": "y"}]}', format_type="json"
    )
    assert result == [{"label": "y"}]


def test_extract_labels_json_with_code_fence_recovery():
    text = '```json\n[{"label": "z"}]\n```'
    result = extract_labels_and_coordinates(text, format_type="json")
    assert result == [{"label": "z"}]


def test_extract_labels_json_unrecoverable_returns_empty():
    result = extract_labels_and_coordinates("definitely not json", format_type="json")
    assert result == []


# ---------------------------------------------------------------------------
# format_detection_result
# ---------------------------------------------------------------------------


def test_format_detection_empty_returns_empty():
    assert format_detection_result([]) == ""


def test_format_detection_florence_normalized():
    dets = [{"label": "cat", "bbox": [0.1, 0.2, 0.3, 0.4]}]
    out = format_detection_result(dets, format_type="florence")
    assert out == "cat<loc_100><loc_200><loc_300><loc_400>"


def test_format_detection_florence_with_image_size():
    dets = [{"label": "cat", "bbox": [50, 100, 150, 200]}]
    out = format_detection_result(dets, format_type="florence", image_size=(200, 200))
    assert out == "cat<loc_250><loc_500><loc_750><loc_1000>"


def test_format_detection_json():
    dets = [{"label": "cat", "bbox": [1, 2, 3, 4]}]
    out = format_detection_result(dets, format_type="json")
    assert '"label"' in out


def test_format_detection_text():
    dets = [
        {"label": "cat", "bbox": [0.1, 0.2, 0.3, 0.4], "confidence": 0.9},
        {"label": "dog", "confidence": 0.5},
    ]
    out = format_detection_result(dets, format_type="text")
    assert "1. cat" in out
    assert "2. dog" in out


def test_format_detection_invalid_type_raises():
    with pytest.raises(ValueError, match="不支持的格式类型"):
        format_detection_result([{"label": "x", "bbox": [1, 2, 3, 4]}], format_type="xml")


# ---------------------------------------------------------------------------
# parse_ocr_result
# ---------------------------------------------------------------------------


def test_parse_ocr_result():
    text = "hello<loc_10><loc_20><loc_30><loc_40>"
    result = parse_ocr_result(text)
    assert result[0]["text"] == "hello"
    assert result[0]["bbox"] == [0.01, 0.02, 0.03, 0.04]


# ---------------------------------------------------------------------------
# extract_caption_keywords
# ---------------------------------------------------------------------------


def test_extract_caption_keywords_filters_and_dedups():
    caption = "The big brown dog and the big cat"
    keywords = extract_caption_keywords(caption)
    assert "big" in keywords
    assert keywords.count("big") == 1
    assert "the" not in keywords  # stopword
    assert "and" not in keywords


def test_extract_caption_keywords_custom_exclude():
    keywords = extract_caption_keywords("apple banana", exclude_words=["apple"])
    assert keywords == ["banana"]


# ---------------------------------------------------------------------------
# calculate_text_similarity (includes regression for missing Counter import)
# ---------------------------------------------------------------------------


def test_similarity_empty_returns_zero():
    assert calculate_text_similarity("", "x") == 0.0


def test_similarity_jaccard():
    score = calculate_text_similarity("a b c", "b c d", method="jaccard")
    assert score == pytest.approx(2 / 4)


def test_similarity_cosine_does_not_raise_and_is_bounded():
    score = calculate_text_similarity("a b c", "a b c", method="cosine")
    assert score == pytest.approx(1.0)
    partial = calculate_text_similarity("a b c", "a b d", method="cosine")
    assert 0.0 < partial < 1.0


def test_similarity_overlap():
    score = calculate_text_similarity("a b", "a b c d", method="overlap")
    assert score == pytest.approx(1.0)


def test_similarity_invalid_method_raises():
    with pytest.raises(ValueError, match="不支持的相似度计算方法"):
        calculate_text_similarity("a", "b", method="bogus")


# ---------------------------------------------------------------------------
# TextProcessor
# ---------------------------------------------------------------------------


def test_text_processor_process_and_tokenize():
    processor = TextProcessor()
    assert processor.process_text("  a   b ") == "a b"
    assert processor.tokenize("Hello, world") == ["Hello", "world"]


def test_text_processor_extract_entities_detection():
    processor = TextProcessor()
    result = processor.extract_entities(
        "person<loc_100><loc_200><loc_300><loc_400>", entity_type="detection"
    )
    assert result[0]["label"] == "person"


def test_text_processor_extract_entities_ocr():
    processor = TextProcessor()
    result = processor.extract_entities(
        "abc<loc_1><loc_2><loc_3><loc_4>", entity_type="ocr"
    )
    assert result[0]["text"] == "abc"


def test_text_processor_extract_entities_invalid():
    with pytest.raises(ValueError, match="不支持的实体类型"):
        TextProcessor().extract_entities("x", entity_type="bogus")


def test_text_processor_format_output():
    processor = TextProcessor()
    out = processor.format_output(
        [{"label": "cat", "bbox": [0.1, 0.2, 0.3, 0.4], "confidence": 0.8}]
    )
    assert "cat" in out


# ---------------------------------------------------------------------------
# validate_florence_format
# ---------------------------------------------------------------------------


def test_validate_florence_format():
    assert validate_florence_format("a<loc_1><loc_2><loc_3><loc_4>") is True
    assert validate_florence_format("a<loc_1><loc_2><loc_3>") is False
    assert validate_florence_format("no markup") is True  # zero markers -> 0 % 4 == 0


# ---------------------------------------------------------------------------
# convert_coordinates
# ---------------------------------------------------------------------------


def test_convert_coordinates_requires_four_values():
    with pytest.raises(ValueError, match="4个值"):
        convert_coordinates([1, 2, 3], "normalized", "florence")


def test_convert_coordinates_normalized_to_florence():
    out = convert_coordinates([0.1, 0.2, 0.3, 0.4], "normalized", "florence")
    assert out == [100, 200, 300, 400]


def test_convert_coordinates_absolute_roundtrip():
    out = convert_coordinates(
        [50, 100, 150, 200], "absolute", "normalized", image_size=(200, 200)
    )
    assert out == [0.25, 0.5, 0.75, 1.0]


def test_convert_coordinates_florence_to_absolute():
    out = convert_coordinates(
        [100, 200, 300, 400], "florence", "absolute", image_size=(200, 200)
    )
    assert out == [20.0, 40.0, 60.0, 80.0]


def test_convert_coordinates_absolute_without_size_raises():
    with pytest.raises(ValueError, match="图像尺寸"):
        convert_coordinates([1, 2, 3, 4], "absolute", "normalized")


def test_convert_coordinates_invalid_formats_raise():
    with pytest.raises(ValueError, match="不支持的源格式"):
        convert_coordinates([0.1, 0.2, 0.3, 0.4], "bogus", "normalized")
    with pytest.raises(ValueError, match="不支持的目标格式"):
        convert_coordinates([0.1, 0.2, 0.3, 0.4], "normalized", "bogus")
