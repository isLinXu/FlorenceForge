"""Data validator import and behavior regression tests."""

import json

from PIL import Image

from florence_forge.data import (
    DataValidator,
    MultiTaskDataset,
    TaskSample,
    validate_data_format,
)


def _write_image(path):
    Image.new("RGB", (64, 64), color=(255, 0, 0)).save(path)


def _valid_sample(image_name="image.png"):
    return {
        "image": image_name,
        "task_type": "CAPTION",
        "conversations": [
            {"from": "human", "value": "Describe the image."},
            {"from": "gpt", "value": "A small red square."},
        ],
    }


def test_data_validator_imports_from_package_namespace(tmp_path):
    image_path = tmp_path / "image.png"
    _write_image(image_path)
    data_path = tmp_path / "data.jsonl"
    data_path.write_text(json.dumps(_valid_sample()) + "\n", encoding="utf-8")

    result = DataValidator().validate_dataset(data_path)

    assert result["is_valid"] is True
    assert result["total_samples"] == 1
    assert result["valid_samples"] == 1
    assert result["task_distribution"] == {"CAPTION": 1}
    assert result["image_formats"] == {"PNG": 1}
    assert MultiTaskDataset.__name__ == "MultiTaskDataset"
    assert TaskSample.__name__ == "TaskSample"


def test_validate_data_format_reports_jsonl_parse_and_schema_errors(tmp_path):
    data_path = tmp_path / "bad.jsonl"
    missing_fields = {"image": "missing.png", "task_type": "OCR"}
    data_path.write_text(
        "{not-json}\n" + json.dumps(missing_fields) + "\n",
        encoding="utf-8",
    )

    result = validate_data_format(data_path)
    messages = [entry["message"] for entry in result["validation_results"]]

    assert result["is_valid"] is False
    assert result["error_count"] == 2
    assert any("JSON解析错误" in message for message in messages)
    assert any("缺少必需字段 'conversations'" in message for message in messages)


def test_data_validator_resets_state_between_validate_calls(tmp_path):
    validator = DataValidator()
    missing_result = validator.validate_dataset(tmp_path / "missing.jsonl")

    image_path = tmp_path / "image.png"
    _write_image(image_path)
    data_path = tmp_path / "data.jsonl"
    data_path.write_text(json.dumps(_valid_sample()) + "\n", encoding="utf-8")
    valid_result = validator.validate_dataset(data_path)

    assert missing_result["is_valid"] is False
    assert valid_result["is_valid"] is True
    assert valid_result["error_count"] == 0
    assert valid_result["validation_results"] == []


def test_strict_mode_treats_warnings_as_failed_validation(tmp_path):
    image_path = tmp_path / "image.png"
    _write_image(image_path)
    sample = _valid_sample()
    sample["conversations"][-1]["value"] = "bad"
    data_path = tmp_path / "data.jsonl"
    data_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")

    loose_result = validate_data_format(data_path, strict_mode=False)
    strict_result = validate_data_format(data_path, strict_mode=True)

    assert loose_result["is_valid"] is True
    assert loose_result["warning_count"] == 1
    assert strict_result["is_valid"] is False
    assert strict_result["error_count"] == 0
    assert strict_result["warning_count"] == 1


def test_missing_images_are_counted_as_invalid_samples(tmp_path):
    data_path = tmp_path / "data.jsonl"
    data_path.write_text(
        json.dumps(_valid_sample("missing.png")) + "\n", encoding="utf-8"
    )

    result = validate_data_format(data_path)

    assert result["is_valid"] is False
    assert result["missing_images"] == 1
    assert result["valid_samples"] == 0
    assert result["invalid_samples"] == 1


def test_data_validator_reports_missing_path(tmp_path):
    result = DataValidator().validate_dataset(tmp_path / "missing.jsonl")

    assert result["is_valid"] is False
    assert result["error_count"] == 1
    assert "路径不存在" in result["validation_results"][0]["message"]


# ---------------------------------------------------------------------------
# JSON dataset variants
# ---------------------------------------------------------------------------


def test_validate_json_list_dataset(tmp_path):
    _write_image(tmp_path / "image.png")
    data_path = tmp_path / "data.json"
    data_path.write_text(json.dumps([_valid_sample()]), encoding="utf-8")

    result = validate_data_format(data_path)
    assert result["is_valid"] is True
    assert result["total_samples"] == 1


def test_validate_json_dict_with_data_key(tmp_path):
    _write_image(tmp_path / "image.png")
    data_path = tmp_path / "data.json"
    data_path.write_text(json.dumps({"data": [_valid_sample()]}), encoding="utf-8")

    result = validate_data_format(data_path)
    assert result["is_valid"] is True
    assert result["total_samples"] == 1


def test_validate_json_dict_without_data_key_errors(tmp_path):
    data_path = tmp_path / "data.json"
    data_path.write_text(json.dumps({"samples": []}), encoding="utf-8")

    result = validate_data_format(data_path)
    messages = [e["message"] for e in result["validation_results"]]
    assert result["is_valid"] is False
    assert any("缺少'data'字段" in m for m in messages)


def test_validate_json_non_container_root_errors(tmp_path):
    data_path = tmp_path / "data.json"
    data_path.write_text(json.dumps(42), encoding="utf-8")

    result = validate_data_format(data_path)
    messages = [e["message"] for e in result["validation_results"]]
    assert result["is_valid"] is False
    assert any("必须是列表或字典" in m for m in messages)


def test_validate_json_decode_error(tmp_path):
    data_path = tmp_path / "data.json"
    data_path.write_text("{not json}", encoding="utf-8")

    result = validate_data_format(data_path)
    messages = [e["message"] for e in result["validation_results"]]
    assert result["is_valid"] is False
    assert any("JSON解析错误" in m for m in messages)


def test_validate_json_read_error_on_directory(tmp_path):
    bogus = tmp_path / "data.json"
    bogus.mkdir()

    result = validate_data_format(bogus)
    messages = [e["message"] for e in result["validation_results"]]
    assert result["is_valid"] is False
    assert any("读取文件错误" in m for m in messages)


def test_validate_jsonl_read_error_on_directory(tmp_path):
    bogus = tmp_path / "data.jsonl"
    bogus.mkdir()

    result = validate_data_format(bogus)
    messages = [e["message"] for e in result["validation_results"]]
    assert result["is_valid"] is False
    assert any("读取JSONL文件错误" in m for m in messages)


def test_validate_unsupported_format(tmp_path):
    data_path = tmp_path / "data.txt"
    data_path.write_text("hello", encoding="utf-8")

    result = validate_data_format(data_path)
    messages = [e["message"] for e in result["validation_results"]]
    assert result["is_valid"] is False
    assert any("不支持的数据格式" in m for m in messages)


def test_validate_empty_dataset(tmp_path):
    data_path = tmp_path / "data.json"
    data_path.write_text(json.dumps([]), encoding="utf-8")

    result = validate_data_format(data_path)
    messages = [e["message"] for e in result["validation_results"]]
    assert result["is_valid"] is False
    assert any("数据集为空" in m for m in messages)


# ---------------------------------------------------------------------------
# Sample / conversation schema errors
# ---------------------------------------------------------------------------


def _write_jsonl(tmp_path, samples):
    data_path = tmp_path / "data.jsonl"
    data_path.write_text(
        "\n".join(json.dumps(s) for s in samples) + "\n", encoding="utf-8"
    )
    return data_path


def test_conversations_must_be_non_empty_list(tmp_path):
    _write_image(tmp_path / "image.png")
    sample = {"image": "image.png", "task_type": "CAPTION", "conversations": []}
    result = validate_data_format(_write_jsonl(tmp_path, [sample]))
    messages = [e["message"] for e in result["validation_results"]]
    assert any("非空列表" in m for m in messages)


def test_conversation_entry_must_be_dict(tmp_path):
    _write_image(tmp_path / "image.png")
    sample = {"image": "image.png", "task_type": "CAPTION", "conversations": ["x"]}
    result = validate_data_format(_write_jsonl(tmp_path, [sample]))
    messages = [e["message"] for e in result["validation_results"]]
    assert any("必须是字典" in m for m in messages)


def test_conversation_missing_keys(tmp_path):
    _write_image(tmp_path / "image.png")
    sample = {
        "image": "image.png",
        "task_type": "CAPTION",
        "conversations": [{"from": "human"}],
    }
    result = validate_data_format(_write_jsonl(tmp_path, [sample]))
    messages = [e["message"] for e in result["validation_results"]]
    assert any("缺少'from'或'value'字段" in m for m in messages)


def test_conversation_unexpected_from_value_warns(tmp_path):
    _write_image(tmp_path / "image.png")
    sample = {
        "image": "image.png",
        "task_type": "CAPTION",
        "conversations": [
            {"from": "system", "value": "hi"},
            {"from": "gpt", "value": "A small red square."},
        ],
    }
    result = validate_data_format(_write_jsonl(tmp_path, [sample]))
    messages = [e["message"] for e in result["validation_results"]]
    assert any("值异常" in m for m in messages)


def test_no_gpt_response_warns(tmp_path):
    _write_image(tmp_path / "image.png")
    sample = {
        "image": "image.png",
        "task_type": "CAPTION",
        "conversations": [{"from": "human", "value": "Describe."}],
    }
    result = validate_data_format(_write_jsonl(tmp_path, [sample]))
    messages = [e["message"] for e in result["validation_results"]]
    assert any("没有找到模型回答" in m for m in messages)


# ---------------------------------------------------------------------------
# Image checks
# ---------------------------------------------------------------------------


def test_small_image_emits_warning(tmp_path):
    Image.new("RGB", (16, 16)).save(tmp_path / "image.png")
    result = validate_data_format(_write_jsonl(tmp_path, [_valid_sample()]))
    messages = [e["message"] for e in result["validation_results"]]
    assert any("图像尺寸过小" in m for m in messages)


def test_corrupt_image_reports_error(tmp_path):
    (tmp_path / "image.png").write_bytes(b"not an image")
    result = validate_data_format(_write_jsonl(tmp_path, [_valid_sample()]))
    messages = [e["message"] for e in result["validation_results"]]
    assert result["is_valid"] is False
    assert any("无法读取图像" in m for m in messages)


def test_many_missing_images_summarized(tmp_path):
    samples = [_valid_sample(f"missing_{i}.png") for i in range(12)]
    result = validate_data_format(_write_jsonl(tmp_path, samples))
    messages = [e["message"] for e in result["validation_results"]]
    assert result["missing_images"] == 12
    assert any("还有2个样本的图像文件缺失" in m for m in messages)


# ---------------------------------------------------------------------------
# Task-specific content checks
# ---------------------------------------------------------------------------


def _task_sample(task_type, response, image_name="image.png"):
    return {
        "image": image_name,
        "task_type": task_type,
        "conversations": [
            {"from": "human", "value": "Q"},
            {"from": "gpt", "value": response},
        ],
    }


def test_detection_coord_out_of_range_warns(tmp_path):
    _write_image(tmp_path / "image.png")
    sample = _task_sample("object_detection", "cat<loc_1200><loc_10><loc_20><loc_30>")
    result = validate_data_format(_write_jsonl(tmp_path, [sample]))
    messages = [e["message"] for e in result["validation_results"]]
    assert any("检测坐标超出范围" in m for m in messages)


def test_detection_unrecognized_format_warns(tmp_path):
    _write_image(tmp_path / "image.png")
    sample = _task_sample("object_detection", "just some free text")
    result = validate_data_format(_write_jsonl(tmp_path, [sample]))
    messages = [e["message"] for e in result["validation_results"]]
    assert any("检测结果格式无法识别" in m for m in messages)


def test_caption_too_long_warns(tmp_path):
    _write_image(tmp_path / "image.png")
    sample = _task_sample("CAPTION", "x" * 600)
    result = validate_data_format(_write_jsonl(tmp_path, [sample]))
    messages = [e["message"] for e in result["validation_results"]]
    assert any("图像描述过长" in m for m in messages)


def test_ocr_empty_result_warns(tmp_path):
    _write_image(tmp_path / "image.png")
    sample = _task_sample("OCR", "   ")
    result = validate_data_format(_write_jsonl(tmp_path, [sample]))
    messages = [e["message"] for e in result["validation_results"]]
    assert any("OCR结果为空" in m for m in messages)


def test_segmentation_format_warns(tmp_path):
    _write_image(tmp_path / "image.png")
    sample = _task_sample("REGION_TO_SEGMENTATION", "no polygon markup here")
    result = validate_data_format(_write_jsonl(tmp_path, [sample]))
    messages = [e["message"] for e in result["validation_results"]]
    assert any("分割结果格式可能不正确" in m for m in messages)
