"""Tests for PaliGemma data conversion helpers."""

import json
import importlib.util
from pathlib import Path

from PIL import Image


def _load_converter():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "data-conversion"
        / "convert_florence_od_to_paligemma.py"
    )
    spec = importlib.util.spec_from_file_location(
        "convert_florence_od_to_paligemma",
        script_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_convert_florence_od_to_paligemma_detection(tmp_path):
    image_path = tmp_path / "image.jpg"
    Image.new("RGB", (100, 200), color="white").save(image_path)

    input_jsonl = tmp_path / "florence.jsonl"
    output_jsonl = tmp_path / "paligemma.jsonl"
    record = {
        "image": str(image_path),
        "prefix": "<OD>",
        "suffix": json.dumps(
            {
                "<OD>": {
                    "bboxes": [[10, 20, 60, 120]],
                    "labels": ["cat"],
                }
            }
        ),
    }
    input_jsonl.write_text(json.dumps(record) + "\n", encoding="utf-8")

    converter = _load_converter()
    count = converter.convert_file(input_jsonl, output_jsonl)

    assert count == 1
    converted = json.loads(output_jsonl.read_text(encoding="utf-8"))
    assert converted["prefix"] == "detect"
    assert converted["suffix"] == "<loc0102><loc0102><loc0614><loc0614> cat"
    assert converted["source_format"] == "florence_od"
    assert converted["target_format"] == "paligemma_detection"
