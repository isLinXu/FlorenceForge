"""DataProfiler 单元测试。"""

import json

import pytest
from PIL import Image

from florence_forge.core.config import DataConfig
from florence_forge.data.dataset import MultiTaskDataset
from florence_forge.data.profiler import DataProfiler

pytestmark = [pytest.mark.unit]


@pytest.fixture
def sample_bundle(tmp_path):
    img_a = tmp_path / "a.jpg"
    img_b = tmp_path / "b.jpg"
    Image.new("RGB", (100, 200), color="red").save(img_a)
    Image.new("RGB", (300, 150), color="blue").save(img_b)

    jsonl = tmp_path / "data.jsonl"
    rows = [
        {"image": "a.jpg", "prefix": "<CAPTION>", "suffix": "short"},
        {"image": "b.jpg", "prefix": "<CAPTION>", "suffix": "a longer caption text"},
        {"image": "a.jpg", "prefix": "<OD>", "suffix": "box1"},
    ]
    with jsonl.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return tmp_path, jsonl


def test_profile_jsonl_reports_task_and_image_stats(sample_bundle):
    base, jsonl = sample_bundle
    report = DataProfiler().profile_jsonl(jsonl, image_base_path=base)

    assert report["total_samples"] == 3
    assert report["task_distribution"]["UNKNOWN"] == 3
    assert report["image_width"]["count"] == 3
    assert report["aspect_ratio"]["count"] == 3
    assert report["duplicate_samples"] == 0


def test_profile_dataset_detects_imbalance(sample_bundle):
    base, jsonl = sample_bundle
    config = DataConfig()
    dataset = MultiTaskDataset(
        data_configs=[
            {"task_type": "CAPTION", "data_path": str(jsonl), "weight": 1.0},
        ],
        image_base_path=str(base),
        config=config,
    )
    report = DataProfiler(imbalance_threshold=1.5).profile_dataset(dataset)

    assert report["total_samples"] == 3
    assert "CAPTION" in report["task_distribution"]
    assert report["suffix_length"]["count"] == 3


def test_profile_jsonl_flags_duplicates(sample_bundle):
    base, jsonl = sample_bundle
    dup_path = base / "dup.jsonl"
    row = {"image": "a.jpg", "prefix": "<CAPTION>", "suffix": "same"}
    with dup_path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.write(json.dumps(row) + "\n")

    report = DataProfiler().profile_jsonl(dup_path, image_base_path=base)
    assert report["duplicate_samples"] == 1
    assert any("重复" in w for w in report["warnings"])
