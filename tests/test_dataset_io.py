"""dataset_io 模块单元测试。"""

import json
from pathlib import Path

from florence_forge.data import dataset_io
from florence_forge.data.dataset_types import TaskSample


def test_load_and_scan_jsonl_roundtrip(tmp_path):
    jsonl = tmp_path / "data.jsonl"
    jsonl.write_text(
        "\n".join(
            [
                json.dumps({"image": "a.jpg", "prefix": "", "suffix": "one"}),
                json.dumps({"image": "b.jpg", "prefix": "p", "suffix": "two"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "b.jpg").write_bytes(b"y")

    samples = []
    count = dataset_io.load_jsonl_task(
        samples,
        task_type="CAPTION",
        data_path=str(jsonl),
        image_base_path=tmp_path,
        weight=1.0,
    )
    assert count == 2
    assert samples[0].suffix == "one"

    index = []
    offsets = {}
    scanned = dataset_io.scan_jsonl_task(
        index,
        offsets,
        task_type="CAPTION",
        data_path=str(jsonl),
        weight=1.0,
    )
    assert scanned == 2
    loaded = dataset_io.load_jsonl_sample_by_index(index, offsets, tmp_path, 1)
    assert loaded.suffix == "two"


def test_persist_and_restore_dataset_json(tmp_path):
    sample = TaskSample(
        task_type="CAPTION",
        image_path="img.jpg",
        prefix="",
        suffix="cat",
    )
    out = tmp_path / "ds.json"
    from florence_forge.core.config import DataConfig

    dataset_io.persist_dataset_json(
        out,
        data_configs=[{"task_type": "CAPTION", "data_path": "x.jsonl", "weight": 1.0}],
        image_base_path=Path("/data"),
        config=DataConfig(),
        samples_data=[sample.to_dict()],
        task_weights={"CAPTION": 1.0},
    )
    restored = dataset_io.restore_dataset_json(out)
    assert restored["samples"][0]["suffix"] == "cat"
