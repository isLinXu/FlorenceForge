"""DatasetSampleCache 单元测试。"""

import torch
from collections import OrderedDict

from florence_forge.data.dataset_sample_cache import DatasetSampleCache
from florence_forge.data.dataset_types import TaskSample


def test_memory_lru_eviction():
    cache = DatasetSampleCache(
        use_cache=True,
        cache_dir=None,
        cache_max_size=2,
        lazy_load=False,
        sample_index=[],
        samples=[],
    )
    cache.put_memory(0, {"a": 1})
    cache.put_memory(1, {"b": 2})
    cache.put_memory(2, {"c": 3})

    assert cache.get_memory(0) is None
    assert cache.get_memory(1)["b"] == 2
    assert cache.get_memory(2)["c"] == 3


def test_resolve_disk_path_uses_sample_metadata(tmp_path):
    samples = [
        TaskSample(
            task_type="CAPTION",
            image_path="img.jpg",
            prefix="",
            suffix="a cat",
            metadata={"source_file": "data.jsonl"},
        )
    ]
    cache = DatasetSampleCache(
        use_cache=True,
        cache_dir=str(tmp_path / "cache"),
        cache_max_size=10,
        lazy_load=False,
        sample_index=[],
        samples=samples,
    )
    path = cache.resolve_disk_path(0)
    assert path.parent.name != ""
    assert path.name == "sample_0.pt"


def test_save_and_load_disk_excludes_pixel_values(tmp_path):
    samples = [
        TaskSample(
            task_type="CAPTION",
            image_path="img.jpg",
            prefix="",
            suffix="dog",
        )
    ]
    cache_dir = tmp_path / "cache"
    cache = DatasetSampleCache(
        use_cache=True,
        cache_dir=str(cache_dir),
        cache_max_size=10,
        lazy_load=False,
        sample_index=[],
        samples=samples,
    )
    payload = {
        "input_ids": torch.tensor([1, 2, 3]),
        "pixel_values": torch.randn(3, 4, 4),
    }
    cache_path = cache.resolve_disk_path(0)
    cache.save_disk(payload, cache_path)

    from florence_forge.utils.torch_serialization import safe_torch_load_cpu

    loaded = safe_torch_load_cpu(cache_path, context="test")
    assert "pixel_values" not in loaded
    assert loaded["input_ids"].tolist() == [1, 2, 3]
