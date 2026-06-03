"""测试数据管线功能"""
import pytest
import json
import tempfile
from pathlib import Path
from collections import OrderedDict
from PIL import Image
import numpy as np
import torch

from florence_forge.data.dataset import MultiTaskDataset, TaskSample, _load_image_cached
from florence_forge.data.builder import DatasetBuilder
from florence_forge.core.config import DataConfig


@pytest.fixture
def temp_image():
    """创建临时测试图像"""
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        img = Image.new('RGB', (100, 100), color='red')
        img.save(f.name)
        yield f.name
    Path(f.name).unlink()


@pytest.fixture
def temp_jsonl_data(temp_image):
    """创建临时 JSONL 数据文件"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        data = [
            {"image": Path(temp_image).name, "prefix": "<CAPTION>", "suffix": "A red image"},
            {"image": Path(temp_image).name, "prefix": "<OD>", "suffix": "<loc_10><loc_20>cat<loc_50><loc_60>"}
        ]
        for item in data:
            f.write(json.dumps(item) + '\n')
        f.flush()
        yield f.name
    Path(f.name).unlink()


class TestTaskSample:
    """测试 TaskSample 数据结构"""
    
    def test_task_sample_creation(self):
        """测试 TaskSample 创建"""
        sample = TaskSample(
            task_type="CAPTION",
            image_path="/path/to/image.jpg",
            prefix="<CAPTION>",
            suffix="A beautiful scene",
            weight=1.0
        )
        assert sample.task_type == "CAPTION"
        assert sample.weight == 1.0
    
    def test_task_sample_to_dict(self):
        """测试转换为字典"""
        sample = TaskSample(
            task_type="CAPTION",
            image_path="/path/to/image.jpg",
            prefix="<CAPTION>",
            suffix="A beautiful scene"
        )
        data = sample.to_dict()
        assert data["task_type"] == "CAPTION"
        assert "metadata" in data
    
    def test_task_sample_from_dict(self):
        """测试从字典创建"""
        data = {
            "task_type": "OD",
            "image_path": "/path/to/image.jpg",
            "prefix": "<OD>",
            "suffix": "<loc_10><loc_20>cat",
            "weight": 2.0
        }
        sample = TaskSample.from_dict(data)
        assert sample.task_type == "OD"
        assert sample.weight == 2.0


def test_dataset_builder_get_statistics_empty_builder():
    builder = DatasetBuilder()

    stats = builder.get_statistics()

    assert stats["num_tasks"] == 0
    assert stats["estimated_samples"] == {}


class TestMultiTaskDataset:
    """测试 MultiTaskDataset"""
    
    def test_dataset_initialization(self, temp_jsonl_data, temp_image):
        """测试数据集初始化"""
        config = DataConfig()
        data_configs = [
            {
                "task_type": "CAPTION",
                "data_path": temp_jsonl_data,
                "weight": 1.0
            }
        ]
        
        dataset = MultiTaskDataset(
            data_configs=data_configs,
            image_base_path=str(Path(temp_image).parent),
            config=config
        )
        
        assert len(dataset) > 0
        assert "CAPTION" in dataset.task_indices
    
    def test_dataset_getitem(self, temp_jsonl_data, temp_image):
        """测试获取样本"""
        config = DataConfig()
        data_configs = [
            {
                "task_type": "CAPTION",
                "data_path": temp_jsonl_data,
                "weight": 1.0
            }
        ]
        
        dataset = MultiTaskDataset(
            data_configs=data_configs,
            image_base_path=str(Path(temp_image).parent),
            config=config
        )
        
        sample = dataset[0]
        assert "image" in sample or "pixel_values" in sample
        assert "task_type" in sample
        assert sample["task_type"] == "CAPTION"

    def test_from_hf_dataset_accepts_path_rows(self, temp_image):
        rows = [
            {
                "image": Path(temp_image).name,
                "caption": "A red image",
                "id": "row-1",
            }
        ]

        dataset = MultiTaskDataset.from_hf_dataset(
            rows,
            task_type="CAPTION",
            image_column="image",
            text_column="caption",
            image_base_path=str(Path(temp_image).parent),
        )

        assert len(dataset) == 1
        assert dataset.samples[0].suffix == "A red image"
        assert dataset.samples[0].metadata["source"] == "hf_dataset"
        assert dataset.samples[0].metadata["id"] == "row-1"
        assert "CAPTION" in dataset.task_indices

    def test_from_hf_dataset_materializes_pil_images(self, tmp_path):
        image = Image.new("RGB", (16, 16), color="blue")
        config = DataConfig(cache_dir=str(tmp_path / "cache"))

        dataset = MultiTaskDataset.from_hf_dataset(
            [{"image": image, "text": "A blue square"}],
            task_type="CAPTION",
            config=config,
        )

        image_path = Path(dataset.samples[0].image_path)
        assert image_path.exists()
        assert image_path.parent == tmp_path / "cache" / "hf_images"
        assert dataset[0]["answer"] == "A blue square"

    def test_encoded_prompt_preserves_prefix_extras_without_double_image_processing(self, temp_image):
        """编码时应保留 prefix/text_input，且图像只经 processor 处理一次。"""
        class RecordingTokenizer:
            def __call__(self, text, return_tensors="pt", add_special_tokens=True):
                length = max(1, len(text))
                return {
                    "input_ids": torch.arange(length).unsqueeze(0),
                    "attention_mask": torch.ones(1, length, dtype=torch.long),
                }

        class RecordingProcessor:
            def __init__(self):
                self.calls = []
                self.tokenizer = RecordingTokenizer()

            def __call__(self, text=None, images=None, return_tensors="pt"):
                self.calls.append({"text": text, "has_images": images is not None})
                length = max(1, len(text or ""))
                return {
                    "input_ids": torch.arange(length).unsqueeze(0),
                    "attention_mask": torch.ones(1, length, dtype=torch.long),
                    "pixel_values": torch.zeros(1, 3, 224, 224),
                }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            row = {
                "image": Path(temp_image).name,
                "prefix": "<CAPTION_TO_PHRASE_GROUNDING>",
                "suffix": "answer",
                "text_input": "a red square",
            }
            f.write(json.dumps(row) + "\n")
            data_path = f.name

        try:
            processor = RecordingProcessor()
            dataset = MultiTaskDataset(
                data_configs=[{
                    "task_type": "CAPTION_TO_PHRASE_GROUNDING",
                    "data_path": data_path,
                    "weight": 1.0,
                }],
                image_base_path=str(Path(temp_image).parent),
                config=DataConfig(),
                processor=processor,
            )

            item = dataset[0]
            prompt = "<CAPTION_TO_PHRASE_GROUNDING>a red square"
            assert item["prompt"] == prompt
            assert processor.calls == [{"text": f"{prompt}answer", "has_images": True}]
            assert torch.all(item["labels"][:len(prompt)] == -100)
            assert torch.equal(item["labels"][len(prompt):], item["input_ids"][len(prompt):])
        finally:
            Path(data_path).unlink()
    
    def test_lazy_load_mode(self, temp_jsonl_data, temp_image):
        """测试延迟加载模式"""
        config = DataConfig()
        data_configs = [
            {
                "task_type": "CAPTION",
                "data_path": temp_jsonl_data,
                "weight": 1.0
            }
        ]
        
        dataset = MultiTaskDataset(
            data_configs=data_configs,
            image_base_path=str(Path(temp_image).parent),
            config=config,
            lazy_load=True
        )
        
        assert len(dataset._sample_index) > 0
        assert len(dataset._sample_offset_cache) > 0
        
        # 测试按需加载
        sample = dataset[0]
        assert "task_type" in sample
    
    def test_cache_operations(self, temp_jsonl_data, temp_image):
        """测试缓存操作（验证 OrderedDict LRU）"""
        config = DataConfig()
        config.use_cache = True
        config.cache_max_size = 2
        
        data_configs = [
            {
                "task_type": "CAPTION",
                "data_path": temp_jsonl_data,
                "weight": 1.0
            }
        ]
        
        dataset = MultiTaskDataset(
            data_configs=data_configs,
            image_base_path=str(Path(temp_image).parent),
            config=config
        )
        
        # 验证缓存是 OrderedDict
        assert isinstance(dataset._cache_index, OrderedDict)
        
        # 添加多个条目测试 LRU
        dataset._cache_put(0, {"data": "sample0"})
        dataset._cache_put(1, {"data": "sample1"})
        assert len(dataset._cache_index) == 2
        
        # 添加第三个条目应触发淘汰
        dataset._cache_put(2, {"data": "sample2"})
        assert len(dataset._cache_index) == 2
        assert 0 not in dataset._cache_index  # 最老的条目被淘汰
        assert 1 in dataset._cache_index
        assert 2 in dataset._cache_index
    
    def test_task_statistics(self, temp_jsonl_data, temp_image):
        """测试任务统计"""
        config = DataConfig()
        data_configs = [
            {
                "task_type": "CAPTION",
                "data_path": temp_jsonl_data,
                "weight": 1.0
            }
        ]
        
        dataset = MultiTaskDataset(
            data_configs=data_configs,
            image_base_path=str(Path(temp_image).parent),
            config=config
        )
        
        stats = dataset.get_task_statistics()
        assert "total_samples" in stats
        assert "task_counts" in stats
        assert "CAPTION" in stats["task_counts"]

    def test_create_task_subset_preserves_runtime_fields(self, temp_jsonl_data, temp_image):
        """任务子集应保留评估时依赖的运行时字段。"""
        config = DataConfig()
        data_configs = [
            {
                "task_type": "CAPTION",
                "data_path": temp_jsonl_data,
                "weight": 1.0
            }
        ]

        dataset = MultiTaskDataset(
            data_configs=data_configs,
            image_base_path=str(Path(temp_image).parent),
            config=config
        )

        subset = dataset.create_task_subset("CAPTION", max_samples=1)
        assert hasattr(subset, "collate_fn")
        assert subset.collate_fn is dataset.collate_fn
        assert hasattr(subset, "_sample_offset_cache")


class TestImageCaching:
    """测试图像加载缓存"""
    
    def test_image_cached_load(self, temp_image):
        """测试图像 LRU 缓存"""
        # 清除缓存统计
        _load_image_cached.cache_clear()
        
        # 第一次加载
        img1 = _load_image_cached(temp_image)
        assert isinstance(img1, Image.Image)
        
        # 第二次加载应该命中缓存
        img2 = _load_image_cached(temp_image)
        assert img1 is not img2  # 返回独立对象，避免共享可变 PIL Image
        assert list(img1.getdata()) == list(img2.getdata())
        
        # 检查缓存统计
        cache_info = _load_image_cached.cache_info()
        assert cache_info.hits == 1
        assert cache_info.misses == 1

    def test_image_cache_returns_fresh_image_objects(self, temp_image):
        """修改一次返回的 Image 不应污染缓存。"""
        _load_image_cached.cache_clear()

        img1 = _load_image_cached(temp_image)
        img1.putpixel((0, 0), (0, 0, 0))

        img2 = _load_image_cached(temp_image)

        assert img2.getpixel((0, 0)) != (0, 0, 0)

    def test_image_cache_evicts_by_byte_budget(self, tmp_path):
        """图像 payload 缓存应按字节预算淘汰，而不是只按条数增长。"""
        img_a = tmp_path / "a.png"
        img_b = tmp_path / "b.png"
        Image.new("RGB", (10, 10), color="red").save(img_a)
        Image.new("RGB", (10, 10), color="blue").save(img_b)

        old_budget = _load_image_cached.set_cache_max_bytes(500)
        try:
            _load_image_cached.cache_clear()
            _load_image_cached(str(img_a))
            _load_image_cached(str(img_b))

            assert _load_image_cached.cache_bytes() <= 500

            _load_image_cached(str(img_a))
            cache_info = _load_image_cached.cache_info()
            assert cache_info.misses == 3
            assert cache_info.hits == 0
            assert cache_info.currsize == 1
        finally:
            _load_image_cached.set_cache_max_bytes(old_budget)
            _load_image_cached.cache_clear()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
