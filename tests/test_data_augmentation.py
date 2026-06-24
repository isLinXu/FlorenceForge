"""测试数据增强接入 MultiTaskDataset 的行为。

覆盖 P1-1: 数据增强从死代码接入到 __getitem__。
"""
import json
import tempfile
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]

from PIL import Image

from florence_forge.data.augmentation import (
    BBoxAugmentation,
    ImageAugmentation,
    TextAugmentation,
)
from florence_forge.data.dataset import MultiTaskDataset
from florence_forge.core.config import DataConfig


@pytest.fixture
def temp_image():
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        Image.new("RGB", (64, 64), color="blue").save(f.name)
        yield f.name
    Path(f.name).unlink()


@pytest.fixture
def temp_jsonl_data(temp_image):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        data = [
            {"image": Path(temp_image).name, "prefix": "<CAPTION>", "suffix": "a blue square"},
            {"image": Path(temp_image).name, "prefix": "<CAPTION>", "suffix": "another caption"},
        ]
        for item in data:
            f.write(json.dumps(item) + "\n")
        f.flush()
        yield f.name
    Path(f.name).unlink()


def _make_dataset(temp_jsonl_data, temp_image, config):
    return MultiTaskDataset(
        data_configs=[{"task_type": "CAPTION", "data_path": temp_jsonl_data, "weight": 1.0}],
        image_base_path=str(Path(temp_image).parent),
        config=config,
    )


class TestAugmentationUnits:
    """增强类的单元行为（probability=1.0 确保确定性触发）。"""

    def test_image_augmentation_returns_image(self, temp_image):
        aug = ImageAugmentation(probability=1.0)
        img = Image.open(temp_image).convert("RGB")
        out = aug.apply_augmentations(img)
        assert isinstance(out, Image.Image)
        assert out.size == img.size

    def test_text_augmentation_returns_str(self):
        aug = TextAugmentation(probability=1.0)
        out = aug.apply_augmentations("hello world foo bar")
        assert isinstance(out, str)

    def test_bbox_augmentation_keeps_normalized_range(self):
        aug = BBoxAugmentation(probability=1.0)
        boxes = [{"xmin": 0.1, "ymin": 0.1, "xmax": 0.5, "ymax": 0.5}]
        out = aug.apply_augmentations(boxes)
        for bb in out:
            for v in bb.values():
                assert 0.0 <= v <= 1.0


class TestDatasetAugmentationWiring:
    """验证增强被正确接入 MultiTaskDataset。"""

    def test_disabled_by_default(self, temp_jsonl_data, temp_image):
        config = DataConfig()  # use_augmentation 默认 False
        ds = _make_dataset(temp_jsonl_data, temp_image, config)
        assert ds._image_aug is None
        assert ds._text_aug is None

    def test_image_aug_enabled(self, temp_jsonl_data, temp_image):
        config = DataConfig(use_augmentation=True, use_cache=False)
        ds = _make_dataset(temp_jsonl_data, temp_image, config)
        assert ds._image_aug is not None
        # text 默认关闭（会破坏结构化标签）
        assert ds._text_aug is None

    def test_text_aug_opt_in(self, temp_jsonl_data, temp_image):
        config = DataConfig(
            use_augmentation=True, augment_text=True, use_cache=False
        )
        ds = _make_dataset(temp_jsonl_data, temp_image, config)
        assert ds._text_aug is not None

    def test_image_aug_can_be_disabled(self, temp_jsonl_data, temp_image):
        config = DataConfig(
            use_augmentation=True, augment_image=False, use_cache=False
        )
        ds = _make_dataset(temp_jsonl_data, temp_image, config)
        assert ds._image_aug is None

    def test_augmentation_disabled_under_cache(self, temp_jsonl_data, temp_image):
        # 缓存模式下增强应被自动禁用以保证确定性
        config = DataConfig(use_augmentation=True, use_cache=True)
        ds = _make_dataset(temp_jsonl_data, temp_image, config)
        assert ds._image_aug is None
        assert ds._text_aug is None

    def test_text_aug_does_not_mutate_shared_sample(self, temp_jsonl_data, temp_image):
        """文本增强必须用副本，不能污染 self.samples[idx]。"""
        config = DataConfig(
            use_augmentation=True,
            augment_text=True,
            augmentation_prob=1.0,
            use_cache=False,
        )
        ds = _make_dataset(temp_jsonl_data, temp_image, config)
        original_suffix = ds.samples[0].suffix
        # 多次取样
        for _ in range(5):
            _ = ds[0]
        # 原始 sample 的 suffix 不应被增强修改
        assert ds.samples[0].suffix == original_suffix

    def test_getitem_returns_valid_sample_with_aug(self, temp_jsonl_data, temp_image):
        config = DataConfig(use_augmentation=True, augmentation_prob=1.0, use_cache=False)
        ds = _make_dataset(temp_jsonl_data, temp_image, config)
        sample = ds[0]
        assert "image" in sample or "pixel_values" in sample
        assert sample["task_type"] == "CAPTION"
