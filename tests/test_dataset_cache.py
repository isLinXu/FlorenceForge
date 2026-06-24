"""测试 MultiTaskDataset 的预编码缓存功能

覆盖缓存初始化、预编码、磁盘缓存读写、清理等逻辑。
"""

import pytest
import json
import tempfile
import pickle
from pathlib import Path
from unittest.mock import MagicMock

import torch

from florence_forge.data.dataset import MultiTaskDataset
from florence_forge.core.config import DataConfig


class TestDatasetCache:
    def _create_dummy_jsonl(self, tmpdir, n_samples=3):
        path = Path(tmpdir) / "data.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for i in range(n_samples):
                item = {
                    "image": f"images/img_{i}.jpg",  # 与 _create_dummy_images 路径一致
                    "prefix": "",
                    "suffix": f"caption {i}",
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        return path

    def _create_dummy_images(self, tmpdir, n_samples=3):
        from PIL import Image
        img_dir = Path(tmpdir) / "images"
        img_dir.mkdir(exist_ok=True)
        for i in range(n_samples):
            img = Image.new("RGB", (100, 100), color=(i * 50, 0, 0))
            img.save(img_dir / f"img_{i}.jpg")
        return img_dir

    def test_cache_disabled_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = self._create_dummy_jsonl(tmpdir)
            self._create_dummy_images(tmpdir)
            cfg = DataConfig()
            dataset = MultiTaskDataset(
                data_configs=[{"task_type": "CAPTION", "data_path": str(data_path), "weight": 1.0}],
                image_base_path=tmpdir,
                config=cfg,
                processor=None,
            )
            assert dataset.use_cache is False
            assert len(dataset._cache_index) == 0

    def test_memory_cache_with_mock_processor(self):
        """使用 mock processor 测试内存缓存"""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = self._create_dummy_jsonl(tmpdir, n_samples=2)
            self._create_dummy_images(tmpdir, n_samples=2)

            mock_processor = MagicMock()
            mock_processor.return_value = {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
                "pixel_values": torch.randn(1, 3, 224, 224),
            }

            cfg = DataConfig()
            cfg.use_cache = True
            dataset = MultiTaskDataset(
                data_configs=[{"task_type": "CAPTION", "data_path": str(data_path), "weight": 1.0}],
                image_base_path=tmpdir,
                config=cfg,
                processor=mock_processor,
            )
            assert dataset.use_cache is True
            assert len(dataset._cache_index) == 2
            # 再次获取应从缓存命中
            item = dataset[0]
            assert item is not None
            assert 0 in dataset._cache_index

    def test_dataset_collator_uses_processor_pad_token_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = self._create_dummy_jsonl(tmpdir, n_samples=1)
            self._create_dummy_images(tmpdir, n_samples=1)

            mock_processor = MagicMock()
            mock_processor.tokenizer.pad_token_id = 7

            dataset = MultiTaskDataset(
                data_configs=[{"task_type": "CAPTION", "data_path": str(data_path), "weight": 1.0}],
                image_base_path=tmpdir,
                config=DataConfig(),
                processor=mock_processor,
            )

            assert dataset.collate_fn.pad_token_id == 7

    def test_disk_cache_roundtrip(self):
        """测试磁盘缓存读写"""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = self._create_dummy_jsonl(tmpdir, n_samples=2)
            self._create_dummy_images(tmpdir, n_samples=2)
            cache_dir = Path(tmpdir) / "cache"

            mock_processor = MagicMock()
            mock_processor.return_value = {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
                "pixel_values": torch.randn(1, 3, 224, 224),
            }

            cfg = DataConfig()
            cfg.use_cache = True
            cfg.cache_dir = str(cache_dir)

            dataset = MultiTaskDataset(
                data_configs=[{"task_type": "CAPTION", "data_path": str(data_path), "weight": 1.0}],
                image_base_path=tmpdir,
                config=cfg,
                processor=mock_processor,
            )
            assert len(dataset._cache_index) == 2
            # 检查磁盘缓存文件是否生成
            assert any(cache_dir.iterdir())

            # 创建新数据集实例，验证能从磁盘缓存加载
            dataset2 = MultiTaskDataset(
                data_configs=[{"task_type": "CAPTION", "data_path": str(data_path), "weight": 1.0}],
                image_base_path=tmpdir,
                config=cfg,
                processor=mock_processor,
            )
            assert len(dataset2._cache_index) == 2

    def test_disk_cache_excludes_pixel_values_and_restores_them(self):
        """磁盘缓存不应持久化大体积 pixel_values，加载时可按需补回。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = self._create_dummy_jsonl(tmpdir, n_samples=1)
            self._create_dummy_images(tmpdir, n_samples=1)
            cache_dir = Path(tmpdir) / "cache"

            mock_processor = MagicMock()
            mock_processor.return_value = {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
                "pixel_values": torch.ones(1, 3, 16, 16),
            }

            cfg = DataConfig(use_cache=True, cache_dir=str(cache_dir))
            dataset = MultiTaskDataset(
                data_configs=[{"task_type": "CAPTION", "data_path": str(data_path), "weight": 1.0}],
                image_base_path=tmpdir,
                config=cfg,
                processor=mock_processor,
            )

            cache_path = dataset._get_cache_path(0)
            cached_on_disk = torch.load(cache_path, map_location="cpu", weights_only=True)
            assert "pixel_values" not in cached_on_disk

            dataset._cache_index.clear()
            restored = dataset._load_cached_sample(0, cache_path)
            assert "pixel_values" in restored
            assert restored["pixel_values"].shape == (3, 16, 16)

    def test_disk_cache_without_processor_falls_back_to_raw_sample(self):
        """worker 中没有 processor 时，不应返回缺少 pixel_values 的半编码缓存。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = self._create_dummy_jsonl(tmpdir, n_samples=1)
            self._create_dummy_images(tmpdir, n_samples=1)
            cache_dir = Path(tmpdir) / "cache"

            mock_processor = MagicMock()
            mock_processor.return_value = {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
                "pixel_values": torch.ones(1, 3, 16, 16),
            }

            cfg = DataConfig(use_cache=False, cache_dir=str(cache_dir))
            dataset = MultiTaskDataset(
                data_configs=[{"task_type": "CAPTION", "data_path": str(data_path), "weight": 1.0}],
                image_base_path=tmpdir,
                config=cfg,
                processor=mock_processor,
            )
            _ = dataset[0]
            dataset.processor = None
            dataset._cache_index.clear()

            item = dataset[0]

            assert item["_needs_encoding"] is True
            assert "pixel_values" not in item

    def test_disk_cache_restore_requires_pixel_values_from_processor(self):
        """缓存恢复阶段 processor 未返回 pixel_values 时应明确失败。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = self._create_dummy_jsonl(tmpdir, n_samples=1)
            self._create_dummy_images(tmpdir, n_samples=1)
            cache_dir = Path(tmpdir) / "cache"

            good_processor = MagicMock()
            good_processor.return_value = {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
                "pixel_values": torch.ones(1, 3, 16, 16),
            }

            cfg = DataConfig(use_cache=False, cache_dir=str(cache_dir))
            dataset = MultiTaskDataset(
                data_configs=[{"task_type": "CAPTION", "data_path": str(data_path), "weight": 1.0}],
                image_base_path=tmpdir,
                config=cfg,
                processor=good_processor,
            )
            _ = dataset[0]
            dataset._cache_index.clear()

            bad_processor = MagicMock()
            bad_processor.return_value = {"input_ids": torch.tensor([[1]])}
            dataset.processor = bad_processor

            with pytest.raises(RuntimeError, match="pixel_values"):
                dataset._load_cached_sample(0, dataset._get_cache_path(0))

    def test_backend_encoding_path_writes_disk_cache(self):
        """backend.encode_with_task 成功时也应执行统一缓存写入。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = self._create_dummy_jsonl(tmpdir, n_samples=1)
            self._create_dummy_images(tmpdir, n_samples=1)
            cache_dir = Path(tmpdir) / "cache"

            mock_processor = MagicMock()
            mock_processor.return_value = {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
                "pixel_values": torch.ones(1, 3, 16, 16),
            }

            class Backend:
                def get_task_prompt(self, task_type):
                    return f"<{task_type}>"

                def encode_with_task(self, **kwargs):
                    return {
                        "input_ids": torch.tensor([[1, 2, 3, 4]]),
                        "pixel_values": torch.ones(1, 3, 16, 16),
                    }

                def prepare_labels(self, encoded_prompt, encoded_full):
                    labels = encoded_full["input_ids"].clone()
                    labels[:, :2] = -100
                    return labels

            cfg = DataConfig(use_cache=False, cache_dir=str(cache_dir))
            dataset = MultiTaskDataset(
                data_configs=[{"task_type": "CAPTION", "data_path": str(data_path), "weight": 1.0}],
                image_base_path=tmpdir,
                config=cfg,
                processor=mock_processor,
                backend=Backend(),
            )

            item = dataset[0]
            cache_path = dataset._get_cache_path(0)
            cached_on_disk = torch.load(cache_path, map_location="cpu", weights_only=True)

            assert item["metadata"]["source_file"] == str(data_path)
            assert cache_path.exists()
            assert "pixel_values" not in cached_on_disk

    def test_clear_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = self._create_dummy_jsonl(tmpdir, n_samples=1)
            self._create_dummy_images(tmpdir, n_samples=1)
            cache_dir = Path(tmpdir) / "cache"

            mock_processor = MagicMock()
            mock_processor.return_value = {
                "input_ids": torch.tensor([[1, 2]]),
                "attention_mask": torch.tensor([[1, 1]]),
                "pixel_values": torch.randn(1, 3, 224, 224),
            }

            cfg = DataConfig()
            cfg.use_cache = True
            cfg.cache_dir = str(cache_dir)

            dataset = MultiTaskDataset(
                data_configs=[{"task_type": "CAPTION", "data_path": str(data_path), "weight": 1.0}],
                image_base_path=tmpdir,
                config=cfg,
                processor=mock_processor,
            )
            assert len(dataset._cache_index) == 1
            dataset.clear_cache()
            assert len(dataset._cache_index) == 0
            assert not cache_dir.exists()

    def test_runtime_cache_fields_survive_subset_pickle_and_file_load(self):
        """手工构造/反序列化的数据集也应具备缓存运行时字段。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = self._create_dummy_jsonl(tmpdir, n_samples=2)
            self._create_dummy_images(tmpdir, n_samples=2)
            cfg = DataConfig(use_cache=True, cache_max_size=1)

            dataset = MultiTaskDataset(
                data_configs=[{"task_type": "CAPTION", "data_path": str(data_path), "weight": 1.0}],
                image_base_path=tmpdir,
                config=cfg,
                processor=None,
            )

            subset = dataset.create_task_subset("CAPTION", max_samples=1)
            subset._cache_put(0, {"ok": True})
            assert len(subset._cache_index) == 1

            roundtripped = pickle.loads(pickle.dumps(dataset))
            roundtripped._cache_put(0, {"ok": True})
            assert len(roundtripped._cache_index) == 1

            save_path = Path(tmpdir) / "dataset.json"
            dataset.save_to_file(save_path)
            loaded = MultiTaskDataset.load_from_file(save_path)
            loaded._cache_put(0, {"ok": True})
            assert hasattr(loaded, "collate_fn")
            assert len(loaded._cache_index) == 1

    def test_getitem_without_processor(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = self._create_dummy_jsonl(tmpdir, n_samples=1)
            self._create_dummy_images(tmpdir, n_samples=1)
            cfg = DataConfig()
            dataset = MultiTaskDataset(
                data_configs=[{"task_type": "CAPTION", "data_path": str(data_path), "weight": 1.0}],
                image_base_path=tmpdir,
                config=cfg,
                processor=None,
            )
            item = dataset[0]
            assert "image" in item
            assert "prompt" in item
            assert "answer" in item

    def test_task_statistics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = self._create_dummy_jsonl(tmpdir, n_samples=3)
            self._create_dummy_images(tmpdir, n_samples=3)
            cfg = DataConfig()
            dataset = MultiTaskDataset(
                data_configs=[{"task_type": "CAPTION", "data_path": str(data_path), "weight": 1.0}],
                image_base_path=tmpdir,
                config=cfg,
                processor=None,
            )
            stats = dataset.get_task_statistics()
            assert stats["total_samples"] == 3
            assert stats["task_counts"]["CAPTION"] == 3
