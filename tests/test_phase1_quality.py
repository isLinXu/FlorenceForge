"""Phase-1 质量基线回归测试（F821 / 增强 / CLI 评估数据集）。"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

pytestmark = [pytest.mark.unit]


class TestDatasetAugmentation:
    def test_bbox_augmentation_updates_metadata(self):
        from florence_forge.core.config import DataConfig
        from florence_forge.data.dataset import MultiTaskDataset
        from florence_forge.data.dataset_types import TaskSample

        config = DataConfig(
            use_augmentation=True,
            augment_image=False,
            augment_bbox=True,
            augmentation_prob=1.0,
        )
        sample = TaskSample(
            task_type="OD",
            image_path="demo.jpg",
            prefix="<OD>",
            suffix="cat",
            metadata={
                "bboxes": [
                    {"xmin": 0.1, "ymin": 0.1, "xmax": 0.5, "ymax": 0.5},
                ]
            },
        )
        dataset = MultiTaskDataset.__new__(MultiTaskDataset)
        dataset.config = config
        dataset._sample_cache = MagicMock(use_cache=False)
        dataset._init_augmentation()

        augmented = dataset._maybe_augment_bboxes(sample)
        assert augmented.metadata["bboxes"] is not sample.metadata["bboxes"]
        assert len(augmented.metadata["bboxes"]) == 1

    def test_augmentation_disabled_when_cache_enabled(self):
        from florence_forge.core.config import DataConfig
        from florence_forge.data.dataset import MultiTaskDataset

        config = DataConfig(use_augmentation=True, use_cache=True)
        dataset = MultiTaskDataset.__new__(MultiTaskDataset)
        dataset.config = config
        dataset._sample_cache = MagicMock(use_cache=True)
        dataset._image_aug = None
        dataset._text_aug = None
        dataset._bbox_aug = None
        dataset._init_augmentation()

        assert dataset._image_aug is None
        assert dataset._bbox_aug is None


class TestCommandsEvalDataset:
    def test_build_eval_dataset_from_jsonl_groups_by_task(self, tmp_path):
        from florence_forge.cli.commands_eval import _build_eval_dataset_from_jsonl

        jsonl = tmp_path / "eval.jsonl"
        jsonl.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "image": "a.jpg",
                            "prefix": "<OD>",
                            "suffix": "cat",
                        }
                    ),
                    json.dumps(
                        {
                            "image": "b.jpg",
                            "prefix": "<CAPTION>",
                            "suffix": "a photo",
                        }
                    ),
                ]
            ),
            encoding="utf-8",
        )

        model = MagicMock()
        model.processor = MagicMock()
        model.backend = None

        dataset = _build_eval_dataset_from_jsonl(str(jsonl), model)

        assert len(dataset) == 2
        assert hasattr(dataset, "collate_fn")
        task_types = {dataset.samples[i].task_type for i in range(len(dataset))}
        assert "OD" in task_types or "CAPTION" in task_types


class TestServerCorsDefaults:
    def test_resolve_cors_origins_defaults_to_localhost(self, monkeypatch):
        from florence_forge.deployment.server import ModelServer

        monkeypatch.delenv("FLORENCE_CORS_ORIGINS", raising=False)
        origins = ModelServer._resolve_cors_origins(None)
        assert "*" not in origins
        assert any("127.0.0.1" in o or "localhost" in o for o in origins)

    def test_resolve_cors_origins_from_env(self, monkeypatch):
        from florence_forge.deployment.server import ModelServer

        monkeypatch.setenv(
            "FLORENCE_CORS_ORIGINS",
            "https://app.example.com,https://api.example.com",
        )
        origins = ModelServer._resolve_cors_origins(None)
        assert origins == ["https://app.example.com", "https://api.example.com"]
