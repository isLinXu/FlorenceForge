"""Phase-1 覆盖率补强：针对近期修复路径的轻量单元测试。"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

pytestmark = [pytest.mark.unit]


class TestDataProfilerExtras:
    def test_histogram_empty_and_constant(self):
        from florence_forge.data.profiler import _histogram

        empty = _histogram([])
        assert empty["count"] == 0
        assert empty["bins"] == []

        flat = _histogram([3.0, 3.0, 3.0])
        assert flat["mean"] == 3.0
        assert len(flat["bins"]) == 1

    def test_profile_jsonl_emits_imbalance_warning(self, tmp_path, caplog):
        from florence_forge.data.profiler import DataProfiler

        jsonl = tmp_path / "imbalanced.jsonl"
        rows = [
            {"task_type": "CAPTION", "image": "a.jpg", "prefix": "<CAPTION>", "suffix": "x"}
        ] * 5
        rows += [
            {"task_type": "OD", "image": "b.jpg", "prefix": "<OD>", "suffix": "y"}
        ]
        with jsonl.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")

        with caplog.at_level("WARNING"):
            report = DataProfiler(imbalance_threshold=2.0).profile_jsonl(jsonl)

        assert report["imbalance_ratio"] > 2.0
        assert any("不平衡" in w for w in report["warnings"])

    def test_data_profiler_exported_from_package(self):
        from florence_forge.data import DataProfiler

        assert DataProfiler is not None


class TestModelMergerApplyWeightDelta:
    def test_skips_missing_keys(self):
        from florence_forge.training.model_merger import ModelMerger

        model = nn.Linear(2, 2)
        before = model.weight.data.clone()
        ModelMerger.apply_weight_delta(model, {"missing.weight": torch.ones(2, 2)})
        assert torch.allclose(model.weight.data, before)


class TestSFTTrainerCheckpoint:
    def test_load_checkpoint_restores_state(self, tmp_path):
        from florence_forge.training.sft_trainer import SFTConfig, SFTTrainer

        model = nn.Linear(2, 2)
        config = SFTConfig(device="cpu")
        trainer = SFTTrainer(model, MagicMock(), train_dataloader=MagicMock(), config=config)
        trainer.global_step = 0
        trainer.best_eval_loss = float("inf")

        ckpt_dir = tmp_path / "ckpt"
        ckpt_dir.mkdir()
        torch.save(
            {
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "global_step": 7,
                "best_eval_loss": 0.25,
            },
            ckpt_dir / "trainer_state.pt",
        )

        trainer.load_checkpoint(str(ckpt_dir))
        assert trainer.global_step == 7
        assert trainer.best_eval_loss == 0.25


class TestGenericHFBackendEncode:
    def test_encode_skips_redundant_device_move(self):
        from florence_forge.core.backends.generic_hf_backend import GenericHFBackend

        backend = GenericHFBackend.__new__(GenericHFBackend)
        backend._device = torch.device("cpu")
        backend._processor = MagicMock()
        backend._processor.return_value = {
            "input_ids": torch.tensor([[1, 2]], device="cpu"),
            "pixel_values": torch.randn(1, 3, 2, 2, device="cpu"),
        }
        backend._tokenizer = None
        backend._image_processor = None

        out = backend.encode(images=MagicMock(), text="hi")
        assert out["input_ids"].device.type == "cpu"


class TestVPLoadExport:
    def test_load_vp_quality_summary_from_helpers(self, tmp_path):
        from florence_forge.evaluation.vp_detection_quality import load_vp_quality_summary

        path = tmp_path / "summary.json"
        path.write_text(json.dumps({"records": []}), encoding="utf-8")
        loaded = load_vp_quality_summary(path)
        assert loaded["records"] == []


class TestSpatialCompression:
    def test_spatial_compression_project_mode(self):
        from florence_forge.training.spatial_compression import SpatialCompression

        layer = SpatialCompression(in_channels=4, out_channels=8, kernel_size=3, mode="project")
        x = torch.randn(2, 6, 6, 4)
        out = layer(x)
        assert out.shape == (2, 2, 2, 8)

    def test_spatial_compression_concat_mode(self):
        from florence_forge.training.spatial_compression import SpatialCompression

        layer = SpatialCompression(in_channels=2, kernel_size=2, mode="concat")
        x = torch.randn(1, 4, 4, 2)
        out = layer(x)
        assert out.shape == (1, 2, 2, 8)

    def test_patch_embed_any_resolution(self):
        from florence_forge.training.spatial_compression import PatchEmbedAnyResolution

        embed = PatchEmbedAnyResolution(patch_size=14, embed_dim=16)
        pixels = torch.randn(1, 3, 28, 28)
        out = embed(pixels)
        assert out.shape == (1, 2, 2, 16)

    def test_multi_stage_spatial_compression(self):
        from florence_forge.training.spatial_compression import MultiStageSpatialCompression

        pipeline = MultiStageSpatialCompression(in_channels=8, kernel_size=2, num_pool_queries=4)
        grid = torch.randn(1, 4, 4, 8)
        out = pipeline(grid)
        assert out.shape == (1, 4, pipeline.spatial_compress.out_channels)


class TestModelMergerDelegation:
    def test_merge_lora_weights_delegates_to_merge_and_unload(self):
        from florence_forge.training.model_merger import ModelMerger

        merger = ModelMerger()
        lora_model = MagicMock()
        expected = MagicMock()
        merger.merge_and_unload = MagicMock(return_value=expected)
        result = merger.merge_lora_weights(MagicMock(), lora_model, task_name="od")
        merger.merge_and_unload.assert_called_once_with(lora_model)
        assert result is expected
