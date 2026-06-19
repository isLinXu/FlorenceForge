"""Tests for TVP training bridge (P3) and synthetic generators (P4)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path


class TestTVPTrainingBridge:
    def test_normalize_tvp_task_type_aliases(self):
        from florence_forge.training.tvp_training import normalize_tvp_task_type

        assert normalize_tvp_task_type("od_vp") == "OD_VP"
        assert normalize_tvp_task_type("counting") == "COUNT_VP_COT"
        assert normalize_tvp_task_type("maze") == "MAZE_VP"
        assert normalize_tvp_task_type("path") == "PATH_VP"

    def test_build_training_config_from_tvp(self):
        from florence_forge.training.tvp_training import (
            build_training_config_from_tvp,
            load_tvp_yaml,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "sample.jsonl"
            jsonl_path.write_text(
                json.dumps({
                    "image": "demo.png",
                    "prefix": "<MAZE_VP>",
                    "suffix": "answer",
                }) + "\n",
                encoding="utf-8",
            )
            yaml_path = Path(tmpdir) / "sft.yaml"
            yaml_path.write_text(
                f"""
model_name_or_path: microsoft/Florence-2-base
output_dir: {tmpdir}/out
datasets:
  - path: {jsonl_path}
    image_root: {tmpdir}
    task_type: maze
    weight: 1.0
mixed_training:
  enabled: false
epochs: 1
batch_size: 2
""".strip(),
                encoding="utf-8",
            )

            cfg = load_tvp_yaml(yaml_path)
            training_config = build_training_config_from_tvp(cfg)
            assert "MAZE_VP" in training_config.tasks
            assert training_config.num_epochs == 1
            assert training_config.batch_size == 2
            assert hasattr(training_config, "_tvp_data_configs")
            assert training_config._tvp_data_configs[0]["task_type"] == "MAZE_VP"

    def test_apply_mixed_training_weights(self):
        from florence_forge.training.tvp_training import apply_mixed_training_weights

        configs = [
            {"task_type": "CAPTION", "weight": 1.0},
            {"task_type": "MAZE_VP", "weight": 1.0},
        ]
        scaled = apply_mixed_training_weights(configs, tvp_ratio=0.3)
        tvp_weight = next(item["weight"] for item in scaled if item["task_type"] == "MAZE_VP")
        general_weight = next(item["weight"] for item in scaled if item["task_type"] == "CAPTION")
        assert tvp_weight < general_weight


class TestTVPSynthetic:
    def test_write_maze_jsonl(self):
        from florence_forge.data.tvp_synthetic import write_maze_jsonl

        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = write_maze_jsonl(tmpdir, num_samples=3, rows=4, cols=4, seed=7)
            assert jsonl_path.exists()
            lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
            assert len(lines) == 3
            record = json.loads(lines[0])
            assert "start_point" in record
            assert "end_point" in record
            assert Path(tmpdir, record["image"]).exists()

    def test_write_path_jsonl(self):
        from florence_forge.data.tvp_synthetic import write_path_jsonl

        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = write_path_jsonl(tmpdir, num_samples=2, seed=11)
            record = json.loads(jsonl_path.read_text(encoding="utf-8").splitlines()[0])
            assert record["end_label"] in {"A", "B", "C", "D", "E"}
            assert len(record["points"]) >= 2
            assert Path(tmpdir, record["image"]).exists()

    def test_synthetic_to_tvp_maze_chain(self):
        from florence_forge.data.tvp_converter import TVPDataConverter
        from florence_forge.data.tvp_synthetic import write_maze_jsonl

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_jsonl = write_maze_jsonl(tmpdir, num_samples=1, rows=5, cols=5, seed=3)
            vp_jsonl = Path(tmpdir) / "maze_vp.jsonl"
            TVPDataConverter.maze_jsonl_to_vp(
                input_path=str(raw_jsonl),
                output_path=str(vp_jsonl),
                image_dir=tmpdir,
            )
            sample = json.loads(vp_jsonl.read_text(encoding="utf-8").strip())
            assert sample["vp_task_type"] == "MAZE_VP"
            assert "prefix" in sample and "suffix" in sample


class TestTVPOPDGRPOBridge:
    def test_resolve_opd_task_type_from_metadata(self):
        from florence_forge.training.opd_trainer import resolve_opd_task_type

        assert resolve_opd_task_type({"base_task": "maze"}) == "maze"
        assert resolve_opd_task_type({"vp_task_type": "COUNT_VP_COT"}) == "counting"
        assert resolve_opd_task_type({"task_type": "PATH_VP"}) == "path"

    def test_prepare_grpo_prompt_batch_trims_labels(self):
        import torch
        from florence_forge.training.tvp_training import prepare_grpo_prompt_batch

        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
            "labels": torch.tensor([[-100, -100, 3, 4, 5]]),
            "metadata": [{"base_task": "maze"}],
        }
        trimmed = prepare_grpo_prompt_batch(batch)
        assert trimmed["input_ids"].shape[1] == 2
        assert "labels" not in trimmed

    def test_build_grpo_reward_models_from_yaml(self):
        from florence_forge.training.tvp_training import _build_grpo_reward_models

        reward_fns, weights = _build_grpo_reward_models({
            "reward_models": {
                "format_rm": {"enabled": True, "weight": 0.3},
                "quality_rm": {"enabled": True, "weight": 0.2},
                "accuracy_rm": {"enabled": True, "weight": 0.5, "task_type": "mixed"},
            }
        })
        assert len(reward_fns) == 3
        assert weights is not None
        assert sum(weights) == 1.0


class TestTVPSpatialSynthetic:
    def test_write_spatial_jsonl(self):
        from florence_forge.data.tvp_synthetic import write_spatial_jsonl

        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = write_spatial_jsonl(tmpdir, num_samples=2, seed=5)
            record = json.loads(jsonl_path.read_text(encoding="utf-8").splitlines()[0])
            assert record["answer"] in {"left", "right", "above", "below"}
            assert "object_A" in record["supporting_boxes"]
            assert Path(tmpdir, record["image"]).exists()

    def test_spatial_to_vp_chain(self):
        from florence_forge.data.tvp_converter import TVPDataConverter
        from florence_forge.data.tvp_synthetic import write_spatial_jsonl

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_jsonl = write_spatial_jsonl(tmpdir, num_samples=1, seed=9)
            vp_jsonl = Path(tmpdir) / "spatial_vp.jsonl"
            TVPDataConverter.spatial_reasoning_jsonl_to_vp(
                input_path=str(raw_jsonl),
                output_path=str(vp_jsonl),
                image_dir=tmpdir,
            )
            sample = json.loads(vp_jsonl.read_text(encoding="utf-8").strip())
            assert sample["vp_task_type"] == "SPATIAL_VP"
            assert "Conclusion" in sample["suffix"]

    def test_write_all_tvp_synthetic(self):
        from florence_forge.data.tvp_synthetic import write_all_tvp_synthetic

        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = write_all_tvp_synthetic(tmpdir, num_samples=2, seed=1)
            assert set(outputs) == {"maze", "path", "spatial"}
            for path in outputs.values():
                assert path.exists()
                assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 2


class TestTVPAlignmentSmokeScript:
    def test_tvp_alignment_smoke_runs(self, tmp_path):
        import importlib.util

        script_path = Path("scripts/smoke/tvp_alignment_smoke.py")
        spec = importlib.util.spec_from_file_location("tvp_alignment_smoke", script_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        report = module.run_smoke(tmp_path / "tvp_smoke", num_samples=2, with_training=True)
        assert report["ok"] is True
        assert report["checks"]["dataset_rows"] == 6
        assert report["training"]["dataset_entries"] == 3
