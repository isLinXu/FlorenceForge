"""CLI inference helper regressions."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from florence_forge.cli.main import (
    create_parser,
    _is_supported_image_file,
    _iter_image_files,
    _normalize_inference_stats,
    run_doctor_task,
    run_inference_task,
)
from florence_forge.cli.commands import _select_trainer_class


def test_supported_image_file_is_case_insensitive(tmp_path):
    image = tmp_path / "sample.WeBp"
    image.write_bytes(b"not a real image; helper only checks suffix")

    assert _is_supported_image_file(image)


def test_iter_image_files_supports_mixed_case_and_filters_non_images(tmp_path):
    for name in ["a.JPG", "b.PnG", "c.tif", "notes.txt"]:
        (tmp_path / name).write_text("x")

    names = [path.name for path in _iter_image_files(tmp_path)]

    assert names == ["a.JPG", "b.PnG", "c.tif"]


def test_supported_image_file_requires_regular_file(tmp_path):
    directory = tmp_path / "folder.jpg"
    directory.mkdir()

    assert not _is_supported_image_file(directory)


def test_normalize_inference_stats_fills_missing_fields():
    stats = _normalize_inference_stats({"total_inferences": 2})

    assert stats["total_inferences"] == 2
    assert stats["total_time"] == 0.0
    assert stats["avg_inference_time"] == 0.0
    assert stats["throughput"] == 0.0


def test_train_parser_accepts_trainer_version_v2():
    parser = create_parser()

    args = parser.parse_args(["train", "--task", "caption", "--trainer-version", "v2"])

    assert args.trainer_version == "v2"


def test_select_trainer_class_supports_v2_aliases():
    from florence_forge.training.trainer import MultiTaskTrainer as TrainerV1
    from florence_forge.training.trainer_refactored import MultiTaskTrainer as TrainerV2

    assert _select_trainer_class("v1") is TrainerV1
    assert _select_trainer_class("legacy") is TrainerV1
    assert _select_trainer_class("v2") is TrainerV2
    assert _select_trainer_class("modular") is TrainerV2


def test_run_inference_task_accepts_minimal_engine_stats(tmp_path):
    class MinimalStatsEngine:
        def __init__(self, model, device, batch_size, use_amp):
            self.calls = 0

        def predict(self, image, **kwargs):
            self.calls += 1
            return {"mode": image.mode}

        def get_stats(self):
            return {"total_inferences": self.calls}

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    image_path = tmp_path / "sample.PNG"
    Image.new("RGB", (8, 8), color="red").save(image_path)
    output_dir = tmp_path / "out"

    args = SimpleNamespace(
        model=str(model_dir),
        input=str(image_path),
        output=str(output_dir),
        device="mps",
        batch_size=1,
        use_amp=False,
        task_prompt="<CAPTION>",
        text_input=None,
        visualize=False,
        save_visualizations=False,
    )

    with patch("florence_forge.deployment.inference.InferenceEngine", MinimalStatsEngine):
        assert run_inference_task(args) is True

    summary = json.loads((output_dir / "inference_summary.json").read_text())
    assert summary["total_images"] == 1
    assert summary["stats"]["total_inferences"] == 1
    assert summary["stats"]["total_time"] == 0.0
    assert summary["stats"]["throughput"] == 0.0


def test_run_doctor_task_can_emit_json(monkeypatch, capsys):
    report = {
        "ok": True,
        "platform": {"python": "3.11", "system": "Darwin", "release": "1", "machine": "arm64"},
        "torch": {
            "available": True,
            "version": "test",
            "selected_device": "mps",
            "selected_device_available": True,
            "mps_available": True,
            "mps_built": True,
            "cuda_available": False,
            "cuda_device_count": 0,
            "cuda_version": None,
        },
        "dependencies": [],
        "missing_required": [],
        "model": {
            "model_id": "microsoft/Florence-2-base",
            "model_path": None,
            "local_snapshot": "/tmp/model",
            "local_snapshot_exists": True,
        },
        "recommended_torch_dtype": "float32",
        "suggested_smoke_command": "python scripts/smoke/real_florence_mps_smoke.py --mode forward --device mps",
        "warnings": [],
    }

    def fake_collect(**kwargs):
        assert kwargs["requested_device"] == "mps"
        assert kwargs["require_model"] is True
        return report

    monkeypatch.setattr("florence_forge.cli.main.collect_environment_diagnostics", fake_collect)

    ok = run_doctor_task(SimpleNamespace(
        device="mps",
        model_id="microsoft/Florence-2-base",
        model_path=None,
        require_model=True,
        json=True,
    ))

    assert ok is True
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["torch"]["selected_device"] == "mps"
