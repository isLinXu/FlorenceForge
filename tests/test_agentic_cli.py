"""CLI integration tests for the ``agentic`` subcommand.

These tests verify:
  1. The argparser registers the ``agentic`` subcommand with all expected flags.
  2. ``run_agentic_task`` dispatches correctly to the orchestrator using a
     mock-based InferenceEngine (no torch/weights needed).
  3. Single-image and batch modes produce valid JSON output files.
  4. ``InferenceEngineAdapter`` correctly translates ``predict_task`` → ``predict``.
  5. The ``--save-transcript`` flag writes a transcript file.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# 1. Argparser registration
# ---------------------------------------------------------------------------

class TestAgenticArgparser:
    """Verify the agentic subcommand is registered in the CLI parser."""

    def _make_parser(self):
        from florence_forge.cli.main import create_parser
        return create_parser()

    def test_agentic_subcommand_exists(self):
        parser = self._make_parser()
        # Should not raise
        args = parser.parse_args([
            'agentic', '--model', 'microsoft/Florence-2-base',
            '--input', '/tmp/img.png', '--output', '/tmp/out',
            '--goal', 'detect objects',
        ])
        assert args.command == 'agentic'
        assert args.goal == 'detect objects'

    def test_agentic_has_expected_flags(self):
        parser = self._make_parser()
        args = parser.parse_args([
            'agentic',
            '--model', '/path/to/model',
            '--input', '/path/to/image.png',
            '--output', '/path/to/output',
            '--goal', 'detect and count',
            '--device', 'cpu',
            '--use-amp',
            '--max-steps', '5',
            '--max-retries', '2',
            '--summarize-every', '1',
            '--save-transcript',
        ])
        assert args.model == '/path/to/model'
        assert args.device == 'cpu'
        assert args.use_amp is True
        assert args.max_steps == 5
        assert args.max_retries == 2
        assert args.summarize_every == 1
        assert args.save_transcript is True

    def test_agentic_goal_required(self):
        parser = self._make_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                'agentic', '--model', 'm', '--input', 'i', '--output', 'o',
            ])

    def test_agentic_defaults(self):
        parser = self._make_parser()
        args = parser.parse_args([
            'agentic', '--model', 'm', '--input', 'i', '--output', 'o',
            '--goal', 'g',
        ])
        assert args.device == 'auto'
        assert args.use_amp is False
        assert args.max_steps == 12
        assert args.max_retries == 1
        assert args.summarize_every == 3
        assert args.save_transcript is False


# ---------------------------------------------------------------------------
# 2. InferenceEngineAdapter
# ---------------------------------------------------------------------------

class TestInferenceEngineAdapter:
    """Test the adapter that bridges InferenceEngine → ToolBackend protocol."""

    def test_predict_task_translates_to_predict(self):
        from florence_forge.cli.commands_agentic import InferenceEngineAdapter

        class FakeEngine:
            def __init__(self):
                self.calls: List[Dict] = []

            def predict(self, image, task_prompt=None, text_input=None, **kwargs):
                self.calls.append({
                    "image": image,
                    "task_prompt": task_prompt,
                    "text_input": text_input,
                })
                return f"result for {task_prompt}"

        engine = FakeEngine()
        adapter = InferenceEngineAdapter(engine)
        result = adapter.predict_task(images="fake_image", task_name="OD")

        assert result == "result for <OD>"
        assert len(engine.calls) == 1
        assert engine.calls[0]["task_prompt"] == "<OD>"
        assert engine.calls[0]["image"] == "fake_image"

    def test_predict_task_with_text_input(self):
        from florence_forge.cli.commands_agentic import InferenceEngineAdapter

        class FakeEngine:
            def predict(self, image, task_prompt=None, text_input=None, **kwargs):
                return f"{task_prompt}:{text_input}"

        adapter = InferenceEngineAdapter(FakeEngine())
        result = adapter.predict_task(
            images=None, task_name="COUNT_VP", text_input="cars",
        )
        assert "cars" in result

    def test_predict_task_non_str_coerced(self):
        from florence_forge.cli.commands_agentic import InferenceEngineAdapter

        class FakeEngine:
            def predict(self, image, task_prompt=None, text_input=None, **kwargs):
                return {"bboxes": [1, 2, 3]}  # non-str

        adapter = InferenceEngineAdapter(FakeEngine())
        result = adapter.predict_task(images=None, task_name="OD")
        assert isinstance(result, str)
        assert "bboxes" in result


# ---------------------------------------------------------------------------
# 3. run_agentic_task with mocked engine + orchestrator
# ---------------------------------------------------------------------------

class _MockOrchestratorResult:
    """Mimic OrchestratorResult for testing."""

    def __init__(self, goal: str, final_answer: str, success: bool, num_steps: int):
        self.goal = goal
        self.final_answer = final_answer
        self.success = success
        self.transcript = "<PLAN>test</PLAN><ACT>act</ACT><VERIFY>ok</VERIFY><DECIDE>done</DECIDE>"
        self.steps = [{"sub_task_index": i, "intent": "detect", "verified": True} for i in range(num_steps)]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "final_answer": self.final_answer,
            "success": self.success,
            "steps": self.steps,
            "state": {"detected_objects": [], "extracted_text": []},
            "plan": [{"index": 0, "goal": self.goal, "intent": "detect"}],
            "transcript": self.transcript,
        }


class TestRunAgenticTask:
    """Test run_agentic_task end-to-end with mocked InferenceEngine."""

    @pytest.fixture
    def fake_image(self, tmp_path):
        """Create a minimal real image file for path checks."""
        from PIL import Image
        img = Image.new("RGB", (64, 64), color=(128, 128, 128))
        path = tmp_path / "test.png"
        img.save(path)
        return path

    @pytest.fixture
    def fake_model_dir(self, tmp_path):
        """Create a fake model directory."""
        d = tmp_path / "model"
        d.mkdir()
        (d / "config.json").write_text("{}")
        return d

    @pytest.fixture
    def mock_engine_and_orchestrator(self):
        """Patch InferenceEngine and AgenticOrchestrator with mocks."""
        mock_engine = MagicMock()
        mock_result = _MockOrchestratorResult(
            goal="detect objects",
            final_answer="Detected 2 objects",
            success=True,
            num_steps=2,
        )
        mock_orchestrator = MagicMock()
        mock_orchestrator.run.return_value = mock_result

        patches = [
            patch(
                "florence_forge.deployment.inference.InferenceEngine",
                return_value=mock_engine,
            ),
            patch(
                "florence_forge.agentic.AgenticOrchestrator",
                return_value=mock_orchestrator,
            ),
        ]
        for p in patches:
            p.start()
        yield mock_engine, mock_orchestrator, mock_result
        for p in patches:
            p.stop()

    def test_single_image_success(self, fake_image, fake_model_dir, mock_engine_and_orchestrator, tmp_path):
        from florence_forge.cli.commands_agentic import run_agentic_task

        mock_engine, mock_orchestrator, mock_result = mock_engine_and_orchestrator
        output_dir = tmp_path / "output"

        args = MagicMock(
            model=str(fake_model_dir),
            input=str(fake_image),
            output=str(output_dir),
            goal="detect objects",
            device="cpu",
            use_amp=False,
            max_steps=12,
            max_retries=1,
            summarize_every=3,
            save_transcript=False,
        )

        success = run_agentic_task(args)
        assert success is True

        # Output files should exist
        result_file = output_dir / "test_agentic.json"
        assert result_file.exists()

        data = json.loads(result_file.read_text())
        assert data["goal"] == "detect objects"
        assert data["success"] is True
        assert data["final_answer"] == "Detected 2 objects"

        # Summary should exist
        summary_file = output_dir / "agentic_summary.json"
        assert summary_file.exists()
        summary = json.loads(summary_file.read_text())
        assert summary["total_images"] == 1
        assert summary["successful"] == 1

    def test_save_transcript_flag(self, fake_image, fake_model_dir, mock_engine_and_orchestrator, tmp_path):
        from florence_forge.cli.commands_agentic import run_agentic_task

        mock_engine, mock_orchestrator, mock_result = mock_engine_and_orchestrator
        output_dir = tmp_path / "output"

        args = MagicMock(
            model=str(fake_model_dir),
            input=str(fake_image),
            output=str(output_dir),
            goal="detect objects",
            device="cpu",
            use_amp=False,
            max_steps=12,
            max_retries=1,
            summarize_every=3,
            save_transcript=True,
        )

        success = run_agentic_task(args)
        assert success is True

        transcript_file = output_dir / "test_transcript.txt"
        assert transcript_file.exists()
        content = transcript_file.read_text()
        assert "<PLAN>" in content
        assert "<DECIDE>" in content

    def test_batch_mode(self, fake_model_dir, mock_engine_and_orchestrator, tmp_path):
        from PIL import Image
        from florence_forge.cli.commands_agentic import run_agentic_task

        mock_engine, mock_orchestrator, mock_result = mock_engine_and_orchestrator

        # Create a batch directory with 3 images
        batch_dir = tmp_path / "batch"
        batch_dir.mkdir()
        for i in range(3):
            img = Image.new("RGB", (32, 32), color=(i * 50, 0, 0))
            img.save(batch_dir / f"img_{i}.png")

        output_dir = tmp_path / "output"

        args = MagicMock(
            model=str(fake_model_dir),
            input=str(batch_dir),
            output=str(output_dir),
            goal="describe each image",
            device="cpu",
            use_amp=False,
            max_steps=8,
            max_retries=1,
            summarize_every=3,
            save_transcript=False,
        )

        success = run_agentic_task(args)
        assert success is True

        # Should have 3 result files
        result_files = list(output_dir.glob("*_agentic.json"))
        assert len(result_files) == 3

        summary_file = output_dir / "agentic_summary.json"
        summary = json.loads(summary_file.read_text())
        assert summary["total_images"] == 3
        assert summary["successful"] == 3

    def test_model_not_found(self, tmp_path):
        from florence_forge.cli.commands_agentic import run_agentic_task

        args = MagicMock(
            model="/nonexistent/model/path",
            input=str(tmp_path),
            output=str(tmp_path / "out"),
            goal="test",
            device="cpu",
            use_amp=False,
            max_steps=12,
            max_retries=1,
            summarize_every=3,
            save_transcript=False,
        )

        success = run_agentic_task(args)
        assert success is False

    def test_input_not_found(self, fake_model_dir, tmp_path):
        from florence_forge.cli.commands_agentic import run_agentic_task

        # Patch InferenceEngine so it doesn't try to load
        with patch("florence_forge.deployment.inference.InferenceEngine") as mock_ie:
            mock_ie.return_value = MagicMock()

            args = MagicMock(
                model=str(fake_model_dir),
                input="/nonexistent/image.png",
                output=str(tmp_path / "out"),
                goal="test",
                device="cpu",
                use_amp=False,
                max_steps=12,
                max_retries=1,
                summarize_every=3,
                save_transcript=False,
            )

            success = run_agentic_task(args)
            assert success is False

    def test_orchestrator_config_passed(self, fake_image, fake_model_dir, mock_engine_and_orchestrator, tmp_path):
        from florence_forge.cli.commands_agentic import run_agentic_task

        mock_engine, mock_orchestrator, mock_result = mock_engine_and_orchestrator
        output_dir = tmp_path / "output"

        args = MagicMock(
            model=str(fake_model_dir),
            input=str(fake_image),
            output=str(output_dir),
            goal="detect",
            device="cpu",
            use_amp=False,
            max_steps=5,
            max_retries=3,
            summarize_every=1,
            save_transcript=False,
        )

        run_agentic_task(args)

        # AgenticOrchestrator should have been called with the config
        # Verify the constructor was called (patched)
        from florence_forge.agentic import AgenticOrchestrator
        assert AgenticOrchestrator.called


# ---------------------------------------------------------------------------
# 4. Commands facade re-export
# ---------------------------------------------------------------------------

class TestCommandsFacade:
    """Verify commands.py re-exports the agentic handler."""

    def test_run_agentic_task_importable_from_commands(self):
        from florence_forge.cli.commands import run_agentic_task
        assert callable(run_agentic_task)

    def test_inference_engine_adapter_importable(self):
        from florence_forge.cli.commands import InferenceEngineAdapter
        assert InferenceEngineAdapter is not None

    def test_run_agentic_task_in_all(self):
        from florence_forge.cli import commands
        assert "run_agentic_task" in commands.__all__
        assert "InferenceEngineAdapter" in commands.__all__


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
