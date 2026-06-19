"""Tests for FlorenceForge x TVP paper alignment (P0-P2)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


class TestTVPTaskRegistry:
    def test_tvp_tasks_registered(self):
        from florence_forge.core.tasks import (
            TVP_TASK_NAMES,
            get_tvp_tasks,
            is_tvp_task,
            validate_task_name,
        )

        expected = {"COUNT_VP_COT", "SPATIAL_VP", "MAZE_VP", "PATH_VP"}
        assert expected.issubset(set(TVP_TASK_NAMES))
        assert set(get_tvp_tasks()) == expected
        for task_name in expected:
            assert validate_task_name(task_name)
            assert is_tvp_task(task_name)
        assert not is_tvp_task("OD_VP")


class TestVisualPrimitiveSorting:
    def test_sort_boxes_left_to_right(self):
        from florence_forge.core.visual_primitives import sort_boxes_left_to_right

        boxes = [[300, 10, 400, 20], [100, 5, 200, 15], [250, 1, 350, 9]]
        assert sort_boxes_left_to_right(boxes) == [
            [100, 5, 200, 15],
            [250, 1, 350, 9],
            [300, 10, 400, 20],
        ]

    def test_resolve_marker_style_aliases(self):
        from florence_forge.core.visual_primitives import resolve_marker_style

        assert resolve_marker_style("special") == "special"
        assert resolve_marker_style("angle_bracket") == "plain"


class TestTVPBenchmark:
    def test_evaluate_tvp_predictions_counting(self):
        from florence_forge.evaluation.tvp_benchmark import evaluate_tvp_predictions

        records = [{
            "base_task": "counting",
            "count": 1,
            "gt_boxes": [(100, 200, 300, 400)],
        }]
        predictions = ["<|box|>[[100,200,300,400]]<|/box|>"]
        results = evaluate_tvp_predictions(records, predictions)
        assert results["overall_metrics"]["sample_count"] == 1
        assert results["overall_metrics"]["composite_mean"] > 0.0

    def test_evaluate_tvp_predictions_spatial(self):
        from florence_forge.evaluation.tvp_benchmark import evaluate_tvp_predictions

        records = [{
            "vp_task_type": "SPATIAL_VP",
            "answer": "left",
        }]
        predictions = ["3. Conclusion\nleft"]
        results = evaluate_tvp_predictions(records, predictions)
        assert "spatial_vp" in results["task_metrics"]


class TestCLIConversionHelpers:
    def test_resolve_vp_box_format_aliases(self):
        from florence_forge.cli.commands import _resolve_vp_box_format

        assert _resolve_vp_box_format("quad") == "json"
        assert _resolve_vp_box_format("loc_tokens") == "loc_tokens"

    def test_tvp_spatial_converter_writes_task_type(self):
        from florence_forge.data.tvp_converter import TVPDataConverter

        samples = [{
            "image": "img1.jpg",
            "observation": "Two objects visible.",
            "reasoning": "Object A is above B.",
            "answer": "above",
        }]

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "spatial.jsonl"
            output_path = Path(tmpdir) / "spatial_vp.jsonl"
            image_dir = Path(tmpdir)

            with open(input_path, "w", encoding="utf-8") as handle:
                for sample in samples:
                    handle.write(json.dumps(sample) + "\n")

            TVPDataConverter.spatial_reasoning_jsonl_to_vp(
                str(input_path),
                str(output_path),
                str(image_dir),
            )

            row = json.loads(output_path.read_text(encoding="utf-8").strip())
            assert row["vp_task_type"] == "SPATIAL_VP"
            assert row["task_family"] == "tvprimitives"
            assert "Analyzing the request" in row["suffix"]
