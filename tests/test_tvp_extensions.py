"""Tests for TVP extension modules added in the supplementary pass.

Covers:
  - visual_primitives: new dataclasses and batch utilities
  - reward_models: FormatRewardModel coordinate validation, QualityRewardModel heuristic
  - tvp_metrics: CountingDetectionMetric, TVPCompositeMetric counting branch
  - tvp_converter: TVPChainBuilder, spatial_reasoning_jsonl_to_vp
  - sft_trainer: SFTConfig defaults, from_config factory
  - tvp_pipeline: _stage_checkpoint_exists
  - grpo_trainer: weighted reward computation
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest

# ────────────────────────────────────────────────────────────────────────────
# visual_primitives
# ────────────────────────────────────────────────────────────────────────────

class TestVisualPrimitiveExtensions:
    def test_visual_primitive_line_as_list(self):
        from florence_forge.core.visual_primitives import VisualPrimitiveLine
        line = VisualPrimitiveLine(0, 0, 300, 400, "edge")
        assert line.as_list() == [0, 0, 300, 400]

    def test_visual_primitive_line_length(self):
        from florence_forge.core.visual_primitives import VisualPrimitiveLine
        line = VisualPrimitiveLine(0, 0, 300, 400)
        assert abs(line.length() - 500.0) < 1.0

    def test_normalize_point(self):
        from florence_forge.core.visual_primitives import normalize_point
        x, y = normalize_point(64, 48, (640, 480))
        assert x == 100
        assert y == 100

    def test_normalize_bboxes_batch(self):
        from florence_forge.core.visual_primitives import normalize_bboxes_batch
        bboxes = [[0, 0, 100, 100], [50, 50, 200, 200]]
        result = normalize_bboxes_batch(bboxes, (200, 200))
        assert len(result) == 2
        # 100/200 * 999 = 499.5 → rounds to 500; verify valid VP range
        assert result[0][0] == 0 and result[0][1] == 0
        assert all(0 <= v <= 999 for v in result[0])

    def test_iou_normalized_identical(self):
        from florence_forge.core.visual_primitives import iou_normalized
        assert iou_normalized([0, 0, 500, 500], [0, 0, 500, 500]) == 1.0

    def test_iou_normalized_no_overlap(self):
        from florence_forge.core.visual_primitives import iou_normalized
        assert iou_normalized([0, 0, 100, 100], [200, 200, 400, 400]) == 0.0

    def test_iou_normalized_partial(self):
        from florence_forge.core.visual_primitives import iou_normalized
        iou = iou_normalized([0, 0, 200, 200], [100, 100, 300, 300])
        assert 0.0 < iou < 1.0

    def test_parse_vp_boxes_special(self):
        from florence_forge.core.visual_primitives import parse_vp_boxes
        text = "<|box|>[[100,200,300,400],[10,20,30,40]]<|/box|>"
        boxes = parse_vp_boxes(text)
        assert boxes == [[100, 200, 300, 400], [10, 20, 30, 40]]

    def test_parse_vp_boxes_plain(self):
        from florence_forge.core.visual_primitives import parse_vp_boxes
        text = "<box>[[5,6,7,8]]</box>"
        boxes = parse_vp_boxes(text)
        assert boxes == [[5, 6, 7, 8]]

    def test_parse_vp_boxes_loc_tokens(self):
        from florence_forge.core.visual_primitives import parse_vp_boxes
        text = "<box><loc_5><loc_6><loc_7><loc_8><loc_9><loc_10><loc_11><loc_12></box>"
        boxes = parse_vp_boxes(text)
        assert boxes == [[5, 6, 7, 8], [9, 10, 11, 12]]

    def test_parse_vp_boxes_python_tuple_payload(self):
        from florence_forge.core.visual_primitives import parse_vp_boxes
        text = "<|box|>(5, 6, 7, 8)<|/box|>"
        boxes = parse_vp_boxes(text)
        assert boxes == [[5, 6, 7, 8]]

    def test_parse_vp_boxes_filters_invalid_coordinate_groups(self):
        from florence_forge.core.visual_primitives import parse_vp_boxes

        text = "<box>[[0,0,1200,10],[500,20,100,40],[1,2,3,4]]</box>"

        assert parse_vp_boxes(text) == [[1, 2, 3, 4]]

    def test_parse_vp_boxes_filters_invalid_loc_token_groups(self):
        from florence_forge.core.visual_primitives import parse_vp_boxes

        text = "<box><loc_0><loc_0><loc_1200><loc_10><loc_2><loc_3><loc_4><loc_5></box>"

        assert parse_vp_boxes(text) == [[2, 3, 4, 5]]

    def test_parse_vp_points_special(self):
        from florence_forge.core.visual_primitives import parse_vp_points
        text = "<|point|>[[100,200],[300,400]]<|/point|>"
        points = parse_vp_points(text)
        assert points == [[100, 200], [300, 400]]

    def test_parse_vp_points_python_tuple_payload(self):
        from florence_forge.core.visual_primitives import parse_vp_points
        text = "<point>(100, 200)</point>"
        points = parse_vp_points(text)
        assert points == [[100, 200]]

    def test_parse_vp_points_filters_out_of_range_coordinates(self):
        from florence_forge.core.visual_primitives import parse_vp_points

        text = "<point>[[100,200],[300,1400]]</point>"

        assert parse_vp_points(text) == [[100, 200]]

    def test_parse_vp_boxes_empty(self):
        from florence_forge.core.visual_primitives import parse_vp_boxes
        assert parse_vp_boxes("no boxes here") == []


# ────────────────────────────────────────────────────────────────────────────
# reward_models
# ────────────────────────────────────────────────────────────────────────────

class TestFormatRewardModel:
    def setup_method(self):
        from florence_forge.training.reward_models import FormatRewardModel
        self.rm = FormatRewardModel()

    def test_perfect_box(self):
        text = "<|box|>[[100,200,300,400]]<|/box|>"
        score = self.rm(text)
        assert score == pytest.approx(1.0)

    def test_invalid_box_format(self):
        text = "<|box|>invalid content<|/box|>"
        score = self.rm(text)
        assert score < 1.0

    def test_out_of_range_coordinate(self):
        text = "<|box|>[[100,200,1200,400]]<|/box|>"  # 1200 > 999
        score = self.rm(text)
        assert score < 1.0

    def test_degenerate_box(self):
        text = "<|box|>[[500,200,100,400]]<|/box|>"  # x1 > x2
        score = self.rm(text)
        assert score < 1.0

    def test_duplicate_box_penalty(self):
        text = "<|box|>[[100,200,300,400],[100,200,300,400]]<|/box|>"
        score = self.rm(text)
        assert score < 1.0

    def test_empty_output_penalty(self):
        score = self.rm("no visual primitives here")
        assert score < 1.0

    def test_perfect_point(self):
        text = "<|point|>[[500,500]]<|/point|>"
        score = self.rm(text)
        assert score == pytest.approx(1.0)

    def test_out_of_range_point(self):
        text = "<|point|>[[500,1500]]<|/point|>"  # 1500 > 999
        score = self.rm(text)
        assert score < 1.0


class TestQualityRewardModel:
    def setup_method(self):
        from florence_forge.training.reward_models import QualityRewardModel
        self.rm = QualityRewardModel()  # no judge model → heuristic mode

    def test_short_response(self):
        score = self.rm("Short response.", {})
        assert score == pytest.approx(1.0)

    def test_long_response_penalized(self):
        text = "a\n" * 4000
        score = self.rm(text, {})
        assert score < 1.0

    def test_repetitive_response_penalized(self):
        text = "same line\n" * 100
        score = self.rm(text, {})
        assert score < 1.0

    def test_parse_judge_score(self):
        from florence_forge.training.reward_models import QualityRewardModel
        rm = QualityRewardModel()
        assert rm._parse_judge_score("1.0") == pytest.approx(1.0)
        assert rm._parse_judge_score("0.5") == pytest.approx(0.5)
        assert rm._parse_judge_score("0.0") == pytest.approx(0.0)
        assert rm._parse_judge_score("garbage") == pytest.approx(0.5)


# ────────────────────────────────────────────────────────────────────────────
# tvp_metrics
# ────────────────────────────────────────────────────────────────────────────

class TestCountingDetectionMetric:
    def setup_method(self):
        from florence_forge.evaluation.tvp_metrics import CountingDetectionMetric
        self.metric = CountingDetectionMetric(iou_threshold=0.5)

    def test_perfect_count_and_box(self):
        text = "<|box|>[[100,200,300,400]]<|/box|>"
        scores = self.metric.compute(text, gt_count=1, gt_boxes=[(100, 200, 300, 400)])
        assert scores["count_accuracy"] == pytest.approx(1.0)
        assert scores["count_f1"] == pytest.approx(1.0)

    def test_wrong_count(self):
        text = "<|box|>[[100,200,300,400],[10,20,50,60]]<|/box|>"  # pred=2
        scores = self.metric.compute(text, gt_count=1, gt_boxes=[(100, 200, 300, 400)])
        assert scores["count_accuracy"] < 1.0

    def test_no_gt_boxes(self):
        text = "<|box|>[[100,200,300,400]]<|/box|>"
        scores = self.metric.compute(text, gt_count=1)
        assert "count_accuracy" in scores

    def test_empty_prediction(self):
        scores = self.metric.compute("no boxes", gt_count=1, gt_boxes=[(100, 200, 300, 400)])
        assert scores["box_recall"] == 0.0


class TestTVPCompositeMetric:
    def setup_method(self):
        from florence_forge.evaluation.tvp_metrics import TVPCompositeMetric
        self.metric = TVPCompositeMetric()

    def test_counting_task(self):
        text = "<|box|>[[100,200,300,400]]<|/box|>"
        r = self.metric.compute(text, task_type="counting", gt_count=1,
                                gt_boxes=[(100, 200, 300, 400)])
        assert "count_f1" in r
        assert "composite" in r
        assert 0.0 <= r["composite"] <= 1.0

    def test_composite_field_always_present(self):
        r = self.metric.compute("some text", task_type="counting")
        assert "composite" in r

    def test_path_task(self):
        text = "<|point|>[[100,100],[500,500],[900,900]]<|/point|>"
        r = self.metric.compute(
            text,
            task_type="path",
            gt_points=[(100, 100), (500, 500), (900, 900)],
            gt_label="A",
            pred_label="A",
        )
        assert "trajectory_similarity" in r
        assert "composite" in r


# ────────────────────────────────────────────────────────────────────────────
# tvp_converter
# ────────────────────────────────────────────────────────────────────────────

class TestTVPChainBuilder:
    def test_counting_chain_contains_label(self):
        from florence_forge.data.tvp_converter import TVPChainBuilder
        chain = TVPChainBuilder.build_counting_chain(
            "cat", [[100, 100, 200, 200]], count=1
        )
        assert "cat" in chain
        assert "1 cat" in chain
        assert "<|box|>" in chain

    def test_maze_chain_solvable(self):
        from florence_forge.data.tvp_converter import TVPChainBuilder
        chain = TVPChainBuilder.build_maze_chain(
            solvable=True,
            exploration_points=[(100, 100), (200, 200)],
            solution_points=[(100, 100), (500, 500)],
            answer="true",
        )
        assert "true" in chain
        assert "<|point|>" in chain

    def test_maze_chain_unsolvable(self):
        from florence_forge.data.tvp_converter import TVPChainBuilder
        chain = TVPChainBuilder.build_maze_chain(
            solvable=False,
            exploration_points=[(100, 100)],
            answer="false",
        )
        assert "false" in chain
        assert "No valid path" in chain

    def test_path_chain(self):
        from florence_forge.data.tvp_converter import TVPChainBuilder
        chain = TVPChainBuilder.build_path_chain(
            trajectory_points=[(100, 200), (300, 400)],
            endpoint=(300, 400),
            end_label="B",
        )
        assert "B" in chain
        assert "<|point|>" in chain

    def test_spatial_chain(self):
        from florence_forge.data.tvp_converter import TVPChainBuilder
        chain = TVPChainBuilder.build_spatial_chain(
            observation="I see two objects.",
            reasoning="Object A is left of B.",
            answer="left",
        )
        assert "left" in chain
        assert "Observation" in chain


class TestSpatialReasoningConverter:
    def test_spatial_reasoning_jsonl_to_vp(self):
        from florence_forge.data.tvp_converter import TVPDataConverter

        samples = [
            {
                "image": "img1.jpg",
                "observation": "Two objects visible.",
                "reasoning": "Object A is above B.",
                "answer": "above",
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "spatial.jsonl"
            output_path = Path(tmpdir) / "spatial_vp.jsonl"
            image_dir = Path(tmpdir)

            with open(input_path, "w") as f:
                for s in samples:
                    f.write(json.dumps(s) + "\n")

            TVPDataConverter.spatial_reasoning_jsonl_to_vp(
                str(input_path),
                str(output_path),
                str(image_dir),
            )

            assert output_path.exists()
            with open(output_path) as f:
                lines = [json.loads(l) for l in f if l.strip()]

            assert len(lines) == 1
            assert lines[0]["base_task"] == "spatial"
            assert "above" in lines[0]["suffix"]


# ────────────────────────────────────────────────────────────────────────────
# sft_trainer
# ────────────────────────────────────────────────────────────────────────────

class TestSFTConfig:
    def test_defaults(self):
        from florence_forge.training.sft_trainer import SFTConfig
        cfg = SFTConfig()
        assert cfg.lr == pytest.approx(2e-5)
        assert cfg.num_epochs == 3
        assert cfg.gradient_accumulation_steps == 4
        assert cfg.save_best is True

    def test_from_config_dict(self):
        from unittest.mock import MagicMock
        from torch.utils.data import DataLoader, TensorDataset
        import torch
        from florence_forge.training.sft_trainer import SFTTrainer

        dummy_ds = TensorDataset(torch.zeros(2, 4), torch.zeros(2, 4, dtype=torch.long))
        dl = DataLoader(dummy_ds, batch_size=2)

        model = MagicMock()
        model.parameters.return_value = iter([torch.nn.Parameter(torch.zeros(1))])

        cfg = {
            "model": model,
            "tokenizer": MagicMock(),
            "dataloader": dl,
            "lr": 1e-5,
            "epochs": 2,
            "device": "cpu",
            "use_amp": False,
        }
        trainer = SFTTrainer.from_config(cfg)
        assert trainer.config.lr == pytest.approx(1e-5)
        assert trainer.config.num_epochs == 2


# ────────────────────────────────────────────────────────────────────────────
# tvp_pipeline
# ────────────────────────────────────────────────────────────────────────────

class TestTVPPipelineCheckpoint:
    def test_no_checkpoint_dir(self):
        from florence_forge.training.tvp_pipeline import TVPPipeline, PipelineStageConfig
        stage = PipelineStageConfig(name="sft", checkpoint_dir="")
        assert TVPPipeline._stage_checkpoint_exists(stage) is False

    def test_nonexistent_checkpoint(self):
        from florence_forge.training.tvp_pipeline import TVPPipeline, PipelineStageConfig
        stage = PipelineStageConfig(name="sft", checkpoint_dir="/nonexistent/path")
        assert TVPPipeline._stage_checkpoint_exists(stage) is False

    def test_existing_hf_checkpoint(self):
        from florence_forge.training.tvp_pipeline import TVPPipeline, PipelineStageConfig
        with tempfile.TemporaryDirectory() as tmpdir:
            final_dir = Path(tmpdir) / "final"
            final_dir.mkdir()
            (final_dir / "config.json").write_text("{}")
            stage = PipelineStageConfig(name="sft", checkpoint_dir=tmpdir)
            assert TVPPipeline._stage_checkpoint_exists(stage) is True

    def test_existing_pt_checkpoint(self):
        from florence_forge.training.tvp_pipeline import TVPPipeline, PipelineStageConfig
        with tempfile.TemporaryDirectory() as tmpdir:
            final_dir = Path(tmpdir) / "final"
            final_dir.mkdir()
            (final_dir / "model.pt").write_text("dummy")
            stage = PipelineStageConfig(name="sft", checkpoint_dir=tmpdir)
            assert TVPPipeline._stage_checkpoint_exists(stage) is True


# ────────────────────────────────────────────────────────────────────────────
# grpo_trainer: reward weighting
# ────────────────────────────────────────────────────────────────────────────

class TestGRPORewardWeighting:
    def test_default_three_reward_weights(self):
        """Default weights for 3 RMs should be [0.1, 0.2, 0.7]."""
        import torch
        from unittest.mock import MagicMock
        from florence_forge.training.grpo_trainer import GRPOTrainer

        model = MagicMock()
        model.parameters.return_value = iter([torch.nn.Parameter(torch.zeros(1))])
        model.config = MagicMock()
        model.config.use_cache = True
        ref_model = MagicMock()
        ref_model.parameters.return_value = iter([torch.nn.Parameter(torch.zeros(1))])

        reward_fns = [lambda t, m: 1.0, lambda t, m: 1.0, lambda t, m: 1.0]
        trainer = GRPOTrainer(
            model=model, ref_model=ref_model, tokenizer=MagicMock(),
            reward_fns=reward_fns, device="cpu",
        )
        assert trainer.reward_weights == pytest.approx([0.1, 0.2, 0.7])

    def test_custom_reward_weights_normalized(self):
        import torch
        from unittest.mock import MagicMock
        from florence_forge.training.grpo_trainer import GRPOTrainer

        model = MagicMock()
        model.parameters.return_value = iter([torch.nn.Parameter(torch.zeros(1))])
        ref_model = MagicMock()
        ref_model.parameters.return_value = iter([torch.nn.Parameter(torch.zeros(1))])

        reward_fns = [lambda t, m: 1.0, lambda t, m: 1.0]
        trainer = GRPOTrainer(
            model=model, ref_model=ref_model, tokenizer=MagicMock(),
            reward_fns=reward_fns, device="cpu",
            reward_weights=[1.0, 3.0],
        )
        assert sum(trainer.reward_weights) == pytest.approx(1.0)
        assert trainer.reward_weights[1] > trainer.reward_weights[0]
