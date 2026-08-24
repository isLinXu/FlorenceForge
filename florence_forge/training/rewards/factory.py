"""Reward model factory functions."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from .accuracy import (
    CountingRewardModel,
    DetectionAccuracyRewardModel,
    MazeRewardModel,
    MixedAccuracyRewardModel,
    PathTracingRewardModel,
    SpatialReasoningRewardModel,
)
from .format import FormatRewardModel
from .quality import QualityRewardModel

def build_reward_models(
    task_type: str = "mixed",
    judge_model: Optional[Any] = None,
    judge_tokenizer: Optional[Any] = None,
) -> List[Callable[[str, Dict], float]]:
    """Build a list of reward functions for the given task type.

    Args:
        task_type: One of "counting", "spatial", "maze", "path",
                   "od", "grounding", or "mixed".
        judge_model: Optional LLM judge for QualityRewardModel.
        judge_tokenizer: Optional tokenizer for the judge model.

    Returns:
        List of reward callables [(text, metadata) -> float].
    """
    format_rm = FormatRewardModel()
    quality_rm = QualityRewardModel(judge_model, judge_tokenizer)

    accuracy_map = {
        "counting": CountingRewardModel,
        "spatial": SpatialReasoningRewardModel,
        "maze": MazeRewardModel,
        "path": PathTracingRewardModel,
        "od": DetectionAccuracyRewardModel,
        "grounding": DetectionAccuracyRewardModel,
    }

    if task_type in accuracy_map:
        acc_rm = accuracy_map[task_type]()
    else:
        acc_rm = MixedAccuracyRewardModel()

    return [format_rm, quality_rm, acc_rm]
