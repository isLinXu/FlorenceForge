"""Reward models for GRPO / RL — compatibility shim; see rewards/ subpackage."""

from .rewards import (
    AgenticFormatRewardModel,
    AgenticQualityRewardModel,
    AgenticSelfCorrectionRewardModel,
    CountingRewardModel,
    DetectionAccuracyRewardModel,
    FormatRewardModel,
    MazeRewardModel,
    MixedAccuracyRewardModel,
    PathTracingRewardModel,
    QualityRewardModel,
    SpatialReasoningRewardModel,
    build_agentic_reward_models,
    build_reward_models,
)

__all__ = [
    "FormatRewardModel",
    "QualityRewardModel",
    "DetectionAccuracyRewardModel",
    "CountingRewardModel",
    "SpatialReasoningRewardModel",
    "MazeRewardModel",
    "PathTracingRewardModel",
    "MixedAccuracyRewardModel",
    "build_reward_models",
    "AgenticFormatRewardModel",
    "AgenticQualityRewardModel",
    "AgenticSelfCorrectionRewardModel",
    "build_agentic_reward_models",
]
