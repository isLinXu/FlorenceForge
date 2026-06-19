"""TVP Three-Stage Training Pipeline.

Implements the Thinking-with-Visual-Primitives training paradigm:
  Stage 1: Supervised Fine-Tuning (SFT) — teach basic VP format
  Stage 2: On-Policy Distillation (OPD) — unify expert models
  Stage 3: GRPO Reinforcement Learning — refine with reward signals

Each stage builds upon the previous one, progressively improving
the model's ability to reason with visual primitives.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class PipelineStageConfig:
    """Configuration for a single pipeline stage."""

    name: str = ""
    enabled: bool = True
    config_path: str = ""
    checkpoint_dir: str = ""
    depends_on: str = ""  # Name of preceding stage


@dataclass
class TVPPipelineConfig:
    """Full three-stage pipeline configuration."""

    stages: List[PipelineStageConfig] = field(default_factory=lambda: [
        PipelineStageConfig(
            name="sft",
            enabled=True,
            config_path="configs/tvp/sft.yaml",
            checkpoint_dir="outputs/tvp/sft",
        ),
        PipelineStageConfig(
            name="opd",
            enabled=True,
            config_path="configs/tvp/opd.yaml",
            checkpoint_dir="outputs/tvp/opd",
            depends_on="sft",
        ),
        PipelineStageConfig(
            name="grpo",
            enabled=True,
            config_path="configs/tvp/grpo.yaml",
            checkpoint_dir="outputs/tvp/grpo",
            depends_on="opd",
        ),
    ])

    output_dir: str = "outputs/tvp"
    resume_from: Optional[str] = None  # Stage name to resume from


class TVPPipeline:
    """Orchestrate the three-stage TVP training pipeline.

    Manages stage execution order, checkpoint propagation between
    stages, and resumption from failures.
    """

    def __init__(self, config: TVPPipelineConfig):
        self.config = config
        self.stage_map = {s.name: s for s in config.stages}
        self.completed_stages: set = set()

    def run(self) -> Dict[str, Any]:
        """Execute the full pipeline.

        Returns:
            Dictionary mapping stage names to their final metrics.
        """
        all_results: Dict[str, Any] = {}
        resume_from = self.config.resume_from

        for stage in self.config.stages:
            if not stage.enabled:
                logger.info("Skipping disabled stage: %s", stage.name)
                continue

            # Check dependency
            if stage.depends_on and stage.depends_on not in self.completed_stages:
                if resume_from and resume_from != stage.name:
                    logger.warning(
                        "Stage '%s' depends on '%s' which hasn't completed. "
                        "Attempting to use existing checkpoint.",
                        stage.name, stage.depends_on,
                    )
                else:
                    raise RuntimeError(
                        f"Stage '{stage.name}' depends on '{stage.depends_on}' "
                        f"which hasn't completed yet."
                    )

            # Skip if resuming and this stage is before resume point
            if resume_from and stage.name != resume_from and \
               stage.name not in self.completed_stages:
                # Find if resume_from comes after this stage
                stage_names = [s.name for s in self.config.stages]
                if stage.name in stage_names and resume_from in stage_names:
                    if stage_names.index(stage.name) < stage_names.index(resume_from):
                        logger.info("Skipping stage '%s' (before resume point '%s')",
                                    stage.name, resume_from)
                        self.completed_stages.add(stage.name)
                        continue

            logger.info("=" * 60)
            logger.info("Starting stage: %s", stage.name)
            logger.info("=" * 60)

            # Skip if checkpoint already exists (avoid re-training)
            if self._stage_checkpoint_exists(stage):
                logger.info(
                    "Stage '%s' checkpoint already exists at '%s'. Skipping.",
                    stage.name, stage.checkpoint_dir,
                )
                self.completed_stages.add(stage.name)
                all_results[stage.name] = {"status": "skipped_existing_checkpoint"}
                continue

            try:
                if stage.name == "sft":
                    results = self._run_sft(stage)
                elif stage.name == "opd":
                    results = self._run_opd(stage)
                elif stage.name == "grpo":
                    results = self._run_grpo(stage)
                else:
                    logger.warning("Unknown stage: %s, skipping", stage.name)
                    continue

                all_results[stage.name] = results
                self.completed_stages.add(stage.name)
                logger.info("Stage '%s' completed: %s", stage.name, results)

            except Exception as exc:
                logger.error("Stage '%s' failed: %s", stage.name, exc)
                raise

        return all_results

    def _run_sft(self, stage: PipelineStageConfig) -> Dict[str, Any]:
        """Run Stage 1: Supervised Fine-Tuning via MultiTaskTrainer bridge."""
        from .tvp_training import run_tvp_sft_with_multitask_trainer

        config_path = stage.config_path
        if not config_path:
            raise ValueError("SFT stage is missing config_path")

        return run_tvp_sft_with_multitask_trainer(
            config_path,
            checkpoint_dir=stage.checkpoint_dir or None,
        )

    def _run_opd(self, stage: PipelineStageConfig) -> Dict[str, Any]:
        """Run Stage 2: On-Policy Distillation."""
        from .tvp_training import run_tvp_opd

        sft_stage = self.stage_map.get("sft")
        student_checkpoint = None
        if sft_stage and sft_stage.checkpoint_dir:
            student_checkpoint = str(Path(sft_stage.checkpoint_dir) / "final")

        return run_tvp_opd(
            stage.config_path,
            checkpoint_dir=stage.checkpoint_dir or None,
            student_checkpoint=student_checkpoint,
        )

    def _run_grpo(self, stage: PipelineStageConfig) -> Dict[str, Any]:
        """Run Stage 3: GRPO Reinforcement Learning."""
        from .tvp_training import run_tvp_grpo

        opd_stage = self.stage_map.get("opd")
        sft_stage = self.stage_map.get("sft")
        model_checkpoint = None
        ref_checkpoint = None
        if opd_stage and opd_stage.checkpoint_dir:
            model_checkpoint = str(Path(opd_stage.checkpoint_dir) / "final")
        if sft_stage and sft_stage.checkpoint_dir:
            ref_checkpoint = str(Path(sft_stage.checkpoint_dir) / "final")

        return run_tvp_grpo(
            stage.config_path,
            checkpoint_dir=stage.checkpoint_dir or None,
            model_checkpoint=model_checkpoint,
            ref_checkpoint=ref_checkpoint,
        )

    @staticmethod
    def _stage_checkpoint_exists(stage: PipelineStageConfig) -> bool:
        """Return True if a usable checkpoint already exists for *stage*."""
        if not stage.checkpoint_dir:
            return False
        final_dir = Path(stage.checkpoint_dir) / "final"
        # Accept either a HF-style config.json or a raw model.pt
        return (final_dir / "config.json").exists() or (final_dir / "model.pt").exists()

    @staticmethod
    def _load_stage_config(stage: PipelineStageConfig) -> Dict[str, Any]:
        """Load YAML configuration for a stage."""
        config_path = Path(stage.config_path)
        if config_path.exists():
            with open(config_path, "r") as f:
                return yaml.safe_load(f) or {}
        logger.warning("Config not found: %s, using defaults", config_path)
        return {}


def create_pipeline_from_yaml(config_path: str) -> TVPPipeline:
    """Create a TVPPipeline from a YAML configuration file."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    stages = [
        PipelineStageConfig(**stage_cfg)
        for stage_cfg in cfg.get("stages", [])
    ]
    pipeline_cfg = TVPPipelineConfig(
        stages=stages,
        output_dir=cfg.get("output_dir", "outputs/tvp"),
        resume_from=cfg.get("resume_from"),
    )
    return TVPPipeline(pipeline_cfg)
