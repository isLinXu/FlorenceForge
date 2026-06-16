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
        """Run Stage 1: Supervised Fine-Tuning."""
        from ..training.sft_trainer import SFTTrainer  # lazy import

        cfg = self._load_stage_config(stage)
        trainer = SFTTrainer.from_config(cfg)
        results = trainer.train()
        return results

    def _run_opd(self, stage: PipelineStageConfig) -> Dict[str, Any]:
        """Run Stage 2: On-Policy Distillation."""
        from ..training.opd_trainer import OPDTrainer

        cfg = self._load_stage_config(stage)

        # Resolve student model from previous stage checkpoint
        sft_stage = self.stage_map.get("sft")
        if sft_stage and sft_stage.checkpoint_dir:
            cfg.setdefault("student_model", str(
                Path(sft_stage.checkpoint_dir) / "final"
            ))

        trainer = OPDTrainer(
            student=cfg.get("student"),
            teachers=cfg.get("teachers", []),
            teacher_weights=cfg.get("teacher_weights"),
            temperature=cfg.get("temperature", 2.0),
            ce_coeff=cfg.get("ce_coeff", 0.3),
        )
        results = {}
        for epoch in range(cfg.get("epochs", 2)):
            epoch_results = trainer.train_epoch(
                dataloader=cfg.get("dataloader"),
                epoch=epoch,
                save_dir=Path(stage.checkpoint_dir),
            )
            results[f"epoch_{epoch}"] = epoch_results
        return results

    def _run_grpo(self, stage: PipelineStageConfig) -> Dict[str, Any]:
        """Run Stage 3: GRPO Reinforcement Learning."""
        from ..training.grpo_trainer import GRPOTrainer
        from ..training.reward_models import build_reward_models

        cfg = self._load_stage_config(stage)

        # Resolve model from previous stage checkpoint
        opd_stage = self.stage_map.get("opd")
        if opd_stage and opd_stage.checkpoint_dir:
            cfg.setdefault("model_name_or_path", str(
                Path(opd_stage.checkpoint_dir) / "final"
            ))

        reward_fns = build_reward_models(
            task_type=cfg.get("task_type", "mixed"),
        )

        trainer = GRPOTrainer(
            model=cfg.get("model"),
            ref_model=cfg.get("ref_model"),
            tokenizer=cfg.get("tokenizer"),
            reward_fns=reward_fns,
            group_size=cfg.get("group_size", 4),
            kl_coeff=cfg.get("kl_coeff", 0.04),
        )

        results = {}
        for epoch in range(cfg.get("epochs", 2)):
            epoch_results = trainer.train_epoch(
                dataloader=cfg.get("dataloader"),
                epoch=epoch,
            )
            results[f"epoch_{epoch}"] = epoch_results
        return results

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
