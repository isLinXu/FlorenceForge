"""Native task preservation data mixer for Agentic training.

When training with agentic meta-cognitive tokens, it's critical to mix in
native Florence-2 task data (30-50%) to prevent catastrophic forgetting
of basic visual capabilities (OD, OCR, CAPTION, etc.).

This module provides:
  1. ``NativeTaskPreserver`` — mixes agentic and native task data at a
     configurable ratio
  2. ``NativeTaskSampler`` — samples native task data from existing JSONL
     files or generates synthetic samples
  3. ``compute_mix_ratio`` — utility to calculate the correct number of
     native samples needed

Usage::

    from florence_forge.data.native_preservation import NativeTaskPreserver

    preserver = NativeTaskPreserver(native_ratio=0.3)
    mixed = preserver.mix(
        agentic_jsonl="data/agentic_train.jsonl",
        native_jsonl="data/native_train.jsonl",
        output_path="data/mixed_train.jsonl",
    )
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)

#: Default ratio of native task data to include (prevents catastrophic forgetting).
DEFAULT_NATIVE_RATIO: float = 0.3

#: Tasks suitable for preservation (basic visual capabilities).
PRESERVATION_TASKS: List[str] = [
    "CAPTION", "DETAILED_CAPTION", "MORE_DETAILED_CAPTION",
    "OD", "OCR", "OCR_WITH_REGION",
    "DENSE_REGION_CAPTION", "REGION_PROPOSAL",
    "OPEN_VOCABULARY_DETECTION", "CAPTION_TO_PHRASE_GROUNDING",
    "COUNT_VP", "OD_VP", "PHRASE_GROUNDING_VP",
    "COUNT_VP_COT", "SPATIAL_VP", "MAZE_VP", "PATH_VP",
]


class NativeTaskSampler:
    """Sample native task data from existing JSONL files.

    Supports sampling from multiple task files with configurable per-task
    weights to ensure balanced coverage of basic capabilities.
    """

    def __init__(
        self,
        task_files: Optional[Dict[str, str]] = None,
        seed: int = 42,
    ):
        """Initialize the sampler.

        Args:
            task_files: Mapping of task_type → JSONL file path.
            seed: Random seed for reproducible sampling.
        """
        self.task_files: Dict[str, str] = task_files or {}
        self.rng = random.Random(seed)

    def add_task_file(self, task_type: str, file_path: str) -> None:
        self.task_files[task_type] = file_path

    def sample(
        self,
        n_samples: int,
        *,
        task_weights: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """Sample n_samples from native task files.

        Args:
            n_samples: Total number of samples to draw.
            task_weights: Optional per-task weights for sampling distribution.
                         If None, samples are distributed uniformly across tasks.

        Returns:
            List of sample dicts (JSONL line format).
        """
        if not self.task_files:
            logger.warning("No native task files configured; returning empty list")
            return []

        # Load all samples from each task file
        task_samples: Dict[str, List[Dict[str, Any]]] = {}
        for task_type, file_path in self.task_files.items():
            file_path = Path(file_path)
            if not file_path.exists():
                logger.warning("Native task file not found: %s", file_path)
                continue
            samples = []
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        samples.append(json.loads(line))
            task_samples[task_type] = samples
            logger.info("Loaded %d samples for native task %s", len(samples), task_type)

        if not task_samples:
            return []

        # Distribute samples across tasks
        weights = task_weights or {t: 1.0 for t in task_samples}
        total_weight = sum(weights.get(t, 1.0) for t in task_samples)
        per_task_counts = {
            t: max(1, int(n_samples * weights.get(t, 1.0) / total_weight))
            for t in task_samples
        }

        # Sample from each task
        result: List[Dict[str, Any]] = []
        for task_type, samples in task_samples.items():
            count = min(per_task_counts[task_type], len(samples))
            if count <= 0:
                continue
            chosen = self.rng.sample(samples, count) if count < len(samples) else samples[:count]
            for s in chosen:
                s = dict(s)  # copy
                s.setdefault("task_family", "native")
                s.setdefault("agentic", False)
                result.append(s)

        # Shuffle the combined result
        self.rng.shuffle(result)
        return result[:n_samples]


class NativeTaskPreserver:
    """Mix agentic and native task data at a configurable ratio.

    This prevents catastrophic forgetting of basic Florence-2 visual
    capabilities when training with agentic meta-cognitive tokens.

    The mixer reads agentic JSONL data, samples native task data from
    provided files, and writes a combined JSONL file with the specified
    ratio.
    """

    def __init__(
        self,
        native_ratio: float = DEFAULT_NATIVE_RATIO,
        seed: int = 42,
    ):
        """Initialize the preserver.

        Args:
            native_ratio: Proportion of native task data (0.0 to 1.0).
                         0.3 means 30% native + 70% agentic.
            seed: Random seed for reproducible mixing.
        """
        self.native_ratio = max(0.0, min(1.0, native_ratio))
        self.rng = random.Random(seed)

    def compute_native_count(self, agentic_count: int) -> int:
        """Calculate the number of native samples needed.

        native_count / (native_count + agentic_count) = native_ratio
        => native_count = agentic_count * native_ratio / (1 - native_ratio)
        """
        if self.native_ratio >= 1.0:
            return agentic_count  # edge case
        if self.native_ratio <= 0.0:
            return 0
        return int(agentic_count * self.native_ratio / (1.0 - self.native_ratio))

    def mix(
        self,
        agentic_jsonl: str | Path,
        native_sampler: NativeTaskSampler,
        output_path: str | Path,
        *,
        task_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Mix agentic and native data into a single JSONL file.

        Args:
            agentic_jsonl: Path to the agentic training data JSONL.
            native_sampler: Sampler for native task data.
            output_path: Path to write the mixed JSONL.
            task_weights: Optional per-task weights for native sampling.

        Returns:
            Summary dict with counts and ratio.
        """
        agentic_path = Path(agentic_jsonl)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load agentic samples
        agentic_samples: List[Dict[str, Any]] = []
        with open(agentic_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    agentic_samples.append(json.loads(line))

        agentic_count = len(agentic_samples)
        native_count = self.compute_native_count(agentic_count)

        logger.info(
            "Mixing: %d agentic + %d native (ratio=%.2f)",
            agentic_count, native_count, self.native_ratio,
        )

        # Sample native data
        native_samples = native_sampler.sample(
            native_count,
            task_weights=task_weights,
        )

        # Combine and shuffle
        combined = agentic_samples + native_samples
        self.rng.shuffle(combined)

        # Write output
        with open(output_path, "w", encoding="utf-8") as f:
            for sample in combined:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        actual_native = len(native_samples)
        actual_ratio = actual_native / max(len(combined), 1)

        summary = {
            "agentic_count": agentic_count,
            "native_count": actual_native,
            "total_count": len(combined),
            "target_native_ratio": self.native_ratio,
            "actual_native_ratio": round(actual_ratio, 4),
            "output_path": str(output_path),
        }
        logger.info("Mixed data written to %s: %s", output_path, summary)
        return summary

    @staticmethod
    def mix_jsonl_files(
        agentic_jsonl: str | Path,
        native_jsonl: str | Path,
        output_path: str | Path,
        *,
        native_ratio: float = DEFAULT_NATIVE_RATIO,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Convenience: mix two JSONL files at a given ratio.

        Both files should already be in prefix/suffix JSONL format.
        Native samples are sampled (not all used) to achieve the target ratio.
        """
        agentic_path = Path(agentic_jsonl)
        native_path = Path(native_jsonl)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        rng = random.Random(seed)

        # Load agentic
        agentic_samples = []
        with open(agentic_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    agentic_samples.append(json.loads(line))

        # Load native
        native_samples = []
        with open(native_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    native_samples.append(json.loads(line))

        agentic_count = len(agentic_samples)
        if native_ratio >= 1.0:
            needed_native = len(native_samples)
        elif native_ratio <= 0.0:
            needed_native = 0
        else:
            needed_native = int(agentic_count * native_ratio / (1.0 - native_ratio))

        # Sample native
        if needed_native < len(native_samples):
            native_samples = rng.sample(native_samples, needed_native)
        else:
            logger.info(
                "Requested %d native samples but only %d available; using all",
                needed_native, len(native_samples),
            )

        combined = agentic_samples + native_samples
        rng.shuffle(combined)

        with open(output_path, "w", encoding="utf-8") as f:
            for sample in combined:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        summary = {
            "agentic_count": agentic_count,
            "native_count": len(native_samples),
            "total_count": len(combined),
            "target_native_ratio": native_ratio,
            "actual_native_ratio": round(len(native_samples) / max(len(combined), 1), 4),
            "output_path": str(output_path),
        }
        logger.info("Mixed JSONL written to %s: %s", output_path, summary)
        return summary
