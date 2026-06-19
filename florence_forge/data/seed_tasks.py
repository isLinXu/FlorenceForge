"""Seed task templates and LLM-based trajectory augmentation for Agentic training.

This module provides:
  1. ``SeedTaskTemplate`` — structured definitions of high-level visual goals
  2. ``SEED_TASKS`` — a curated library of seed tasks across domains
  3. ``LLMTrajectoryAugmenter`` — generates agentic trajectories from seeds
     using an external LLM (GPT-4o, Claude, etc.)
  4. ``SeedTaskLibrary`` — manages, filters, and expands the seed collection

The seed task approach follows the data construction pipeline:
  seed_tasks → LLM trajectory generation → virtual execution validation
  → human refinement → data augmentation → training JSONL

Usage::

    from florence_forge.data.seed_tasks import SeedTaskLibrary, LLMTrajectoryAugmenter

    library = SeedTaskLibrary()
    seeds = library.get_seeds_by_domain("document")

    augmenter = LLMTrajectoryAugmenter(llm_client=my_openai_client)
    for seed in seeds:
        trajectory = augmenter.generate_trajectory(seed)
        # trajectory is ready for AgenticChainBuilder or direct JSONL export
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Seed task definition
# ---------------------------------------------------------------------------

@dataclass
class SeedTaskTemplate:
    """A single seed task for agentic trajectory generation.

    Attributes:
        task_id: Unique identifier.
        goal: High-level visual goal description.
        domain: Task domain (document, industrial, medical, retail, etc.).
        expected_steps: List of expected sub-task descriptions.
        difficulty: "easy", "medium", or "hard".
        native_prompts_used: Florence-2 native prompts expected in the trajectory.
        error_scenarios: Possible error scenarios for REFLECT training.
        num_rounds: Expected number of PLAN→ACT→VERIFY rounds.
        metadata: Additional task-specific metadata.
    """
    task_id: str
    goal: str
    domain: str
    expected_steps: List[str]
    difficulty: str = "medium"
    native_prompts_used: List[str] = field(default_factory=list)
    error_scenarios: List[str] = field(default_factory=list)
    num_rounds: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "goal": self.goal,
            "domain": self.domain,
            "expected_steps": self.expected_steps,
            "difficulty": self.difficulty,
            "native_prompts_used": self.native_prompts_used,
            "error_scenarios": self.error_scenarios,
            "num_rounds": self.num_rounds,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SeedTaskTemplate:
        return cls(
            task_id=data["task_id"],
            goal=data["goal"],
            domain=data["domain"],
            expected_steps=data.get("expected_steps", []),
            difficulty=data.get("difficulty", "medium"),
            native_prompts_used=data.get("native_prompts_used", []),
            error_scenarios=data.get("error_scenarios", []),
            num_rounds=data.get("num_rounds", 1),
            metadata=data.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# Curated seed task library
# ---------------------------------------------------------------------------

SEED_TASKS: List[SeedTaskTemplate] = [
    # --- Document domain ---
    SeedTaskTemplate(
        task_id="doc_invoice_001",
        goal="Analyze this invoice image, extract all field information, and verify amount consistency.",
        domain="document",
        expected_steps=[
            "Detect all text regions in the invoice",
            "Extract text from each detected region using OCR",
            "Parse extracted text into structured fields (vendor, date, items, total)",
            "Verify that line items sum matches the stated total",
        ],
        difficulty="medium",
        native_prompts_used=["<OCR_WITH_REGION>", "<OD>"],
        error_scenarios=[
            "OCR misses a small-amount line item",
            "Currency symbol misrecognized",
            "Date format inconsistency",
        ],
        num_rounds=3,
    ),
    SeedTaskTemplate(
        task_id="doc_blueprint_001",
        goal="Extract all dimension annotations and tolerance requirements from this engineering drawing.",
        domain="document",
        expected_steps=[
            "Detect all annotation boxes and text regions",
            "OCR each annotation region",
            "Classify annotations as dimensions vs tolerances",
            "Verify completeness against drawing borders",
        ],
        difficulty="hard",
        native_prompts_used=["<OCR_WITH_REGION>", "<OD>", "<DENSE_REGION_CAPTION>"],
        error_scenarios=[
            "Missing diameter symbol detection",
            "Tolerance values misread",
            "Annotation box overlap confusion",
        ],
        num_rounds=4,
    ),
    SeedTaskTemplate(
        task_id="doc_form_001",
        goal="Extract form fields and their values from this filled form image.",
        domain="document",
        expected_steps=[
            "Detect form field regions",
            "OCR each field label and value",
            "Match labels to corresponding values",
            "Verify all required fields are captured",
        ],
        difficulty="medium",
        native_prompts_used=["<OCR_WITH_REGION>", "<OD>"],
        error_scenarios=[
            "Handwritten text misrecognized",
            "Checkbox state misidentified",
            "Field-value pairing error",
        ],
        num_rounds=3,
    ),

    # --- Industrial domain ---
    SeedTaskTemplate(
        task_id="industrial_pcb_001",
        goal="Detect all solder joints on this PCB image, identify defects (cold solder, bridges), and generate a quality report.",
        domain="industrial",
        expected_steps=[
            "Detect all solder joint regions",
            "Classify each joint as good or defective",
            "Identify specific defect types (bridge, cold, insufficient)",
            "Generate summary defect report with locations",
        ],
        difficulty="hard",
        native_prompts_used=["<OD>", "<DENSE_REGION_CAPTION>"],
        error_scenarios=[
            "Small defects missed at low resolution",
            "Bridge defects confused with good joints",
            "False positive on reflective surfaces",
        ],
        num_rounds=4,
    ),
    SeedTaskTemplate(
        task_id="industrial_surface_001",
        goal="Inspect this surface for scratches, dents, and discoloration defects.",
        domain="industrial",
        expected_steps=[
            "Scan surface for anomaly regions",
            "Classify each anomaly type",
            "Measure defect severity (size, depth)",
            "Verify no missed regions in scan path",
        ],
        difficulty="hard",
        native_prompts_used=["<OD>", "<REGION_TO_DESCRIPTION>"],
        error_scenarios=[
            "Subtle scratches missed",
            "Lighting reflection mistaken for defect",
            "Defect boundaries imprecise",
        ],
        num_rounds=3,
    ),

    # --- Medical domain ---
    SeedTaskTemplate(
        task_id="medical_xray_001",
        goal="Analyze this chest X-ray, detect abnormal regions, and describe findings.",
        domain="medical",
        expected_steps=[
            "Detect regions of interest in the X-ray",
            "Describe each detected region",
            "Cross-reference findings for consistency",
            "Summarize diagnostic observations",
        ],
        difficulty="hard",
        native_prompts_used=["<OD>", "<REGION_TO_DESCRIPTION>", "<DENSE_REGION_CAPTION>"],
        error_scenarios=[
            "Subtle opacity missed",
            "Normal variant mistaken for pathology",
            "Bilateral findings not compared",
        ],
        num_rounds=4,
    ),

    # --- Retail domain ---
    SeedTaskTemplate(
        task_id="retail_shelf_001",
        goal="Analyze this retail shelf image, count product facings, and identify out-of-stock positions.",
        domain="retail",
        expected_steps=[
            "Detect all product regions on the shelf",
            "Count product facings per row",
            "Identify empty/gap positions",
            "Verify count against shelf planogram",
        ],
        difficulty="medium",
        native_prompts_used=["<OD>", "<COUNT>", "<DENSE_REGION_CAPTION>"],
        error_scenarios=[
            "Overlapping products counted as one",
            "Empty shelf space missed",
            "Product variants confused",
        ],
        num_rounds=3,
    ),

    # --- Spatial reasoning ---
    SeedTaskTemplate(
        task_id="spatial_scene_001",
        goal="Analyze the spatial relationships between objects in this scene image.",
        domain="spatial",
        expected_steps=[
            "Detect all major objects in the scene",
            "Ground each object with bounding boxes",
            "Determine pairwise spatial relationships",
            "Verify spatial consistency (transitivity check)",
        ],
        difficulty="medium",
        native_prompts_used=["<OD>", "<CAPTION_TO_PHRASE_GROUNDING>"],
        error_scenarios=[
            "Left/right confusion for mirrored objects",
            "Occluded objects not detected",
            "Depth ordering incorrect",
        ],
        num_rounds=2,
    ),

    # --- Counting ---
    SeedTaskTemplate(
        task_id="count_crowd_001",
        goal="Count the number of people in this crowded scene image.",
        domain="counting",
        expected_steps=[
            "Detect all person instances",
            "Handle occlusions and partial detections",
            "Count detected instances",
            "Verify count by recounting dense regions",
        ],
        difficulty="hard",
        native_prompts_used=["<OD>", "<COUNT>"],
        error_scenarios=[
            "Occluded persons missed",
            "Double counting in dense clusters",
            "Background patterns mistaken for people",
        ],
        num_rounds=3,
    ),

    # --- Maze navigation ---
    SeedTaskTemplate(
        task_id="maze_complex_001",
        goal="Determine if there is a valid path from start to end in this maze.",
        domain="maze",
        expected_steps=[
            "Identify start and end points",
            "Explore passages systematically",
            "Mark dead-ends and backtrack",
            "Verify path validity",
        ],
        difficulty="hard",
        native_prompts_used=["<REGION_PROPOSAL>"],
        error_scenarios=[
            "Premature conclusion before exhausting branches",
            "Wall violation in path",
            "Dead-end confused with through-passage",
        ],
        num_rounds=5,
    ),
]


# ---------------------------------------------------------------------------
# Seed task library manager
# ---------------------------------------------------------------------------

class SeedTaskLibrary:
    """Manage and query the seed task collection."""

    def __init__(self, seeds: Optional[List[SeedTaskTemplate]] = None):
        self._seeds: Dict[str, SeedTaskTemplate] = {}
        for seed in (seeds or SEED_TASKS):
            self._seeds[seed.task_id] = seed

    def add_seed(self, seed: SeedTaskTemplate) -> None:
        self._seeds[seed.task_id] = seed

    def get_seed(self, task_id: str) -> Optional[SeedTaskTemplate]:
        return self._seeds.get(task_id)

    def get_all_seeds(self) -> List[SeedTaskTemplate]:
        return list(self._seeds.values())

    def get_seeds_by_domain(self, domain: str) -> List[SeedTaskTemplate]:
        return [s for s in self._seeds.values() if s.domain == domain]

    def get_seeds_by_difficulty(self, difficulty: str) -> List[SeedTaskTemplate]:
        return [s for s in self._seeds.values() if s.difficulty == difficulty]

    def get_seeds_by_prompt(self, prompt: str) -> List[SeedTaskTemplate]:
        return [s for s in self._seeds.values() if prompt in s.native_prompts_used]

    def random_seeds(
        self,
        n: int,
        *,
        domain: Optional[str] = None,
        difficulty: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> List[SeedTaskTemplate]:
        """Get n random seeds, optionally filtered by domain/difficulty."""
        candidates = list(self._seeds.values())
        if domain:
            candidates = [s for s in candidates if s.domain == domain]
        if difficulty:
            candidates = [s for s in candidates if s.difficulty == difficulty]
        rng = random.Random(seed)
        return rng.sample(candidates, min(n, len(candidates)))

    def to_json(self, path: str | Path) -> Path:
        """Export all seeds to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [s.to_dict() for s in self._seeds.values()]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return path

    @classmethod
    def from_json(cls, path: str | Path) -> SeedTaskLibrary:
        """Load seeds from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        seeds = [SeedTaskTemplate.from_dict(d) for d in data]
        return cls(seeds=seeds)

    def __len__(self) -> int:
        return len(self._seeds)


# ---------------------------------------------------------------------------
# LLM-based trajectory augmenter
# ---------------------------------------------------------------------------

class LLMTrajectoryAugmenter:
    """Generate agentic trajectories from seed tasks using an external LLM.

    This is the LLM trajectory generation step in the data construction pipeline.
    The LLM receives the seed task + Florence-2 capability description and
    produces a structured agentic trajectory with meta-cognitive tokens.

    The generated trajectory is validated against the agentic format spec
    before being accepted.

    Usage::

        augmenter = LLMTrajectoryAugmenter(llm_client=openai_client)
        trajectory = augmenter.generate_trajectory(seed_task)
        if trajectory:
            # trajectory is a dict with "suffix" containing the agentic chain
            # ready for JSONL export
    """

    SYSTEM_PROMPT = """You are a visual task planning expert. Given a high-level visual goal and the capabilities of Florence-2 (a vision model with prompts: <OD>, <OCR>, <OCR_WITH_REGION>, <CAPTION>, <DETAILED_CAPTION>, <DENSE_REGION_CAPTION>, <REGION_PROPOSAL>, <REGION_TO_DESCRIPTION>, <OPEN_VOCABULARY_DETECTION>, <CAPTION_TO_PHRASE_GROUNDING>, <COUNT>), you need to:

1. Decompose the goal into a sequence of steps
2. For each step, produce structured reasoning using these meta-cognitive tokens:
   - <PLAN>strategic planning</PLAN>
   - <ACT>concrete action with Florence-2 prompts</ACT>
   - <VERIFY>verification of the action result</VERIFY>
   - <REFLECT>self-reflection on errors (only if an error occurred)</REFLECT>
   - <DECIDE>final decision / answer</DECIDE>
   - <SUMMARIZE_STATE>compress history to key state</SUMMARIZE_STATE>
   - <DONE>task completion signal</DONE>

Output format: A single text string containing the concatenated phases.
Each phase must use the exact token format shown above.
Content inside <ACT> must reference at least one Florence-2 native prompt.
If the task involves multiple rounds, repeat PLAN→ACT→VERIFY for each round.
End with <DECIDE> containing the final answer and <DONE>.

Example output:
<PLAN>I need to detect all objects first, then OCR text regions.</PLAN><ACT>Running <OD> to detect objects. Found 5 regions.</ACT><VERIFY>Verified: 5 regions detected, no duplicates.</VERIFY><PLAN>Now extract text from each region.</PLAN><ACT>Running <OCR_WITH_REGION> on detected regions.</ACT><VERIFY>Text extracted from all 5 regions.</VERIFY><DECIDE>Final result: 5 objects with text annotations.</DECIDE><DONE>Task completed.</DONE>"""

    def __init__(
        self,
        llm_client: Any,
        model_name: str = "gpt-4o",
        temperature: float = 0.7,
        error_injection_rate: float = 0.2,
    ):
        self.llm = llm_client
        self.model_name = model_name
        self.temperature = temperature
        self.error_injection_rate = error_injection_rate

    def generate_trajectory(
        self,
        seed: SeedTaskTemplate,
        *,
        inject_error: Optional[bool] = None,
    ) -> Optional[Dict[str, Any]]:
        """Generate a single agentic trajectory from a seed task.

        Args:
            seed: The seed task template.
            inject_error: Whether to inject an error scenario. If None,
                         uses the error_injection_rate probability.

        Returns:
            A dict with the trajectory text and metadata, or None if generation failed.
        """
        if inject_error is None:
            inject_error = random.random() < self.error_injection_rate

        user_prompt = self._build_user_prompt(seed, inject_error)

        try:
            response = self.llm.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=2048,
            )
            trajectory_text = response.choices[0].message.content.strip()
        except Exception as exc:
            logger.warning("LLM trajectory generation failed for %s: %s", seed.task_id, exc)
            return None

        # Validate the generated trajectory
        if not self._validate_trajectory(trajectory_text):
            logger.warning("Generated trajectory for %s failed validation", seed.task_id)
            return None

        return {
            "task_id": seed.task_id,
            "suffix": trajectory_text,
            "goal": seed.goal,
            "domain": seed.domain,
            "difficulty": seed.difficulty,
            "error_injected": inject_error,
            "num_rounds": seed.num_rounds,
            "expected_steps": seed.expected_steps,
            "native_prompts_used": seed.native_prompts_used,
        }

    def _build_user_prompt(self, seed: SeedTaskTemplate, inject_error: bool) -> str:
        parts = [
            f"Task: {seed.goal}",
            f"Domain: {seed.domain}",
            f"Expected steps: {seed.num_rounds} round(s)",
            f"Expected sub-tasks: {json.dumps(seed.expected_steps)}",
            f"Available Florence-2 prompts: {', '.join(seed.native_prompts_used)}",
        ]
        if inject_error and seed.error_scenarios:
            scenario = random.choice(seed.error_scenarios)
            parts.append(
                f"\nIMPORTANT: Introduce an error scenario in one of the ACT phases: '{scenario}'. "
                "Then catch it in VERIFY, acknowledge in REFLECT, and correct in DECIDE."
            )
        parts.append("\nGenerate the complete agentic trajectory now.")
        return "\n".join(parts)

    def _validate_trajectory(self, text: str) -> bool:
        """Validate that the trajectory has required agentic phases."""
        from ..core.agentic_tokens import has_required_phases
        return has_required_phases(text)

    def batch_generate(
        self,
        seeds: List[SeedTaskTemplate],
        *,
        max_trajectories_per_seed: int = 1,
    ) -> List[Dict[str, Any]]:
        """Generate trajectories for multiple seeds.

        Returns a list of trajectory dicts.
        """
        results: List[Dict[str, Any]] = []
        for seed in seeds:
            for _ in range(max_trajectories_per_seed):
                traj = self.generate_trajectory(seed)
                if traj:
                    results.append(traj)
        return results

    def to_jsonl(
        self,
        trajectories: List[Dict[str, Any]],
        output_path: str | Path,
        *,
        task_type: str = "AGENTIC_COUNT",
    ) -> Path:
        """Write trajectories to a JSONL file for training.

        Each line is a JSON object with prefix/suffix schema compatible
        with MultiTaskDataset.
        """
        from ..core.tasks import get_task_config

        try:
            prompt = get_task_config(task_type).get("prompt", f"<{task_type}>")
        except KeyError:
            prompt = f"<{task_type}>"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for traj in trajectories:
                record = {
                    "image": "",  # to be filled by caller
                    "prefix": prompt,
                    "suffix": traj["suffix"],
                    "task_family": "agentic",
                    "vp_task_type": task_type,
                    "agentic": True,
                    "error_injected": traj.get("error_injected", False),
                    "num_rounds": traj.get("num_rounds", 1),
                    "goal": traj.get("goal", ""),
                    "domain": traj.get("domain", ""),
                    "expected_steps": traj.get("expected_steps", []),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info("Wrote %d LLM-augmented trajectories to %s", len(trajectories), output_path)
        return output_path
