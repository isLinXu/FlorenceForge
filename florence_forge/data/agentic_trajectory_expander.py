"""Convert TVP chain-of-thought samples into Agentic meta-cognitive trajectories.

This module bridges the existing TVP ``TVPChainBuilder`` output (plain-text CoT)
into structured agentic format with ``<PLAN>``/``<ACT>``/``<VERIFY>``/``<REFLECT>``/``<DECIDE>``
delimiters, and provides error-injection augmentation for self-correction training.

Key design decisions (validated against FlorenceForge architecture):
  - Output samples use the same ``prefix``/``suffix`` JSONL schema as TVP converters,
    so they drop into ``MultiTaskDataset`` without any loader changes.
  - The ``prefix`` remains a single Florence-2 task token (e.g. ``<COUNT>``) per
    processor constraint — no multi-turn conversational text.
  - Agentic tokens are embedded in the ``suffix`` (the answer text), which gets
    encoded by ``encode_with_task()`` → ``processor(text=prefix, ...)`` then
    suffix is tokenized separately for label construction.
  - Error injection produces both *corrected* and *uncorrected* variants for
    self-correction RL: the corrected version teaches the ``<REFLECT>`` → fix
    pattern, while uncorrected variants test whether the model self-corrects.
  - Multi-round chains (PLAN→ACT→VERIFY repeated) are supported via
    ``build_multi_round_chain`` for long-horizon tasks.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..core.agentic_tokens import (
    wrap_phase,
    has_required_phases,
)
from ..core.visual_primitives import (
    COORDINATE_MAX,
    format_box,
    format_point,
    format_ref_box,
    sort_boxes_left_to_right,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agentic chain builders
# ---------------------------------------------------------------------------

class AgenticChainBuilder:
    """Build agentic meta-cognitive reasoning chains from structured inputs.

    Unlike ``TVPChainBuilder`` which produces free-form CoT text, this builder
    wraps each reasoning phase in meta-cognitive delimiter tokens.
    """

    # ---- Counting ----

    @staticmethod
    def build_counting_chain(
        label: str,
        boxes: List[List[int]],
        count: int,
        *,
        marker_style: str = "special",
        inject_error: bool = False,
    ) -> str:
        """Agentic counting: PLAN strategy → ACT scan → VERIFY count → DECIDE.

        With ``inject_error=True``, introduces a counting error in ACT that
        gets caught and corrected in VERIFY→REFLECT.
        """
        sorted_boxes = sort_boxes_left_to_right(boxes)

        plan = (
            f"I need to count all {label} instances in the image. "
            "Strategy: scan left-to-right, ground each instance, then verify the tally."
        )

        if inject_error:
            # Introduce a wrong count (skip last box)
            error_count = max(1, count - 1)
            act = (
                f"Scanning left-to-right, I find {error_count} {label} instances. "
                f"{format_ref_box(label, sorted_boxes[:-1] if len(sorted_boxes) > 1 else sorted_boxes, marker_style=marker_style)}"
            )
            verify = (
                f"Wait, let me recount. The last grounded box "
                f"{format_box([sorted_boxes[-1]], marker_style=marker_style)} "
                f"was not included in my tally. Actual count should be {count}, not {error_count}."
            )
            reflect = (
                f"I missed the rightmost {label} during my initial scan. "
                "I must ensure complete left-to-right coverage and recount after grounding."
            )
            decide = f"The correct count is {count} {label}."
        else:
            act = (
                f"Scanning left-to-right, I find {count} {label} instances. "
                f"{format_ref_box(label, sorted_boxes, marker_style=marker_style)}"
            )
            verify = (
                f"Verifying: I grounded {len(sorted_boxes)} boxes. "
                f"No overlaps or duplicates detected. Count confirmed: {count}."
            )
            reflect = ""
            decide = f"The total count is {count} {label}."

        parts = [
            wrap_phase("plan", plan),
            wrap_phase("act", act),
            wrap_phase("verify", verify),
        ]
        if reflect:
            parts.append(wrap_phase("reflect", reflect))
        parts.append(wrap_phase("decide", decide))
        return "".join(parts)

    # ---- Spatial reasoning ----

    @staticmethod
    def build_spatial_chain(
        observation: str,
        reasoning: str,
        answer: str,
        supporting_boxes: Optional[Dict[str, List[List[int]]]] = None,
        *,
        object_groundings: Optional[List[Tuple[str, List[List[int]], str]]] = None,
        marker_style: str = "special",
        inject_error: bool = False,
    ) -> str:
        """Agentic spatial reasoning: PLAN→ACT→VERIFY→(REFLECT)→DECIDE."""
        plan = (
            f"{observation} "
            "I need to ground the relevant objects and reason about their spatial relationship."
        )

        act_parts: List[str] = []
        if object_groundings:
            for lbl, bxs, note in object_groundings:
                sorted_bxs = sort_boxes_left_to_right(bxs)
                act_parts.append(f"{note}: {format_ref_box(lbl, sorted_bxs, marker_style=marker_style)}")
        elif supporting_boxes:
            for lbl, bxs in supporting_boxes.items():
                sorted_bxs = sort_boxes_left_to_right(bxs)
                act_parts.append(f"Grounded {lbl}: {format_ref_box(lbl, sorted_bxs, marker_style=marker_style)}")
        act_parts.append(reasoning)
        act = " ".join(act_parts)

        if inject_error:
            wrong_answer = _flip_spatial_answer(answer)
            verify = (
                f"Initial answer: {wrong_answer}. Let me verify by re-checking positions. "
                f"The spatial relationship actually supports '{answer}', not '{wrong_answer}'."
            )
            reflect = (
                "I initially confused left/right (or up/down). "
                "I should double-check coordinate relationships before committing."
            )
            decide = answer
        else:
            verify = f"Verified: spatial analysis is consistent. Answer: {answer}."
            reflect = ""
            decide = answer

        parts = [
            wrap_phase("plan", plan),
            wrap_phase("act", act),
            wrap_phase("verify", verify),
        ]
        if reflect:
            parts.append(wrap_phase("reflect", reflect))
        parts.append(wrap_phase("decide", decide))
        return "".join(parts)

    # ---- Maze navigation ----

    @staticmethod
    def build_maze_chain(
        solvable: bool,
        exploration_points: List[Tuple[int, int]],
        solution_points: Optional[List[Tuple[int, int]]] = None,
        answer: str = "",
        *,
        start_point: Optional[Tuple[int, int]] = None,
        end_point: Optional[Tuple[int, int]] = None,
        exploration_steps: Optional[List[Dict[str, Any]]] = None,
        marker_style: str = "special",
        inject_error: bool = False,
    ) -> str:
        """Agentic maze: multi-round PLAN→ACT→VERIFY exploration loop."""
        plan = (
            "I will use trial-and-error exploration. "
            "Strategy: probe passages, mark dead-ends, and backtrack when blocked."
        )

        act_parts: List[str] = []
        if start_point and end_point:
            act_parts.append(
                f"Start: {format_point([start_point], marker_style=marker_style)}, "
                f"Destination: {format_point([end_point], marker_style=marker_style)}"
            )

        if exploration_steps:
            for i, step in enumerate(exploration_steps, start=1):
                step_pts = [tuple(p) for p in step.get("points", [])]
                note = str(step.get("note", "")).strip()
                if step_pts:
                    act_parts.append(
                        f"Step {i}: {note} {format_point(step_pts, marker_style=marker_style)}"
                    )
        elif exploration_points:
            act_parts.append(format_point(exploration_points, marker_style=marker_style))

        act = " ".join(act_parts)

        if inject_error and solvable and solution_points:
            # Claim unsolvable when actually solvable, then catch error
            verify = (
                "Checking: I claimed no path exists. But re-examining the last passage, "
                f"I find a valid route: {format_point(solution_points, marker_style=marker_style)}. "
                "My earlier conclusion was wrong."
            )
            reflect = (
                "I prematurely concluded unsolvable without exhausting all branches. "
                "I must verify by checking every reachable cell before deciding."
            )
            decide = "true" if solvable else "false"
        elif inject_error and not solvable:
            # Claim solvable when actually unsolvable, then catch
            verify = (
                "Checking: I claimed a path exists. But re-tracing my route, "
                "I hit a wall at every branch. The maze has no valid path."
            )
            reflect = (
                "I confused a dead-end with a through-passage. "
                "I need to verify wall connectivity before committing."
            )
            decide = "false"
        else:
            if solvable and solution_points:
                verify = (
                    f"Path verified: {format_point(solution_points, marker_style=marker_style)} "
                    "reaches the destination without wall violations."
                )
            else:
                verify = "Exhaustive exploration complete. No valid path found."
            reflect = ""
            decide = answer or ("true" if solvable else "false")

        parts = [
            wrap_phase("plan", plan),
            wrap_phase("act", act),
            wrap_phase("verify", verify),
        ]
        if reflect:
            parts.append(wrap_phase("reflect", reflect))
        parts.append(wrap_phase("decide", decide))
        return "".join(parts)

    # ---- Phrase grounding ----

    @staticmethod
    def build_grounding_chain(
        caption: str,
        label: str,
        boxes: List[List[int]],
        *,
        marker_style: str = "special",
        inject_error: bool = False,
    ) -> str:
        """Agentic phrase grounding: PLAN→ACT→VERIFY→DECIDE."""
        sorted_boxes = sort_boxes_left_to_right(boxes)
        plan = (
            f"Task: locate '{label}' mentioned in caption: \"{caption}\". "
            "Strategy: scan the image for matching visual features."
        )

        if inject_error and len(sorted_boxes) >= 2:
            # Ground wrong region, then correct
            wrong_box = sorted_boxes[0]
            correct_box = sorted_boxes[-1]
            act = (
                f"Found {label} at {format_box([wrong_box], marker_style=marker_style)}."
            )
            verify = (
                f"Verifying: the box {format_box([wrong_box], marker_style=marker_style)} "
                f"doesn't match '{label}'. The correct location is "
                f"{format_box([correct_box], marker_style=marker_style)}."
            )
            reflect = (
                "I grabbed the first salient region instead of matching the caption. "
                "I must cross-check caption semantics with visual content."
            )
            decide = format_ref_box(label, sorted_boxes, marker_style=marker_style)
        else:
            act = f"Found {label} at {format_ref_box(label, sorted_boxes, marker_style=marker_style)}."
            verify = f"Verified: {len(sorted_boxes)} region(s) match '{label}'."
            reflect = ""
            decide = format_ref_box(label, sorted_boxes, marker_style=marker_style)

        parts = [
            wrap_phase("plan", plan),
            wrap_phase("act", act),
            wrap_phase("verify", verify),
        ]
        if reflect:
            parts.append(wrap_phase("reflect", reflect))
        parts.append(wrap_phase("decide", decide))
        return "".join(parts)

    # ---- Multi-round chain ----

    @staticmethod
    def build_multi_round_chain(
        rounds: List[Dict[str, Any]],
        final_answer: str,
        *,
        marker_style: str = "special",
        inject_error_at: Optional[int] = None,
    ) -> str:
        """Build a multi-round agentic chain with repeated PLAN→ACT→VERIFY cycles.

        Each round is a dict with keys: ``plan``, ``act``, ``verify``, and
        optional ``reflect``. If ``inject_error_at`` is set, that round index
        gets an error injected into ACT and a REFLECT phase is added.

        The final output wraps the answer in ``<DECIDE>`` and appends ``<DONE>``.

        Args:
            rounds: List of round dicts, each with plan/act/verify keys.
            final_answer: The final committed answer.
            marker_style: VP marker style for coordinates.
            inject_error_at: Round index (0-based) to inject an error.

        Returns:
            Concatenated agentic chain text.
        """
        parts: List[str] = []

        # Initial state summary if there are multiple rounds
        if len(rounds) > 1:
            state_summary = (
                f"Task requires {len(rounds)} rounds of exploration. "
                "I will plan, act, and verify at each step."
            )
            parts.append(wrap_phase("summarize_state", state_summary))

        for i, round_data in enumerate(rounds):
            inject = (inject_error_at is not None and i == inject_error_at)

            plan_text = round_data.get("plan", f"Round {i+1}: continue exploration.")
            act_text = round_data.get("act", "")
            verify_text = round_data.get("verify", "")

            if inject:
                # Modify act to contain an error, and verify to catch it
                error_act = round_data.get("error_act", act_text)
                act_text = error_act
                verify_text = (
                    f"Wait, checking round {i+1}: "
                    + round_data.get("error_verify", verify_text)
                )
                reflect_text = round_data.get("reflect", (
                    f"In round {i+1}, I made an error. "
                    "I need to be more careful with my observation."
                ))
            else:
                reflect_text = round_data.get("reflect", "")

            parts.append(wrap_phase("plan", plan_text))
            parts.append(wrap_phase("act", act_text))
            parts.append(wrap_phase("verify", verify_text))
            if reflect_text:
                parts.append(wrap_phase("reflect", reflect_text))

            # Mid-round state summary for long chains
            if len(rounds) > 2 and i < len(rounds) - 1:
                mid_summary = f"After round {i+1}: {round_data.get('state', 'progress made.')}"
                parts.append(wrap_phase("summarize_state", mid_summary))

        parts.append(wrap_phase("decide", final_answer))
        parts.append(wrap_phase("done", "Task completed."))
        return "".join(parts)


def wrap_done(content: str = "Task completed.") -> str:
    """Wrap content in <DONE> tags."""
    return wrap_phase("done", content)


def wrap_summarize_state(content: str) -> str:
    """Wrap content in <SUMMARIZE_STATE> tags."""
    return wrap_phase("summarize_state", content)


# ---------------------------------------------------------------------------
# Error injection utilities
# ---------------------------------------------------------------------------

_SPATIAL_FLIP_MAP = {
    "left": "right", "right": "left",
    "up": "down", "down": "up",
    "above": "below", "below": "above",
    "front": "behind", "behind": "front",
    "inside": "outside", "outside": "inside",
    "yes": "no", "no": "yes",
    "true": "false", "false": "true",
}


def _flip_spatial_answer(answer: str) -> str:
    """Flip a spatial answer to simulate an error."""
    lowered = answer.strip().lower()
    for key, val in _SPATIAL_FLIP_MAP.items():
        if key in lowered:
            return val
    return answer


def inject_coordinate_noise(
    boxes: List[List[int]],
    noise_range: int = 50,
    seed: Optional[int] = None,
) -> List[List[int]]:
    """Add random noise to box coordinates for error-injection augmentation."""
    rng = random.Random(seed)
    noisy: List[List[int]] = []
    for box in boxes:
        x1, y1, x2, y2 = box
        dx1 = rng.randint(-noise_range, noise_range)
        dy1 = rng.randint(-noise_range, noise_range)
        dx2 = rng.randint(-noise_range, noise_range)
        dy2 = rng.randint(-noise_range, noise_range)
        nx1 = max(0, min(COORDINATE_MAX, x1 + dx1))
        ny1 = max(0, min(COORDINATE_MAX, y1 + dy1))
        nx2 = max(0, min(COORDINATE_MAX, x2 + dx2))
        ny2 = max(0, min(COORDINATE_MAX, y2 + dy2))
        # Ensure x1 <= x2, y1 <= y2
        if nx1 > nx2:
            nx1, nx2 = nx2, nx1
        if ny1 > ny2:
            ny1, ny2 = ny2, ny1
        noisy.append([nx1, ny1, nx2, ny2])
    return noisy


# ---------------------------------------------------------------------------
# TVP → Agentic trajectory converter
# ---------------------------------------------------------------------------

class AgenticTrajectoryExpander:
    """Convert existing TVP JSONL samples to agentic meta-cognitive format.

    Reads TVP chain-of-thought JSONL files and produces agentic-format
    samples with structured ``<PLAN>``/``<ACT>``/``<VERIFY>``/``<DECIDE>``
    phases. Optionally produces error-injected variants for self-correction training.
    """

    def __init__(
        self,
        error_injection_rate: float = 0.3,
        seed: Optional[int] = None,
    ):
        self.error_injection_rate = max(0.0, min(1.0, error_injection_rate))
        self.rng = random.Random(seed)

    def expand_file(
        self,
        input_path: str,
        output_path: str,
        *,
        task_type: str = "AGENTIC_COUNT",
        marker_style: str = "special",
    ) -> int:
        """Expand a TVP JSONL file into agentic-format samples.

        Returns the number of samples written.
        """
        input_path = Path(input_path).absolute()
        output_path = Path(output_path).absolute()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        from ..core.tasks import get_task_config
        try:
            prompt = get_task_config(task_type).get("prompt", f"<{task_type}>")
        except KeyError:
            prompt = f"<{task_type}>"

        count = 0
        with open(input_path, "r", encoding="utf-8") as fin, \
             open(output_path, "w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                sample = self._expand_sample(
                    data, prompt=prompt, task_type=task_type,
                    marker_style=marker_style,
                )
                if sample:
                    fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    count += 1

        logger.info("Agentic expansion: %d samples -> %s", count, output_path)
        return count

    def _expand_sample(
        self,
        data: Dict[str, Any],
        *,
        prompt: str,
        task_type: str,
        marker_style: str,
    ) -> Optional[Dict[str, Any]]:
        """Convert a single TVP sample to agentic format."""
        base_task = str(data.get("base_task", "")).strip().lower()
        inject_error = self.rng.random() < self.error_injection_rate

        chain = self._build_chain(data, base_task, inject_error, marker_style)
        if chain is None:
            return None

        if not has_required_phases(chain):
            logger.warning("Generated chain missing required phases, skipping")
            return None

        sample = dict(data)  # preserve image path and metadata
        sample["prefix"] = prompt
        sample["suffix"] = chain
        sample["task_family"] = "agentic"
        sample["vp_task_type"] = task_type
        sample["agentic"] = True
        sample["error_injected"] = inject_error
        return sample

    def _build_chain(
        self,
        data: Dict[str, Any],
        base_task: str,
        inject_error: bool,
        marker_style: str,
    ) -> Optional[str]:
        """Dispatch to the appropriate agentic chain builder."""
        if base_task in ("counting", "count"):
            label = data.get("count_label", "object")
            boxes = data.get("vp_boxes") or _extract_boxes_from_suffix(data.get("suffix", ""))
            count = data.get("count", len(boxes))
            return AgenticChainBuilder.build_counting_chain(
                label=label, boxes=boxes, count=count,
                marker_style=marker_style, inject_error=inject_error,
            )

        if base_task in ("spatial", "vqa"):
            observation = data.get("observation", "Analyzing the scene.")
            reasoning = data.get("reasoning", "")
            answer = data.get("answer", "")
            supporting_boxes = data.get("supporting_boxes")
            return AgenticChainBuilder.build_spatial_chain(
                observation=observation, reasoning=reasoning, answer=answer,
                supporting_boxes=supporting_boxes,
                marker_style=marker_style, inject_error=inject_error,
            )

        if base_task == "maze":
            solvable = data.get("solvable", True)
            exploration_points = [tuple(p) for p in data.get("exploration_points", [])]
            solution_points = [tuple(p) for p in data.get("solution_points", [])] if solvable else None
            start_point = tuple(data["start_point"]) if "start_point" in data else None
            end_point = tuple(data["end_point"]) if "end_point" in data else None
            return AgenticChainBuilder.build_maze_chain(
                solvable=solvable,
                exploration_points=exploration_points,
                solution_points=solution_points,
                answer=data.get("answer", ""),
                start_point=start_point, end_point=end_point,
                marker_style=marker_style, inject_error=inject_error,
            )

        if base_task in ("grounding", "phrase_grounding"):
            caption = data.get("caption", data.get("text_input", ""))
            label = data.get("label", "object")
            boxes = data.get("vp_boxes") or _extract_boxes_from_suffix(data.get("suffix", ""))
            return AgenticChainBuilder.build_grounding_chain(
                caption=caption, label=label, boxes=boxes,
                marker_style=marker_style, inject_error=inject_error,
            )

        logger.warning("Unknown base_task for agentic expansion: %s", base_task)
        return None


def _extract_boxes_from_suffix(suffix: str) -> List[List[int]]:
    """Extract VP boxes from a TVP suffix text."""
    from ..core.visual_primitives import parse_vp_boxes
    return parse_vp_boxes(suffix)
