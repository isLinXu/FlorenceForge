"""Evaluation metrics for Agentic meta-cognitive reasoning.

Provides specialized metrics beyond standard task accuracy:
  1. **Planning accuracy**: Does the PLAN phase produce a viable step sequence?
  2. **Tool-call correctness**: Does ACT select appropriate Florence-2 prompts?
  3. **Error recovery rate**: Given an error, does REFLECT→corrected DECIDE work?
  4. **Consistency retention**: Do multi-round chains maintain object/scene consistency?
  5. **Format validity**: Are all required phases present and well-formed?
  6. **Native capability preservation**: How much do base OD/OCR/etc degrade?

These metrics operate on decoded text and ground-truth metadata,
requiring no torch dependencies for the core logic.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..core.agentic_tokens import (
    AGENTIC_PHASE_ORDER,
    AGENTIC_PHASE_TOKENS,
    REQUIRED_PHASES,
    extract_all_phases,
    extract_phase,
    get_phase_order,
    has_required_phases,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AgenticMetrics:
    """Container for all agentic evaluation metrics."""
    format_validity: float = 0.0
    planning_accuracy: float = 0.0
    tool_call_correctness: float = 0.0
    error_recovery_rate: float = 0.0
    consistency_score: float = 0.0
    native_capability_preservation: float = 0.0
    total_samples: int = 0
    error_injected_samples: int = 0
    details: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format_validity": round(self.format_validity, 4),
            "planning_accuracy": round(self.planning_accuracy, 4),
            "tool_call_correctness": round(self.tool_call_correctness, 4),
            "error_recovery_rate": round(self.error_recovery_rate, 4),
            "consistency_score": round(self.consistency_score, 4),
            "native_capability_preservation": round(self.native_capability_preservation, 4),
            "total_samples": self.total_samples,
            "error_injected_samples": self.error_injected_samples,
            "details": self.details,
        }

    def summary(self) -> str:
        lines = [
            "Agentic Evaluation Summary",
            "=" * 50,
            f"  Total samples:          {self.total_samples}",
            f"  Error-injected samples: {self.error_injected_samples}",
            f"  Format validity:        {self.format_validity:.2%}",
            f"  Planning accuracy:      {self.planning_accuracy:.2%}",
            f"  Tool-call correctness:  {self.tool_call_correctness:.2%}",
            f"  Error recovery rate:    {self.error_recovery_rate:.2%}",
            f"  Consistency score:      {self.consistency_score:.2%}",
            f"  Native preservation:    {self.native_capability_preservation:.2%}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

# Native Florence-2 task prompts that can appear inside <ACT> tags
_NATIVE_PROMPTS = {
    "<OD>", "<OCR>", "<OCR_WITH_REGION>", "<CAPTION>",
    "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>",
    "<DENSE_REGION_CAPTION>", "<REGION_PROPOSAL>",
    "<REGION_TO_SEGMENTATION>", "<OPEN_VOCABULARY_DETECTION>",
    "<CAPTION_TO_PHRASE_GROUNDING>", "<REGION_TO_CATEGORY>",
    "<REGION_TO_DESCRIPTION>", "<COUNT>",
    "<REFERRING_EXPRESSION_SEGMENTATION>",
}


def evaluate_format_validity(text: str) -> Tuple[bool, List[str]]:
    """Check if the output conforms to agentic format specification.

    Returns (is_valid, list_of_issues).
    """
    issues: List[str] = []

    # 1. Must have required phases
    if not has_required_phases(text):
        missing = [p for p in REQUIRED_PHASES if not extract_phase(text, p)]
        issues.append(f"Missing required phases: {missing}")

    # 2. All opened tags must be closed
    for phase in AGENTIC_PHASE_ORDER:
        open_tok, close_tok = AGENTIC_PHASE_TOKENS[phase]
        open_count = text.count(open_tok)
        close_count = text.count(close_tok)
        if open_count != close_count:
            issues.append(
                f"Tag mismatch for {phase}: {open_count} open vs {close_count} close"
            )

    # 3. Phase ordering check (PLAN should come before ACT, ACT before DECIDE)
    phase_seq = get_phase_order(text)
    canonical = [p for p in AGENTIC_PHASE_ORDER if p in phase_seq]
    if phase_seq != canonical:
        issues.append(f"Phase order incorrect: {phase_seq} vs expected {canonical}")

    # 4. Non-empty phase content
    all_phases = extract_all_phases(text)
    for phase, contents in all_phases.items():
        for content in contents:
            if not content.strip():
                issues.append(f"Empty content in <{phase}> phase")

    return len(issues) == 0, issues


def evaluate_planning_accuracy(
    plan_text: str,
    expected_steps: Optional[List[str]] = None,
) -> float:
    """Evaluate whether the PLAN phase produces a viable step sequence.

    If ``expected_steps`` is provided, checks for keyword overlap.
    Otherwise, checks if the plan references at least one native Florence-2 prompt.

    Returns score in [0, 1].
    """
    if not plan_text.strip():
        return 0.0

    if expected_steps:
        # Check keyword overlap between plan and expected steps
        plan_lower = plan_text.lower()
        matched = sum(
            1 for step in expected_steps
            if any(kw in plan_lower for kw in step.lower().split()[:3])
        )
        return matched / max(len(expected_steps), 1)

    # Heuristic: check if plan references native Florence-2 capabilities
    plan_lower = plan_text.lower()
    native_mentions = sum(1 for p in _NATIVE_PROMPTS if p.lower() in plan_lower)
    if native_mentions > 0:
        return min(1.0, 0.3 + 0.2 * native_mentions)

    # Check for VP marker references (indirect capability usage)
    vp_in_plan = bool(re.search(r"<\|box\|>|<\|point\|>|<\|ref\|>", plan_lower))
    if vp_in_plan:
        base_score = 0.3
    else:
        base_score = 0.0

    # Check for action verbs and planning language suggesting a viable plan
    plan_verbs = ["scan", "detect", "count", "identify", "locate", "extract",
                  "verify", "check", "ground", "find", "analyze", "explore",
                  "probe", "search", "inspect", "classify", "measure",
                  "compare", "cross-reference", "recount", "backtrack"]
    plan_nouns = ["strategy", "approach", "plan", "step", "method", "pipeline",
                  "trial", "exploration", "dead-end", "passage", "route"]
    verb_count = sum(1 for v in plan_verbs if v in plan_lower)
    noun_count = sum(1 for n in plan_nouns if n in plan_lower)

    # Each verb contributes 0.15, each planning noun 0.1, VP markers +0.3
    score = base_score + verb_count * 0.15 + noun_count * 0.1
    return min(1.0, score)


def evaluate_tool_call_correctness(act_text: str) -> float:
    """Evaluate whether ACT uses appropriate Florence-2 native prompts.

    Returns score in [0, 1]. Full score if at least one native prompt is used.
    Also recognizes VP marker formats (``<|box|>``, ``<|point|>``, ``<|ref|>``).
    """
    if not act_text.strip():
        return 0.0

    native_found = [p for p in _NATIVE_PROMPTS if p in act_text]

    if not native_found:
        # Check for VP coordinate markers as alternative
        vp_markers = [
            r"<\|box\|>.*?\[.*?\d+.*?\]",
            r"<\|point\|>.*?\[.*?\d+.*?\]",
            r"<\|ref\|>.*?<\|/ref\|>",
            r"\[\d+,\s*\d+",
        ]
        has_vp = any(re.search(p, act_text, re.DOTALL) for p in vp_markers)
        if has_vp:
            return 0.75  # VP markers are a valid tool-call representation
        return 0.0

    return min(1.0, 0.5 + 0.25 * len(native_found))


def evaluate_error_recovery(
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> float:
    """Evaluate self-correction behavior in error-injected trajectories.

    Checks if:
      1. An error was detected in VERIFY
      2. A REFLECT phase acknowledges the error
      3. The DECIDE phase contains a corrected answer

    Returns score in [0, 1].
    """
    phases = extract_all_phases(text)

    verify_contents = phases.get("verify", [])
    reflect_contents = phases.get("reflect", [])
    act_contents = phases.get("act", [])
    decide_contents = phases.get("decide", [])

    score = 0.0

    # 1. Error detection in VERIFY
    verify_text = " ".join(verify_contents).lower()
    error_detected = any(
        kw in verify_text
        for kw in ["wrong", "error", "mistake", "incorrect", "not", "but", "wait",
                    "missing", "missed", "recount", "re-check", "recheck",
                    "doesn", "isn", "wasn", "aren", "weren",  # contractions
                    "no valid", "not match", "doesn't match",
                    "actually", "correct location", "instead of"]
    )
    if error_detected:
        score += 0.3

    # 2. Error acknowledgment in REFLECT
    reflect_text = " ".join(reflect_contents).lower()
    error_acknowledged = bool(reflect_contents) and any(
        kw in reflect_text
        for kw in ["miss", "wrong", "error", "mistake", "confus", "incorrect",
                    "forgot", "skip", "overlook", "premature", "should",
                    "grabbed", "first", "cross-check", "instead"]
    )
    if error_acknowledged:
        score += 0.3

    # 3. Correction in DECIDE (differs from initial ACT answer)
    act_text = " ".join(act_contents).strip()
    decide_text = " ".join(decide_contents).strip()
    if act_text and decide_text and act_text != decide_text:
        score += 0.2

    # 4. If metadata says error was injected, check full recovery
    if metadata and metadata.get("error_injected"):
        if error_detected and error_acknowledged:
            score += 0.2
        elif not error_detected:
            # Failed to detect an injected error
            score = max(0.0, score - 0.1)

    return min(1.0, score)


def evaluate_consistency(text: str, num_rounds: int = 1) -> float:
    """Evaluate consistency across multi-round agentic chains.

    Checks:
      - Object references are consistent across rounds
      - No contradictory spatial claims
      - Coordinate values don't wildly fluctuate

    Returns score in [0, 1].
    """
    if num_rounds <= 1:
        return 1.0  # Single round is trivially consistent

    phases = extract_all_phases(text)
    all_act_texts = phases.get("act", [])

    if len(all_act_texts) < 2:
        return 1.0

    score = 1.0

    # Check for contradictory spatial terms across rounds
    spatial_terms = {
        "left": "right", "right": "left",
        "up": "down", "down": "up",
        "above": "below", "below": "above",
    }

    for i in range(len(all_act_texts) - 1):
        act_i = all_act_texts[i].lower()
        act_j = all_act_texts[i + 1].lower()

        for term, opposite in spatial_terms.items():
            if term in act_i and opposite in act_j:
                # Potential contradiction — penalize if both texts
                # mention coordinates (suggesting they refer to
                # the same visual scene) or share common nouns
                has_coords_i = bool(re.search(r"\[\d+", act_i))
                has_coords_j = bool(re.search(r"\[\d+", act_j))
                if (has_coords_i and has_coords_j) or _shares_object_reference(act_i, act_j):
                    score -= 0.2

    # Check coordinate consistency (no wild jumps)
    all_coords = []
    for act_text in all_act_texts:
        coords = re.findall(r"\[(\d+),\s*(\d+)", act_text)
        all_coords.extend([(int(x), int(y)) for x, y in coords])

    if len(all_coords) > 2:
        for i in range(len(all_coords) - 1):
            x1, y1 = all_coords[i]
            x2, y2 = all_coords[i + 1]
            dist = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            if dist > 800:  # Wild jump in [0, 999] space
                score -= 0.1

    return max(0.0, min(1.0, score))


def _shares_object_reference(text_a: str, text_b: str) -> bool:
    """Heuristic: check if two texts share common object references."""
    # Extract potential object names (nouns between VP markers)
    objects_a = set(re.findall(r"(\w+)\s*[:=]\s*\[", text_a.lower()))
    objects_b = set(re.findall(r"(\w+)\s*[:=]\s*\[", text_b.lower()))
    if objects_a & objects_b:
        return True
    # Also check for common significant nouns (objects like "box", "block", etc.)
    significant_nouns = {"box", "block", "object", "person", "car", "text",
                         "region", "item", "circle", "square", "triangle",
                         "red", "blue", "green", "yellow"}
    words_a = set(re.findall(r"\b(\w+)\b", text_a.lower()))
    words_b = set(re.findall(r"\b(\w+)\b", text_b.lower()))
    common_significant = (words_a & words_b) & significant_nouns
    return len(common_significant) >= 2


def evaluate_native_preservation(
    native_accuracy_before: float,
    native_accuracy_after: float,
) -> float:
    """Evaluate how well native Florence-2 capabilities are preserved after training.

    Returns the preservation ratio: 1.0 means no degradation, 0.0 means total loss.
    """
    if native_accuracy_before <= 0:
        return 1.0
    return max(0.0, min(1.0, native_accuracy_after / native_accuracy_before))


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

class AgenticEvaluator:
    """Comprehensive evaluator for Agentic meta-cognitive reasoning.

    Usage::

        evaluator = AgenticEvaluator()
        for pred_text, metadata in zip(predictions, ground_truths):
            evaluator.add_sample(pred_text, metadata)
        metrics = evaluator.compute()
        print(metrics.summary())
    """

    def __init__(self):
        self._predictions: List[str] = []
        self._metadata: List[Dict[str, Any]] = []
        self._native_before: Optional[float] = None
        self._native_after: Optional[float] = None

    def add_sample(
        self,
        prediction: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a single prediction for evaluation."""
        self._predictions.append(prediction)
        self._metadata.append(metadata or {})

    def add_batch(
        self,
        predictions: List[str],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add a batch of predictions."""
        if metadata_list is None:
            metadata_list = [{}] * len(predictions)
        for pred, meta in zip(predictions, metadata_list):
            self.add_sample(pred, meta)

    def set_native_accuracy(
        self,
        before: float,
        after: float,
    ) -> None:
        """Set native task accuracy before and after agentic training."""
        self._native_before = before
        self._native_after = after

    def compute(self) -> AgenticMetrics:
        """Compute all agentic metrics from accumulated predictions."""
        n = len(self._predictions)
        if n == 0:
            return AgenticMetrics()

        format_scores: List[float] = []
        plan_scores: List[float] = []
        tool_scores: List[float] = []
        recovery_scores_error: List[float] = []  # only error-injected samples
        consistency_scores: List[float] = []
        error_injected_count = 0

        for pred, meta in zip(self._predictions, self._metadata):
            # Format validity
            is_valid, _ = evaluate_format_validity(pred)
            format_scores.append(1.0 if is_valid else 0.0)

            # Planning accuracy
            plan_contents = extract_phase(pred, "plan")
            plan_text = " ".join(plan_contents) if plan_contents else ""
            expected_steps = meta.get("expected_steps")
            plan_scores.append(evaluate_planning_accuracy(plan_text, expected_steps))

            # Tool-call correctness
            act_contents = extract_phase(pred, "act")
            act_text = " ".join(act_contents) if act_contents else ""
            tool_scores.append(evaluate_tool_call_correctness(act_text))

            # Error recovery (only scored for error-injected samples)
            if meta.get("error_injected"):
                error_injected_count += 1
                recovery_scores_error.append(evaluate_error_recovery(pred, meta))

            # Consistency
            num_rounds = meta.get("num_rounds", 1)
            consistency_scores.append(evaluate_consistency(pred, num_rounds))

        # Native preservation
        native_pres = 1.0
        if self._native_before is not None and self._native_after is not None:
            native_pres = evaluate_native_preservation(
                self._native_before, self._native_after
            )

        return AgenticMetrics(
            format_validity=sum(format_scores) / n,
            planning_accuracy=sum(plan_scores) / n,
            tool_call_correctness=sum(tool_scores) / n,
            error_recovery_rate=(
                sum(recovery_scores_error) / max(len(recovery_scores_error), 1)
                if error_injected_count > 0
                else 1.0
            ),
            consistency_score=sum(consistency_scores) / n,
            native_capability_preservation=native_pres,
            total_samples=n,
            error_injected_samples=error_injected_count,
        )
