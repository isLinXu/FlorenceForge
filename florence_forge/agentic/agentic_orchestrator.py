"""Agentic orchestrator: the outer agent loop over Florence-2 native tools.

This module realizes the ``<DECOMPOSE>`` / ``<NEXT_ACTION>`` semantics from the
original design proposal as **inference-time Python control flow**, rather than
in-model tokens — which is the architecturally correct place for long-horizon
multi-step state management (Florence-2's single forward pass cannot maintain a
long agentic context, but a Python loop can).

Flow::

    goal ──<DECOMPOSE>──▶ [SubTask, SubTask, ...]
              │
              ▼  for each sub-task:
        <NEXT_ACTION> ─▶ ToolCall (native Florence-2 task)
              │              │
              │              ▼ backend.predict_task(...)
              │         raw text output
              ▼              │
        <VERIFY> ◀───────────┘
              │
       ok? ───┴── no ──▶ <REFLECT> ──▶ retry (≤ max_retries)
              │ yes
              ▼
        update AgentState (+ optional <SUMMARIZE_STATE>)
              │
              ▼
     all sub-tasks done ──▶ <DONE> + aggregated final answer

The orchestrator emits a full meta-cognitive transcript (wrapped with the same
``core.agentic_tokens`` delimiters used for training data), so the trace can be
fed straight into ``AgenticEvaluator`` or harvested as SFT trajectories — a
free data-generation byproduct.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, runtime_checkable

from ..core.agentic_tokens import wrap_phase
from ..core.visual_primitives import parse_vp_boxes
from .tool_registry import (
    ToolSpec,
    get_tool_spec,
    list_tools,
    select_tool_for_intent,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class ToolBackend(Protocol):
    """Minimal protocol the orchestrator needs from a model backend.

    ``Florence2MultiTaskModel`` already satisfies this via its ``predict_task``.
    Keeping it a Protocol means the orchestrator never imports torch and can be
    exercised with a deterministic mock in tests.
    """

    def predict_task(
        self,
        images: Any,
        task_name: str,
        text_input: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:  # returns str (or list[str] for batched; we use single image)
        ...


# ---------------------------------------------------------------------------
# State & record dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    """A single resolved tool invocation."""

    intent: str
    task_name: str
    text_input: Optional[str] = None

    def describe(self) -> str:
        arg = f" arg={self.text_input!r}" if self.text_input else ""
        return f"{self.task_name}(<{self.intent}>{arg})"


@dataclass
class StepRecord:
    """Outcome of one sub-task execution (possibly with retries)."""

    sub_task_index: int
    intent: str
    tool_call: ToolCall
    raw_output: str = ""
    parsed: Dict[str, Any] = field(default_factory=dict)
    verified: bool = False
    issues: List[str] = field(default_factory=list)
    attempts: int = 1
    transcript: str = ""  # agentic-token-wrapped trace for this sub-task


@dataclass
class SubTask:
    """One decomposed unit of work."""

    index: int
    goal: str
    intent: str
    text_input: Optional[str] = None


@dataclass
class PlanResult:
    """Result of decomposing a high-level goal."""

    sub_tasks: List[SubTask]
    rationale: str = ""

    def __len__(self) -> int:
        return len(self.sub_tasks)


@dataclass
class AgentState:
    """Cross-step accumulated observations.

    This is the ``<SUMMARIZE_STATE>`` target — the compact memory the agent
    carries across sub-tasks for a long-horizon visual task.
    """

    detected_objects: List[Dict[str, Any]] = field(default_factory=list)
    extracted_text: List[str] = field(default_factory=list)
    located_regions: List[Dict[str, Any]] = field(default_factory=list)
    counts: Dict[str, int] = field(default_factory=dict)
    descriptions: List[str] = field(default_factory=list)
    pending_issues: List[str] = field(default_factory=list)

    def summarize(self) -> str:
        """Produce a compact textual state summary."""
        parts: List[str] = []
        if self.detected_objects:
            parts.append(f"{len(self.detected_objects)} detected boxes")
        if self.extracted_text:
            parts.append(f"{len(self.extracted_text)} text segments")
        if self.located_regions:
            parts.append(f"{len(self.located_regions)} located regions")
        if self.counts:
            parts.append("counts={" + ", ".join(
                f"{k}:{v}" for k, v in self.counts.items()) + "}")
        if self.descriptions:
            parts.append(f"{len(self.descriptions)} descriptions")
        if self.pending_issues:
            parts.append(f"{len(self.pending_issues)} pending issues")
        return "; ".join(parts) if parts else "no observations yet"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detected_objects": self.detected_objects,
            "extracted_text": self.extracted_text,
            "located_regions": self.located_regions,
            "counts": self.counts,
            "descriptions": self.descriptions,
            "pending_issues": self.pending_issues,
        }


@dataclass
class OrchestratorConfig:
    """Tunable knobs for the orchestrator."""

    max_steps: int = 12
    max_retries: int = 1          # extra attempts per sub-task on verify failure
    summarize_every: int = 3      # emit <SUMMARIZE_STATE> every N completed steps
    min_boxes_expected: int = 0   # if >0, detect/locate must return >= this many
    emit_transcript: bool = True  # build agentic-token-wrapped trace
    max_total_seconds: Optional[float] = None  # overall wall-clock budget for a run
    backend_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestratorResult:
    """Full result of an orchestration run."""

    goal: str
    plan: PlanResult
    steps: List[StepRecord]
    state: AgentState
    final_answer: str
    transcript: str
    success: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "plan": [
                {"index": s.index, "goal": s.goal, "intent": s.intent,
                 "text_input": s.text_input}
                for s in self.plan.sub_tasks
            ],
            "steps": [
                {
                    "sub_task_index": r.sub_task_index,
                    "intent": r.intent,
                    "tool_call": r.tool_call.describe(),
                    "verified": r.verified,
                    "attempts": r.attempts,
                    "issues": r.issues,
                    "raw_output": r.raw_output,
                }
                for r in self.steps
            ],
            "state": self.state.to_dict(),
            "final_answer": self.final_answer,
            "success": self.success,
        }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class AgenticOrchestrator:
    """Drive a Florence-2 backend through a multi-step visual task.

    Args:
        backend: Any object satisfying :class:`ToolBackend` (e.g.
            ``Florence2MultiTaskModel``).
        config: Optional :class:`OrchestratorConfig`.
    """

    def __init__(
        self,
        backend: ToolBackend,
        config: Optional[OrchestratorConfig] = None,
    ):
        if not hasattr(backend, "predict_task"):
            raise TypeError(
                "backend must expose predict_task(images, task_name, "
                "text_input=None, **kwargs); got "
                f"{type(backend).__name__}"
            )
        self.backend = backend
        self.config = config or OrchestratorConfig()

    def _budget_exceeded(self, start_time: Optional[float]) -> bool:
        """Return True if the overall time budget (``max_total_seconds``) elapsed."""
        budget = self.config.max_total_seconds
        if not budget or start_time is None:
            return False
        return (time.monotonic() - start_time) > budget

    # -- public entrypoint --------------------------------------------------

    def run(
        self,
        image: Any,
        goal: str,
        plan: Optional[Sequence[SubTask]] = None,
    ) -> OrchestratorResult:
        """Execute *goal* over *image* and return a full transcript + state.

        Args:
            image: A PIL image (or whatever the backend's predict_task accepts).
            goal: High-level natural-language objective.
            plan: Optional pre-built sub-task list; if omitted, the goal is
                decomposed heuristically via :meth:`decompose`.
        """
        if plan is None:
            plan_result = self.decompose(goal)
        else:
            plan_result = PlanResult(sub_tasks=list(plan))

        state = AgentState()
        steps: List[StepRecord] = []
        transcript_parts: List[str] = []

        if self.config.emit_transcript:
            transcript_parts.append(
                wrap_phase("plan", f"Goal: {goal}. {plan_result.rationale}")
            )
            transcript_parts.append(
                wrap_phase(
                    "summarize_state",
                    "Decomposed into "
                    + " -> ".join(f"{st.intent}" for st in plan_result.sub_tasks),
                )
            )

        start_time = time.monotonic()
        completed = 0
        for sub in plan_result.sub_tasks:
            if len(steps) >= self.config.max_steps:
                logger.warning("Reached max_steps=%d, stopping.", self.config.max_steps)
                break
            if self._budget_exceeded(start_time):
                logger.warning(
                    "Reached max_total_seconds=%.1fs, stopping with %d/%d sub-tasks done.",
                    self.config.max_total_seconds, len(steps), len(plan_result.sub_tasks),
                )
                break

            record = self._execute_sub_task(image, sub, state)
            steps.append(record)
            if self.config.emit_transcript and record.transcript:
                transcript_parts.append(record.transcript)

            completed += 1
            if (self.config.summarize_every > 0
                    and completed % self.config.summarize_every == 0):
                summary = state.summarize()
                if self.config.emit_transcript:
                    transcript_parts.append(wrap_phase("summarize_state", summary))

        final_answer = self._aggregate(goal, state, steps)
        success = all(r.verified for r in steps) and len(steps) > 0
        if self.config.emit_transcript:
            transcript_parts.append(wrap_phase("decide", final_answer))
            transcript_parts.append(wrap_phase("done", "Task completed."))

        return OrchestratorResult(
            goal=goal,
            plan=plan_result,
            steps=steps,
            state=state,
            final_answer=final_answer,
            transcript="".join(transcript_parts),
            success=success,
        )

    # -- public streaming entrypoint ---------------------------------------

    @staticmethod
    def step_to_dict(record: StepRecord) -> Dict[str, Any]:
        """Serialize a :class:`StepRecord` to a JSON-friendly dict.

        Public helper so streaming consumers (e.g. the FastAPI SSE endpoint)
        can serialize step events without reaching into orchestrator internals.
        """
        return {
            "sub_task_index": record.sub_task_index,
            "intent": record.intent,
            "tool_call": record.tool_call.describe(),
            "raw_output": record.raw_output,
            "verified": record.verified,
            "attempts": record.attempts,
            "issues": record.issues,
        }

    def run_stream(
        self,
        image: Any,
        goal: str,
        plan: Optional[Sequence[SubTask]] = None,
    ):
        """Execute *goal* over *image*, yielding events incrementally.

        Public streaming variant of :meth:`run`. Yields plain dicts with a
        ``"type"`` discriminator so callers never touch private methods:

        * ``{"type": "plan", ...}`` — emitted once after decomposition.
        * ``{"type": "step", "step": {...}, "record": StepRecord}`` — one per
          executed sub-task. ``record`` is the live dataclass for callers that
          need richer data (e.g. to render a visualization); ``step`` is the
          JSON-friendly serialization.
        * ``{"type": "done", ...}`` — emitted once at the end.
        """
        if plan is None:
            plan_result = self.decompose(goal)
        else:
            plan_result = PlanResult(sub_tasks=list(plan))

        yield {
            "type": "plan",
            "goal": goal,
            "sub_tasks": [
                {"index": s.index, "intent": s.intent, "goal": s.goal}
                for s in plan_result.sub_tasks
            ],
            "rationale": plan_result.rationale,
        }

        state = AgentState()
        steps: List[StepRecord] = []
        start_time = time.monotonic()
        for sub in plan_result.sub_tasks:
            if len(steps) >= self.config.max_steps:
                logger.warning("Reached max_steps=%d, stopping.", self.config.max_steps)
                break
            if self._budget_exceeded(start_time):
                logger.warning(
                    "Reached max_total_seconds=%.1fs, stopping with %d/%d sub-tasks done.",
                    self.config.max_total_seconds, len(steps), len(plan_result.sub_tasks),
                )
                break
            record = self._execute_sub_task(image, sub, state)
            steps.append(record)
            yield {
                "type": "step",
                "step": self.step_to_dict(record),
                "record": record,
            }

        final_answer = self._aggregate(goal, state, steps)
        success = all(r.verified for r in steps) and len(steps) > 0
        yield {
            "type": "done",
            "final_answer": final_answer,
            "state": state.summarize(),
            "state_detail": state.to_dict(),
            "success": success,
        }

    # -- <DECOMPOSE> --------------------------------------------------------

    def decompose(self, goal: str) -> PlanResult:
        """Split *goal* into ordered sub-tasks (the ``<DECOMPOSE>`` step).

        Heuristic, dependency-free decomposition: scan the goal for tool
        keywords (via the registry) and order the matched intents in a sensible
        perception→reasoning sequence. Falls back to a generic detect→describe
        plan when nothing specific matches.

        For production use, this method is the natural seam to swap in an LLM
        planner; the rest of the loop is agnostic to how the plan was produced.
        """
        goal_l = (goal or "").lower()
        matched: List[ToolSpec] = []
        seen_intents = set()

        # Greedy keyword scan across all tools, preserving registry order so the
        # plan is deterministic.
        for spec in list_tools():
            if spec.intent in seen_intents:
                continue
            if any(kw.lower() in goal_l for kw in spec.keywords):
                matched.append(spec)
                seen_intents.add(spec.intent)

        if not matched:
            # Generic fallback: detect then describe.
            matched = [get_tool_spec("detect"), get_tool_spec("describe")]

        # Canonical ordering: detection/location first, then reading, then
        # counting/description (perception → extraction → reasoning).
        order_priority = {
            "detect": 0, "open_detect": 0, "locate": 1, "region_describe": 1,
            "read_text": 2, "read_text_plain": 2, "region_category": 3,
            "count": 4, "describe": 5,
        }
        matched.sort(key=lambda s: order_priority.get(s.intent, 9))

        sub_tasks: List[SubTask] = []
        for i, spec in enumerate(matched):
            text_input = self._infer_text_input(goal, spec) if spec.needs_text_input else None
            sub_tasks.append(SubTask(
                index=i,
                goal=f"{spec.description}",
                intent=spec.intent,
                text_input=text_input,
            ))

        rationale = (
            f"Identified {len(sub_tasks)} sub-task(s): "
            + " -> ".join(s.intent for s in sub_tasks)
        )
        return PlanResult(sub_tasks=sub_tasks, rationale=rationale)

    @staticmethod
    def _infer_text_input(goal: str, spec: ToolSpec) -> Optional[str]:
        """Best-effort extraction of the text argument for text-input tools.

        For ``count``/``locate``/``open_detect`` we try to pull the object noun
        from the goal; this is intentionally simple and can be replaced by an
        LLM. Returns ``None`` when nothing reasonable is found (the backend may
        still handle a bare prompt).
        """
        import re
        goal_l = goal.lower()
        # "count the <X>" / "how many <X>" / "locate the <X>" / "find the <X>"
        patterns = [
            r"(?:count|how many|number of)\s+(?:the\s+)?([a-z][a-z\s]{1,30}?)(?:\s+(?:in|on|are|is|\?)|$)",
            r"(?:locate|find|where is)\s+(?:the\s+)?([a-z][a-z\s]{1,30}?)(?:\s+(?:in|on|\?)|$)",
            r"(?:detect|ground)\s+(?:the\s+)?([a-z][a-z\s]{1,30}?)(?:\s+(?:in|on|\?)|$)",
        ]
        for pat in patterns:
            m = re.search(pat, goal_l)
            if m:
                return m.group(1).strip()
        return None

    # -- <NEXT_ACTION> ------------------------------------------------------

    def next_action(self, sub_task: SubTask) -> ToolCall:
        """Resolve a sub-task into a concrete tool call (``<NEXT_ACTION>``)."""
        try:
            spec = get_tool_spec(sub_task.intent)
        except KeyError:
            # Fall back to keyword matching on the sub-task goal text.
            spec = select_tool_for_intent(sub_task.goal) or get_tool_spec("detect")
        return ToolCall(
            intent=spec.intent,
            task_name=spec.task_name,
            text_input=sub_task.text_input,
        )

    # -- single sub-task execution -----------------------------------------

    def _execute_sub_task(
        self,
        image: Any,
        sub: SubTask,
        state: AgentState,
    ) -> StepRecord:
        tool_call = self.next_action(sub)
        spec = get_tool_spec(tool_call.intent)
        attempts = 0
        raw = ""
        parsed: Dict[str, Any] = {}
        verified = False
        issues: List[str] = []
        local_parts: List[str] = []

        max_attempts = 1 + max(0, self.config.max_retries)
        while attempts < max_attempts:
            attempts += 1
            raw = self._invoke_tool(image, tool_call)
            parsed = self._parse_output(spec, raw)
            verified, issues = self._verify(spec, parsed, raw)

            if self.config.emit_transcript:
                local_parts.append(wrap_phase(
                    "act",
                    f"[sub {sub.index}] {tool_call.describe()} -> {raw[:120]}",
                ))
                local_parts.append(wrap_phase(
                    "verify",
                    ("OK: " + spec.intent) if verified
                    else ("issues: " + "; ".join(issues)),
                ))

            if verified:
                break

            # <REFLECT> + retry strategy
            if attempts < max_attempts:
                strategy = self._reflect_strategy(spec, issues)
                if self.config.emit_transcript:
                    local_parts.append(wrap_phase(
                        "reflect",
                        f"Attempt {attempts} failed ({'; '.join(issues)}). "
                        f"Strategy: {strategy}",
                    ))

        if verified:
            self._commit_to_state(spec, parsed, state)
        else:
            state.pending_issues.extend(issues)

        return StepRecord(
            sub_task_index=sub.index,
            intent=sub.intent,
            tool_call=tool_call,
            raw_output=raw,
            parsed=parsed,
            verified=verified,
            issues=issues,
            attempts=attempts,
            transcript="".join(local_parts),
        )

    def _invoke_tool(self, image: Any, call: ToolCall) -> str:
        """Call the backend's native task; normalize output to a string."""
        try:
            out = self.backend.predict_task(
                images=image,
                task_name=call.task_name,
                text_input=call.text_input,
                **self.config.backend_kwargs,
            )
        except Exception as exc:  # noqa: BLE001 — surface as an issue, don't crash the loop
            logger.warning("Tool %s raised: %s", call.task_name, exc)
            return f"__ERROR__: {exc}"
        if isinstance(out, list):
            out = out[0] if out else ""
        return out if isinstance(out, str) else str(out)

    # -- output parsing -----------------------------------------------------

    @staticmethod
    def _parse_output(spec: ToolSpec, raw: str) -> Dict[str, Any]:
        """Parse a tool's raw text into structured fields per ``output_kind``."""
        if raw.startswith("__ERROR__"):
            return {"error": raw}

        kind = spec.output_kind
        if kind in ("boxes", "regions"):
            boxes = parse_vp_boxes(raw)
            return {"boxes": boxes, "n_boxes": len(boxes), "text": raw}
        if kind == "count":
            import re
            nums = re.findall(r"\d+", raw)
            count = int(nums[0]) if nums else 0
            boxes = parse_vp_boxes(raw)
            return {"count": count, "boxes": boxes, "text": raw}
        # plain text
        return {"text": raw.strip()}

    # -- <VERIFY> -----------------------------------------------------------

    def _verify(
        self,
        spec: ToolSpec,
        parsed: Dict[str, Any],
        raw: str,
    ) -> tuple[bool, List[str]]:
        """Validate parsed tool output. Returns (ok, issues)."""
        issues: List[str] = []
        if "error" in parsed:
            issues.append(f"backend error: {parsed['error']}")
            return False, issues

        kind = spec.output_kind
        if kind in ("boxes", "regions"):
            n = parsed.get("n_boxes", 0)
            if n == 0:
                issues.append(f"{spec.intent} returned no boxes")
            elif self.config.min_boxes_expected and n < self.config.min_boxes_expected:
                issues.append(
                    f"{spec.intent} returned {n} boxes "
                    f"(< expected {self.config.min_boxes_expected})"
                )
        elif kind == "count":
            if parsed.get("count", 0) <= 0:
                issues.append("count is zero or unparseable")
        else:  # text
            if not parsed.get("text"):
                issues.append(f"{spec.intent} returned empty text")

        return len(issues) == 0, issues

    # -- <REFLECT> ----------------------------------------------------------

    @staticmethod
    def _reflect_strategy(spec: ToolSpec, issues: List[str]) -> str:
        """Pick a corrective strategy given verification issues."""
        joined = " ".join(issues).lower()
        if "no boxes" in joined or "returned 0" in joined:
            return ("retry with denser detection "
                    "(<DENSE_REGION_CAPTION>) for finer granularity")
        if "count is zero" in joined:
            return "re-scan and recount after grounding each instance"
        if "empty text" in joined:
            return "retry OCR at higher region resolution"
        if "backend error" in joined:
            return "retry the same tool once (transient failure)"
        return "retry the same tool with adjusted inputs"

    # -- state commit -------------------------------------------------------

    @staticmethod
    def _commit_to_state(
        spec: ToolSpec,
        parsed: Dict[str, Any],
        state: AgentState,
    ) -> None:
        kind = spec.output_kind
        if kind == "boxes":
            for box in parsed.get("boxes", []):
                state.detected_objects.append({"box": box, "via": spec.intent})
        elif kind == "regions":
            for box in parsed.get("boxes", []):
                state.located_regions.append({"box": box, "via": spec.intent})
            if parsed.get("text"):
                state.extracted_text.append(parsed["text"])
        elif kind == "count":
            label = spec.intent
            state.counts[label] = parsed.get("count", 0)
        else:  # text
            txt = parsed.get("text", "")
            if txt:
                if spec.intent in ("read_text", "read_text_plain"):
                    state.extracted_text.append(txt)
                else:
                    state.descriptions.append(txt)

    # -- aggregation --------------------------------------------------------

    @staticmethod
    def _aggregate(goal: str, state: AgentState, steps: List[StepRecord]) -> str:
        """Compose a final natural-language answer from accumulated state."""
        bits: List[str] = []
        if state.detected_objects:
            bits.append(f"detected {len(state.detected_objects)} object(s)")
        if state.counts:
            bits.append("counts: " + ", ".join(
                f"{k}={v}" for k, v in state.counts.items()))
        if state.located_regions:
            bits.append(f"localized {len(state.located_regions)} region(s)")
        if state.extracted_text:
            joined = " | ".join(t[:60] for t in state.extracted_text[:5])
            bits.append(f"text: {joined}")
        if state.descriptions:
            bits.append(f"description: {state.descriptions[0][:120]}")
        if state.pending_issues:
            bits.append(f"unresolved: {len(state.pending_issues)} issue(s)")
        if not bits:
            return "No observations could be extracted for the goal."
        return f"For goal '{goal}': " + "; ".join(bits) + "."
