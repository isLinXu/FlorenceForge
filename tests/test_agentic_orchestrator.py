"""Unit tests for the outer-layer AgenticOrchestrator.

These tests exercise the inference-time orchestrator (``florence_forge.agentic``)
with a deterministic mock backend — no torch, no model weights, no PIL images.
The mock returns canned outputs keyed by task name, so we can assert on the full
DECOMPOSE → NEXT_ACTION → VERIFY → REFLECT → DECIDE flow deterministically.

Coverage areas:
  1. Tool registry: intent→task mapping, keyword selection, validation
  2. Decompose: heuristic planning from natural-language goals
  3. Single-tool execution: invoke→parse→verify→commit
  4. Retry on verify failure (REFLECT strategy)
  5. Multi-step orchestration with state accumulation
  6. Transcript emission (agentic token wrapping)
  7. Aggregation / final answer
  8. Edge cases: empty output, backend error, max_steps cap
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# Mock backend
# ---------------------------------------------------------------------------

class MockBackend:
    """Deterministic mock satisfying the ToolBackend protocol.

    Returns canned strings per task_name. Supports injectable errors and
    a call log for asserting on orchestration behavior.
    """

    def __init__(
        self,
        responses: Optional[Dict[str, str]] = None,
        error_tasks: Optional[set] = None,
    ):
        self._responses = responses or {}
        self._error_tasks = error_tasks or set()
        self.calls: List[Dict[str, Any]] = []

    def predict_task(
        self,
        images: Any,
        task_name: str,
        text_input: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        self.calls.append({
            "task_name": task_name,
            "text_input": text_input,
            "images": images,
        })
        if task_name in self._error_tasks:
            raise RuntimeError(f"mock error for {task_name}")
        return self._responses.get(task_name, "")


def _box_payload(boxes: List[List[int]]) -> str:
    """Format a VP box payload as Florence-2 would emit it."""
    from florence_forge.core.visual_primitives import format_box
    return format_box(boxes)


# ---------------------------------------------------------------------------
# 1. Tool registry
# ---------------------------------------------------------------------------

class TestToolRegistry:
    """Test the intent→task mapping and keyword-based selection."""

    def test_all_tools_validated_at_import(self):
        """Every ToolSpec.task_name must exist in FLORENCE2_TASKS."""
        from florence_forge.agentic.tool_registry import TOOL_REGISTRY
        from florence_forge.core.tasks import FLORENCE2_TASKS

        for intent, spec in TOOL_REGISTRY.items():
            assert spec.task_name in FLORENCE2_TASKS, (
                f"intent {intent!r} references unknown task {spec.task_name!r}"
            )

    def test_tool_spec_prompt_property(self):
        from florence_forge.agentic.tool_registry import get_tool_spec
        from florence_forge.core.tasks import FLORENCE2_TASKS

        spec = get_tool_spec("detect")
        assert spec.prompt == FLORENCE2_TASKS["OD"].prompt

    def test_get_tool_spec_unknown_raises(self):
        from florence_forge.agentic.tool_registry import get_tool_spec

        with pytest.raises(KeyError, match="Unknown tool intent"):
            get_tool_spec("nonexistent_intent")

    def test_list_tools_count(self):
        from florence_forge.agentic.tool_registry import list_tools

        tools = list_tools()
        assert len(tools) >= 9

    def test_select_tool_for_intent_detect(self):
        from florence_forge.agentic.tool_registry import select_tool_for_intent

        spec = select_tool_for_intent("detect all objects in the image")
        assert spec is not None
        assert spec.intent == "detect"

    def test_select_tool_for_intent_ocr(self):
        from florence_forge.agentic.tool_registry import select_tool_for_intent

        spec = select_tool_for_intent("read all text in the image")
        assert spec is not None
        assert spec.intent in ("read_text", "read_text_plain")

    def test_select_tool_for_intent_count(self):
        from florence_forge.agentic.tool_registry import select_tool_for_intent

        spec = select_tool_for_intent("count the number of cars")
        assert spec is not None
        assert spec.intent == "count"

    def test_select_tool_for_intent_empty(self):
        from florence_forge.agentic.tool_registry import select_tool_for_intent

        assert select_tool_for_intent("") is None
        assert select_tool_for_intent("   ") is None

    def test_select_tool_for_intent_no_match(self):
        from florence_forge.agentic.tool_registry import select_tool_for_intent

        assert select_tool_for_intent("xyzzy frobnicate") is None

    def test_select_tool_chinese_keywords(self):
        from florence_forge.agentic.tool_registry import select_tool_for_intent

        spec = select_tool_for_intent("检测所有目标")
        assert spec is not None
        assert spec.intent == "detect"


# ---------------------------------------------------------------------------
# 2. Decompose
# ---------------------------------------------------------------------------

class TestDecompose:
    """Test the heuristic <DECOMPOSE> planner."""

    def test_decompose_detect_and_count(self):
        from florence_forge.agentic import AgenticOrchestrator

        orch = AgenticOrchestrator(MockBackend())
        plan = orch.decompose("detect all objects and count the cars")

        intents = [s.intent for s in plan.sub_tasks]
        assert "detect" in intents
        assert "count" in intents
        # detect should come before count (perception → reasoning order)
        assert intents.index("detect") < intents.index("count")

    def test_decompose_fallback_detect_describe(self):
        from florence_forge.agentic import AgenticOrchestrator

        orch = AgenticOrchestrator(MockBackend())
        plan = orch.decompose("frobnicate the widget")

        intents = [s.intent for s in plan.sub_tasks]
        assert intents == ["detect", "describe"]

    def test_decompose_ocr_and_describe(self):
        from florence_forge.agentic import AgenticOrchestrator

        orch = AgenticOrchestrator(MockBackend())
        plan = orch.decompose("read text and describe the scene")

        intents = [s.intent for s in plan.sub_tasks]
        assert "read_text" in intents
        assert "describe" in intents

    def test_decompose_rationale_nonempty(self):
        from florence_forge.agentic import AgenticOrchestrator

        orch = AgenticOrchestrator(MockBackend())
        plan = orch.decompose("detect objects")
        assert plan.rationale
        assert len(plan) > 0

    def test_decompose_count_infers_text_input(self):
        from florence_forge.agentic import AgenticOrchestrator

        orch = AgenticOrchestrator(MockBackend())
        plan = orch.decompose("count the cars in the image")
        count_sub = next(s for s in plan.sub_tasks if s.intent == "count")
        assert count_sub.text_input is not None
        assert "car" in count_sub.text_input.lower()

    def test_decompose_locate_infers_text_input(self):
        from florence_forge.agentic import AgenticOrchestrator

        orch = AgenticOrchestrator(MockBackend())
        plan = orch.decompose("locate the red box")
        locate_sub = next(s for s in plan.sub_tasks if s.intent == "locate")
        assert locate_sub.text_input is not None
        assert "red box" in locate_sub.text_input.lower()


# ---------------------------------------------------------------------------
# 3. Single sub-task execution
# ---------------------------------------------------------------------------

class TestSingleSubTask:
    """Test the execute → verify → commit pipeline for one sub-task."""

    def test_detect_returns_boxes(self):
        from florence_forge.agentic import (
            AgenticOrchestrator, SubTask, OrchestratorConfig, AgentState,
        )

        backend = MockBackend(responses={
            "OD": _box_payload([[10, 20, 50, 60], [70, 80, 100, 110]]),
        })
        orch = AgenticOrchestrator(backend, OrchestratorConfig(emit_transcript=False))
        state = AgentState()
        sub = SubTask(index=0, goal="detect", intent="detect")
        record = orch._execute_sub_task(image=None, sub=sub, state=state)

        assert record.verified is True
        assert record.attempts == 1
        assert len(state.detected_objects) == 2

    def test_ocr_returns_text(self):
        from florence_forge.agentic import (
            AgenticOrchestrator, SubTask, AgentState, OrchestratorConfig,
        )

        backend = MockBackend(responses={
            "OCR": "Hello World 123",
        })
        orch = AgenticOrchestrator(backend, OrchestratorConfig(emit_transcript=False))
        state = AgentState()
        sub = SubTask(index=0, goal="read text", intent="read_text_plain")
        record = orch._execute_sub_task(image=None, sub=sub, state=state)

        assert record.verified is True
        assert len(state.extracted_text) == 1

    def test_empty_output_not_verified(self):
        from florence_forge.agentic import (
            AgenticOrchestrator, SubTask, AgentState, OrchestratorConfig,
        )

        backend = MockBackend(responses={"OD": ""})
        orch = AgenticOrchestrator(backend, OrchestratorConfig(emit_transcript=False, max_retries=0))
        state = AgentState()
        sub = SubTask(index=0, goal="detect", intent="detect")
        record = orch._execute_sub_task(image=None, sub=sub, state=state)

        assert record.verified is False
        assert len(record.issues) > 0
        assert len(state.pending_issues) > 0

    def test_backend_error_not_verified(self):
        from florence_forge.agentic import (
            AgenticOrchestrator, SubTask, AgentState, OrchestratorConfig,
        )

        backend = MockBackend(error_tasks={"OD"})
        orch = AgenticOrchestrator(backend, OrchestratorConfig(emit_transcript=False, max_retries=0))
        state = AgentState()
        sub = SubTask(index=0, goal="detect", intent="detect")
        record = orch._execute_sub_task(image=None, sub=sub, state=state)

        assert record.verified is False
        assert any("error" in i.lower() for i in record.issues)


# ---------------------------------------------------------------------------
# 4. Retry / REFLECT
# ---------------------------------------------------------------------------

class TestRetryReflect:
    """Test that the orchestrator retries on verify failure."""

    def test_retry_succeeds_on_second_attempt(self):
        """If first call returns empty but second returns boxes, verify passes."""
        from florence_forge.agentic import (
            AgenticOrchestrator, SubTask, AgentState, OrchestratorConfig,
        )

        # First call returns empty, second returns boxes
        call_count = [0]

        class RetryBackend:
            def predict_task(self, images, task_name, text_input=None, **kwargs):
                call_count[0] += 1
                if call_count[0] == 1:
                    return ""
                return _box_payload([[10, 20, 50, 60]])

        orch = AgenticOrchestrator(
            RetryBackend(),
            OrchestratorConfig(emit_transcript=False, max_retries=1),
        )
        state = AgentState()
        sub = SubTask(index=0, goal="detect", intent="detect")
        record = orch._execute_sub_task(image=None, sub=sub, state=state)

        assert record.verified is True
        assert record.attempts == 2
        assert len(state.detected_objects) == 1

    def test_no_retry_when_max_retries_zero(self):
        from florence_forge.agentic import (
            AgenticOrchestrator, SubTask, AgentState, OrchestratorConfig,
        )

        backend = MockBackend(responses={"OD": ""})
        orch = AgenticOrchestrator(
            backend,
            OrchestratorConfig(emit_transcript=False, max_retries=0),
        )
        state = AgentState()
        sub = SubTask(index=0, goal="detect", intent="detect")
        record = orch._execute_sub_task(image=None, sub=sub, state=state)

        assert record.attempts == 1
        assert record.verified is False


# ---------------------------------------------------------------------------
# 5. Multi-step orchestration
# ---------------------------------------------------------------------------

class TestMultiStepOrchestration:
    """Test the full run() loop with multiple sub-tasks."""

    def test_run_detect_and_describe(self):
        from florence_forge.agentic import AgenticOrchestrator, OrchestratorConfig

        backend = MockBackend(responses={
            "OD": _box_payload([[10, 20, 50, 60], [70, 80, 100, 110]]),
            "DETAILED_CAPTION": "A scene with two objects on a table.",
        })
        orch = AgenticOrchestrator(backend, OrchestratorConfig(emit_transcript=False))
        result = orch.run(image=None, goal="detect objects and describe the scene")

        assert result.success is True
        assert len(result.steps) == 2
        assert len(result.state.detected_objects) == 2
        assert len(result.state.descriptions) == 1
        assert "detected 2" in result.final_answer

    def test_run_count_task(self):
        from florence_forge.agentic import AgenticOrchestrator

        backend = MockBackend(responses={
            "COUNT_VP": "3",  # count = 3
        })
        orch = AgenticOrchestrator(backend)
        result = orch.run(image=None, goal="count the boxes")

        assert len(result.steps) >= 1
        # count should be parsed
        assert result.state.counts.get("count", 0) == 3

    def test_run_with_predefined_plan(self):
        from florence_forge.agentic import (
            AgenticOrchestrator, SubTask, OrchestratorConfig,
        )

        backend = MockBackend(responses={
            "OD": _box_payload([[10, 20, 50, 60]]),
            "OCR": "Some text here",
        })
        plan = [
            SubTask(index=0, goal="detect", intent="detect"),
            SubTask(index=1, goal="read text", intent="read_text_plain"),
        ]
        orch = AgenticOrchestrator(backend, OrchestratorConfig(emit_transcript=False))
        result = orch.run(image=None, goal="custom goal", plan=plan)

        assert len(result.steps) == 2
        assert all(r.verified for r in result.steps)

    def test_max_steps_cap(self):
        from florence_forge.agentic import (
            AgenticOrchestrator, SubTask, OrchestratorConfig,
        )

        backend = MockBackend(responses={"OD": _box_payload([[10, 20, 50, 60]])})
        # 5 sub-tasks but max_steps=2
        plan = [SubTask(index=i, goal="detect", intent="detect") for i in range(5)]
        orch = AgenticOrchestrator(backend, OrchestratorConfig(max_steps=2, emit_transcript=False))
        result = orch.run(image=None, goal="detect many times", plan=plan)

        assert len(result.steps) == 2


# ---------------------------------------------------------------------------
# 6. Transcript emission
# ---------------------------------------------------------------------------

class TestTranscript:
    """Test that the orchestrator emits agentic-token-wrapped transcripts."""

    def test_transcript_has_phases(self):
        from florence_forge.agentic import AgenticOrchestrator
        from florence_forge.core.agentic_tokens import (
            extract_phase, has_required_phases,
        )

        backend = MockBackend(responses={
            "OD": _box_payload([[10, 20, 50, 60]]),
            "DETAILED_CAPTION": "A scene with objects.",
        })
        orch = AgenticOrchestrator(backend)  # emit_transcript=True by default
        result = orch.run(image=None, goal="detect and describe")

        assert result.transcript
        # Should contain PLAN, ACT, VERIFY, DECIDE, DONE
        assert len(extract_phase(result.transcript, "plan")) > 0
        assert len(extract_phase(result.transcript, "act")) > 0
        assert len(extract_phase(result.transcript, "verify")) > 0
        assert len(extract_phase(result.transcript, "decide")) > 0
        assert len(extract_phase(result.transcript, "done")) > 0
        assert has_required_phases(result.transcript)

    def test_transcript_disabled(self):
        from florence_forge.agentic import AgenticOrchestrator, OrchestratorConfig

        backend = MockBackend(responses={
            "OD": _box_payload([[10, 20, 50, 60]]),
        })
        orch = AgenticOrchestrator(backend, OrchestratorConfig(emit_transcript=False))
        result = orch.run(image=None, goal="detect objects")

        assert result.transcript == ""

    def test_transcript_has_summarize_state(self):
        from florence_forge.agentic import AgenticOrchestrator, OrchestratorConfig
        from florence_forge.core.agentic_tokens import extract_phase

        backend = MockBackend(responses={
            "OD": _box_payload([[10, 20, 50, 60]]),
            "DETAILED_CAPTION": "A scene.",
            "OCR": "text",
            "OCR_WITH_REGION": _box_payload([[10, 20, 50, 60]]) + "text",
        })
        # summarize_every=1 so every step triggers a summary
        orch = AgenticOrchestrator(backend, OrchestratorConfig(summarize_every=1))
        result = orch.run(image=None, goal="detect, describe, and read text")

        summaries = extract_phase(result.transcript, "summarize_state")
        assert len(summaries) >= 2  # at least initial + one mid-run


# ---------------------------------------------------------------------------
# 7. Aggregation
# ---------------------------------------------------------------------------

class TestAggregation:
    """Test the final answer aggregation."""

    def test_aggregate_empty_state(self):
        from florence_forge.agentic import AgenticOrchestrator, AgentState

        state = AgentState()
        answer = AgenticOrchestrator._aggregate("unknown goal", state, [])
        assert "No observations" in answer

    def test_aggregate_with_detections(self):
        from florence_forge.agentic import AgenticOrchestrator, AgentState

        state = AgentState()
        state.detected_objects = [{"box": [10, 20, 50, 60]}, {"box": [70, 80, 100, 110]}]
        answer = AgenticOrchestrator._aggregate("detect objects", state, [])
        assert "2" in answer
        assert "detected" in answer.lower()

    def test_aggregate_with_counts(self):
        from florence_forge.agentic import AgenticOrchestrator, AgentState

        state = AgentState()
        state.counts = {"count": 5}
        answer = AgenticOrchestrator._aggregate("count cars", state, [])
        assert "5" in answer

    def test_result_to_dict_serializable(self):
        import json
        from florence_forge.agentic import AgenticOrchestrator, OrchestratorConfig

        backend = MockBackend(responses={
            "OD": _box_payload([[10, 20, 50, 60]]),
        })
        orch = AgenticOrchestrator(backend, OrchestratorConfig(emit_transcript=False))
        result = orch.run(image=None, goal="detect objects")

        d = result.to_dict()
        # Should be JSON-serializable
        serialized = json.dumps(d, default=str)
        assert "goal" in json.loads(serialized)
        assert "steps" in json.loads(serialized)


# ---------------------------------------------------------------------------
# 8. State management
# ---------------------------------------------------------------------------

class TestAgentState:
    """Test the AgentState accumulate-and-summarize behavior."""

    def test_state_summarize_empty(self):
        from florence_forge.agentic import AgentState

        state = AgentState()
        assert "no observations" in state.summarize()

    def test_state_summarize_with_data(self):
        from florence_forge.agentic import AgentState

        state = AgentState()
        state.detected_objects = [{"box": [1, 2, 3, 4]}]
        state.extracted_text = ["hello", "world"]
        state.counts = {"count": 3}
        summary = state.summarize()
        assert "1 detected" in summary
        assert "2 text" in summary
        assert "count:3" in summary

    def test_state_to_dict(self):
        from florence_forge.agentic import AgentState

        state = AgentState()
        state.detected_objects = [{"box": [1, 2, 3, 4]}]
        d = state.to_dict()
        assert d["detected_objects"] == [{"box": [1, 2, 3, 4]}]
        assert d["counts"] == {}
        assert d["extracted_text"] == []


# ---------------------------------------------------------------------------
# 9. ToolCall and StepRecord
# ---------------------------------------------------------------------------

class TestDataclasses:
    """Test the helper dataclasses."""

    def test_tool_call_describe_no_arg(self):
        from florence_forge.agentic import ToolCall

        call = ToolCall(intent="detect", task_name="OD")
        assert "OD" in call.describe()
        assert "detect" in call.describe()

    def test_tool_call_describe_with_arg(self):
        from florence_forge.agentic import ToolCall

        call = ToolCall(intent="count", task_name="COUNT_VP", text_input="cars")
        desc = call.describe()
        assert "cars" in desc

    def test_step_record_defaults(self):
        from florence_forge.agentic import StepRecord, ToolCall

        call = ToolCall(intent="detect", task_name="OD")
        record = StepRecord(sub_task_index=0, intent="detect", tool_call=call)
        assert record.verified is False
        assert record.attempts == 1
        assert record.issues == []

    def test_plan_result_len(self):
        from florence_forge.agentic import PlanResult, SubTask

        plan = PlanResult(sub_tasks=[
            SubTask(index=0, goal="a", intent="detect"),
            SubTask(index=1, goal="b", intent="describe"),
        ])
        assert len(plan) == 2


class TestOrchestratorTimeout:
    """max_total_seconds wall-clock budget stops the run early."""

    def test_budget_stops_run_early(self):
        import time as _time

        from florence_forge.agentic import AgenticOrchestrator, OrchestratorConfig

        class SlowBackend:
            def predict_task(self, images, task_name, text_input=None, **kwargs):
                _time.sleep(0.05)
                return "a photo"

        goal = "detect all cars and describe scene"
        orch = AgenticOrchestrator(
            SlowBackend(),
            OrchestratorConfig(emit_transcript=False, max_total_seconds=0.01),
        )
        planned = len(orch.decompose(goal))
        assert planned >= 2  # sanity: goal decomposes to multiple sub-tasks

        result = orch.run(image=object(), goal=goal)
        # After the first (slow) step the budget is exceeded, so the run stops
        # before completing all planned sub-tasks.
        assert 0 < len(result.steps) < planned

    def test_no_budget_runs_all_steps(self):
        from florence_forge.agentic import AgenticOrchestrator, OrchestratorConfig

        class FastBackend:
            def predict_task(self, images, task_name, text_input=None, **kwargs):
                return "a photo of a street"

        orch = AgenticOrchestrator(
            FastBackend(),
            OrchestratorConfig(emit_transcript=False, max_total_seconds=None),
        )
        result = orch.run(image=object(), goal="describe the scene")
        assert len(result.steps) >= 1

    def test_run_stream_respects_budget(self):
        import time as _time

        from florence_forge.agentic import AgenticOrchestrator, OrchestratorConfig

        class SlowBackend:
            def predict_task(self, images, task_name, text_input=None, **kwargs):
                _time.sleep(0.05)
                return "a photo"

        goal = "detect cars and describe the scene"
        orch = AgenticOrchestrator(
            SlowBackend(),
            OrchestratorConfig(emit_transcript=False, max_total_seconds=0.01),
        )
        planned = len(orch.decompose(goal))
        events = list(orch.run_stream(image=object(), goal=goal))
        types = [e["type"] for e in events]
        assert types[0] == "plan"
        assert types[-1] == "done"
        # Stopped early: fewer step events than planned sub-tasks.
        assert 0 < types.count("step") < planned


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
