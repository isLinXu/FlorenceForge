"""Agentic inference-time orchestration for FlorenceForge.

This subpackage implements the *outer* agent loop that treats Florence-2's
native tasks (``<OD>``, ``<OCR>``, ``<COUNT>``, grounding, ...) as **tools**,
and drives long-horizon visual reasoning by maintaining cross-step state.

Unlike the in-model meta-cognitive chains in ``core.agentic_tokens`` /
``data.agentic_trajectory_expander`` (which teach Florence-2 to emit a single
``<PLAN>...<DECIDE>`` chain in one forward pass), this layer lives entirely in
Python and is responsible for:

  * ``<DECOMPOSE>`` — splitting a high-level goal into ordered sub-tasks
  * ``<NEXT_ACTION>`` — choosing which native Florence-2 tool to call next
  * ``<VERIFY>`` / ``<REFLECT>`` — validating tool output and retrying on failure
  * ``<SUMMARIZE_STATE>`` — compressing accumulated observations

The orchestrator is backend-agnostic: it only needs an object exposing
``predict_task(images, task_name, text_input=None, **kwargs) -> str`` (which the
existing ``Florence2MultiTaskModel`` already provides). A lightweight protocol
plus a deterministic mock backend keep the whole module importable and testable
without torch or model weights — matching the CI-friendly style of the rest of
the codebase.
"""

from __future__ import annotations

from .agentic_orchestrator import (
    ToolBackend,
    ToolCall,
    StepRecord,
    AgentState,
    SubTask,
    PlanResult,
    OrchestratorConfig,
    AgenticOrchestrator,
)
from .tool_registry import (
    ToolSpec,
    TOOL_REGISTRY,
    get_tool_spec,
    list_tools,
    select_tool_for_intent,
)

__all__ = [
    "ToolBackend",
    "ToolCall",
    "StepRecord",
    "AgentState",
    "SubTask",
    "PlanResult",
    "OrchestratorConfig",
    "AgenticOrchestrator",
    "ToolSpec",
    "TOOL_REGISTRY",
    "get_tool_spec",
    "list_tools",
    "select_tool_for_intent",
]
