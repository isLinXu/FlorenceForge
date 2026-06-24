"""Tool registry: maps agentic intents to native Florence-2 tasks.

The outer orchestrator chooses *intents* (detect / read_text / count / locate /
describe / region_describe); this registry resolves each intent to a concrete
Florence-2 task name (validated against ``core.tasks.FLORENCE2_TASKS``) plus
metadata describing how to parse the tool's raw text output.

Keeping this mapping in one place means the orchestrator's planning logic never
hard-codes Florence-2 task tokens — adding a new tool is a single registry entry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..core.tasks import FLORENCE2_TASKS, validate_task_name


@dataclass(frozen=True)
class ToolSpec:
    """Specification of a single agentic tool (a native Florence-2 task).

    Attributes:
        intent: High-level intent keyword the planner reasons about.
        task_name: Concrete Florence-2 task name (key in ``FLORENCE2_TASKS``).
        needs_text_input: Whether the task requires a ``text_input`` argument.
        output_kind: One of ``"boxes"``, ``"text"``, ``"regions"``, ``"count"``
            — tells the orchestrator how to parse the raw output into state.
        description: Human-readable purpose, used in ``<DECOMPOSE>`` reasoning.
        keywords: Trigger words that map a free-form sub-goal to this intent.
    """

    intent: str
    task_name: str
    needs_text_input: bool
    output_kind: str
    description: str
    keywords: List[str] = field(default_factory=list)

    @property
    def prompt(self) -> str:
        """The underlying Florence-2 task prompt token."""
        return FLORENCE2_TASKS[self.task_name].prompt


# Canonical intent → tool mapping. Task names are validated at import time.
_RAW_TOOLS: List[ToolSpec] = [
    ToolSpec(
        intent="detect",
        task_name="OD",
        needs_text_input=False,
        output_kind="boxes",
        description="Detect all objects in the image and return bounding boxes.",
        keywords=["detect", "object", "find all", "bounding box", "bbox",
                  "检测", "目标", "框"],
    ),
    ToolSpec(
        intent="read_text",
        task_name="OCR_WITH_REGION",
        needs_text_input=False,
        output_kind="regions",
        description="Read text in the image with per-region bounding boxes.",
        keywords=["ocr", "text", "read", "字", "文字", "识别", "标注"],
    ),
    ToolSpec(
        intent="read_text_plain",
        task_name="OCR",
        needs_text_input=False,
        output_kind="text",
        description="Read all text in the image as a plain string.",
        keywords=["plain text", "transcribe", "全文"],
    ),
    ToolSpec(
        intent="count",
        task_name="COUNT_VP",
        needs_text_input=True,
        output_kind="count",
        description="Count instances of a specified object class.",
        keywords=["count", "how many", "number of", "数量", "计数", "多少"],
    ),
    ToolSpec(
        intent="locate",
        task_name="CAPTION_TO_PHRASE_GROUNDING",
        needs_text_input=True,
        output_kind="boxes",
        description="Locate a phrase / object described in text and ground it.",
        keywords=["locate", "where", "ground", "find the", "定位", "在哪"],
    ),
    ToolSpec(
        intent="open_detect",
        task_name="OPEN_VOCABULARY_DETECTION",
        needs_text_input=True,
        output_kind="boxes",
        description="Open-vocabulary detection of an arbitrary described class.",
        keywords=["open vocabulary", "detect the", "开放词汇"],
    ),
    ToolSpec(
        intent="describe",
        task_name="DETAILED_CAPTION",
        needs_text_input=False,
        output_kind="text",
        description="Produce a detailed natural-language description of the image.",
        keywords=["describe", "caption", "what is", "描述", "说明"],
    ),
    ToolSpec(
        intent="region_describe",
        task_name="DENSE_REGION_CAPTION",
        needs_text_input=False,
        output_kind="regions",
        description="Describe every salient region with its bounding box.",
        keywords=["dense", "every region", "all regions", "区域描述"],
    ),
    ToolSpec(
        intent="region_category",
        task_name="REGION_TO_CATEGORY",
        needs_text_input=True,
        output_kind="text",
        description="Classify the object inside a given region box.",
        keywords=["classify region", "what category", "区域类别"],
    ),
]


def _build_registry() -> Dict[str, ToolSpec]:
    registry: Dict[str, ToolSpec] = {}
    for spec in _RAW_TOOLS:
        if not validate_task_name(spec.task_name):
            raise ValueError(
                f"ToolSpec {spec.intent!r} references unknown Florence-2 task "
                f"{spec.task_name!r}. Valid tasks must exist in FLORENCE2_TASKS."
            )
        registry[spec.intent] = spec
    return registry


#: intent -> ToolSpec
TOOL_REGISTRY: Dict[str, ToolSpec] = _build_registry()


def get_tool_spec(intent: str) -> ToolSpec:
    """Return the :class:`ToolSpec` for *intent* or raise ``KeyError``."""
    if intent not in TOOL_REGISTRY:
        raise KeyError(
            f"Unknown tool intent: {intent!r}. "
            f"Available: {sorted(TOOL_REGISTRY)}"
        )
    return TOOL_REGISTRY[intent]


def list_tools() -> List[ToolSpec]:
    """Return all registered tool specs (stable order)."""
    return list(TOOL_REGISTRY.values())


def select_tool_for_intent(text: str) -> Optional[ToolSpec]:
    """Heuristically map a free-form sub-goal *text* to a tool via keywords.

    Returns the best-matching :class:`ToolSpec`, or ``None`` if nothing matches.
    Matching is case-insensitive and scored by number of keyword hits, so a
    sub-goal like "read all dimension text" prefers the OCR tool.
    """
    if not text or not text.strip():
        return None
    lowered = text.lower()
    best: Optional[ToolSpec] = None
    best_score = 0
    for spec in TOOL_REGISTRY.values():
        score = sum(1 for kw in spec.keywords if kw.lower() in lowered)
        if score > best_score:
            best_score = score
            best = spec
    return best if best_score > 0 else None
