"""Meta-cognitive token definitions for Agentic Visual Reasoning.

These tokens extend Florence-2's BART tokenizer with structured reasoning
delimiters that scaffold multi-step visual reasoning:

  ``<PLAN>`` / ``</PLAN>``              — strategic planning before acting
  ``<ACT>`` / ``</ACT>``               — concrete action (grounding, scanning)
  ``<VERIFY>`` / ``</VERIFY>``          — verification of previous action result
  ``<REFLECT>`` / ``</REFLECT>``        — self-reflection on errors or progress
  ``<DECIDE>`` / ``</DECIDE>``          — final decision / answer commitment
  ``<SUMMARIZE_STATE>`` / ``</SUMMARIZE_STATE>`` — compress history to key state
  ``<DONE>`` / ``</DONE>``              — signal task completion

Usage flow::

    1. ``register_agentic_tokens(tokenizer)`` — add tokens + resize embeddings
    2. Chain builders emit text with these delimiters
    3. Reward models parse token-delimited segments for structured scoring
    4. Phase-aware loss weighting uses token positions to scale gradients

All tokens are registered as *special tokens* so BART never splits them
across sub-words, and ``add_special_tokens=False`` during answer encoding
keeps them in the suffix without re-injecting ``<s>`` / ``</s>``.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token vocabulary
# ---------------------------------------------------------------------------

#: Paired meta-cognitive delimiter tokens (open, close).
AGENTIC_TOKEN_PAIRS: List[Tuple[str, str]] = [
    ("<PLAN>", "</PLAN>"),
    ("<ACT>", "</ACT>"),
    ("<VERIFY>", "</VERIFY>"),
    ("<REFLECT>", "</REFLECT>"),
    ("<DECIDE>", "</DECIDE>"),
    ("<SUMMARIZE_STATE>", "</SUMMARIZE_STATE>"),
    ("<DONE>", "</DONE>"),
]

#: Flat list of all agentic special tokens (for tokenizer.add_tokens).
AGENTIC_SPECIAL_TOKENS: List[str] = [
    tok for pair in AGENTIC_TOKEN_PAIRS for tok in pair
]

#: Mapping from logical phase name to (open, close) tuple.
AGENTIC_PHASE_TOKENS: Dict[str, Tuple[str, str]] = {
    "plan":            ("<PLAN>", "</PLAN>"),
    "act":             ("<ACT>", "</ACT>"),
    "verify":          ("<VERIFY>", "</VERIFY>"),
    "reflect":         ("<REFLECT>", "</REFLECT>"),
    "decide":          ("<DECIDE>", "</DECIDE>"),
    "summarize_state": ("<SUMMARIZE_STATE>", "</SUMMARIZE_STATE>"),
    "done":            ("<DONE>", "</DONE>"),
}

#: Ordered phase names — the canonical agentic reasoning loop.
#: Core phases (PLAN→ACT→VERIFY→REFLECT→DECIDE) followed by control phases.
AGENTIC_PHASE_ORDER: List[str] = [
    "plan", "act", "verify", "reflect", "decide",
    "summarize_state", "done",
]

#: Core reasoning phases that form the main agentic loop.
CORE_PHASES: List[str] = ["plan", "act", "verify", "reflect", "decide"]

#: Control / signaling phases (not part of the main reasoning loop).
CONTROL_PHASES: List[str] = ["summarize_state", "done"]

# Phases that must appear at least once for a well-formed agentic trajectory.
REQUIRED_PHASES: List[str] = ["act", "decide"]

#: Loss weight per phase for phase-aware training.
#: Higher weight → model learns that phase more aggressively.
#: DECIDE gets the highest weight since it's the final answer commitment.
#: REFLECT gets a boost to encourage self-correction learning.
PHASE_LOSS_WEIGHTS: Dict[str, float] = {
    "plan":            0.8,
    "act":             1.0,
    "verify":          1.2,
    "reflect":         1.5,
    "decide":          2.0,
    "summarize_state": 0.6,
    "done":            1.0,
}


# ---------------------------------------------------------------------------
# Tokenizer registration
# ---------------------------------------------------------------------------

def register_agentic_tokens(tokenizer) -> int:
    """Add agentic meta-cognitive tokens to *tokenizer* if not already present.

    Returns the number of newly added tokens (0 if all already exist).
    The caller is responsible for ``model.resize_token_embeddings()``.
    """
    if tokenizer is None:
        return 0

    existing = set(tokenizer.get_vocab())
    new_tokens = [t for t in AGENTIC_SPECIAL_TOKENS if t not in existing]
    if not new_tokens:
        return 0

    added = tokenizer.add_tokens(new_tokens, special_tokens=True)
    logger.info("Registered %d agentic meta-cognitive tokens: %s", added, new_tokens)
    return added


# ---------------------------------------------------------------------------
# Token-based text helpers
# ---------------------------------------------------------------------------

def wrap_phase(phase: str, content: str) -> str:
    """Wrap *content* in the open/close tokens for *phase*.

    >>> wrap_phase("plan", "Scan left-to-right")
    '<PLAN>Scan left-to-right</PLAN>'
    """
    pair = AGENTIC_PHASE_TOKENS.get(phase)
    if pair is None:
        raise ValueError(f"Unknown agentic phase: {phase!r}. "
                         f"Valid phases: {list(AGENTIC_PHASE_TOKENS)}")
    open_tok, close_tok = pair
    return f"{open_tok}{content}{close_tok}"


def extract_phase(text: str, phase: str) -> List[str]:
    """Extract all contents wrapped in *phase* delimiters from *text*.

    Returns a list of inner texts, preserving order of appearance.
    """
    import re
    pair = AGENTIC_PHASE_TOKENS.get(phase)
    if pair is None:
        raise ValueError(f"Unknown agentic phase: {phase!r}")
    open_tok, close_tok = pair
    pattern = re.compile(
        re.escape(open_tok) + r"(.*?)" + re.escape(close_tok),
        re.DOTALL,
    )
    return [m.group(1).strip() for m in pattern.finditer(text)]


def extract_all_phases(text: str) -> Dict[str, List[str]]:
    """Extract contents for all agentic phases from *text*."""
    return {phase: extract_phase(text, phase) for phase in AGENTIC_PHASE_ORDER}


def has_required_phases(text: str) -> bool:
    """Return True if *text* contains all :data:`REQUIRED_PHASES`."""
    return all(extract_phase(text, p) for p in REQUIRED_PHASES)


def get_phase_order(text: str) -> List[str]:
    """Return the sequence of phase open-tokens as they appear in *text*."""
    import re
    found: List[str] = []
    for phase in AGENTIC_PHASE_ORDER:
        open_tok = AGENTIC_PHASE_TOKENS[phase][0]
        if open_tok in text:
            found.append(phase)
    return found


# ---------------------------------------------------------------------------
# Phase span analysis (for phase-aware loss weighting)
# ---------------------------------------------------------------------------

def find_phase_spans(text: str) -> List[Tuple[str, int, int]]:
    """Find all (phase, start_char, end_char) spans in *text*.

    Returns a list of tuples in order of appearance. Each span covers the
    content *between* open and close tokens (exclusive of the tokens themselves).

    This is used by phase-aware loss weighting to scale gradient signals
    differently for PLAN vs ACT vs REFLECT vs DECIDE phases.
    """
    import re
    spans: List[Tuple[str, int, int]] = []
    for phase in AGENTIC_PHASE_ORDER:
        open_tok, close_tok = AGENTIC_PHASE_TOKENS[phase]
        pattern = re.compile(
            re.escape(open_tok) + r"(.*?)" + re.escape(close_tok),
            re.DOTALL,
        )
        for m in pattern.finditer(text):
            content_start = m.start(1)
            content_end = m.end(1)
            spans.append((phase, content_start, content_end))
    spans.sort(key=lambda s: s[1])
    return spans


def get_phase_loss_weights() -> Dict[str, float]:
    """Return the phase-to-loss-weight mapping (copy)."""
    return dict(PHASE_LOSS_WEIGHTS)
