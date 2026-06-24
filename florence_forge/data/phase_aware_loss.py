"""Phase-aware loss weighting for Agentic meta-cognitive training.

When training with agentic tokens (``<PLAN>``, ``<ACT>``, ``<VERIFY>``,
``<REFLECT>``, ``<DECIDE>``), different reasoning phases deserve different
gradient emphasis:

  - ``DECIDE`` gets the highest weight (final answer commitment)
  - ``REFLECT`` gets a boost (encourage self-correction learning)
  - ``VERIFY`` gets a moderate boost (verification is critical for robustness)
  - ``PLAN`` gets a slightly lower weight (planning is important but less direct)
  - ``SUMMARIZE_STATE`` gets a low weight (auxiliary compression signal)

This module provides utilities to:
  1. Identify token positions belonging to each agentic phase
  2. Construct a per-token weight tensor that scales the loss accordingly

Integration point: ``dataset_encoding.py`` can call
``apply_phase_weights_to_labels()`` to produce a ``loss_weights`` tensor
alongside the standard ``labels`` tensor. The trainer then uses this to
scale the cross-entropy loss per token.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import torch

from ..core.agentic_tokens import (
    AGENTIC_PHASE_TOKENS,
    PHASE_LOSS_WEIGHTS,
    find_phase_spans,
)

logger = logging.getLogger(__name__)

#: Default weight for tokens outside any agentic phase (e.g. connecting text).
DEFAULT_OUTSIDE_WEIGHT: float = 1.0


def _char_spans_to_token_spans(
    text: str,
    token_ids: torch.Tensor,
    tokenizer: Any,
) -> List[Tuple[str, int, int]]:
    """Map character-level phase spans to token-level spans.

    Uses offset mapping from the tokenizer's fast encoding to translate
    character positions to token indices. Falls back to a heuristic
    search if offset mapping is unavailable.

    Returns list of (phase, start_token_idx, end_token_idx) tuples.
    """
    # Try fast tokenizer offset mapping
    try:
        encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        offsets = encoding.get("offset_mapping", None)
        if offsets is not None:
            char_spans = find_phase_spans(text)
            token_spans: List[Tuple[str, int, int]] = []
            for phase, char_start, char_end in char_spans:
                # Find token indices whose offsets overlap with [char_start, char_end)
                tok_start = None
                tok_end = None
                for i, (off_start, off_end) in enumerate(offsets):
                    if off_start >= char_end:
                        break
                    if off_end > char_start:
                        if tok_start is None:
                            tok_start = i
                        tok_end = i + 1
                if tok_start is not None and tok_end is not None:
                    token_spans.append((phase, tok_start, tok_end))
            return token_spans
    except Exception:
        pass

    # Fallback: find token IDs that match agentic special tokens
    token_spans_fallback: List[Tuple[str, int, int]] = []

    id_to_token = {}
    if hasattr(tokenizer, "convert_ids_to_tokens"):
        for i in range(len(token_ids)):
            id_to_token[i] = tokenizer.convert_ids_to_tokens(int(token_ids[i]))

    for phase in AGENTIC_PHASE_TOKENS:
        open_tok, close_tok = AGENTIC_PHASE_TOKENS[phase]
        open_id = tokenizer.convert_tokens_to_ids(open_tok) if hasattr(tokenizer, "convert_tokens_to_ids") else None
        close_id = tokenizer.convert_tokens_to_ids(close_tok) if hasattr(tokenizer, "convert_tokens_to_ids") else None

        in_phase = False
        phase_start = None
        for i in range(len(token_ids)):
            tid = int(token_ids[i])
            if open_id is not None and tid == open_id:
                in_phase = True
                phase_start = i + 1  # content starts after open token
            elif close_id is not None and tid == close_id and in_phase:
                if phase_start is not None and i > phase_start:
                    token_spans_fallback.append((phase, phase_start, i))
                in_phase = False
                phase_start = None

    return token_spans_fallback


def build_phase_weight_tensor(
    labels: torch.Tensor,
    answer_text: str,
    tokenizer: Any,
    *,
    prompt_length: int = 0,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Build a per-token loss weight tensor based on agentic phase membership.

    Args:
        labels: The labels tensor (1D) with ``ignore_index`` for masked positions.
        answer_text: The raw suffix text containing agentic token delimiters.
        tokenizer: The tokenizer (for mapping tokens to phase IDs).
        prompt_length: Number of leading tokens that are prompt (not supervised).
        ignore_index: The ignore index used in labels.

    Returns:
        A float tensor of the same shape as ``labels``, where each supervised
        token gets a weight from ``PHASE_LOSS_WEIGHTS`` and ignored tokens
        get 0.0.
    """
    weights = torch.zeros_like(labels, dtype=torch.float32)

    # Tokens before prompt_length are not supervised
    if prompt_length > 0:
        supervised = labels.clone()
        supervised[:prompt_length] = ignore_index
    else:
        supervised = labels

    # Find phase token spans in the answer text
    token_spans = _char_spans_to_token_spans(answer_text, supervised, tokenizer)

    # Build a mapping: token_idx -> phase
    token_phase: Dict[int, str] = {}
    for phase, tok_start, tok_end in token_spans:
        # Adjust for prompt_length offset
        adjusted_start = tok_start + prompt_length
        adjusted_end = tok_end + prompt_length
        for i in range(adjusted_start, min(adjusted_end, len(weights))):
            if supervised[i] != ignore_index:
                token_phase[i] = phase

    # Assign weights
    for i in range(len(weights)):
        if supervised[i] == ignore_index:
            weights[i] = 0.0
        elif i in token_phase:
            weights[i] = PHASE_LOSS_WEIGHTS.get(token_phase[i], DEFAULT_OUTSIDE_WEIGHT)
        else:
            weights[i] = DEFAULT_OUTSIDE_WEIGHT

    return weights


def apply_phase_weights_to_labels(
    labels: torch.Tensor,
    answer_text: str,
    tokenizer: Any,
    *,
    prompt_length: int = 0,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Convenience wrapper — alias for ``build_phase_weight_tensor``."""
    return build_phase_weight_tensor(
        labels=labels,
        answer_text=answer_text,
        tokenizer=tokenizer,
        prompt_length=prompt_length,
        ignore_index=ignore_index,
    )


def phase_weighted_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_weights: torch.Tensor,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Compute phase-weighted cross-entropy loss.

    Args:
        logits: Model logits [batch, seq_len, vocab_size].
        labels: Ground truth labels [batch, seq_len].
        loss_weights: Per-token weights [batch, seq_len].
        ignore_index: Tokens to ignore in loss computation.

    Returns:
        Scalar loss tensor.
    """
    # Standard cross-entropy (reduction='none' to get per-token losses)
    loss_fn = torch.nn.CrossEntropyLoss(
        ignore_index=ignore_index,
        reduction="none",
    )
    # logits: [batch, seq, vocab] → [batch*vocab, seq] for CE
    # labels: [batch, seq]
    per_token_loss = loss_fn(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
    )  # [batch * seq]

    weights_flat = loss_weights.view(-1)
    # Mask out ignored positions
    mask = (labels.view(-1) != ignore_index).float()
    weighted = per_token_loss * weights_flat * mask
    total_weight = weights_flat * mask
    # Avoid division by zero
    loss = weighted.sum() / (total_weight.sum() + 1e-8)
    return loss
