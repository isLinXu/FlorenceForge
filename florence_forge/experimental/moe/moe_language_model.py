"""Experimental MoE language model alias."""

from __future__ import annotations

from .moe_model import MoEModel


class MoELanguageModel(MoEModel):
    """Backward-compatible name for the experimental token MoE model."""

