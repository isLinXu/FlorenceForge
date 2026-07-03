"""Trainable sparse gate for experimental MoE layers."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseGate(nn.Module):
    """Project hidden states to per-token expert weights.

    Args:
        d_model: Input hidden size.
        d_state: Kept for backward compatibility with older constructors.
        n_heads: Number of experts/gating heads.
        top_k: Optional number of experts to keep per token. ``None`` keeps all.
        threshold: Optional post-softmax pruning threshold. Values below it are
            zeroed and the remaining weights are renormalized.
        hard_routing: If ``True``, use hard (argmax) routing: each token is
            assigned to exactly one expert with weight 1.0. The gradient still
            flows through the soft logits via a straight-through estimator.

    Shape:
        Input: ``(batch, seq, d_model)``
        Output: ``(batch, seq, n_heads)``, normalized over the expert dimension.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int,
        n_heads: int = 1,
        top_k: Optional[int] = None,
        threshold: float = 0.0,
        hard_routing: bool = False,
    ):
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if n_heads <= 0:
            raise ValueError("n_heads must be positive")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive when provided")

        self.d_model = d_model
        self.d_state = d_state
        self.n_heads = n_heads
        self.top_k = min(top_k, n_heads) if top_k is not None else None
        self.threshold = float(threshold)
        self.hard_routing = bool(hard_routing)

        self.proj = nn.Linear(d_model, n_heads)
        self.temperature = nn.Parameter(torch.ones(()))
        self.last_logits: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(
                f"SparseGate expects a 3D tensor (batch, seq, dim), got {tuple(x.shape)}"
            )
        if x.shape[-1] != self.d_model:
            raise ValueError(f"Expected hidden size {self.d_model}, got {x.shape[-1]}")

        temperature = self.temperature.clamp_min(1e-4)
        logits = self.proj(x) / temperature
        self.last_logits = logits.detach().clone()

        if self.top_k is not None and self.top_k < self.n_heads:
            values, indices = torch.topk(logits, k=self.top_k, dim=-1)
            sparse_logits = torch.full_like(logits, torch.finfo(logits.dtype).min)
            logits = sparse_logits.scatter(-1, indices, values)

        weights = F.softmax(logits, dim=-1)

        if self.threshold > 0:
            weights = weights.masked_fill(weights < self.threshold, 0.0)
            normalizer = weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            weights = weights / normalizer

        # ── Hard routing: straight-through estimator ───────────
        if self.hard_routing:
            # Forward: one-hot (argmax)
            hard_indices = torch.argmax(weights, dim=-1, keepdim=True)  # (B, S, 1)
            hard_weights = torch.zeros_like(weights)
            hard_weights = hard_weights.scatter(-1, hard_indices, 1.0)
            # Backward: gradient flows through soft weights
            weights = hard_weights + weights - weights.detach()

        return weights
