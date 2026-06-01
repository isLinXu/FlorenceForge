"""Minimal trainable MoE layer for experimental use."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .sparse_gate import SparseGate


class MoELayer(nn.Module):
    """Mixture-of-Experts layer with per-token routing weights.

    The layer is intentionally small and explicit: every expert runs for every
    token, then the trainable gate mixes their outputs. This is not a production
    sparse-kernel implementation, but it gives the experimental module a correct
    shape contract and gradient flow.
    """

    def __init__(
        self,
        num_experts: int,
        d_model: int,
        d_state: int,
        top_k: Optional[int] = None,
        gate_threshold: float = 0.0,
    ):
        super().__init__()
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if d_model <= 0 or d_state <= 0:
            raise ValueError("d_model and d_state must be positive")

        self.num_experts = num_experts
        self.d_model = d_model
        self.d_state = d_state

        self.experts = nn.ModuleList(
            [nn.Linear(d_model, d_state) for _ in range(num_experts)]
        )
        self.gate = SparseGate(
            d_model=d_model,
            d_state=d_state,
            n_heads=num_experts,
            top_k=top_k,
            threshold=gate_threshold,
        )
        self.last_gate_weights: Optional[torch.Tensor] = None

    @property
    def selective_params(self) -> torch.Tensor:
        """Backward-compatible view of the gate projection weights."""
        return self.gate.proj.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"MoELayer expects a 3D tensor (batch, seq, dim), got {tuple(x.shape)}")
        if x.shape[-1] != self.d_model:
            raise ValueError(f"Expected hidden size {self.d_model}, got {x.shape[-1]}")

        gate_weights = self.gate(x)
        expert_outputs = torch.stack(
            [expert(x) for expert in self.experts],
            dim=2,
        )
        self.last_gate_weights = gate_weights
        return torch.einsum("bse,bsed->bsd", gate_weights, expert_outputs)
