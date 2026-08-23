"""Trainable SelectiveSSM-style mixer for experimental MoE work."""

from __future__ import annotations

import torch
import torch.nn as nn

from .moe_encoder import MoEEncoder
from .moe_layer import MoELayer
from .sparse_gate import SparseGate


class SelectiveSSMMixer(nn.Module):
    """Small residual mixer combining state projection and sparse gating.

    This is a conservative experimental placeholder, not a full selective SSM
    implementation. The previous version generated random constant tensors and
    indexed inputs with incompatible dimensions. This implementation keeps the
    public constructor intact while making all parameters trainable and the
    forward pass shape-safe.
    """

    def __init__(self, d_model: int, d_state: int, n_heads: int = 1):
        super().__init__()
        if d_model <= 0 or d_state <= 0:
            raise ValueError("d_model and d_state must be positive")
        if n_heads <= 0:
            raise ValueError("n_heads must be positive")

        self.d_model = d_model
        self.d_state = d_state
        self.n_heads = n_heads

        self.state_in = nn.Linear(d_model, d_state)
        self.state_out = nn.Linear(d_state, d_model)
        self.head_values = nn.Linear(d_model, n_heads * d_model)
        self.mix_proj = nn.Linear(d_model, d_model)
        self.sparse_gate = SparseGate(d_model, d_state, n_heads)

    @property
    def selective_params(self) -> torch.Tensor:
        """Backward-compatible view of trainable selective parameters."""
        return self.state_in.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(
                f"SelectiveSSMMixer expects a 3D tensor (batch, seq, dim), got {tuple(x.shape)}"
            )
        if x.shape[-1] != self.d_model:
            raise ValueError(f"Expected hidden size {self.d_model}, got {x.shape[-1]}")

        state_update = self.state_out(torch.tanh(self.state_in(x)))
        head_values = self.head_values(x).view(*x.shape[:2], self.n_heads, self.d_model)
        gate_weights = self.sparse_gate(x)
        sparse_update = torch.einsum("bsh,bshd->bsd", gate_weights, head_values)
        mix = torch.sigmoid(self.mix_proj(x))
        return x + mix * (state_update + sparse_update)


__all__ = ["SelectiveSSMMixer", "SparseGate", "MoELayer", "MoEEncoder"]
