"""Experimental MoE decoder wrapper."""

from __future__ import annotations

import torch
import torch.nn as nn

from .moe_layer import MoELayer


class MoEDecoder(nn.Module):
    """Thin decoder wrapper around :class:`MoELayer`."""

    def __init__(self, num_experts: int, d_model: int, d_state: int):
        super().__init__()
        self.num_experts = num_experts
        self.d_model = d_model
        self.d_state = d_state
        self.moe_layer = MoELayer(num_experts, d_model, d_state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.moe_layer(x)
