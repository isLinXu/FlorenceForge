"""Lightweight validator for experimental MoE layers."""

from __future__ import annotations

import torch

from .moe_layer import MoELayer


class MoEValidator:
    """Validate basic MoE shape, finiteness, and routing invariants."""

    def __init__(self, moe_layer: MoELayer):
        self.moe_layer = moe_layer

    def validate(self, x: torch.Tensor) -> bool:
        with torch.no_grad():
            output = self.moe_layer(x)
            if output.shape != (*x.shape[:2], self.moe_layer.d_state):
                return False
            if not torch.isfinite(output).all():
                return False

            gate_weights = self.moe_layer.last_gate_weights
            if gate_weights is None:
                return False
            expected_gate_shape = (*x.shape[:2], self.moe_layer.num_experts)
            if gate_weights.shape != expected_gate_shape:
                return False
            # 若启用容量因子，溢出 token 的权重和可能为 0，因此同时接受 0 和 1
            sums = gate_weights.sum(dim=-1)
            valid = torch.isclose(
                sums, torch.ones_like(sums), atol=1e-5
            ) | torch.isclose(sums, torch.zeros_like(sums), atol=1e-5)
            return bool(valid.all().item())
