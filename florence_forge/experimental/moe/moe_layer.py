"""Minimal trainable MoE layer for experimental use."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .sparse_gate import SparseGate


class MoELayer(nn.Module):
    """Mixture-of-Experts layer with per-token routing weights and capacity factor.

    Supports sparse forward (only top-k experts computed per token) and
    capacity-factor-based overflow tracking for production-grade load balancing.
    """

    def __init__(
        self,
        num_experts: int,
        d_model: int,
        d_state: int,
        top_k: Optional[int] = None,
        gate_threshold: float = 0.0,
        capacity_factor: Optional[float] = 1.25,
    ):
        super().__init__()
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if d_model <= 0 or d_state <= 0:
            raise ValueError("d_model and d_state must be positive")

        self.num_experts = num_experts
        self.d_model = d_model
        self.d_state = d_state
        self.capacity_factor = capacity_factor

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
        self._overflow_stats: Optional[torch.Tensor] = None

    @property
    def selective_params(self) -> torch.Tensor:
        """Backward-compatible view of the gate projection weights."""
        return self.gate.proj.weight

    def _apply_capacity(self, gate_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply capacity-factor-based overflow tracking.

        Returns:
            (gate_weights, overflow_stats) where overflow_stats[e] is the number
            of tokens overflowed for expert e.
        """
        if self.capacity_factor is None or self.capacity_factor <= 0:
            return gate_weights, torch.zeros(self.num_experts, device=gate_weights.device)

        B, S, E = gate_weights.shape
        num_tokens = B * S
        capacity = max(1, int(self.capacity_factor * num_tokens / E))

        # 对每个专家，保留 top capacity 个 token（按权重）
        capacity_mask = torch.zeros_like(gate_weights, dtype=torch.bool)
        for e in range(E):
            weights_e = gate_weights[:, :, e].reshape(-1)  # (B*S,)
            k = min(capacity, num_tokens)
            top_indices = torch.topk(weights_e, k, sorted=False).indices
            capacity_mask.view(-1, E)[top_indices, e] = True

        # 溢出统计：原始 mask (gate > 0) - 容量 mask
        active_mask = gate_weights > 0
        overflow_mask = active_mask & ~capacity_mask
        overflow_stats = overflow_mask.sum(dim=(0, 1)).float()

        # 截断溢出 token 的权重（生产级 hard routing）
        gate_weights = gate_weights * capacity_mask.float()
        # 重新归一化：确保每个 token 的权重和为 1（如果至少有一个专家未被溢出）
        weight_sum = gate_weights.sum(dim=-1, keepdim=True)
        gate_weights = torch.where(
            weight_sum > 0,
            gate_weights / (weight_sum + 1e-9),
            gate_weights,
        )
        return gate_weights, overflow_stats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"MoELayer expects a 3D tensor (batch, seq, dim), got {tuple(x.shape)}")
        if x.shape[-1] != self.d_model:
            raise ValueError(f"Expected hidden size {self.d_model}, got {x.shape[-1]}")

        gate_weights = self.gate(x)

        # 应用容量因子（记录溢出统计并截断溢出权重）
        if self.capacity_factor is not None:
            gate_weights, overflow_stats = self._apply_capacity(gate_weights)
            self._overflow_stats = overflow_stats.detach().clone()
        else:
            self._overflow_stats = None

        B, S, E = gate_weights.shape

        # 稀疏前向：仅计算 gate 权重 > 0 的专家
        output = torch.zeros(B, S, self.d_state, device=x.device, dtype=x.dtype)
        for e_idx in range(self.num_experts):
            expert_weights = gate_weights[:, :, e_idx]  # (B, S)
            mask = expert_weights > 0
            if not mask.any():
                continue
            selected_x = x[mask]                     # (N, d_model)
            selected_w = expert_weights[mask]        # (N,)
            expert_out = self.experts[e_idx](selected_x)  # (N, d_state)
            output[mask] = output[mask] + expert_out * selected_w.unsqueeze(-1)

        self.last_gate_weights = gate_weights.detach().clone()
        self._routing_sums = gate_weights.sum(dim=(0, 1)).detach().clone()
        return output
