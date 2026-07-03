"""Experimental MoE model container."""

from __future__ import annotations

from typing import Union

import torch
import torch.nn as nn

from .moe_layer import MoELayer


class MoEModel(nn.Module):
    """Small token model backed by a single experimental MoE layer."""

    def __init__(
        self,
        vocab_size: int = 30592,
        max_position_embeddings: int = 512,
        num_experts: int = 8,
        d_model: int = 768,
        d_state: int = 256,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.num_experts = num_experts
        self.d_model = d_model
        self.d_state = d_state

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_position_embeddings, d_model)
        self.moe_layer = MoELayer(num_experts, d_model, d_state)
        self.output_projection = nn.Linear(d_state, vocab_size)

    def _embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.shape[1]
        if seq_len > self.max_position_embeddings:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_position_embeddings={self.max_position_embeddings}"
            )
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        return self.token_embedding(input_ids) + self.position_embedding(positions)

    def forward(
        self, input_ids: Union[torch.Tensor, torch.LongTensor], **kwargs
    ) -> torch.Tensor:
        if input_ids.dtype in (torch.long, torch.int64, torch.int32):
            hidden = self._embed_tokens(input_ids.long())
            return self.output_projection(self.moe_layer(hidden))

        return self.moe_layer(input_ids)
