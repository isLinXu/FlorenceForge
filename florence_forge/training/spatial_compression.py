"""Spatial token compression module for Visual Primitive reasoning.

Inspired by Thinking-with-Visual-Primitives paper:
  At the ViT output, apply a 3x3 spatial compression
  (compress every 9 adjacent patch tokens into a single token along the channel dimension).

Example:
  756x756 image -> 14x14 patch -> 54x54 = 2916 patch tokens
  After 3x3 compression: 18x18 = 324 tokens
  Then CSA further compresses by 4x -> 81 KV entries.

This module implements the 3x3 spatial compression and a configurable
multi-stage compression pipeline. CSA (Compressed Sparse Attention) is
proprietary, so we substitute with standard attention pooling.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class SpatialCompression(nn.Module):
    """Compress a 2D grid of tokens using a non-overlapping kernel.

    Input:  (B, H, W, C) where H and W are spatial dimensions of patch tokens.
    Output: (B, H//k, W//k, C_out) where k = kernel_size.

    Two modes:
      - ``"project"``: flatten each k×k block, then linear-project to out_channels.
      - ``"concat"``: flatten each k×k block, concatenate channels (no projection).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        kernel_size: int = 3,
        mode: str = "project",
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.mode = mode

        if mode == "project":
            out_channels = out_channels or in_channels
            self.proj = nn.Linear(
                in_channels * (kernel_size ** 2),
                out_channels,
            )
            self.out_channels = out_channels
        elif mode == "concat":
            self.out_channels = in_channels * (kernel_size ** 2)
        else:
            raise ValueError(f"mode must be 'project' or 'concat', got '{mode}'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, H, W, C) tensor of patch tokens.

        Returns:
            (B, H//k, W//k, C_out) compressed tensor.
        """
        B, H, W, C = x.shape
        k = self.kernel_size
        if H % k != 0 or W % k != 0:
            # Pad to make divisible
            pad_h = (k - H % k) % k
            pad_w = (k - W % k) % k
            x = torch.nn.functional.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            _, H, W, _ = x.shape

        # (B, H, W, C) -> (B, H//k, k, W//k, k, C)
        x = x.reshape(B, H // k, k, W // k, k, C)
        # -> (B, H//k, W//k, k, k, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        # -> (B, H//k, W//k, k*k*C)
        x = x.reshape(B, H // k, W // k, k * k * C)

        if self.mode == "project":
            x = self.proj(x)
        return x


class PatchEmbedAnyResolution(nn.Module):
    """Patch embedding supporting arbitrary-resolution images.

    Partitions the image using patch_size × patch_size patches.
    """

    def __init__(
        self,
        patch_size: int = 14,
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            pixel_values: (B, C, H, W)

        Returns:
            (B, H//p, W//p, D) where p = patch_size
        """
        x = self.proj(pixel_values)
        x = x.permute(0, 2, 3, 1).contiguous()
        return x


class AttentionPooling(nn.Module):
    """Attention-based pooling to compress token sequences.

    A lightweight substitute for CSA (Compressed Sparse Attention)
    from the TVP paper. Uses learnable query vectors to attend
    over the spatial token sequence.
    """

    def __init__(
        self,
        in_channels: int,
        num_queries: int = 81,
        num_heads: int = 8,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.randn(1, num_queries, in_channels) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, N, C) flattened spatial tokens.

        Returns:
            (B, num_queries, C) compressed representation.
        """
        B = x.shape[0]
        q = self.queries.expand(B, -1, -1)
        q = q + self.norm(q)
        out, _ = self.attn(q, x, x)
        return out


class MultiStageSpatialCompression(nn.Module):
    """Multi-stage spatial compression pipeline.

    Stage 1: SpatialCompression (3×3 kernel)
    Stage 2: Flatten + AttentionPooling (substitute for CSA)

    This mirrors the TVP paper's two-stage compression:
    ViT tokens → 3×3 spatial → CSA → compressed KV
    """

    def __init__(
        self,
        in_channels: int,
        compressed_channels: Optional[int] = None,
        kernel_size: int = 3,
        num_pool_queries: int = 81,
        num_heads: int = 8,
        mode: str = "project",
    ):
        super().__init__()
        self.spatial_compress = SpatialCompression(
            in_channels=in_channels,
            out_channels=compressed_channels or in_channels,
            kernel_size=kernel_size,
            mode=mode,
        )
        self.attn_pool = AttentionPooling(
            in_channels=self.spatial_compress.out_channels,
            num_queries=num_pool_queries,
            num_heads=num_heads,
        )

    def forward(self, x: torch.Tensor, hw: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, N, C) flattened spatial tokens, or (B, H, W, C) grid tokens.
            hw: Optional (H, W) hint if input is flattened.

        Returns:
            (B, num_queries, C_out) compressed representation.
        """
        if x.dim() == 3 and hw is not None:
            H, W = hw
            x = x.reshape(x.shape[0], H, W, -1)
        elif x.dim() == 3:
            # Infer square-ish layout
            N = x.shape[1]
            side = int(math.sqrt(N))
            if side * side == N:
                x = x.reshape(x.shape[0], side, side, -1)
            else:
                # Cannot infer grid; skip spatial compression, use pooling only
                return self.attn_pool(x)

        x = self.spatial_compress(x)
        B, H2, W2, C = x.shape
        x = x.reshape(B, H2 * W2, C)
        x = self.attn_pool(x)
        return x
