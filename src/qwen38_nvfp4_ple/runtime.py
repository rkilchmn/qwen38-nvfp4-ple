"""Runtime-only packed embedding lookup independent of vLLM internals."""

from __future__ import annotations

import math

import torch
from torch import nn

from .format import GROUP_SIZE, dequantize_nvfp4


class PackedNvfp4Embedding(nn.Module):
    """CPU-resident NVFP4 table that dequantizes only selected rows."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        split_parts: int,
        *,
        output_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        if num_embeddings <= 0 or embedding_dim <= 0 or split_parts <= 0:
            raise ValueError("Embedding dimensions and split_parts must be positive")
        if embedding_dim % GROUP_SIZE:
            raise ValueError(
                f"Embedding width must be divisible by {GROUP_SIZE}, got {embedding_dim}"
            )
        self.num_embeddings = num_embeddings
        self.org_vocab_size = num_embeddings
        self.embedding_dim = embedding_dim
        self.split_parts = split_parts
        self.shard_size = math.ceil(num_embeddings / split_parts)
        self.output_dtype = output_dtype
        self.register_buffer(
            "weight",
            torch.empty((num_embeddings, embedding_dim // 2), dtype=torch.uint8),
        )
        self.register_buffer(
            "weight_scale",
            torch.empty(
                (num_embeddings, embedding_dim // GROUP_SIZE),
                dtype=torch.float8_e4m3fn,
            ),
        )
        self.register_buffer(
            "weight_scale_2", torch.empty(split_parts, dtype=torch.float32)
        )

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        shape = tuple(indices.shape)
        flat = indices.reshape(-1).to(device=self.weight.device, dtype=torch.long)
        packed = self.weight.index_select(0, flat)
        scales = self.weight_scale.index_select(0, flat)
        shard_ids = torch.div(flat, self.shard_size, rounding_mode="floor")
        shard_ids.clamp_max_(self.split_parts - 1)
        global_scales = self.weight_scale_2.index_select(0, shard_ids)
        values = dequantize_nvfp4(
            packed,
            scales,
            global_scales,
            output_dtype=self.output_dtype,
        )
        return values.reshape(*shape, self.embedding_dim)
