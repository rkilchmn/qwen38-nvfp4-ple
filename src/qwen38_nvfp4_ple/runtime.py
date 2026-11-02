"""Runtime-only packed embedding lookup independent of vLLM internals.

Baseline design (recreated from FINDINGS.md):
- PLE is fully on SSD: 128 NVFP4-quantized shards (~28.8 GB),
  geometry rows=2,500,012, packed_w=80, dim=160.
- Every shard is mmap'd zero-copy (MADV_RANDOM) inside the PleOffloadWorker
  process. No anonymous copies are ever made: only the touched rows of the
  mapping are gathered per lookup and dequantized (NVFP4: uint8-e2m1 values,
  fp8-e4m3 block scales (columns // 16 per row), one fp32 global scale/shard).
- resident RAM = page cache of touched pages (~MB), plus a minimal
  weight_scale_2 buffer (split_parts floats) filled by load_weights.
"""

from __future__ import annotations

import glob
import json
import mmap
import os
import struct
from typing import Dict

import torch
from torch import nn

from .format import GROUP_SIZE, dequantize_nvfp4

FORMAT_NAME = "qwen38-nvfp4-ple-v1"

_MODEL_DIRS = (
    os.environ.get("VLLM_PLE_MODEL_DIR", ""),
    os.environ.get("MOE_SSD_MODEL_DIR", ""),
    os.environ.get("VLLM_PLE_OFFLOAD_MODEL_DIR", ""),
)


def _resolve_shard_files() -> list[str]:
    """Resolve the 128 PLE shard paths from the model directory."""
    for d in _MODEL_DIRS:
        if not d:
            continue
        base = os.path.expanduser(d)
        files = sorted(glob.glob(os.path.join(base, "ple-nvfp4-*-of-*.safetensors")))
        if files:
            files.sort(
                key=lambda p: int(
                    os.path.basename(p).split("-")[2]
                )
            )
            return files
    # Fallback: search under any *_NVFP4* model root next to the workdir.
    # Group per directory, dedupe by shard index, pick the directory with the
    # most unique shards (never merge multiple directories).
    best: dict[int, str] = {}
    for pattern in (
        "/home/roger/models/*NVFP4*/ple-nvfp4-*-of-*.safetensors",
    ):
        per_dir: dict[str, dict[int, str]] = {}
        for p in sorted(glob.glob(pattern)):
            name = os.path.basename(p)
            try:
                shard = int(name.split("-")[2])
            except ValueError:
                continue
            per_dir.setdefault(os.path.dirname(p), {})[shard] = p
        for mapping in per_dir.values():
            if len(mapping) > len(best):
                best = mapping
    return [best[k] for k in sorted(best)]


class _MappedShard:
    """One zero-copy mmap of a Packed NVFP4 PLE shard.

    Exposure:
      packed (rows, packed_w)   uint8  -- two 4-bit values per byte
      scales (rows, n_blocks)   float8_e4m3fn -- per 16-column block
      global_scale              float32 scalar for this shard
    Views are built straight over the mapped file (no copies); MADV_RANDOM
    keeps read-ahead off so each row read faults exactly one 4 KB page.
    """

    _DT = {
        "U8": torch.uint8,
        "F32": torch.float32,
        "F16": torch.float16,
        "BF16": torch.bfloat16,
    }

    def __init__(self, path: str) -> None:
        self.path = path
        self._fd = os.open(path, os.O_RDONLY)
        size = os.fstat(self._fd).st_size
        self._mmap = mmap.mmap(self._fd, size, access=mmap.ACCESS_READ, flags=mmap.MAP_SHARED)
        try:
            self._mmap.madvise(mmap.MADV_RANDOM, 0, size)
        except (AttributeError, OSError):
            pass  # older python: read-ahead stays on, still functional

        # safetensors header: 8-byte LE length + JSON, then the data section.
        (header_len,) = struct.unpack("<Q", self._mmap[:8])
        header = json.loads(self._mmap[8 : 8 + header_len])
        data_start = 8 + header_len

        views: Dict[str, torch.Tensor] = {}
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            dtype_str = meta["dtype"]
            shape = tuple(meta["shape"])
            start, end = meta["data_offsets"]
            off = data_start + start
            nbytes = end - start
            # Zero-copy: a memoryview slice of the mapping, handed to torch.
            mv = memoryview(self._mmap)[off : off + nbytes]
            if dtype_str == "F8_E4M3":
                # torch.frombuffer has no fp8 dtype; map via uint8 then cast view.
                raw = torch.frombuffer(mv, dtype=torch.uint8)
                tensor = raw.view(torch.float8_e4m3fn)
            else:
                t = self._DT.get(dtype_str)
                if t is None:
                    raise ValueError(f"Unsupported PLE dtype {dtype_str} in {path}")
                tensor = torch.frombuffer(mv, dtype=t)
            views[key.rsplit(".", 1)[-1]] = tensor.reshape(shape)

        self.packed: torch.Tensor = views["weight"].view(-1, views["weight"].shape[-1])
        self.scales: torch.Tensor = views["weight_scale"].view(
            self.packed.shape[0], views["weight_scale"].numel() // self.packed.shape[0]
        )
        gs = views["weight_scale_2"]
        self.global_scale: float = float(gs.reshape(-1)[0].item())
        self.rows = self.packed.shape[0]

    def close(self) -> None:
        try:
            self._mmap.close()
        except Exception:
            pass
        try:
            os.close(self._fd)
        except Exception:
            pass

    def __del__(self) -> None:
        self.close()


class PackedNvfp4Embedding(nn.Module):
    """CPU-resident NVFP4 PLE table streamed from SSD.

    Baseline interface: (num_embeddings, embedding_dim, split_parts).
    All shard data lives on SSD and is mmap'd zero-copy; the only CPU-resident
    state is the minimal ``weight_scale_2`` buffer (split_parts floats).
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        split_parts: int,
        *,
        output_dtype: torch.dtype = torch.bfloat16,
        open_mappings: bool = False,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.split_parts = split_parts
        self.output_dtype = output_dtype
        self.shard_size = max(1, num_embeddings // split_parts)

        # Zero-size packed/scale buffers: rows are dequantized on the fly
        # from the mmap views, never materialized.
        self.register_buffer("weight", torch.empty(0, dtype=torch.uint8))
        self.register_buffer("weight_scale", torch.empty(0, dtype=torch.float8_e4m3fn))
        # Minimal buffer: one fp32 global decode scale per shard.
        self.register_buffer("weight_scale_2", torch.zeros(split_parts, dtype=torch.float32))

        self.entries: Dict[int, _MappedShard] = {}
        if open_mappings:
            for sid, path in enumerate(_resolve_shard_files()):
                self.entries[sid] = _MappedShard(path)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """Gather + dequantize only the touched rows (NVFP4 on-demand)."""
        shape = tuple(indices.shape)
        flat = indices.reshape(-1).to(dtype=torch.long).cpu()
        out = torch.empty(
            flat.shape[0], self.embedding_dim, dtype=self.output_dtype
        )
        if not self.entries:
            out.zero_()
            return out.reshape(*shape, self.embedding_dim)

        shard_of_row = torch.div(flat, self.shard_size, rounding_mode="floor")
        for sid in shard_of_row.unique().tolist():
            entry = self.entries.get(int(sid))
            if entry is None:
                continue
            mask = shard_of_row == sid
            local = (flat[mask] - int(sid) * self.shard_size).clamp_(0, entry.rows - 1)
            rows = entry.packed.index_select(0, local)
            scales = entry.scales.index_select(0, local)
            out[mask] = dequantize_nvfp4(
                rows,
                scales,
                self.weight_scale_2[sid].item() if self.weight_scale_2[sid] > 0 else entry.global_scale,
                output_dtype=self.output_dtype,
            )
        return out.reshape(*shape, self.embedding_dim)
