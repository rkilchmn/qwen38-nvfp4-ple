"""vLLM general plugin for Qwen3.8 NVFP4 PLE CPU lookup."""
from __future__ import annotations

from collections.abc import Iterable
import importlib
import logging
import os
import re
from typing import Any

import torch

from .format import FORMAT_NAME, Nvfp4PleConfig
from .runtime import PackedNvfp4Embedding

LOGGER = logging.getLogger(__name__)
_PATCH_MARKER = "_qwen38_nvfp4_ple_plugin_v1"
_PROCESS_PATCH_MARKER = "_qwen38_nvfp4_ple_original_proc_main"
_PACKED_SHARD_RE = re.compile(
    r"^ngram_embedding\.shard_(?P<shard>\d+)\."
    r"(?P<component>weight|weight_scale|weight_scale_2)$"
)


def _recompute_hash_buffers(
    module: Any, config: Any, ple_dense_layer_id: int
) -> dict[str, torch.Tensor]:
    """Recompute the PLE hash-table buffers on CPU with real values.

    The original constructor computes ``layer_multipliers`` (splitmix64 of a
    per-layer seed) and ``ngram_heads_vocab_sizes``/``ngram_heads_offsets``
    (nth-prime table strides) at __init__ time. The plugin builds the base
    class under ``torch.device("meta")``, so those arithmetic results carry no
    values; without recomputation the buffers are zeros and the first
    forward_impl raises ``torch.remainder`` ZeroDivisionError. These functions
    are deterministic and cheap, so mirror them exactly.
    """
    splitmix64 = module._splitmix64
    nth_prime_after = module._nth_prime_after
    gamma = module._SPLITMIX_GAMMA
    layer_prime = module._PLE_LAYER_PRIME

    ngram_size = int(config.ngram_size)
    heads_per_ngram = int(config.heads_per_ngram)
    ngram_heads = (ngram_size - 1) * heads_per_ngram
    unigram_vocab_size = int(config.vocab_size)

    max_multiplier = ((1 << 63) - 1) // unigram_vocab_size
    half_bound = max(1, max_multiplier // 2)
    seed = int(getattr(config, "seed", 1234))
    base_seed = seed + layer_prime * ple_dense_layer_id
    multipliers = [
        2 * (splitmix64(base_seed + gamma * (index + 1)) % half_bound) + 1
        for index in range(ngram_size)
    ]

    sizes: list[int] = []
    offsets: list[int] = []
    offset = 0
    for local_head in range(ngram_heads):
        global_head = ple_dense_layer_id * ngram_heads + local_head
        size = nth_prime_after(int(config.ngram_vocab_size_base) - 1, global_head + 1)
        sizes.append(size)
        offsets.append(offset)
        offset += size

    return {
        "layer_multipliers": torch.tensor(multipliers, dtype=torch.long),
        "ngram_heads_vocab_sizes": torch.tensor(sizes, dtype=torch.long),
        "ngram_heads_offsets": torch.tensor(offsets, dtype=torch.long),
    }


def _make_embedding_class(module: Any) -> type:
    original = module.Qwen3_8FlashNextNGramEmbedding
    envs = importlib.import_module("vllm.envs")
    offload_module = importlib.import_module(
        "vllm.model_executor.layers.ple_offload_layer"
    )

    class Nvfp4NGramEmbedding(original):
        """Drop-in Qwen PLE layer with packed CPU-owned table storage."""

        _qwen38_nvfp4_ple_plugin_v1 = True

        def __init__(
            self,
            config: Any,
            embedding_dim: int,
            ple_dense_layer_id: int,
            max_total_tokens: int,
            max_num_reqs: int,
            prefix: str,
            quant_config: Any = None,
            params_dtype: torch.dtype | None = None,
        ) -> None:
            metadata = getattr(config, "ple_quantization_config", None)
            self._nvfp4_ple_enabled = metadata is not None
            if not self._nvfp4_ple_enabled:
                super().__init__(
                    config,
                    embedding_dim,
                    ple_dense_layer_id,
                    max_total_tokens,
                    max_num_reqs,
                    prefix,
                    quant_config=quant_config,
                    params_dtype=params_dtype,
                )
                return

            parsed = Nvfp4PleConfig.from_mapping(metadata)
            if parsed.format != FORMAT_NAME:
                raise ValueError(f"Unsupported NVFP4 PLE format: {parsed.format}")
            if not envs.VLLM_PLE_CPU_OFFLOAD:
                raise RuntimeError(
                    f"{FORMAT_NAME} requires VLLM_PLE_CPU_OFFLOAD=1; "
                    "the packed table is a CPU lookup format"
                )
            output_dtype = params_dtype or torch.bfloat16
            if output_dtype != torch.bfloat16:
                raise ValueError(
                    f"{FORMAT_NAME} was evaluated with BF16 lookup output; "
                    f"got model dtype {output_dtype}"
                )

            with torch.device("meta"):
                super().__init__(
                    config,
                    embedding_dim,
                    ple_dense_layer_id,
                    max_total_tokens,
                    max_num_reqs,
                    prefix,
                    quant_config=quant_config,
                    params_dtype=params_dtype,
                )
            original_embedding = self.ngram_embedding
            num_embeddings = int(original_embedding.org_vocab_size)
            head_dim = int(original_embedding.embedding_dim)
            split_parts = int(self.split_ngram_parts)

            recomputed = _recompute_hash_buffers(module, config, ple_dense_layer_id)
            for name, meta_buffer in (
                (n, getattr(self, n))
                for n in (
                    "layer_multipliers",
                    "ngram_heads_vocab_sizes",
                    "ngram_heads_offsets",
                )
            ):
                real = recomputed[name]
                if tuple(real.shape) != tuple(meta_buffer.shape):
                    raise ValueError(
                        f"{name}: recomputed shape {tuple(real.shape)} != "
                        f"expected {tuple(meta_buffer.shape)}"
                    )
                self._buffers[name] = real
            self._buffers["positions_buffer"] = torch.arange(
                max_total_tokens, dtype=torch.int64, device="cpu"
            )
            self._buffers["padded_buffer"] = torch.full(
                (max_num_reqs, max_total_tokens),
                self.eos_token_id,
                dtype=torch.int64,
                device="cpu",
            )
            with torch.device("cpu"):
                self.ngram_embedding = PackedNvfp4Embedding(
                    num_embeddings,
                    head_dim,
                    split_parts,
                    output_dtype=output_dtype,
                    # mmap views are opened only in the offload subprocess; GPU
                    # workers keep an empty placeholder (streaming lives there).
                    open_mappings=offload_module.is_offload_process(),
                )
            self._nvfp4_output_dtype = output_dtype

        def forward_impl(
            self,
            hidden_states: torch.Tensor,
            input_ids: torch.Tensor,
            query_start_loc: torch.Tensor,
            ngram_context: torch.Tensor,
            output_buffer: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if not getattr(self, "_nvfp4_ple_enabled", False):
                return super().forward_impl(
                    hidden_states,
                    input_ids,
                    query_start_loc,
                    ngram_context,
                    output_buffer=output_buffer,
                )
            if os.environ.get("VLLM_PLE_ZERO"):
                # Diagnostic: zero the PLE contribution (isolates core model).
                if output_buffer is not None:
                    output_buffer.zero_()
                    return output_buffer
                result = super().forward_impl(
                    hidden_states,
                    input_ids,
                    query_start_loc,
                    ngram_context,
                    output_buffer=None,
                )
                return torch.zeros_like(result)
            result = super().forward_impl(
                hidden_states,
                input_ids,
                query_start_loc,
                ngram_context,
                output_buffer=None,
            )
            if os.environ.get("VLLM_PLE_DEBUG") and not getattr(
                Nvfp4NGramEmbedding, "_dbg_logged", False
            ):
                LOGGER.info(
                    "PLE forward_impl serving: rows=%d abs_sum=%.6e out[0,0]=%.6e "
                    "is_offload=%s",
                    result.shape[0],
                    float(result.float().abs().sum()),
                    float(result.float()[0, 0]) if result.numel() else 0.0,
                    offload_module.is_offload_process(),
                )
                Nvfp4NGramEmbedding._dbg_logged = True
            if output_buffer is None:
                return result
            output = output_buffer[: result.shape[0], : result.shape[1]]
            output.copy_(result)
            return output

        def get_offload_output_dtype(self, default_dtype: torch.dtype) -> torch.dtype:
            if getattr(self, "_nvfp4_ple_enabled", False):
                return self._nvfp4_output_dtype
            return super().get_offload_output_dtype(default_dtype)

        def load_weights(
            self, weights: Iterable[tuple[str, torch.Tensor]]
        ) -> set[str]:
            items = list(weights)
            packed_checkpoint = getattr(self, "_nvfp4_ple_enabled", False) or any(
                name.endswith(".weight_scale_2") for name, _tensor in items
            )
            if not packed_checkpoint:
                return super().load_weights(items)

            # GPU workers own only the IPC placeholder. The dedicated CPU
            # process below loads the table metadata; row data streams from SSD.
            if envs.VLLM_PLE_CPU_OFFLOAD and not offload_module.is_offload_process():
                return {name for name, _tensor in items}
            if not getattr(self, "_nvfp4_ple_enabled", False):
                raise RuntimeError("Packed PLE weights reached an unconfigured layer")

            embedding = self.ngram_embedding
            persistent = {
                "layer_multipliers": self.layer_multipliers,
                "ngram_heads_offsets": self.ngram_heads_offsets,
                "ngram_heads_vocab_sizes": self.ngram_heads_vocab_sizes,
            }
            loaded_components: dict[str, set[int]] = {
                "weight": set(),
                "weight_scale": set(),
                "weight_scale_2": set(),
            }
            loaded: set[str] = set()
            for name, tensor in items:
                if name in persistent:
                    target = persistent[name]
                    if tuple(tensor.shape) != tuple(target.shape):
                        raise ValueError(
                            f"Shape mismatch for {name}: expected {tuple(target.shape)}, "
                            f"got {tuple(tensor.shape)}"
                        )
                    target.copy_(tensor.to(device="cpu", dtype=target.dtype))
                    loaded.add(name)
                    continue
                match = _PACKED_SHARD_RE.match(name)
                if match is None:
                    leaf = name.rsplit(".", 1)[-1]
                    if leaf.startswith("hashstats_") or leaf == "token_lookup":
                        continue
                    raise ValueError(f"Unexpected packed PLE tensor: {name}")
                shard = int(match.group("shard"))
                component = match.group("component")
                if not 0 <= shard < embedding.split_parts:
                    raise ValueError(f"PLE shard index is out of range: {shard}")
                # STREAMING baseline: packed rows/block-scales stay on SSD and
                # stream through the mmap views. Only the tiny per-shard global
                # scale (the "minimal buffer") is resident.
                if component == "weight_scale_2":
                    embedding.weight_scale_2[shard] = float(
                        tensor.to(device="cpu", dtype=torch.float32).reshape(-1)[0]
                    )
                loaded_components[component].add(shard)
                loaded.add(f"ngram_embedding.{component}")

            expected_shards = set(range(embedding.split_parts))
            missing = {
                component: sorted(expected_shards - present)
                for component, present in loaded_components.items()
                if present != expected_shards
            }
            if missing:
                LOGGER.info(
                    "PLE streaming: partial shard metadata %s (rows stream from SSD)",
                    {k: len(v) for k, v in missing.items()},
                )
            return loaded

    Nvfp4NGramEmbedding.__name__ = "Qwen3_8FlashNextNvfp4NGramEmbedding"
    Nvfp4NGramEmbedding.__qualname__ = Nvfp4NGramEmbedding.__name__
    Nvfp4NGramEmbedding.__module__ = __name__
    return Nvfp4NGramEmbedding


def _ple_offload_proc_main(*args: Any, **kwargs: Any) -> None:
    """Register the streaming PLE class inside the offload subprocess.

    This patch is *required* (not debugging): the subprocess is spawned fresh,
    so without register() it builds the ORIGINAL embedding class, whose
    constructor materializes the full BF16 table (num_embeddings x dim x 2 B
    = ~102 GB here) and OOMs. register() swaps in the streaming class first.
    """
    import traceback
    try:
        logging.getLogger(__name__).info(
            "PLE offload worker: registering streaming plugin..."
        )
        register()
        worker_module = importlib.import_module("vllm.v1.ple_offload.worker")
        original = getattr(
            worker_module.PleOffloadWorker, _PROCESS_PATCH_MARKER,
            worker_module.PleOffloadWorker.proc_main,
        )
        original(*args, **kwargs)
    except Exception as e:
        logging.getLogger(__name__).error(
            "PLE offload worker crashed: %s\n%s", e, traceback.format_exc()
        )
        raise


def _patch_ple_process_entrypoint() -> None:
    worker_module = importlib.import_module("vllm.v1.ple_offload.worker")
    worker_class = worker_module.PleOffloadWorker
    if getattr(worker_class, _PROCESS_PATCH_MARKER, None) is not None:
        return
    setattr(worker_class, _PROCESS_PATCH_MARKER, worker_class.proc_main)
    worker_class.proc_main = staticmethod(_ple_offload_proc_main)


def register() -> None:
    """Install idempotent Qwen3.8 PLE hooks in this vLLM process."""

    module = importlib.import_module("vllm.models.qwen3_8_flash_next.nvidia.ple_layer")
    current = module.Qwen3_8FlashNextNGramEmbedding
    if not getattr(current, _PATCH_MARKER, False):
        module.Qwen3_8FlashNextNGramEmbedding = _make_embedding_class(module)
        LOGGER.info("Registered %s vLLM PLE lookup support", FORMAT_NAME)
    # The offload subprocess must also build the streaming class (see
    # _ple_offload_proc_main), otherwise it materializes the full BF16 table.
    _patch_ple_process_entrypoint()
