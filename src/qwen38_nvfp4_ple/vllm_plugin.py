"""vLLM general plugin for Qwen3.8 NVFP4 PLE CPU lookup."""

from __future__ import annotations

from collections.abc import Iterable
import importlib
import logging
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

            # The official constructor computes the hashes and logical table
            # dimensions. Meta allocation avoids ever creating its BF16 table.
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

            # Persistent hash buffers are populated by load_weights. Workspace
            # buffers are deterministic and rebuilt here on real CPU storage.
            for name in (
                "layer_multipliers",
                "ngram_heads_vocab_sizes",
                "ngram_heads_offsets",
            ):
                meta_buffer = getattr(self, name)
                self._buffers[name] = torch.empty(
                    tuple(meta_buffer.shape), dtype=meta_buffer.dtype, device="cpu"
                )
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
            result = super().forward_impl(
                hidden_states,
                input_ids,
                query_start_loc,
                ngram_context,
                output_buffer=None,
            )
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
            # process below loads and decodes every packed table component.
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
                start = shard * embedding.shard_size
                end = min(start + embedding.shard_size, embedding.num_embeddings)
                if component == "weight":
                    target = embedding.weight[start:end]
                    expected_dtype = torch.uint8
                elif component == "weight_scale":
                    target = embedding.weight_scale[start:end]
                    expected_dtype = torch.float8_e4m3fn
                else:
                    target = embedding.weight_scale_2[shard]
                    expected_dtype = torch.float32
                expected_shape = tuple(target.shape)
                if tuple(tensor.shape) != expected_shape or tensor.dtype != expected_dtype:
                    raise ValueError(
                        f"Invalid {name}: expected {expected_dtype} {expected_shape}, "
                        f"got {tensor.dtype} {tuple(tensor.shape)}"
                    )
                target.copy_(tensor.to(device="cpu"))
                loaded_components[component].add(shard)
                loaded.add(f"ngram_embedding.{component}")

            expected_shards = set(range(embedding.split_parts))
            missing = {
                component: sorted(expected_shards - present)
                for component, present in loaded_components.items()
                if present != expected_shards
            }
            if missing:
                raise RuntimeError(f"Packed PLE checkpoint is incomplete: {missing}")
            return loaded

    Nvfp4NGramEmbedding.__name__ = "Qwen3_8FlashNextNvfp4NGramEmbedding"
    Nvfp4NGramEmbedding.__qualname__ = Nvfp4NGramEmbedding.__name__
    Nvfp4NGramEmbedding.__module__ = __name__
    return Nvfp4NGramEmbedding


def _ple_offload_proc_main(*args: Any, **kwargs: Any) -> None:
    """Register the plugin inside vLLM's special PLE subprocess."""

    worker_module = importlib.import_module("vllm.v1.ple_offload.worker")
    worker_class = worker_module.PleOffloadWorker
    original = getattr(worker_class, _PROCESS_PATCH_MARKER, worker_class.proc_main)
    register()
    original(*args, **kwargs)


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
    _patch_ple_process_entrypoint()
