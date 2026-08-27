# Qwen3.8 NVFP4 PLE for vLLM

This project stores the 51.2-billion-parameter Qwen3.8 position-learning
enhancement (PLE) n-gram table in NVFP4. Its vLLM plugin decodes only the rows
selected by each request to BF16. The plugin does not change attention, MoE,
sampling, or scheduling.

The conversion changes the storage footprint as follows:

| Artifact | Exact bytes | GiB |
|---|---:|---:|
| Original BF16 PLE table | 102,400,491,520 | 95.368 |
| NVFP4 PLE table | 28,800,138,752 | 26.822 |
| Complete converted checkpoint | 109,138,439,672 | 101.643 |

This saves 68.546 GiB without REAP. The physical checkpoint and an end-to-end
vLLM generation path have been tested on NVIDIA GB10. Read
[Findings](docs/findings.md) before treating the alignment measurements as a
quality benchmark.

## Use with docker

The tested versions are:

| Component | Tested value |
|---|---|
| Base image | `vllm/vllm-openai:qwen38-flash-next@sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8` |
| vLLM | `0.1.dev20073+g8e685d198` |
| GPU | NVIDIA GB10, compute capability 12.1 |
| Serving mode | TP1, `mp` executor, PLE CPU offload, BF16 lookup output |

Build it locally:

```bash
cd qwen38-nvfp4-ple
docker build --tag qwen38-nvfp4-ple:local .
```

The standard vLLM-style entrypoint is available immediately:

```bash
docker run --rm --gpus all qwen38-nvfp4-ple:local --help
```

## Start with Compose

Start the published checkpoint with the single-DGX-Spark profile:

```bash
docker compose up --build --detach
```

Compose downloads
`provsalt/Qwen3.8-Flash-Next-NVFP4-PLE-NVFP4` into a persistent Docker volume.
The checkpoint is about 101.6 GiB so be patient.

```bash
docker compose logs --follow api
```

Compose marks the service healthy only after vLLM's `/health` endpoint returns
HTTP 200. Inspect the status and API:

```bash
docker compose ps
curl http://127.0.0.1:8000/v1/models
```

Stop the service:

```bash
docker compose down
```

The defaults can be changed with environment variables:

```bash
MODEL_ID=organization/another-packed-checkpoint \
PORT=8080 \
docker compose up --build --detach
```

To keep overrides between runs, place them in an untracked `.env` file:

```dotenv
MODEL_ID=provsalt/Qwen3.8-Flash-Next-NVFP4-PLE-NVFP4
BIND_ADDRESS=127.0.0.1
PORT=8000
VLLM_IMAGE=qwen38-nvfp4-ple:local
VLLM_PLE_OFFLOAD_READY_TIMEOUT=900
```

The Compose defaults reserve every NVIDIA GPU, use host IPC, and publish the
API only on localhost. The command follows the Qwen3.8-27B profile with a
262,144-token context, FP8 KV cache, Qwen reasoning and tool parsers, automatic
tool choice, data-parallel multimodal encoding, and three-token MTP. It also
uses the `mp` executor required by PLE CPU offload.

## Run without Compose

Pass the model first and then any normal `vllm serve` options. This is the
equivalent direct invocation for multimodal serving:

```bash
docker run --rm \
  --gpus all \
  --ipc host \
  --publish 127.0.0.1:8000:8000 \
  --volume qwen38-hf-cache:/root/.cache/huggingface \
  qwen38-nvfp4-ple:local \
  provsalt/Qwen3.8-Flash-Next-NVFP4-PLE-NVFP4 \
  --max-model-len 262144 \
  --tensor-parallel-size 1 \
  --kv-cache-dtype fp8 \
  --trust-remote-code \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_xml  \
  --enable-auto-tool-choice \
  --gpu-memory-utilization 0.9 \
  --mm-encoder-tp-mode data \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --distributed-executor-backend mp
```

## Convert a source checkpoint

Download the BF16-PLE source checkpoint:

```bash
hf download Inferact/Qwen3.8-Flash-Next-NVFP4 \
  --local-dir /models/Qwen3.8-Flash-Next-NVFP4
```

Preview the conversion without writing files:

```bash
docker run --rm \
  --entrypoint qwen38-nvfp4-ple \
  --volume /models:/models:ro \
  qwen38-nvfp4-ple:local \
  convert /models/Qwen3.8-Flash-Next-NVFP4 /unused --dry-run
```

Convert the PLE table:

```bash
docker run --rm \
  --user "$(id -u):$(id -g)" \
  --entrypoint qwen38-nvfp4-ple \
  --volume /models:/models \
  qwen38-nvfp4-ple:local \
  convert \
    /models/Qwen3.8-Flash-Next-NVFP4 \
    /models/Qwen3.8-Flash-Next-NVFP4-PLE-NVFP4
```

The converter is resumable. It validates completed shards before skipping
them. It hardlinks unchanged safetensors when the source and output share a
filesystem. Pass `--copy-unchanged` to create an independent copy.

Inspect the converted checkpoint:

```bash
docker run --rm \
  --entrypoint qwen38-nvfp4-ple \
  --volume /models:/models:ro \
  qwen38-nvfp4-ple:local \
  inspect /models/Qwen3.8-Flash-Next-NVFP4-PLE-NVFP4
```

## Why the dedicated image is required

Qwen3.8-Flash-Next is a Qwen4 architecture preview. The regular vLLM image and
PyPI release do not contain its model implementation. This Dockerfile pins the
official Qwen3.8 image digest used in testing.

Upstream vLLM computes n-gram IDs on CPU, owns the PLE table in a dedicated
process, and copies selected rows asynchronously into a GPU buffer. This plugin
replaces only the CPU table and row lookup:

```text
token IDs -> upstream n-gram hashing -> packed-row gather
          -> NVFP4 decode to BF16 -> upstream pinned buffer -> GPU
```

## Limits

- NVIDIA only. The plugin targets NVIDIA's PLE offload implementation.
- Packed checkpoints require PLE CPU offload.
- Format version 1 returns BF16 PLE rows.
- The converter requires separate BF16 PLE shard files.
- The CPU decoder uses ordinary PyTorch tensor operations.

The upstream requirements remain in the
[official vLLM Qwen3.8 recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next).
NVIDIA's
[NVFP4 documentation](https://nvidia.github.io/TransformerEngine/features/low_precision_training/nvfp4/nvfp4.html)
defines the E2M1 values, E4M3 block scales, and FP32 global scale used here.

## License

The source is Apache-2.0. Converted weights retain the source model's license.
No model weights are included in this repository.
