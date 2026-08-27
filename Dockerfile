# Keep this digest synchronized with the compatibility matrix in README.md.
FROM vllm/vllm-openai:qwen38-flash-next@sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8

LABEL org.opencontainers.image.title="Qwen3.8 NVFP4 PLE for vLLM"
LABEL org.opencontainers.image.description="Pinned Qwen4-preview vLLM runtime with on-demand NVFP4 PLE lookup"
LABEL org.opencontainers.image.licenses="Apache-2.0"

WORKDIR /opt/qwen38-nvfp4-ple
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python3 -m pip install --no-cache-dir --no-deps .

ENV VLLM_PLE_CPU_OFFLOAD=1 \
    VLLM_PLE_OFFLOAD_READY_TIMEOUT=900 \
    VLLM_PLUGINS=qwen38_nvfp4_ple

EXPOSE 8000

# Preserve the vllm/vllm-openai interface. Pass MODEL [vllm serve options]
# after the image name.
ENTRYPOINT ["vllm", "serve"]
