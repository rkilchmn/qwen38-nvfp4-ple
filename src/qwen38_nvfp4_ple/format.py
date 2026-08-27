"""Pure tensor operations for the Qwen3.8 NVFP4 PLE storage format."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

import torch


FORMAT_NAME = "qwen38-nvfp4-ple-v1"
GROUP_SIZE = 16
FP4_MAX = 6.0
FP8_MAX = 448.0
PACKING = "low-nibble-first"

# Magnitude codes 0..7 for NVIDIA E2M1. Bit 3 is the sign bit.
E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


@dataclass(frozen=True)
class Nvfp4PleConfig:
    """Metadata stored under ``text_config.ple_quantization_config``."""

    format: str = FORMAT_NAME
    group_size: int = GROUP_SIZE
    value_dtype: str = "e2m1"
    packed_dtype: str = "uint8"
    block_scale_dtype: str = "float8_e4m3fn"
    global_scale_dtype: str = "float32"
    output_dtype: str = "bfloat16"
    packing: str = PACKING

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Nvfp4PleConfig":
        if not isinstance(value, Mapping):
            raise ValueError("PLE quantization metadata must be an object")
        try:
            config = cls(**dict(value))
        except TypeError as exc:
            raise ValueError(f"Invalid PLE quantization metadata: {exc}") from exc
        config.validate()
        return config

    def validate(self) -> None:
        expected = Nvfp4PleConfig()
        for field, expected_value in asdict(expected).items():
            actual = getattr(self, field)
            if actual != expected_value:
                raise ValueError(
                    f"Unsupported PLE quantization {field}: "
                    f"expected {expected_value!r}, got {actual!r}"
                )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_matrix(weight: torch.Tensor) -> tuple[int, int]:
    if weight.ndim != 2 or not weight.is_floating_point():
        raise ValueError("NVFP4 quantization expects a 2-D floating tensor")
    rows, columns = weight.shape
    if columns <= 0 or columns % GROUP_SIZE:
        raise ValueError(
            f"NVFP4 width must be divisible by {GROUP_SIZE}, got {columns}"
        )
    return rows, columns


def _encode_e2m1(values: torch.Tensor) -> torch.Tensor:
    """Round normalized values with NVIDIA's deterministic E2M1 thresholds."""

    magnitude = values.abs()
    codes = torch.zeros_like(magnitude, dtype=torch.uint8)
    codes = torch.where(magnitude > 0.25, 1, codes)
    codes = torch.where(magnitude >= 0.75, 2, codes)
    codes = torch.where(magnitude > 1.25, 3, codes)
    codes = torch.where(magnitude >= 1.75, 4, codes)
    codes = torch.where(magnitude > 2.5, 5, codes)
    codes = torch.where(magnitude >= 3.5, 6, codes)
    codes = torch.where(magnitude > 5.0, 7, codes)
    sign = torch.signbit(values).to(torch.uint8) << 3
    return codes | sign


def quantize_nvfp4(
    weight: torch.Tensor,
    *,
    global_amax: float | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack a row-major matrix as E2M1 plus block and global decode scales.

    The last dimension is divided into groups of 16. Two E2M1 values are
    packed in each byte; the first value occupies the low nibble. The returned
    global scale is the *decode* scale ``amax / (448 * 6)``.
    """

    rows, columns = _validate_matrix(weight)
    values = weight.float()
    if global_amax is None:
        amax = values.abs().amax().to(dtype=torch.float32)
    else:
        amax = torch.as_tensor(global_amax, dtype=torch.float32, device=values.device)
        if amax.numel() != 1 or not torch.isfinite(amax) or float(amax) < 0:
            raise ValueError(f"Invalid global amax: {global_amax!r}")

    if float(amax) == 0.0:
        packed = torch.zeros((rows, columns // 2), dtype=torch.uint8, device=values.device)
        block_scales = torch.zeros(
            (rows, columns // GROUP_SIZE),
            dtype=torch.float8_e4m3fn,
            device=values.device,
        )
        return packed, block_scales, torch.zeros((), dtype=torch.float32, device=values.device)

    encode_scale = (FP8_MAX * FP4_MAX) / amax
    decode_scale = amax / (FP8_MAX * FP4_MAX)
    groups = values.reshape(rows, columns // GROUP_SIZE, GROUP_SIZE)
    block_amax = groups.abs().amax(dim=-1)
    block_scales = ((block_amax / FP4_MAX) * encode_scale).to(torch.float8_e4m3fn)
    block_scales_f32 = block_scales.float()
    normalized = torch.where(
        block_scales_f32.unsqueeze(-1) > 0,
        groups * encode_scale / block_scales_f32.unsqueeze(-1),
        torch.zeros_like(groups),
    ).reshape(rows, columns)
    codes = _encode_e2m1(normalized)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return packed.contiguous(), block_scales.contiguous(), decode_scale.reshape(())


def dequantize_nvfp4(
    packed: torch.Tensor,
    block_scales: torch.Tensor,
    global_scale: torch.Tensor | float,
    *,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Decode packed PLE rows without materializing the full BF16 table."""

    if packed.ndim != 2 or packed.dtype != torch.uint8:
        raise ValueError("Packed NVFP4 values must be a 2-D uint8 tensor")
    rows, packed_columns = packed.shape
    columns = packed_columns * 2
    expected_scale_shape = (rows, columns // GROUP_SIZE)
    if tuple(block_scales.shape) != expected_scale_shape:
        raise ValueError(
            f"Expected block scales shaped {expected_scale_shape}, "
            f"got {tuple(block_scales.shape)}"
        )
    if block_scales.dtype != torch.float8_e4m3fn:
        raise ValueError(
            "NVFP4 block scales must use torch.float8_e4m3fn, "
            f"got {block_scales.dtype}"
        )
    scale2 = torch.as_tensor(global_scale, dtype=torch.float32, device=packed.device)
    if scale2.numel() not in (1, rows):
        raise ValueError(
            "NVFP4 global decode scale must be scalar or have one value per row"
        )
    if not bool(torch.isfinite(scale2).all()) or bool((scale2 < 0).any()):
        raise ValueError("NVFP4 global decode scales must be finite and nonnegative")

    low = packed & 0x0F
    high = packed >> 4
    codes = torch.empty((rows, columns), dtype=torch.uint8, device=packed.device)
    codes[:, 0::2] = low
    codes[:, 1::2] = high
    lookup = torch.tensor(E2M1_MAGNITUDES, dtype=torch.float32, device=packed.device)
    values = lookup[(codes & 0x07).long()]
    values = torch.where((codes & 0x08) != 0, -values, values)
    values = values.reshape(rows, columns // GROUP_SIZE, GROUP_SIZE)
    row_scale = scale2.reshape(-1, 1, 1) if scale2.numel() == rows else scale2
    values = values * block_scales.float().unsqueeze(-1) * row_scale
    return values.reshape(rows, columns).to(output_dtype)


def nvfp4_storage_bytes(rows: int, columns: int, shard_count: int = 1) -> int:
    """Return the exact tensor payload size for this row-wise PLE format."""

    if rows < 0 or columns <= 0 or columns % GROUP_SIZE or shard_count <= 0:
        raise ValueError("Invalid NVFP4 storage dimensions")
    values = rows * columns
    return math.ceil(values / 2) + math.ceil(values / GROUP_SIZE) + 4 * shard_count
