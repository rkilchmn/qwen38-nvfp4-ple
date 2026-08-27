"""Streaming conversion and inspection for Qwen3.8 PLE checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
from typing import Any, Callable, Iterable

from safetensors import safe_open
from safetensors.torch import save_file
import torch

from .format import (
    FORMAT_NAME,
    GROUP_SIZE,
    Nvfp4PleConfig,
    nvfp4_storage_bytes,
    quantize_nvfp4,
)


LOGGER = logging.getLogger(__name__)
INDEX_NAME = "model.safetensors.index.json"
CONFIG_NAME = "config.json"
STATE_NAME = ".nvfp4-ple-conversion.json"
STATE_FORMAT = "qwen38-nvfp4-ple-conversion-v1"
PLE_WEIGHT_RE = re.compile(
    r"^(?P<prefix>.+\.ple\.ple_embedding\.ngram_embedding)"
    r"\.shard_(?P<shard>\d+)\.weight$"
)


@dataclass(frozen=True)
class PleShard:
    number: int
    key: str
    filename: str
    rows: int
    columns: int

    @property
    def stem(self) -> str:
        return self.key[: -len(".weight")]

    @property
    def scale_key(self) -> str:
        return f"{self.stem}.weight_scale"

    @property
    def global_scale_key(self) -> str:
        return f"{self.stem}.weight_scale_2"


@dataclass(frozen=True)
class SourceCheckpoint:
    directory: Path
    config: dict[str, Any]
    index: dict[str, Any]
    index_sha256: str
    prefix: str
    shards: tuple[PleShard, ...]

    @property
    def rows(self) -> int:
        return sum(shard.rows for shard in self.shards)

    @property
    def columns(self) -> int:
        return self.shards[0].columns

    @property
    def bf16_payload_bytes(self) -> int:
        return self.rows * self.columns * 2

    @property
    def nvfp4_payload_bytes(self) -> int:
        return nvfp4_storage_bytes(self.rows, self.columns, len(self.shards))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing required file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_config(config: dict[str, Any]) -> dict[str, Any]:
    value = config.get("text_config", config)
    if not isinstance(value, dict):
        raise ValueError("config.json text_config must be an object")
    return value


def inspect_source(directory: str | Path) -> SourceCheckpoint:
    """Validate a BF16-sharded Qwen3.8 PLE source without loading its table."""

    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Checkpoint directory does not exist: {root}")
    config = _read_json(root / CONFIG_NAME)
    text_config = _text_config(config)
    if text_config.get("ple_quantization_config") is not None:
        raise ValueError("Source checkpoint already declares quantized PLE storage")
    if config.get("model_type") != "qwen4_exp":
        raise ValueError(
            "Expected a Qwen3.8/Qwen4Exp checkpoint with model_type='qwen4_exp', "
            f"got {config.get('model_type')!r}"
        )

    index_path = root / INDEX_NAME
    index = _read_json(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"{INDEX_NAME} has no weight_map object")

    matches: list[tuple[int, str, str, str]] = []
    for key, filename in weight_map.items():
        match = PLE_WEIGHT_RE.match(key)
        if match:
            matches.append((int(match.group("shard")), key, str(filename), match.group("prefix")))
    matches.sort(key=lambda item: item[0])
    if not matches:
        raise ValueError("Checkpoint contains no split Qwen3.8 PLE embedding weights")
    prefixes = {item[3] for item in matches}
    if len(prefixes) != 1:
        raise ValueError(f"Expected one PLE table, found prefixes: {sorted(prefixes)}")
    actual_numbers = [item[0] for item in matches]
    if actual_numbers != list(range(len(matches))):
        raise ValueError(
            "PLE shard numbers must be contiguous from zero; "
            f"found {actual_numbers[:16]} ({len(actual_numbers)} total)"
        )
    configured_parts = int(text_config.get("split_ngram_parts", len(matches)))
    if configured_parts != len(matches):
        raise ValueError(
            f"config split_ngram_parts={configured_parts}, but index has {len(matches)}"
        )

    ple_keys = {item[1] for item in matches}
    ple_files = {item[2] for item in matches}
    for filename in ple_files:
        indexed_keys = {key for key, mapped in weight_map.items() if mapped == filename}
        unexpected = sorted(indexed_keys - ple_keys)
        if unexpected:
            raise ValueError(
                "PLE source files must not mix table and backbone tensors. "
                f"{filename} also contains {unexpected[:5]}"
            )

    shards: list[PleShard] = []
    width: int | None = None
    for number, key, filename, _prefix in matches:
        shard_path = root / filename
        if not shard_path.is_file():
            raise FileNotFoundError(f"Missing PLE safetensors file: {shard_path}")
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            tensor_slice = handle.get_slice(key)
            shape = tuple(tensor_slice.get_shape())
            dtype = tensor_slice.get_dtype()
        if len(shape) != 2 or dtype != "BF16":
            raise ValueError(f"Expected {key} to be a BF16 matrix, got {dtype} {shape}")
        rows, columns = shape
        if columns % GROUP_SIZE:
            raise ValueError(f"PLE width {columns} is not divisible by {GROUP_SIZE}")
        if width is not None and columns != width:
            raise ValueError(f"Inconsistent PLE widths: {width} and {columns}")
        width = columns
        shards.append(PleShard(number, key, filename, rows, columns))

    prefix = next(iter(prefixes))
    required_buffers = {
        f"{prefix.rsplit('.ngram_embedding', 1)[0]}.{name}"
        for name in (
            "layer_multipliers",
            "ngram_heads_offsets",
            "ngram_heads_vocab_sizes",
        )
    }
    missing_buffers = sorted(required_buffers - set(weight_map))
    if missing_buffers:
        raise ValueError(f"Checkpoint is missing PLE hash buffers: {missing_buffers}")

    return SourceCheckpoint(
        directory=root,
        config=config,
        index=index,
        index_sha256=_sha256(index_path),
        prefix=prefix,
        shards=tuple(shards),
    )


def _output_filename(shard: PleShard, shard_count: int) -> str:
    digits = max(5, len(str(shard_count)))
    return (
        f"ple-nvfp4-{shard.number + 1:0{digits}d}"
        f"-of-{shard_count:0{digits}d}.safetensors"
    )


def _new_state(source: SourceCheckpoint) -> dict[str, Any]:
    return {
        "format": STATE_FORMAT,
        "source_directory": str(source.directory),
        "source_index_sha256": source.index_sha256,
        "ple_prefix": source.prefix,
        "shard_count": len(source.shards),
        "global_amax": {},
        "completed_shards": {},
        "status": "converting",
    }


def _load_or_create_state(output: Path, source: SourceCheckpoint) -> dict[str, Any]:
    state_path = output / STATE_NAME
    if state_path.exists():
        state = _read_json(state_path)
        expected = {
            "format": STATE_FORMAT,
            "source_directory": str(source.directory),
            "source_index_sha256": source.index_sha256,
            "ple_prefix": source.prefix,
            "shard_count": len(source.shards),
        }
        mismatches = {
            key: (state.get(key), value)
            for key, value in expected.items()
            if state.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Existing conversion state does not match source: {mismatches}")
        return state
    if any(output.iterdir()):
        raise FileExistsError(
            f"Output directory is nonempty and has no {STATE_NAME}: {output}"
        )
    state = _new_state(source)
    _atomic_json(state_path, state)
    return state


def _scan_amax(shard: PleShard, source: Path, chunk_rows: int) -> float:
    maximum = 0.0
    with safe_open(source / shard.filename, framework="pt", device="cpu") as handle:
        tensor_slice = handle.get_slice(shard.key)
        for start in range(0, shard.rows, chunk_rows):
            chunk = tensor_slice[start : min(start + chunk_rows, shard.rows)]
            if chunk.numel():
                maximum = max(maximum, float(chunk.abs().amax().float()))
    return maximum


def _validate_packed_shard(path: Path, shard: PleShard) -> bool:
    if not path.is_file():
        return False
    expected = {
        shard.key: ("U8", (shard.rows, shard.columns // 2)),
        shard.scale_key: ("F8_E4M3", (shard.rows, shard.columns // GROUP_SIZE)),
        shard.global_scale_key: ("F32", ()),
    }
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            if set(handle.keys()) != set(expected):
                return False
            for key, (dtype, shape) in expected.items():
                view = handle.get_slice(key)
                if view.get_dtype() != dtype or tuple(view.get_shape()) != shape:
                    return False
    except Exception:
        return False
    return True


def _convert_shard(
    source: SourceCheckpoint,
    shard: PleShard,
    destination: Path,
    *,
    global_amax: float,
    chunk_rows: int,
) -> None:
    packed = torch.empty((shard.rows, shard.columns // 2), dtype=torch.uint8)
    block_scales = torch.empty(
        (shard.rows, shard.columns // GROUP_SIZE), dtype=torch.float8_e4m3fn
    )
    global_scale = torch.tensor(
        global_amax / (448.0 * 6.0) if global_amax else 0.0,
        dtype=torch.float32,
    )
    with safe_open(
        source.directory / shard.filename, framework="pt", device="cpu"
    ) as handle:
        tensor_slice = handle.get_slice(shard.key)
        for start in range(0, shard.rows, chunk_rows):
            end = min(start + chunk_rows, shard.rows)
            packed_chunk, scale_chunk, _ = quantize_nvfp4(
                tensor_slice[start:end], global_amax=global_amax
            )
            packed[start:end].copy_(packed_chunk)
            block_scales[start:end].copy_(scale_chunk)

    temporary = destination.with_name(f".{destination.name}.partial")
    if temporary.exists():
        temporary.unlink()
    save_file(
        {
            shard.key: packed,
            shard.scale_key: block_scales,
            shard.global_scale_key: global_scale,
        },
        temporary,
        metadata={"format": FORMAT_NAME},
    )
    os.replace(temporary, destination)


def _link_or_copy(source: Path, destination: Path, *, copy_unchanged: bool) -> None:
    if destination.exists():
        if destination.stat().st_size != source.stat().st_size:
            raise FileExistsError(f"Existing output file differs in size: {destination}")
        return
    if copy_unchanged:
        shutil.copy2(source, destination)
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _copy_checkpoint_assets(
    source: SourceCheckpoint,
    output: Path,
    *,
    copy_unchanged: bool,
) -> None:
    ple_files = {shard.filename for shard in source.shards}
    indexed_files = set(source.index["weight_map"].values())
    for filename in sorted(indexed_files - ple_files):
        source_path = source.directory / filename
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing indexed checkpoint file: {source_path}")
        _link_or_copy(source_path, output / filename, copy_unchanged=copy_unchanged)

    reserved = {CONFIG_NAME, INDEX_NAME, *indexed_files}
    for source_path in sorted(source.directory.iterdir()):
        if not source_path.is_file() or source_path.name in reserved:
            continue
        destination = output / source_path.name
        if destination.exists():
            if destination.stat().st_size != source_path.stat().st_size:
                raise FileExistsError(f"Existing output asset differs: {destination}")
            continue
        shutil.copy2(source_path, destination)


def _finalize(source: SourceCheckpoint, output: Path) -> dict[str, Any]:
    config = json.loads(json.dumps(source.config))
    _text_config(config)["ple_quantization_config"] = Nvfp4PleConfig().to_dict()

    index = json.loads(json.dumps(source.index))
    weight_map: dict[str, str] = index["weight_map"]
    for shard in source.shards:
        filename = _output_filename(shard, len(source.shards))
        weight_map[shard.key] = filename
        weight_map[shard.scale_key] = filename
        weight_map[shard.global_scale_key] = filename
    metadata = index.setdefault("metadata", {})
    source_total = metadata.get("total_size")
    if isinstance(source_total, int):
        metadata["total_size"] = (
            source_total - source.bf16_payload_bytes + source.nvfp4_payload_bytes
        )

    _atomic_json(output / CONFIG_NAME, config)
    _atomic_json(output / INDEX_NAME, index)
    return {
        "format": FORMAT_NAME,
        "source": str(source.directory),
        "output": str(output),
        "shards": len(source.shards),
        "rows": source.rows,
        "embedding_width": source.columns,
        "bf16_ple_bytes": source.bf16_payload_bytes,
        "nvfp4_ple_bytes": source.nvfp4_payload_bytes,
        "saved_bytes": source.bf16_payload_bytes - source.nvfp4_payload_bytes,
    }


def conversion_preview(source_dir: str | Path) -> dict[str, Any]:
    source = inspect_source(source_dir)
    return {
        "format": FORMAT_NAME,
        "source": str(source.directory),
        "ple_prefix": source.prefix,
        "shards": len(source.shards),
        "rows": source.rows,
        "embedding_width": source.columns,
        "bf16_ple_bytes": source.bf16_payload_bytes,
        "nvfp4_ple_bytes": source.nvfp4_payload_bytes,
        "saved_bytes": source.bf16_payload_bytes - source.nvfp4_payload_bytes,
        "source_files_replaced": sorted({shard.filename for shard in source.shards}),
    }


def convert_checkpoint(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    chunk_rows: int = 65_536,
    copy_unchanged: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Convert only PLE shards and hardlink the unchanged checkpoint tensors."""

    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    source = inspect_source(source_dir)
    output = Path(output_dir).expanduser().resolve()
    if source.directory == output:
        raise ValueError("Source and output checkpoint directories must differ")
    if source.directory in output.parents:
        raise ValueError("Output must not be nested inside the source checkpoint")
    output.mkdir(parents=True, exist_ok=True)
    state = _load_or_create_state(output, source)
    emit = progress or LOGGER.info

    amax_values: dict[str, float] = {
        str(key): float(value) for key, value in state.get("global_amax", {}).items()
    }
    completed: dict[str, Any] = dict(state.get("completed_shards", {}))
    for shard in source.shards:
        key = str(shard.number)
        filename = _output_filename(shard, len(source.shards))
        destination = output / filename
        if key in completed and _validate_packed_shard(destination, shard):
            emit(f"[{shard.number + 1}/{len(source.shards)}] verified existing {filename}")
            continue
        if key not in amax_values:
            emit(f"[{shard.number + 1}/{len(source.shards)}] scanning global amax")
            amax_values[key] = _scan_amax(shard, source.directory, chunk_rows)
            state["global_amax"] = amax_values
            _atomic_json(output / STATE_NAME, state)
        emit(f"[{shard.number + 1}/{len(source.shards)}] packing {filename}")
        _convert_shard(
            source,
            shard,
            destination,
            global_amax=amax_values[key],
            chunk_rows=chunk_rows,
        )
        if not _validate_packed_shard(destination, shard):
            raise RuntimeError(f"Packed shard validation failed: {destination}")
        completed[key] = {"filename": filename, "global_amax": amax_values[key]}
        state["completed_shards"] = completed
        _atomic_json(output / STATE_NAME, state)

    emit("Linking or copying unchanged checkpoint files")
    _copy_checkpoint_assets(source, output, copy_unchanged=copy_unchanged)
    result = _finalize(source, output)
    state["status"] = "complete"
    state["result"] = result
    _atomic_json(output / STATE_NAME, state)
    return result


def _packed_shards(root: Path, index: dict[str, Any]) -> tuple[PleShard, ...]:
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"{INDEX_NAME} has no weight_map object")
    matches: list[tuple[int, str, str]] = []
    for key, filename in weight_map.items():
        match = PLE_WEIGHT_RE.match(key)
        if match:
            matches.append((int(match.group("shard")), key, str(filename)))
    matches.sort()
    if [item[0] for item in matches] != list(range(len(matches))):
        raise ValueError("Packed PLE shards are missing or non-contiguous")
    shards: list[PleShard] = []
    for number, key, filename in matches:
        stem = key[: -len(".weight")]
        scale_key = f"{stem}.weight_scale"
        global_key = f"{stem}.weight_scale_2"
        if weight_map.get(scale_key) != filename or weight_map.get(global_key) != filename:
            raise ValueError(f"Packed PLE scale mappings are incomplete for shard {number}")
        path = root / filename
        with safe_open(path, framework="pt", device="cpu") as handle:
            value_view = handle.get_slice(key)
            scale_view = handle.get_slice(scale_key)
            global_view = handle.get_slice(global_key)
            packed_shape = tuple(value_view.get_shape())
            scale_shape = tuple(scale_view.get_shape())
            global_shape = tuple(global_view.get_shape())
            dtypes = (
                value_view.get_dtype(),
                scale_view.get_dtype(),
                global_view.get_dtype(),
            )
        if len(packed_shape) != 2:
            raise ValueError(f"Packed shard {number} is not a matrix: {packed_shape}")
        rows, packed_width = packed_shape
        columns = packed_width * 2
        expected_scale = (rows, columns // GROUP_SIZE)
        if dtypes != ("U8", "F8_E4M3", "F32"):
            raise ValueError(f"Packed shard {number} has wrong dtypes: {dtypes}")
        if scale_shape != expected_scale or global_shape != ():
            raise ValueError(
                f"Packed shard {number} has wrong scale shapes: "
                f"{scale_shape}, {global_shape}"
            )
        shards.append(PleShard(number, key, filename, rows, columns))
    if not shards:
        raise ValueError("Checkpoint declares NVFP4 PLE but has no packed shards")
    return tuple(shards)


def inspect_checkpoint(directory: str | Path) -> dict[str, Any]:
    """Return a machine-readable format and storage report for a checkpoint."""

    root = Path(directory).expanduser().resolve()
    config = _read_json(root / CONFIG_NAME)
    text_config = _text_config(config)
    metadata = text_config.get("ple_quantization_config")
    if metadata is None:
        source = inspect_source(root)
        return conversion_preview(source.directory) | {"kind": "bf16-source"}

    format_config = Nvfp4PleConfig.from_mapping(metadata)
    index = _read_json(root / INDEX_NAME)
    shards = _packed_shards(root, index)
    widths = {shard.columns for shard in shards}
    if len(widths) != 1:
        raise ValueError(f"Packed PLE shards have inconsistent widths: {sorted(widths)}")
    rows = sum(shard.rows for shard in shards)
    columns = next(iter(widths))
    payload = nvfp4_storage_bytes(rows, columns, len(shards))
    return {
        "kind": "nvfp4-ple",
        "format": format_config.format,
        "checkpoint": str(root),
        "shards": len(shards),
        "rows": rows,
        "embedding_width": columns,
        "nvfp4_ple_bytes": payload,
        "equivalent_bf16_ple_bytes": rows * columns * 2,
        "saved_bytes": rows * columns * 2 - payload,
        "index_total_size": index.get("metadata", {}).get("total_size"),
    }
