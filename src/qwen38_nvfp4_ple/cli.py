"""Command-line interface for PLE checkpoint conversion and inspection."""

from __future__ import annotations

import argparse
import json
import logging
from typing import Sequence

from .checkpoint import convert_checkpoint, conversion_preview, inspect_checkpoint


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qwen38-nvfp4-ple",
        description="Convert only Qwen3.8 PLE embeddings from BF16 to NVFP4.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert = subparsers.add_parser("convert", help="create or resume a converted checkpoint")
    convert.add_argument("source", help="source Hugging Face checkpoint directory")
    convert.add_argument("output", help="new checkpoint directory")
    convert.add_argument(
        "--chunk-rows",
        type=int,
        default=65_536,
        help="rows processed at once (default: 65536)",
    )
    convert.add_argument(
        "--copy-unchanged",
        action="store_true",
        help="copy unchanged model shards instead of hardlinking when possible",
    )
    convert.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the source and print the exact storage plan",
    )

    inspect = subparsers.add_parser("inspect", help="validate and describe a checkpoint")
    inspect.add_argument("checkpoint", help="checkpoint directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.command == "inspect":
        result = inspect_checkpoint(args.checkpoint)
    elif args.dry_run:
        result = conversion_preview(args.source)
    else:
        result = convert_checkpoint(
            args.source,
            args.output,
            chunk_rows=args.chunk_rows,
            copy_unchanged=args.copy_unchanged,
            progress=lambda message: print(message, flush=True),
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
