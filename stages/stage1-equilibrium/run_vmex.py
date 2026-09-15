"""Run VMEX and publish its WOUT at the requested pipeline output path."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import tempfile
from typing import Sequence


logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Stage 1 VMEX equilibrium solve.")
    parser.add_argument("--input", required=True, type=Path, help="Path to the VMEC input file.")
    parser.add_argument("--output", required=True, type=Path, help="Path to the output WOUT NetCDF file.")
    # VMEX's --device auto picks CPU for small resolutions even in the GPU image,
    # so the workflow passes the configured device.
    parser.add_argument(
        "--device",
        default="auto",
        help="Solve device passed to VMEX. Defaults to its automatic placement policy.",
    )
    return parser


def run_vmex(input_path: Path, output_path: Path, device: str = "auto") -> int:
    """Run VMEX and move its WOUT to the requested path after success."""
    # Match VMEX's resolved input path for relative references and WOUT naming.
    input_path = input_path.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Stage 1 input file does not exist: {input_path}")

    output_path = output_path.parent.resolve() / output_path.name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from vmex.core.cli import main as vmex_main, resolve_wout_path

    # VMEX can write WOUT before returning a nonzero status.
    with tempfile.TemporaryDirectory(prefix=".vmex-", dir=output_path.parent) as temporary:
        status = vmex_main(
            [str(input_path), "--outdir", temporary, "--device", device]
        )
        if status:
            return status
        resolve_wout_path(input_path=input_path, outdir=Path(temporary)).replace(output_path)

    logger.info("Wrote Stage 1 WOUT file: %s", output_path)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)
    try:
        return run_vmex(args.input, args.output, args.device)
    except Exception:
        logger.exception("Stage 1 VMEX failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
