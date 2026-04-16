# Author: Perry Radau
# Date: 2026-04-16
# CLI launcher for the tSNR calculator.
# Dependencies: Python 3.10+, tsnr module dependencies
# Usage: uv run python main.py /path/to/input.nii.gz phantom
#        uv run python main.py /path/to/input.nii.gz brain

"""Main entry point for tSNR CLI."""

from __future__ import annotations

from tsnr import cli


def main() -> int:
    """Execute CLI and return exit code.

    Returns:
        int: CLI exit code.
    """
    return cli()


if __name__ == "__main__":
    raise SystemExit(main())
