"""Command-line entry point for blogsimi."""

from __future__ import annotations

from typing import Optional, Sequence

from . import core


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the blogsimi command-line interface."""
    return core.run(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
