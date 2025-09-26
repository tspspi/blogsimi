"""blogsimi package public API."""

from __future__ import annotations

try:
    from ._version import version as __version__
except ImportError:  # pragma: no cover - fallback for editable installs
    __version__ = "0.0.0"

from .core import run

__all__ = ["run", "__version__"]
