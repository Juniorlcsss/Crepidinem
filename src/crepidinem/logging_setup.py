"""Logging configuration"""

from __future__ import annotations

import logging
import sys

__all__ = ["configure_console", "configure_logging", "get_logger"]

_FORMAT = "%(asctime)s %(levelname)-8s %(name)-38s %(message)s"
_DATEFMT = "%H:%M:%S"


def configure_console() -> None:
    """Make stdout and stderr survive whatever the model decides to write."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            continue


def configure_logging(level: str = "INFO") -> None:
    """Install a single stderr handler on the root logger (idempotent)."""
    root = logging.getLogger()
    resolved = getattr(logging, level.upper(), logging.INFO)
    root.setLevel(resolved)

    for handler in root.handlers:
        if getattr(handler, "_crepidinem", False):
            handler.setLevel(resolved)
            return

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    handler.setLevel(resolved)
    setattr(handler, "_crepidinem", True)
    root.addHandler(handler)

    logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a package-namespaced logger."""
    return logging.getLogger(name)
