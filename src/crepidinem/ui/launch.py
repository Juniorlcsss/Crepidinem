"""start the dashboard."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

__all__ = ["main"]

_APP = Path(__file__).with_name("app.py")

_MISSING = """\
crepidinem: the dashboard needs the `ui` extra.

    uv sync --extra ui

"""


def main(argv: list[str] | None = None) -> int:
    """launch Streamlit dashboard"""
    try:
        import streamlit
    except ImportError:
        sys.stderr.write(_MISSING)
        return 2

    command = [sys.executable, "-m", "streamlit", "run", str(_APP), *(argv or sys.argv[1:])]
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
