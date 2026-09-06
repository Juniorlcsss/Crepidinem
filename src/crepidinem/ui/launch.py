"""start the dashboard."""

from __future__ import annotations

import importlib.util
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
    if importlib.util.find_spec("streamlit") is None:
        sys.stderr.write(_MISSING)
        return 2

    command = [sys.executable, "-m", "streamlit", "run", str(_APP), *(argv or sys.argv[1:])]
    return subprocess.call(command)  # noqa: S603


if __name__ == "__main__":
    raise SystemExit(main())
