from __future__ import annotations

import json
import re
from typing import Any

from crepidinem.exceptions import AgentParseError

__all__ = ["extract_json_object", "extract_python_code"]

_FENCE = re.compile(
    r"```[ \t]*(?P<lang>[A-Za-z0-9_+-]*)[ \t]*\r?\n(?P<body>.*?)```",
    re.DOTALL,
)


def _fenced_blocks(text: str) -> list[tuple[str, str]]:
    """return language,body"""
    return [(m.group("lang").lower(), m.group("body")) for m in _FENCE.finditer(text)]


def _balanced_json_slice(text: str) -> str | None:
    """find the first complete span"""

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def extract_json_object(text: str) -> dict[str, Any]:
    """pull json out of response"""

    candidates: list[str] = []
    blocks = _fenced_blocks(text)
    candidates.extend(body for lang, body in blocks if lang in {"json", "json5", ""})
    candidates.extend(body for lang, body in blocks if lang not in {"json", "json5", ""})
    candidates.append(text)
    balanced = _balanced_json_slice(text)
    if balanced is not None:
        candidates.append(balanced)

    for candidate in candidates:
        stripped = candidate.strip()
        if not stripped.startswith("{"):
            inner = _balanced_json_slice(stripped)

            if inner is None:
                continue
            stripped = inner

        try:
            parsed: Any = json.loads(stripped)

        except json.JSONDecodeError:
            continue

        if isinstance(parsed, dict):
            return parsed

    msg = "No JSON object could be parsed out of the response."
    raise AgentParseError(msg, raw=text)


def extract_python_code(text: str) -> str:
    """pull script out of response"""

    blocks = _fenced_blocks(text)
    ordered = [body for lang, body in blocks if lang in {"python", "py", "python3"}]
    ordered += [body for lang, body in blocks if lang not in {"python", "py", "python3"}]
    ordered.append(text)

    first_syntax_error: SyntaxError | None = None
    for candidate in ordered:
        code = candidate.strip()
        if not code:
            continue
        try:
            compile(code, "control_script.py", "exec")

        except SyntaxError as exc:
            if first_syntax_error is None:
                first_syntax_error = exc

            continue
        return code

    if first_syntax_error is not None:
        msg = (
            f"The response contained Python that does not parse: "
            f"{first_syntax_error.msg} (line {first_syntax_error.lineno})."
        )
        raise AgentParseError(msg, raw=text)

    msg = "The response contained no Python code block."
    raise AgentParseError(msg, raw=text)
