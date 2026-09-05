"""value types shared by every sandbox backend"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["ExecutionResult"]


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """outcome of running a script inside a sandbox"""

    stdout: str
    stderr: str
    exit_code: int
    duration_s: float = 0.0
    timed_out: bool = False
    sandbox_id: str = ""
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """true when the process exited cleanly"""
        return self.exit_code == 0 and not self.timed_out

    def summary(self, *, limit: int = 400) -> str:
        """compact oneline description"""
        tail = (self.stderr or self.stdout).strip().replace("\n", " ⏎ ")
        if len(tail) > limit:
            tail = f"{tail[:limit]}…"
        return (
            f"exit={self.exit_code} timed_out={self.timed_out} "
            f"duration={self.duration_s:.2f}s :: {tail}"
        )
