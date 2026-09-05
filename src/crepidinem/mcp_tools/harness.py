"""physics validation harness"""

from __future__ import annotations

import base64
import json
import secrets
from dataclasses import asdict, dataclass, field

__all__ = [
    "EXIT_HARNESS_ERROR",
    "EXIT_OK",
    "EXIT_PHYSICS_VIOLATION",
    "EXIT_RUNTIME_ERROR",
    "MAX_WAYPOINTS",
    "RESULT_MARKER",
    "VIOLATION_CODES",
    "PhysicsLimits",
    "build_harness",
    "new_nonce",
]

#cap on trajectory points
MAX_WAYPOINTS = 2000

#prefixes
RESULT_MARKER = "__CREPIDINEM_RESULT__"
VIOLATION_MARKER = "PHYSICS_VIOLATION"

EXIT_OK = 0
EXIT_HARNESS_ERROR = 1
EXIT_PHYSICS_VIOLATION = 3
EXIT_RUNTIME_ERROR = 4

#all violations
VIOLATION_CODES: frozenset[str] = frozenset(
    {
        "ACCELERATION_LIMIT_EXCEEDED",
        "COLLISION_DETECTED",
        "COMMAND_BUDGET_EXCEEDED",
        "GRIPPER_FORCE_EXCEEDED",
        "NO_COMMANDS_ISSUED",
        "PAYLOAD_LIMIT_EXCEEDED",
        "VELOCITY_LIMIT_EXCEEDED",
        "WORKSPACE_BOUNDS_EXCEEDED",
    }
)


@dataclass(frozen=True, slots=True)
class PhysicsLimits:
    """constraints in simulated cell"""

    max_velocity: float = 0.5
    max_acceleration: float = 2.0
    max_payload_kg: float = 1.0
    max_gripper_force_n: float = 20.0

    #axis reachable volume
    workspace_x: tuple[float, float] = (-0.6, 0.6)
    workspace_y: tuple[float, float] = (-0.6, 0.6)
    workspace_z: tuple[float, float] = (-0.10, 0.80)

    #tool tip must stay at least this far above the table surface
    table_height: float = 0.0
    table_clearance: float = 0.02

    #static obstacles as axis aligned boxes
    obstacles: tuple[tuple[float, float, float, float, float, float], ...] = field(
        default_factory=lambda: (
            (-0.30, -0.10, 0.00, -0.10, 0.10, 0.15),
        )
    )

    #duration of control tick (seconds)
    control_tick: float = 0.2

    #spatial resolution of the swept-path collision check (meters)
    collision_step: float = 0.005

    #upper bound on motion commands
    max_commands: int = 2000

    def to_json(self) -> str:
        """Serialise for embedding into the harness."""
        return json.dumps(asdict(self), sort_keys=True)


_HARNESS_TEMPLATE = '''\
"""Crepidinem physics validation harness (generated - do not edit by hand)."""

import base64
import json
import math
import sys
import traceback

LIMITS = json.loads(__LIMITS_JSON__)
RESULT_MARKER = __RESULT_MARKER__
RESULT_NONCE = __RESULT_NONCE__
VIOLATION_MARKER = __VIOLATION_MARKER__
EXIT_OK, EXIT_HARNESS_ERROR, EXIT_VIOLATION, EXIT_RUNTIME = 0, 1, 3, 4
MAX_WAYPOINTS = __MAX_WAYPOINTS__


class PhysicsViolation(BaseException):
    """A hard physical constraint was broken.

    Derives from BaseException, not Exception, so a bare ``except Exception``
    in the control script cannot swallow it. Every violation is also recorded
    on the arm before it is raised, so even a script that catches
    BaseException is still reported as a failure.
    """

    def __init__(self, code, message, **detail):
        super().__init__("[%s] %s" % (code, message))
        self.code = code
        self.message = message
        self.detail = detail


class RoboticArm:
    """Deterministic kinematic model of the cell's arm.

    Every motion command is validated *before* the state is mutated, and the
    straight-line path between waypoints is swept at ``collision_step``
    resolution - so a move that would clip the table midway is caught even
    when both endpoints are legal.
    """

    def __init__(self, position=(0.0, 0.0, 0.30), name="arm0"):
        self.name = name
        self.position = tuple(float(v) for v in position)
        self.velocity = min(0.1, LIMITS["max_velocity"])
        self.payload_kg = 0.0
        self.gripper_open = True
        self.gripper_force_n = 0.0
        self.sim_time = 0.0
        self.path_length = 0.0
        self.commands = 0
        self.log = []
        self.violations = []
        self.waypoints = []
        self._check_point(self.position, "initial pose")
        self._add_waypoint("start")

    def _add_waypoint(self, cmd):
        """Record the tool-tip pose after a command, for trajectory rendering.

        This is observation only - nothing here affects the verdict. It exists
        so a certified script can be shown as the physical path it describes
        rather than as a wall of text.
        """
        if len(self.waypoints) >= MAX_WAYPOINTS:
            return
        self.waypoints.append(
            {
                "index": len(self.waypoints),
                "x": round(self.position[0], 6),
                "y": round(self.position[1], 6),
                "z": round(self.position[2], 6),
                "cmd": cmd,
                "velocity": round(self.velocity, 6),
                "sim_time": round(self.sim_time, 6),
                "gripper_open": self.gripper_open,
                "payload_kg": self.payload_kg,
            }
        )

    def _violate(self, code, message, **detail):
        """Announce, record, then raise - in that order.

        The stderr write comes first and is flushed immediately because it is
        the one record the control script cannot retract: bytes already on the
        pipe are outside this process. The exception and the violations list
        both live in memory the script can reach, so neither can be the only
        evidence that a limit was broken.
        """
        sys.stderr.write("%s: %s\\n" % (VIOLATION_MARKER, code))
        sys.stderr.flush()
        violation = PhysicsViolation(code, message, **detail)
        self.violations.append(violation)
        raise violation

    # ---------------------------------------------------------- bookkeeping

    def _record(self, entry):
        self.log.append(entry)
        self.commands += 1
        if self.commands > LIMITS["max_commands"]:
            self._violate(
                "COMMAND_BUDGET_EXCEEDED",
                "Control script issued more than %d commands; "
                "suspected unbounded loop." % LIMITS["max_commands"],
                commands=self.commands,
            )

    # ------------------------------------------------------------- geometry

    def _check_point(self, point, context):
        x, y, z = point
        for axis, value, (lo, hi) in (
            ("x", x, LIMITS["workspace_x"]),
            ("y", y, LIMITS["workspace_y"]),
            ("z", z, LIMITS["workspace_z"]),
        ):
            if value < lo - 1e-9 or value > hi + 1e-9:
                self._violate(
                    "WORKSPACE_BOUNDS_EXCEEDED",
                    "%s=%.4f leaves the reachable volume [%.3f, %.3f] during %s."
                    % (axis, value, lo, hi, context),
                    axis=axis,
                    value=value,
                    limit=[lo, hi],
                    point=list(point),
                )

        floor = LIMITS["table_height"] + LIMITS["table_clearance"]
        if z < floor - 1e-9:
            self._violate(
                "COLLISION_DETECTED",
                "Tool tip reached z=%.4f during %s; the table surface is at "
                "z=%.3f and requires %.3fm clearance."
                % (z, context, LIMITS["table_height"], LIMITS["table_clearance"]),
                obstacle="table",
                point=list(point),
                min_z=floor,
            )

        for index, box in enumerate(LIMITS["obstacles"]):
            x0, y0, z0, x1, y1, z1 = box
            if x0 <= x <= x1 and y0 <= y <= y1 and z0 <= z <= z1:
                self._violate(
                    "COLLISION_DETECTED",
                    "Tool tip entered obstacle #%d %s at (%.3f, %.3f, %.3f) during %s."
                    % (index, box, x, y, z, context),
                    obstacle="obstacle_%d" % index,
                    box=list(box),
                    point=list(point),
                )

    def _sweep(self, start, end, context):
        """Sample the straight-line path and validate every sample."""
        dx, dy, dz = (end[0] - start[0], end[1] - start[1], end[2] - start[2])
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        steps = max(1, int(math.ceil(distance / LIMITS["collision_step"])))
        for step in range(1, steps + 1):
            t = step / steps
            self._check_point(
                (start[0] + dx * t, start[1] + dy * t, start[2] + dz * t), context
            )
        return distance

    # ------------------------------------------------------------- commands

    def set_velocity(self, velocity):
        """Set the commanded tool-tip speed in m/s."""
        velocity = float(velocity)
        if velocity <= 0.0:
            self._violate(
                "VELOCITY_LIMIT_EXCEEDED",
                "Commanded velocity %.4f m/s is not positive." % velocity,
                requested=velocity,
            )
        if velocity > LIMITS["max_velocity"] + 1e-9:
            self._violate(
                "VELOCITY_LIMIT_EXCEEDED",
                "Commanded velocity %.4f m/s exceeds the %.4f m/s joint limit."
                % (velocity, LIMITS["max_velocity"]),
                requested=velocity,
                limit=LIMITS["max_velocity"],
            )
        delta = abs(velocity - self.velocity)
        # A step change in commanded speed is executed over one control tick.
        acceleration = delta / LIMITS["control_tick"]
        if acceleration > LIMITS["max_acceleration"] + 1e-9:
            self._violate(
                "ACCELERATION_LIMIT_EXCEEDED",
                "Stepping velocity %.4f -> %.4f m/s in one tick implies %.2f m/s^2, "
                "over the %.2f m/s^2 limit."
                % (self.velocity, velocity, acceleration, LIMITS["max_acceleration"]),
                implied_acceleration=acceleration,
                limit=LIMITS["max_acceleration"],
            )
        self.velocity = velocity
        self._record({"cmd": "set_velocity", "velocity": velocity})

    def move_to(self, x, y, z, velocity=None):
        """Move the tool tip to an absolute position, in a straight line."""
        if velocity is not None:
            self.set_velocity(velocity)
        target = (float(x), float(y), float(z))
        distance = self._sweep(self.position, target, "move_to%s" % (target,))
        self.position = target
        self.path_length += distance
        self.sim_time += distance / self.velocity if self.velocity else 0.0
        self._add_waypoint("move_to")
        self._record(
            {
                "cmd": "move_to",
                "target": list(target),
                "velocity": self.velocity,
                "distance": distance,
            }
        )

    def move_by(self, dx, dy, dz, velocity=None):
        """Move the tool tip by a relative offset."""
        self.move_to(
            self.position[0] + float(dx),
            self.position[1] + float(dy),
            self.position[2] + float(dz),
            velocity=velocity,
        )

    def home(self):
        """Return to the safe home pose."""
        self.move_to(0.0, 0.0, 0.30)

    def grip(self, force_n=5.0, payload_kg=0.0):
        """Close the gripper with ``force_n`` newtons onto ``payload_kg``."""
        force_n = float(force_n)
        payload_kg = float(payload_kg)
        if force_n > LIMITS["max_gripper_force_n"] + 1e-9:
            self._violate(
                "GRIPPER_FORCE_EXCEEDED",
                "Gripper force %.2f N exceeds the %.2f N limit; sample would be crushed."
                % (force_n, LIMITS["max_gripper_force_n"]),
                requested=force_n,
                limit=LIMITS["max_gripper_force_n"],
            )
        if payload_kg > LIMITS["max_payload_kg"] + 1e-9:
            self._violate(
                "PAYLOAD_LIMIT_EXCEEDED",
                "Payload %.3f kg exceeds the %.3f kg limit."
                % (payload_kg, LIMITS["max_payload_kg"]),
                requested=payload_kg,
                limit=LIMITS["max_payload_kg"],
            )
        self.gripper_open = False
        self.gripper_force_n = force_n
        self.payload_kg = payload_kg
        self._add_waypoint("grip")
        self._record({"cmd": "grip", "force_n": force_n, "payload_kg": payload_kg})

    def release(self):
        """Open the gripper and drop any payload."""
        self.gripper_open = True
        self.gripper_force_n = 0.0
        self.payload_kg = 0.0
        self._add_waypoint("release")
        self._record({"cmd": "release"})

    def wait(self, seconds):
        """Advance the simulated clock. Never sleeps for real."""
        seconds = float(seconds)
        self.sim_time += seconds
        self._record({"cmd": "wait", "seconds": seconds})

    def telemetry(self):
        return {
            "commands": self.commands,
            "final_position": list(self.position),
            "path_length_m": round(self.path_length, 6),
            "sim_time_s": round(self.sim_time, 6),
            "final_velocity": self.velocity,
            "payload_kg": self.payload_kg,
            "gripper_open": self.gripper_open,
        }


#: Builtins a control script is allowed to see. The omissions are the point:
#: no ``open``, no ``eval``/``exec``/``compile``, no ``globals``, and an
#: ``__import__`` that only yields ``math``. Without those a script cannot
#: reach ``sys`` or ``os``, so it can neither end the process before the
#: harness reports nor edit the arm's own bookkeeping from the outside.
_ALLOWED_BUILTINS = (
    "abs bool dict divmod enumerate filter float format frozenset int len list map max min "
    "next print range reversed round set slice sorted str sum tuple zip "
    "True False None ArithmeticError AssertionError AttributeError BaseException "
    "Exception IndexError "
    "KeyError NameError RuntimeError StopIteration TypeError ValueError ZeroDivisionError"
).split()

#: Deterministic, side-effect-free modules a control script may import.
#: ``random`` and ``time`` are absent deliberately: both would make a verdict
#: depend on something other than the script.
_ALLOWED_MODULES = ("math",)


def _safe_builtins():
    """Build the restricted builtins mapping handed to the control script."""
    import builtins

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name not in _ALLOWED_MODULES:
            raise ImportError(
                "Control scripts may only import %s; %r is not available."
                % (", ".join(_ALLOWED_MODULES), name)
            )
        return __import__(name, globals, locals, fromlist, level)

    safe = {}
    for entry in _ALLOWED_BUILTINS:
        if hasattr(builtins, entry):
            safe[entry] = getattr(builtins, entry)
    safe["__import__"] = guarded_import
    return safe


def _emit(payload, exit_code):
    # Generated per run on the host. A verdict line without it did not come
    # from this harness, so the host refuses to read it.
    payload["nonce"] = RESULT_NONCE
    sys.stdout.write(RESULT_MARKER + json.dumps(payload) + "\\n")
    sys.stdout.flush()
    sys.exit(exit_code)


def main():
    arm = RoboticArm()

    def emit(payload, exit_code):
        """Attach the trajectory to every verdict, then report it."""
        payload["waypoints"] = arm.waypoints
        payload["waypoints_truncated"] = len(arm.waypoints) >= MAX_WAYPOINTS
        _emit(payload, exit_code)

    namespace = {
        "__name__": "__crepidinem_control__",
        "__builtins__": _safe_builtins(),
        "arm": arm,
        "RoboticArm": RoboticArm,
        "PhysicsViolation": PhysicsViolation,
        "LIMITS": dict(LIMITS),
        "MAX_VELOCITY": LIMITS["max_velocity"],
        "TABLE_HEIGHT": LIMITS["table_height"],
    }
    source = base64.b64decode(__CONTROL_B64__).decode("utf-8")

    try:
        exec(compile(source, "control_script.py", "exec"), namespace)
    except PhysicsViolation as violation:
        sys.stderr.write("%s: %s\\n" % (VIOLATION_MARKER, violation.code))
        emit(
            {
                "status": "violation",
                "violation": {
                    "code": violation.code,
                    "message": violation.message,
                    "detail": violation.detail,
                },
                "telemetry": arm.telemetry(),
                "command_log": arm.log[-25:],
                "traceback": traceback.format_exc(limit=6),
            },
            EXIT_VIOLATION,
        )
    except SystemExit as exc:
        if exc.code not in (0, None):
            emit(
                {
                    "status": "error",
                    "error": "Control script exited with status %r." % (exc.code,),
                    "telemetry": arm.telemetry(),
                    "command_log": arm.log[-25:],
                    "traceback": None,
                },
                EXIT_RUNTIME,
            )
    except BaseException:
        emit(
            {
                "status": "error",
                "error": "Control script raised before completing.",
                "telemetry": arm.telemetry(),
                "command_log": arm.log[-25:],
                "traceback": traceback.format_exc(limit=6),
            },
            EXIT_RUNTIME,
        )

    if arm.violations:
        swallowed = arm.violations[0]
        sys.stderr.write("%s: %s\\n" % (VIOLATION_MARKER, swallowed.code))
        emit(
            {
                "status": "violation",
                "violation": {
                    "code": swallowed.code,
                    "message": swallowed.message
                    + " (the control script suppressed this exception; "
                    "the harness does not allow that)",
                    "detail": swallowed.detail,
                },
                "telemetry": arm.telemetry(),
                "command_log": arm.log[-25:],
                "traceback": None,
            },
            EXIT_VIOLATION,
        )

    if arm.commands == 0:
        sys.stderr.write("%s: NO_COMMANDS_ISSUED\\n" % VIOLATION_MARKER)
        emit(
            {
                "status": "violation",
                "violation": {
                    "code": "NO_COMMANDS_ISSUED",
                    "message": "The control script ran but never commanded the arm.",
                    "detail": {},
                },
                "telemetry": arm.telemetry(),
                "command_log": [],
                "traceback": None,
            },
            EXIT_VIOLATION,
        )

    emit(
        {
            "status": "ok",
            "violation": None,
            "telemetry": arm.telemetry(),
            "command_log": arm.log[-25:],
            "traceback": None,
        },
        EXIT_OK,
    )


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        sys.stderr.write(traceback.format_exc())
        sys.exit(EXIT_HARNESS_ERROR)
'''


def new_nonce() -> str:
    """mint a verdict nonce for one run"""
    return secrets.token_hex(16)


def build_harness(
    control_code: str,
    limits: PhysicsLimits | None = None,
    *,
    nonce: str | None = None,
) -> str:
    """Wrap ``control_code`` in the physics validation harness.

    Args:
        control_code: Raw Python emitted by the Code Smith. Treated as opaque
            text; it is base64-embedded, never interpolated.
        limits: Constraints to enforce. Defaults to :class:`PhysicsLimits`.
        nonce: Value the harness stamps on its verdict, so the caller can tell
            a real verdict from a line the control script printed. Minted per
            run when omitted.

    Returns:
        A complete, self-contained Python program (stdlib only) ready to be
        written into a sandbox and executed.
    """
    resolved = limits or PhysicsLimits()
    encoded = base64.b64encode(control_code.encode("utf-8")).decode("ascii")
    return (
        _HARNESS_TEMPLATE.replace("__LIMITS_JSON__", repr(resolved.to_json()))
        .replace("__RESULT_NONCE__", repr(nonce or new_nonce()))
        .replace("__RESULT_MARKER__", repr(RESULT_MARKER))
        .replace("__VIOLATION_MARKER__", repr(VIOLATION_MARKER))
        .replace("__MAX_WAYPOINTS__", repr(MAX_WAYPOINTS))
        .replace("__CONTROL_B64__", repr(encoded))
    )
