"""Command-line entry point for the core loop.

uv run crepidinem                       #run the demo goal
uv run crepidinem --goal "..."          #run your own
uv run crepidinem --check script.py     #verify one script and stop
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import sys
from pathlib import Path

from crepidinem.config import Settings, load_settings
from crepidinem.exceptions import CrepidinemError
from crepidinem.logging_setup import configure_console, configure_logging, get_logger
from crepidinem.mcp_tools.harness import PhysicsLimits
from crepidinem.mcp_tools.physics_sandbox import format_feedback, test_code_in_physics_sandbox
from crepidinem.orchestrator.core import (
    DEFAULT_GOAL,
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_REPLAN_AFTER,
    ExperimentReport,
    run_experiment_sync,
)
from crepidinem.research.hardware import HardwareDossier

__all__ = ["main"]

log = get_logger(__name__)

EXIT_SUCCESS = 0
EXIT_NOT_CERTIFIED = 1
EXIT_FAILURE = 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crepidinem",
        description=(
            "Run the Crepidinem loop: plan an experiment, generate control code, and "
            "refuse to release anything the physics sandbox has not certified."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--goal", default=DEFAULT_GOAL, help="Scientific goal to pursue.")
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help="How many times the Code Smith may rewrite before giving up.",
    )
    parser.add_argument(
        "--replan-after",
        type=int,
        default=DEFAULT_REPLAN_AFTER,
        help=(
            "Consecutive rejections before the Principal Investigator is asked "
            "for a new plan rather than the Code Smith for another rewrite. "
            "0 never replans."
        ),
    )
    parser.add_argument(
        "--planner-model",
        default=None,
        help="Token Factory model id for the Principal Investigator.",
    )
    parser.add_argument(
        "--coder-model",
        default=None,
        help="Token Factory model id for the Code Smith.",
    )
    parser.add_argument(
        "--no-brief-limits",
        dest="brief_limits",
        action="store_false",
        default=None,
        help=(
            "Withhold the cell's numeric limits from the Code Smith. Demonstrates "
            "the guardrail catching an under-briefed model, rather than the prompt "
            "preventing the mistake."
        ),
    )
    parser.add_argument(
        "--no-research",
        dest="research",
        action="store_false",
        default=None,
        help=(
            "Skip the hardware research step and use the cell's default limits, "
            "even when TAVILY_API_KEY is set."
        ),
    )
    parser.add_argument(
        "--refresh-research",
        dest="research_refresh",
        action="store_true",
        default=None,
        help=(
            "Ignore the search cache and re-issue every query live. Costs "
            "Tavily credits; use it for a demo, where the search should be real."
        ),
    )
    parser.add_argument(
        "--no-research-cache",
        dest="research_cache",
        action="store_false",
        default=None,
        help="Do not read or write the on-disk search cache at all.",
    )
    parser.add_argument(
        "--research-trust",
        choices=("envelope", "narrow-only"),
        default=None,
        help=(
            "How far a retrieved value may move an enforced limit. narrow-only "
            "never lets retrieved text loosen a safety constraint."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=("local", "docker", "nebius"),
        default=None,
        help="Override CREPIDINEM_SANDBOX_BACKEND for this run.",
    )
    parser.add_argument(
        "--max-velocity",
        type=float,
        default=None,
        help="Override the joint velocity limit, in m/s.",
    )
    parser.add_argument(
        "--check",
        type=Path,
        default=None,
        metavar="SCRIPT",
        help="Verify a single control script against the guardrail and exit.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the machine-readable report instead of the transcript.",
    )
    parser.add_argument("--log-level", default=None, help="DEBUG, INFO, WARNING or ERROR.")
    return parser


def _resolve_settings(args: argparse.Namespace) -> Settings:
    settings = load_settings()
    if args.backend is not None:
        settings = dataclasses.replace(settings, sandbox_backend=args.backend)
    if args.log_level is not None:
        settings = dataclasses.replace(settings, log_level=str(args.log_level).upper())
    if args.planner_model is not None:
        settings = dataclasses.replace(settings, planner_model=args.planner_model)
    if args.coder_model is not None:
        settings = dataclasses.replace(settings, coder_model=args.coder_model)
    if args.brief_limits is not None:
        settings = dataclasses.replace(settings, brief_limits=args.brief_limits)
    if args.research is not None:
        settings = dataclasses.replace(settings, research_enabled=args.research)
    if args.research_trust is not None:
        settings = dataclasses.replace(settings, research_trust=args.research_trust)
    if args.research_cache is not None:
        settings = dataclasses.replace(settings, research_cache_enabled=args.research_cache)
    if args.research_refresh is not None:
        settings = dataclasses.replace(settings, research_cache_refresh=args.research_refresh)
    return settings


def _research_to_json(dossier: HardwareDossier | None) -> dict[str, object] | None:
    """Serialise the research dossier, including what it refused to apply."""
    if dossier is None:
        return None
    return {
        "hardware": list(dossier.hardware),
        "queries": list(dossier.queries),
        "sources": [{"title": h.title, "url": h.url} for h in dossier.hits],
        "findings": [
            {
                "field": f.field,
                "value": f.value,
                "unit": f.unit,
                "previous": f.previous,
                "accepted": f.accepted,
                "reason": f.reason,
                "source_url": f.source_url,
            }
            for f in dossier.findings
        ],
        "notes": list(dossier.notes),
    }


def _report_to_json(report: ExperimentReport) -> str:
    return json.dumps(
        {
            "goal": report.goal,
            "success": report.success,
            "limits": dataclasses.asdict(report.limits),
            "research": _research_to_json(report.dossier),
            "plan": report.plan.as_prompt(),
            "plan_revisions": [p.as_prompt() for p in report.revisions],
            "citations": list(report.plan.citations),
            "planner_model": report.planner_model,
            "coder_model": report.coder_model,
            "violations_caught": report.violations,
            "attempts": [
                {
                    "index": attempt.index,
                    "plan_revision": attempt.plan_revision,
                    "passed": attempt.passed,
                    "error": attempt.report["error"],
                    "message": attempt.report["message"],
                    "telemetry": attempt.report["telemetry"],
                    "waypoints": attempt.waypoints,
                    "code": attempt.code,
                    "generation_s": round(attempt.generation_s, 3),
                    "duration_s": round(attempt.duration_s, 3),
                }
                for attempt in report.attempts
            ],
            "certified_code": report.certified_code,
            "certified_telemetry": report.certified_telemetry,
            "certified_waypoints": report.certified_waypoints,
        },
        indent=2,
    )


def _check_single_script(path: Path, settings: Settings, limits: PhysicsLimits) -> int:
    """Verify one script and report. Used by ``--check``."""
    code = path.read_text(encoding="utf-8-sig")
    report = asyncio.run(test_code_in_physics_sandbox(code, limits=limits, settings=settings))
    print(format_feedback(report))
    return EXIT_SUCCESS if report["success"] else EXIT_NOT_CERTIFIED


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the loop, and map the outcome onto an exit code."""
    configure_console()
    args = _build_parser().parse_args(argv)
    settings = _resolve_settings(args)
    configure_logging(settings.log_level)

    limits = (
        PhysicsLimits(max_velocity=args.max_velocity)
        if args.max_velocity is not None
        else PhysicsLimits()
    )

    try:
        if args.check is not None:
            return _check_single_script(args.check, settings, limits)

        report = run_experiment_sync(
            args.goal,
            max_attempts=args.max_attempts,
            replan_after=args.replan_after,
            settings=settings,
            limits=limits,
            print_transcript=not args.json,
        )
    except CrepidinemError as exc:
        log.error("%s", exc)
        print(f"crepidinem: {exc}", file=sys.stderr)
        return EXIT_FAILURE
    except KeyboardInterrupt:
        print("crepidinem: interrupted", file=sys.stderr)
        return EXIT_FAILURE

    if args.json:
        print(_report_to_json(report))

    return EXIT_SUCCESS if report.success else EXIT_NOT_CERTIFIED


if __name__ == "__main__":
    raise SystemExit(main())
