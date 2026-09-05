"""Crepidinem dashboard"""

from __future__ import annotations

import dataclasses
import time
from typing import Any

import streamlit as st

from crepidinem.config import Settings, load_settings
from crepidinem.logging_setup import configure_logging
from crepidinem.mcp_tools.harness import PhysicsLimits
from crepidinem.orchestrator.core import DEFAULT_GOAL, DEFAULT_MAX_ATTEMPTS
from crepidinem.ui.plot import trajectory_figure
from crepidinem.ui.runner import BackgroundRun
from crepidinem.ui.state import AttemptView, DashboardState

__all__ = ["main"]

_POLL_INTERVAL = 0.35

_FORGE_KEY = "forge_attempt"
_FORGE_SEEN = "forge_attempts_seen"

_EXAMPLE_GOALS = (
    DEFAULT_GOAL,
    "Move the robotic arm from coordinates [0, 0, 0.3] to [0.5, 0.5, 0.2], "
    "avoiding the reagent rack in the centre of the bench.",
    "Using a Universal Robots UR5e, transfer a 0.4 kg sample container from "
    "the balance at [0.3, -0.3] to the centrifuge at [0.3, 0.3].",
    "Pipette 50 uL from the vial at station A into the well plate at station B "
    "as quickly as the arm allows.",
)


# setup

def _init_state() -> None:
    st.session_state.setdefault("run", None)
    st.session_state.setdefault("view", DashboardState())
    st.session_state.setdefault("settings", load_settings())


def _sidebar(settings: Settings) -> tuple[Settings, PhysicsLimits, int]:
    """render controls and return the config"""
    with st.sidebar:
        st.header("Cell configuration")

        st.caption(
            "Planning runs on Nemotron 3 Ultra and coding on Nano, both through "
            "Nebius Token Factory. Every run makes real inference calls."
        )
        backend = st.selectbox(
            "Sandbox backend",
            options=("local", "docker", "nebius"),
            index=("local", "docker", "nebius").index(settings.sandbox_backend),
            help=(
                "local: a subprocess, fast but not an isolation boundary. "
                "docker: a locked-down container - use this for code you have "
                "not read."
            ),
        )
        max_attempts = st.slider("Max attempts", 1, 6, DEFAULT_MAX_ATTEMPTS)

        st.divider()
        st.subheader("Physics limits")
        st.caption(
            "The simulator enforces exactly these numbers, and the agents are "
            "briefed with exactly these numbers. There is one source of truth."
        )
        max_velocity = st.slider("Max tool-tip speed (m/s)", 0.05, 2.0, 0.5, step=0.05)
        max_payload = st.slider("Max payload (kg)", 0.05, 5.0, 1.0, step=0.05)

        brief_limits = st.checkbox(
            "Brief the agents with the limits",
            value=settings.brief_limits,
            help=(
                "Uncheck to withhold the cell's numbers. The guardrail then has "
                "to catch an under-informed model, instead of the prompt "
                "preventing the mistake."
            ),
        )

        st.divider()
        st.subheader("Hardware research")
        research = st.checkbox(
            "Ground limits in published specs (Tavily)",
            value=settings.research_enabled and bool(settings.tavily_api_key),
            disabled=not settings.tavily_api_key,
            help=(
                "Search for the hardware named in the goal and enforce its real "
                "published limits. Requires TAVILY_API_KEY and Nemotron agents."
            ),
        )
        if not settings.tavily_api_key:
            st.caption("TAVILY_API_KEY is not set, so the cell defaults apply.")
        trust = st.selectbox(
            "Trust in retrieved values",
            options=("envelope", "narrow-only"),
            index=("envelope", "narrow-only").index(settings.research_trust),
            disabled=not research,
            help=(
                "narrow-only: retrieved text may tighten a safety limit but "
                "never loosen one. envelope: loosening is allowed inside a "
                "hard-coded plausible range, and only when quoted from the "
                "cited document."
            ),
        )

        if not settings.nebius_api_key:
            st.error("NEBIUS_API_KEY is not set. Fill it in .env before running.")

    resolved = dataclasses.replace(
        settings,
        sandbox_backend=backend,
        brief_limits=brief_limits,
        research_enabled=research,
        research_trust=trust,
    )
    limits = PhysicsLimits(max_velocity=max_velocity, max_payload_kg=max_payload)
    return resolved, limits, max_attempts


#panes


def _brain_pane(view: DashboardState) -> None:
    st.subheader("The Brain")
    st.caption(f"Principal Investigator - {view.planner_model or 'Nemotron 3 Ultra'}")

    if view.research:
        _research_block(view.research)

    if view.pi_reasoning:
        with st.expander("Reasoning trace", expanded=not view.plan):
            st.markdown(f"```text\n{view.pi_reasoning[-4000:]}\n```")

    if view.plan_steps:
        st.markdown("**Plan**")
        for number, step in enumerate(view.plan_steps, start=1):
            st.markdown(f"{number}. {step}")
    elif view.pi_answer:
        st.markdown(f"```text\n{view.pi_answer[-4000:]}\n```")
    else:
        st.info("Waiting for the Principal Investigator...")


def _research_block(research: dict[str, Any]) -> None:
    """show what search changed and what it was refused"""
    findings = research.get("findings")
    rows = findings if isinstance(findings, list) else []
    applied = [f for f in rows if f.get("accepted")]
    with st.expander(
        f"Hardware research - {len(applied)} limit(s) applied", expanded=bool(applied)
    ):
        st.caption(str(research.get("summary") or ""))
        for finding in rows:
            mark = "applied" if finding.get("accepted") else "rejected"
            previous = finding.get("previous")
            was = f" (was {previous:g})" if isinstance(previous, int | float) else ""
            st.markdown(
                f"- **{finding.get('field')}** = {finding.get('value')} "
                f"{finding.get('unit', '')}{was} — _{mark}_: {finding.get('reason')}"
            )
            url = str(finding.get("source_url") or "")
            if url:
                st.caption(f"source: {url}")
        for note in research.get("notes") or []:
            st.caption(str(note))


def _forge_pane(view: DashboardState) -> None:
    st.subheader("The Forge")
    st.caption(f"Code Smith - {view.coder_model or 'Nemotron Nano'}")

    current = view.current
    if current is None:
        st.info("Waiting for a plan to build from...")
        return

    labels = [f"Attempt {a.index} · {a.badge}" for a in view.attempts]
    latest = len(view.attempts) - 1

    if st.session_state.get(_FORGE_SEEN) != len(view.attempts):
        st.session_state[_FORGE_SEEN] = len(view.attempts)
        st.session_state[_FORGE_KEY] = latest

    chosen = st.radio(
        "Attempt",
        options=list(range(len(view.attempts))),
        format_func=lambda i: labels[i],
        horizontal=True,
        label_visibility="collapsed",
        key=_FORGE_KEY,
    )
    selected = view.attempts[chosen]

    if selected.feedback:
        with st.expander("Guardrail feedback fed back into this attempt", expanded=False):
            st.code(selected.feedback, language="text")

    code = selected.display_code
    if code:
        st.code(code, language="python")
    else:
        st.info("Generating...")
    if selected.generation_s:
        st.caption(f"{selected.generation_s:.1f}s to generate")


def _crucible_pane(view: DashboardState) -> None:
    st.subheader("The Crucible")
    st.caption(f"Deterministic physics sandbox - backend: {view.backend or 'local'}")

    if view.limits:
        st.markdown("**Enforced limits**")
        st.markdown(
            f"- speed ≤ **{view.limits.get('max_velocity')} m/s**\n"
            f"- acceleration ≤ **{view.limits.get('max_acceleration')} m/s²**\n"
            f"- payload ≤ **{view.limits.get('max_payload_kg')} kg**\n"
            f"- gripper ≤ **{view.limits.get('max_gripper_force_n')} N**\n"
            f"- tool tip ≥ **{view.limits.get('table_clearance')} m** above the table"
        )

    if not view.attempts:
        st.info("Nothing has been submitted to the sandbox yet.")
        return

    st.markdown("**Verdicts**")
    for attempt in view.attempts:
        if attempt.passed is None:
            st.warning(f"Attempt {attempt.index}: running...")
            continue

        if attempt.passed:
            st.success(f"Attempt {attempt.index}: CERTIFIED SAFE\n\n{attempt.verdict}")
            continue

        st.error(f"Attempt {attempt.index}: {attempt.error}")
        st.caption(attempt.verdict.strip()[:600])

    if view.finished:
        if view.certified:
            st.metric("Violations caught before hardware", len(view.violations))

        else:
            st.metric("Certified", "no", delta="nothing released", delta_color="inverse")


def _execution_pane(view: DashboardState) -> None:
    st.subheader("The Execution")

    waypoints = view.plot_waypoints
    if not waypoints:
        st.info(
            "The trajectory appears here once the harness has run a script. "
            "It is drawn from the harness's own observations, not from a second "
            "simulation in the browser."
        )
        return

    current = view.current
    failure = None
    title = "Validated trajectory"
    if not view.certified and current is not None and current.passed is False:
        failure = current.failure_point
        title = f"Rejected trajectory - {current.error}"

    limits = PhysicsLimits(
        max_velocity=float(view.limits.get("max_velocity", 0.5)),
        workspace_x=tuple(view.limits.get("workspace_x", (-0.6, 0.6))),
        workspace_y=tuple(view.limits.get("workspace_y", (-0.6, 0.6))),
        workspace_z=tuple(view.limits.get("workspace_z", (-0.10, 0.80))),
        table_height=float(view.limits.get("table_height", 0.0)),
        table_clearance=float(view.limits.get("table_clearance", 0.02)),
        obstacles=tuple(tuple(b) for b in view.limits.get("obstacles", ())),
    )
    st.plotly_chart(
        trajectory_figure(waypoints, limits, title=title, failure_point=failure),
        use_container_width=True,
    )

    if view.certified_telemetry:
        columns = st.columns(4)
        telemetry = view.certified_telemetry
        columns[0].metric("Commands", telemetry.get("commands", 0))
        columns[1].metric("Path length", f"{telemetry.get('path_length_m', 0):.3f} m")
        columns[2].metric("Simulated time", f"{telemetry.get('sim_time_s', 0):.2f} s")
        columns[3].metric("Waypoints", len(waypoints))


def _verdict_banner(view: DashboardState) -> None:
    if view.error:
        st.error(f"The run could not complete: {view.error}")
        return
    if not view.finished:
        return
    if view.certified:
        st.success(
            f"CERTIFIED after {len(view.attempts)} attempt(s). "
            f"{len(view.violations)} violation(s) caught before hardware."
        )
    else:
        st.error(
            f"NOT CERTIFIED after {len(view.attempts)} attempt(s). Nothing is released to hardware."
        )


def _attempt_strip(attempts: list[AttemptView]) -> None:
    """A one-line summary of the loop so far."""
    if not attempts:
        return
    parts = []
    for attempt in attempts:
        if attempt.passed is None:
            parts.append(f"⏳ {attempt.index}")
        elif attempt.passed:
            parts.append(f"✅ {attempt.index} certified")
        else:
            parts.append(f"❌ {attempt.index} {attempt.error}")
    st.markdown(" &nbsp;→&nbsp; ".join(parts))


#page


def main() -> None:
    """Render one pass of the dashboard."""
    st.set_page_config(page_title="Crepidinem", page_icon="🦾", layout="wide")
    _init_state()
    configure_logging(st.session_state["settings"].log_level)

    st.title("Crepidinem")
    st.caption(
        "An LLM hallucination in software is a bug. In a lab it is a broken arm. "
        "Nothing reaches hardware until a deterministic physics sandbox has certified it."
    )

    settings, limits, max_attempts = _sidebar(st.session_state["settings"])

    goal = st.text_area(
        "Scientific goal",
        value=_EXAMPLE_GOALS[0],
        height=80,
        help="Name specific hardware to let the research step look up its real limits.",
    )
    left, right = st.columns([1, 4])
    launch = left.button("Run experiment", type="primary", use_container_width=True)
    with right.expander("Example goals"):
        for example in _EXAMPLE_GOALS:
            st.markdown(f"- {example}")

    run: BackgroundRun | None = st.session_state["run"]

    if launch:
        run = BackgroundRun(
            goal.strip() or DEFAULT_GOAL,
            settings=settings,
            limits=limits,
            max_attempts=max_attempts,
        )
        st.session_state["run"] = run
        st.session_state["view"] = DashboardState(goal=goal)
        st.session_state.pop(_FORGE_KEY, None)
        st.session_state.pop(_FORGE_SEEN, None)
        run.start()

    view: DashboardState = st.session_state["view"]
    if run is not None:
        view.apply_all(run.drain())
        if run.finished and run.error and not view.error:
            view.error = run.error

    _verdict_banner(view)
    _attempt_strip(view.attempts)

    brain, forge, crucible = st.columns(3, gap="medium")
    with brain:
        _brain_pane(view)
        
    with forge:
        _forge_pane(view)

    with crucible:
        _crucible_pane(view)

    st.divider()
    _execution_pane(view)

    if view.status:
        with st.expander("Run log"):
            for line in view.status:
                st.text(line)

    if run is not None and run.running:
        time.sleep(_POLL_INTERVAL)
        st.rerun()


if __name__ == "__main__":
    main()
