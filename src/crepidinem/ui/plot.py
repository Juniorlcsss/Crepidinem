"""render a validated trajectory"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

from crepidinem.mcp_tools.harness import PhysicsLimits

__all__ = ["trajectory_figure"]

_PATH_COLOUR = "#22d3ee"
_CARRY_COLOUR = "#f59e0b"
_OBSTACLE_COLOUR = "#ef4444"
_TABLE_COLOUR = "#334155"

_MIN_SEGMENT = 2

_BOX_VERTS = (
    (0, 0, 0),
    (1, 0, 0),
    (1, 1, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 0, 1),
    (1, 1, 1),
    (0, 1, 1),
)

#two triangles per cube face
_BOX_FACES = (
    (0, 1, 2),
    (0, 2, 3),
    (4, 5, 6),
    (4, 6, 7),
    (0, 1, 5),
    (0, 5, 4),
    (2, 3, 7),
    (2, 7, 6),
    (1, 2, 6),
    (1, 6, 5),
    (0, 3, 7),
    (0, 7, 4),
)


def _box_mesh(
    box: tuple[float, float, float, float, float, float],
    *,
    name: str,
    colour: str,
    opacity: float,
) -> go.Mesh3d:
    """An axis-aligned box as a translucent solid."""
    x0, y0, z0, x1, y1, z1 = box
    xs = [x0 + (x1 - x0) * v[0] for v in _BOX_VERTS]
    ys = [y0 + (y1 - y0) * v[1] for v in _BOX_VERTS]
    zs = [z0 + (z1 - z0) * v[2] for v in _BOX_VERTS]
    return go.Mesh3d(
        x=xs,
        y=ys,
        z=zs,
        i=[f[0] for f in _BOX_FACES],
        j=[f[1] for f in _BOX_FACES],
        k=[f[2] for f in _BOX_FACES],
        color=colour,
        opacity=opacity,
        name=name,
        hoverinfo="name",
        showlegend=True,
    )


def _segment_traces(points: list[dict[str, Any]]) -> list[go.Scatter3d]:
    """The path, split so carrying a payload is visually distinct from not."""
    traces: list[go.Scatter3d] = []
    run: list[dict[str, Any]] = []
    carrying = bool(points and not points[0].get("gripper_open", True))
    labelled = {"carrying": False, "empty": False}

    def flush(*, is_carrying: bool) -> None:
        if len(run) < _MIN_SEGMENT:
            return
        key = "carrying" if is_carrying else "empty"
        traces.append(
            go.Scatter3d(
                x=[p["x"] for p in run],
                y=[p["y"] for p in run],
                z=[p["z"] for p in run],
                mode="lines",
                line={"width": 7, "color": _CARRY_COLOUR if is_carrying else _PATH_COLOUR},
                name="carrying payload" if is_carrying else "travelling empty",
                showlegend=not labelled[key],
                hoverinfo="skip",
            )
        )
        labelled[key] = True

    for point in points:
        state = not point.get("gripper_open", True)
        if state != carrying and run:
            flush(is_carrying=carrying)
            run = [run[-1]]
            carrying = state
        run.append(point)
    flush(is_carrying=carrying)
    return traces


def trajectory_figure(
    waypoints: list[dict[str, Any]],
    limits: PhysicsLimits,
    *,
    title: str = "Validated trajectory",
    failure_point: tuple[float, float, float] | None = None,
) -> go.Figure:
    """Draw ``waypoints`` inside the cell the harness enforced.

    Args:
        waypoints: Poses recorded by the harness, in order.
        limits: The constraints that were applied, used to draw the cell.
        title: Figure title.
        failure_point: Where a violation occurred, marked in red when given.

    Returns:
        A Plotly figure. Empty input yields an empty-but-labelled cell rather
        than an exception, so the dashboard can render before a run finishes.
    """
    figure = go.Figure()

    x_lo, x_hi = limits.workspace_x
    y_lo, y_hi = limits.workspace_y
    floor = limits.table_height

    figure.add_trace(
        _box_mesh(
            (x_lo, y_lo, floor - 0.01, x_hi, y_hi, floor),
            name="table",
            colour=_TABLE_COLOUR,
            opacity=0.35,
        )
    )
    for index, box in enumerate(limits.obstacles):
        figure.add_trace(
            _box_mesh(box, name=f"obstacle {index}", colour=_OBSTACLE_COLOUR, opacity=0.45)
        )

    if waypoints:
        for trace in _segment_traces(waypoints):
            figure.add_trace(trace)

        figure.add_trace(
            go.Scatter3d(
                x=[p["x"] for p in waypoints],
                y=[p["y"] for p in waypoints],
                z=[p["z"] for p in waypoints],
                mode="markers",
                marker={"size": 4, "color": _PATH_COLOUR},
                name="waypoints",
                text=[
                    f"{p.get('cmd', '?')} #{p.get('index', 0)}<br>"
                    f"v={p.get('velocity', 0):.3f} m/s<br>"
                    f"t={p.get('sim_time', 0):.2f} s"
                    for p in waypoints
                ],
                hovertemplate="%{text}<br>(%{x:.3f}, %{y:.3f}, %{z:.3f})<extra></extra>",
            )
        )
        first, last = waypoints[0], waypoints[-1]
        figure.add_trace(
            go.Scatter3d(
                x=[first["x"], last["x"]],
                y=[first["y"], last["y"]],
                z=[first["z"], last["z"]],
                mode="markers+text",
                marker={"size": 8, "color": ["#22c55e", "#a855f7"], "symbol": "diamond"},
                text=["start", "end"],
                textposition="top center",
                name="endpoints",
                hoverinfo="skip",
            )
        )

    if failure_point is not None:
        figure.add_trace(
            go.Scatter3d(
                x=[failure_point[0]],
                y=[failure_point[1]],
                z=[failure_point[2]],
                mode="markers+text",
                marker={"size": 10, "color": _OBSTACLE_COLOUR, "symbol": "x"},
                text=["violation"],
                textposition="top center",
                name="violation",
            )
        )

    figure.update_layout(
        title=title,
        height=560,
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        scene={
            "xaxis": {"title": "x (m)", "range": [x_lo, x_hi]},
            "yaxis": {"title": "y (m)", "range": [y_lo, y_hi]},
            "zaxis": {"title": "z (m)", "range": list(limits.workspace_z)},
            "aspectmode": "cube",
            "camera": {"eye": {"x": 1.5, "y": 1.5, "z": 1.1}},
        },
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.08},
    )
    return figure
