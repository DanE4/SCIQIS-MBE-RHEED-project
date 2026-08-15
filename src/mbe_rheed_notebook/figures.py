"""Figure builders for the notebook.

Every function takes plain arrays or already-loaded JSON and returns a Plotly or Matplotlib
figure. Nothing here reads global state or touches marimo, so each one can be called from a
plain Python REPL or a test.
"""

import numpy as np
import plotly.graph_objects as go
from matplotlib.figure import Figure
from plotly.subplots import make_subplots

HEX_SYMBOL = "hexagon"
PROXY_COLOR = "#d62728"
SIMULATION_COLOR = "#1f77b4"


def _axial_to_cartesian(heights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Place axial (q, r) lattice sites at Cartesian centers so hex topology reads correctly."""
    row, column = np.indices(heights.shape)
    return (column + 0.5 * row).ravel(), (np.sqrt(3.0) / 2.0 * row).ravel()


def height_surface(heights: np.ndarray, coverage: float, zmax: int) -> go.Figure:
    """Rotatable 3D surface. `uirevision` keeps the camera fixed across playback frames."""
    extent = len(heights) - 0.5
    figure = go.Figure(
        go.Surface(
            z=heights,
            colorscale="Viridis",
            cmin=0,
            cmax=zmax,
            colorbar={"title": "height (ML)", "len": 0.7},
            hovertemplate="x=%{x}<br>y=%{y}<br>height=%{z} ML<extra></extra>",
            name="surface",
        )
    )
    figure.update_layout(
        uirevision="surface-playback",
        height=430,
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
        title=f"Surface at {coverage:.2f} ML",
        scene={
            "uirevision": "surface-playback",
            "xaxis": {"title": "array x", "range": [-0.5, extent], "autorange": False},
            "yaxis": {"title": "array y", "range": [-0.5, extent], "autorange": False},
            "zaxis": {"title": "height (ML)", "range": [0, zmax], "autorange": False},
            "aspectmode": "manual",
            "aspectratio": {"x": 1, "y": 1, "z": 0.55},
            "camera": {"eye": {"x": 1.4, "y": 1.4, "z": 1.0}},
        },
    )
    return figure


def hex_cells(heights: np.ndarray, coverage: float, zmax: int) -> go.Figure:
    """Same lattice drawn on its true six-neighbor geometry rather than a square grid."""
    row, column = np.indices(heights.shape)
    x, y = _axial_to_cartesian(heights)
    figure = go.Figure(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker={
                "symbol": HEX_SYMBOL,
                "size": max(9, min(24, 320 / len(heights))),
                "color": heights.ravel(),
                "colorscale": "Viridis",
                "cmin": 0,
                "cmax": zmax,
                "colorbar": {"title": "height (ML)", "len": 0.7},
                "line": {"color": "rgba(255,255,255,0.65)", "width": 0.5},
            },
            customdata=np.column_stack((column.ravel(), row.ravel(), heights.ravel())),
            hovertemplate=(
                "q=%{customdata[0]}<br>r=%{customdata[1]}<br>"
                "height=%{customdata[2]} ML<extra></extra>"
            ),
            name="hexagonal cells",
        )
    )
    figure.update_layout(
        height=430,
        margin={"l": 20, "r": 20, "t": 50, "b": 30},
        title=f"Six-neighbor axial lattice at {coverage:.2f} ML",
        xaxis={"title": "axial q + r/2", "scaleanchor": "y", "scaleratio": 1},
        yaxis={"title": "sqrt(3) r / 2"},
    )
    return figure


def rheed_trace(
    coverage_axis: np.ndarray,
    proxy: np.ndarray,
    frame: int,
    axis_label: str,
) -> go.Figure:
    """Full proxy trace annotated with the layer-cycle milestones and the current frame."""
    first_layer = np.flatnonzero(coverage_axis <= 1.0)
    most_stepped = int(first_layer[np.argmin(proxy[first_layer])])
    targets = np.arange(0.0, min(2.0, float(coverage_axis[-1])) + 0.01, 0.5)
    milestones = np.array([int(np.argmin(np.abs(coverage_axis - t))) for t in targets])
    coverage, current = coverage_axis[frame], proxy[frame]

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[coverage_axis[0], coverage_axis[most_stepped]],
            y=[proxy[0], proxy[most_stepped]],
            mode="markers",
            marker={"color": ["#16a34a", "#7e22ce"], "symbol": ["diamond", "x"], "size": 12},
            customdata=["initial flat surface", "most stepped stored frame through 1 ML"],
            hovertemplate=(
                "%{customdata}<br>coverage=%{x:.2f} ML<br>proxy=%{y:.3f}<extra></extra>"
            ),
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=coverage_axis,
            y=proxy,
            mode="lines",
            line={"color": PROXY_COLOR, "width": 3},
            name="normalized step-density RHEED proxy",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=coverage_axis[milestones],
            y=proxy[milestones],
            mode="markers+text",
            marker={"color": "#f59e0b", "size": 8, "symbol": "circle-open"},
            text=[f"{target:.1f}" for target in targets],
            textposition="top center",
            name="0–2 ML cycle milestones",
            hovertemplate=(
                "nearest stored coverage=%{x:.2f} ML<br>observed proxy=%{y:.3f}<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[coverage, coverage],
            y=[0, 1.03],
            mode="lines",
            line={"color": "#111827", "dash": "dot"},
            name="current frame",
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[coverage],
            y=[current],
            mode="markers",
            marker={"color": "#111827", "size": 10},
            name="current proxy",
            hovertemplate="coverage=%{x:.2f} ML<br>proxy=%{y:.3f}<extra></extra>",
        )
    )
    figure.update_layout(
        height=430,
        margin={"l": 60, "r": 10, "t": 50, "b": 60},
        title=f"RHEED proxy at {coverage:.2f} ML",
        legend={"orientation": "h", "y": -0.2},
        xaxis_title=axis_label,
        yaxis_title="normalized proxy",
        yaxis_range=[0, 1.03],
    )
    return figure


def observables(
    coverage_axis: np.ndarray,
    roughness: np.ndarray,
    island_density: np.ndarray,
    proxy: np.ndarray,
    axis_label: str,
) -> Figure:
    """Stacked roughness / island-density / proxy panels sharing one growth axis."""
    figure = Figure(figsize=(8, 7), constrained_layout=True)
    axes = figure.subplots(3, 1, sharex=True)
    axes[0].plot(coverage_axis, roughness, color="tab:blue")
    axes[0].set_ylabel("RMS roughness (ML)")
    axes[1].plot(coverage_axis, island_density, color="tab:green")
    axes[1].set_ylabel("islands / site")
    axes[2].plot(coverage_axis, proxy, color="tab:red")
    axes[2].set(xlabel=axis_label, ylabel=r"$1-S_d$ proxy")
    figure.suptitle("Growth observables")
    return figure


def sweep_panels(sweep_data: dict, selection: dict, result) -> Figure:
    """Heatmap of the stored ensemble with the selected point circled, plus that single run."""
    temperatures = np.asarray(sweep_data["temperatures_k"])
    fluxes = np.asarray(sweep_data["fluxes_ml_s"])
    amplitudes = np.asarray(sweep_data["mean_amplitude"])
    row = int(np.flatnonzero(temperatures == selection["temperature_k"])[0])
    column = int(np.flatnonzero(fluxes == selection["flux_ml_s"])[0])

    figure = Figure(figsize=(12, 3.5), constrained_layout=True)
    axes = figure.subplots(1, 3)
    heatmap = axes[0].imshow(amplitudes, origin="lower", cmap="magma", aspect="auto")
    axes[0].scatter(
        column, row, marker="s", s=120, facecolors="none", edgecolors="cyan", linewidths=2
    )
    axes[0].set(
        title="Mean proxy amplitude",
        xlabel="flux (ML/s)",
        ylabel="temperature (K)",
        xticks=range(len(fluxes)),
        xticklabels=fluxes,
        yticks=range(len(temperatures)),
        yticklabels=temperatures,
    )
    figure.colorbar(heatmap, ax=axes[0], shrink=0.8)
    surface = axes[1].imshow(result.final_heights, origin="lower", cmap="viridis")
    axes[1].set(title="Selected final morphology", xlabel="lattice x", ylabel="lattice y")
    figure.colorbar(surface, ax=axes[1], shrink=0.8, label="height (ML)")
    axes[2].plot(result.coverage_ml, result.rheed_proxy, color="tab:red")
    axes[2].set(
        title="Selected proxy trace",
        xlabel="coverage (ML)",
        ylabel=r"$1-S_d$",
        ylim=(0, 1.03),
    )
    return figure


def _format_metric(value: float | None, digits: int = 3, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def figure3_comparison(figure3_data: dict) -> tuple[go.Figure, str]:
    """Reference-vs-model panels plus the Markdown diagnostics table rows."""
    figure = make_subplots(
        rows=3,
        cols=2,
        shared_xaxes=True,
        column_titles=("Figure-derived experimental RHEED", "Morphology-derived raw proxy"),
        horizontal_spacing=0.10,
        vertical_spacing=0.08,
    )
    rows = []
    pairs = zip(figure3_data["traces"], figure3_data["comparisons"], strict=True)
    for index, (trace, comparison) in enumerate(pairs, start=1):
        time = np.asarray(trace["time_s"])
        mean = np.asarray(trace["rheed_proxy_mean"])
        std = np.asarray(trace["rheed_proxy_std"])
        ratio = trace["nominal_ga_n_ratio"]
        reference, simulation = comparison["reference"], comparison["simulation"]
        figure.add_trace(
            go.Scatter(
                x=trace["reference_time_s"],
                y=trace["reference_rheed_panel_coordinate"],
                mode="lines",
                line={"color": PROXY_COLOR, "width": 2},
                name="figure-derived RHEED" if index == 1 else None,
                showlegend=index == 1,
                hovertemplate="time=%{x:.1f} s<br>figure coordinate=%{y:.3f}<extra></extra>",
            ),
            row=index,
            col=1,
        )
        for values, fill in (
            (np.clip(mean - std, 0, 1), None),
            (np.clip(mean + std, 0, 1), "tonexty"),
        ):
            figure.add_trace(
                go.Scatter(
                    x=time,
                    y=values,
                    mode="lines",
                    line={"width": 0},
                    fill=fill,
                    fillcolor="rgba(31,119,180,0.18)",
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=index,
                col=2,
            )
        figure.add_trace(
            go.Scatter(
                x=time,
                y=mean,
                mode="lines",
                line={"color": SIMULATION_COLOR, "width": 2},
                name="proxy mean +/- 1 SD" if index == 1 else None,
                showlegend=index == 1,
                hovertemplate="time=%{x:.1f} s<br>raw 1-Sd=%{y:.3f}<extra></extra>",
            ),
            row=index,
            col=2,
        )
        figure.update_yaxes(
            title_text=f"Ga/N={ratio:.2f}<br>panel coordinate", range=[0, 1], row=index, col=1
        )
        figure.update_yaxes(title_text="raw 1-Sd", range=[0, 1], row=index, col=2)
        rows.append(
            f"| {ratio:.2f} | {_format_metric(reference['period_ml'])} / "
            f"{_format_metric(simulation['period_ml'])} | "
            f"{_format_metric(comparison['absolute_peak_phase_difference_ml'])} | "
            f"{_format_metric(reference['damping_rate_per_ml'], signed=True)} / "
            f"{_format_metric(simulation['damping_rate_per_ml'], signed=True)} | "
            f"{_format_metric(reference['relative_detrended_amplitude'])} / "
            f"{_format_metric(simulation['relative_detrended_amplitude'])} |"
        )
    for column in (1, 2):
        figure.update_xaxes(title_text="time (s)", range=[0, 40], row=3, col=column)
    figure.update_layout(
        height=800,
        margin={"l": 80, "r": 20, "t": 95, "b": 80},
        title="Figure 3 comparison: separate scales, shared time domain",
        legend={"orientation": "h", "x": 0, "y": -0.08},
    )
    return figure, "\n".join(rows)


def morphology_sequence(morphology: dict) -> go.Figure:
    """The stored layer-cycle snapshots drawn side by side on the hex lattice."""
    frames = morphology["frames"]
    figure = make_subplots(
        rows=1,
        cols=len(frames),
        subplot_titles=[
            f"target {frame['target_predicted_coverage_ml']:.1f} ML<br>"
            f"actual {frame['predicted_coverage_ml']:.2f} ML"
            for frame in frames
        ],
        horizontal_spacing=0.03,
    )
    maximum_height = max(np.max(frame["height_ml"]) for frame in frames)
    for column, frame in enumerate(frames, start=1):
        heights = np.asarray(frame["height_ml"])
        x, y = _axial_to_cartesian(heights)
        figure.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker={
                    "symbol": HEX_SYMBOL,
                    "size": 16,
                    "color": heights.ravel(),
                    "coloraxis": "coloraxis",
                    "line": {"color": "white", "width": 0.5},
                },
                customdata=heights.ravel(),
                hovertemplate="height=%{customdata} ML<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=column,
        )
        figure.update_xaxes(visible=False, row=1, col=column)
        figure.update_yaxes(visible=False, row=1, col=column)
    figure.update_layout(
        height=320,
        margin={"l": 10, "r": 50, "t": 90, "b": 20},
        title="Figure 4-inspired layer-cycle morphology - no strain or SK claim",
        coloraxis={
            "colorscale": "Viridis",
            "cmin": 0,
            "cmax": max(1, maximum_height),
            "colorbar": {"title": "height (ML)"},
        },
    )
    return figure
