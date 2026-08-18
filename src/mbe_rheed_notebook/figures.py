"""Figure builders for the notebook.

Every function takes plain arrays or already-loaded JSON and returns a Plotly or Matplotlib
figure. Nothing here reads global state or touches marimo, so each one can be called from a
plain Python REPL or a test.
"""

import numpy as np
import plotly.graph_objects as go
from matplotlib.figure import Figure
from plotly.subplots import make_subplots

from mbe_rheed_sim.lattice import HEX_DIRECTIONS
from mbe_rheed_sim.observables import step_density
from mbe_rheed_sim.rheed import ScreenPattern

HEX_SYMBOL = "hexagon"
PROXY_COLOR = "#d62728"
SIMULATION_COLOR = "#1f77b4"
BEAM_COLOR = "#ff7f0e"
SPECULAR_COLOR = "#0ea5e9"
# Typical RHEED incidence is 1-3 degrees. This is a visualization
# assumption for drawing the geometry, not a value taken from the primary paper, which reports
# no beam angle. Nothing downstream of this figure consumes it.
GRAZING_ANGLE_DEG = 2.0
# Mandated wording for any beam/detector overlay (STATUS.md Stage 6J). Kept as a constant so the
# figure and its test cannot drift apart from the requirement.
BEAM_GEOMETRY_LABEL = "explanatory geometry only — diffraction is not simulated"
# Shown wherever a computed pattern is displayed, so a kinematic image is never mistaken for
# the dynamical scattering a real RHEED screen records.
DIFFRACTION_LABEL = "kinematic single scattering only — not dynamical RHEED"
# Decades of intensity shown below the flat-surface specular value. Diffuse scattering from a
# rough surface sits three to four decades down, so a linear screen would look empty.
SCREEN_LOG_DECADES = 5.0
# One monolayer of height drawn as one in-plane site spacing. For GaN that is c/2 = 0.259 nm
# against a = 0.319 nm, so the true vertical:lateral aspect is 0.81 of what the beam shows.
ML_PER_SITE_SPACING = 1.0


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


def step_edges(heights: np.ndarray, coverage: float, zmax: int) -> go.Figure:
    """Where the steps are, drawn on the same hex geometry the proxy is measured on.

    `zmax` is accepted for a common signature with the other surface views; the colour scale
    here is the fixed 0–6 count of unequal neighbours, not height.
    """
    row, column = np.indices(heights.shape)
    x, y = _axial_to_cartesian(heights)
    unequal = sum(
        heights != np.roll(heights, shift=(-dy, -dx), axis=(0, 1)) for dy, dx in HEX_DIRECTIONS
    )
    figure = go.Figure(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker={
                "symbol": HEX_SYMBOL,
                "size": max(9, min(24, 320 / len(heights))),
                "color": unequal.ravel(),
                "colorscale": "Inferno",
                "cmin": 0,
                "cmax": 6,
                "colorbar": {"title": "stepped<br>neighbours", "len": 0.7},
                "line": {"color": "rgba(255,255,255,0.35)", "width": 0.5},
            },
            customdata=np.column_stack(
                (column.ravel(), row.ravel(), heights.ravel(), unequal.ravel())
            ),
            hovertemplate=(
                "q=%{customdata[0]}<br>r=%{customdata[1]}<br>height=%{customdata[2]} ML"
                "<br>stepped neighbours=%{customdata[3]} of 6<extra></extra>"
            ),
            name="step edges",
        )
    )
    density = step_density(heights)
    figure.update_layout(
        height=430,
        margin={"l": 20, "r": 20, "t": 70, "b": 30},
        title=(
            f"Step edges at {coverage:.2f} ML"
            f"<br><sub>S_d = {density:.3f}, so the proxy 1 - S_d = {1 - density:.3f}</sub>"
        ),
        xaxis={"title": "axial q + r/2", "scaleanchor": "y", "scaleratio": 1},
        yaxis={"title": "sqrt(3) r / 2"},
    )
    return figure


def _screen_decades(pattern: ScreenPattern) -> np.ndarray:
    """Screen intensity in decades below a flat surface, floored so log10 stays finite."""
    return np.log10(np.maximum(pattern.intensity, 10.0**-SCREEN_LOG_DECADES))


def detector_screen(pattern: ScreenPattern, coverage: float) -> go.Figure:
    """The computed detector image, as the screen itself rather than as a single number."""
    figure = go.Figure(
        go.Heatmap(
            x=pattern.deflection_deg,
            y=pattern.exit_angle_deg,
            z=_screen_decades(pattern),
            colorscale="Inferno",
            zmin=-SCREEN_LOG_DECADES,
            zmax=0.0,
            colorbar={
                "title": "log<sub>10</sub> I<br>(flat = 0)",
                "len": 0.7,
                "tickvals": list(range(-int(SCREEN_LOG_DECADES), 1)),
            },
            hovertemplate=(
                "deflection=%{x:.2f}°<br>exit angle=%{y:.2f}°"
                "<br>log10 I=%{z:.2f}<extra></extra>"
            ),
            name="detector screen",
        )
    )
    figure.add_hline(
        y=0.0,
        line={"color": "#94a3b8", "dash": "dot", "width": 1},
        annotation={"text": "shadow edge", "font": {"color": "#94a3b8", "size": 10}},
        annotation_position="top left",
    )
    figure.add_trace(
        go.Scatter(
            x=[0.0],
            y=[pattern.grazing_angle_deg],
            mode="markers",
            marker={"symbol": "circle-open", "color": "#38bdf8", "size": 16, "line": {"width": 2}},
            name="specular (00) beam",
            hovertemplate=(
                f"specular intensity {pattern.specular_intensity:.4f}"
                " of a flat surface<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        height=430,
        margin={"l": 60, "r": 10, "t": 76, "b": 60},
        legend={"orientation": "h", "y": -0.2},
        title=(
            f"Detector screen at {coverage:.2f} ML — specular "
            f"{pattern.specular_intensity:.4f} of flat"
            f"<br><sub>{pattern.beam_energy_kev:g} keV, {pattern.grazing_angle_deg:.2f}° "
            f"grazing, q_z d / pi = {pattern.phase_order:.2f} ({pattern.condition}) · "
            f"{DIFFRACTION_LABEL}</sub>"
        ),
        # Equal angular scales, and `constrain` shrinks the drawing area to the data rather
        # than padding the screen out with empty angles.
        xaxis={
            "title": "horizontal deflection (degrees)",
            "scaleanchor": "y",
            "scaleratio": 1,
            "constrain": "domain",
            "showgrid": False,
            "zeroline": False,
        },
        yaxis={
            "title": "exit angle above surface (degrees)",
            "showgrid": False,
            "zeroline": False,
        },
        plot_bgcolor="#000000",
    )
    return figure


def rheed_geometry(
    heights: np.ndarray,
    coverage: float,
    zmax: int,
    *,
    grazing_angle_deg: float = GRAZING_ANGLE_DEG,
) -> go.Figure:
    """The simulated surface with the RHEED beam and screen placed on it.

    Geometry only: where the beam comes in, and where the specular ray leaves. What lands on
    this screen is computed by `detector_screen` and drawn on its own angular axes underneath.
    It is deliberately not painted onto this screen: the specular spot sits about a degree
    above the shadow edge, so any pattern large enough to read at this scale would require the
    drawn beam angle to be a lie, and these rays carry the true angle in data coordinates.

    The z axis is stretched exactly as in `height_surface` so a few monolayers stay visible
    against a lattice hundreds of sites wide. The rendered beam therefore looks far steeper
    than it is; the returned title states both the true angle and the stretch factor.
    """
    if heights.ndim != 2 or min(heights.shape) < 2 or not 0 < grazing_angle_deg < 45:
        raise ValueError("beam geometry needs a 2D lattice and a grazing angle in (0, 45)")
    rows, columns = heights.shape
    slope = np.tan(np.radians(grazing_angle_deg)) * ML_PER_SITE_SPACING
    # Impact at the lattice centre, on the real local height so the ray meets the drawn surface.
    impact_x, impact_y = 0.5 * (columns - 1), 0.5 * (rows - 1)
    impact_z = float(heights[rows // 2, columns // 2])
    # Half a lattice width of run-up each side: enough for the grazing rise to read, while
    # keeping the drawn box near 2:1 so the wide scene still fits the default camera framing.
    standoff = 0.5 * columns
    entry_x, screen_x = -standoff, (columns - 1) + standoff
    entry_z = impact_z + (impact_x - entry_x) * slope
    spot_z = impact_z + (screen_x - impact_x) * slope

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
    for label, xs, zs in (
        ("incident beam", (entry_x, impact_x), (entry_z, impact_z)),
        ("specular direction", (impact_x, screen_x), (impact_z, spot_z)),
    ):
        figure.add_trace(
            go.Scatter3d(
                x=xs,
                y=(impact_y, impact_y),
                z=zs,
                mode="lines",
                line={"color": BEAM_COLOR, "width": 6},
                name=label,
                hovertemplate=f"{label}<extra></extra>",
            )
        )
    screen_top = max(float(zmax), spot_z * 1.6)
    screen_half_width = 0.35 * rows
    figure.add_trace(
        go.Mesh3d(
            x=[screen_x] * 4,
            y=[
                impact_y - screen_half_width,
                impact_y + screen_half_width,
                impact_y + screen_half_width,
                impact_y - screen_half_width,
            ],
            z=[0.0, 0.0, screen_top, screen_top],
            i=[0, 0],
            j=[1, 2],
            k=[2, 3],
            color="#94a3b8",
            opacity=0.28,
            name="detector screen",
            hovertemplate="detector screen<extra></extra>",
            showlegend=True,
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=[screen_x],
            y=[impact_y],
            z=[spot_z],
            mode="markers",
            marker={"color": BEAM_COLOR, "size": 6, "symbol": "circle"},
            name="specular spot",
            hovertemplate="specular spot (geometric, not diffracted)<extra></extra>",
        )
    )

    z_top = max(float(zmax), entry_z, screen_top)
    x_span = screen_x - entry_x
    # Both axes are in site units, so the disclosed stretch is the ratio of drawn units per
    # data unit on z against x, using the aspect numbers set immediately below.
    x_aspect = x_span / columns
    stretch = (0.55 / z_top) / (x_aspect / x_span)
    figure.update_layout(
        uirevision="beam-geometry",
        height=430,
        margin={"l": 0, "r": 0, "t": 70, "b": 0},
        legend={"orientation": "h", "y": -0.02},
        title=(
            f"Beam geometry at {coverage:.2f} ML — {grazing_angle_deg:.2f}° grazing incidence, "
            + (
                f"z stretched {stretch:.0f}x for visibility"
                if stretch >= 1.5
                else "drawn to scale"
            )
            + f"<br><sub>{BEAM_GEOMETRY_LABEL}</sub>"
        ),
        scene={
            "uirevision": "beam-geometry",
            "xaxis": {"title": "array x", "range": [entry_x, screen_x], "autorange": False},
            "yaxis": {"title": "array y", "range": [-0.5, rows - 0.5], "autorange": False},
            "zaxis": {"title": "height (ML)", "range": [0, z_top], "autorange": False},
            "aspectmode": "manual",
            "aspectratio": {"x": x_aspect, "y": 1, "z": 0.55},
            # Near side-on, looking along the beam's plane, so grazing incidence reads as grazing.
            "camera": {"eye": {"x": 0.15, "y": -2.9, "z": 0.5}},
        },
    )
    return figure


def rheed_trace(
    coverage_axis: np.ndarray,
    proxy: np.ndarray,
    frame: int,
    axis_label: str,
    specular: np.ndarray | None = None,
) -> go.Figure:
    """Full proxy trace annotated with the layer-cycle milestones and the current frame.

    Pass `specular` to overlay the kinematic (00) intensity for the same snapshots. The two
    curves are different quantities on a shared 0-1 scale: one counts step edges, the other
    is a diffraction calculation. Plotting them together is the check that the cheap proxy
    tracks the expensive observable.
    """
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
    if specular is not None:
        figure.add_trace(
            go.Scatter(
                x=coverage_axis,
                y=specular,
                mode="lines",
                line={"color": SPECULAR_COLOR, "width": 2, "dash": "dash"},
                name="kinematic specular (00) intensity",
                hovertemplate=(
                    "coverage=%{x:.2f} ML<br>specular=%{y:.3f} of flat<extra></extra>"
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
        title=(
            f"RHEED proxy at {coverage:.2f} ML"
            if specular is None
            else f"RHEED proxy and kinematic specular intensity at {coverage:.2f} ML"
        ),
        legend={"orientation": "h", "y": -0.2},
        xaxis_title=axis_label,
        yaxis_title="normalized proxy" if specular is None else "normalized signal",
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
