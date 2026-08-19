"""Figure builders for the notebook.

Every function takes plain arrays or already-loaded JSON and returns a Plotly or Matplotlib
figure. Nothing here reads global state or touches marimo, so each one can be called from a
plain Python REPL or a test.
"""

import numpy as np
import plotly.graph_objects as go
from matplotlib.figure import Figure
from plotly.subplots import make_subplots

from mbe_rheed_sim import rheed
from mbe_rheed_sim.lattice import HEX_DIRECTIONS
from mbe_rheed_sim.observables import step_density
from mbe_rheed_sim.rheed import SCREEN_LOG_DECADES, ScreenPattern, screen_decades

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
# The geometry view paints the computed screen, so its label must not claim nothing was
# calculated (the retired BEAM_GEOMETRY_LABEL said exactly that). What it must still say is that
# the rays and the plane are drawn for explanation, at a disclosed distortion.
GEOMETRY_LABEL = "geometry view only — rays and plane are explanatory, the painted screen is the computed one"

# Angular acceptance of the drawn detector in the reachable-orders mode, matching the span the
# azimuth sweep and validate_rheed.py already use. rod_orders still decides what is reachable
# inside it: every order is reachable somewhere -- 40 of them, out past 40 deg exit at 15 keV --
# and drawing those would picture the max_order loop rather than a RHEED screen.
ORDERS_ACCEPTANCE_DEG = 9.0
# Shown wherever a computed pattern is displayed, so a kinematic image is never mistaken for
# the dynamical scattering a real RHEED screen records.
DIFFRACTION_LABEL = "kinematic single scattering only — not dynamical RHEED"
# Decades of intensity shown below the flat-surface specular value. A real screen is viewed
# well short of this range; three keeps the rods bright and the background near black while
# still showing the diffuse scattering that roughening produces.
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


def detector_screen(pattern: ScreenPattern, coverage: float) -> go.Figure:
    """The computed detector image, as the screen itself rather than as a single number."""
    figure = go.Figure(
        go.Heatmap(
            x=pattern.deflection_deg,
            y=pattern.exit_angle_deg,
            z=screen_decades(pattern),
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
    # Only the orders whose rods actually cut the Ewald sphere inside this screen, at the
    # angles the Ewald construction puts them. Nothing here is positioned by hand, and an
    # order that the geometry does not reach is simply not drawn.
    figure.add_trace(
        go.Scatter(
            x=[rod.deflection_deg for rod in pattern.rods],
            y=[rod.exit_angle_deg for rod in pattern.rods],
            mode="markers+text",
            marker={"symbol": "cross-thin", "color": "#94a3b8", "size": 9, "line": {"width": 1}},
            text=[rod.label for rod in pattern.rods],
            textposition="top center",
            textfont={"color": "#94a3b8", "size": 11},
            name="predicted (hk) rod",
            hovertemplate=(
                "%{text} rod<br>deflection=%{x:.3f}°<br>exit angle=%{y:.3f}°<extra></extra>"
            ),
        )
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
        height=460,
        # Three title lines; anything tighter overlaps the y-axis label.
        margin={"l": 60, "r": 10, "t": 112, "b": 60},
        legend={"orientation": "h", "y": -0.2},
        title=(
            f"<b>Kinematic RHEED screen</b> at {coverage:.2f} ML — specular "
            f"{pattern.specular_intensity:.4f} of flat"
            f"<br><sub>{pattern.beam_energy_kev:g} keV · "
            f"{pattern.grazing_angle_deg:.2f}° grazing · "
            f"{pattern.azimuth_deg:g}° azimuth · "
            f"q_z d / π = {pattern.phase_order:.2f} ({pattern.condition}) · "
            f"{pattern.coherence_length_nm:g} nm coherence · "
            f"rods in view: {', '.join(rod.label for rod in pattern.rods)}"
            f"<br>{DIFFRACTION_LABEL}</sub>"
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
    azimuth_deg: float = 0.0,
    pattern: ScreenPattern | None = None,
    show_orders: bool = False,
) -> go.Figure:
    """The surface, the beam, the detector plane, and the computed screen painted on it.

    Geometry, not a diffraction calculation: nothing here is a rod this figure invented. Every
    direction comes from `mbe_rheed_sim.rheed` -- `beam_geometry` for the lab frame,
    `rod_orders` for which `(h, k)` are reachable, `outgoing_direction` for each one's `k_f`,
    `detector_intersection` for where that ray crosses the plane -- and the only intensities
    drawn are `pattern`'s own, the same `ScreenPattern` the 2D detector figure shows. No ray or
    spot is positioned by hand.

    With `show_orders`, every reachable order gets a ray to its own intersection, `(00)` kept
    distinct as the nominal specular. Azimuth turns the sample about its normal, so `k_i`, `n`
    and `k_f(00)` are fixed while the non-specular orders move: that is the whole point of the
    mode. Rays stop at the plane rather than continuing, because a streak is an intensity
    distribution and not one electron trajectory -- the painted screen is where its shape lives.

    Two disclosed distortions, both stated in the title, with a true-aspect side view inset for
    the honest angle. The z axis is stretched as in `height_surface`, so drawn ray angles are
    not to scale. The painted screen keeps its true angular size vertically and its true aspect
    ratio, so its horizontal extent in lattice units carries that same stretch -- and the order
    rays are mapped through the identical transform, so a ray lands exactly on its own feature
    in the painted screen instead of beside it.
    """
    if heights.ndim != 2 or min(heights.shape) < 2 or not 0 < grazing_angle_deg < 45:
        raise ValueError("beam geometry needs a 2D lattice and a grazing angle in (0, 45)")
    geometry = rheed.beam_geometry(
        grazing_angle_deg=grazing_angle_deg, azimuth_deg=azimuth_deg
    )
    rows, columns = heights.shape

    # Physical surface positions, not array indices: the diffraction code places the scatterer
    # of column (r, c) at a(c + r/2, sqrt(3)/2 r), so the geometry view uses the same lattice.
    row_index, column_index = np.indices(heights.shape)
    surface_x = column_index + 0.5 * row_index
    surface_y = np.sqrt(3.0) / 2.0 * row_index
    centre_x, centre_y = 0.5 * surface_x.max(), 0.5 * surface_y.max()
    offset = np.stack(((surface_x - centre_x).ravel(), (surface_y - centre_y).ravel()))
    # Bounding radius of the unrotated sample, so it is rotation invariant. Every fixed part of
    # the scene is sized from this: the beam, the plane and the axes must not move when only the
    # sample turns, or the azimuth would appear to change the instrument.
    radius = float(np.hypot(*offset).max())
    # Rotate the sample about its own normal, through the centre. The morphology is untouched:
    # only where each column sits in the lab frame changes.
    rotated = geometry.sample_rotation @ offset
    surface_x = rotated[0].reshape(heights.shape) + centre_x
    surface_y = rotated[1].reshape(heights.shape) + centre_y

    impact_z = float(heights[rows // 2, columns // 2])
    # One sample radius of run-up each side, so the grazing rise reads without the scene
    # becoming mostly empty space.
    standoff = radius
    entry_x, screen_x = centre_x - standoff, centre_x + standoff
    # Follow k_i backwards from the impact point rather than restating tan(grazing) here.
    entry_z = impact_z - float(
        geometry.incident_direction[2] * standoff / geometry.incident_direction[0]
    )

    coherence_nm = (
        pattern.coherence_length_nm if pattern is not None else rheed.DEFAULT_COHERENCE_LENGTH_NM
    )

    # rod_orders is the only authority for what is reachable: it solves the Ewald intersection
    # and returns nothing for an order the sphere never touches. Intersect each one's own k_f
    # with the plane; these are true plane offsets, independent of how the scene is drawn.
    rods = (
        rheed.rod_orders(
            grazing_angle_deg=grazing_angle_deg,
            azimuth_deg=azimuth_deg,
            span_deg=ORDERS_ACCEPTANCE_DEG,
        )
        if show_orders
        else ()
    )
    hits = {
        (rod.h, rod.k): rheed.detector_intersection(
            rheed.outgoing_direction(rod.exit_angle_deg, rod.deflection_deg), standoff
        )
        for rod in rods
    }
    specular_offset = rheed.detector_intersection(geometry.specular_direction, standoff)

    y_low, y_high = centre_y - radius, centre_y + radius
    x_span, y_span = screen_x - entry_x, y_high - y_low
    reach = max(
        [float(zmax), entry_z, impact_z + specular_offset[1]]
        + [impact_z + vertical for _, vertical in hits.values()]
    )
    if show_orders:
        # True aspect on all three axes, so the drawn ray angles in this mode *are* the real
        # ones and the exit angles can be read off the plane. The surface is a thin sheet at
        # this scale, which is what a grazing-incidence geometry actually looks like.
        z_top = reach * 1.12
        x_aspect = x_span / max(y_span, 1.0)
        y_aspect, z_aspect = 1.0, z_top / max(y_span, 1.0)
        stretch = 1.0
    else:
        z_top = reach * 1.45
        x_aspect, y_aspect, z_aspect = x_span / max(y_span, 1.0), 1.0, 0.55
        # Drawn length per data unit on each axis. The screen patch and the order rays share
        # this, so a square screen stays square and a ray still meets its own feature on it.
        stretch = (z_aspect / z_top) / (y_aspect / y_span)

    def plane_point(horizontal: float, vertical: float) -> tuple[float, float]:
        """One detector-plane offset in scene coordinates, through the disclosed stretch."""
        return centre_y + horizontal * stretch, impact_z + vertical

    specular_y, specular_z = plane_point(*specular_offset)

    figure = go.Figure(
        go.Surface(
            x=surface_x,
            y=surface_y,
            z=heights,
            colorscale="Viridis",
            cmin=0,
            cmax=zmax,
            colorbar={"title": "height (ML)", "len": 0.55, "x": 1.02},
            hovertemplate="lab x=%{x:.1f}<br>lab y=%{y:.1f}<br>height=%{z} ML<extra></extra>",
            name="surface",
        )
    )

    for label, xs, ys, zs, colour in (
        ("incident beam k_i", (entry_x, centre_x), (centre_y, centre_y), (entry_z, impact_z),
         BEAM_COLOR),
        ("nominal specular k_f (00)", (centre_x, screen_x), (centre_y, specular_y),
         (impact_z, specular_z), SPECULAR_COLOR),
    ):
        figure.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                line={"color": colour, "width": 7},
                name=label,
                hovertemplate=f"{label}<extra></extra>",
            )
        )
        figure.add_trace(
            go.Cone(
                x=[xs[1]],
                y=[ys[1]],
                z=[zs[1]],
                u=[xs[1] - xs[0]],
                v=[ys[1] - ys[0]],
                w=[zs[1] - zs[0]],
                sizemode="absolute",
                sizeref=0.06 * x_span,
                anchor="tip",
                colorscale=[[0, colour], [1, colour]],
                showscale=False,
                hoverinfo="skip",
                showlegend=False,
            )
        )

    normal_top = impact_z + 0.75 * z_top
    figure.add_trace(
        go.Scatter3d(
            x=[centre_x, centre_x],
            y=[centre_y, centre_y],
            z=[impact_z, normal_top],
            mode="lines+text",
            line={"color": "#16a34a", "width": 4, "dash": "dash"},
            text=["", "n"],
            textposition="top center",
            textfont={"color": "#16a34a", "size": 13},
            name="surface normal n",
            hovertemplate="surface normal<extra></extra>",
        )
    )

    # The beam samples a coherence-sized patch, not a point: rays leave this footprint, so the
    # figure never implies that diffraction happens at one geometric bounce.
    footprint_sites = coherence_nm / rheed.GAN_IN_PLANE_SPACING_NM
    turn = np.linspace(0.0, 2.0 * np.pi, 61)
    figure.add_trace(
        go.Scatter3d(
            x=centre_x + footprint_sites * np.cos(turn),
            y=centre_y + footprint_sites * np.sin(turn),
            z=np.full(turn.size, impact_z + 0.02 * z_top),
            mode="lines",
            line={"color": "#f8fafc", "width": 3},
            name=f"illuminated footprint ({coherence_nm:.1f} nm coherence)",
            hovertemplate="illuminated footprint<extra></extra>",
        )
    )

    figure.add_trace(
        go.Mesh3d(
            x=[screen_x] * 4,
            y=[centre_y - radius, centre_y + radius, centre_y + radius, centre_y - radius],
            z=[0.0, 0.0, z_top, z_top],
            i=[0, 0],
            j=[1, 2],
            k=[2, 3],
            color="#94a3b8",
            opacity=0.18,
            name="detector plane",
            hovertemplate="detector plane<extra></extra>",
            showlegend=True,
        )
    )

    if pattern is not None:
        horizontal, vertical = rheed.detector_offsets(
            pattern.exit_angle_deg[:, None], pattern.deflection_deg[None, :], standoff
        )
        figure.add_trace(
            go.Surface(
                x=np.full(pattern.intensity.shape, screen_x),
                y=centre_y + np.broadcast_to(horizontal, pattern.intensity.shape) * stretch,
                z=impact_z + np.broadcast_to(vertical, pattern.intensity.shape),
                surfacecolor=rheed.screen_decades(pattern),
                colorscale="Inferno",
                cmin=-rheed.SCREEN_LOG_DECADES,
                cmax=0.0,
                showscale=False,
                name="computed screen",
                showlegend=True,
                hovertemplate=(
                    "exit=%{z:.2f}<br>log10 I=%{surfacecolor:.2f}<extra>computed screen</extra>"
                ),
            )
        )

    if show_orders:
        rays_x: list[float | None] = []
        rays_y: list[float | None] = []
        rays_z: list[float | None] = []
        label_x, label_y, label_z, labels = [], [], [], []
        for rod in rods:
            if (rod.h, rod.k) == (0, 0):
                continue
            hit_y, hit_z = plane_point(*hits[(rod.h, rod.k)])
            rays_x += [centre_x, screen_x, None]
            rays_y += [centre_y, hit_y, None]
            rays_z += [impact_z, hit_z, None]
            label_x.append(screen_x)
            label_y.append(hit_y)
            label_z.append(hit_z)
            labels.append(rod.label)
        if labels:
            figure.add_trace(
                go.Scatter3d(
                    x=rays_x,
                    y=rays_y,
                    z=rays_z,
                    mode="lines",
                    line={"color": "#a78bfa", "width": 3},
                    name=f"reachable orders ({len(labels)} beside (00))",
                    hoverinfo="skip",
                )
            )
            figure.add_trace(
                go.Scatter3d(
                    x=label_x,
                    y=label_y,
                    z=label_z,
                    mode="markers+text",
                    marker={"color": "#a78bfa", "size": 4},
                    text=labels,
                    textposition="middle right",
                    textfont={"color": "#c4b5fd", "size": 11},
                    name="(hk) from rod_orders()",
                    hovertemplate="%{text}<extra>reachable order</extra>",
                )
            )

    figure.add_trace(
        go.Scatter3d(
            x=[screen_x],
            y=[specular_y],
            z=[specular_z],
            mode="markers",
            marker={
                "color": "rgba(0,0,0,0)",
                "size": 9,
                "line": {"color": SPECULAR_COLOR, "width": 3},
            },
            name="specular hit point",
            hovertemplate="specular hit point — the (00) pixel of the screen<extra></extra>",
        )
    )

    # True-aspect side view, 1:1, because the main scene cannot show what 2.75 degrees looks
    # like and a stretched picture of a grazing beam is the one thing readers misread.
    tangent = float(
        geometry.specular_direction[2] / geometry.specular_direction[0]
    )
    for xs, ys, colour in (
        ((-1.0, 0.0), (tangent, 0.0), BEAM_COLOR),
        ((0.0, 1.0), (0.0, tangent), SPECULAR_COLOR),
    ):
        figure.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line={"color": colour, "width": 2},
                xaxis="x2",
                yaxis="y2",
                showlegend=False,
                hoverinfo="skip",
            )
        )
    figure.add_trace(
        go.Scatter(
            x=[-1.0, 1.0],
            y=[0.0, 0.0],
            mode="lines",
            line={"color": "#475569", "width": 2},
            xaxis="x2",
            yaxis="y2",
            showlegend=False,
            hovertemplate="surface<extra></extra>",
        )
    )

    condition = f"{pattern.condition}, " if pattern is not None else ""
    orders_note = (
        f" · {len(rods)} reachable orders inside ±{ORDERS_ACCEPTANCE_DEG:g}°"
        if show_orders
        else ""
    )
    scale_note = (
        "all three axes to scale; displayed ray angles are the real ones"
        if show_orders
        else f"z exaggerated {stretch:.0f}x; displayed ray angles not to scale"
    )
    figure.update_layout(
        uirevision="beam-geometry",
        height=560,
        margin={"l": 0, "r": 0, "t": 92, "b": 0},
        legend={"orientation": "h", "y": -0.02},
        title=(
            f"Beam geometry at {coverage:.2f} ML — {condition}{grazing_angle_deg:.2f}° grazing, "
            f"{azimuth_deg:g}° sample azimuth{orders_note}"
            f"<br><sub>{GEOMETRY_LABEL}</sub>"
            f"<br><sub><b>{scale_note}</b> — the inset is the same beam at 1:1</sub>"
        ),
        xaxis2={
            "domain": [0.015, 0.30],
            "anchor": "y2",
            "title": {"text": f"true {grazing_angle_deg:.2f}° grazing (1:1)", "font": {"size": 10}},
            "showticklabels": False,
            "zeroline": False,
            "showgrid": False,
        },
        yaxis2={
            "domain": [0.80, 0.97],
            "anchor": "x2",
            "scaleanchor": "x2",
            "scaleratio": 1.0,
            "showticklabels": False,
            "zeroline": False,
            "showgrid": False,
        },
        scene={
            "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
            "uirevision": "beam-geometry",
            "xaxis": {
                "title": "lab x (beam axis)",
                "range": [entry_x, screen_x],
                "autorange": False,
            },
            "yaxis": {"title": "lab y", "range": [y_low, y_high], "autorange": False},
            "zaxis": {"title": "height / lab z (ML)", "range": [0, z_top], "autorange": False},
            "aspectmode": "manual",
            "aspectratio": {"x": x_aspect, "y": y_aspect, "z": z_aspect},
            # Looking downstream from upstream-left, roughly 40 degrees off the detector's
            # normal, so the painted screen reads as a screen instead of the edge-on sliver a
            # side-on view gives. Drag to taste: uirevision above keeps whatever you set.
            "camera": {"eye": {"x": -1.22, "y": -1.02, "z": 0.5}, "center": {"z": -0.05}},
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
            line={"color": PROXY_COLOR, "width": 2, "dash": "dash"},
            name="normalized step-density morphology proxy (1 - S_d)",
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
                line={"color": SPECULAR_COLOR, "width": 3},
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
            f"Morphology proxy at {coverage:.2f} ML"
            if specular is None
            else f"Kinematic specular intensity and morphology proxy at {coverage:.2f} ML"
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
