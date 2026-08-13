import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import plotly.graph_objects as go

    from mbe_rheed_sim import SimulationConfig, run
    from mbe_rheed_sim.paper import (
        FIGURE3_NOMINAL_GA_N_RATIOS,
        figure3_config,
        figure3_parameters,
    )

    return (
        FIGURE3_NOMINAL_GA_N_RATIOS,
        Path,
        SimulationConfig,
        figure3_config,
        figure3_parameters,
        go,
        json,
        mo,
        np,
        plt,
        run,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Molecular-beam epitaxy: a minimal KMC-to-RHEED story

    Molecular-beam epitaxy (MBE) builds a crystal by delivering growth units to a hot
    surface. Deposition competes with thermally activated diffusion: particles land,
    explore terraces, nucleate islands, and become incorporated at island edges.

    This notebook implements a **small teaching model**, not the full strained GaN/AlN
    quantum-dot model of Budagosky and Garcia-Cristobal (2022). Its RHEED curve is a
    clearly labelled morphology proxy, not an electron-diffraction calculation.
    """)
    return


@app.cell
def _(mo):
    mo.vstack(
        [
            mo.md("## 1. From source to growing surface"),
            mo.mermaid(
                """
                flowchart LR
                    A[Ga source] -->|beam flux F| B[heated substrate]
                    B --> C[mobile adsorbates]
                    C --> D[islands and growing film]
                """
            ),
            mo.md(
                "The substrate temperature controls how rapidly deposited growth units "
                "diffuse or desorb before they become incorporated into the film."
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. What can a deposited growth unit do?

    | Event | Surface change | Competing control |
    |---|---|---|
    | **Deposit** | add one unit at the landing site | flux $F$ |
    | **Diffuse** | move a top unit to a neighboring column | $T$, bonds, barriers |
    | **Attach** | gain lateral neighbors and become less mobile | island geometry |
    | **Desorb** | remove one top unit | $T$, desorption barrier |

    These are the only physical events in the current single-species, solid-on-solid model.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. How kinetic Monte Carlo advances time

    MBE contains rare, discrete events with very different rates. Residence-time kinetic
    Monte Carlo chooses event $i$ with probability $r_i / R$ and advances the physical
    clock by

    $$\Delta t = -\frac{\ln u}{R}, \qquad R=\sum_i r_i.$$

    This lets deposition, diffusion, and desorption share one clock while preserving a
    reproducible stochastic trajectory when the random seed is fixed.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Model and event rates

    The state is an integer solid-on-solid height field on a periodic lattice with six
    neighbors per site. One deposition event adds one generic growth unit. A top particle
    may diffuse laterally or cross one step upward or downward; multi-step hops are forbidden.
    A downward crossing pays an Ehrlich-Schwoebel step barrier. Desorption removes a top
    particle.

    The global deposition rate and local diffusion rate are

    $$r_{\mathrm{dep}} = F N, \qquad
    r_{\mathrm{diff}} = \nu\exp\left[-\frac{E_{\mathrm{diff}} + nE_b + mE_{\mathrm{step}}}{k_B T}\right],$$

    $$r_{\mathrm{des}} = \nu\exp\left[-\frac{E_{\mathrm{des}} + nE_b}{k_B T}\right].$$

    $F$ is flux in ML/s, $N$ is the number of sites, $\nu$ is an effective attempt
    frequency, $E_{\mathrm{diff}}$ and $E_{\mathrm{des}}$ isolated-particle barriers, and
    $nE_b$ the lateral bond contribution. $m=1$ only for a downward crossing. The energetic
    defaults below are demonstration parameters.
    """)
    return


@app.cell
def _(FIGURE3_NOMINAL_GA_N_RATIOS, mo):
    default_parameters = {
        "experiment_mode": "Generic demonstration",
        "figure3_ratio": 0.82,
        "temperature_k": 800,
        "flux_ml_s": 0.5,
        "barrier_ev": 0.15,
        "step_barrier_ev": 0.05,
        "desorption_barrier_ev": 0.65,
        "size": 16,
        "coverage_ml": 2.0,
        "seed": 7,
    }
    get_parameters, _set_parameters = mo.state(default_parameters)
    parameter_controls = mo.md("""
        ### Simulation mode

        | Choice | Value |
        |---|---|
        | Experiment | {experiment_mode} |
        | Paper condition | {figure3_ratio} |

        ### Growth conditions

        | Quantity | Value |
        |---|---|
        | Temperature | {temperature_k} |
        | Deposition flux | {flux_ml_s} |
        | Diffusion barrier | {barrier_ev} |
        | Down-step barrier | {step_barrier_ev} |
        | Desorption barrier | {desorption_barrier_ev} |

        ### Numerical controls

        | Quantity | Value |
        |---|---|
        | Lattice size | {size} |
        | Target coverage | {coverage_ml} |
        | RNG seed | {seed} |
    """).batch(
        experiment_mode=mo.ui.radio(
            options=["Generic demonstration", "Paper Figure 3 preset"],
            value="Generic demonstration",
            label="Simulation mode",
        ),
        figure3_ratio=mo.ui.dropdown(
            {f"Ga/N = {ratio:.2f}": ratio for ratio in FIGURE3_NOMINAL_GA_N_RATIOS},
            value="Ga/N = 0.82",
            label="Figure 3 nominal Ga/N ratio",
        ),
        temperature_k=mo.ui.slider(600, 1_100, step=25, value=800, label="Temperature (K)"),
        flux_ml_s=mo.ui.slider(0.1, 1.0, step=0.1, value=0.5, label="Flux (ML/s)"),
        barrier_ev=mo.ui.slider(0.10, 0.40, step=0.01, value=0.15, label="Diffusion barrier (eV)"),
        step_barrier_ev=mo.ui.slider(
            0.0, 0.20, step=0.01, value=0.05, label="Down-step barrier (eV)"
        ),
        desorption_barrier_ev=mo.ui.slider(
            0.4, 0.9, step=0.01, value=0.65, label="Desorption barrier (eV)"
        ),
        size=mo.ui.slider(8, 24, step=2, value=16, label="Lattice size"),
        coverage_ml=mo.ui.slider(0.5, 3.0, step=0.5, value=2.0, label="Target coverage (ML)"),
        seed=mo.ui.number(start=0, stop=10_000, value=7, label="RNG seed"),
    )
    parameter_form = parameter_controls.form(
        submit_button_label="Run simulation",
        bordered=True,
        on_change=lambda value: _set_parameters(value or default_parameters),
    )
    mo.vstack(
        [
            mo.md(
                "## 5. Run the virtual experiment\n"
                "Choose **Generic demonstration** for editable teaching parameters or "
                "**Paper Figure 3 preset** to load the complete 40 s paper-derived condition. "
                "In paper mode the generic sliders are ignored. Frame scrubbing below reuses "
                "the stored trajectory and does not rerun KMC."
            ),
            parameter_form,
        ]
    )
    return (get_parameters,)


@app.cell
def _(SimulationConfig, figure3_config, figure3_parameters, get_parameters, run):
    selected_parameters = get_parameters()
    if selected_parameters["experiment_mode"] == "Paper Figure 3 preset":
        _ratio = float(selected_parameters["figure3_ratio"])
        _paper_parameters = figure3_parameters(_ratio)
        simulation_config = figure3_config(
            _ratio,
            lattice_size=7,
            duration_s=40.0,
            seed=int(selected_parameters["seed"]),
        )
        experiment_name = f"Paper Figure 3 preset (Ga/N = {_ratio:.2f})"
        experiment_detail = (
            f"T = {_paper_parameters.temperature_k:.2f} K; effective Ga flux = "
            f"{_paper_parameters.effective_ga_flux_ml_s:.4f} ML/s; predicted growth rate = "
            f"{_paper_parameters.predicted_growth_rate_ml_s:.4f} ML/s; "
            f"7x7 smoke lattice; 40 s; seed {simulation_config.seed}."
        )
    else:
        _paper_parameters = None
        simulation_config = SimulationConfig(
            lattice_size=int(selected_parameters["size"]),
            target_coverage_ml=float(selected_parameters["coverage_ml"]),
            temperature_k=float(selected_parameters["temperature_k"]),
            deposition_flux_ml_s=float(selected_parameters["flux_ml_s"]),
            diffusion_barrier_ev=float(selected_parameters["barrier_ev"]),
            step_barrier_ev=float(selected_parameters["step_barrier_ev"]),
            desorption_barrier_ev=float(selected_parameters["desorption_barrier_ev"]),
            seed=int(selected_parameters["seed"]),
        )
        experiment_name = "Generic demonstration"
        experiment_detail = (
            f"T = {simulation_config.temperature_k:.0f} K; flux = "
            f"{simulation_config.deposition_flux_ml_s:.2f} ML/s; "
            f"{simulation_config.lattice_size}x{simulation_config.lattice_size}; "
            f"target {simulation_config.target_coverage_ml:.1f} ML; seed "
            f"{simulation_config.seed}."
        )
    simulation = run(simulation_config)
    coverage_axis = (
        simulation.time_s * _paper_parameters.predicted_growth_rate_ml_s
        if _paper_parameters is not None
        else simulation.coverage_ml
    )
    coverage_axis_label = (
        "paper-predicted film coverage (ML)"
        if _paper_parameters is not None
        else "film coverage (ML)"
    )
    return coverage_axis, coverage_axis_label, experiment_detail, experiment_name, simulation


@app.cell
def _(experiment_detail, experiment_name, mo, simulation):
    mo.md(f"""
    **Active mode:** {experiment_name}

    {experiment_detail}

    **Completed:** {simulation.deposited_events} deposition events,
    {simulation.selected_diffusion_events} selected diffusion events
    ({simulation.diffusion_events} nearest-neighbor hop equivalents), and
    {simulation.desorbed_events} desorption events; simulated time
    {simulation.time_s[-1]:.3f} s; final RMS roughness
    {simulation.roughness_ml[-1]:.3f} ML.
    """)
    return


@app.cell
def _(mo, simulation):
    get_frame, set_frame = mo.state(len(simulation.snapshots) - 1, allow_self_loops=True)
    return get_frame, set_frame


@app.cell
def _(mo, set_frame, simulation):
    playback = mo.ui.refresh(
        options=[0.25, 0.5, 1.0],
        label="Playback interval",
        on_change=lambda _value: set_frame(lambda frame: (frame + 1) % len(simulation.snapshots)),
    )
    return (playback,)


@app.cell
def _(coverage_axis, get_frame, mo, np, playback, set_frame, simulation):
    _current_frame = min(get_frame(), len(simulation.snapshots) - 1)
    _cycle_labels = {
        0.0: "flat reference",
        0.5: "expected island/step maximum",
        1.0: "expected first-layer completion",
        1.5: "expected renewed roughening",
        2.0: "expected second-layer completion",
    }
    _milestone_indices = {
        (
            f"{coverage:.1f} ML — {_cycle_labels[coverage]}"
            if coverage in _cycle_labels
            else f"{coverage:.1f} ML"
        ): int(np.argmin(np.abs(coverage_axis - coverage)))
        for coverage in np.arange(0.0, max(0.5, float(coverage_axis[-1])) + 0.01, 0.5)
    }
    snapshot_slider = mo.ui.slider(
        0,
        len(simulation.snapshots) - 1,
        value=min(get_frame(), len(simulation.snapshots) - 1),
        label="Recorded growth frame",
        show_value=True,
        on_change=set_frame,
    )
    milestone_picker = mo.ui.dropdown(
        options=_milestone_indices,
        value=min(
            _milestone_indices,
            key=lambda label: abs(coverage_axis[_current_frame] - float(label.split()[0])),
        ),
        label="Coverage milestone",
        on_change=set_frame,
    )
    display_mode = mo.ui.dropdown(
        options=["3D height surface", "Hexagonal cells"],
        value="3D height surface",
        label="Surface view",
    )
    mo.vstack(
        [
            mo.md("## 6. Surface morphology and the RHEED proxy"),
            mo.md(
                "The annotated 0–2 ML shortcuts encode the **ideal layer-by-layer "
                "hypothesis**. The linked morphology and proxy show what this stochastic "
                "run actually produces; disagreement is a result, not hidden by relabeling."
            ),
            mo.hstack(
                [snapshot_slider, milestone_picker, playback, display_mode],
                justify="start",
                gap=2,
                wrap=True,
            ),
        ]
    )
    return (display_mode,)


@app.cell
def _(coverage_axis, coverage_axis_label, display_mode, get_frame, go, mo, np, simulation):
    _frame = min(get_frame(), len(simulation.snapshots) - 1)
    _heights = simulation.snapshots[_frame]
    _coverage = coverage_axis[_frame]
    _proxy = simulation.rheed_proxy[_frame]
    _zmax = max(1, int(simulation.snapshots.max()))
    if display_mode.value == "3D height surface":
        surface_figure = go.Figure(
            go.Surface(
                z=_heights,
                colorscale="Viridis",
                cmin=0,
                cmax=_zmax,
                colorbar={"title": "height (ML)", "len": 0.7},
                hovertemplate="x=%{x}<br>y=%{y}<br>height=%{z} ML<extra></extra>",
                name="surface",
            )
        )
        surface_figure.update_layout(
            height=430,
            margin={"l": 0, "r": 0, "t": 50, "b": 0},
            title=f"Surface at {_coverage:.2f} ML",
            scene={
                "xaxis_title": "array x",
                "yaxis_title": "array y",
                "zaxis_title": "height (ML)",
                "zaxis": {"range": [0, _zmax]},
                "aspectmode": "data",
                "camera": {"eye": {"x": 1.4, "y": 1.4, "z": 1.0}},
            },
        )
    else:
        _axial_y, _axial_x = np.indices(_heights.shape)
        _hex_data = np.column_stack((_axial_x.ravel(), _axial_y.ravel(), _heights.ravel()))
        surface_figure = go.Figure(
            go.Scatter(
                x=(_axial_x + 0.5 * _axial_y).ravel(),
                y=(np.sqrt(3.0) / 2.0 * _axial_y).ravel(),
                mode="markers",
                marker={
                    "symbol": "hexagon",
                    "size": max(9, min(24, 320 / len(_heights))),
                    "color": _heights.ravel(),
                    "colorscale": "Viridis",
                    "cmin": 0,
                    "cmax": _zmax,
                    "colorbar": {"title": "height (ML)", "len": 0.7},
                    "line": {"color": "rgba(255,255,255,0.65)", "width": 0.5},
                },
                customdata=_hex_data,
                hovertemplate=(
                    "q=%{customdata[0]}<br>r=%{customdata[1]}<br>"
                    "height=%{customdata[2]} ML<extra></extra>"
                ),
                name="hexagonal cells",
            )
        )
        surface_figure.update_layout(
            height=430,
            margin={"l": 20, "r": 20, "t": 50, "b": 30},
            title=f"Six-neighbor axial lattice at {_coverage:.2f} ML",
            xaxis={"title": "axial q + r/2", "scaleanchor": "y", "scaleratio": 1},
            yaxis={"title": "sqrt(3) r / 2"},
        )
    _first_layer = np.flatnonzero(coverage_axis <= 1.0)
    _most_stepped = int(_first_layer[np.argmin(simulation.rheed_proxy[_first_layer])])
    _cycle_targets = np.arange(0.0, min(2.0, float(coverage_axis[-1])) + 0.01, 0.5)
    _cycle_indices = np.array(
        [int(np.argmin(np.abs(coverage_axis - target))) for target in _cycle_targets]
    )
    rheed_figure = go.Figure()
    rheed_figure.add_trace(
        go.Scatter(
            x=[coverage_axis[0], coverage_axis[_most_stepped]],
            y=[simulation.rheed_proxy[0], simulation.rheed_proxy[_most_stepped]],
            mode="markers",
            marker={
                "color": ["#16a34a", "#7e22ce"],
                "symbol": ["diamond", "x"],
                "size": 12,
            },
            customdata=["initial flat surface", "most stepped stored frame through 1 ML"],
            hovertemplate="%{customdata}<br>coverage=%{x:.2f} ML<br>proxy=%{y:.3f}<extra></extra>",
            showlegend=False,
        )
    )
    rheed_figure.add_trace(
        go.Scatter(
            x=coverage_axis,
            y=simulation.rheed_proxy,
            mode="lines",
            line={"color": "#d62728", "width": 3},
            name="normalized step-density RHEED proxy",
        )
    )
    rheed_figure.add_trace(
        go.Scatter(
            x=coverage_axis[_cycle_indices],
            y=simulation.rheed_proxy[_cycle_indices],
            mode="markers+text",
            marker={"color": "#f59e0b", "size": 8, "symbol": "circle-open"},
            text=[f"{target:.1f}" for target in _cycle_targets],
            textposition="top center",
            name="0–2 ML cycle milestones",
            hovertemplate=(
                "nearest stored coverage=%{x:.2f} ML<br>observed proxy=%{y:.3f}<extra></extra>"
            ),
        )
    )
    rheed_figure.add_trace(
        go.Scatter(
            x=[_coverage, _coverage],
            y=[0, 1.03],
            mode="lines",
            line={"color": "#111827", "dash": "dot"},
            name="current frame",
            hoverinfo="skip",
        )
    )
    rheed_figure.add_trace(
        go.Scatter(
            x=[_coverage],
            y=[_proxy],
            mode="markers",
            marker={"color": "#111827", "size": 10},
            name="current proxy",
            hovertemplate="coverage=%{x:.2f} ML<br>proxy=%{y:.3f}<extra></extra>",
        )
    )
    rheed_figure.update_layout(
        height=430,
        margin={"l": 60, "r": 10, "t": 50, "b": 60},
        title=f"RHEED proxy at {_coverage:.2f} ML",
        legend={"orientation": "h", "y": -0.2},
        xaxis_title=coverage_axis_label,
        yaxis_title="normalized proxy",
        yaxis_range=[0, 1.03],
    )
    mo.vstack(
        [
            mo.Html("""
                <style>
                @media (max-width: 600px) {
                    div:has(> marimo-plotly) { flex-basis: 100% !important; }
                }
                </style>
            """),
            mo.hstack(
                [surface_figure, rheed_figure],
                widths="equal",
                wrap=True,
                align="center",
            ),
            mo.md(
                "The 3D height view uses array coordinates. **Hexagonal cells** maps the "
                "same periodic axial lattice to Cartesian centers so its six-neighbor "
                "topology is shown without implying a square metric. On the RHEED trace, "
                "the **green diamond** is the initially flat maximum and the **purple x** "
                "is the most stepped stored frame through the first monolayer."
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### What does RHEED see here?

    A real RHEED beam strikes the surface at grazing incidence and its diffracted intensity
    depends on electron-scattering geometry. This teaching model does **not** calculate that
    diffraction. It uses surface steps as a morphology-based connection:

    The proxy is $I_{\mathrm{proxy}}=1-S_d$, where $S_d$ is the fraction of unique
    nearest-neighbor bonds whose endpoint heights differ. Smooth, nearly complete layers
    have fewer steps and a larger proxy. Real RHEED phase and amplitude also depend on
    beam geometry, refraction, absorption, reconstruction, and multiple scattering.
    """)
    return


@app.cell
def _(coverage_axis, coverage_axis_label, mo, plt, simulation):
    observable_figure, _axes = plt.subplots(
        3, 1, figsize=(8, 7), sharex=True, constrained_layout=True
    )
    _axes[0].plot(coverage_axis, simulation.roughness_ml, color="tab:blue")
    _axes[0].set_ylabel("RMS roughness (ML)")
    _axes[1].plot(
        coverage_axis,
        simulation.island_density_per_site,
        color="tab:green",
    )
    _axes[1].set_ylabel("islands / site")
    _axes[2].plot(coverage_axis, simulation.rheed_proxy, color="tab:red")
    _axes[2].set(xlabel=coverage_axis_label, ylabel=r"$1-S_d$ proxy")
    observable_figure.suptitle("Growth observables")
    mo.vstack([mo.md("## 7. Growth observables"), observable_figure])
    return


@app.cell
def _(Path, json, mo):
    sweep_data = json.loads(
        (Path(__file__).resolve().parents[1] / "data/processed/parameter_sweep.json").read_text()
    )
    _temperatures = sweep_data["temperatures_k"]
    _fluxes = sweep_data["fluxes_ml_s"]
    _default = {"temperature_k": _temperatures[1], "flux_ml_s": _fluxes[1]}
    get_sweep_selection, _set_sweep_selection = mo.state(_default)
    sweep_controls = mo.md("""
        | Sweep coordinate | Selection |
        |---|---|
        | Temperature | {temperature_k} |
        | Deposition flux | {flux_ml_s} |
    """).batch(
        temperature_k=mo.ui.dropdown(
            {f"{value:.0f} K": value for value in _temperatures},
            value=f"{_default['temperature_k']:.0f} K",
            label="Temperature",
        ),
        flux_ml_s=mo.ui.dropdown(
            {f"{value:.2f} ML/s": value for value in _fluxes},
            value=f"{_default['flux_ml_s']:.2f} ML/s",
            label="Deposition flux",
        ),
    )
    sweep_form = sweep_controls.form(
        submit_button_label="Run selected point",
        bordered=True,
        on_change=lambda value: _set_sweep_selection(value or _default),
    )
    mo.vstack(
        [
            mo.md(
                "## 8. Temperature/flux regime map\n"
                "Select one point from the reproducible three-seed sweep. The linked run "
                "uses seed 0 on the same 16x16 lattice."
            ),
            sweep_form,
        ]
    )
    return get_sweep_selection, sweep_data


@app.cell
def _(SimulationConfig, get_sweep_selection, run):
    sweep_selection = get_sweep_selection()
    selected_sweep_result = run(
        SimulationConfig(
            lattice_size=16,
            target_coverage_ml=2.0,
            temperature_k=float(sweep_selection["temperature_k"]),
            deposition_flux_ml_s=float(sweep_selection["flux_ml_s"]),
            max_isolated_hop_distance=3,
            sample_every_ml=0.05,
            seed=0,
        )
    )
    return selected_sweep_result, sweep_selection


@app.cell
def _(mo, np, plt, selected_sweep_result, sweep_data, sweep_selection):
    _temperatures = np.asarray(sweep_data["temperatures_k"])
    _fluxes = np.asarray(sweep_data["fluxes_ml_s"])
    _amplitudes = np.asarray(sweep_data["mean_amplitude"])
    _amplitude_stds = np.asarray(sweep_data["std_amplitude"])
    _temperature_index = int(np.flatnonzero(_temperatures == sweep_selection["temperature_k"])[0])
    _flux_index = int(np.flatnonzero(_fluxes == sweep_selection["flux_ml_s"])[0])
    sweep_figure, _axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    _heatmap = _axes[0].imshow(_amplitudes, origin="lower", cmap="magma", aspect="auto")
    _axes[0].scatter(
        _flux_index,
        _temperature_index,
        marker="s",
        s=120,
        facecolors="none",
        edgecolors="cyan",
        linewidths=2,
    )
    _axes[0].set(
        title="Mean proxy amplitude",
        xlabel="flux (ML/s)",
        ylabel="temperature (K)",
        xticks=range(len(_fluxes)),
        xticklabels=_fluxes,
        yticks=range(len(_temperatures)),
        yticklabels=_temperatures,
    )
    sweep_figure.colorbar(_heatmap, ax=_axes[0], shrink=0.8)
    _surface = _axes[1].imshow(selected_sweep_result.final_heights, origin="lower", cmap="viridis")
    _axes[1].set(title="Selected final morphology", xlabel="lattice x", ylabel="lattice y")
    sweep_figure.colorbar(_surface, ax=_axes[1], shrink=0.8, label="height (ML)")
    _axes[2].plot(
        selected_sweep_result.coverage_ml,
        selected_sweep_result.rheed_proxy,
        color="tab:red",
    )
    _axes[2].set(
        title="Selected proxy trace",
        xlabel="coverage (ML)",
        ylabel=r"$1-S_d$",
        ylim=(0, 1.03),
    )
    mo.vstack(
        [
            sweep_figure,
            mo.md(
                f"Selected ensemble amplitude: **{_amplitudes[_temperature_index, _flux_index]:.3f} "
                f"+/- {_amplitude_stds[_temperature_index, _flux_index]:.3f}**; "
                f"single-run final roughness: **{selected_sweep_result.roughness_ml[-1]:.3f} ML**."
            ),
        ]
    )
    return


@app.cell
def _(Path, json):
    figure3_data = json.loads(
        (
            Path(__file__).resolve().parents[1] / "data/processed/figure3_simulated_smoke.json"
        ).read_text()
    )
    return (figure3_data,)


@app.cell
def _(figure3_data, go, mo, np):
    _colors = [
        ("#1f77b4", "rgba(31,119,180,0.16)"),
        ("#ff7f0e", "rgba(255,127,14,0.16)"),
        ("#2ca02c", "rgba(44,160,44,0.16)"),
    ]
    figure3_figure = go.Figure()
    _metric_lines = []

    def _format_metric(value, digits=3, signed=False):
        if value is None:
            return "n/a"
        return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"

    for _trace, (_line_color, _fill_color) in zip(figure3_data["traces"], _colors, strict=True):
        _time = np.asarray(_trace["time_s"])
        _mean = np.asarray(_trace["rheed_proxy_mean"])
        _std = np.asarray(_trace["rheed_proxy_std"])
        _ratio = _trace["nominal_ga_n_ratio"]
        _metrics = _trace["oscillation_metrics"]
        _metric_lines.append(
            f"- **Ga/N = {_ratio:.2f}:** {int(_metrics['peak_count'])} peaks; "
            f"period {_format_metric(_metrics['period_ml'])} ML "
            f"(deviation {_format_metric(_metrics['period_deviation_ml'])} ML); "
            f"detrended amplitude {_format_metric(_metrics['detrended_amplitude'])}; "
            f"peak-to-trough {_format_metric(_metrics['peak_to_trough_amplitude'])}; "
            f"near-1-ML spectral fraction "
            f"{_format_metric(_metrics['spectral_power_fraction'])}; peak/trough phase "
            f"{_format_metric(_metrics['peak_phase_ml'], 2)}/"
            f"{_format_metric(_metrics['trough_phase_ml'], 2)} ML; damping rate "
            f"{_format_metric(_metrics['damping_rate_per_ml'], signed=True)} per ML."
        )
        figure3_figure.add_trace(
            go.Scatter(
                x=_time,
                y=np.clip(_mean - _std, 0, 1),
                mode="lines",
                line={"width": 0},
                hoverinfo="skip",
                showlegend=False,
            )
        )
        figure3_figure.add_trace(
            go.Scatter(
                x=_time,
                y=np.clip(_mean + _std, 0, 1),
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor=_fill_color,
                hoverinfo="skip",
                showlegend=False,
            )
        )
        figure3_figure.add_trace(
            go.Scatter(
                x=_time,
                y=_mean,
                mode="lines",
                line={"color": _line_color, "width": 2},
                name=f"Ga/N = {_ratio:.2f} mean +/- 1 SD",
                hovertemplate="time=%{x:.1f} s<br>proxy=%{y:.3f}<extra></extra>",
            )
        )
    figure3_figure.update_layout(
        height=470,
        margin={"l": 60, "r": 20, "t": 65, "b": 105},
        title="Figure 3 target: simulated step-density proxy",
        xaxis_title="time (s)",
        yaxis_title="normalized proxy",
        yaxis_range=[0, 1.03],
        hovermode="x unified",
        legend={"orientation": "h", "x": 0, "y": -0.22},
    )
    mo.vstack(
        [
            mo.md("## 9. Paper Figure 3 smoke reproduction"),
            figure3_figure,
            mo.md(
                f"This committed result is a **{figure3_data['classification']}** on a "
                f"{figure3_data['lattice_size']}x{figure3_data['lattice_size']} lattice "
                f"with seeds `{figure3_data['seeds']}`. The repeated oscillations provide "
                "the qualitative comparison; their amplitude is not paper-scale converged, "
                "and no experimental curve has been silently digitized or normalized."
            ),
            mo.md(
                "### Layer-scale oscillation diagnostics\n\n"
                + "\n".join(_metric_lines)
                + "\n\nThe coverage axis for these diagnostics is paper-predicted growth rate "
                "times time. Classification requires repeated detrended extrema with a "
                "median period within 0.5–1.5 ML; it is a diagnostic, not proof of "
                "experimental RHEED agreement."
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 10. Repeatability and model limits

    The seed makes a run exactly reproducible, but one trajectory is not an uncertainty
    estimate. The scripted Stage 3 workflows now aggregate fixed seed ensembles: run
    `make reproduce-figure3` for 40 s Figure 3 proxy bands, `make sweep` for the first
    temperature/flux map, `make convergence` for the generic finite-size check, and
    `make convergence-figure3` for the practical paper-regime size check.

    The qualitative comparison target is the oscillation of surface step density shown
    alongside experimental RHEED traces in Figure 3 of Budagosky and Garcia-Cristobal.
    The separate Figure 3 workflow implements the paper's homoepitaxial flux-ratio
    calibration and a validated isolated-adatom long-hop approximation. Quantitative
    comparison is still deferred because paper-scale convergence and experimental-curve
    normalization are unresolved. Strain is not needed for that homoepitaxial target and
    remains outside the current model.

    A defensible next RHEED stage is a kinematic layer-interference model. Full dynamical
    electron scattering remains outside the baseline.
    """)
    return


if __name__ == "__main__":
    app.run()
