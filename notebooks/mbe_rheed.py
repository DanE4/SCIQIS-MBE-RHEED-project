import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import numpy as np

    from mbe_rheed_notebook import batch, controls, figures
    from mbe_rheed_sim import SimulationConfig, run
    from mbe_rheed_sim.kmc import SimulationResult
    from mbe_rheed_sim.paper import FIGURE3_NOMINAL_GA_N_RATIOS

    ROOT = Path(__file__).resolve().parents[1]
    GALLERY_DIR = ROOT / "data" / "gallery"
    GALLERY = json.loads((GALLERY_DIR / "index.json").read_text())
    return (
        FIGURE3_NOMINAL_GA_N_RATIOS,
        GALLERY,
        GALLERY_DIR,
        ROOT,
        SimulationConfig,
        SimulationResult,
        batch,
        controls,
        figures,
        json,
        mo,
        np,
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

    Narrative and reactive wiring live here; the physics is in `mbe_rheed_sim` and the
    figure/widget construction is in `mbe_rheed_notebook`. See `INDEX.md`.
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
def _(GALLERY, controls, mo):
    data_source, gallery_choice = controls.source_selector(GALLERY)
    mo.vstack(
        [
            mo.md(
                "## 5. Run the virtual experiment\n"
                "**Pre-computed demo** loads a stored trajectory instantly — use this when "
                "presenting. **Simulate now** runs the model live with the parameters below.\n\n"
                "One KMC trajectory is strictly sequential: every event changes the surface "
                "that the next event is drawn from, so a single run cannot be spread across "
                "cores no matter how many are free. Cores only help across *independent* runs, "
                "which is what the batch workflows in the last section use. That is why a live "
                "run here is single-threaded and why the stored trajectories exist."
            ),
            mo.hstack([data_source, gallery_choice], justify="start", gap=2, wrap=True),
        ]
    )
    return data_source, gallery_choice


@app.cell
def _(GALLERY, controls, mo):
    preset_choice = controls.preset_selector(GALLERY)
    mo.vstack(
        [
            mo.md(
                "### Parameters for a live run\n"
                "**Start from** loads the parameters behind any stored demo into the form below, "
                "so you can reproduce that scenario and then change one thing at a time. Hover "
                "the &#9432; beside a quantity to see what it does.\n\n"
                "**Hand-tuned parameters** lets you edit every growth condition directly. "
                "**GaN parameters from the paper** replaces temperature, effective Ga flux, and "
                "the four barriers with values the paper fits to a Ga/N ratio; the lattice, "
                "stopping criterion, acceleration, sampling, event limit, and seed still apply. "
                "The stopping criterion you did not pick is ignored."
            ),
            preset_choice,
        ]
    )
    return (preset_choice,)


@app.cell
def _(FIGURE3_NOMINAL_GA_N_RATIOS, GALLERY, controls, mo, preset_choice):
    # Rebuilding the form is how a preset moves the sliders: marimo elements take their value
    # at construction, so the cell that owns them re-runs when the preset changes.
    _values = (
        controls.preset_parameters(GALLERY[preset_choice.value])
        if preset_choice.value
        else controls.DEFAULT_PARAMETERS
    )
    get_parameters, _set_parameters = mo.state(_values)
    parameter_form = controls.parameter_form(
        FIGURE3_NOMINAL_GA_N_RATIOS,
        on_change=lambda value: _set_parameters(value or _values),
        values=_values,
    )
    parameter_form
    return (get_parameters,)


@app.cell
def _(mo):
    # A run button lives in its own cell so reading its value does not reset it.
    expensive_override = mo.ui.run_button(label="Run it anyway")
    return (expensive_override,)


@app.cell
def _(
    GALLERY,
    GALLERY_DIR,
    SimulationResult,
    controls,
    data_source,
    expensive_override,
    gallery_choice,
    get_parameters,
    mo,
):
    if data_source.value == controls.PRE_COMPUTED:
        _entry = gallery_choice.value
        _meta = GALLERY[_entry]
        simulation = SimulationResult.load_npz(GALLERY_DIR / f"{_entry}.npz")
        growth_rate = _meta["predicted_growth_rate_ml_s"]
        experiment_name = _meta["title"]
        experiment_detail = controls.gallery_detail(_meta, simulation.config)
        experiment_source = f"stored trajectory `data/gallery/{_entry}.npz` — nothing was simulated"
    else:
        _config, _estimate, growth_rate, experiment_name, experiment_detail = controls.build_run(
            get_parameters()
        )
        mo.stop(
            controls.is_expensive(_config, _estimate) and not expensive_override.value,
            controls.expensive_warning(_estimate, expensive_override),
        )
        simulation = controls.run_with_progress(_config, f"Running KMC: {experiment_name}")
        experiment_source = (
            f"live run, {_config.lattice_size}x{_config.lattice_size}, seed {_config.seed}"
        )

    coverage_axis = simulation.time_s * growth_rate if growth_rate else simulation.coverage_ml
    coverage_axis_label = (
        "paper-predicted film coverage (ML)" if growth_rate else "film coverage (ML)"
    )
    return (
        coverage_axis,
        coverage_axis_label,
        experiment_detail,
        experiment_name,
        experiment_source,
        simulation,
    )


@app.cell
def _(experiment_detail, experiment_name, experiment_source, mo, simulation):
    mo.md(f"""
    **Active mode:** {experiment_name}

    **Source:** {experiment_source}

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
        value=_current_frame,
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
            mo.md(r"""
            **What does RHEED actually see?** A real RHEED beam strikes the surface at grazing
            incidence and its diffracted intensity depends on electron-scattering geometry.
            This teaching model does **not** calculate that diffraction. It uses surface steps
            as a morphology-based stand-in:

            $$I_{\mathrm{proxy}} = 1 - S_d$$

            where $S_d$ is the fraction of unique nearest-neighbor bonds whose endpoint heights
            differ. Smooth, nearly complete layers have fewer steps and a larger proxy. Real
            RHEED phase and amplitude also depend on beam geometry, refraction, absorption,
            surface reconstruction, and multiple scattering.
            """),
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
def _(
    coverage_axis,
    coverage_axis_label,
    display_mode,
    figures,
    get_frame,
    mo,
    simulation,
):
    _frame = min(get_frame(), len(simulation.snapshots) - 1)
    _heights = simulation.snapshots[_frame]
    _zmax = max(1, int(simulation.snapshots.max()))
    _builder = (
        figures.height_surface
        if display_mode.value == "3D height surface"
        else figures.hex_cells
    )
    surface_figure = _builder(_heights, float(coverage_axis[_frame]), _zmax)
    rheed_figure = figures.rheed_trace(
        coverage_axis, simulation.rheed_proxy, _frame, coverage_axis_label
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
            mo.hstack([surface_figure, rheed_figure], widths="equal", wrap=True, align="center"),
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
def _(coverage_axis, coverage_axis_label, figures, mo, simulation):
    observable_figure = figures.observables(
        coverage_axis,
        simulation.roughness_ml,
        simulation.island_density_per_site,
        simulation.rheed_proxy,
        coverage_axis_label,
    )
    mo.vstack([mo.md("## 7. Growth observables"), observable_figure])
    return


@app.cell
def _(mo):
    get_artifact_revision, set_artifact_revision = mo.state(0)
    return get_artifact_revision, set_artifact_revision


@app.cell
def _(ROOT, get_artifact_revision, json, mo):
    get_artifact_revision()
    sweep_data = json.loads((ROOT / "data/processed/parameter_sweep.json").read_text())
    _temperatures = sweep_data["temperatures_k"]
    _fluxes = sweep_data["fluxes_ml_s"]
    _default = {"temperature_k": _temperatures[1], "flux_ml_s": _fluxes[1]}
    get_sweep_selection, _set_sweep_selection = mo.state(_default)
    sweep_form = (
        mo.md("""
        | Sweep coordinate | Selection |
        |---|---|
        | Temperature | {temperature_k} |
        | Deposition flux | {flux_ml_s} |
    """)
        .batch(
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
        .form(
            submit_button_label="Run selected point",
            bordered=True,
            on_change=lambda value: _set_sweep_selection(value or _default),
        )
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
def _(figures, mo, np, selected_sweep_result, sweep_data, sweep_selection):
    sweep_figure = figures.sweep_panels(sweep_data, sweep_selection, selected_sweep_result)
    _row = int(
        np.flatnonzero(np.asarray(sweep_data["temperatures_k"]) == sweep_selection["temperature_k"])[0]
    )
    _column = int(
        np.flatnonzero(np.asarray(sweep_data["fluxes_ml_s"]) == sweep_selection["flux_ml_s"])[0]
    )
    mo.vstack(
        [
            sweep_figure,
            mo.md(
                f"Selected ensemble amplitude: "
                f"**{sweep_data['mean_amplitude'][_row][_column]:.3f} "
                f"+/- {sweep_data['std_amplitude'][_row][_column]:.3f}**; "
                f"single-run final roughness: **{selected_sweep_result.roughness_ml[-1]:.3f} ML**."
            ),
        ]
    )
    return


@app.cell
def _(ROOT, get_artifact_revision, json):
    get_artifact_revision()
    figure3_data = json.loads(
        (ROOT / "data/processed/figure3_simulated_smoke.json").read_text()
    )
    return (figure3_data,)


@app.cell
def _(figure3_data, figures, mo):
    figure3_figure, _metric_rows = figures.figure3_comparison(figure3_data)
    _morphology = figure3_data["morphology_sequence"]
    morphology_figure = figures.morphology_sequence(_morphology)
    _provenance = figure3_data["provenance"]
    mo.vstack(
        [
            mo.md("## 9. Comparison with the published experiment"),
            figure3_figure,
            mo.md(
                "The red curves are **figure-derived experimental RHEED panel coordinates**; "
                "the blue curves are this model's raw morphology-derived `1-S_d` with a "
                "three-seed standard-deviation band. They share time but intentionally use "
                "separate panels and are not the same physical quantity."
            ),
            mo.md(
                "### Quantitative diagnostics\n\n"
                "| Ga/N | period ref/sim (ML) | absolute peak-phase difference (ML) | "
                "damping ref/sim (per ML) | relative amplitude ref/sim |\n"
                "|---:|---:|---:|---:|---:|\n" + _metric_rows + "\n\nRelative amplitude is "
                "referenced to Ga/N = 0.89 separately for each signal. Diagnostics use "
                "paper-predicted coverage = growth rate x time after linear detrending. "
                "Figure-derived normalization and 7x7 finite-size effects prevent a "
                "quantitative agreement claim."
            ),
            morphology_figure,
            mo.md(
                f"The morphology sequence is **{_morphology['classification']}**. "
                f"Artifacts were generated by `{_provenance['generated_by']}` at Git commit "
                f"`{_provenance['code_version']['git_commit']}` using lattice "
                f"{_provenance['lattice_size']}x{_provenance['lattice_size']} and seeds "
                f"`{_provenance['seeds']}`. Run `make figure3` to regenerate them."
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 10. Repeatability and model limits

    The seed makes a run exactly reproducible, but one trajectory is not an uncertainty
    estimate. The scripted workflows aggregate fixed seed ensembles instead: `make figure3`
    for the Ga/N comparison and morphology figures, `make sweep` for the temperature/flux
    map, `make convergence` for the generic finite-size check, and `make convergence-figure3`
    for the paper-regime size check.

    The comparison target is the oscillation of surface step density shown alongside
    experimental RHEED traces in Figure 3 of Budagosky and Garcia-Cristobal. That workflow
    implements the paper's homoepitaxial flux-ratio calibration and a validated
    isolated-adatom long-hop approximation, and reports figure-derived reference curves,
    period, phase, damping, relative amplitude, and provenance.

    It stays a **qualitative** comparison: the authors' normalization is unavailable, and the
    7x7 amplitude is not finite-size converged. Strain is not needed for this homoepitaxial
    target and is outside the model.

    A defensible next RHEED stage is a kinematic layer-interference model. Full dynamical
    electron scattering remains outside the baseline.
    """)
    return


@app.cell
def _(batch, mo):
    get_batch_request, set_batch_request = mo.state(None)
    get_batch_process, set_batch_process = mo.state(None, allow_self_loops=True)
    batch_form = batch.controls(
        on_change=lambda value: set_batch_request(dict(value) if value else None)
    )
    mo.vstack(
        [
            mo.md(
                "## 11. Batch workflows (regenerating the data)\n"
                "These controls launch the same reproducible CLI used by `make`. Independent "
                "seeds and parameter points use bounded spawn-based worker processes; one KMC "
                "trajectory remains sequential. Blank overrides retain the preset below.\n\n"
                "| Workflow | Canonical seeds | Canonical sizes / grid |\n"
                "|---|---|---|\n"
                "| Baseline | 2026 | 8x8 |\n"
                "| GaN Ga/N comparison | 2026-2028 | 7x7, three Ga/N ratios |\n"
                "| Sweep | 0-2 | 16x16, 3 temperatures x 3 fluxes |\n"
                "| Generic convergence | 0-2 | 8/16/24 |\n"
                "| Figure 3 convergence | 0-2 | 8/16/32; add 64 via the size override |\n"
                "| Acceleration validation | 0-99 | 7x7 exact/accelerated pairs |\n"
                "| Scientific trends | 0-4 | 8x8, three physics configurations |\n"
                "| Sweep validation | 0-2 | 24x24, 3 temperatures x 2 fluxes |\n"
                "| Runtime benchmark | 0 | 64/128/256, sequential |\n\n"
                "Measured on the development M4 Pro: the Ga/N comparison is about 16 s with four "
                "workers; Figure 3 convergence through 64x64 is about 52 s with three effective "
                "workers; the sequential 64/128/256 benchmark is about 34 s. Runtime varies "
                "with load."
            ),
            batch_form,
            mo.md(
                "Successful jobs are first retained under `outputs/batches/`, then atomically "
                "promoted to the canonical notebook artifacts. Failed or cancelled jobs never "
                "replace canonical data. Only one batch can run in this notebook session.\n\n"
                "This panel regenerates data; it does not plot it. On promotion, sections 8 "
                "and 9 above re-read `data/processed/` and redraw themselves — scroll back up "
                "to see what a batch changed."
            ),
        ]
    )
    return get_batch_process, get_batch_request, set_batch_process


@app.cell
def _(batch, get_batch_process, get_batch_request, mo, set_batch_process):
    _request = get_batch_request()
    _existing = get_batch_process()
    if _request is None:
        batch_launch_message = mo.md("Select a workflow and press **Launch batch workflow**.")
    elif _existing is not None and _existing["process"].poll() is None:
        batch_launch_message = mo.callout(
            "A batch is already running. Cancel it or wait for it to finish before "
            "launching another.",
            kind="warn",
        )
    elif batch.needs_confirmation(_request) and not _request["confirm_expensive"]:
        batch_launch_message = mo.callout(
            "This workflow is intentionally gated. Tick the expensive-workflow confirmation "
            "and submit again.",
            kind="warn",
        )
    else:
        _state = batch.launch(_request)
        set_batch_process(_state)
        batch_launch_message = mo.callout(
            f"Launched `{_request['workflow']}` as process {_state['process'].pid}.", kind="info"
        )
    batch_launch_message
    return


@app.cell
def _(mo):
    batch_refresh = mo.ui.refresh(
        options=[1.0, 2.0, 5.0], default_interval=1.0, label="Batch status refresh"
    )
    return (batch_refresh,)


@app.cell
def _(batch, batch_refresh, get_batch_process, mo, set_artifact_revision):
    batch_refresh.value
    _state = get_batch_process()
    _status, _elapsed, _just_promoted = batch.read_status(_state)
    if _just_promoted:
        set_artifact_revision(lambda revision: revision + 1)
    cancel_batch = mo.ui.run_button(
        label="Cancel active batch",
        kind="danger",
        disabled=_state is None or _state["process"].poll() is not None,
        on_change=lambda _value: batch.cancel(get_batch_process()),
    )
    mo.vstack(
        [
            mo.md(batch.status_markdown(_status, _elapsed)),
            mo.hstack([batch_refresh, cancel_batch], justify="start", gap=2),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
