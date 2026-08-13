import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mbe_rheed_sim import SimulationConfig, run

    return SimulationConfig, mo, np, plt, run


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
    mo.md(r"""
    ## 1. Why kinetic Monte Carlo?

    MBE contains rare, discrete events with very different rates. Residence-time kinetic
    Monte Carlo chooses event $i$ with probability $r_i / R$ and advances the physical
    clock by

    $$\Delta t = -\frac{\ln u}{R}, \qquad R=\sum_i r_i.$$

    This lets deposition and diffusion share one clock while preserving a reproducible
    stochastic trajectory when the random seed is fixed.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Baseline growth model

    The state is an integer solid-on-solid height field on a periodic lattice with six
    neighbors per site. One deposition event adds one generic growth unit. A top particle
    may diffuse laterally or down one monolayer; upward and multi-step hops are omitted.

    The global deposition rate and local diffusion rate are

    $$r_{\mathrm{dep}} = F N, \qquad
    r_{\mathrm{diff}} = \nu\exp\left[-\frac{E_{\mathrm{diff}} + nE_b}{k_B T}\right].$$

    $F$ is flux in ML/s, $N$ is the number of sites, $\nu$ is an effective attempt
    frequency, $E_{\mathrm{diff}}$ the isolated-particle barrier, and $nE_b$ the lateral
    bond contribution. The energetic defaults below are demonstration parameters.
    """)
    return


@app.cell
def _(mo):
    default_parameters = {
        "temperature_k": 800,
        "flux_ml_s": 0.5,
        "barrier_ev": 0.15,
        "size": 12,
        "coverage_ml": 2.0,
        "seed": 7,
    }
    get_parameters, _set_parameters = mo.state(default_parameters)
    parameter_form = mo.ui.dictionary(
        {
            "temperature_k": mo.ui.slider(600, 1_100, step=25, value=800, label="Temperature (K)"),
            "flux_ml_s": mo.ui.slider(0.1, 1.0, step=0.1, value=0.5, label="Flux (ML/s)"),
            "barrier_ev": mo.ui.slider(0.10, 0.40, step=0.01, value=0.15, label="Diffusion barrier (eV)"),
            "size": mo.ui.slider(8, 20, step=2, value=12, label="Lattice size"),
            "coverage_ml": mo.ui.slider(0.5, 3.0, step=0.5, value=2.0, label="Target coverage (ML)"),
            "seed": mo.ui.number(start=0, stop=10_000, value=7, label="RNG seed"),
        }
    ).form(
        submit_button_label="Run simulation",
        bordered=True,
        on_change=lambda value: _set_parameters(value or default_parameters),
    )
    mo.vstack(
        [
            mo.md("## 3. Interactive experiment\nChange values, then submit once."),
            parameter_form,
        ]
    )
    return (get_parameters,)


@app.cell
def _(SimulationConfig, get_parameters, run):
    selected_parameters = get_parameters()
    simulation_config = SimulationConfig(
        lattice_size=int(selected_parameters["size"]),
        target_coverage_ml=float(selected_parameters["coverage_ml"]),
        temperature_k=float(selected_parameters["temperature_k"]),
        deposition_flux_ml_s=float(selected_parameters["flux_ml_s"]),
        diffusion_barrier_ev=float(selected_parameters["barrier_ev"]),
        seed=int(selected_parameters["seed"]),
    )
    simulation = run(simulation_config)
    return (simulation,)


@app.cell
def _(mo, simulation):
    mo.md(f"""
    **Completed:** {simulation.deposited_events} deposition events and
    {simulation.diffusion_events} diffusion events; simulated time
    {simulation.time_s[-1]:.3f} s; final RMS roughness
    {simulation.roughness_ml[-1]:.3f} ML.
    """)
    return


@app.cell
def _(mo, simulation):
    snapshot_slider = mo.ui.slider(
        0,
        len(simulation.snapshots) - 1,
        value=len(simulation.snapshots) - 1,
        label="Recorded growth frame",
        show_value=True,
    )
    mo.vstack([mo.md("## 4. Evolving surface and RHEED connection"), snapshot_slider])
    return (snapshot_slider,)


@app.cell
def _(plt, simulation, snapshot_slider):
    _frame = snapshot_slider.value
    _heights = simulation.snapshots[_frame]
    evolution_figure, _axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    _image = _axes[0].imshow(_heights, origin="lower", cmap="viridis", vmin=0)
    _axes[0].set(
        title=f"Surface at {simulation.coverage_ml[_frame]:.2f} ML",
        xlabel="lattice x",
        ylabel="lattice y",
    )
    evolution_figure.colorbar(_image, ax=_axes[0], label="height (ML)")
    _axes[1].plot(simulation.coverage_ml, simulation.rheed_proxy, color="tab:red")
    _axes[1].scatter(
        simulation.coverage_ml[_frame],
        simulation.rheed_proxy[_frame],
        color="black",
        zorder=3,
    )
    _axes[1].set(
        xlabel="deposited coverage (ML)",
        ylabel=r"$1 - S_d$ (dimensionless)",
        title="Step-density RHEED proxy",
        ylim=(0, 1.03),
    )
    evolution_figure
    return


@app.cell
def _(mo):
    mo.md(r"""
    The proxy is $I_{\mathrm{proxy}}=1-S_d$, where $S_d$ is the fraction of unique
    nearest-neighbor bonds whose endpoint heights differ. Smooth, nearly complete layers
    have fewer steps and a larger proxy. Real RHEED phase and amplitude also depend on
    beam geometry, refraction, absorption, reconstruction, and multiple scattering.
    """)
    return


@app.cell
def _(plt, simulation):
    observable_figure, _axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True, constrained_layout=True)
    _axes[0].plot(simulation.coverage_ml, simulation.roughness_ml, color="tab:blue")
    _axes[0].set_ylabel("RMS roughness (ML)")
    _axes[1].plot(
        simulation.coverage_ml,
        simulation.island_density_per_site,
        color="tab:green",
    )
    _axes[1].set_ylabel("islands / site")
    _axes[2].plot(simulation.coverage_ml, simulation.rheed_proxy, color="tab:red")
    _axes[2].set(xlabel="deposited coverage (ML)", ylabel=r"$1-S_d$ proxy")
    observable_figure.suptitle("Growth observables")
    observable_figure
    return


@app.cell
def _(np, plt, simulation):
    _size = simulation.config.lattice_size
    _x, _y = np.meshgrid(np.arange(_size), np.arange(_size))
    morphology_3d, _axis = plt.subplots(figsize=(7, 5), subplot_kw={"projection": "3d"})
    _surface = _axis.plot_surface(
        _x,
        _y,
        simulation.final_heights,
        cmap="viridis",
        edgecolor="none",
    )
    _axis.set(xlabel="lattice x", ylabel="lattice y", zlabel="height (ML)", title="Final morphology")
    morphology_3d.colorbar(_surface, ax=_axis, shrink=0.65, label="height (ML)")
    morphology_3d
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Repeatability, comparison, and next physics

    The seed makes a run exactly reproducible, but one trajectory is not an uncertainty
    estimate. A later parameter sweep should aggregate independent seeds and report bands.

    The qualitative comparison target is the oscillation of surface step density shown
    alongside experimental RHEED traces in Figure 3 of Budagosky and Garcia-Cristobal.
    Quantitative comparison is intentionally deferred because this baseline omits their
    GaN/AlN flux calibration, desorption, step barrier, strain field, and multiscale hops.

    A defensible next RHEED stage is a kinematic layer-interference model. Full dynamical
    electron scattering remains outside the baseline.
    """)
    return


if __name__ == "__main__":
    app.run()
