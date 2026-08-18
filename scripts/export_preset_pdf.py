"""One PDF page per gallery preset: final 3D surface, RHEED signals and the numbers used.

Every preset runs on the paper's fitted GaN physics (`figure3_config`, Ga/N = 0.82) rather
than the notebook's teaching energetics, with one knob moved per preset. Run with
`make preset-pdf` (add `SIZES=64` for a cheaper pass).
"""

import argparse
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from mbe_rheed_notebook.figures import SCREEN_LOG_DECADES, screen_decades
from mbe_rheed_sim import rheed, run
from mbe_rheed_sim.analysis import rheed_oscillation_metrics
from mbe_rheed_sim.paper import figure3_config, figure3_parameters
from mbe_rheed_sim.workflows import log_progress, setup_logging

ROOT = Path(__file__).resolve().parents[1]
RATIO = 0.82
COVERAGE_ML = 3.0
SEED = 7

# Only the fields a preset is allowed to move, and how to name them in the provenance line.
MODIFIED_FIELD_LABELS = {
    "temperature_k": ("T", "{:.1f} K"),
    "deposition_flux_ml_s": ("effective Ga flux", "{:.4f} ML/s"),
    "attempt_frequency_hz": ("attempt frequency", "{:.3g} Hz"),
    "step_barrier_ev": ("E_step", "{:.4f} eV"),
}

# Each preset is the paper parameterization with one physical knob moved. Keys match
# data/gallery/index.json; the titles deliberately do not, because they state what these runs
# measured rather than what the teaching-parameter gallery claims. Removing the down-step
# barrier comes out *smoother* than the paper preset, not more islanded, and 3x flux roughens
# only slightly and still oscillates, so neither "island growth" nor "flux too high" survives.
PRESETS = {
    "gan-paper-082": ("Layer-by-layer growth (GaN paper parameters)", {}),
    "island-growth": (
        "Enhanced interlayer smoothing (no Ehrlich-Schwoebel barrier)",
        {"step_barrier_ev": 0.0},
    ),
    "step-barrier-mounding": (
        "Mounding from a strong Ehrlich-Schwoebel barrier",
        {"step_barrier_ev": 0.25},
    ),
    # The island-density row on the page carries the evidence for "dense": 850 K nucleates
    # several times denser than the paper preset, so this claim is measured, not asserted.
    "too-cold": ("Low temperature: reduced mobility, dense nucleation", {"temperature_k": 850.0}),
    "too-fast": ("High flux: 3x effective Ga flux", {"flux_scale": 3.0}),
    "no-diffusion": ("No diffusion: pure random deposition", {"attempt_frequency_hz": 0.0}),
}


def preset_config(overrides: dict, lattice_size: int) -> "object":
    base = figure3_config(RATIO, lattice_size=lattice_size, seed=SEED)
    scale = overrides.pop("flux_scale", 1.0)
    return replace(
        base,
        target_time_s=None,
        target_coverage_ml=COVERAGE_ML,
        deposition_flux_ml_s=base.deposition_flux_ml_s * scale,
        **overrides,
    )


def modification_note(config, base) -> str:
    """Which fields this preset moved away from the unmodified paper configuration."""
    changes = [
        f"{label} = {form.format(getattr(config, field))} "
        f"(paper {form.format(getattr(base, field))})"
        for field, (label, form) in MODIFIED_FIELD_LABELS.items()
        if getattr(config, field) != getattr(base, field)
    ]
    return ", ".join(changes)


def _surface_panel(figure: Figure, result) -> None:
    heights = np.asarray(result.final_heights, dtype=float)
    row, column = np.indices(heights.shape)
    # Axial (q, r) sites sit at these Cartesian centres, so the hex lattice reads correctly.
    x, y = column + 0.5 * row, np.sqrt(3.0) / 2.0 * row
    # Anchor the height axis at zero: autoscaling a nearly flat surface magnifies one adatom
    # into a spike and makes pages incomparable.
    zmax = max(1.0, np.ceil(heights.max()))
    axes = figure.add_subplot(2, 2, 1, projection="3d")
    surface = axes.plot_surface(
        x, y, heights, cmap="viridis", vmin=0.0, vmax=zmax, linewidth=0, rstride=1, cstride=1
    )
    axes.set(
        title=f"Final surface, {result.coverage_ml[-1]:.2f} ML",
        xlabel="q + r/2",
        ylabel="sqrt(3) r / 2",
        zlabel="height (ML)",
        zlim=(0.0, zmax),
    )
    axes.set_box_aspect((1.0, 1.0, 0.55))
    axes.view_init(elev=32, azim=-135)
    figure.colorbar(surface, ax=axes, shrink=0.55, pad=0.1, label="height (ML)")


def _screen_panel(figure: Figure, pattern) -> None:
    # Log scale, because the specular beam sits orders above the diffuse background. The zero
    # point is the *flat surface* at the same beam condition, which is what `intensity` is
    # normalized to, so pages are directly comparable: a run whose specular has collapsed
    # reads as dark here rather than being renormalized back to full brightness.
    axes = figure.add_subplot(2, 2, 2)
    image = axes.imshow(
        screen_decades(pattern),
        origin="lower",
        cmap="inferno",
        aspect="auto",
        vmin=-SCREEN_LOG_DECADES,
        vmax=0.0,
        extent=(
            pattern.deflection_deg[0],
            pattern.deflection_deg[-1],
            pattern.exit_angle_deg[0],
            pattern.exit_angle_deg[-1],
        ),
    )
    axes.set(
        title=f"Kinematic screen, {pattern.condition} ({pattern.grazing_angle_deg:.2f}°)",
        xlabel="deflection (deg)",
        ylabel="exit angle (deg)",
    )
    figure.colorbar(image, ax=axes, label="log10 intensity (flat surface = 0)")


def _trace_panel(figure: Figure, result, specular) -> None:
    axes = figure.add_subplot(2, 2, 3)
    axes.plot(result.coverage_ml, result.rheed_proxy, color="tab:red", label=r"$1-S_d$ proxy")
    axes.plot(
        result.coverage_ml,
        specular / specular[0],
        color="tab:cyan",
        linestyle="--",
        label="kinematic specular (00)",
    )
    twin = axes.twinx()
    twin.plot(result.coverage_ml, result.roughness_ml, color="tab:blue", linewidth=1, alpha=0.6)
    twin.set_ylabel("RMS roughness (ML)", color="tab:blue")
    axes.set(xlabel="coverage (ML)", ylabel="normalized signal", ylim=(0, 1.05))
    axes.legend(loc="lower left", fontsize=7)


def _numbers_panel(
    figure: Figure, index: int, name: str, result, pattern, specular, elapsed: float
) -> None:
    config = asdict(result.config)
    metrics = rheed_oscillation_metrics(result.coverage_ml, result.rheed_proxy)
    rows = [
        ("run number", f"{index} of {len(PRESETS)}"),
        ("preset key", name),
        (
            "lattice",
            (
                f"{config['lattice_size']}x{config['lattice_size']}"
                f" ({config['lattice_size'] ** 2} sites)"
            ),
        ),
        ("target coverage", f"{config['target_coverage_ml']:.2f} ML"),
        ("temperature", f"{config['temperature_k']:.1f} K"),
        ("deposition flux", f"{config['deposition_flux_ml_s']:.4f} ML/s"),
        ("attempt frequency", f"{config['attempt_frequency_hz']:.3g} Hz"),
        ("E_diff", f"{config['diffusion_barrier_ev']:.4f} eV"),
        ("E_bond", f"{config['lateral_bond_energy_ev']:.4f} eV"),
        ("E_step", f"{config['step_barrier_ev']:.4f} eV"),
        ("E_des", f"{config['desorption_barrier_ev']:.4f} eV"),
        ("max hop distance", str(config["max_isolated_hop_distance"])),
        ("seed", str(config["seed"])),
        ("frames stored", str(len(result.snapshots))),
        ("simulated time", f"{result.time_s[-1]:.3f} s"),
        (
            "deposited / hops / desorbed",
            f"{result.deposited_events} / {result.diffusion_events} / {result.desorbed_events}",
        ),
        ("final RMS roughness", f"{result.roughness_ml[-1]:.4f} ML"),
        (
            "island density peak / final",
            (
                f"{result.island_density_per_site.max():.5f}"
                f" / {result.island_density_per_site[-1]:.5f} per site"
            ),
        ),
        ("proxy period", "n/a" if metrics.period_ml is None else f"{metrics.period_ml:.3f} ML"),
        ("proxy oscillatory", str(metrics.is_oscillatory)),
        ("specular final / flat", f"{specular[-1] / specular[0]:.4f}"),
        (
            "beam",
            (
                f"{pattern.beam_energy_kev:.0f} keV, {pattern.grazing_angle_deg:.3f}°, "
                f"{pattern.coherence_length_nm:.1f} nm coherence"
            ),
        ),
        # Box geometry only: lattice size against coherence length. Identical on every page
        # here by construction, and not a morphology measurement.
        ("satellite ratio (box geometry)", f"{pattern.satellite_artifact_ratio:.3f}"),
        ("wall time", f"{elapsed:.1f} s"),
    ]
    axes = figure.add_subplot(2, 2, 4)
    axes.axis("off")
    table = axes.table(cellText=rows, colWidths=[0.5, 0.5], loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.15)


def page(index: int, name: str, title: str, result, base, elapsed: float) -> Figure:
    angle = rheed.antiphase_grazing_angle_deg(rheed.DEFAULT_PHASE_ORDER)
    pattern = rheed.diffraction_screen(result.final_heights, grazing_angle_deg=angle)
    specular = rheed.specular_intensity(result.snapshots, grazing_angle_deg=angle)
    figure = Figure(figsize=(14, 9), constrained_layout=True)
    _surface_panel(figure, result)
    _screen_panel(figure, pattern)
    _trace_panel(figure, result, specular)
    _numbers_panel(figure, index, name, result, pattern, specular, elapsed)
    # The paper's predicted growth rate only describes the unmodified preset, so a run that
    # moved the flux or the temperature does not get to quote it.
    modified = modification_note(result.config, base)
    provenance = (
        f"Ga/N = {RATIO} paper preset, unmodified; predicted growth rate "
        f"{figure3_parameters(RATIO).predicted_growth_rate_ml_s:.4f} ML/s"
        if not modified
        else f"derived from the Ga/N = {RATIO} paper preset; modified: {modified}"
    )
    figure.suptitle(f"Run {index} of {len(PRESETS)} — {title}\n{provenance}", fontsize=13)
    return figure


def main(*, lattice_size: int, output: Path) -> None:
    setup_logging()
    output.parent.mkdir(parents=True, exist_ok=True)
    base = preset_config({}, lattice_size)
    with PdfPages(output) as pdf:
        for index, (name, (title, overrides)) in enumerate(PRESETS.items(), start=1):
            config = preset_config(dict(overrides), lattice_size)
            started = perf_counter()
            result = run(config, on_progress=log_progress(f"{name} {lattice_size}x{lattice_size}"))
            elapsed = perf_counter() - started
            pdf.savefig(page(index, name, title, result, base, elapsed))
            print(f"{index} {name:24s} {elapsed:7.1f} s  roughness {result.roughness_ml[-1]:.4f} ML")
    print(f"wrote {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "preset_gallery.pdf")
    arguments = parser.parse_args()
    main(lattice_size=arguments.size, output=arguments.output)
