"""Visual companion to `make validate-rheed`: the RHEED calculation, not the growth model.

`make validate-rheed` already checks the numbers against analytic values with tolerances
(rod positions, uniform-lift invariance, coherence FWHM, lattice-size independence). Nothing
there is a picture. These four pages are the pictures, on surfaces whose answer is known:

1. six synthetic surfaces beside their screens,
2. a 0 -> 2 ML coverage montage from a real KMC run,
3. a phase-order sweep, anti-phase -> in-phase -> anti-phase -> in-phase, on one frozen
   half-covered surface,
4. an azimuth sweep on one frozen surface, at a grazing angle where first orders exist,
5. a coherence sweep, which is what decides whether a stepped surface cancels at all.

Run with `make rheed-visuals`.
"""

import argparse
from pathlib import Path

import numpy as np
from export_preset_pdf import RATIO, preset_config
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from mbe_rheed_sim import rheed, run
from mbe_rheed_sim.lattice import initial_lattice
from mbe_rheed_sim.observables import rms_roughness_ml, step_density
from mbe_rheed_sim.rheed import SCREEN_LOG_DECADES, screen_decades
from mbe_rheed_sim.workflows import log_progress, setup_logging

ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_SIZE = 64
MONTAGE_ML = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
PHASE_ORDERS = (1, 2, 3, 4)
AZIMUTHS_DEG = (0.0, 10.0, 20.0, 30.0)
# First orders only exist above ~2.06 deg at 15 keV, so the azimuth sweep has nothing to move
# at the shallow default. Order 5 clears that, and is the angle `validate_rheed.py` already uses.
AZIMUTH_ORDER = 5
AZIMUTH_SPAN_DEG = 9.0
COHERENCE_NM = (1.0, 2.0, 4.0, 8.0, 16.0)


def synthetic_surfaces(size: int = SYNTHETIC_SIZE) -> dict[str, np.ndarray]:
    """The model's own starting surfaces, captioned for this page.

    These are exactly what `initial_surface` offers a growth run, so a screen here is the
    screen of frame 0 of the matching run rather than a separate set of test arrays.
    """
    return {
        f"{letter}  {caption}": initial_lattice(name, size)
        for letter, name, caption in (
            ("A", "flat", "perfectly flat"),
            ("B", "half-layer", "half layer, two levels"),
            ("C", "straight-step", "straight step (x2, periodic)"),
            ("D", "island", "isolated island"),
            ("E", "mounds", "periodic pyramid array"),
            ("F", "rough", "random rough"),
        )
    }


def _draw_screen(axes, pattern, *, title: str, mark_rods: bool = False) -> object:
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
    if mark_rods:
        rods = rheed.rod_orders(
            grazing_angle_deg=pattern.grazing_angle_deg,
            azimuth_deg=pattern.azimuth_deg,
            span_deg=AZIMUTH_SPAN_DEG,
        )
        axes.scatter(
            [rod.deflection_deg for rod in rods],
            [rod.exit_angle_deg for rod in rods],
            s=70,
            facecolors="none",
            edgecolors="#38bdf8",
            linewidths=1.0,
        )
    # Only draw the horizon when this screen actually reaches it. An unconditional axhline
    # stretches the axes below the data, and that empty strip renders as white — which reads
    # as high intensity next to the black below-horizon shadow.
    if pattern.exit_angle_deg[0] < 0.0 < pattern.exit_angle_deg[-1]:
        axes.axhline(0.0, color="#94a3b8", linestyle=":", linewidth=0.8)
    axes.set(
        title=title,
        xlabel="deflection (deg)",
        ylabel="exit angle (deg)",
        ylim=(pattern.exit_angle_deg[0], pattern.exit_angle_deg[-1]),
    )
    return image


def _draw_surface(axes, heights: np.ndarray, title: str) -> object:
    # Array view, not the axial-Cartesian one: this page is about what the transform was fed.
    # vmax is floored at 1 so a perfectly flat surface gets a sane colour bar, not +/-0.1.
    image = axes.imshow(
        heights, origin="lower", cmap="viridis", vmin=0, vmax=max(1, int(heights.max()))
    )
    axes.set(title=title, xticks=[], yticks=[])
    return image


def page_synthetic(surfaces: dict[str, np.ndarray], angle: float) -> Figure:
    figure = Figure(figsize=(19, 7), constrained_layout=True)
    axes = figure.subplots(2, len(surfaces))
    for column, (name, heights) in enumerate(surfaces.items()):
        pattern = rheed.diffraction_screen(heights, grazing_angle_deg=angle)
        surface_image = _draw_surface(
            axes[0, column],
            heights,
            f"{name}\n{np.mean(heights):.2f} ML, "
            f"$\\sigma_h$={rms_roughness_ml(heights):.3f}, "
            f"$S_d$={step_density(heights):.3f}",
        )
        axes[0, column].title.set_fontsize(9)
        figure.colorbar(surface_image, ax=axes[0, column], shrink=0.8)
        screen_image = _draw_screen(
            axes[1, column], pattern, title=f"$I_{{00}}$ = {pattern.specular_intensity:.4f}"
        )
        if column == len(surfaces) - 1:
            figure.colorbar(
                screen_image, ax=axes[1, column], label="log10 I (flat surface = 0)"
            )
    figure.suptitle(
        "Synthetic surfaces and their kinematic screens — "
        f"{angle:.3f}° grazing, {pattern.beam_energy_kev:.0f} keV, "
        f"{pattern.coherence_length_nm:.1f} nm coherence.\n"
        "A is the reference: any departure from it is disorder.\n"
        "B and C are both exactly 0.50 ML on two levels, yet only B cancels: cancellation needs "
        "both terraces inside one coherence area,\nand C's terraces are 10.2 nm wide against a "
        "4 nm beam. See the coherence sweep on the last page.",
        fontsize=11,
    )
    return figure


def page_montage(result, angle: float) -> Figure:
    frames = [int(np.argmin(np.abs(result.coverage_ml - target))) for target in MONTAGE_ML]
    specular = rheed.specular_intensity(result.snapshots, grazing_angle_deg=angle)

    figure = Figure(figsize=(20, 8.5), constrained_layout=True)
    grid = figure.add_gridspec(3, len(frames), height_ratios=[1.0, 1.0, 0.75])
    for column, frame in enumerate(frames):
        heights = result.snapshots[frame]
        pattern = rheed.diffraction_screen(heights, grazing_angle_deg=angle)
        _draw_surface(
            figure.add_subplot(grid[0, column]),
            heights,
            f"{result.coverage_ml[frame]:.2f} ML\n$\\sigma_h$="
            f"{result.roughness_ml[frame]:.3f}",
        )
        screen_axes = figure.add_subplot(grid[1, column])
        _draw_screen(screen_axes, pattern, title=f"$I_{{00}}$={pattern.specular_intensity:.3f}")
        if column:
            screen_axes.set_ylabel("")

    trace = figure.add_subplot(grid[2, :])
    trace.plot(result.coverage_ml, specular, color="tab:cyan", label="kinematic specular (00)")
    trace.plot(result.coverage_ml, result.rheed_proxy, color="tab:red", label=r"$1-S_d$ proxy")
    trace.scatter(
        result.coverage_ml[frames],
        specular[frames],
        color="#111827",
        zorder=3,
        label="montage frames",
    )
    trace.set(xlabel="coverage (ML)", ylabel="normalized signal", xlim=(0, 2.05), ylim=(0, 1.05))
    trace.legend(loc="lower right", fontsize=8, ncol=3)
    figure.suptitle(
        f"Coverage montage, 0 to 2 ML — Ga/N = {RATIO} paper preset, "
        f"{result.config.lattice_size}x{result.config.lattice_size}, {angle:.3f}° grazing.\n"
        "Roughens near half coverage, partially smooths near integer ML: one cycle per\n"
        "monolayer, but not a return to flat — at 1 ML $\\sigma_h$ is still 0.246 and the "
        "specular recovers only to 0.781. The incomplete recovery is the damping.",
        fontsize=12,
    )
    return figure


def page_phase_orders(heights: np.ndarray) -> Figure:
    figure = Figure(figsize=(15, 8), constrained_layout=True)
    axes = figure.subplots(2, len(PHASE_ORDERS))
    for column, order in enumerate(PHASE_ORDERS):
        angle = rheed.antiphase_grazing_angle_deg(order)
        pattern = rheed.diffraction_screen(heights, grazing_angle_deg=angle, span_deg=1.5)
        image = _draw_screen(
            axes[0, column],
            pattern,
            title=(
                f"$q_z d/\\pi$ = {order} — {pattern.condition}\n"
                f"{angle:.3f}° grazing, $I_{{00}}$ = {pattern.specular_intensity:.4f}"
            ),
        )
        if column == len(PHASE_ORDERS) - 1:
            figure.colorbar(image, ax=axes[0, column], label="log10 I (flat surface = 0)")
        # Horizontal cut along the specular row. Two different quantities live on this line and
        # they must not be conflated: the specular pixel at zero deflection, which is what
        # `specular_intensity` reports, and the largest value anywhere on the row, which at an
        # anti-phase order is a diffuse feature well away from the beam.
        row = rheed.specular_row(pattern)
        axes[1, column].semilogy(
            pattern.deflection_deg,
            np.maximum(row, 10.0**-SCREEN_LOG_DECADES),
            color="tab:cyan",
        )
        axes[1, column].axvline(0.0, color="#111827", linestyle=":", linewidth=0.8)
        axes[1, column].set(
            xlabel="deflection (deg)",
            ylabel="I (flat = 1)" if column == 0 else "",
            ylim=(10.0**-SCREEN_LOG_DECADES, 1.5),
            title=(
                f"specular pixel {pattern.specular_intensity:.4f}\n"
                f"row max {row.max():.4f} at "
                f"{pattern.deflection_deg[int(np.argmax(row))]:+.2f}°"
            ),
        )
        axes[1, column].title.set_fontsize(9)
    figure.suptitle(
        "Phase-order sweep on one frozen half-covered two-level surface "
        f"({np.mean(heights):.2f} ML).\n"
        "Odd orders put adjacent terraces pi out of phase and must cancel; even orders must "
        "recover. The surface never changes — only the beam does.",
        fontsize=12,
    )
    return figure


def page_azimuths(heights: np.ndarray) -> Figure:
    angle = rheed.antiphase_grazing_angle_deg(AZIMUTH_ORDER)
    figure = Figure(figsize=(17, 5.5), constrained_layout=True)
    axes = figure.subplots(1, len(AZIMUTHS_DEG))
    for column, azimuth in enumerate(AZIMUTHS_DEG):
        pattern = rheed.diffraction_screen(
            heights,
            grazing_angle_deg=angle,
            azimuth_deg=azimuth,
            span_deg=AZIMUTH_SPAN_DEG,
        )
        rods = rheed.rod_orders(
            grazing_angle_deg=angle, azimuth_deg=azimuth, span_deg=AZIMUTH_SPAN_DEG
        )
        image = _draw_screen(
            axes[column],
            pattern,
            title=(
                f"azimuth {azimuth:g}° — {len(rods)} reachable rods\n"
                + "\n".join(
                    ", ".join(rod.label for rod in rods[start : start + 4])
                    for start in range(0, len(rods), 4)
                )
            ),
            mark_rods=True,
        )
        axes[column].title.set_fontsize(9)
        if column == len(AZIMUTHS_DEG) - 1:
            figure.colorbar(image, ax=axes[column], label="log10 I (flat surface = 0)")
    figure.suptitle(
        f"Azimuth sweep on one frozen surface — {angle:.3f}° grazing (phase order "
        f"{AZIMUTH_ORDER}), where first orders exist.\n"
        "Rotating the sample sweeps the Ewald sphere across a hexagonal reciprocal lattice, so "
        "which rods are reachable changes. Circles are `rod_orders()`, not hand placement.\n"
        "0° and 30° are the two inequivalent high-symmetry directions; the pattern repeats "
        "every 60° and mirrors at 30°.",
        fontsize=10,
    )
    return figure


def page_coherence(surfaces: dict[str, np.ndarray], angle: float) -> Figure:
    """Why `coherence_length_nm` is the parameter that decides whether a step cancels.

    The rod width is an instrument property, and so is anti-phase cancellation on a stepped
    surface: a beam that only ever illuminates one terrace at a time sees a flat surface.
    """
    step_name = "C  straight step (x2, periodic)"
    half_name = "B  half layer, two levels"
    terrace_nm = len(surfaces[step_name]) // 2 * rheed.GAN_IN_PLANE_SPACING_NM

    figure = Figure(figsize=(17, 8), constrained_layout=True)
    grid = figure.add_gridspec(2, len(COHERENCE_NM), height_ratios=[1.0, 0.8])
    patterns = []
    for column, coherence in enumerate(COHERENCE_NM):
        pattern = rheed.diffraction_screen(
            surfaces[step_name],
            grazing_angle_deg=angle,
            coherence_length_nm=coherence,
            span_deg=1.5,
        )
        patterns.append(pattern)
        axes = figure.add_subplot(grid[0, column])
        image = _draw_screen(
            axes,
            pattern,
            title=(
                f"$L_c$ = {coherence:g} nm — $I_{{00}}$ = {pattern.specular_intensity:.4f}\n"
                f"rod FWHM {pattern.streak_width_deg:.3f}°"
            ),
        )
        axes.title.set_fontsize(9)
        if column == len(COHERENCE_NM) - 1:
            figure.colorbar(image, ax=axes, label="log10 I (flat surface = 0)")

    specular_axes = figure.add_subplot(grid[1, :2])
    for name, colour in ((step_name, "tab:orange"), (half_name, "tab:purple")):
        specular_axes.semilogx(
            COHERENCE_NM,
            [
                float(
                    rheed.specular_intensity(
                        surfaces[name], grazing_angle_deg=angle, coherence_length_nm=coherence
                    )
                )
                for coherence in COHERENCE_NM
            ],
            marker="o",
            color=colour,
            label=name,
        )
    specular_axes.axvline(terrace_nm, color="#94a3b8", linestyle="--")
    specular_axes.annotate(
        f"terrace width {terrace_nm:.1f} nm",
        (terrace_nm, 0.55),
        fontsize=8,
        rotation=90,
        ha="right",
    )
    specular_axes.set(
        xlabel="coherence length $L_c$ (nm)",
        ylabel="specular $I_{00}$",
        ylim=(-0.05, 1.05),
        title="Cancellation switches on once the beam spans both terraces",
    )
    specular_axes.legend(fontsize=8)

    # Measured against predicted, not the prediction on its own. The rod is a fraction of a
    # degree wide, so it needs a finely resolved screen or the measurement returns its own
    # pixel pitch; a flat surface isolates the instrument width from any surface disorder.
    measured = []
    for coherence in COHERENCE_NM:
        fine = rheed.diffraction_screen(
            surfaces["A  perfectly flat"],
            grazing_angle_deg=angle,
            coherence_length_nm=coherence,
            span_deg=1.0,
            shape=(41, 801),
        )
        measured.append(rheed.measured_rod_fwhm_deg(fine))

    width_axes = figure.add_subplot(grid[1, 2:])
    width_axes.loglog(
        COHERENCE_NM,
        [pattern.streak_width_deg for pattern in patterns],
        color="tab:cyan",
        label=r"analytic $4\sqrt{2}\ln 2 / (L_c k \cos\theta)$",
    )
    width_axes.loglog(
        COHERENCE_NM,
        measured,
        marker="o",
        linestyle="none",
        color="#111827",
        label="measured off the screen (FWHM of the specular row)",
    )
    width_axes.set(
        xlabel="coherence length $L_c$ (nm)",
        ylabel="rod FWHM (deg)",
        title=(
            "Measured rod width follows the analytic $1/L_c$ prediction\n"
            f"(largest deviation {max(abs(m - p.streak_width_deg) for m, p in zip(measured, patterns, strict=True)):.4f}°"
            " — `make validate-rheed` asserts this)"
        ),
    )
    width_axes.legend(fontsize=8)
    width_axes.title.set_fontsize(9)
    figure.suptitle(
        "Coherence sweep on one frozen stepped surface (0.50 ML, two levels) — "
        f"{angle:.3f}° grazing.\n"
        "Longer coherence narrows the rods and, on this surface, destroys the specular beam:\n"
        "at 1 nm the beam sees one flat terrace ($I_{00}$ = 0.99), at 16 nm it spans both and "
        "they cancel ($I_{00}$ = 0.002). Same surface throughout.",
        fontsize=11,
    )
    return figure


def main(*, lattice_size: int, output: Path) -> None:
    setup_logging()
    output.parent.mkdir(parents=True, exist_ok=True)
    angle = rheed.antiphase_grazing_angle_deg(rheed.DEFAULT_PHASE_ORDER)
    surfaces = synthetic_surfaces()
    result = run(
        preset_config({}, lattice_size), on_progress=log_progress(f"montage {lattice_size}")
    )
    with PdfPages(output) as pdf:
        pdf.savefig(page_synthetic(surfaces, angle))
        pdf.savefig(page_montage(result, angle))
        pdf.savefig(page_phase_orders(surfaces["B  half layer, two levels"]))
        pdf.savefig(page_azimuths(surfaces["B  half layer, two levels"]))
        pdf.savefig(page_coherence(surfaces, angle))
    print(f"wrote {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "rheed_visuals.pdf")
    arguments = parser.parse_args()
    main(lattice_size=arguments.size, output=arguments.output)
