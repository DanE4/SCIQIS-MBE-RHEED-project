"""Numerical validation of the kinematic RHEED module.

Writes `outputs/rheed_validation/` with the geometry it used, the analytic-versus-numeric
error table, and the screens for flat, stepped and rough morphologies plus the azimuth and
broadening comparisons. Every claim here is a number, not a picture: the pictures only show
what the numbers already assert, and the script fails if any tolerance is missed.

The published reference case is the exact Ewald-to-screen transformation of Liu, Chang and
Zou, J. Vac. Sci. Technol. B 40, 054002 (2022), Eqs. (5)-(6). That transformation is
implemented here independently, from the paper, and required to agree with this module's
angular screen mapping. Dynamical intensities are not compared, because this model does not
compute them.
"""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mbe_rheed_sim import rheed
from mbe_rheed_sim.workflows import artifact_root, update_progress

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = "outputs/rheed_validation"
GRAZING_DEG = rheed.antiphase_grazing_angle_deg(5)
SIZE = 32
SCREEN_DECADES = 3.0
# Liu et al. section II: 10 keV, about 2.5 degrees grazing, Ge(111) surface mesh.
GE111_SPACING_NM = 0.4001


def _surfaces() -> dict[str, np.ndarray]:
    """Three morphologies from the same lattice: layer-complete, stepped, three-dimensional."""
    generator = np.random.default_rng(20260818)
    flat = np.zeros((SIZE, SIZE), dtype=np.int64)
    stepped = flat.copy()
    stepped.ravel()[generator.permutation(SIZE * SIZE)[: (SIZE * SIZE) // 2]] = 1
    rough = generator.integers(0, 6, (SIZE, SIZE)).astype(np.int64)
    return {"flat": flat, "stepped": stepped, "rough": rough}


def _liu_screen_coordinates(
    exit_angle_deg: float, deflection_deg: float, distance_mm: float
) -> tuple[float, float]:
    """Flat-screen position of an outgoing direction, screen perpendicular to the surface."""
    exit_angle = math.radians(exit_angle_deg)
    deflection = math.radians(deflection_deg)
    return (
        distance_mm * math.tan(deflection),
        distance_mm * math.tan(exit_angle) / math.cos(deflection),
    )


def _liu_reciprocal(
    screen_x_mm: float, screen_y_mm: float, distance_mm: float, wavenumber: float, grazing_deg: float
) -> tuple[float, float]:
    """Liu et al. Eqs. (5)-(6): flat-screen position back to in-plane reciprocal coordinates.

    Returns their `(x, y)`, where `x` is the transverse component of the momentum transfer
    and `y` is its component along the beam with the opposite sign.
    """
    radius = math.sqrt(distance_mm**2 + screen_x_mm**2 + screen_y_mm**2)
    return (
        wavenumber * screen_x_mm / radius,
        wavenumber * (-distance_mm / radius + math.cos(math.radians(grazing_deg))),
    )


def _published_geometry_check() -> dict[str, object]:
    """Cross-check our screen mapping against the primary paper's exact transformation."""
    wavenumber = rheed.wavenumber_per_nm(10.0)
    distance_mm = 300.0
    worst = 0.0
    rods = rheed.rod_orders(
        grazing_angle_deg=2.5,
        energy_kev=10.0,
        in_plane_spacing_nm=GE111_SPACING_NM,
        span_deg=6.0,
    )
    reciprocal = rheed.reciprocal_basis(GE111_SPACING_NM)
    for rod in rods:
        screen = _liu_screen_coordinates(rod.exit_angle_deg, rod.deflection_deg, distance_mm)
        transverse, along = _liu_reciprocal(*screen, distance_mm, wavenumber, 2.5)
        vector = rod.h * reciprocal[0] + rod.k * reciprocal[1]
        worst = max(worst, abs(transverse - vector[1]), abs(along + vector[0]))

    # Liu et al. Fig. 3(a): at this condition the 1x1 first order and a 2x2 half order are
    # both on the screen while the 1x1 second order is not.
    lengths = {
        "half_order_2x2": 2.0 * math.pi / (GE111_SPACING_NM * math.sqrt(3.0)),
        "first_order_1x1": 4.0 * math.pi / (GE111_SPACING_NM * math.sqrt(3.0)),
        "second_order_1x1": 8.0 * math.pi / (GE111_SPACING_NM * math.sqrt(3.0)),
    }
    reachable = wavenumber * math.sin(math.radians(2.5))
    return {
        "reference": "Liu, Chang & Zou, J. Vac. Sci. Technol. B 40, 054002 (2022), Eqs. (5)-(6)",
        "case": "Ge(111), 10 keV, 2.5 deg grazing, a = 0.4001 nm",
        "sample_screen_distance_mm": distance_mm,
        "orders_compared": len(rods),
        "max_reciprocal_error_per_nm": worst,
        "reachable_in_plane_momentum_per_nm": reachable,
        "rod_lengths_per_nm": lengths,
        "reachable": {name: bool(value <= reachable) for name, value in lengths.items()},
    }


def _geometry_record() -> dict[str, object]:
    wavelength = rheed.electron_wavelength_nm(rheed.DEFAULT_BEAM_ENERGY_KEV)
    wavenumber = rheed.wavenumber_per_nm(rheed.DEFAULT_BEAM_ENERGY_KEV)
    real = rheed.surface_basis()
    reciprocal = rheed.reciprocal_basis()
    incident = rheed.incident_wavevector(GRAZING_DEG)
    return {
        "beam_energy_kev": rheed.DEFAULT_BEAM_ENERGY_KEV,
        "electron_wavelength_nm": wavelength,
        "wavenumber_per_nm": wavenumber,
        "grazing_angle_deg": GRAZING_DEG,
        "phase_order_qz_d_over_pi": rheed.phase_order(GRAZING_DEG),
        "incident_wavevector_per_nm": incident.tolist(),
        "incident_wavevector_norm_error": abs(float(np.linalg.norm(incident)) - wavenumber),
        "in_plane_spacing_nm": rheed.GAN_IN_PLANE_SPACING_NM,
        "layer_height_nm": rheed.GAN_LAYER_HEIGHT_NM,
        "real_basis_nm": real.tolist(),
        "reciprocal_basis_per_nm": reciprocal.tolist(),
        "duality_max_error": float(np.abs(real @ reciprocal.T - 2 * math.pi * np.eye(2)).max()),
        "first_order_critical_grazing_deg": math.degrees(
            math.asin(float(np.linalg.norm(reciprocal[0])) / wavenumber)
        ),
        "coherence_length_nm": rheed.DEFAULT_COHERENCE_LENGTH_NM,
        "rod_fwhm_per_nm": 4.0
        * math.sqrt(2.0)
        * math.log(2.0)
        / rheed.DEFAULT_COHERENCE_LENGTH_NM,
        "satellite_artifact_ratio_at_size_32": rheed.satellite_artifact_ratio(32),
        "rods": [rod._asdict() | {"label": rod.label} for rod in rheed.rod_orders(
            grazing_angle_deg=GRAZING_DEG, span_deg=3.0
        )],
        "published_reference_case": _published_geometry_check(),
    }


def _elastic_error(pattern: rheed.ScreenPattern) -> float:
    wavenumber = rheed.wavenumber_per_nm(pattern.beam_energy_kev)
    exit_angle = np.radians(pattern.exit_angle_deg)[:, None]
    deflection = np.radians(pattern.deflection_deg)[None, :]
    norm = np.hypot(
        wavenumber * np.cos(exit_angle) * np.hypot(np.cos(deflection), np.sin(deflection)),
        wavenumber * np.sin(exit_angle) + 0.0 * deflection,
    )
    return float(np.abs(norm - wavenumber).max() / wavenumber)


def _direct_sum_error(lattice: np.ndarray, samples: int = 64) -> float:
    """Largest gap between the interpolated transform path and the literal scatterer sum."""
    spacing, height = rheed.GAN_IN_PLANE_SPACING_NM, rheed.GAN_LAYER_HEIGHT_NM
    wavenumber = rheed.wavenumber_per_nm()
    pattern = rheed.diffraction_screen(
        lattice, grazing_angle_deg=GRAZING_DEG, coherence_patches=1
    )
    windows, profile = rheed._illuminated_patches(
        lattice, rheed.DEFAULT_COHERENCE_LENGTH_NM, spacing, 1
    )
    window = windows[0]
    half = len(profile) // 2
    row, column = np.indices(window.shape)
    x = spacing * (column - half + 0.5 * (row - half))
    y = spacing * (math.sqrt(3.0) / 2.0) * (row - half)

    incidence = math.radians(GRAZING_DEG)
    generator = np.random.default_rng(7)
    lit = np.nonzero(pattern.exit_angle_deg > 0.0)[0]
    worst = 0.0
    for _ in range(samples):
        row_index = int(generator.choice(lit))
        column_index = int(generator.integers(len(pattern.deflection_deg)))
        exit_angle = math.radians(pattern.exit_angle_deg[row_index])
        deflection = math.radians(pattern.deflection_deg[column_index])
        q_x = wavenumber * (math.cos(exit_angle) * math.cos(deflection) - math.cos(incidence))
        q_y = wavenumber * math.cos(exit_angle) * math.sin(deflection)
        q_z = wavenumber * (math.sin(exit_angle) + math.sin(incidence))
        phase = q_x * x + q_y * y + q_z * height * window
        direct = abs(np.sum(profile * np.exp(-1j * phase)) / profile.sum()) ** 2
        worst = max(worst, abs(direct - pattern.intensity[row_index, column_index]))
    return worst


def _analytic_versus_numeric(surfaces: dict[str, np.ndarray]) -> dict[str, object]:
    flat = surfaces["flat"]
    reference = rheed.diffraction_screen(
        flat, grazing_angle_deg=GRAZING_DEG, span_deg=1.0, shape=(41, 801)
    )
    row = reference.intensity[reference.intensity.shape[0] // 2]

    rod_errors = []
    for rod in rheed.rod_orders(grazing_angle_deg=GRAZING_DEG, span_deg=3.0):
        expected_parallel = np.array(
            [
                rheed.wavenumber_per_nm() * math.cos(math.radians(GRAZING_DEG)),
                0.0,
            ]
        ) + rod.h * rheed.reciprocal_basis()[0] + rod.k * rheed.reciprocal_basis()[1]
        vertical = math.sqrt(
            rheed.wavenumber_per_nm() ** 2 - float(expected_parallel @ expected_parallel)
        )
        rod_errors.append(
            {
                "label": rod.label,
                "exit_angle_deg": rod.exit_angle_deg,
                "deflection_deg": rod.deflection_deg,
                "exit_angle_error_deg": abs(
                    rod.exit_angle_deg
                    - math.degrees(math.asin(vertical / rheed.wavenumber_per_nm()))
                ),
                "deflection_error_deg": abs(
                    rod.deflection_deg
                    - math.degrees(math.atan2(expected_parallel[1], expected_parallel[0]))
                ),
            }
        )

    lifted = rheed.diffraction_screen(surfaces["stepped"] + 5, grazing_angle_deg=GRAZING_DEG)
    plain = rheed.diffraction_screen(surfaces["stepped"], grazing_angle_deg=GRAZING_DEG)

    sizes = {}
    for size in (16, 32, 64, 128):
        pattern = rheed.diffraction_screen(
            np.zeros((size, size), dtype=np.int64),
            grazing_angle_deg=GRAZING_DEG,
            span_deg=1.0,
            shape=(41, 801),
        )
        line = pattern.intensity[pattern.intensity.shape[0] // 2]
        sizes[str(size)] = {
            "specular_intensity": pattern.specular_intensity,
            "measured_rod_fwhm_deg": rheed.half_max_width(pattern.deflection_deg, line),
        }

    resolutions = {}
    for shape in ((91, 201), (181, 401), (361, 801)):
        pattern = rheed.diffraction_screen(
            surfaces["stepped"], grazing_angle_deg=GRAZING_DEG, span_deg=1.0, shape=shape
        )
        lit = pattern.exit_angle_deg > 0.0
        step_x = pattern.deflection_deg[1] - pattern.deflection_deg[0]
        step_y = pattern.exit_angle_deg[1] - pattern.exit_angle_deg[0]
        resolutions[f"{shape[0]}x{shape[1]}"] = {
            "specular_intensity": pattern.specular_intensity,
            "integrated_lit_intensity_deg2": float(pattern.intensity[lit].sum())
            * step_x
            * step_y,
        }

    broadening = {}
    for name, keywords in (
        ("coherence_2nm", {"coherence_length_nm": 2.0}),
        ("coherence_4nm", {}),
        ("coherence_8nm", {"coherence_length_nm": 8.0}),
        ("divergence_0.30deg", {"divergence_deg": 0.30}),
        ("detector_psf_0.20deg", {"detector_psf_deg": 0.20}),
    ):
        pattern = rheed.diffraction_screen(
            flat, grazing_angle_deg=GRAZING_DEG, span_deg=1.0, shape=(41, 801), **keywords
        )
        line = pattern.intensity[pattern.intensity.shape[0] // 2]
        broadening[name] = {
            "measured_fwhm_deg": rheed.half_max_width(pattern.deflection_deg, line),
            "analytic_coherence_fwhm_deg": pattern.streak_width_deg,
            "peak_intensity": float(line.max()),
        }

    azimuths = {
        f"{azimuth:g}deg": [
            {"label": rod.label, "exit_angle_deg": rod.exit_angle_deg,
             "deflection_deg": rod.deflection_deg}
            for rod in rheed.rod_orders(
                grazing_angle_deg=GRAZING_DEG, azimuth_deg=azimuth, span_deg=9.0
            )
        ]
        for azimuth in (0.0, 10.0, 20.0, 30.0)
    }

    mirror = rheed.diffraction_screen(flat, grazing_angle_deg=GRAZING_DEG)
    return {
        "elastic_relative_error": _elastic_error(reference),
        "direct_sum_max_absolute_error": _direct_sum_error(surfaces["stepped"]),
        "flat_specular_intensity": reference.specular_intensity,
        "measured_rod_fwhm_deg": rheed.half_max_width(reference.deflection_deg, row),
        "analytic_rod_fwhm_deg": reference.streak_width_deg,
        "rod_position_errors": rod_errors,
        "uniform_lift_max_intensity_change": float(
            np.abs(lifted.intensity - plain.intensity).max()
        ),
        "mirror_symmetry_max_error": float(
            np.abs(mirror.intensity - mirror.intensity[:, ::-1]).max()
        ),
        "specular_curve_vs_screen_pixel_error": abs(
            float(rheed.specular_intensity(surfaces["stepped"], grazing_angle_deg=GRAZING_DEG))
            - plain.specular_intensity
        ),
        "below_horizon_max_intensity": float(
            plain.intensity[plain.exit_angle_deg < 0.0].max()
        ),
        "lattice_size_independence": sizes,
        "detector_resolution_convergence": resolutions,
        "broadening": broadening,
        "azimuth_rod_positions": azimuths,
    }


def _draw(pattern: rheed.ScreenPattern, title: str, path: Path) -> None:
    figure, axes = plt.subplots(figsize=(6.2, 4.4), constrained_layout=True)
    image = axes.pcolormesh(
        pattern.deflection_deg,
        pattern.exit_angle_deg,
        np.log10(np.maximum(pattern.intensity, 10.0**-SCREEN_DECADES)),
        cmap="inferno",
        vmin=-SCREEN_DECADES,
        vmax=0.0,
        shading="nearest",
    )
    axes.axhline(0.0, color="#94a3b8", linestyle=":", linewidth=1)
    for rod in pattern.rods:
        axes.plot(rod.deflection_deg, rod.exit_angle_deg, "+", color="#94a3b8", markersize=9)
        axes.annotate(
            rod.label,
            (rod.deflection_deg, rod.exit_angle_deg),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            color="#94a3b8",
            fontsize=8,
        )
    axes.set_xlabel("horizontal deflection (degrees)")
    axes.set_ylabel("exit angle above surface (degrees)")
    axes.set_title(title, fontsize=10)
    axes.set_aspect("equal")
    axes.set_xlim(pattern.deflection_deg[0], pattern.deflection_deg[-1])
    axes.set_ylim(pattern.exit_angle_deg[0], pattern.exit_angle_deg[-1])
    figure.colorbar(image, ax=axes, label="log$_{10}$ I (flat = 0)")
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _draw_comparison(patterns: list[tuple[str, rheed.ScreenPattern]], title: str, path: Path):
    figure, axes = plt.subplots(
        1, len(patterns), figsize=(4.0 * len(patterns), 3.8), constrained_layout=True, sharey=True
    )
    for axis, (label, pattern) in zip(np.atleast_1d(axes), patterns, strict=True):
        axis.pcolormesh(
            pattern.deflection_deg,
            pattern.exit_angle_deg,
            np.log10(np.maximum(pattern.intensity, 10.0**-SCREEN_DECADES)),
            cmap="inferno",
            vmin=-SCREEN_DECADES,
            vmax=0.0,
            shading="nearest",
        )
        axis.axhline(0.0, color="#94a3b8", linestyle=":", linewidth=1)
        for rod in pattern.rods:
            axis.plot(rod.deflection_deg, rod.exit_angle_deg, "+", color="#94a3b8", markersize=8)
        axis.set_title(label, fontsize=9)
        axis.set_xlabel("deflection (deg)")
        axis.set_aspect("equal")
        # The shadow-edge line would otherwise stretch the axes down to zero.
        axis.set_xlim(pattern.deflection_deg[0], pattern.deflection_deg[-1])
        axis.set_ylim(pattern.exit_angle_deg[0], pattern.exit_angle_deg[-1])
    np.atleast_1d(axes)[0].set_ylabel("exit angle (deg)")
    figure.suptitle(title, fontsize=10)
    figure.savefig(path, dpi=140)
    plt.close(figure)


TOLERANCES = {
    "elastic_relative_error": 1e-12,
    "direct_sum_max_absolute_error": 5e-3,
    "uniform_lift_max_intensity_change": 1e-12,
    "mirror_symmetry_max_error": 5e-3,
    "specular_curve_vs_screen_pixel_error": 1e-9,
    "below_horizon_max_intensity": 0.0,
}


def main() -> None:
    directory = artifact_root(ROOT) / OUTPUT
    directory.mkdir(parents=True, exist_ok=True)
    surfaces = _surfaces()

    update_progress(stage="geometry", completed=0, total=4)
    geometry = _geometry_record()
    (directory / "geometry.json").write_text(json.dumps(geometry, indent=2) + "\n")

    update_progress(stage="analytic vs numeric", completed=1, total=4)
    comparison = _analytic_versus_numeric(surfaces)
    (directory / "analytic_vs_numeric.json").write_text(json.dumps(comparison, indent=2) + "\n")

    update_progress(stage="screens", completed=2, total=4)
    for name, lattice in surfaces.items():
        pattern = rheed.diffraction_screen(lattice, grazing_angle_deg=GRAZING_DEG)
        _draw(
            pattern,
            f"{name} surface - {pattern.beam_energy_kev:g} keV, "
            f"{pattern.grazing_angle_deg:.2f}° grazing, specular "
            f"{pattern.specular_intensity:.4f} of flat",
            directory / f"{name}_surface.png",
        )

    update_progress(stage="comparisons", completed=3, total=4)
    _draw_comparison(
        [
            (
                f"azimuth {azimuth:g}°",
                rheed.diffraction_screen(
                    surfaces["flat"],
                    grazing_angle_deg=GRAZING_DEG,
                    azimuth_deg=azimuth,
                    # Wide enough to follow the rods up the Laue circle as the sample turns.
                    span_deg=9.0,
                    shape=(241, 241),
                ),
            )
            for azimuth in (0.0, 10.0, 20.0, 30.0)
        ],
        "Sample azimuth rotates the reciprocal lattice against a fixed beam",
        directory / "azimuth_comparison.png",
    )
    _draw_comparison(
        [
            (label, rheed.diffraction_screen(
                surfaces["flat"], grazing_angle_deg=GRAZING_DEG, span_deg=1.0,
                shape=(121, 241), **keywords
            ))
            for label, keywords in (
                ("coherence 2 nm", {"coherence_length_nm": 2.0}),
                ("coherence 8 nm", {"coherence_length_nm": 8.0}),
                ("divergence 0.3°", {"divergence_deg": 0.3}),
                ("detector PSF 0.2°", {"detector_psf_deg": 0.2}),
            )
        ],
        "Three broadenings, each applied in its own domain",
        directory / "broadening_comparison.png",
    )

    failures = {
        key: comparison[key]
        for key, limit in TOLERANCES.items()
        if not comparison[key] <= limit
    }
    reference_error = geometry["published_reference_case"]["max_reciprocal_error_per_nm"]
    if reference_error > 1e-9:
        failures["published_reference_max_error_per_nm"] = reference_error
    widths = comparison["broadening"]
    if not (
        widths["coherence_2nm"]["measured_fwhm_deg"]
        > widths["coherence_4nm"]["measured_fwhm_deg"]
        > widths["coherence_8nm"]["measured_fwhm_deg"]
    ):
        failures["coherence_ordering"] = widths
    for name in ("divergence_0.30deg", "detector_psf_0.20deg"):
        if widths[name]["measured_fwhm_deg"] <= widths["coherence_4nm"]["measured_fwhm_deg"]:
            failures[f"{name}_did_not_broaden"] = widths[name]
    measured = [entry["measured_rod_fwhm_deg"] for entry in comparison["lattice_size_independence"].values()]
    if max(measured) - min(measured) > 0.05 * np.mean(measured):
        failures["rod_width_depends_on_box_size"] = comparison["lattice_size_independence"]
    if failures:
        raise RuntimeError(f"RHEED validation tolerances missed: {json.dumps(failures, indent=2)}")

    summary = {
        "output_directory": OUTPUT,
        "geometry": geometry,
        "checks": comparison,
        "tolerances": TOLERANCES,
        "status": "passed",
    }
    update_progress(stage="complete", completed=4, total=4)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
