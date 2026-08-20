"""Checks for the kinematic diffraction calculation.

These are the textbook consequences of the kinematic sum and of the Ewald construction, so a
sign error, a wrong wavelength, a mis-indexed transform or a broken reciprocal basis breaks
at least one of them. Nothing here snapshots pixels; every assertion is a physical statement
that can be derived on paper first.
"""

import math

import numpy as np
import pytest

from mbe_rheed_sim import rheed

ANGLE = rheed.antiphase_grazing_angle_deg(3)
# High enough that the first-order rods clear the Ewald sphere; see `test_first_order_rods_*`.
OPEN_ANGLE = rheed.antiphase_grazing_angle_deg(5)
# Integrated lit-screen intensity of the reference rough surface, from the finest grid below.
RESOLVED_INTEGRAL = 0.0631


def _half_filled(size: int = 16, seed: int = 0) -> np.ndarray:
    lattice = np.zeros((size, size), dtype=np.int64)
    chosen = np.random.default_rng(seed).permutation(size * size)[: size * size // 2]
    lattice.ravel()[chosen] = 1
    return lattice


def _profile_fwhm(axis: np.ndarray, values: np.ndarray) -> float:
    """Full width at half maximum of a single peak, linearly interpolated between samples."""
    peak = int(np.argmax(values))
    half = 0.5 * values[peak]
    edges = []
    for direction in (-1, 1):
        index = peak
        while 0 < index < len(values) - 1 and values[index] > half:
            index += direction
        span = values[index]
        previous = values[index - direction]
        weight = (previous - half) / (previous - span) if previous != span else 0.0
        edges.append(axis[index - direction] + weight * (axis[index] - axis[index - direction]))
    return abs(edges[1] - edges[0])


def _specular_row(pattern: rheed.ScreenPattern) -> np.ndarray:
    return pattern.intensity[len(pattern.exit_angle_deg) // 2]


def _off_rod_mask(pattern: rheed.ScreenPattern, margins: float = 6.0) -> np.ndarray:
    """Lit pixels whose in-plane momentum transfer is far from every reciprocal rod.

    Distance is measured in reciprocal space, not in screen angles: a rod is a line in q_z,
    so the Ewald sphere can run close to it over a wide band of exit angles and the intensity
    there is rod, not background.
    """
    wavenumber = rheed.wavenumber_per_nm(pattern.beam_energy_kev)
    incidence = math.radians(pattern.grazing_angle_deg)
    exit_angle = np.radians(pattern.exit_angle_deg)[:, None]
    deflection = np.radians(pattern.deflection_deg)[None, :]
    q_x = wavenumber * (np.cos(exit_angle) * np.cos(deflection) - math.cos(incidence))
    q_y = wavenumber * np.cos(exit_angle) * np.sin(deflection)
    rotation = rheed._rotation(pattern.azimuth_deg)
    sample_x = rotation[0, 0] * q_x + rotation[1, 0] * q_y
    sample_y = rotation[0, 1] * q_x + rotation[1, 1] * q_y
    reciprocal = rheed.reciprocal_basis()
    rod_width = 4.0 * math.sqrt(2.0) * math.log(2.0) / pattern.coherence_length_nm
    near = np.zeros(np.broadcast(sample_x, sample_y).shape, dtype=bool)
    for h in range(-3, 4):
        for k in range(-3, 4):
            vector = h * reciprocal[0] + k * reciprocal[1]
            near |= np.hypot(sample_x - vector[0], sample_y - vector[1]) < margins * rod_width
    return (~near) & (pattern.exit_angle_deg[:, None] > 0.0)


# --------------------------------------------------------------------------- beam and units


def test_electron_wavelength_matches_the_standard_value() -> None:
    # 15 keV electrons are about 9.94 pm; 100 keV about 3.70 pm.
    assert rheed.electron_wavelength_nm(15.0) == pytest.approx(0.00994, abs=5e-5)
    assert rheed.electron_wavelength_nm(100.0) == pytest.approx(0.00370, abs=5e-5)
    assert rheed.wavenumber_per_nm(15.0) == pytest.approx(
        2.0 * math.pi / rheed.electron_wavelength_nm(15.0)
    )
    with pytest.raises(ValueError):
        rheed.electron_wavelength_nm(0.0)


def test_the_incident_wavevector_points_along_the_beam_and_downwards() -> None:
    incident = rheed.incident_wavevector(ANGLE)
    wavenumber = rheed.wavenumber_per_nm()
    assert np.linalg.norm(incident) == pytest.approx(wavenumber)
    assert incident[0] > 0 and incident[1] == 0.0 and incident[2] < 0
    assert math.degrees(math.asin(-incident[2] / wavenumber)) == pytest.approx(ANGLE)


@pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
def test_the_angle_that_sets_an_order_reports_that_order_back(order: int) -> None:
    angle = rheed.antiphase_grazing_angle_deg(order)
    assert rheed.phase_order(angle) == pytest.approx(float(order))
    pattern = rheed.diffraction_screen(np.zeros((8, 8), dtype=np.int64), grazing_angle_deg=angle)
    assert pattern.condition == ("anti-phase" if order % 2 else "in-phase")


# ------------------------------------------------------------------- lattice and reciprocal


def test_the_reciprocal_basis_is_dual_to_the_real_one() -> None:
    """`a_i . b_j = 2 pi delta_ij` is the definition; nothing else in here is meaningful."""
    for spacing in (0.2, rheed.GAN_IN_PLANE_SPACING_NM, 0.5):
        real = rheed.surface_basis(spacing)
        reciprocal = rheed.reciprocal_basis(spacing)
        assert real @ reciprocal.T == pytest.approx(2.0 * math.pi * np.eye(2))
        # Both reciprocal vectors of a triangular lattice have length 4 pi / (a sqrt 3).
        expected = 4.0 * math.pi / (spacing * math.sqrt(3.0))
        assert np.linalg.norm(reciprocal, axis=1) == pytest.approx([expected, expected])
    with pytest.raises(ValueError):
        rheed.surface_basis(0.0)


def test_the_reciprocal_basis_matches_the_positions_the_sum_actually_uses() -> None:
    """A reciprocal vector must give every scatterer a phase of a whole 2 pi."""
    spacing = rheed.GAN_IN_PLANE_SPACING_NM
    real = rheed.surface_basis(spacing)
    reciprocal = rheed.reciprocal_basis(spacing)
    for row in range(-3, 4):
        for column in range(-3, 4):
            position = row * real[1] + column * real[0]
            for h in (-2, 1, 3):
                for k in (-1, 2):
                    phase = (h * reciprocal[0] + k * reciprocal[1]) @ position
                    assert phase / (2.0 * math.pi) == pytest.approx(round(phase / (2 * math.pi)))


# --------------------------------------------------------------------------- Ewald geometry


def test_every_screen_direction_is_elastic() -> None:
    """`|k_f| = |k_i|` for every pixel, which is what puts the screen on the Ewald sphere."""
    pattern = rheed.diffraction_screen(
        np.zeros((8, 8), dtype=np.int64), grazing_angle_deg=ANGLE, shape=(41, 51)
    )
    wavenumber = rheed.wavenumber_per_nm(pattern.beam_energy_kev)
    exit_angle = np.radians(pattern.exit_angle_deg)[:, None]
    deflection = np.radians(pattern.deflection_deg)[None, :]
    final = np.stack(
        np.broadcast_arrays(
            wavenumber * np.cos(exit_angle) * np.cos(deflection),
            wavenumber * np.cos(exit_angle) * np.sin(deflection),
            wavenumber * np.sin(exit_angle) + 0.0 * deflection,
        )
    )
    assert np.abs(np.linalg.norm(final, axis=0) - wavenumber).max() < 1e-9 * wavenumber


def test_rod_positions_match_the_analytic_ewald_construction() -> None:
    """Solve `|k_i,par + G| <= k` by hand and demand the same angles back."""
    spacing = rheed.GAN_IN_PLANE_SPACING_NM
    wavenumber = rheed.wavenumber_per_nm()
    incident_parallel = wavenumber * math.cos(math.radians(OPEN_ANGLE))
    reciprocal = rheed.reciprocal_basis(spacing)
    orders = {(rod.h, rod.k): rod for rod in rheed.rod_orders(grazing_angle_deg=OPEN_ANGLE)}
    for (h, k), rod in orders.items():
        vector = h * reciprocal[0] + k * reciprocal[1]
        parallel = np.array([incident_parallel + vector[0], vector[1]])
        vertical = math.sqrt(wavenumber**2 - float(parallel @ parallel))
        assert rod.exit_angle_deg == pytest.approx(
            math.degrees(math.asin(vertical / wavenumber))
        )
        assert rod.deflection_deg == pytest.approx(
            math.degrees(math.atan2(parallel[1], parallel[0]))
        )
    assert (0, 0) in orders
    assert orders[(0, 0)].exit_angle_deg == pytest.approx(OPEN_ANGLE)
    assert orders[(0, 0)].deflection_deg == pytest.approx(0.0)


def test_first_order_rods_open_only_above_their_critical_grazing_angle() -> None:
    """A rod exists when `|G| <= k sin(incidence)`; below that it is a geometric absence."""
    spacing = rheed.GAN_IN_PLANE_SPACING_NM
    wavenumber = rheed.wavenumber_per_nm()
    length = 4.0 * math.pi / (spacing * math.sqrt(3.0))
    critical = math.degrees(math.asin(length / wavenumber))
    assert ANGLE < critical < OPEN_ANGLE

    closed = {
        (rod.h, rod.k)
        for rod in rheed.rod_orders(grazing_angle_deg=critical - 0.1, span_deg=3.0)
    }
    opened = {
        (rod.h, rod.k)
        for rod in rheed.rod_orders(grazing_angle_deg=critical + 0.1, span_deg=3.0)
    }
    assert closed == {(0, 0)}
    assert {(0, 1), (0, -1)} <= opened


def test_the_accessible_rods_show_up_as_maxima_on_the_screen() -> None:
    """The detector must contain the orders geometry predicts, at the predicted place."""
    pattern = rheed.diffraction_screen(
        np.zeros((16, 16), dtype=np.int64), grazing_angle_deg=OPEN_ANGLE
    )
    side = [rod for rod in pattern.rods if (rod.h, rod.k) == (0, 1)]
    assert side, "the first-order rod should be inside this screen"
    rod = side[0]
    row = np.argmin(abs(pattern.exit_angle_deg - rod.exit_angle_deg))
    column = np.argmin(abs(pattern.deflection_deg - rod.deflection_deg))
    patch = pattern.intensity[row - 3 : row + 4, column - 3 : column + 4]
    assert pattern.intensity[row, column] == pytest.approx(patch.max(), rel=0.05)
    assert pattern.intensity[row, column] > 0.5


def test_the_screen_is_centred_on_the_specular_beam_and_shadows_the_substrate() -> None:
    pattern = rheed.diffraction_screen(
        _half_filled(), grazing_angle_deg=ANGLE, span_deg=3.0, shape=(60, 80)
    )
    rows, columns = pattern.intensity.shape
    assert rows % 2 == 1 and columns % 2 == 1
    assert pattern.exit_angle_deg[rows // 2] == pytest.approx(ANGLE)
    assert pattern.deflection_deg[columns // 2] == pytest.approx(0.0)
    below_horizon = pattern.exit_angle_deg < 0.0
    assert below_horizon.any()
    assert not pattern.intensity[below_horizon].any()


def test_the_horizon_stays_dark_even_after_the_detector_blurs_the_edge() -> None:
    pattern = rheed.diffraction_screen(
        np.zeros((16, 16), dtype=np.int64),
        grazing_angle_deg=ANGLE,
        detector_psf_deg=0.4,
        span_deg=3.0,
    )
    assert not pattern.intensity[pattern.exit_angle_deg < 0.0].any()


@pytest.mark.parametrize("azimuth", [0.0, 30.0, 60.0])
def test_a_mirror_azimuth_gives_a_mirror_symmetric_screen(azimuth: float) -> None:
    """The triangular lattice has mirror planes every 30 degrees; the screen must show it."""
    pattern = rheed.diffraction_screen(
        np.zeros((16, 16), dtype=np.int64), grazing_angle_deg=OPEN_ANGLE, azimuth_deg=azimuth
    )
    assert np.abs(pattern.intensity - pattern.intensity[:, ::-1]).max() < 5e-3


def test_azimuth_rotates_the_reciprocal_lattice_against_the_beam() -> None:
    """Turning the sample must move the rods, and by the rotation actually asked for."""
    straight = rheed.rod_orders(grazing_angle_deg=OPEN_ANGLE, azimuth_deg=0.0)
    turned = rheed.rod_orders(grazing_angle_deg=OPEN_ANGLE, azimuth_deg=30.0)
    assert {(rod.h, rod.k) for rod in straight} != {(rod.h, rod.k) for rod in turned}

    # A full 60 degrees is a symmetry of the triangular lattice, so the pattern returns.
    reciprocal = rheed.reciprocal_basis()
    for azimuth in (17.0, 45.0):
        rotation = rheed._rotation(azimuth)
        wavenumber = rheed.wavenumber_per_nm()
        for rod in rheed.rod_orders(grazing_angle_deg=OPEN_ANGLE, azimuth_deg=azimuth):
            vector = rotation @ (rod.h * reciprocal[0] + rod.k * reciprocal[1])
            deflection = math.degrees(
                math.atan2(vector[1], wavenumber * math.cos(math.radians(OPEN_ANGLE)) + vector[0])
            )
            assert rod.deflection_deg == pytest.approx(deflection)

    sixty = rheed.diffraction_screen(
        np.zeros((12, 12), dtype=np.int64), grazing_angle_deg=OPEN_ANGLE, azimuth_deg=60.0
    )
    zero = rheed.diffraction_screen(
        np.zeros((12, 12), dtype=np.int64), grazing_angle_deg=OPEN_ANGLE, azimuth_deg=0.0
    )
    assert np.abs(sixty.intensity - zero.intensity).max() < 5e-3


# ------------------------------------------------------------------------ the kinematic sum


def test_the_transform_path_reproduces_the_direct_kinematic_sum() -> None:
    """The interpolated transform is an optimization; the sum over scatterers is the physics."""
    spacing, height = rheed.GAN_IN_PLANE_SPACING_NM, rheed.GAN_LAYER_HEIGHT_NM
    wavenumber = rheed.wavenumber_per_nm()
    lattice = _half_filled(16, seed=3)
    pattern = rheed.diffraction_screen(
        lattice, grazing_angle_deg=OPEN_ANGLE, coherence_patches=1
    )
    windows, profile = rheed._illuminated_patches(
        lattice, rheed.DEFAULT_COHERENCE_LENGTH_NM, spacing, 1
    )
    window = windows[0]
    half = len(profile) // 2
    row, column = np.indices(window.shape)
    x = spacing * (column - half + 0.5 * (row - half))
    y = spacing * (math.sqrt(3.0) / 2.0) * (row - half)

    incidence = math.radians(OPEN_ANGLE)
    worst = 0.0
    for row_index in range(4, len(pattern.exit_angle_deg), 23):
        for column_index in range(2, len(pattern.deflection_deg), 29):
            exit_angle = math.radians(pattern.exit_angle_deg[row_index])
            deflection = math.radians(pattern.deflection_deg[column_index])
            if exit_angle < 0:
                continue
            q_x = wavenumber * (math.cos(exit_angle) * math.cos(deflection) - math.cos(incidence))
            q_y = wavenumber * math.cos(exit_angle) * math.sin(deflection)
            q_z = wavenumber * (math.sin(exit_angle) + math.sin(incidence))
            phase = q_x * x + q_y * y + q_z * height * window
            direct = abs(np.sum(profile * np.exp(-1j * phase)) / profile.sum()) ** 2
            worst = max(worst, abs(direct - pattern.intensity[row_index, column_index]))
    assert worst < 5e-3


def test_a_half_filled_layer_cancels_at_anti_phase_and_survives_in_phase() -> None:
    """The whole reason RHEED oscillates: odd order makes adjacent terraces interfere away."""
    lattice = _half_filled()
    anti = rheed.specular_intensity(lattice, grazing_angle_deg=ANGLE)
    in_phase = rheed.specular_intensity(
        lattice, grazing_angle_deg=rheed.antiphase_grazing_angle_deg(4)
    )
    assert anti < 0.02
    assert in_phase == pytest.approx(1.0)


@pytest.mark.parametrize("coverage", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_the_two_level_specular_intensity_follows_the_interference_law(coverage: float) -> None:
    """`I_00 = (1 - 2 theta)^2`, the closed form of the two-terrace interference term.

    On the specular rod `q` is purely vertical, so the in-plane phases cancel and a surface
    with only heights 0 and 1 has amplitude `(1 - theta) + theta exp(-i q_z d)`. At anti-phase
    that is `1 - 2 theta`, which is the whole oscillation, derived without any morphology
    argument. The tolerance covers the finite patch: the Gaussian illumination weights sites,
    so a random arrangement realizes the nominal coverage only to within its own sampling.
    """
    size = 64
    lattice = np.zeros((size, size), dtype=np.int64)
    filled = round(coverage * size * size)
    lattice.ravel()[np.random.default_rng(3).permutation(size * size)[:filled]] = 1
    intensity = float(rheed.specular_intensity(lattice, grazing_angle_deg=ANGLE))
    assert intensity == pytest.approx((1.0 - 2.0 * coverage) ** 2, abs=0.01)


def test_a_flat_surface_is_the_brightest_the_specular_beam_ever_gets() -> None:
    pattern = rheed.diffraction_screen(
        np.zeros((16, 16), dtype=np.int64), grazing_angle_deg=ANGLE
    )
    assert pattern.specular_intensity == pytest.approx(1.0)
    assert pattern.intensity.max() == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("offset", [1, 4, 9])
def test_raising_the_whole_surface_is_a_global_phase_and_changes_no_intensity(
    offset: int,
) -> None:
    """Translation only multiplies the amplitude by `exp(-i q_z d n)`; `|A|^2` cannot move."""
    lattice = _half_filled(16, seed=5)
    reference = rheed.diffraction_screen(lattice, grazing_angle_deg=OPEN_ANGLE)
    lifted = rheed.diffraction_screen(lattice + offset, grazing_angle_deg=OPEN_ANGLE)
    assert lifted.intensity == pytest.approx(reference.intensity, abs=1e-12)
    assert rheed.specular_intensity(
        lattice + offset, grazing_angle_deg=OPEN_ANGLE
    ) == pytest.approx(rheed.specular_intensity(lattice, grazing_angle_deg=OPEN_ANGLE))


def test_the_screen_specular_pixel_is_the_curve_it_is_plotted_against() -> None:
    """The overlaid trace and the marked pixel must be one number, not two calculations."""
    stack = np.stack([_half_filled(seed=seed) for seed in range(3)])
    curve = rheed.specular_intensity(stack, grazing_angle_deg=ANGLE)
    assert curve.shape == (3,)
    for index, lattice in enumerate(stack):
        pattern = rheed.diffraction_screen(lattice, grazing_angle_deg=ANGLE)
        assert pattern.specular_intensity == pytest.approx(curve[index], abs=1e-9)


def test_the_screen_tracks_the_morphology_from_flat_through_stepped_to_three_dimensional(
) -> None:
    """Flat -> narrow rods on black; partial layer -> diffuse; 3D -> specular nearly gone."""

    def partial_layer(coverage: float, size: int = 16, seed: int = 0) -> np.ndarray:
        lattice = np.zeros((size, size), dtype=np.int64)
        count = int(coverage * size * size)
        lattice.ravel()[np.random.default_rng(seed).permutation(size * size)[:count]] = 1
        return lattice

    patterns = [
        rheed.diffraction_screen(lattice, grazing_angle_deg=OPEN_ANGLE)
        for lattice in (
            np.zeros((16, 16), dtype=np.int64),
            partial_layer(0.35),
            np.random.default_rng(7).integers(0, 6, (16, 16)).astype(np.int64),
        )
    ]
    specular = [pattern.specular_intensity for pattern in patterns]
    background = [float(p.intensity[_off_rod_mask(p)].mean()) for p in patterns]

    assert specular[0] == pytest.approx(1.0)
    assert specular[0] > specular[1] > specular[2]
    assert specular[2] < 1e-3
    assert background[0] < 1e-6
    assert background[0] < background[1] < background[2]


# ------------------------------------------------------------------------- structure factor


def test_a_two_atom_basis_extinguishes_the_orders_it_should() -> None:
    """`F = 1 + exp(-i G . r)` vanishes wherever `G . r` is an odd multiple of pi."""
    spacing = rheed.GAN_IN_PLANE_SPACING_NM
    real = rheed.surface_basis(spacing)
    # Half of a1: rods with odd h then get exactly opposite contributions.
    basis = ((real[0, 0] / 2, real[0, 1] / 2, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0))
    orders = {
        (rod.h, rod.k): rod.relative_intensity
        for rod in rheed.rod_orders(grazing_angle_deg=OPEN_ANGLE, basis=basis, max_order=2)
    }
    assert orders[(0, 0)] == pytest.approx(1.0)
    for (h, _), value in orders.items():
        assert value == pytest.approx(0.0 if h % 2 else 1.0, abs=1e-9)

    plain = rheed.rod_orders(grazing_angle_deg=OPEN_ANGLE)
    assert all(rod.relative_intensity == pytest.approx(1.0) for rod in plain)


def test_the_default_scattering_model_is_the_effective_single_scatterer() -> None:
    lattice = _half_filled()
    without = rheed.diffraction_screen(lattice, grazing_angle_deg=OPEN_ANGLE)
    trivial = rheed.diffraction_screen(
        lattice, grazing_angle_deg=OPEN_ANGLE, basis=((0.0, 0.0, 0.0, 1.0),)
    )
    assert trivial.intensity == pytest.approx(without.intensity)


# ------------------------------------------------------------------------------- broadening


def test_a_longer_coherence_length_narrows_the_rod_and_nothing_else_does() -> None:
    widths = {}
    for coherence in (2.0, 4.0, 8.0):
        pattern = rheed.diffraction_screen(
            np.zeros((16, 16), dtype=np.int64),
            grazing_angle_deg=ANGLE,
            coherence_length_nm=coherence,
            span_deg=1.0,
            shape=(41, 401),
        )
        widths[coherence] = _profile_fwhm(pattern.deflection_deg, _specular_row(pattern))
        # The analytic Gaussian rod width, 4 sqrt(2) ln2 / L, in deflection.
        assert widths[coherence] == pytest.approx(pattern.streak_width_deg, rel=0.05)
    assert widths[2.0] > widths[4.0] > widths[8.0]
    assert widths[2.0] == pytest.approx(2.0 * widths[4.0], rel=0.05)


def test_beam_divergence_broadens_the_detector_feature_without_touching_the_lattice() -> None:
    sharp, diverged = (
        rheed.diffraction_screen(
            np.zeros((16, 16), dtype=np.int64),
            grazing_angle_deg=ANGLE,
            divergence_deg=divergence,
            span_deg=1.0,
            shape=(41, 401),
        )
        for divergence in (0.0, 0.3)
    )
    assert _profile_fwhm(diverged.deflection_deg, _specular_row(diverged)) > 1.5 * _profile_fwhm(
        sharp.deflection_deg, _specular_row(sharp)
    )
    assert diverged.specular_intensity < sharp.specular_intensity


def test_detector_point_spread_broadens_the_detector_feature() -> None:
    sharp, blurred = (
        rheed.diffraction_screen(
            np.zeros((16, 16), dtype=np.int64),
            grazing_angle_deg=ANGLE,
            detector_psf_deg=psf,
            span_deg=1.0,
            shape=(41, 401),
        )
        for psf in (0.0, 0.2)
    )
    assert _profile_fwhm(blurred.deflection_deg, _specular_row(blurred)) > 1.5 * _profile_fwhm(
        sharp.deflection_deg, _specular_row(sharp)
    )


def test_the_three_broadenings_are_independent_knobs() -> None:
    """Each one must move the screen on its own; none may be a rename of another."""
    base = {"grazing_angle_deg": ANGLE, "span_deg": 1.0, "shape": (41, 401)}
    lattice = np.zeros((16, 16), dtype=np.int64)
    reference = rheed.diffraction_screen(lattice, **base)
    for keyword, value in (
        ("coherence_length_nm", 8.0),
        ("divergence_deg", 0.2),
        ("detector_psf_deg", 0.1),
    ):
        changed = rheed.diffraction_screen(lattice, **base, **{keyword: value})
        assert not np.allclose(changed.intensity, reference.intensity)


# ------------------------------------------------------------------- numerical independence


@pytest.mark.parametrize("size", [16, 32, 64, 128])
def test_the_screen_follows_the_surface_not_the_simulation_box(size: int) -> None:
    """The same physical surface tiled into a bigger box must diffract identically."""
    motif = np.array(
        [[0, 0, 1, 1, 1, 1, 0, 0][((row // 2) + column) % 8] for row in range(8) for column in range(8)],
        dtype=np.int64,
    ).reshape(8, 8)
    lattice = np.tile(motif, (size // 8, size // 8))
    reference = rheed.diffraction_screen(
        np.tile(motif, (2, 2)), grazing_angle_deg=OPEN_ANGLE, coherence_patches=1
    )
    pattern = rheed.diffraction_screen(
        lattice, grazing_angle_deg=OPEN_ANGLE, coherence_patches=1
    )
    assert pattern.intensity == pytest.approx(reference.intensity, abs=1e-12)
    assert pattern.streak_width_deg == pytest.approx(reference.streak_width_deg)


@pytest.mark.parametrize("size", [16, 32, 64, 128])
def test_rod_width_and_position_do_not_depend_on_the_box_size(size: int) -> None:
    flat = np.zeros((size, size), dtype=np.int64)
    pattern = rheed.diffraction_screen(
        flat, grazing_angle_deg=ANGLE, span_deg=1.0, shape=(41, 401)
    )
    row = _specular_row(pattern)
    assert pattern.deflection_deg[int(np.argmax(row))] == pytest.approx(0.0, abs=1e-9)
    assert _profile_fwhm(pattern.deflection_deg, row) == pytest.approx(
        pattern.streak_width_deg, rel=0.05
    )


@pytest.mark.parametrize("shape", [(91, 201), (181, 401), (361, 801)])
def test_finer_detector_sampling_converges_on_the_same_physics(shape) -> None:
    """Peak position, width and integrated intensity must be pixel-grid independent."""
    flat = rheed.diffraction_screen(
        np.zeros((16, 16), dtype=np.int64),
        grazing_angle_deg=OPEN_ANGLE,
        span_deg=1.0,
        shape=shape,
    )
    row = _specular_row(flat)
    assert flat.deflection_deg[int(np.argmax(row))] == pytest.approx(0.0, abs=1e-9)
    assert flat.specular_intensity == pytest.approx(1.0, abs=1e-6)
    assert _profile_fwhm(flat.deflection_deg, row) == pytest.approx(
        flat.streak_width_deg, rel=0.08
    )

    rough = rheed.diffraction_screen(
        _half_filled(16, seed=2), grazing_angle_deg=OPEN_ANGLE, span_deg=1.0, shape=shape
    )
    lit = rough.exit_angle_deg > 0.0
    step_x = rough.deflection_deg[1] - rough.deflection_deg[0]
    step_y = rough.exit_angle_deg[1] - rough.exit_angle_deg[0]
    integrated = float(rough.intensity[lit].sum()) * step_x * step_y
    # Pinned from the finest grid; the Riemann sum must not drift with resolution.
    assert integrated == pytest.approx(RESOLVED_INTEGRAL, rel=0.02)


def test_the_finite_box_satellite_warning_tracks_the_coherence_length() -> None:
    """Below about 1 the diffuse background is continuous; above it, it breaks into spots."""
    assert rheed.satellite_artifact_ratio(128, coherence_length_nm=4.0) < 0.2
    assert rheed.satellite_artifact_ratio(8, coherence_length_nm=4.0) > 1.0
    assert rheed.satellite_artifact_ratio(32, coherence_length_nm=40.0) > rheed.satellite_artifact_ratio(
        32, coherence_length_nm=4.0
    )


# ---------------------------------------------------------------------------- input guards


def test_invalid_geometry_is_refused() -> None:
    lattice = np.zeros((8, 8), dtype=np.int64)
    with pytest.raises(ValueError):
        rheed.diffraction_screen(lattice[0], grazing_angle_deg=ANGLE)
    with pytest.raises(ValueError):
        rheed.diffraction_screen(lattice, grazing_angle_deg=ANGLE, span_deg=0.0)
    with pytest.raises(ValueError):
        rheed.diffraction_screen(lattice, grazing_angle_deg=0.0)
    with pytest.raises(ValueError):
        rheed.diffraction_screen(lattice, grazing_angle_deg=ANGLE, divergence_deg=-1.0)
    with pytest.raises(ValueError):
        rheed.diffraction_screen(lattice, grazing_angle_deg=ANGLE, detector_psf_deg=-1.0)
    with pytest.raises(ValueError):
        rheed.diffraction_screen(lattice, grazing_angle_deg=ANGLE, coherence_length_nm=0.0)
    with pytest.raises(ValueError):
        rheed.antiphase_grazing_angle_deg(0)
    with pytest.raises(ValueError):
        rheed.antiphase_grazing_angle_deg(400)
