"""Checks for the kinematic diffraction calculation.

These are the textbook consequences of the kinematic sum, so a sign error, a wrong
wavelength, or a mis-indexed transform breaks at least one of them.
"""

import numpy as np
import pytest

from mbe_rheed_sim import rheed


def _half_filled(size: int = 16, seed: int = 0) -> np.ndarray:
    lattice = np.zeros((size, size), dtype=np.int64)
    chosen = np.random.default_rng(seed).permutation(size * size)[: size * size // 2]
    lattice.ravel()[chosen] = 1
    return lattice


def test_electron_wavelength_matches_the_standard_value() -> None:
    # 15 keV electrons are about 9.97 pm; 100 keV about 3.70 pm.
    assert rheed.electron_wavelength_nm(15.0) == pytest.approx(0.00997, abs=5e-5)
    assert rheed.electron_wavelength_nm(100.0) == pytest.approx(0.00370, abs=5e-5)
    with pytest.raises(ValueError):
        rheed.electron_wavelength_nm(0.0)


@pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
def test_the_angle_that_sets_an_order_reports_that_order_back(order: int) -> None:
    angle = rheed.antiphase_grazing_angle_deg(order)
    assert rheed.phase_order(angle) == pytest.approx(float(order))
    pattern = rheed.diffraction_screen(np.zeros((8, 8), dtype=np.int64), grazing_angle_deg=angle)
    assert pattern.condition == ("anti-phase" if order % 2 else "in-phase")


def test_a_half_filled_layer_cancels_at_anti_phase_and_survives_in_phase() -> None:
    """The whole reason RHEED oscillates: odd order makes adjacent terraces interfere away."""
    lattice = _half_filled()
    anti = rheed.specular_intensity(
        lattice, grazing_angle_deg=rheed.antiphase_grazing_angle_deg(3)
    )
    in_phase = rheed.specular_intensity(
        lattice, grazing_angle_deg=rheed.antiphase_grazing_angle_deg(4)
    )
    assert anti < 0.02
    assert in_phase == pytest.approx(1.0)


def test_a_flat_surface_is_the_brightest_the_specular_beam_ever_gets() -> None:
    angle = rheed.antiphase_grazing_angle_deg(3)
    pattern = rheed.diffraction_screen(np.zeros((16, 16), dtype=np.int64), grazing_angle_deg=angle)
    assert pattern.specular_intensity == pytest.approx(1.0)
    assert pattern.intensity.max() == pytest.approx(1.0)
    # Height offsets are a phase common to every column, so they cannot change any intensity.
    shifted = rheed.diffraction_screen(
        np.full((16, 16), 4, dtype=np.int64), grazing_angle_deg=angle
    )
    assert shifted.intensity == pytest.approx(pattern.intensity)


def test_the_screen_is_centred_on_the_specular_beam_and_shadows_the_substrate() -> None:
    angle = rheed.antiphase_grazing_angle_deg(3)
    pattern = rheed.diffraction_screen(
        _half_filled(), grazing_angle_deg=angle, span_deg=3.0, shape=(60, 80)
    )
    rows, columns = pattern.intensity.shape
    assert rows % 2 == 1 and columns % 2 == 1
    assert pattern.exit_angle_deg[rows // 2] == pytest.approx(angle)
    assert pattern.deflection_deg[columns // 2] == pytest.approx(0.0)
    below_horizon = pattern.exit_angle_deg < 0.0
    assert below_horizon.any()
    assert not pattern.intensity[below_horizon].any()


def test_the_screen_specular_pixel_is_the_curve_it_is_plotted_against() -> None:
    """The overlaid trace and the marked pixel must be one number, not two calculations."""
    angle = rheed.antiphase_grazing_angle_deg(3)
    stack = np.stack([_half_filled(seed=seed) for seed in range(3)])
    curve = rheed.specular_intensity(stack, grazing_angle_deg=angle)
    assert curve.shape == (3,)
    for index, lattice in enumerate(stack):
        pattern = rheed.diffraction_screen(lattice, grazing_angle_deg=angle)
        assert pattern.specular_intensity == pytest.approx(curve[index])


def test_streaks_land_on_the_hexagonal_reciprocal_lattice() -> None:
    """A flat surface diffracts only into rods; they must sit where `a` puts them."""
    angle = rheed.antiphase_grazing_angle_deg(3)
    pattern = rheed.diffraction_screen(
        np.zeros((16, 16), dtype=np.int64), grazing_angle_deg=angle
    )
    specular_row = pattern.intensity[len(pattern.exit_angle_deg) // 2]
    bright = pattern.deflection_deg[specular_row > 0.5 * specular_row.max()]
    assert max(bright) == pytest.approx(pattern.rod_spacing_deg, abs=0.1)
    assert min(bright) == pytest.approx(-pattern.rod_spacing_deg, abs=0.1)
    # Between the rods the screen must be dark, not a filled texture.
    between = abs(pattern.deflection_deg % pattern.rod_spacing_deg - 0.5 * pattern.rod_spacing_deg)
    assert specular_row[between < 0.3].max() < 1e-3


@pytest.mark.parametrize("size", [7, 16, 33])
def test_streak_width_follows_the_beam_not_the_simulation_box(size: int) -> None:
    """The regression this guards: rod width must be an instrument property.

    Windowing the raw lattice made a 7x7 run produce degree-wide bands and a 33x33 run
    produce narrow ones from identical physics, which says nothing about the surface.
    """
    angle = rheed.antiphase_grazing_angle_deg(3)
    pattern = rheed.diffraction_screen(
        np.zeros((size, size), dtype=np.int64), grazing_angle_deg=angle, transfer_width_nm=4.0
    )
    specular_row = pattern.intensity[len(pattern.exit_angle_deg) // 2]
    above_half = pattern.deflection_deg[specular_row > 0.5 * specular_row.max()]
    central = above_half[abs(above_half) < 0.5 * pattern.rod_spacing_deg]
    measured = central.max() - central.min()
    step = pattern.deflection_deg[1] - pattern.deflection_deg[0]
    assert measured == pytest.approx(pattern.streak_width_deg, abs=2 * step)

    # Halving the transfer width must double the streak width; the lattice is untouched.
    wider = rheed.diffraction_screen(
        np.zeros((size, size), dtype=np.int64), grazing_angle_deg=angle, transfer_width_nm=2.0
    )
    assert wider.streak_width_deg == pytest.approx(2 * pattern.streak_width_deg)


def _off_rod_background(pattern: rheed.ScreenPattern) -> float:
    """Mean lit-screen intensity away from every diffraction rod."""
    to_nearest_rod = abs(
        (pattern.deflection_deg + 0.5 * pattern.rod_spacing_deg) % pattern.rod_spacing_deg
        - 0.5 * pattern.rod_spacing_deg
    )
    lit = pattern.intensity[pattern.exit_angle_deg > 0.0]
    return float(lit[:, to_nearest_rod > 0.3].mean())


def test_roughening_fills_the_gaps_between_the_rods() -> None:
    """The signature the screen exists to show: streaks on black, then diffuse everywhere."""
    angle = rheed.antiphase_grazing_angle_deg(3)
    flat = rheed.diffraction_screen(np.zeros((16, 16), dtype=np.int64), grazing_angle_deg=angle)
    rough = rheed.diffraction_screen(_half_filled(), grazing_angle_deg=angle)
    assert rough.specular_intensity < 0.05 * flat.specular_intensity
    assert _off_rod_background(flat) < 1e-5
    assert _off_rod_background(rough) > 100 * _off_rod_background(flat)


def test_invalid_geometry_is_refused() -> None:
    lattice = np.zeros((8, 8), dtype=np.int64)
    angle = rheed.antiphase_grazing_angle_deg(3)
    with pytest.raises(ValueError):
        rheed.diffraction_screen(lattice[0], grazing_angle_deg=angle)
    with pytest.raises(ValueError):
        rheed.diffraction_screen(lattice, grazing_angle_deg=angle, span_deg=0.0)
    with pytest.raises(ValueError):
        rheed.diffraction_screen(lattice, grazing_angle_deg=0.0)
    with pytest.raises(ValueError):
        rheed.antiphase_grazing_angle_deg(0)
    with pytest.raises(ValueError):
        rheed.antiphase_grazing_angle_deg(400)
