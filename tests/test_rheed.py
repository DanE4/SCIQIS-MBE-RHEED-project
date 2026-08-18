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
    """A flat surface diffracts only into rods; the first pair must sit where `a` puts them."""
    angle = rheed.antiphase_grazing_angle_deg(3)
    pattern = rheed.diffraction_screen(
        np.zeros((16, 16), dtype=np.int64), grazing_angle_deg=angle
    )
    specular_row = pattern.intensity[len(pattern.exit_angle_deg) // 2]
    bright = pattern.deflection_deg[specular_row > 0.2 * specular_row.max()]
    # Rods repeat every 4*pi / (a*sqrt(3)) in q_y, which at this wavelength is this deflection.
    spacing = 4.0 * np.pi / (rheed.GAN_IN_PLANE_SPACING_NM * np.sqrt(3.0))
    expected = np.degrees(spacing / (2.0 * np.pi / rheed.electron_wavelength_nm(15.0)))
    assert max(bright) == pytest.approx(expected, abs=0.15)
    assert min(bright) == pytest.approx(-expected, abs=0.15)


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
