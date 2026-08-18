import math

import numpy as np
import pytest

from mbe_rheed_notebook import figures
from mbe_rheed_sim import rheed
from mbe_rheed_sim.lattice import initial_lattice

GRAZING_DEG = rheed.antiphase_grazing_angle_deg(5)
SIZE = 32


def _named_traces(figure) -> dict:
    return {trace.name: trace for trace in figure.data if trace.name}


def _geometry(azimuth_deg: float = 0.0, grazing_angle_deg: float = GRAZING_DEG):
    heights = initial_lattice("island", SIZE)
    pattern = rheed.diffraction_screen(
        heights, grazing_angle_deg=grazing_angle_deg, azimuth_deg=azimuth_deg
    )
    figure = figures.rheed_geometry(
        heights,
        1.0,
        max(1, int(heights.max())),
        grazing_angle_deg=grazing_angle_deg,
        azimuth_deg=azimuth_deg,
        pattern=pattern,
    )
    return heights, pattern, figure


def test_specular_is_the_mirror_of_the_incident_beam_and_ignores_azimuth() -> None:
    straight = rheed.beam_geometry(grazing_angle_deg=GRAZING_DEG)
    turned = rheed.beam_geometry(grazing_angle_deg=GRAZING_DEG, azimuth_deg=37.0)
    for geometry in (straight, turned):
        assert np.linalg.norm(geometry.incident_direction) == pytest.approx(1.0)
        assert np.linalg.norm(geometry.specular_direction) == pytest.approx(1.0)
        # Mirror in the surface: the normal component flips, the in-plane part does not.
        assert geometry.specular_direction[:2] == pytest.approx(geometry.incident_direction[:2])
        assert geometry.specular_direction[2] == pytest.approx(-geometry.incident_direction[2])
        assert geometry.incident_direction[2] < 0.0 < geometry.specular_direction[2]
    # Turning a sample about its own normal cannot move the mirror direction.
    assert turned.specular_direction == pytest.approx(straight.specular_direction)
    assert turned.incident_direction == pytest.approx(straight.incident_direction)


def test_sample_rotation_is_a_rotation_and_carries_the_reciprocal_lattice() -> None:
    rotation = rheed.beam_geometry(grazing_angle_deg=GRAZING_DEG, azimuth_deg=23.0).sample_rotation
    assert rotation @ rotation.T == pytest.approx(np.eye(2))
    assert np.linalg.det(rotation) == pytest.approx(1.0)

    def rod_geometry(azimuth_deg: float) -> list[tuple[float, float]]:
        return sorted(
            (round(rod.exit_angle_deg, 6), round(rod.deflection_deg, 6))
            for rod in rheed.rod_orders(
                grazing_angle_deg=GRAZING_DEG, azimuth_deg=azimuth_deg, span_deg=9.0
            )
        )

    # A triangular lattice repeats every 60 degrees and mirrors at 30, and 0 and 30 are the two
    # inequivalent high-symmetry directions. This is what pins the azimuth convention.
    assert rod_geometry(0.0) == rod_geometry(60.0)
    assert rod_geometry(0.0) != rod_geometry(30.0)
    assert rod_geometry(10.0) == sorted((e, -d) for e, d in rod_geometry(50.0))


def test_detector_offsets_are_the_gnomonic_projection() -> None:
    distance = 120.0
    horizontal, vertical = rheed.detector_offsets(GRAZING_DEG, 0.0, distance)
    # The specular beam pierces the plane on the beam axis, one d tan(theta) up.
    assert float(horizontal) == pytest.approx(0.0)
    assert float(vertical) == pytest.approx(distance * math.tan(math.radians(GRAZING_DEG)))

    horizontal, vertical = rheed.detector_offsets(GRAZING_DEG, 2.0, distance)
    assert float(horizontal) == pytest.approx(distance * math.tan(math.radians(2.0)))
    assert float(vertical) == pytest.approx(
        distance * math.tan(math.radians(GRAZING_DEG)) / math.cos(math.radians(2.0))
    )
    with pytest.raises(ValueError):
        rheed.detector_offsets(1.0, 0.0, 0.0)


@pytest.mark.parametrize("azimuth_deg", [0.0, 30.0, 60.0])
def test_painted_screen_centre_is_where_the_specular_ray_lands(azimuth_deg: float) -> None:
    """The one number that makes the 3D view and the 2D screen the same picture."""
    _, _, figure = _geometry(azimuth_deg)
    traces = _named_traces(figure)
    screen, spot = traces["computed screen"], traces["specular hit point"]
    centre = (screen.z.shape[0] // 2, screen.z.shape[1] // 2)
    assert screen.y[centre] == pytest.approx(spot.y[0])
    assert screen.z[centre] == pytest.approx(spot.z[0])


@pytest.mark.parametrize("azimuth_deg", [0.0, 30.0, 60.0])
def test_azimuth_turns_the_sample_and_leaves_the_instrument_alone(azimuth_deg: float) -> None:
    heights, _, turned = _geometry(azimuth_deg)
    _, _, straight = _geometry(0.0)
    turned_traces, straight_traces = _named_traces(turned), _named_traces(straight)

    # The morphology is the same surface, only placed differently in the lab frame.
    assert np.array_equal(np.asarray(turned_traces["surface"].z), heights)

    # Beam, plane and axes are instrument, not sample: none of them may move with the azimuth.
    assert turned_traces["specular hit point"].z[0] == pytest.approx(
        straight_traces["specular hit point"].z[0]
    )
    assert turned_traces["detector plane"].x == pytest.approx(
        straight_traces["detector plane"].x
    )
    for axis in ("xaxis", "yaxis", "zaxis"):
        assert turned.layout.scene[axis].range == straight.layout.scene[axis].range


def test_the_sample_footprint_actually_rotates() -> None:
    _, _, straight = _geometry(0.0)
    _, _, turned = _geometry(30.0)
    flat = _named_traces(straight)["surface"]
    tilted = _named_traces(turned)["surface"]
    assert not np.allclose(np.asarray(flat.x), np.asarray(tilted.x))
    # Rotation preserves distances, so the bounding radius about the centre is unchanged.
    def radius(trace) -> float:
        x, y = np.asarray(trace.x), np.asarray(trace.y)
        return float(np.hypot(x - x.mean(), y - y.mean()).max())

    assert radius(flat) == pytest.approx(radius(tilted), rel=1e-9)


@pytest.mark.parametrize("order", [1, 3, 5])
def test_the_ray_follows_the_beam_condition(order: int) -> None:
    angle = rheed.antiphase_grazing_angle_deg(order)
    _, _, figure = _geometry(0.0, grazing_angle_deg=angle)
    traces = _named_traces(figure)
    beam = traces["incident beam k_i"]
    # The drawn ray rises at the true tangent of the grazing angle in data coordinates.
    run = beam.x[1] - beam.x[0]
    rise = beam.z[0] - beam.z[1]
    assert rise / run == pytest.approx(math.tan(math.radians(angle)), rel=1e-9)
    assert f"{angle:.2f}° grazing" in figure.layout.title.text


def test_geometry_without_a_pattern_paints_nothing() -> None:
    heights = initial_lattice("flat", SIZE)
    figure = figures.rheed_geometry(heights, 0.0, 1, grazing_angle_deg=GRAZING_DEG)
    assert "computed screen" not in _named_traces(figure)
    # Still a geometry view, and still says so.
    assert figures.GEOMETRY_LABEL in figure.layout.title.text
