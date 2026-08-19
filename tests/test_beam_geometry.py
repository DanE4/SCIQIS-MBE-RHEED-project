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


def _orders(azimuth_deg: float = 0.0, grazing_angle_deg: float = GRAZING_DEG):
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
        show_orders=True,
    )
    return heights, pattern, figure


def _order_labels(figure) -> dict[str, tuple[float, float]]:
    """Drawn (h,k) label positions, keyed by label, from the orders trace."""
    trace = next(t for t in figure.data if t.name and t.name.startswith("(hk)"))
    return {
        label: (float(y), float(z))
        for label, y, z in zip(trace.text, trace.y, trace.z, strict=True)
    }


@pytest.mark.parametrize("azimuth_deg", [0.0, 10.0, 25.0, 30.0])
def test_drawn_orders_are_exactly_what_rod_orders_returns(azimuth_deg: float) -> None:
    _, _, figure = _orders(azimuth_deg)
    expected = {
        rod.label
        for rod in rheed.rod_orders(
            grazing_angle_deg=GRAZING_DEG,
            azimuth_deg=azimuth_deg,
            span_deg=figures.ORDERS_ACCEPTANCE_DEG,
        )
        if (rod.h, rod.k) != (0, 0)
    }
    assert _order_labels(figure).keys() == expected
    # Nothing is drawn that the Ewald construction did not return, in either direction.
    assert expected


@pytest.mark.parametrize("azimuth_deg", [0.0, 10.0, 25.0])
def test_each_order_ray_ends_on_its_own_ewald_intersection(azimuth_deg: float) -> None:
    """Every drawn endpoint is the ray/plane intersection, not a hand-placed spot."""
    _, _, figure = _orders(azimuth_deg)
    traces = _named_traces(figure)
    plane_x = float(traces["detector plane"].x[0])
    centre_y = float(traces["specular hit point"].y[0])
    standoff = plane_x - float(traces["incident beam k_i"].x[1])
    impact_z = float(traces["incident beam k_i"].z[1])
    drawn = _order_labels(figure)

    for rod in rheed.rod_orders(
        grazing_angle_deg=GRAZING_DEG,
        azimuth_deg=azimuth_deg,
        span_deg=figures.ORDERS_ACCEPTANCE_DEG,
    ):
        if (rod.h, rod.k) == (0, 0):
            continue
        direction = rheed.outgoing_direction(rod.exit_angle_deg, rod.deflection_deg)
        horizontal, vertical = rheed.detector_intersection(direction, standoff)
        # This mode draws to scale, so the intersection is the screen coordinate itself.
        assert drawn[rod.label] == pytest.approx((centre_y + horizontal, impact_z + vertical))
        # And that is the coordinate diffraction_screen's own angular mapping gives.
        assert (horizontal, vertical) == pytest.approx(
            rheed.detector_offsets(rod.exit_angle_deg, rod.deflection_deg, standoff)
        )


def test_order_rays_land_on_the_painted_screen_they_belong_to() -> None:
    """A ray inside the painted window must meet its own feature, not sit beside it."""
    _, pattern, figure = _orders(0.0)
    traces = _named_traces(figure)
    screen = traces["computed screen"]
    drawn = _order_labels(figure)
    inside = [
        rod
        for rod in rheed.rod_orders(
            grazing_angle_deg=GRAZING_DEG, azimuth_deg=0.0, span_deg=GRAZING_DEG
        )
        if (rod.h, rod.k) != (0, 0)
        and abs(rod.deflection_deg) <= pattern.deflection_deg[-1]
        and abs(rod.exit_angle_deg - GRAZING_DEG) <= pattern.deflection_deg[-1]
    ]
    assert inside, "need at least one order inside the painted window to compare"
    for rod in inside:
        row = int(np.argmin(np.abs(pattern.exit_angle_deg - rod.exit_angle_deg)))
        column = int(np.argmin(np.abs(pattern.deflection_deg - rod.deflection_deg)))
        pixel = (float(screen.y[row, column]), float(screen.z[row, column]))
        # Within one screen pixel of the painted position for that direction.
        pitch = abs(float(screen.y[row, column + 1]) - pixel[0]) + abs(
            float(screen.z[row + 1, column]) - pixel[1]
        )
        assert drawn[rod.label] == pytest.approx(pixel, abs=2.0 * pitch)


def test_azimuth_periodicity_and_mirror_symmetry_of_the_drawn_orders() -> None:
    """The hexagonal surface repeats every 60 deg and mirrors at 30, so the picture must too."""

    def geometry(azimuth_deg: float) -> np.ndarray:
        return np.array(sorted(_order_labels(_orders(azimuth_deg)[2]).values()))

    centre_y = float(_named_traces(_orders(0.0)[2])["specular hit point"].y[0])
    assert np.allclose(geometry(0.0), geometry(60.0))
    # 0 and 30 deg are the two inequivalent high-symmetry directions, so they must differ.
    assert geometry(0.0).shape != geometry(30.0).shape

    # Mirroring about the plane of incidence turns +50 deg into -50, which is +10 by periodicity.
    reflected = geometry(50.0).copy()
    reflected[:, 0] = 2.0 * centre_y - reflected[:, 0]
    assert np.allclose(geometry(10.0), reflected[np.lexsort(reflected.T[::-1])])


def test_azimuth_moves_only_the_non_specular_orders() -> None:
    straight, turned = _orders(0.0)[2], _orders(25.0)[2]
    straight_traces, turned_traces = _named_traces(straight), _named_traces(turned)
    for name in ("incident beam k_i", "nominal specular k_f (00)", "specular hit point"):
        assert turned_traces[name].y == pytest.approx(straight_traces[name].y)
        assert turned_traces[name].z == pytest.approx(straight_traces[name].z)
    assert _order_labels(straight) != _order_labels(turned)


@pytest.mark.parametrize("order", [3, 5])
def test_grazing_angle_moves_the_nominal_specular(order: int) -> None:
    angle = rheed.antiphase_grazing_angle_deg(order)
    _, _, figure = _orders(0.0, grazing_angle_deg=angle)
    traces = _named_traces(figure)
    beam = traces["nominal specular k_f (00)"]
    standoff = float(traces["detector plane"].x[0]) - float(beam.x[0])
    rise = float(beam.z[1]) - float(beam.z[0])
    assert rise == pytest.approx(rheed.detector_offsets(angle, 0.0, standoff)[1])


def test_the_geometry_view_touches_neither_the_surface_nor_the_pattern() -> None:
    """A visualization that mutates its inputs would corrupt the run it is drawing."""
    heights = initial_lattice("mounds", SIZE)
    pattern = rheed.diffraction_screen(heights, grazing_angle_deg=GRAZING_DEG)
    before_heights = heights.copy()
    before_intensity = pattern.intensity.copy()
    before_specular = pattern.specular_intensity
    for show_orders in (False, True):
        figures.rheed_geometry(
            heights,
            1.0,
            max(1, int(heights.max())),
            grazing_angle_deg=GRAZING_DEG,
            pattern=pattern,
            show_orders=show_orders,
        )
    assert np.array_equal(heights, before_heights)
    assert np.array_equal(pattern.intensity, before_intensity)
    assert pattern.specular_intensity == before_specular


def test_the_footprint_is_the_coherence_area_not_a_point() -> None:
    _, pattern, figure = _orders(0.0)
    footprint = next(
        t for t in figure.data if t.name and t.name.startswith("illuminated footprint")
    )
    centre_y = float(_named_traces(figure)["specular hit point"].y[0])
    expected = pattern.coherence_length_nm / rheed.GAN_IN_PLANE_SPACING_NM
    assert max(np.asarray(footprint.y)) - centre_y == pytest.approx(expected)
