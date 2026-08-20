"""Checks for the illustrative Stranski-Krastanov overlay.

The point of these is the guarantee around it: the notebook must behave exactly as it does today
unless a caller explicitly asks for the prescribed frames, and the frames it does append must be
legal height fields on an increasing coverage axis.
"""

from functools import lru_cache

import numpy as np
import pytest

from mbe_rheed_notebook import figures, reconstruction
from mbe_rheed_sim.config import SimulationConfig
from mbe_rheed_sim.kmc import run

# The overlay needs a lattice that can resolve the template, so the fixtures run at its minimum.
_SIZE = reconstruction.MIN_LATTICE_SIZE


# Cached because the seed is fixed: every caller wants the same trajectory, and simulating it once
# instead of per test takes seconds off the suite. Nothing here mutates a result - the test below
# that checks `extend` leaves the recording alone would fail first if anything did.
@lru_cache(maxsize=2)
def _run(lattice_size: int = _SIZE) -> object:
    return run(
        SimulationConfig(
            lattice_size=lattice_size,
            target_coverage_ml=1.0,
            seed=7,
            sample_every_ml=0.25,
            max_events=50_000_000,
        )
    )


def test_off_by_default_returns_the_recorded_run_untouched() -> None:
    result = _run()
    extended, axis, order = reconstruction.extend(result, result.coverage_ml, enabled=False)

    assert extended is result
    assert order is None
    np.testing.assert_array_equal(axis, result.coverage_ml)


def test_a_lattice_too_small_to_resolve_the_template_declines() -> None:
    result = _run(8)
    extended, _axis, order = reconstruction.extend(result, result.coverage_ml, enabled=True)

    assert extended is result
    assert order is None


def test_the_order_parameter_is_one_on_the_template_and_zero_on_a_flat_surface() -> None:
    template = reconstruction.frame_pattern(64, 0)
    flat = np.zeros((1, 64, 64), dtype=np.int64)

    assert reconstruction.order_parameter(template[None], template)[0] == pytest.approx(1.0)
    assert reconstruction.order_parameter(flat, template)[0] == 0.0


def test_the_stored_template_becomes_relief_standing_proud_of_the_substrate() -> None:
    """Brightness is read as monolayers, so a frame's tones *are* its height levels."""
    assert reconstruction.frame_count() > 1
    pattern = reconstruction.frame_pattern(128, 0)

    assert pattern.shape == (128, 128)
    # Ten levels of relief spanning the tonal range, and nothing outside it.
    assert pattern.max() - pattern.min() == pytest.approx(
        reconstruction.HEIGHT_LEVELS - 1, abs=0.5
    )
    # Inverted: the template's bright background must end up *below* its dark structure, or the
    # structure reads as a pit in the film rather than something grown on it. Stated as the sign of
    # the correlation over the whole frame, since any single pixel is averaged with neighbours.
    edge = round(128 * reconstruction._IMAGE_FRACTION)
    left = (128 - edge) // 2
    luminance = reconstruction._resample(
        reconstruction._luminance()[0].astype(float), edge
    )[::-1]
    heights = pattern[128 - edge :, left : left + edge]
    correlation = np.corrcoef(luminance.ravel(), heights.ravel())[0, 1]
    assert correlation < -0.99


def test_the_label_sits_below_the_template_and_skips_itself_when_it_cannot_fit() -> None:
    pattern = reconstruction.frame_pattern(128, 0)
    edge = round(128 * reconstruction._IMAGE_FRACTION)
    band = 128 - edge

    # Row 0 is the bottom of the rendered surface, so the label belongs in the low rows, and it
    # must reach the tallest level there - that is what makes it read against the template.
    assert pattern[:band].max() == pytest.approx(pattern.max())
    # Too little room is not an error and must not overrun the template: the label is skipped.
    cramped = np.zeros((12, 12))
    reconstruction._stamp_label(cramped, top=8, scale=1)
    assert not cramped.any()


def test_the_camera_faces_the_viewer_once_the_surface_orders() -> None:
    result = _run()
    extended, _axis, order = reconstruction.extend(result, result.coverage_ml, enabled=True)
    last = len(order.r_sk) - 1

    def scene(frame: int, of: object = order) -> reconstruction.Scene:
        return reconstruction.scene(of, frame, extended.snapshots, result.snapshots)

    assert scene(order.first_appended_frame - 1).camera is None
    assert scene(0, None).camera is None
    emerging = scene(order.first_appended_frame)
    ordered = scene(last)
    # Higher elevation means more top-down: a pattern lying flat only reads when faced.
    assert ordered.camera["eye"]["z"] > emerging.camera["eye"]["z"]
    # And it has to get properly overhead. The label's strokes are one site wide, so below
    # roughly 85 degrees their vertical walls hide their tops and the text becomes a fence.
    eye = ordered.camera["eye"]
    assert np.degrees(np.arctan2(eye["z"], np.hypot(eye["x"], eye["y"]))) > 85.0
    # Emerging frames each publish their own revision, or the new angle never takes effect...
    assert emerging.revision != scene(order.first_appended_frame + 1).revision
    # ...and every ordered frame shares one, or Plotly resets the scene on every playback tick.
    assert ordered.revision == scene(last - 1).revision
    assert scene(last - 1).camera == ordered.camera
    # Only the ordered frames are titled, and only they hold one colour scale between them.
    assert ordered.title and not emerging.title
    assert ordered.zmax == emerging.zmax
    # The recorded frames keep their own scale, or the extension would flatten them into the
    # bottom of the colourmap; the appended ones stand taller than the recording.
    recorded = scene(order.first_appended_frame - 1)
    assert recorded.zmax == max(1, int(result.snapshots.max()))
    assert ordered.zmax > recorded.zmax


def test_appended_frames_are_legal_height_fields_on_an_increasing_axis() -> None:
    result = _run()
    extended, axis, order = reconstruction.extend(result, result.coverage_ml, enabled=True)
    appended = extended.snapshots[order.first_appended_frame :]

    assert extended.snapshots.dtype == result.snapshots.dtype
    assert appended.shape[1:] == (_SIZE, _SIZE)
    assert appended.min() >= 0
    # Every stored array has to stay the same length as the frame axis, or the slider and the
    # trace would disagree about which frame is showing.
    assert all(
        len(values) == len(extended.snapshots)
        for values in (
            axis,
            extended.coverage_ml,
            extended.time_s,
            extended.roughness_ml,
            extended.island_density_per_site,
            extended.rheed_proxy,
            order.r_sk,
        )
    )
    # The axis advances while the phase emerges, then holds: past the transition the film has
    # stopped thickening and only the ordering moves. A growing axis there rescaled the scene on
    # every tick. It does not hold *exactly* - the template's frames differ in content, and an
    # integer height field cannot represent an arbitrary mean - but the residual is a few per cent
    # of the fixed height range, which shifts tones rather than rescaling the scene.
    assert np.all(np.diff(axis[: order.transition_frame + 1]) > 0)
    ordered = axis[order.transition_frame :]
    height_range = float(appended.max() - appended.min())
    assert ordered.max() - ordered.min() < 0.1 * height_range
    # The coverage reported for an appended frame is the mean height it actually has.
    np.testing.assert_allclose(
        axis[order.first_appended_frame :], [frame.mean() for frame in appended]
    )
    # The real frames must not resemble the template, or the "transition" would have no baseline.
    assert np.max(np.abs(order.r_sk[: order.first_appended_frame])) < 0.5
    assert order.theta_ml is not None
    assert axis[order.first_appended_frame] < order.theta_ml < axis[-1]


def test_playback_loops_inside_the_ordered_phase_once_the_surface_orders() -> None:
    result = _run()
    extended, _axis, order = reconstruction.extend(result, result.coverage_ml, enabled=True)
    total = len(extended.snapshots)

    # Before the transition the ticker walks the whole trajectory as it always did.
    assert reconstruction.next_frame(order, 0, total) == 1
    assert reconstruction.next_frame(None, total - 1, total) == 0
    # After it, playback stays in the ordered phase instead of replaying the entire recorded run.
    assert reconstruction.next_frame(order, total - 1, total) == order.transition_frame
    assert (
        reconstruction.next_frame(order, order.transition_frame, total)
        == order.transition_frame + 1
    )
    # And the loop visits every ordered frame, so none is skipped.
    frame, seen = order.transition_frame, set()
    for _ in range(total - order.transition_frame):
        seen.add(frame)
        frame = reconstruction.next_frame(order, frame, total)
    assert seen == set(range(order.transition_frame, total))


def test_every_section_3_figure_builds_on_the_extended_run() -> None:
    """`make check` only executes the notebook with the overlay off, so cover the other path.

    A figure that raises here would raise on a projector, in front of the audience.
    """
    result = _run()
    extended, axis, order = reconstruction.extend(result, result.coverage_ml, enabled=True)
    last = len(extended.snapshots) - 1
    heights = extended.snapshots[last]
    scene = reconstruction.scene(order, last, extended.snapshots, result.snapshots)

    for builder in (
        figures.height_surface,
        figures.hex_cells,
        figures.step_edges,
        figures.rheed_geometry,
    ):
        assert builder(heights, float(axis[last]), scene.zmax) is not None
    assert (
        figures.height_surface(
            heights, float(axis[last]), scene.zmax, camera=scene.camera, revision=scene.revision
        ).layout.scene.camera.eye.z
        == scene.camera["eye"]["z"]
    )
    assert figures.rheed_trace(axis, extended.rheed_proxy, last, "coverage (ML)") is not None
    assert (
        figures.observables(
            axis,
            extended.roughness_ml,
            extended.island_density_per_site,
            extended.rheed_proxy,
            "coverage (ML)",
        )
        is not None
    )
    assert (
        figures.reconstruction_order_parameter(
            axis,
            order.r_sk,
            last,
            order.theta_ml,
            reconstruction.CRITICAL_ORDER_PARAMETER,
            "coverage (ML)",
        )
        is not None
    )


def test_the_appended_frames_never_touch_the_real_result() -> None:
    result = _run()
    before = result.snapshots.copy()
    extended, _axis, _order = reconstruction.extend(result, result.coverage_ml, enabled=True)

    np.testing.assert_array_equal(result.snapshots, before)
    # `final_heights` stays the real last surface, so no figure reporting a final can be fed one
    # of the prescribed fields.
    np.testing.assert_array_equal(extended.final_heights, result.final_heights)


def test_camera_swing_starts_from_the_active_view_and_ends_facing() -> None:
    """Every mode's swing begins at its own default eye and lands overhead, centre preserved."""
    for start in (figures.SURFACE_CAMERA, figures.GEOMETRY_CAMERA, figures.GEOMETRY_ORDERS_CAMERA):
        begun = reconstruction._eye(0.0, start)
        assert np.allclose(
            [begun["eye"][axis] for axis in "xyz"],
            [start["eye"][axis] for axis in "xyz"],
        ), start
        # Overhead: the eye is almost all z, which is what makes the label legible.
        ended = reconstruction._eye(1.0, start)["eye"]
        assert ended["z"] > 5.0 * np.hypot(ended["x"], ended["y"])
        assert reconstruction._eye(0.5, start).get("center") == start.get("center")
