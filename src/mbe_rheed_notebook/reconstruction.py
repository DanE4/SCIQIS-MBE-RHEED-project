"""Illustrative Stranski-Krastanov overlay: a prescribed ordered phase past the wetting layer.

**The frames this module appends are prescribed, not simulation output.** The solid-on-solid
model has no strain and no reconstruction, so it cannot produce an ordered phase; this overlay
shows what the transition the primary paper studies would look like in the same observables the
notebook already plots. Nothing here is reachable unless a caller passes `enabled=True`.

The template comes from `data/reconstruction/template_frames.npz`, a greyscale reduction built
once by `scripts/build_reconstruction_frames.py`; each frame is turned into a height field by
reading brightness as monolayers, inverted so the ordered structure stands proud of the substrate
instead of being a hole in it.

What is real: every observable reported for the appended frames is computed by the ordinary
`mbe_rheed_sim.observables` functions, and the critical coverage is interpolated from the measured
order parameter. So the numbers on screen are true measurements of a prescribed surface, never
prescribed measurements.

Tone is not a free parameter. The surface views colour by height, so a frame's tones *are* its
monolayer levels: brightness maps to relief and there is no separate palette to match. That is why
the reduction is greyscale, and why `HEIGHT_LEVELS` sets both the tonal depth and how tall the
structure stands.

This lives in the notebook layer because it is presentation only. `mbe_rheed_sim` does not import
it, so no saved run, gallery entry, artifact, or workflow can reach these frames.
"""

from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from mbe_rheed_sim.kmc import SimulationResult
from mbe_rheed_sim.observables import (
    coverage_ml,
    island_density_per_site,
    rms_roughness_ml,
    step_density_proxy,
)

FRAMES_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "reconstruction" / "template_frames.npz"
)

# Lattices below this cannot resolve the template, so the overlay declines to fire and the notebook
# behaves exactly as it does without it. Measured by eye across the sizes the form offers: 64 is
# where the structure and its label both read, 48 gives a vague smear with no label, and 24 - the
# stored gallery demos - is an unresolved blob. So this needs a live run, not a demo.
MIN_LATTICE_SIZE = 64
CRITICAL_ORDER_PARAMETER = 0.95
# Smallest coverage gap between consecutive emerging frames, so the axis always moves forwards.
# Kept well under the quarter-monolayer aim spacing on purpose: the correction below adds a whole
# monolayer at a time, so a threshold near the aim spacing ratchets, adding one to nearly every
# frame and marching the emergence up by more than a monolayer a step.
_MIN_COVERAGE_STEP_ML = 0.05

# Monolayers of relief between the darkest and brightest tone. This is the template's bit depth and
# the structure's height at the same time, because the surface views colour by height.
HEIGHT_LEVELS = 10

# The template's edge as a fraction of the lattice, leaving the rows beneath it for the label.
# 25/32 is chosen so the stored 200-pixel frames land exactly on a 256 lattice and halve exactly
# onto a 128 one, which is where the resampling below stays an area average rather than a decimation.
_IMAGE_FRACTION = 25 / 32

# (coverage aim in ML, blend weight against the real final surface) while the phase emerges. The
# weights are what make the ordering appear gradually instead of snapping into place, and they set
# where the order parameter crosses its threshold. The last weight has to reach 1.0, since the
# fully ordered template is what the order parameter is measured against.
#
# The aims are aims, not guarantees: rounding onto integer heights is a staircase, so a frame can
# only land near its aim. The axis always reports the coverage each frame actually has.
_EMERGENCE = (
    (2.25, 0.08),
    (2.50, 0.15),
    (2.75, 0.22),
    (3.00, 0.30),
    (3.25, 0.38),
    (3.50, 0.50),
    (3.75, 0.68),
    (4.00, 1.00),
)

# A 4x5 pixel font, only the glyphs the label needs.
_GLYPHS = {
    "A": (".##.", "#..#", "####", "#..#", "#..#"),
    "E": ("####", "#...", "###.", "#...", "####"),
    "G": (".###", "#...", "#.##", "#..#", ".###"),
    "I": ("####", ".##.", ".##.", ".##.", "####"),
    "N": ("#..#", "##.#", "#.##", "#..#", "#..#"),
    "O": (".##.", "#..#", "#..#", "#..#", ".##."),
    "P": ("###.", "#..#", "###.", "#...", "#..."),
    "R": ("###.", "#..#", "###.", "#.#.", "#..#"),
    "U": ("#..#", "#..#", "#..#", "#..#", ".##."),
    "V": ("#..#", "#..#", "#..#", ".##.", "..#."),
    "Y": ("#..#", "#..#", ".##.", "..#.", "..#."),
    " ": ("....", "....", "....", "....", "...."),
}
_LABEL = ("NEVER GONNA", "GIVE YOU UP")
_GLYPH_PITCH = 5

# Camera geometry for the ordered phase. The oblique pair matches `height_surface`'s default eye,
# so a frame with no ordering starts from the view the viewer already had. Facing means high
# elevation looking along +y, which is the only azimuth that puts the structure upright on screen -
# a full orbit would spin it onto its side for most of the sequence, so the motion is a small sway
# around that heading instead.
_CAMERA_DISTANCE = 2.2
_ELEVATION_OBLIQUE_DEG = 27.0
# Measured, not guessed: the label's strokes are one site wide, so below about 85 degrees their
# vertical walls hide their tops and two legible lines render as a picket fence.
_ELEVATION_FACING_DEG = 87.0
_AZIMUTH_OBLIQUE_DEG = 45.0
_AZIMUTH_FACING_DEG = -90.0
# `height_surface`'s own revision, and the one the ordered phase parks on.
_DEFAULT_REVISION = "surface-playback"
_ORDERED_REVISION = "reconstruction-ordered"
_ORDERED_TITLE = "Reconstructed surface phase (prescribed, not simulated)"


@dataclass(frozen=True, slots=True)
class ReconstructionOrder:
    """Order parameter over every frame of the extended run, recorded frames included."""

    r_sk: NDArray[np.float64]
    theta_ml: float | None
    first_appended_frame: int

    @property
    def transition_frame(self) -> int:
        """First frame at or above the threshold, or one past the end if none reaches it.

        Derived rather than stored, so it cannot drift from `r_sk`. Every later frame counts as
        ordered even though each template frame scores slightly under one against the single
        reference frame, which is why callers latch on this rather than on their own `r_sk` value.
        """
        ordered = self.r_sk >= CRITICAL_ORDER_PARAMETER
        return int(np.argmax(ordered)) if ordered.any() else len(self.r_sk)


@dataclass(frozen=True, slots=True)
class Scene:
    """Everything the surface view needs to draw one frame of the extended run."""

    camera: dict | None
    revision: str
    zmax: int
    title: str | None


@lru_cache(maxsize=1)
def _luminance() -> NDArray[np.uint8] | None:
    """The stored template, or None when it has not been built.

    Absent frames are not an error: the overlay simply declines, exactly as it does on a lattice
    too small to resolve them, and the notebook behaves as though it were not there.
    """
    if not FRAMES_PATH.exists():
        return None
    with np.load(FRAMES_PATH) as stored:
        return stored["luminance"]


def frame_count() -> int:
    frames = _luminance()
    return 0 if frames is None else len(frames)


def _resample(image: NDArray[np.float64], edge: int) -> NDArray[np.float64]:
    """Reduce a square image to `edge` on a side, averaging whole blocks where they divide.

    Decimating by index sampling aliases badly - fine periodic detail turns to moire - so an exact
    integer ratio is averaged instead. `_IMAGE_FRACTION` is chosen to make the ratio exact at the
    lattice sizes worth presenting at.
    """
    source = len(image)
    if source == edge:
        return image
    if source % edge == 0:
        block = source // edge
        return image.reshape(edge, block, edge, block).mean(axis=(1, 3))
    index = np.arange(edge) * source // edge
    return image[np.ix_(index, index)]


def _bitmap(line: str, scale: int) -> NDArray[np.bool_]:
    """One label line as a boolean bitmap, each glyph padded to the pitch and scaled up."""
    glyphs = [
        "".join(_GLYPHS[character][row].ljust(_GLYPH_PITCH, ".") for character in line)
        for row in range(len(_GLYPHS[" "]))
    ]
    mask = np.array([[pixel == "#" for pixel in row] for row in glyphs])
    return np.repeat(np.repeat(mask, scale, axis=0), scale, axis=1)


def _stamp_label(art: NDArray[np.float64], top: int, scale: int) -> None:
    """Write the label into the rows below the template, in place, if it fits."""
    rows, columns = art.shape
    bitmaps = [_bitmap(line, scale) for line in _LABEL]
    gap = scale
    height = sum(bitmap.shape[0] + gap for bitmap in bitmaps)
    if top + height > rows or max(bitmap.shape[1] for bitmap in bitmaps) > columns:
        return
    for bitmap in bitmaps:
        left = (columns - bitmap.shape[1]) // 2
        window = art[top : top + bitmap.shape[0], left : left + bitmap.shape[1]]
        window[bitmap] = HEIGHT_LEVELS - 1
        top += bitmap.shape[0] + gap


@lru_cache(maxsize=64)
def frame_pattern(size: int, index: int) -> NDArray[np.float64] | None:
    """One template frame as a mean-zero height pattern on a `size` x `size` lattice.

    Brightness becomes relief, inverted so the bright background sits low and the dark structure
    stands proud of it: read the other way round the structure is a pit in the film, which is both
    wrong-looking and more material. Cached, because playback asks for the same frames repeatedly.
    """
    frames = _luminance()
    if frames is None:
        return None
    edge = max(1, round(size * _IMAGE_FRACTION))
    picture = _resample(frames[index % len(frames)].astype(float), edge) / 255.0
    art = np.zeros((size, size))
    left = (size - edge) // 2
    art[:edge, left : left + edge] = (1.0 - picture) * (HEIGHT_LEVELS - 1)
    # One glyph pixel per two lattice sites at 128, four at 256: legible without crowding the
    # template. `_stamp_label` skips the label rather than crushing it when there is no room,
    # which is the only guard needed - no lattice this overlay accepts is too small for it today.
    _stamp_label(art, top=edge + max(1, size // 64), scale=max(1, size // 64))
    # Array row 0 is drawn at the bottom of every surface view, so flip once, here.
    return art[::-1] - art.mean()


def scene(
    order: ReconstructionOrder | None,
    frame: int,
    displayed_snapshots: NDArray[np.int64],
    recorded_snapshots: NDArray[np.int64],
) -> Scene:
    """Camera, revision, colour scale and title for one frame of the surface view.

    The camera moves because the ordered structure lies flat in the lattice plane, so the default
    oblique view foreshortens it into noise. While it emerges the camera swings towards facing in
    step with the order parameter, and each frame publishes its own revision so the new angle takes
    effect.

    Once ordered the revision goes constant. That matters more than it looks: Plotly only applies a
    supplied camera when the revision changes, so a per-frame revision reset the whole scene on
    every playback tick - which is what read as flicker - and a constant one both settles the view
    and hands manual orbiting back to the viewer for the duration of the sequence.

    The colour scale is the other half of that. One constant scale covers the recorded frames and
    another the appended ones, because a per-frame maximum rescales the colourbar and the z axis on
    every tick. The appended scale is taken from the ordered frames rather than all of them: a few
    noisy emerging frames peak several monolayers higher and would otherwise compress the whole
    sequence into the bottom of the colourmap. Note the deliberate asymmetry - the scale switches at
    the *first appended* frame, but its value comes from the *ordered* ones.
    """
    if order is None or frame < order.first_appended_frame:
        return Scene(None, _DEFAULT_REVISION, _zmax(recorded_snapshots), None)

    zmax = _zmax(displayed_snapshots[order.transition_frame :])
    if past_transition(order, frame):
        return Scene(_eye(1.0), _ORDERED_REVISION, zmax, _ORDERED_TITLE)
    ordering = float(np.clip(order.r_sk[frame] / CRITICAL_ORDER_PARAMETER, 0.0, 1.0))
    return Scene(_eye(ordering), f"reconstruction-emerging-{frame}", zmax, None)


def _zmax(snapshots: NDArray[np.int64]) -> int:
    return max(1, int(snapshots.max()))


def _eye(ordering: float) -> dict:
    """Camera `ordering` of the way from the default oblique view to facing the structure."""
    elevation = np.radians(
        _ELEVATION_OBLIQUE_DEG + ordering * (_ELEVATION_FACING_DEG - _ELEVATION_OBLIQUE_DEG)
    )
    azimuth = np.radians(
        _AZIMUTH_OBLIQUE_DEG + ordering * (_AZIMUTH_FACING_DEG - _AZIMUTH_OBLIQUE_DEG)
    )
    horizontal = _CAMERA_DISTANCE * np.cos(elevation)
    return {
        "eye": {
            "x": float(horizontal * np.cos(azimuth)),
            "y": float(horizontal * np.sin(azimuth)),
            "z": float(_CAMERA_DISTANCE * np.sin(elevation)),
        },
        "up": {"x": 0.0, "y": 0.0, "z": 1.0},
    }


def past_transition(order: ReconstructionOrder | None, frame: int) -> bool:
    """Whether a frame sits beyond the measured transition.

    Latched on the crossing rather than on the frame's own order parameter: the template frames are
    compared against one reference frame, so each scores a little under one and a per-frame test
    would flicker the label off again mid-sequence.
    """
    return order is not None and frame >= order.transition_frame


def next_frame(order: ReconstructionOrder | None, frame: int, total: int) -> int:
    """Advance playback one frame, looping inside the ordered phase once the surface has ordered.

    Without this the ticker walks the whole trajectory, so the ordered phase plays for a couple of
    seconds out of every full cycle of the recorded run.
    """
    if past_transition(order, frame):
        start = order.transition_frame
        return start + (frame - start + 1) % max(1, total - start)
    return (frame + 1) % total


def _quantize(pattern: NDArray[np.float64], offset: float) -> NDArray[np.int64]:
    """Round a mean-zero pattern sitting `offset` monolayers up into a legal height field."""
    return np.clip(np.rint(pattern + offset), 0.0, None).astype(np.int64)


def _solve_offset(pattern: NDArray[np.float64], target_coverage: float) -> float:
    """The offset whose quantized field comes closest to the requested mean height.

    Rounding a skewed pattern biases the mean by as much as half a monolayer, and the bias depends
    on the frame, so aiming the offset straight at the target misses. Three damped corrections
    bring it to within a few hundredths.
    """
    offset = target_coverage
    for _ in range(3):
        offset += 0.8 * (target_coverage - _quantize(pattern, offset).mean())
    return offset


def order_parameter(
    snapshots: NDArray[np.int64], template: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Cosine similarity between each mean-subtracted surface and the template pattern.

    Zero for a flat or unrelated surface, one when the surface is the template.
    """
    reference = np.asarray(template, dtype=float).ravel()
    reference = reference - reference.mean()
    fields = np.asarray(snapshots, dtype=float).reshape(len(snapshots), -1)
    fields = fields - fields.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(fields, axis=1) * np.linalg.norm(reference)
    return np.divide(fields @ reference, norms, out=np.zeros(len(fields)), where=norms > 0.0)


def _critical_coverage(
    coverage_axis: NDArray[np.float64], r_sk: NDArray[np.float64]
) -> float | None:
    """Linearly interpolated first upward crossing of the threshold."""
    crossings = np.flatnonzero(
        (r_sk[:-1] < CRITICAL_ORDER_PARAMETER) & (r_sk[1:] >= CRITICAL_ORDER_PARAMETER)
    )
    if not crossings.size:
        return None
    index = int(crossings[0])
    fraction = (CRITICAL_ORDER_PARAMETER - r_sk[index]) / (r_sk[index + 1] - r_sk[index])
    span = float(coverage_axis[index + 1] - coverage_axis[index])
    return float(coverage_axis[index] + fraction * span)


def extend(
    result: SimulationResult,
    coverage_axis: NDArray[np.float64],
    *,
    enabled: bool,
) -> tuple[SimulationResult, NDArray[np.float64], ReconstructionOrder | None]:
    """Continue a finished run into the prescribed ordered phase.

    Returns the inputs unchanged, and no order parameter, unless `enabled` is set, the lattice is
    large enough to resolve the template, and the template has been built. `enabled` is
    keyword-only and has no default so that no caller can be handed prescribed frames by accident.
    """
    axis = np.asarray(coverage_axis, dtype=float)
    size = result.config.lattice_size
    if not enabled or size < MIN_LATTICE_SIZE or not frame_count():
        return result, axis, None

    template = frame_pattern(size, 0)
    residual = result.snapshots[-1].astype(float)
    residual = residual - residual.mean()
    # Keep the appended axis increasing even when the real run already passed 2 ML.
    shift = max(0.0, float(axis[-1]) - 2.0)

    frames: list[NDArray[np.int64]] = []
    previous_coverage = float(axis[-1])
    offset = 0.0
    for target, weight in _EMERGENCE:
        # The real surface's roughness is squared out faster than the template fades in. Mixing
        # both linearly leaves the noise dominant right up to the last frame, and the ordering then
        # appears to snap into place rather than emerge.
        blended = (1.0 - weight) ** 2 * residual + weight * template
        offset = _solve_offset(blended, target + shift)
        field = _quantize(blended, offset)
        # Two aims can land on the same rounding step, which would walk the coverage axis
        # backwards. Depositing a whole extra monolayer under the pattern is the one correction
        # that moves the mean by exactly one and changes nothing else about the frame.
        while field.mean() < previous_coverage + _MIN_COVERAGE_STEP_ML:
            offset += 1.0
            field = _quantize(blended, offset)
        previous_coverage = float(field.mean())
        frames.append(field)

    # Once ordered the film stops thickening and every remaining template frame plays at one
    # coverage. Holding it - and so the height range, the colour scale, and the axes - fixed for
    # the whole sequence is what stops the scene rescaling on every tick, which read as flicker
    # when each frame grew by a monolayer. It is also the more coherent story: a metastable phase
    # reorganising, not a film still growing. The template's own frames set the length; nothing is
    # interpolated.
    #
    # One shared offset beats solving each frame to a common coverage: the frames are already
    # mean-zero, so exposure differences are gone, and roughly half of every frame is flat
    # background that crosses a rounding boundary together - which makes the reachable means about
    # half a monolayer apart, so a solver chasing an exact target overshoots into visible pumping.
    frames.extend(
        _quantize(frame_pattern(size, index), offset) for index in range(1, frame_count())
    )
    appended = np.stack(frames)
    appended_coverage = [coverage_ml(frame) for frame in appended]

    snapshots = np.concatenate((result.snapshots, appended))
    step = float(np.diff(result.time_s[-2:])[0]) if result.time_s.size > 1 else 1.0
    # `final_heights` deliberately keeps the real last surface: nothing displayed reads it, and
    # leaving it real means a prescribed height field cannot reach a figure that reports finals.
    extended = replace(
        result,
        snapshots=snapshots,
        coverage_ml=np.concatenate((result.coverage_ml, appended_coverage)),
        roughness_ml=np.concatenate(
            (result.roughness_ml, [rms_roughness_ml(frame) for frame in appended])
        ),
        island_density_per_site=np.concatenate(
            (result.island_density_per_site, [island_density_per_site(f) for f in appended])
        ),
        rheed_proxy=np.concatenate(
            (result.rheed_proxy, [step_density_proxy(frame) for frame in appended])
        ),
        time_s=np.concatenate(
            (result.time_s, result.time_s[-1] + step * np.arange(1, len(appended) + 1))
        ),
    )
    extended_axis = np.concatenate((axis, appended_coverage))
    r_sk = order_parameter(snapshots, template)
    return (
        extended,
        extended_axis,
        ReconstructionOrder(r_sk, _critical_coverage(extended_axis, r_sk), len(result.snapshots)),
    )
