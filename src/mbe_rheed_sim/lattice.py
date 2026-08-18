from collections.abc import Iterator
from functools import cache

import numpy as np
from numpy.typing import NDArray

HeightField = NDArray[np.int64]

# Axial-coordinate connectivity represented on a periodic rectangular array.
HEX_DIRECTIONS = ((0, 1), (0, -1), (1, 0), (-1, 0), (1, -1), (-1, 1))


@cache
def hex_ring_offsets(radius: int) -> tuple[tuple[int, int], ...]:
    """The 6*radius axial offsets exactly `radius` hex steps away."""
    return tuple(
        (dy, dx)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
        if max(abs(dx), abs(dy), abs(dx + dy)) == radius
    )


@cache
def hex_disk_offsets(radius: int) -> tuple[tuple[int, int], ...]:
    return ((0, 0),) + tuple(
        offset for ring in range(1, radius + 1) for offset in hex_ring_offsets(ring)
    )


def empty_lattice(size: int) -> HeightField:
    if size < 2:
        raise ValueError("size must be at least 2")
    return np.zeros((size, size), dtype=np.int64)


def _half_layer(size: int) -> HeightField:
    """Exactly half the sites raised one level, scattered. The anti-phase extreme."""
    heights = empty_lattice(size)
    # Fixed seed: the surface has to be a pure function of its name, or a config stops
    # reproducing its own trajectory.
    chosen = np.random.default_rng(20260818).permutation(size * size)[: size * size // 2]
    heights.ravel()[chosen] = 1
    return heights


def _straight_step(size: int) -> HeightField:
    """One terrace boundary, which on a torus necessarily means two straight steps."""
    heights = empty_lattice(size)
    heights[size // 2 :] = 1
    return heights


def _island(size: int) -> HeightField:
    """A single compact island, one level high, on the true hex geometry."""
    heights = empty_lattice(size)
    centre = size // 2
    for dy, dx in hex_disk_offsets(max(1, size // 8)):
        heights[(centre + dy) % size, (centre + dx) % size] = 1
    return heights


def _mounds(size: int) -> HeightField:
    """A periodic array of pyramids: triangular ramps in *both* directions.

    Ramping only along rows gives a 1D grating, which diffracts quite differently from a mound
    array, so both axes carry the ramp and the peaks are genuinely localized. Triangular rather
    than sawtooth so the array wraps without a cliff at the seam.
    """
    period = max(2, size // 4)
    row, column = np.indices((size, size))
    ridge_y = np.abs((row % period) - period // 2)
    ridge_x = np.abs((column % period) - period // 2)
    # Minimum, not sum: that is a pyramid, and it keeps the peak height at period/2 instead
    # of doubling it into a spike taller than the mound spacing.
    return np.minimum(ridge_y.max() - ridge_y, ridge_x.max() - ridge_x).astype(np.int64)


def _rough(size: int) -> HeightField:
    """Uncorrelated heights: the fully disordered limit."""
    return np.random.default_rng(20260819).integers(0, 6, (size, size)).astype(np.int64)


# Named starting surfaces. A growth run may begin from any of these instead of a bare
# substrate, which is what makes step-flow and regrowth-on-rough reachable at all. Names are
# stored in SimulationConfig, so each one must stay a deterministic function of the size.
INITIAL_SURFACES = {
    "flat": empty_lattice,
    "half-layer": _half_layer,
    "straight-step": _straight_step,
    "island": _island,
    "mounds": _mounds,
    "rough": _rough,
}


def initial_lattice(name: str, size: int) -> HeightField:
    if name not in INITIAL_SURFACES:
        raise ValueError(
            f"unknown initial surface {name!r}; expected one of {sorted(INITIAL_SURFACES)}"
        )
    return INITIAL_SURFACES[name](size)


def neighbors(y: int, x: int, size: int) -> Iterator[tuple[int, int]]:
    for dy, dx in HEX_DIRECTIONS:
        yield (y + dy) % size, (x + dx) % size


def lateral_bonds(heights: HeightField, y: int, x: int) -> int:
    level = int(heights[y, x])
    if level == 0:
        return 0
    return sum(heights[ny, nx] >= level for ny, nx in neighbors(y, x, len(heights)))


def open_terrace_hop_distance(
    heights: HeightField, y: int, x: int, maximum: int
) -> int:
    """Conservative long-hop distance for an isolated adatom on a flat terrace."""
    if maximum <= 1 or heights[y, x] <= 0 or lateral_bonds(heights, y, x) != 0:
        return 1

    base_height = int(heights[y, x]) - 1
    size = len(heights)
    for radius in range(1, maximum):
        for dy, dx in hex_ring_offsets(radius):
            if int(heights[(y + dy) % size, (x + dx) % size]) != base_height:
                return max(1, radius - 1)
    return maximum


def long_hop(
    heights: HeightField,
    source: tuple[int, int],
    direction: tuple[int, int],
    distance: int,
) -> None:
    """Move a top particle across a verified open terrace."""
    if distance < 1 or direction not in HEX_DIRECTIONS:
        raise ValueError("invalid long hop")
    if distance == 1:
        target = (
            (source[0] + direction[0]) % len(heights),
            (source[1] + direction[1]) % len(heights),
        )
        hop(heights, source, target)
        return
    if open_terrace_hop_distance(heights, *source, distance) != distance:
        raise ValueError("long hop crosses a surface obstacle")
    target = (
        (source[0] + distance * direction[0]) % len(heights),
        (source[1] + distance * direction[1]) % len(heights),
    )
    heights[source] -= 1
    heights[target] += 1


def hop_allowed(heights: HeightField, source: tuple[int, int], target: tuple[int, int]) -> bool:
    y, x = source
    ny, nx = target
    return (
        heights[y, x] > 0
        and target in neighbors(y, x, len(heights))
        and abs(int(heights[y, x]) - (int(heights[ny, nx]) + 1)) <= 1
    )


def deposit(heights: HeightField, y: int, x: int) -> None:
    heights[y, x] += 1


def hop(heights: HeightField, source: tuple[int, int], target: tuple[int, int]) -> None:
    if not hop_allowed(heights, source, target):
        raise ValueError("illegal surface hop")
    heights[source] -= 1
    heights[target] += 1
