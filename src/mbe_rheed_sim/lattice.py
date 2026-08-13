from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray

HeightField = NDArray[np.int64]

# Axial-coordinate connectivity represented on a periodic rectangular array.
HEX_DIRECTIONS = ((0, 1), (0, -1), (1, 0), (-1, 0), (1, -1), (-1, 1))


def empty_lattice(size: int) -> HeightField:
    if size < 2:
        raise ValueError("size must be at least 2")
    return np.zeros((size, size), dtype=np.int64)


def neighbors(y: int, x: int, size: int) -> Iterator[tuple[int, int]]:
    for dy, dx in HEX_DIRECTIONS:
        yield (y + dy) % size, (x + dx) % size


def lateral_bonds(heights: HeightField, y: int, x: int) -> int:
    level = int(heights[y, x])
    if level == 0:
        return 0
    return sum(heights[ny, nx] >= level for ny, nx in neighbors(y, x, len(heights)))


def hop_allowed(heights: HeightField, source: tuple[int, int], target: tuple[int, int]) -> bool:
    y, x = source
    ny, nx = target
    return (
        heights[y, x] > 0
        and target in neighbors(y, x, len(heights))
        and 0 <= int(heights[y, x]) - (int(heights[ny, nx]) + 1) <= 1
    )


def deposit(heights: HeightField, y: int, x: int) -> None:
    heights[y, x] += 1


def hop(heights: HeightField, source: tuple[int, int], target: tuple[int, int]) -> None:
    if not hop_allowed(heights, source, target):
        raise ValueError("illegal surface hop")
    heights[source] -= 1
    heights[target] += 1
