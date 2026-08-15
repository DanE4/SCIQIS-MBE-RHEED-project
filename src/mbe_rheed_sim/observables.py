from collections import deque

import numpy as np

from mbe_rheed_sim.lattice import HEX_DIRECTIONS, HeightField, neighbors


def coverage_ml(heights: HeightField) -> float:
    return float(np.mean(heights))


def rms_roughness_ml(heights: HeightField) -> float:
    return float(np.std(heights))


def step_density(heights: HeightField) -> float:
    # Three directions count every undirected hex-lattice bond once.
    differing = 0
    for dy, dx in HEX_DIRECTIONS[::2]:
        neighbor_heights = np.roll(heights, shift=(-dy, -dx), axis=(0, 1))
        differing += int(np.count_nonzero(heights != neighbor_heights))
    return differing / (3 * heights.size)


def island_sizes(heights: HeightField, layer: int | None = None) -> list[int]:
    """Periodic connected components occupied in the selected monolayer."""
    if layer is None:
        layer = int(np.floor(coverage_ml(heights))) + 1
    if layer < 1:
        raise ValueError("layer must be positive")

    occupied = heights >= layer
    visited = np.zeros_like(occupied)
    sizes: list[int] = []
    size = len(heights)

    for y, x in zip(*np.nonzero(occupied), strict=True):
        if visited[y, x]:
            continue
        visited[y, x] = True
        queue = deque([(int(y), int(x))])
        component_size = 0
        while queue:
            site_y, site_x = queue.popleft()
            component_size += 1
            for neighbor_y, neighbor_x in neighbors(site_y, site_x, size):
                if occupied[neighbor_y, neighbor_x] and not visited[neighbor_y, neighbor_x]:
                    visited[neighbor_y, neighbor_x] = True
                    queue.append((neighbor_y, neighbor_x))
        sizes.append(component_size)
    return sizes


def island_density_per_site(heights: HeightField) -> float:
    return len(island_sizes(heights)) / heights.size


def step_density_proxy(heights: HeightField) -> float:
    """Dimensionless morphology proxy; this is not a RHEED diffraction calculation."""
    return 1.0 - step_density(heights)
