"""Optional compiled kernels for the accelerated-catalogue hot path.

The KMC event loop, the RNG stream, and every rate value stay in `kmc.py`. This module
only re-expresses three per-event operations - local rate refresh, Fenwick update, and
Fenwick sampling - as scalar loops that Numba can compile, because at a few hundred
affected sites per event the NumPy versions are dominated by call overhead rather than
arithmetic.

The kernels reproduce the reference floating-point operations in the same order (rates
come from the same precomputed Arrhenius tables, sums accumulate left to right, Fenwick
deltas propagate level by level), so a compiled run is bit-identical to a reference run.
`tests/test_fastpath.py` asserts that; `MBE_KMC_BACKEND=reference` disables the kernels.
"""

import os

import numpy as np

try:  # Numba is optional: without it the reference NumPy path runs unchanged.
    from numba import njit
except ImportError:  # pragma: no cover - exercised only on platforms without a wheel
    njit = None


def enabled() -> bool:
    backend = os.environ.get("MBE_KMC_BACKEND", "auto").lower()
    if backend not in {"auto", "fast", "reference"}:
        raise ValueError("MBE_KMC_BACKEND must be auto, fast or reference")
    if backend == "reference":
        return False
    if njit is None:
        if backend == "fast":
            raise RuntimeError("MBE_KMC_BACKEND=fast requires numba to be installed")
        return False
    return True


if njit is not None:

    @njit(cache=True)
    def refresh_and_update(
        heights,
        changed_y,
        changed_x,
        hex_dy,
        hex_dx,
        disk_dy,
        disk_dx,
        ring_dy,
        ring_dx,
        ring_start,
        max_hop,
        diffusion_table,
        desorption_table,
        diffusion_rates,
        distances,
        desorption_rates,
        site_buffer,
        total_buffer,
        tree,
        values,
        use_tree,
    ):
        """Recompute every site a mutation can reach, then fold it into the rate tree."""
        size = heights.shape[0]

        count = 0
        for changed in range(changed_y.size):
            for offset in range(disk_dy.size):
                y = (changed_y[changed] + disk_dy[offset]) % size
                x = (changed_x[changed] + disk_dx[offset]) % size
                site_buffer[count] = y * size + x
                count += 1
        site_buffer[:count].sort()
        unique = 0
        for index in range(count):
            if unique == 0 or site_buffer[index] != site_buffer[unique - 1]:
                site_buffer[unique] = site_buffer[index]
                unique += 1

        for index in range(unique):
            flat = site_buffer[index]
            y = flat // size
            x = flat % size
            height = heights[y, x]
            if height == 0:
                for direction in range(6):
                    diffusion_rates[y, x, direction] = 0.0
                distances[y, x] = 1
                desorption_rates[y, x] = 0.0
                total_buffer[index] = 0.0
                continue

            bonds = 0
            for direction in range(6):
                neighbor_y = (y + hex_dy[direction]) % size
                neighbor_x = (x + hex_dx[direction]) % size
                if heights[neighbor_y, neighbor_x] >= height:
                    bonds += 1

            distance = 1
            if bonds == 0:
                cleared = 0
                for radius in range(ring_start.size):
                    stop = ring_dy.size if radius + 1 == ring_start.size else ring_start[radius + 1]
                    open_ring = True
                    for offset in range(ring_start[radius], stop):
                        neighbor_y = (y + ring_dy[offset]) % size
                        neighbor_x = (x + ring_dx[offset]) % size
                        if heights[neighbor_y, neighbor_x] != height - 1:
                            open_ring = False
                            break
                    if not open_ring:
                        break
                    cleared += 1
                if cleared == ring_start.size:
                    distance = max_hop
                elif cleared > 1:
                    distance = cleared

            total = 0.0
            span = 6.0 * float(distance * distance)
            for direction in range(6):
                neighbor_y = (y + hex_dy[direction]) % size
                neighbor_x = (x + hex_dx[direction]) % size
                target = heights[neighbor_y, neighbor_x]
                rate = 0.0
                if distance == 1:
                    step = height - (target + 1)
                    if step >= -1 and step <= 1:
                        rate = diffusion_table[bonds, 1 if step > 0 else 0] / span
                else:
                    rate = diffusion_table[bonds, 0] / span
                diffusion_rates[y, x, direction] = rate
                total += rate
            distances[y, x] = distance
            desorption_rates[y, x] = desorption_table[bonds]
            total_buffer[index] = total + desorption_table[bonds]

        if not use_tree:
            return unique

        # Same level-by-level delta propagation as the NumPy Fenwick update, so the tree
        # accumulates the identical rounding.
        deltas = np.empty(unique, dtype=np.float64)
        indices = np.empty(unique, dtype=np.int64)
        for index in range(unique):
            flat = site_buffer[index]
            deltas[index] = total_buffer[index] - values[flat]
            values[flat] = total_buffer[index]
            indices[index] = flat + 1
        active = unique
        while active > 0:
            for index in range(active):
                tree[indices[index]] += deltas[index]
            kept = 0
            for index in range(active):
                parent = indices[index] + (indices[index] & -indices[index])
                if parent < tree.size:
                    indices[kept] = parent
                    deltas[kept] = deltas[index]
                    kept += 1
            active = kept
        return unique

    @njit(cache=True)
    def occupied_totals(heights, diffusion_rates, desorption_rates):
        """Running sums in flat lattice order, skipping the empty sites that add zero."""
        diffusion = 0.0
        desorption = 0.0
        for y in range(heights.shape[0]):
            for x in range(heights.shape[1]):
                if heights[y, x] == 0:
                    continue
                for direction in range(6):
                    diffusion += diffusion_rates[y, x, direction]
                desorption += desorption_rates[y, x]
        return diffusion, desorption

    @njit(cache=True)
    def occupied_select_diffusion(heights, diffusion_rates, rate):
        running = 0.0
        for y in range(heights.shape[0]):
            for x in range(heights.shape[1]):
                if heights[y, x] == 0:
                    continue
                for direction in range(6):
                    running += diffusion_rates[y, x, direction]
                    if running > rate:
                        return y, x, direction
        return -1, -1, -1

    @njit(cache=True)
    def occupied_select_desorption(heights, desorption_rates, rate):
        running = 0.0
        for y in range(heights.shape[0]):
            for x in range(heights.shape[1]):
                if heights[y, x] == 0:
                    continue
                running += desorption_rates[y, x]
                if running > rate:
                    return y, x
        return -1, -1

    @njit(cache=True)
    def tree_total(tree, size):
        total = 0.0
        index = size
        while index:
            total += tree[index]
            index -= index & -index
        return total

    @njit(cache=True)
    def tree_select(tree, size, step, rate, diffusion_rates, distances):
        """Site, direction (-1 for desorption) and hop distance for a sampled rate."""
        index = 0
        residual = rate
        while step:
            candidate = index + step
            if candidate <= size and tree[candidate] <= residual:
                index = candidate
                residual -= tree[candidate]
            step >>= 1

        lattice_size = diffusion_rates.shape[0]
        y = index // lattice_size
        x = index - y * lattice_size
        site_rate = 0.0
        for direction in range(6):
            site_rate += diffusion_rates[y, x, direction]
        if residual >= site_rate:
            return y, x, -1, 0
        running = 0.0
        for direction in range(6):
            running += diffusion_rates[y, x, direction]
            if running > residual:
                return y, x, direction, distances[y, x]
        return y, x, 5, distances[y, x]

else:  # pragma: no cover - import guard only
    refresh_and_update = tree_total = tree_select = None
    occupied_totals = occupied_select_diffusion = occupied_select_desorption = None
