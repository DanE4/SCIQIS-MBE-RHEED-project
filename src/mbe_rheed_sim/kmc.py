import json
import math
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from mbe_rheed_sim.config import SimulationConfig
from mbe_rheed_sim.lattice import (
    HEX_DIRECTIONS,
    HeightField,
    deposit,
    empty_lattice,
    long_hop,
)
from mbe_rheed_sim.observables import (
    coverage_ml,
    island_density_per_site,
    rms_roughness_ml,
    step_density_proxy,
)
from mbe_rheed_sim.rates import BOLTZMANN_EV_PER_K


@dataclass(frozen=True, slots=True)
class SimulationResult:
    config: SimulationConfig
    final_heights: HeightField
    coverage_ml: NDArray[np.float64]
    time_s: NDArray[np.float64]
    roughness_ml: NDArray[np.float64]
    island_density_per_site: NDArray[np.float64]
    rheed_proxy: NDArray[np.float64]
    snapshots: NDArray[np.int64]
    deposited_events: int
    selected_diffusion_events: int
    diffusion_events: int
    long_hop_events: int
    desorbed_events: int

    def save_npz(self, path: str | Path) -> None:
        """Serialize arrays and configuration without custom object pickling."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            config_json=json.dumps(asdict(self.config), sort_keys=True),
            final_heights=self.final_heights,
            coverage_ml=self.coverage_ml,
            time_s=self.time_s,
            roughness_ml=self.roughness_ml,
            island_density_per_site=self.island_density_per_site,
            rheed_proxy=self.rheed_proxy,
            snapshots=self.snapshots,
            deposited_events=self.deposited_events,
            selected_diffusion_events=self.selected_diffusion_events,
            diffusion_events=self.diffusion_events,
            long_hop_events=self.long_hop_events,
            desorbed_events=self.desorbed_events,
        )


@dataclass(frozen=True, slots=True)
class _DiffusionCatalogue:
    cumulative_rates: NDArray[np.float64]
    sources: NDArray[np.int64]
    directions: NDArray[np.int64]
    distances: NDArray[np.int64]

    @property
    def total_rate(self) -> float:
        return float(self.cumulative_rates[-1]) if self.cumulative_rates.size else 0.0


class _RateTree:
    """Update and sample site rates without rebuilding a full cumulative array."""

    def __init__(self, values: NDArray[np.float64]) -> None:
        self.values = np.asarray(values, dtype=float).ravel().copy()
        self.tree = np.zeros(self.values.size + 1, dtype=float)
        self.tree[1:] = self.values
        for index in range(1, self.values.size + 1):
            parent = index + (index & -index)
            if parent <= self.values.size:
                self.tree[parent] += self.tree[index]

    @property
    def total_rate(self) -> float:
        total = 0.0
        index = self.values.size
        while index:
            total += self.tree[index]
            index -= index & -index
        return total

    def update(self, indices: NDArray[np.int64], values: NDArray[np.float64]) -> None:
        deltas = values - self.values[indices]
        self.values[indices] = values
        tree_indices = indices + 1
        while tree_indices.size:
            np.add.at(self.tree, tree_indices, deltas)
            tree_indices = tree_indices + (tree_indices & -tree_indices)
            active = tree_indices < self.tree.size
            tree_indices = tree_indices[active]
            deltas = deltas[active]

    def select(self, rate: float) -> tuple[int, float]:
        total = self.total_rate
        if not 0 <= rate < total:
            raise ValueError("selected rate must be inside the positive total rate")
        index = 0
        step = 1 << (self.values.size.bit_length() - 1)
        residual = rate
        while step:
            candidate = index + step
            if candidate <= self.values.size and self.tree[candidate] <= residual:
                index = candidate
                residual -= self.tree[candidate]
            step >>= 1
        return index, residual


@cache
def _hex_ring_offsets(radius: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (dy, dx)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
        if max(abs(dx), abs(dy), abs(dx + dy)) == radius
    )


@cache
def _hex_disk_offsets(radius: int) -> tuple[tuple[int, int], ...]:
    return ((0, 0),) + tuple(
        offset for ring in range(1, radius + 1) for offset in _hex_ring_offsets(ring)
    )


def _bond_counts(heights: HeightField) -> NDArray[np.int64]:
    counts = np.zeros_like(heights)
    for dy, dx in HEX_DIRECTIONS:
        neighbor_heights = np.roll(heights, shift=(-dy, -dx), axis=(0, 1))
        counts += neighbor_heights >= heights
    counts[heights == 0] = 0
    return counts


@cache
def _long_hop_rate_tables(
    config: SimulationConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    bonds = np.arange(7)
    thermal_energy = BOLTZMANN_EV_PER_K * config.temperature_k
    diffusion = config.attempt_frequency_hz * np.exp(
        -(
            config.diffusion_barrier_ev
            + bonds[:, None] * config.lateral_bond_energy_ev
            + np.array([0.0, config.step_barrier_ev])
        )
        / thermal_energy
    )
    desorption = config.attempt_frequency_hz * np.exp(
        -(config.desorption_barrier_ev + bonds * config.lateral_bond_energy_ev) / thermal_energy
    )
    return diffusion, desorption


def _exact_diffusion_catalogue(
    heights: HeightField,
    config: SimulationConfig,
    bonds: NDArray[np.int64],
) -> _DiffusionCatalogue:
    size = config.lattice_size
    rates = np.zeros((*heights.shape, 6), dtype=float)
    for index, (dy, dx) in enumerate(HEX_DIRECTIONS):
        target_heights = np.roll(heights, shift=(-dy, -dx), axis=(0, 1))
        allowed = (heights > 0) & (np.abs(heights - (target_heights + 1)) <= 1)
        downward = heights > target_heights + 1
        barriers = (
            config.diffusion_barrier_ev
            + bonds * config.lateral_bond_energy_ev
            + downward * config.step_barrier_ev
        )
        rates[..., index] = np.where(
            allowed,
            config.attempt_frequency_hz
            * np.exp(-barriers / (BOLTZMANN_EV_PER_K * config.temperature_k))
            / 6.0,
            0.0,
        )

    selected = np.flatnonzero(rates)
    site_indices, directions = np.divmod(selected, 6)
    sources = np.column_stack(np.divmod(site_indices, size)).astype(np.int64)
    selected_rates = rates.ravel()[selected]
    return _DiffusionCatalogue(
        cumulative_rates=np.cumsum(selected_rates),
        sources=sources,
        directions=directions.astype(np.int64),
        distances=np.ones(len(selected), dtype=np.int64),
    )


def _long_hop_site_rates(
    heights: HeightField,
    config: SimulationConfig,
    sources: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]]:
    if not len(sources):
        return (
            np.empty((0, 6), dtype=float),
            np.array([], dtype=np.int64),
            np.array([], dtype=float),
        )

    size = config.lattice_size
    source_y, source_x = sources.T
    source_heights = heights[source_y, source_x]
    diffusion_table, desorption_table = _long_hop_rate_tables(config)
    bonds = np.zeros(len(sources), dtype=np.int64)
    for dy, dx in HEX_DIRECTIONS:
        bonds += heights[(source_y + dy) % size, (source_x + dx) % size] >= source_heights
    bonds[source_heights == 0] = 0
    distances = np.ones(len(sources), dtype=np.int64)
    open_sites = (source_heights > 0) & (bonds == 0)
    distances[open_sites] = config.max_isolated_hop_distance
    for radius in range(1, config.max_isolated_hop_distance):
        ring_open = np.ones_like(open_sites)
        for dy, dx in _hex_ring_offsets(radius):
            ring_open &= (
                heights[(source_y + dy) % size, (source_x + dx) % size] == source_heights - 1
            )
        blocked = open_sites & ~ring_open
        distances[blocked] = max(1, radius - 1)
        open_sites &= ring_open

    rates = np.zeros((len(sources), 6), dtype=float)
    for index, (dy, dx) in enumerate(HEX_DIRECTIONS):
        target_heights = heights[(source_y + dy) % size, (source_x + dx) % size]
        short_hop = distances == 1
        allowed = (source_heights > 0) & (
            ~short_hop | (np.abs(source_heights - (target_heights + 1)) <= 1)
        )
        downward = short_hop & (source_heights > target_heights + 1)
        rates[..., index] = np.where(
            allowed,
            diffusion_table[bonds, downward.astype(np.int64)] / (6.0 * distances**2),
            0.0,
        )
    desorption_rates = np.where(
        source_heights > 0,
        desorption_table[bonds],
        0.0,
    )
    return rates, distances, desorption_rates


def _long_hop_diffusion_catalogue(
    heights: HeightField,
    config: SimulationConfig,
    sources: NDArray[np.int64],
) -> _DiffusionCatalogue:
    rates, distances, _ = _long_hop_site_rates(heights, config, sources)

    selected = np.flatnonzero(rates)
    source_indices, directions = np.divmod(selected, 6)
    selected_rates = rates.ravel()[selected]
    return _DiffusionCatalogue(
        cumulative_rates=np.cumsum(selected_rates),
        sources=sources[source_indices],
        directions=directions.astype(np.int64),
        distances=np.repeat(distances.ravel(), 6)[selected],
    )


def _diffusion_events(
    heights: HeightField,
    config: SimulationConfig,
    bond_counts: NDArray[np.int64] | None = None,
    sources: NDArray[np.int64] | None = None,
) -> _DiffusionCatalogue:
    if config.max_isolated_hop_distance == 1:
        if bond_counts is None:
            bond_counts = _bond_counts(heights)
        return _exact_diffusion_catalogue(heights, config, bond_counts)
    if sources is None:
        sources = np.argwhere(heights > 0).astype(np.int64)
    return _long_hop_diffusion_catalogue(heights, config, sources)


def _desorption_events(
    heights: HeightField,
    config: SimulationConfig,
    bond_counts: NDArray[np.int64] | None = None,
    sources: NDArray[np.int64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    if sources is None:
        sources = np.argwhere(heights > 0).astype(np.int64)
    if not len(sources):
        return np.array([], dtype=float), sources
    if bond_counts is None:
        bond_counts = _bond_counts(heights)
    bonds = bond_counts[tuple(sources.T)]
    barriers = config.desorption_barrier_ev + bonds * config.lateral_bond_energy_ev
    rates = config.attempt_frequency_hz * np.exp(
        -barriers / (BOLTZMANN_EV_PER_K * config.temperature_k)
    )
    return np.cumsum(rates), sources


class _LocalLongHopCatalogue:
    """Locally update accelerated rates after a surface mutation."""

    def __init__(self, heights: HeightField, config: SimulationConfig) -> None:
        self.heights = heights
        self.config = config
        self.diffusion_rates = np.zeros((*heights.shape, 6), dtype=float)
        self.distances = np.ones_like(heights)
        self.desorption_rates = np.zeros_like(heights, dtype=float)
        self.rate_tree: _RateTree | None = None
        coordinates = np.indices(heights.shape).reshape(2, -1).T
        self._refresh_sources(coordinates)
        if heights.size >= 128**2:
            self.rate_tree = _RateTree(self.diffusion_rates.sum(axis=2) + self.desorption_rates)

    def _refresh_sources(self, sources: NDArray[np.int64]) -> None:
        rates, distances, desorption_rates = _long_hop_site_rates(
            self.heights, self.config, sources
        )
        y, x = sources.T
        self.diffusion_rates[y, x] = rates
        self.distances[y, x] = distances
        self.desorption_rates[y, x] = desorption_rates
        if self.rate_tree is not None:
            self.rate_tree.update(
                y * self.config.lattice_size + x,
                rates.sum(axis=1) + desorption_rates,
            )

    @property
    def total_rate(self) -> float:
        if self.rate_tree is None:
            raise RuntimeError("rate tree is disabled for this small lattice")
        return self.rate_tree.total_rate

    def select(self, rate: float) -> tuple[tuple[int, int], int | None, int]:
        if self.rate_tree is None:
            raise RuntimeError("rate tree is disabled for this small lattice")
        site_index, residual = self.rate_tree.select(rate)
        source = divmod(site_index, self.config.lattice_size)
        direction_rates = self.diffusion_rates[source]
        total_diffusion_rate = float(direction_rates.sum())
        if residual >= total_diffusion_rate:
            return source, None, 0
        direction = int(np.searchsorted(np.cumsum(direction_rates), residual, side="right"))
        return source, direction, int(self.distances[source])

    def refresh_near(self, changed_sites: tuple[tuple[int, int], ...]) -> None:
        size = self.config.lattice_size
        radius = max(1, self.config.max_isolated_hop_distance - 1)
        affected = {
            ((y + dy) % size, (x + dx) % size)
            for y, x in changed_sites
            for dy, dx in _hex_disk_offsets(radius)
        }
        self._refresh_sources(np.asarray(sorted(affected), dtype=np.int64))


def run(config: SimulationConfig) -> SimulationResult:
    """Run the baseline residence-time KMC from an empty surface."""
    rng = np.random.default_rng(config.seed)
    heights = empty_lattice(config.lattice_size)
    sites = heights.size
    target_atoms = (
        None
        if config.target_coverage_ml is None
        else math.ceil(config.target_coverage_ml * sites - 1e-12)
    )
    sample_atoms = max(1, round(config.sample_every_ml * sites))
    next_sample = sample_atoms
    deposition_rate = config.deposition_flux_ml_s * sites

    deposited = 0
    diffusion_selections = 0
    diffused = 0
    long_hops = 0
    desorbed = 0
    time = 0.0
    coverage_history: list[float] = []
    time_history: list[float] = []
    roughness_history: list[float] = []
    island_history: list[float] = []
    rheed_history: list[float] = []
    snapshots: list[HeightField] = []

    def record() -> None:
        coverage_history.append(coverage_ml(heights))
        time_history.append(time)
        roughness_history.append(rms_roughness_ml(heights))
        island_history.append(island_density_per_site(heights))
        rheed_history.append(step_density_proxy(heights))
        snapshots.append(heights.copy())

    local_catalogue = (
        _LocalLongHopCatalogue(heights, config) if config.max_isolated_hop_distance > 1 else None
    )
    record()
    for _ in range(config.max_events):
        if target_atoms is not None and deposited - desorbed >= target_atoms:
            break

        if local_catalogue is None:
            bond_counts = _bond_counts(heights)
            occupied_sources = np.argwhere(heights > 0).astype(np.int64)
            diffusion_events = _diffusion_events(heights, config, bond_counts, occupied_sources)
            desorption_rates, desorption_sources = _desorption_events(
                heights, config, bond_counts, occupied_sources
            )
            total_diffusion_rate = diffusion_events.total_rate
            total_desorption_rate = float(desorption_rates[-1]) if desorption_rates.size else 0.0
            total_surface_rate = total_diffusion_rate + total_desorption_rate
        elif local_catalogue.rate_tree is None:
            diffusion_rates = np.cumsum(local_catalogue.diffusion_rates.ravel())
            desorption_rates = np.cumsum(local_catalogue.desorption_rates.ravel())
            total_diffusion_rate = float(diffusion_rates[-1])
            total_desorption_rate = float(desorption_rates[-1])
            total_surface_rate = total_diffusion_rate + total_desorption_rate
        else:
            total_surface_rate = local_catalogue.total_rate
        total_rate = deposition_rate + total_surface_rate
        next_time = time - math.log(max(float(rng.random()), np.finfo(float).tiny)) / total_rate
        if config.target_time_s is not None and next_time >= config.target_time_s:
            time = config.target_time_s
            record()
            break
        time = next_time
        selected_rate = float(rng.random()) * total_rate

        if selected_rate < deposition_rate:
            y, x = rng.integers(0, config.lattice_size, size=2)
            deposit(heights, int(y), int(x))
            changed_sites = ((int(y), int(x)),)
            deposited += 1
        elif local_catalogue is not None and local_catalogue.rate_tree is not None:
            selected_rate -= deposition_rate
            source, direction_index, distance = local_catalogue.select(selected_rate)
            if direction_index is None:
                heights[source] -= 1
                changed_sites = (source,)
                desorbed += 1
            else:
                direction = HEX_DIRECTIONS[direction_index]
                target = (
                    (source[0] + distance * direction[0]) % config.lattice_size,
                    (source[1] + distance * direction[1]) % config.lattice_size,
                )
                long_hop(heights, source, direction, distance)
                changed_sites = source, target
                diffusion_selections += 1
                diffused += distance**2
                if distance > 1:
                    long_hops += 1
        elif selected_rate < deposition_rate + total_diffusion_rate:
            selected_rate -= deposition_rate
            if local_catalogue is None:
                event_index = int(
                    np.searchsorted(diffusion_events.cumulative_rates, selected_rate, side="right")
                )
                source = tuple(diffusion_events.sources[event_index])
                direction = HEX_DIRECTIONS[diffusion_events.directions[event_index]]
                distance = int(diffusion_events.distances[event_index])
            else:
                event_index = int(np.searchsorted(diffusion_rates, selected_rate, side="right"))
                site_index, direction_index = divmod(event_index, 6)
                source = divmod(site_index, config.lattice_size)
                direction = HEX_DIRECTIONS[direction_index]
                distance = int(local_catalogue.distances[source])
            target = (
                (source[0] + distance * direction[0]) % config.lattice_size,
                (source[1] + distance * direction[1]) % config.lattice_size,
            )
            long_hop(heights, source, direction, distance)
            changed_sites = source, target
            diffusion_selections += 1
            diffused += distance**2
            if distance > 1:
                long_hops += 1
        else:
            selected_rate -= deposition_rate + total_diffusion_rate
            event_index = int(np.searchsorted(desorption_rates, selected_rate, side="right"))
            source = (
                tuple(desorption_sources[event_index])
                if local_catalogue is None
                else divmod(event_index, config.lattice_size)
            )
            heights[source] -= 1
            changed_sites = (source,)
            desorbed += 1

        if local_catalogue is not None:
            local_catalogue.refresh_near(changed_sites)

        if deposited - desorbed >= next_sample:
            record()
            next_sample += sample_atoms
    else:
        raise RuntimeError(
            f"simulation target not reached within max_events={config.max_events}; "
            "increase max_events or use isolated-adatom acceleration"
        )

    if coverage_history[-1] != coverage_ml(heights):
        record()
    if int(heights.sum()) != deposited - desorbed or np.any(heights < 0):
        raise RuntimeError("KMC mass/non-negativity invariant failed")

    return SimulationResult(
        config=config,
        final_heights=heights.copy(),
        coverage_ml=np.asarray(coverage_history),
        time_s=np.asarray(time_history),
        roughness_ml=np.asarray(roughness_history),
        island_density_per_site=np.asarray(island_history),
        rheed_proxy=np.asarray(rheed_history),
        snapshots=np.stack(snapshots),
        deposited_events=deposited,
        selected_diffusion_events=diffusion_selections,
        diffusion_events=diffused,
        long_hop_events=long_hops,
        desorbed_events=desorbed,
    )
