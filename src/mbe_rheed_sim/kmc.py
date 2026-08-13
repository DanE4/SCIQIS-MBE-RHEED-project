import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from mbe_rheed_sim.config import SimulationConfig
from mbe_rheed_sim.lattice import (
    HEX_DIRECTIONS,
    HeightField,
    deposit,
    empty_lattice,
    hop_allowed,
    lateral_bonds,
    long_hop,
    open_terrace_hop_distance,
)
from mbe_rheed_sim.lattice import neighbors as lattice_neighbors
from mbe_rheed_sim.observables import coverage_ml, island_density_per_site, rms_roughness_ml
from mbe_rheed_sim.rates import BOLTZMANN_EV_PER_K, diffusion_rate
from mbe_rheed_sim.rheed import step_density_proxy


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
    diffusion_events: int
    long_hop_events: int
    desorbed_events: int

    def save_npz(self, path: str | Path) -> None:
        """Serialize arrays and configuration without custom object pickling."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            config_json=json.dumps(self.config.as_dict(), sort_keys=True),
            final_heights=self.final_heights,
            coverage_ml=self.coverage_ml,
            time_s=self.time_s,
            roughness_ml=self.roughness_ml,
            island_density_per_site=self.island_density_per_site,
            rheed_proxy=self.rheed_proxy,
            snapshots=self.snapshots,
            deposited_events=self.deposited_events,
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


def _bond_counts(heights: HeightField) -> NDArray[np.int64]:
    counts = np.zeros_like(heights)
    for dy, dx in HEX_DIRECTIONS:
        neighbor_heights = np.roll(heights, shift=(-dy, -dx), axis=(0, 1))
        counts += neighbor_heights >= heights
    counts[heights == 0] = 0
    return counts


def _exact_diffusion_catalogue(
    heights: HeightField, config: SimulationConfig
) -> _DiffusionCatalogue:
    size = config.lattice_size
    bonds = _bond_counts(heights)
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


def _long_hop_diffusion_catalogue(
    heights: HeightField, config: SimulationConfig
) -> _DiffusionCatalogue:
    events: list[tuple[float, tuple[int, int], tuple[int, int], int]] = []
    cumulative_rate = 0.0
    size = config.lattice_size

    for y, x in zip(*np.nonzero(heights), strict=True):
        source = int(y), int(x)
        bonds = lateral_bonds(heights, *source)
        distance = open_terrace_hop_distance(
            heights, *source, config.max_isolated_hop_distance
        )
        if distance > 1:
            directional_rate = diffusion_rate(
                config.attempt_frequency_hz,
                config.diffusion_barrier_ev,
                config.lateral_bond_energy_ev,
                bonds,
                config.temperature_k,
            ) / (6.0 * distance**2)
            for direction in HEX_DIRECTIONS:
                cumulative_rate += directional_rate
                events.append((cumulative_rate, source, direction, distance))
            continue
        for direction, target in zip(
            HEX_DIRECTIONS, lattice_neighbors(*source, size), strict=True
        ):
            if hop_allowed(heights, source, target):
                downward = heights[source] > heights[target] + 1
                directional_rate = diffusion_rate(
                    config.attempt_frequency_hz,
                    config.diffusion_barrier_ev,
                    config.lateral_bond_energy_ev,
                    bonds,
                    config.temperature_k,
                    config.step_barrier_ev if downward else 0.0,
                ) / 6.0
                cumulative_rate += directional_rate
                events.append((cumulative_rate, source, direction, 1))
    if not events:
        return _DiffusionCatalogue(
            cumulative_rates=np.array([], dtype=float),
            sources=np.empty((0, 2), dtype=np.int64),
            directions=np.array([], dtype=np.int64),
            distances=np.array([], dtype=np.int64),
        )
    direction_indices = {direction: index for index, direction in enumerate(HEX_DIRECTIONS)}
    return _DiffusionCatalogue(
        cumulative_rates=np.array([event[0] for event in events]),
        sources=np.array([event[1] for event in events], dtype=np.int64),
        directions=np.array([direction_indices[event[2]] for event in events], dtype=np.int64),
        distances=np.array([event[3] for event in events], dtype=np.int64),
    )


def _diffusion_events(heights: HeightField, config: SimulationConfig) -> _DiffusionCatalogue:
    if config.max_isolated_hop_distance == 1:
        return _exact_diffusion_catalogue(heights, config)
    return _long_hop_diffusion_catalogue(heights, config)


def _desorption_events(
    heights: HeightField, config: SimulationConfig
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    sources = np.argwhere(heights > 0).astype(np.int64)
    if not len(sources):
        return np.array([], dtype=float), sources
    bonds = _bond_counts(heights)[tuple(sources.T)]
    barriers = config.desorption_barrier_ev + bonds * config.lateral_bond_energy_ev
    rates = config.attempt_frequency_hz * np.exp(
        -barriers / (BOLTZMANN_EV_PER_K * config.temperature_k)
    )
    return np.cumsum(rates), sources


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

    record()
    for _ in range(config.max_events):
        if target_atoms is not None and deposited - desorbed >= target_atoms:
            break

        diffusion_events = _diffusion_events(heights, config)
        desorption_rates, desorption_sources = _desorption_events(heights, config)
        total_diffusion_rate = diffusion_events.total_rate
        total_desorption_rate = float(desorption_rates[-1]) if desorption_rates.size else 0.0
        total_rate = deposition_rate + total_diffusion_rate + total_desorption_rate
        next_time = time - math.log(
            max(float(rng.random()), np.finfo(float).tiny)
        ) / total_rate
        if config.target_time_s is not None and next_time >= config.target_time_s:
            time = config.target_time_s
            record()
            break
        time = next_time
        selected_rate = float(rng.random()) * total_rate

        if selected_rate < deposition_rate:
            y, x = rng.integers(0, config.lattice_size, size=2)
            deposit(heights, int(y), int(x))
            deposited += 1
        elif selected_rate < deposition_rate + total_diffusion_rate:
            selected_rate -= deposition_rate
            event_index = int(
                np.searchsorted(diffusion_events.cumulative_rates, selected_rate, side="right")
            )
            source = tuple(diffusion_events.sources[event_index])
            direction = HEX_DIRECTIONS[diffusion_events.directions[event_index]]
            distance = int(diffusion_events.distances[event_index])
            long_hop(heights, source, direction, distance)
            diffused += distance**2
            if distance > 1:
                long_hops += 1
        else:
            selected_rate -= deposition_rate + total_diffusion_rate
            event_index = int(np.searchsorted(desorption_rates, selected_rate, side="right"))
            source = tuple(desorption_sources[event_index])
            heights[source] -= 1
            desorbed += 1

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
        diffusion_events=diffused,
        long_hop_events=long_hops,
        desorbed_events=desorbed,
    )
