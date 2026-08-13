import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from mbe_rheed_sim.config import SimulationConfig
from mbe_rheed_sim.lattice import (
    HeightField,
    deposit,
    empty_lattice,
    hop,
    hop_allowed,
    lateral_bonds,
)
from mbe_rheed_sim.lattice import neighbors as lattice_neighbors
from mbe_rheed_sim.observables import coverage_ml, island_density_per_site, rms_roughness_ml
from mbe_rheed_sim.rates import diffusion_rate
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
        )


def _diffusion_events(
    heights: HeightField, config: SimulationConfig
) -> tuple[list[tuple[float, tuple[int, int], tuple[int, int]]], float]:
    events: list[tuple[float, tuple[int, int], tuple[int, int]]] = []
    cumulative_rate = 0.0
    size = config.lattice_size

    for y, x in zip(*np.nonzero(heights), strict=True):
        source = int(y), int(x)
        source_rate = diffusion_rate(
            config.attempt_frequency_hz,
            config.diffusion_barrier_ev,
            config.lateral_bond_energy_ev,
            lateral_bonds(heights, *source),
            config.temperature_k,
        )
        directional_rate = source_rate / 6.0
        if directional_rate == 0:
            continue
        for target in lattice_neighbors(*source, size):
            if hop_allowed(heights, source, target):
                cumulative_rate += directional_rate
                events.append((cumulative_rate, source, target))
    return events, cumulative_rate


def run(config: SimulationConfig) -> SimulationResult:
    """Run the baseline residence-time KMC from an empty surface."""
    rng = np.random.default_rng(config.seed)
    heights = empty_lattice(config.lattice_size)
    sites = heights.size
    target_atoms = math.ceil(config.target_coverage_ml * sites - 1e-12)
    sample_atoms = max(1, round(config.sample_every_ml * sites))
    next_sample = sample_atoms
    deposition_rate = config.deposition_flux_ml_s * sites

    deposited = 0
    diffused = 0
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
        if deposited >= target_atoms:
            break

        diffusion_events, total_diffusion_rate = _diffusion_events(heights, config)
        total_rate = deposition_rate + total_diffusion_rate
        time -= math.log(max(float(rng.random()), np.finfo(float).tiny)) / total_rate
        selected_rate = float(rng.random()) * total_rate

        if selected_rate < deposition_rate:
            y, x = rng.integers(0, config.lattice_size, size=2)
            deposit(heights, int(y), int(x))
            deposited += 1
        else:
            selected_rate -= deposition_rate
            for cumulative_rate, source, target in diffusion_events:
                if selected_rate < cumulative_rate:
                    hop(heights, source, target)
                    diffused += 1
                    break

        if deposited >= next_sample:
            record()
            next_sample += sample_atoms
    else:
        raise RuntimeError(
            f"target coverage not reached within max_events={config.max_events}; "
            "increase max_events or reduce diffusion relative to deposition"
        )

    if coverage_history[-1] != coverage_ml(heights):
        record()
    if int(heights.sum()) != deposited or np.any(heights < 0):
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
    )
