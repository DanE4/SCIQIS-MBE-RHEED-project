import itertools
import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from mbe_rheed_sim import fastpath
from mbe_rheed_sim.config import SimulationConfig
from mbe_rheed_sim.lattice import (
    HEX_DIRECTIONS,
    HeightField,
    deposit,
    empty_lattice,
    hex_disk_offsets,
    hex_ring_offsets,
)
from mbe_rheed_sim.observables import (
    coverage_ml,
    island_density_per_site,
    rms_roughness_ml,
    step_density_proxy,
)
from mbe_rheed_sim.rates import BOLTZMANN_EV_PER_K

_EMPTY_RATES = np.zeros(0, dtype=float)
_SMALLEST_NORMAL = float(np.finfo(float).tiny)
# Events between progress reports, on top of the coverage-sampled ones.
_PROGRESS_EVENT_INTERVAL = 4_096


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

    @classmethod
    def load_npz(cls, path: str | Path) -> "SimulationResult":
        """Rebuild a result written by save_npz, without pickling."""
        with np.load(path) as stored:
            fields = {
                name: stored[name]
                for name in cls.__slots__
                if name not in {"config", "final_heights"}
            }
            return cls(
                config=SimulationConfig(**json.loads(str(stored["config_json"]))),
                final_heights=stored["final_heights"],
                **{
                    name: int(value) if value.ndim == 0 else value
                    for name, value in fields.items()
                },
            )

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
def _compiled_arguments(config: SimulationConfig) -> tuple[NDArray, ...]:
    """Geometry and rate tables the compiled refresh kernel needs, built once per config."""
    max_hop = config.max_isolated_hop_distance
    hex_dy, hex_dx = _offset_arrays(HEX_DIRECTIONS)
    disk_dy, disk_dx = _offset_arrays(hex_disk_offsets(max(1, max_hop - 1)))
    rings = tuple(offset for radius in range(1, max_hop) for offset in hex_ring_offsets(radius))
    ring_dy, ring_dx = _offset_arrays(rings or ((0, 0),))
    ring_start = np.cumsum([0, *(6 * radius for radius in range(1, max_hop - 1))], dtype=np.int64)
    diffusion_table, desorption_table = _long_hop_rate_tables(config)
    return (
        hex_dy,
        hex_dx,
        disk_dy,
        disk_dx,
        ring_dy,
        ring_dx,
        ring_start[: max_hop - 1],
        max_hop,
        diffusion_table,
        desorption_table,
    )


@cache
def _offset_arrays(offsets: tuple[tuple[int, int], ...]) -> tuple[NDArray, NDArray]:
    array = np.asarray(offsets, dtype=np.int64)
    return array[:, 0], array[:, 1]


@cache
def _neighborhood_offsets(max_hop: int) -> tuple[NDArray, NDArray, NDArray]:
    """One gather covers every site a rate depends on.

    Column layout: the six hop targets, then the open-terrace rings 1..max_hop-1.
    `starts` marks where each ring begins so `reduceat` can test them in one call.
    """
    offsets = list(HEX_DIRECTIONS)
    starts = []
    for radius in range(1, max_hop):
        starts.append(len(offsets))
        offsets.extend(hex_ring_offsets(radius))
    array = np.asarray(offsets, dtype=np.int64)
    return array[:, 0], array[:, 1], np.asarray(starts, dtype=np.intp)


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
    max_hop = config.max_isolated_hop_distance
    source_y, source_x = sources.T
    source_heights = heights[source_y, source_x]
    diffusion_table, desorption_table = _long_hop_rate_tables(config)

    offset_y, offset_x, ring_starts = _neighborhood_offsets(max_hop)
    neighborhood = heights[
        (source_y[:, None] + offset_y) % size, (source_x[:, None] + offset_x) % size
    ]
    target_heights = neighborhood[:, :6]

    bonds = np.count_nonzero(target_heights >= source_heights[:, None], axis=1)
    bonds[source_heights == 0] = 0

    distances = np.ones(len(sources), dtype=np.int64)
    open_sites = (source_heights > 0) & (bonds == 0)
    if ring_starts.size:
        # Number of leading rings that are entirely one level below the adatom.
        cleared = np.logical_and.accumulate(
            np.logical_and.reduceat(
                neighborhood == (source_heights - 1)[:, None], ring_starts, axis=1
            ),
            axis=1,
        ).sum(axis=1)
        distances[open_sites] = np.where(
            cleared[open_sites] == ring_starts.size,
            max_hop,
            np.maximum(1, cleared[open_sites]),
        )
    else:
        distances[open_sites] = max_hop

    short_hop = (distances == 1)[:, None]
    allowed = (source_heights > 0)[:, None] & (
        ~short_hop | (np.abs(source_heights[:, None] - (target_heights + 1)) <= 1)
    )
    downward = short_hop & (source_heights[:, None] > target_heights + 1)
    rates = np.where(
        allowed,
        diffusion_table[bonds[:, None], downward.astype(np.int64)]
        / (6.0 * distances[:, None] ** 2),
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
        self._compiled = _compiled_arguments(config) if fastpath.enabled() else None
        if self._compiled is not None:
            disk_size = self._compiled[2].size
            self._site_buffer = np.empty(2 * disk_size, dtype=np.int64)
            self._total_buffer = np.empty(2 * disk_size, dtype=float)

    def _refresh_sources(self, sources: NDArray[np.int64]) -> None:
        y, x = sources.T
        # An empty site has no rate and no hop distance, so only occupied sites are
        # worth a neighbourhood scan. On a sparse surface that is most of the saving.
        occupied = self.heights[y, x] > 0
        occupied_y, occupied_x = y[occupied], x[occupied]
        rates, distances, desorption_rates = _long_hop_site_rates(
            self.heights, self.config, sources[occupied]
        )
        self.diffusion_rates[y, x] = 0.0
        self.distances[y, x] = 1
        self.desorption_rates[y, x] = 0.0
        self.diffusion_rates[occupied_y, occupied_x] = rates
        self.distances[occupied_y, occupied_x] = distances
        self.desorption_rates[occupied_y, occupied_x] = desorption_rates
        if self.rate_tree is not None:
            totals = np.zeros(len(sources), dtype=float)
            totals[occupied] = rates.sum(axis=1) + desorption_rates
            self.rate_tree.update(y * self.config.lattice_size + x, totals)

    @property
    def total_rate(self) -> float:
        if self.rate_tree is None:
            raise RuntimeError("rate tree is disabled for this small lattice")
        if self._compiled is not None:
            return fastpath.tree_total(self.rate_tree.tree, self.rate_tree.values.size)
        return self.rate_tree.total_rate

    def select(self, rate: float) -> tuple[tuple[int, int], int | None, int]:
        if self.rate_tree is None:
            raise RuntimeError("rate tree is disabled for this small lattice")
        if self._compiled is not None:
            size = self.rate_tree.values.size
            if not 0 <= rate < self.total_rate:
                raise ValueError("selected rate must be inside the positive total rate")
            y, x, direction, distance = fastpath.tree_select(
                self.rate_tree.tree,
                size,
                1 << (size.bit_length() - 1),
                rate,
                self.diffusion_rates,
                self.distances,
            )
            return (y, x), (None if direction < 0 else direction), distance
        site_index, residual = self.rate_tree.select(rate)
        source = divmod(site_index, self.config.lattice_size)
        direction_rates = self.diffusion_rates[source]
        total_diffusion_rate = float(direction_rates.sum())
        if residual >= total_diffusion_rate:
            return source, None, 0
        direction = int(np.searchsorted(np.cumsum(direction_rates), residual, side="right"))
        return source, direction, int(self.distances[source])

    def _occupied_cumulative_rates(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
        """Cumulative rates over occupied sites only, in flat lattice order.

        Empty sites carry zero rate, and adding zero leaves a running sum bit-identical,
        so skipping them selects exactly the event the full-lattice cumulative sum would.
        """
        occupied = np.flatnonzero(self.heights)
        return (
            np.cumsum(self.diffusion_rates.reshape(-1, 6)[occupied].ravel()),
            np.cumsum(self.desorption_rates.reshape(-1)[occupied]),
            occupied,
        )

    def surface_totals(self) -> tuple[float, float]:
        """Total diffusion and desorption rate, for lattices sampled without a rate tree."""
        if self._compiled is not None:
            return fastpath.occupied_totals(
                self.heights, self.diffusion_rates, self.desorption_rates
            )
        diffusion, desorption, _ = self._occupied_cumulative_rates()
        return (
            float(diffusion[-1]) if diffusion.size else 0.0,
            float(desorption[-1]) if desorption.size else 0.0,
        )

    def select_diffusion(self, rate: float) -> tuple[tuple[int, int], int, int]:
        if self._compiled is not None:
            y, x, direction = fastpath.occupied_select_diffusion(
                self.heights, self.diffusion_rates, rate
            )
        else:
            cumulative, _, occupied = self._occupied_cumulative_rates()
            index = int(np.searchsorted(cumulative, rate, side="right"))
            site, direction = divmod(index, 6)
            y, x = divmod(int(occupied[site]), self.config.lattice_size)
        return (y, x), direction, int(self.distances[y, x])

    def select_desorption(self, rate: float) -> tuple[int, int]:
        if self._compiled is not None:
            return fastpath.occupied_select_desorption(self.heights, self.desorption_rates, rate)
        _, cumulative, occupied = self._occupied_cumulative_rates()
        index = int(np.searchsorted(cumulative, rate, side="right"))
        return divmod(int(occupied[index]), self.config.lattice_size)

    def refresh_near(self, changed_sites: tuple[tuple[int, int], ...]) -> None:
        if self._compiled is not None:
            changed = np.asarray(changed_sites, dtype=np.int64)
            tree = self.rate_tree
            fastpath.refresh_and_update(
                self.heights,
                changed[:, 0],
                changed[:, 1],
                *self._compiled,
                self.diffusion_rates,
                self.distances,
                self.desorption_rates,
                self._site_buffer,
                self._total_buffer,
                tree.tree if tree is not None else _EMPTY_RATES,
                tree.values if tree is not None else _EMPTY_RATES,
                tree is not None,
            )
            return

        size = self.config.lattice_size
        radius = max(1, self.config.max_isolated_hop_distance - 1)
        offset_y, offset_x = _offset_arrays(hex_disk_offsets(radius))
        changed = np.asarray(changed_sites, dtype=np.int64)
        # Flat indices sort exactly like the (y, x) tuples they encode.
        affected = np.unique(
            ((changed[:, :1] + offset_y) % size) * size + (changed[:, 1:] + offset_x) % size
        )
        self._refresh_sources(np.column_stack(np.divmod(affected, size)))


def run(
    config: SimulationConfig, on_progress: Callable[[float], None] | None = None
) -> SimulationResult:
    """Run the baseline residence-time KMC from an empty surface.

    `on_progress` is called at every sampling point with the completed fraction of the
    stopping criterion, in [0, 1].
    """
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

    def progress_fraction() -> float:
        if target_atoms:
            return min(1.0, (deposited - desorbed) / target_atoms)
        if config.target_time_s:
            return min(1.0, time / config.target_time_s)
        return 0.0

    def report_progress() -> None:
        if on_progress is not None:
            on_progress(progress_fraction())

    def record() -> None:
        coverage_history.append(coverage_ml(heights))
        time_history.append(time)
        roughness_history.append(rms_roughness_ml(heights))
        island_history.append(island_density_per_site(heights))
        rheed_history.append(step_density_proxy(heights))
        snapshots.append(heights.copy())
        report_progress()

    local_catalogue = _LocalLongHopCatalogue(heights, config)
    record()
    events = itertools.count() if config.max_events is None else range(config.max_events)
    for event_index in events:
        if target_atoms is not None and deposited - desorbed >= target_atoms:
            break

        if local_catalogue.rate_tree is None:
            total_diffusion_rate, total_desorption_rate = local_catalogue.surface_totals()
            total_surface_rate = total_diffusion_rate + total_desorption_rate
        else:
            total_surface_rate = local_catalogue.total_rate
        total_rate = deposition_rate + total_surface_rate
        next_time = time - math.log(max(float(rng.random()), _SMALLEST_NORMAL)) / total_rate
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
        elif local_catalogue.rate_tree is not None:
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
                # The catalogue only lists legal hops, so long_hop()'s re-derivation of the
                # distance would repeat work already done; tests/test_kmc.py cross-checks it.
                heights[source] -= 1
                heights[target] += 1
                changed_sites = source, target
                diffusion_selections += 1
                diffused += distance**2
                if distance > 1:
                    long_hops += 1
        elif selected_rate < deposition_rate + total_diffusion_rate:
            selected_rate -= deposition_rate
            source, direction_index, distance = local_catalogue.select_diffusion(selected_rate)
            direction = HEX_DIRECTIONS[direction_index]
            target = (
                (source[0] + distance * direction[0]) % config.lattice_size,
                (source[1] + distance * direction[1]) % config.lattice_size,
            )
            heights[source] -= 1
            heights[target] += 1
            changed_sites = source, target
            diffusion_selections += 1
            diffused += distance**2
            if distance > 1:
                long_hops += 1
        else:
            selected_rate -= deposition_rate + total_diffusion_rate
            source = local_catalogue.select_desorption(selected_rate)
            heights[source] -= 1
            changed_sites = (source,)
            desorbed += 1

        local_catalogue.refresh_near(changed_sites)

        if deposited - desorbed >= next_sample:
            record()
            next_sample += sample_atoms
        elif not event_index % _PROGRESS_EVENT_INTERVAL:
            # Sampling is coverage-based, so a slow run can sit between two frames for
            # minutes. Reporting on an event count as well keeps the caller's progress bar
            # and ETA alive without storing a snapshot.
            report_progress()
    else:
        done = progress_fraction()
        needed = math.ceil(config.max_events / done) if done > 0 else None
        raise RuntimeError(
            f"simulation target not reached within max_events={config.max_events:,}: "
            f"only {done:.1%} of the target was reached. Raise the event safety limit to "
            + (f"at least {needed:,} " if needed else "a larger value ")
            + "or use isolated-adatom acceleration (hop distance > 1)."
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
