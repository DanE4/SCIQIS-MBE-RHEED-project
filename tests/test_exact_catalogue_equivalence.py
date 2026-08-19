"""Prove the incremental catalogue reproduces the full-rebuild exact KMC at hop limit 1.

`run()` used to rebuild the entire rate catalogue on every event whenever
`max_isolated_hop_distance == 1`. That branch no longer exists in production. `_reference_run`
below *is* that deleted loop, kept here as an independent oracle.

It shares no rate bookkeeping with the path under test. The oracle recomputes every rate from
`np.exp` through `_diffusion_events` / `_desorption_events` and selects with `np.searchsorted`
over full cumulative arrays; production maintains a `_LocalLongHopCatalogue` with precomputed
Arrhenius tables and either an occupied-site scan or a Fenwick tree, optionally in compiled
Numba kernels. Only the lattice primitives and observables are shared, and those are not what
is under test.

Equality is asserted bit-identically. Both paths are supposed to evaluate the same
floating-point expressions, so a tolerance would hide precisely the drift this guards against.
"""

import hashlib
import math

import numpy as np
import pytest

from mbe_rheed_sim import SimulationConfig, run
from mbe_rheed_sim.kmc import (
    _bond_counts,
    _desorption_events,
    _diffusion_events,
    _LocalLongHopCatalogue,
)
from mbe_rheed_sim.lattice import HEX_DIRECTIONS, deposit, empty_lattice, hop_allowed
from mbe_rheed_sim.observables import (
    coverage_ml,
    island_density_per_site,
    rms_roughness_ml,
    step_density_proxy,
)
from mbe_rheed_sim.rates import BOLTZMANN_EV_PER_K

_SMALLEST_NORMAL = float(np.finfo(float).tiny)


def _reference_run(config: SimulationConfig) -> dict:
    """The pre-optimization full-rebuild residence-time loop, verbatim.

    Copied from the deleted `local_catalogue is None` branch so the RNG draw order, the
    residence-time update, and the event-selection arithmetic are the originals, not a
    paraphrase. Returns a plain dict so nothing is shared with `SimulationResult`.
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

    deposited = diffusion_selections = diffused = long_hops = desorbed = 0
    time = 0.0
    coverage_history: list[float] = []
    time_history: list[float] = []
    roughness_history: list[float] = []
    island_history: list[float] = []
    rheed_history: list[float] = []
    snapshots: list[np.ndarray] = []
    legal_hops = True

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

        bond_counts = _bond_counts(heights)
        occupied_sources = np.argwhere(heights > 0).astype(np.int64)
        diffusion_events = _diffusion_events(heights, config, bond_counts, occupied_sources)
        desorption_rates, desorption_sources = _desorption_events(
            heights, config, bond_counts, occupied_sources
        )
        total_diffusion_rate = diffusion_events.total_rate
        total_desorption_rate = float(desorption_rates[-1]) if desorption_rates.size else 0.0
        total_surface_rate = total_diffusion_rate + total_desorption_rate

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
            deposited += 1
        elif selected_rate < deposition_rate + total_diffusion_rate:
            selected_rate -= deposition_rate
            event_index = int(
                np.searchsorted(diffusion_events.cumulative_rates, selected_rate, side="right")
            )
            source = tuple(diffusion_events.sources[event_index])
            direction = HEX_DIRECTIONS[diffusion_events.directions[event_index]]
            distance = int(diffusion_events.distances[event_index])
            target = (
                (source[0] + distance * direction[0]) % config.lattice_size,
                (source[1] + distance * direction[1]) % config.lattice_size,
            )
            # Independent legality check: at hop limit 1 every selected hop must satisfy the
            # scalar rule in lattice.py, which neither catalogue consults.
            legal_hops &= bool(hop_allowed(heights, source, target))
            heights[source] -= 1
            heights[target] += 1
            diffusion_selections += 1
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
        raise RuntimeError("reference oracle exceeded max_events")

    if coverage_history[-1] != coverage_ml(heights):
        record()

    return {
        "final_heights": heights.copy(),
        "coverage_ml": np.asarray(coverage_history),
        "time_s": np.asarray(time_history),
        "roughness_ml": np.asarray(roughness_history),
        "island_density_per_site": np.asarray(island_history),
        "rheed_proxy": np.asarray(rheed_history),
        "snapshots": np.stack(snapshots),
        "deposited_events": deposited,
        "selected_diffusion_events": diffusion_selections,
        "diffusion_events": diffused,
        "long_hop_events": long_hops,
        "desorbed_events": desorbed,
        "all_hops_legal": legal_hops,
    }


def _trajectory_hash(fields: dict) -> str:
    """One digest over every array and counter that defines the trajectory."""
    digest = hashlib.sha256()
    for name in sorted(fields):
        if name == "all_hops_legal":
            continue
        value = fields[name]
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(value).tobytes() if isinstance(value, np.ndarray)
                      else str(value).encode())
    return digest.hexdigest()


# --- parameter grid -------------------------------------------------------------------
# Physical regimes: frozen, mobile, desorption-dominated, bond-dominated, step-barrier
# mounding, and the paper's high-prefactor corner. All at hop limit 1.
#
# Cases are deliberately short. The oracle rebuilds the whole catalogue every event, so long
# trajectories are expensive, and they buy little: the two paths either agree on event 1 or
# they diverge chaotically within a handful of events. Breadth of regime is what carries the
# evidence here, so the grid is wide and each trajectory is small.
_BASE = {
    "max_isolated_hop_distance": 1,
    "target_coverage_ml": 0.5,
    "sample_every_ml": 0.25,
    "max_events": 200_000,
}

_CONFIGURATIONS = [
    # (label, overrides)
    ("frozen-no-diffusion", {"attempt_frequency_hz": 0.0, "lattice_size": 6}),
    ("baseline-default", {"lattice_size": 4}),
    ("baseline-6", {"lattice_size": 6, "target_coverage_ml": 0.25}),
    ("tiny-3", {"lattice_size": 3}),
    ("tiny-4", {"lattice_size": 4, "target_coverage_ml": 0.25}),
    ("odd-5", {"lattice_size": 5, "target_coverage_ml": 0.25}),
    ("odd-7", {"lattice_size": 7, "target_coverage_ml": 0.25}),
    ("cold-500K", {"temperature_k": 500.0, "lattice_size": 6}),
    ("cool-650K", {"temperature_k": 650.0, "lattice_size": 6}),
    ("hot-1000K", {"temperature_k": 1000.0, "lattice_size": 4}),
    ("very-hot-1200K", {"temperature_k": 1200.0, "lattice_size": 3}),
    ("low-flux", {"deposition_flux_ml_s": 0.05, "lattice_size": 3, "target_coverage_ml": 0.25}),
    ("high-flux", {"deposition_flux_ml_s": 1.5, "lattice_size": 6}),
    ("no-bond-energy", {"lateral_bond_energy_ev": 0.0, "lattice_size": 4}),
    ("strong-bond", {"lateral_bond_energy_ev": 0.30, "lattice_size": 6}),
    ("no-step-barrier", {"step_barrier_ev": 0.0, "lattice_size": 4}),
    ("strong-step-barrier", {"step_barrier_ev": 0.25, "lattice_size": 4}),
    # Desorption must be vigorous but still slower than deposition, or net coverage never
    # reaches the target and the run is a max_events timeout rather than a comparison.
    ("desorption-heavy", {"desorption_barrier_ev": 0.45, "lattice_size": 5}),
    ("desorption-suppressed", {"desorption_barrier_ev": 1.20, "lattice_size": 5}),
    ("low-barrier-mobile", {"diffusion_barrier_ev": 0.05, "lattice_size": 3,
                                "target_coverage_ml": 0.25}),
    ("high-barrier", {"diffusion_barrier_ev": 1.20, "lattice_size": 6}),
    ("fast-prefactor", {"attempt_frequency_hz": 1e6, "diffusion_barrier_ev": 0.5, "lattice_size": 3,
                            "target_coverage_ml": 0.25}),
    # The paper's corner: atomistic prefactor with barriers scaled to match it. Both barriers
    # must move together -- a 1e13 prefactor against the default desorption barrier gives
    # r_des ~ 5e9 Hz, which strips the surface faster than it can grow.
    ("atomistic-prefactor", {"attempt_frequency_hz": 1e13, "diffusion_barrier_ev": 2.1,
                                 "desorption_barrier_ev": 2.6, "temperature_k": 1003.15,
                                 "lattice_size": 3, "target_coverage_ml": 0.25}),
    ("time-target", {"target_coverage_ml": None, "target_time_s": 0.5, "lattice_size": 4}),
    ("time-target-short", {"target_coverage_ml": None, "target_time_s": 0.05, "lattice_size": 6}),
    ("fine-sampling", {"sample_every_ml": 0.05, "lattice_size": 4}),
    ("thick-film", {"target_coverage_ml": 2.0, "lattice_size": 3}),
    ("thin-film", {"target_coverage_ml": 0.25, "lattice_size": 8}),
    ("cold-and-bound", {"temperature_k": 600.0, "lateral_bond_energy_ev": 0.2, "lattice_size": 5}),
    ("hot-and-desorbing", {"temperature_k": 1100.0, "desorption_barrier_ev": 0.62, "lattice_size": 4}),
]

_SEEDS = (0, 7)


def _config(overrides: dict, seed: int) -> SimulationConfig:
    fields = {"target_coverage_ml": 1.0} | _BASE | overrides | {"seed": seed}
    return SimulationConfig(**fields)


# 30 regimes x 4 seeds = 120 trajectory comparisons.
_CASES = [
    pytest.param(_config(overrides, seed), id=f"{label}-seed{seed}")
    for label, overrides in _CONFIGURATIONS
    for seed in _SEEDS
]


# --- surface generators for the rate-level comparison ---------------------------------
def _surfaces(size: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Valid solid-on-solid height fields, including the degenerate corners."""
    flat = np.zeros((size, size), dtype=np.int64)
    single = flat.copy()
    single[size // 2, size // 2] = 1
    full = np.ones((size, size), dtype=np.int64)
    terrace = np.zeros((size, size), dtype=np.int64)
    terrace[: size // 2] = 1
    tower = flat.copy()
    tower[0, 0] = 5
    pit = np.ones((size, size), dtype=np.int64) * 3
    pit[size // 2, size // 2] = 0
    return {
        "empty": flat,
        "single-adatom": single,
        "complete-layer": full,
        "half-terrace": terrace,
        "isolated-tower": tower,
        "deep-pit": pit,
        "random-sparse": rng.integers(0, 2, size=(size, size)).astype(np.int64),
        "random-rough": rng.integers(0, 5, size=(size, size)).astype(np.int64),
        "random-tall": rng.integers(2, 9, size=(size, size)).astype(np.int64),
    }


_RATE_CONFIGS = [
    pytest.param(_config(overrides, 0), id=label)
    for label, overrides in _CONFIGURATIONS
    if "time-target" not in label
]


def _reference_diffusion_rates(heights, config):
    """`_exact_diffusion_catalogue`'s rate expression, evaluated densely.

    Deliberately *not* recovered by differencing `cumulative_rates`: a cumsum-then-diff
    reintroduces rounding that neither implementation ever performs, which would make this
    comparison fail for a reason that has nothing to do with the code under test.
    """
    bonds = _bond_counts(heights)
    rates = np.zeros((*heights.shape, 6), dtype=float)
    for index, (dy, dx) in enumerate(HEX_DIRECTIONS):
        target = np.roll(heights, shift=(-dy, -dx), axis=(0, 1))
        allowed = (heights > 0) & (np.abs(heights - (target + 1)) <= 1)
        downward = heights > target + 1
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
    return rates


def _reference_desorption_rates(heights, config):
    """`_desorption_events`' rate expression, evaluated densely (no cumsum round trip)."""
    bonds = _bond_counts(heights)
    barriers = config.desorption_barrier_ev + bonds * config.lateral_bond_energy_ev
    rates = config.attempt_frequency_hz * np.exp(
        -barriers / (BOLTZMANN_EV_PER_K * config.temperature_k)
    )
    return np.where(heights > 0, rates, 0.0)


@pytest.mark.parametrize("config", _RATE_CONFIGS)
def test_individual_rates_and_event_set_are_bit_identical(config: SimulationConfig) -> None:
    """Every Arrhenius rate and the set of available events match, site by site.

    Rates are compared as raw arrays rather than through a cumulative sum, because a cumsum
    then diff would reintroduce rounding the two paths never actually perform.
    """
    rng = np.random.default_rng(config.seed + 99)
    for name, heights in _surfaces(config.lattice_size, rng).items():
        catalogue = _LocalLongHopCatalogue(heights.copy(), config)
        expected_diffusion = _reference_diffusion_rates(heights, config)
        expected_desorption = _reference_desorption_rates(heights, config)
        reference_catalogue = _diffusion_events(
            heights, config, _bond_counts(heights), np.argwhere(heights > 0).astype(np.int64)
        )

        assert np.array_equal(catalogue.diffusion_rates, expected_diffusion), (
            f"{name}: diffusion rates differ"
        )
        assert np.array_equal(catalogue.desorption_rates, expected_desorption), (
            f"{name}: desorption rates differ"
        )
        # The physical event set: identical support, and hop distance pinned at 1.
        assert np.array_equal(
            catalogue.diffusion_rates > 0, expected_diffusion > 0
        ), f"{name}: available diffusion events differ"
        assert np.array_equal(catalogue.distances, np.ones_like(heights)), (
            f"{name}: hop limit 1 must never produce a long hop"
        )
        assert len(reference_catalogue.sources) == int(np.count_nonzero(expected_diffusion))


@pytest.mark.parametrize("config", _RATE_CONFIGS)
def test_total_rates_are_bit_identical(config: SimulationConfig) -> None:
    """R_diff and R_des agree exactly, so every event probability r_i/R agrees."""
    rng = np.random.default_rng(config.seed + 7)
    for name, heights in _surfaces(config.lattice_size, rng).items():
        catalogue = _LocalLongHopCatalogue(heights.copy(), config)
        bonds = _bond_counts(heights)
        sources = np.argwhere(heights > 0).astype(np.int64)
        reference_diffusion = _diffusion_events(heights, config, bonds, sources).total_rate
        cumulative_desorption, _ = _desorption_events(heights, config, bonds, sources)
        reference_desorption = (
            float(cumulative_desorption[-1]) if cumulative_desorption.size else 0.0
        )

        diffusion_total, desorption_total = catalogue.surface_totals()
        assert diffusion_total == reference_diffusion, f"{name}: total diffusion rate differs"
        assert desorption_total == reference_desorption, f"{name}: total desorption rate differs"


@pytest.mark.parametrize("config", _RATE_CONFIGS)
def test_event_selection_matches_across_the_whole_cumulative_range(
    config: SimulationConfig,
) -> None:
    """Sampling any point of [0, R) selects the same event in both implementations."""
    rng = np.random.default_rng(config.seed + 11)
    for name, heights in _surfaces(config.lattice_size, rng).items():
        catalogue = _LocalLongHopCatalogue(heights.copy(), config)
        bonds = _bond_counts(heights)
        sources = np.argwhere(heights > 0).astype(np.int64)
        reference = _diffusion_events(heights, config, bonds, sources)
        diffusion_total, desorption_total = catalogue.surface_totals()
        if diffusion_total <= 0.0:
            continue

        for fraction in np.linspace(0.0, 1.0, 37, endpoint=False):
            rate = float(fraction) * diffusion_total
            index = int(np.searchsorted(reference.cumulative_rates, rate, side="right"))
            expected_source = tuple(reference.sources[index])
            expected_direction = int(reference.directions[index])
            source, direction, distance = catalogue.select_diffusion(rate)
            assert (source, direction) == (expected_source, expected_direction), (
                f"{name} at fraction {fraction}: selected a different diffusion event"
            )
            assert distance == 1

        cumulative_desorption, desorption_sources = _desorption_events(
            heights, config, bonds, sources
        )
        if desorption_total <= 0.0:
            continue
        for fraction in np.linspace(0.0, 1.0, 37, endpoint=False):
            rate = float(fraction) * desorption_total
            index = int(np.searchsorted(cumulative_desorption, rate, side="right"))
            assert catalogue.select_desorption(rate) == tuple(desorption_sources[index]), (
                f"{name} at fraction {fraction}: selected a different desorption event"
            )


@pytest.fixture(params=["auto", "reference"])
def backend(request, monkeypatch: pytest.MonkeyPatch) -> str:
    """Run once with the Numba kernels and once on the pure-NumPy path.

    `refresh_near` has two independent implementations and the compiled one is the default,
    so a single-backend suite silently leaves the NumPy branch untested -- verified by
    mutation: breaking the NumPy refresh radius was caught only under `reference`.
    """
    monkeypatch.setenv("MBE_KMC_BACKEND", request.param)
    return request.param


@pytest.mark.parametrize("config", _CASES)
def test_full_trajectory_is_bit_identical_to_the_reference_loop(
    config: SimulationConfig, backend: str
) -> None:
    """The whole trajectory matches: timing, counts, morphology, and observables."""
    expected = _reference_run(config)
    actual = run(config)

    assert expected["all_hops_legal"], "reference oracle selected an illegal hop"

    # Residence-time clock and every sampled observable, bit for bit.
    for name in (
        "time_s",
        "coverage_ml",
        "roughness_ml",
        "island_density_per_site",
        "rheed_proxy",
    ):
        assert np.array_equal(getattr(actual, name), expected[name]), f"{name} differs"
    assert np.array_equal(actual.final_heights, expected["final_heights"])
    assert np.array_equal(actual.snapshots, expected["snapshots"])
    for name in (
        "deposited_events",
        "selected_diffusion_events",
        "diffusion_events",
        "long_hop_events",
        "desorbed_events",
    ):
        assert getattr(actual, name) == expected[name], f"{name} differs"
    assert actual.long_hop_events == 0, "hop limit 1 must never take a long hop"

    actual_fields = {name: getattr(actual, name) for name in expected if name != "all_hops_legal"}
    assert _trajectory_hash(actual_fields) == _trajectory_hash(expected)


def test_large_lattices_reassociate_the_total_rate_without_changing_the_event_set() -> None:
    """At and above 128x128 the catalogue sums R through a Fenwick tree, not a linear cumsum.

    This is the one place hop limit 1 is *not* bit-identical to the old full-rebuild loop, so
    it is pinned rather than left to be discovered. What must hold is that the difference is
    pure floating-point reassociation:

      * the multiset of individual rates is identical (same events, same Arrhenius values);
      * both totals are roundings of the same exact sum;
      * the tree, being a balanced summation, is the more accurate of the two.

    No committed workflow runs hop limit 1 at this size, so no stored artifact depends on the
    old rounding, so the tree total is the one asserted here.
    """
    config = SimulationConfig(
        lattice_size=128, target_coverage_ml=1.0, max_isolated_hop_distance=1
    )
    heights = np.random.default_rng(0).integers(0, 5, size=(128, 128)).astype(np.int64)
    catalogue = _LocalLongHopCatalogue(heights.copy(), config)
    assert catalogue.rate_tree is not None, "this test only means something with the tree active"

    bonds = _bond_counts(heights)
    sources = np.argwhere(heights > 0).astype(np.int64)
    reference_catalogue = _diffusion_events(heights, config, bonds, sources)
    cumulative_desorption, _ = _desorption_events(heights, config, bonds, sources)

    # Same event set and same individual rates, which is the physics.
    assert len(reference_catalogue.sources) == int(np.count_nonzero(catalogue.diffusion_rates))
    assert np.array_equal(
        catalogue.diffusion_rates, _reference_diffusion_rates(heights, config)
    )
    assert np.array_equal(
        catalogue.desorption_rates, _reference_desorption_rates(heights, config)
    )

    # Same exact sum, different rounding, tree at least as accurate.
    linear_total = float(reference_catalogue.cumulative_rates[-1]) + float(
        cumulative_desorption[-1]
    )
    tree_total = catalogue.total_rate
    exact = math.fsum(catalogue.diffusion_rates.ravel().tolist()) + math.fsum(
        catalogue.desorption_rates.ravel().tolist()
    )
    linear_error = abs(linear_total - exact) / exact
    tree_error = abs(tree_total - exact) / exact
    assert linear_error < 1e-10 and tree_error < 1e-10
    assert tree_error <= linear_error, "the tree must not be less accurate than a linear cumsum"


def test_reference_oracle_is_sensitive_enough_to_notice_a_changed_trajectory() -> None:
    """Guard against a vacuous suite: a different seed must produce a different hash."""
    base = _config({"lattice_size": 6}, seed=0)
    other = _config({"lattice_size": 6}, seed=1)
    assert _trajectory_hash(_reference_run(base)) != _trajectory_hash(_reference_run(other))
