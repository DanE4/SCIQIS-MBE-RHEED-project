import numpy as np
import pytest

from mbe_rheed_sim import SimulationConfig, run
from mbe_rheed_sim.kmc import (
    _desorption_events,
    _diffusion_events,
    _LocalLongHopCatalogue,
    _RateTree,
)
from mbe_rheed_sim.lattice import (
    HEX_DIRECTIONS,
    deposit,
    hop_allowed,
    lateral_bonds,
    long_hop,
    open_terrace_hop_distance,
)
from mbe_rheed_sim.rates import diffusion_rate


def test_deposition_only_limit_and_invariants() -> None:
    config = SimulationConfig(
        lattice_size=4,
        target_coverage_ml=1.0,
        attempt_frequency_hz=0.0,
        sample_every_ml=0.25,
        seed=7,
    )
    result = run(config)
    assert result.deposited_events == 16
    assert result.diffusion_events == 0
    assert result.desorbed_events == 0
    assert int(result.final_heights.sum()) == 16
    assert np.all(result.final_heights >= 0)
    assert result.coverage_ml[-1] == 1.0


def test_rate_tree_updates_and_selects_weighted_sites() -> None:
    tree = _RateTree(np.array([1.0, 2.0, 3.0]))
    assert tree.total_rate == pytest.approx(6.0)
    assert tree.select(0.5) == pytest.approx((0, 0.5))
    assert tree.select(1.5) == pytest.approx((1, 0.5))

    tree.update(np.array([1]), np.array([0.0]))
    assert tree.total_rate == pytest.approx(4.0)
    assert tree.select(1.5) == pytest.approx((2, 0.5))
    with pytest.raises(ValueError):
        tree.select(4.0)


def test_diffusion_desorption_and_net_mass() -> None:
    result = run(
        SimulationConfig(
            lattice_size=4,
            target_coverage_ml=0.5,
            diffusion_barrier_ev=0.2,
            desorption_barrier_ev=0.45,
            sample_every_ml=0.25,
            seed=3,
        )
    )
    assert result.diffusion_events > 0
    assert result.desorbed_events > 0
    assert int(result.final_heights.sum()) == result.deposited_events - result.desorbed_events


def test_isolated_adatom_acceleration_preserves_mass() -> None:
    result = run(
        SimulationConfig(
            lattice_size=7,
            target_coverage_ml=0.25,
            max_isolated_hop_distance=3,
            sample_every_ml=0.25,
            seed=4,
        )
    )
    assert result.long_hop_events > 0
    assert result.selected_diffusion_events >= result.long_hop_events
    assert result.diffusion_events >= result.long_hop_events * 4
    assert int(result.final_heights.sum()) == result.deposited_events - result.desorbed_events


def test_vectorized_long_hop_catalogue_matches_scalar_rules() -> None:
    config = SimulationConfig(lattice_size=7, max_isolated_hop_distance=3)
    heights = np.zeros((7, 7), dtype=np.int64)
    heights[1, 1] = heights[3, 3] = heights[3, 4] = 1
    heights[5, 5] = 2
    catalogue = _diffusion_events(heights, config)
    rates = np.diff(catalogue.cumulative_rates, prepend=0)
    expected_count = sum(
        6
        if open_terrace_hop_distance(heights, int(y), int(x), 3) > 1
        else sum(
            hop_allowed(
                heights,
                (int(y), int(x)),
                ((int(y) + dy) % 7, (int(x) + dx) % 7),
            )
            for dy, dx in HEX_DIRECTIONS
        )
        for y, x in zip(*np.nonzero(heights), strict=True)
    )
    assert len(rates) == expected_count

    for source_array, direction_index, distance, rate in zip(
        catalogue.sources,
        catalogue.directions,
        catalogue.distances,
        rates,
        strict=True,
    ):
        source = tuple(source_array)
        direction = HEX_DIRECTIONS[direction_index]
        scalar_distance = open_terrace_hop_distance(heights, *source, 3)
        assert distance == scalar_distance
        step_barrier = 0.0
        if distance == 1:
            target = tuple((source[i] + direction[i]) % 7 for i in (0, 1))
            assert hop_allowed(heights, source, target)
            if heights[source] > heights[target] + 1:
                step_barrier = config.step_barrier_ev
        expected = diffusion_rate(
            config.attempt_frequency_hz,
            config.diffusion_barrier_ev,
            config.lateral_bond_energy_ev,
            lateral_bonds(heights, *source),
            config.temperature_k,
            step_barrier,
        ) / (6 * distance**2)
        assert rate == pytest.approx(expected)


def test_local_long_hop_catalogue_matches_full_rebuild_after_changes() -> None:
    config = SimulationConfig(lattice_size=7, max_isolated_hop_distance=3)
    heights = np.zeros((7, 7), dtype=np.int64)
    heights[1, 1] = heights[3, 3] = heights[3, 4] = 1
    local = _LocalLongHopCatalogue(heights, config)

    def assert_matches_full_rebuild() -> None:
        full = _diffusion_events(heights, config)
        full_rates = np.diff(full.cumulative_rates, prepend=0)
        assert np.count_nonzero(local.diffusion_rates) == len(full_rates)
        for source, direction, distance, rate in zip(
            full.sources, full.directions, full.distances, full_rates, strict=True
        ):
            source_tuple = tuple(source)
            assert local.diffusion_rates[*source_tuple, direction] == pytest.approx(rate)
            assert local.distances[source_tuple] == distance

        full_desorption, sources = _desorption_events(heights, config)
        expected_desorption = np.zeros_like(heights, dtype=float)
        expected_desorption[tuple(sources.T)] = np.diff(full_desorption, prepend=0)
        assert local.desorption_rates == pytest.approx(expected_desorption)
        assert local.rate_tree is None

    assert_matches_full_rebuild()
    distance = open_terrace_hop_distance(heights, 1, 1, 3)
    target = (1, (1 + distance) % 7)
    long_hop(heights, (1, 1), (0, 1), distance)
    local.refresh_near(((1, 1), target))
    assert_matches_full_rebuild()
    deposit(heights, 0, 0)
    local.refresh_near(((0, 0),))
    assert_matches_full_rebuild()


def test_large_local_catalogue_updates_rate_tree() -> None:
    config = SimulationConfig(lattice_size=128, max_isolated_hop_distance=3)
    heights = np.zeros((128, 128), dtype=np.int64)
    local = _LocalLongHopCatalogue(heights, config)
    assert local.rate_tree is not None

    deposit(heights, 0, 0)
    local.refresh_near(((0, 0),))
    assert local.total_rate == pytest.approx(
        local.diffusion_rates.sum() + local.desorption_rates.sum()
    )


def test_time_target_stops_at_requested_time() -> None:
    result = run(
        SimulationConfig(
            lattice_size=3,
            target_coverage_ml=None,
            target_time_s=0.25,
            attempt_frequency_hz=0,
            seed=2,
        )
    )
    assert result.time_s[-1] == 0.25
    assert result.coverage_ml[-1] == result.deposited_events / 9
