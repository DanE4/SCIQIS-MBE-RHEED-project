import numpy as np

from mbe_rheed_sim import SimulationConfig, run


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
    assert result.diffusion_events >= result.long_hop_events * 4
    assert int(result.final_heights.sum()) == result.deposited_events - result.desorbed_events


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
