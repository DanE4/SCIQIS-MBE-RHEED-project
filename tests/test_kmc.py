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
    assert int(result.final_heights.sum()) == 16
    assert np.all(result.final_heights >= 0)
    assert result.coverage_ml[-1] == 1.0


def test_diffusion_occurs_and_conserves_deposited_mass() -> None:
    result = run(
        SimulationConfig(
            lattice_size=4,
            target_coverage_ml=0.5,
            diffusion_barrier_ev=0.2,
            sample_every_ml=0.25,
            seed=3,
        )
    )
    assert result.diffusion_events > 0
    assert int(result.final_heights.sum()) == result.deposited_events
