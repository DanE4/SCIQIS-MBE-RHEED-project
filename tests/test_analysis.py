import numpy as np
import pytest

from mbe_rheed_sim import SimulationConfig
from mbe_rheed_sim.analysis import oscillation_amplitude, rheed_proxy_ensemble


def test_oscillation_amplitude_and_seed_ensemble() -> None:
    assert oscillation_amplitude(np.tile([0.0, 1.0], 50)) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        oscillation_amplitude(np.array([1.0]))

    grid, traces = rheed_proxy_ensemble(
        SimulationConfig(
            lattice_size=4,
            target_coverage_ml=0.5,
            attempt_frequency_hz=0,
            sample_every_ml=0.25,
        ),
        seeds=(1, 2, 3),
        points=5,
    )
    assert grid.shape == (5,)
    assert traces.shape == (3, 5)
    assert np.all((0 <= traces) & (traces <= 1))
