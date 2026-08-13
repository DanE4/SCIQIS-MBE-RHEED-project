import numpy as np
import pytest

from mbe_rheed_sim import SimulationConfig
from mbe_rheed_sim.analysis import (
    oscillation_amplitude,
    rheed_oscillation_metrics,
    rheed_proxy_ensemble,
    successive_size_check,
)


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


def test_rheed_oscillation_metrics_distinguish_periodic_and_monotonic_traces() -> None:
    coverage = np.linspace(0.0, 6.0, 601)
    periodic = 0.75 + 0.2 * np.exp(-0.05 * coverage) * np.cos(2 * np.pi * coverage)
    metrics = rheed_oscillation_metrics(coverage, periodic)

    assert metrics.is_oscillatory
    assert metrics.peak_count >= 5
    assert metrics.period_ml == pytest.approx(1.0, abs=0.03)
    assert metrics.period_deviation_ml == pytest.approx(0.0, abs=0.03)
    assert metrics.spectral_power_fraction > 0.9
    assert metrics.damping_rate_per_ml is not None
    assert metrics.damping_rate_per_ml < 0

    monotonic = rheed_oscillation_metrics(coverage, np.linspace(0.0, 1.0, coverage.size))
    assert not monotonic.is_oscillatory
    assert monotonic.peak_count == 0
    assert monotonic.period_ml is None


def test_successive_size_check_includes_uncertainty_in_acceptance() -> None:
    resolved = successive_size_check(16, 32, 0.101, 0.100, 0.001, 0.001, 3)
    noisy = successive_size_check(16, 32, 0.101, 0.100, 0.02, 0.02, 3)

    assert resolved["passes"]
    assert not noisy["passes"]
    with pytest.raises(ValueError):
        successive_size_check(32, 16, 0.1, 0.1, 0.01, 0.01, 3)
