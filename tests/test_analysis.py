import numpy as np
import pytest

from mbe_rheed_sim import SimulationConfig, run
from mbe_rheed_sim.analysis import (
    oscillation_amplitude,
    result_array_bytes,
    rheed_oscillation_metrics,
    run_summary,
    successive_size_check,
)


def test_oscillation_amplitude() -> None:
    assert oscillation_amplitude(np.tile([0.0, 1.0], 50)) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        oscillation_amplitude(np.array([1.0]))


def test_run_summary_reports_events_and_final_surface() -> None:
    result = run(
        SimulationConfig(lattice_size=4, target_coverage_ml=0.5, sample_every_ml=0.25)
    )
    summary = run_summary(result, seed=0, elapsed=1.5)

    assert (summary["seed"], summary["elapsed_s"]) == (0, 1.5)
    assert summary["result_array_bytes"] == result_array_bytes(result)
    assert summary["events"]["deposited"] == result.deposited_events
    assert summary["final"]["maximum_height_ml"] == int(result.final_heights.max())
    assert 0.0 <= summary["final"]["occupied_site_fraction"] <= 1.0


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
