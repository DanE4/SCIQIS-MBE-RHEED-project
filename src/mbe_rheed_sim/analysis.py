"""Small ensemble-analysis helpers for reproducible parameter studies."""

from collections.abc import Iterable
from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from mbe_rheed_sim.config import SimulationConfig
from mbe_rheed_sim.kmc import run


def oscillation_amplitude(values: NDArray[np.float64]) -> float:
    """Robust half peak-to-trough amplitude using the 5th and 95th percentiles."""
    if values.ndim != 1 or values.size < 2:
        raise ValueError("oscillation amplitude requires a one-dimensional trace")
    low, high = np.quantile(values, (0.05, 0.95))
    return float((high - low) / 2.0)


def rheed_proxy_ensemble(
    config: SimulationConfig,
    seeds: Iterable[int],
    *,
    points: int = 201,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Interpolate seeded RHEED-proxy runs onto one coverage or time grid."""
    seed_values = tuple(seeds)
    if not seed_values or points < 2:
        raise ValueError("at least one seed and two interpolation points are required")
    if config.target_time_s is not None and config.target_coverage_ml is None:
        target = config.target_time_s
        coordinate = "time_s"
    elif config.target_coverage_ml is not None and config.target_time_s is None:
        target = config.target_coverage_ml
        coordinate = "coverage_ml"
    else:
        raise ValueError("ensemble config must have exactly one simulation target")

    grid = np.linspace(0.0, target, points)
    traces = []
    for seed in seed_values:
        result = run(replace(config, seed=seed))
        traces.append(np.interp(grid, getattr(result, coordinate), result.rheed_proxy))
    return grid, np.vstack(traces)
