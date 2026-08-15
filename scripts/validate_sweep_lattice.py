"""Check the sweep's high-versus-low flux trend on a 24x24 lattice."""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from mbe_rheed_sim import SimulationConfig, run
from mbe_rheed_sim.analysis import oscillation_amplitude
from mbe_rheed_sim.workflows import artifact_root, parse_int_values, resolve_workers, run_parallel

ROOT = Path(__file__).resolve().parents[1]
TEMPERATURES_K = (700.0, 850.0, 1_000.0)
LOW_FLUX_ML_S = 0.25
HIGH_FLUX_ML_S = 0.75
SEEDS = (0, 1, 2)
BASE = SimulationConfig(
    lattice_size=24,
    target_coverage_ml=2.0,
    max_isolated_hop_distance=3,
    sample_every_ml=0.05,
)


def mean_amplitude(results) -> tuple[float, float]:
    coverage = np.linspace(0.0, float(BASE.target_coverage_ml), 201)
    traces = np.vstack(
        [np.interp(coverage, result.coverage_ml, result.rheed_proxy) for result in results]
    )
    amplitudes = np.array([oscillation_amplitude(trace) for trace in traces])
    return float(amplitudes.mean()), float(amplitudes.std())


def main(*, workers: int = 4, seeds: tuple[int, ...] = SEEDS) -> None:
    configurations = [
        replace(BASE, temperature_k=temperature, deposition_flux_ml_s=flux, seed=seed)
        for temperature in TEMPERATURES_K
        for flux in (LOW_FLUX_ML_S, HIGH_FLUX_ML_S)
        for seed in seeds
    ]
    results = run_parallel(
        run,
        configurations,
        workers=workers,
        description="cross-lattice sweep validation",
    )
    comparisons = []
    for index, temperature in enumerate(TEMPERATURES_K):
        start = index * 2 * len(seeds)
        low_mean, low_std = mean_amplitude(results[start : start + len(seeds)])
        high_mean, high_std = mean_amplitude(
            results[start + len(seeds) : start + 2 * len(seeds)]
        )
        comparisons.append(
            {
                "temperature_k": temperature,
                "low_flux_mean": low_mean,
                "low_flux_std": low_std,
                "high_flux_mean": high_mean,
                "high_flux_std": high_std,
                "high_minus_low": high_mean - low_mean,
            }
        )

    failed = [item for item in comparisons if item["high_minus_low"] <= 0]
    if failed:
        raise RuntimeError(f"high-flux amplitude trend did not survive at 24x24: {failed}")
    summary = {
        "config": asdict(BASE),
        "seeds": seeds,
        "effective_workers": min(workers, len(configurations)),
        "low_flux_ml_s": LOW_FLUX_ML_S,
        "high_flux_ml_s": HIGH_FLUX_ML_S,
        "comparisons": comparisons,
    }
    output = artifact_root(ROOT) / "outputs" / "runs" / "sweep_lattice_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int)
    parser.add_argument("--seeds")
    arguments = parser.parse_args()
    main(
        workers=resolve_workers(arguments.workers),
        seeds=parse_int_values(arguments.seeds, SEEDS),
    )
