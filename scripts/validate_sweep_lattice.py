"""Check the sweep's high-versus-low flux trend on a 24x24 lattice."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from mbe_rheed_sim import SimulationConfig
from mbe_rheed_sim.analysis import oscillation_amplitude, rheed_proxy_ensemble

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "runs" / "sweep_lattice_validation.json"
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


def mean_amplitude(config: SimulationConfig) -> tuple[float, float]:
    _, traces = rheed_proxy_ensemble(config, SEEDS)
    amplitudes = np.array([oscillation_amplitude(trace) for trace in traces])
    return float(amplitudes.mean()), float(amplitudes.std())


def main() -> None:
    comparisons = []
    for temperature in TEMPERATURES_K:
        low_mean, low_std = mean_amplitude(
            replace(BASE, temperature_k=temperature, deposition_flux_ml_s=LOW_FLUX_ML_S)
        )
        high_mean, high_std = mean_amplitude(
            replace(BASE, temperature_k=temperature, deposition_flux_ml_s=HIGH_FLUX_ML_S)
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
        "config": BASE.as_dict(),
        "seeds": SEEDS,
        "low_flux_ml_s": LOW_FLUX_ML_S,
        "high_flux_ml_s": HIGH_FLUX_ML_S,
        "comparisons": comparisons,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
