"""Compare accelerated and exact KMC ensemble observables on a small lattice."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from mbe_rheed_sim import SimulationConfig, run

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "runs" / "acceleration_validation.json"
SEEDS = range(100)
BASE = SimulationConfig(
    lattice_size=7,
    target_coverage_ml=0.5,
    sample_every_ml=0.25,
)


def ensemble(maximum_hop: int) -> np.ndarray:
    return np.array(
        [
            (
                result.roughness_ml[-1],
                result.island_density_per_site[-1],
                result.rheed_proxy[-1],
            )
            for seed in SEEDS
            for result in [
                run(replace(BASE, seed=seed, max_isolated_hop_distance=maximum_hop))
            ]
        ]
    )


def main() -> None:
    exact = ensemble(1)
    accelerated = ensemble(3)
    exact_mean = exact.mean(axis=0)
    accelerated_mean = accelerated.mean(axis=0)
    exact_std = exact.std(axis=0)
    difference = np.abs(accelerated_mean - exact_mean)
    tolerance = 0.25 * exact_std
    if np.any(difference > tolerance):
        raise RuntimeError(
            f"acceleration validation failed: difference={difference}, tolerance={tolerance}"
        )

    names = ("roughness_ml", "island_density_per_site", "rheed_proxy")
    summary = {
        "lattice_size": BASE.lattice_size,
        "target_coverage_ml": BASE.target_coverage_ml,
        "seeds": len(SEEDS),
        "maximum_hop_distance": 3,
        "acceptance_tolerance": "absolute mean difference <= 0.25 exact-model standard deviation",
        "metrics": {
            name: {
                "exact_mean": float(exact_mean[index]),
                "accelerated_mean": float(accelerated_mean[index]),
                "exact_std": float(exact_std[index]),
                "absolute_difference": float(difference[index]),
                "tolerance": float(tolerance[index]),
            }
            for index, name in enumerate(names)
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
