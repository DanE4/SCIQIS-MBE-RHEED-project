"""Compare accelerated and exact KMC ensemble observables on a small lattice."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from mbe_rheed_sim import SimulationConfig, run
from mbe_rheed_sim.workflows import (
    artifact_root,
    parse_workflow_args,
    run_parallel,
)

ROOT = Path(__file__).resolve().parents[1]
SEEDS = tuple(range(100))
BASE = SimulationConfig(
    lattice_size=7,
    target_coverage_ml=0.5,
    sample_every_ml=0.25,
)


def ensemble(results) -> np.ndarray:
    return np.array(
        [
            (
                result.roughness_ml[-1],
                result.island_density_per_site[-1],
                result.rheed_proxy[-1],
            )
            for result in results
        ]
    )


def main(*, workers: int = 4, seeds: tuple[int, ...] = SEEDS) -> None:
    configurations = [
        replace(BASE, seed=seed, max_isolated_hop_distance=maximum_hop)
        for maximum_hop in (1, 3)
        for seed in seeds
    ]
    results = run_parallel(
        run,
        configurations,
        workers=workers,
        description="exact/accelerated validation",
    )
    exact = ensemble(results[: len(seeds)])
    accelerated = ensemble(results[len(seeds) :])
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
        "seeds": list(seeds),
        "effective_workers": min(workers, len(configurations)),
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
    output = artifact_root(ROOT) / "outputs" / "runs" / "acceleration_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(**parse_workflow_args(seeds=SEEDS))
