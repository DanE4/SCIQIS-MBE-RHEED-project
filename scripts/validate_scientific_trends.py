"""Validate smoothing and step-barrier mounding trends over fixed seed ensembles."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from mbe_rheed_sim import SimulationConfig, run
from mbe_rheed_sim.workflows import artifact_root, parse_int_values, resolve_workers, run_parallel

ROOT = Path(__file__).resolve().parents[1]
SEEDS = tuple(range(5))
BASE = SimulationConfig(lattice_size=8, target_coverage_ml=2.0, sample_every_ml=0.1)


def main(*, workers: int = 4, seeds: tuple[int, ...] = SEEDS) -> None:
    configurations = [
        replace(config, seed=seed)
        for config in (
            replace(BASE, attempt_frequency_hz=0),
            replace(BASE, step_barrier_ev=0),
            BASE,
        )
        for seed in seeds
    ]
    results = run_parallel(
        run,
        configurations,
        workers=workers,
        description="scientific trend validation",
    )
    grouped = [
        [float(result.roughness_ml[-1]) for result in results[index : index + len(seeds)]]
        for index in range(0, len(results), len(seeds))
    ]
    regimes = {
        "deposition_only": grouped[0],
        "diffusion_no_step_barrier": grouped[1],
        "diffusion_with_step_barrier": grouped[2],
    }
    means = {name: float(np.mean(values)) for name, values in regimes.items()}
    if not (
        means["diffusion_no_step_barrier"] < means["deposition_only"]
        < means["diffusion_with_step_barrier"]
    ):
        raise RuntimeError(f"expected smoothing/mounding ordering changed: {means}")

    summary = {
        "config": BASE.as_dict(),
        "seeds": list(seeds),
        "effective_workers": min(resolve_workers(workers), len(configurations)),
        "final_roughness_ml": regimes,
        "mean_final_roughness_ml": means,
    }
    output = artifact_root(ROOT) / "outputs" / "runs" / "scientific_trends.json"
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
