"""Validate smoothing and step-barrier mounding trends over fixed seed ensembles."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from mbe_rheed_sim import SimulationConfig, run

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "runs" / "scientific_trends.json"
SEEDS = range(5)
BASE = SimulationConfig(lattice_size=8, target_coverage_ml=2.0, sample_every_ml=0.1)


def roughnesses(config: SimulationConfig) -> list[float]:
    return [float(run(replace(config, seed=seed)).roughness_ml[-1]) for seed in SEEDS]


def main() -> None:
    regimes = {
        "deposition_only": roughnesses(replace(BASE, attempt_frequency_hz=0)),
        "diffusion_no_step_barrier": roughnesses(replace(BASE, step_barrier_ev=0)),
        "diffusion_with_step_barrier": roughnesses(BASE),
    }
    means = {name: float(np.mean(values)) for name, values in regimes.items()}
    if not (
        means["diffusion_no_step_barrier"] < means["deposition_only"]
        < means["diffusion_with_step_barrier"]
    ):
        raise RuntimeError(f"expected smoothing/mounding ordering changed: {means}")

    summary = {
        "config": BASE.as_dict(),
        "seeds": list(SEEDS),
        "final_roughness_ml": regimes,
        "mean_final_roughness_ml": means,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
