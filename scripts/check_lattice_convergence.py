"""Measure small-lattice sensitivity for the current uncalibrated demonstration regime."""

import json
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mbe_rheed_sim import SimulationConfig, run
from mbe_rheed_sim.analysis import oscillation_amplitude

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "outputs" / "runs"
FIGURE_DIR = ROOT / "outputs" / "figures"
SIZES = (8, 16, 24)
SEEDS = (0, 1, 2)
BASE = SimulationConfig(target_coverage_ml=2.0, sample_every_ml=0.05, step_barrier_ev=0)


def main() -> None:
    roughness_means = []
    roughness_stds = []
    amplitude_means = []
    amplitude_stds = []
    for size in SIZES:
        results = [run(replace(BASE, lattice_size=size, seed=seed)) for seed in SEEDS]
        roughness = np.array([result.roughness_ml[-1] for result in results])
        amplitudes = np.array(
            [oscillation_amplitude(result.rheed_proxy) for result in results]
        )
        roughness_means.append(float(roughness.mean()))
        roughness_stds.append(float(roughness.std()))
        amplitude_means.append(float(amplitudes.mean()))
        amplitude_stds.append(float(amplitudes.std()))

    summary = {
        "base_config": BASE.as_dict(),
        "lattice_sizes": SIZES,
        "seeds": SEEDS,
        "roughness_mean_ml": roughness_means,
        "roughness_std_ml": roughness_stds,
        "proxy_amplitude_mean": amplitude_means,
        "proxy_amplitude_std": amplitude_stds,
    }
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "lattice_convergence.json").write_text(json.dumps(summary, indent=2) + "\n")

    figure, axes = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
    axes[0].errorbar(SIZES, roughness_means, yerr=roughness_stds, marker="o", capsize=4)
    axes[0].set(xlabel="lattice side", ylabel="final RMS roughness (ML)")
    axes[1].errorbar(SIZES, amplitude_means, yerr=amplitude_stds, marker="o", capsize=4)
    axes[1].set(xlabel="lattice side", ylabel="RHEED-proxy amplitude")
    figure.suptitle("Small-lattice sensitivity (mean +/- SD, 3 seeds)")
    figure.savefig(FIGURE_DIR / "lattice_convergence.png", dpi=160)
    plt.close(figure)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
