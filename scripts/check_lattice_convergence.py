"""Measure small-lattice sensitivity for the current uncalibrated demonstration regime."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mbe_rheed_sim import SimulationConfig
from mbe_rheed_sim.analysis import (
    oscillation_amplitude,
    result_array_bytes,
    rheed_oscillation_metrics,
    successive_size_check,
)
from mbe_rheed_sim.workflows import (
    artifact_root,
    parse_int_values,
    resolve_workers,
    run_parallel,
    run_timed,
)

ROOT = Path(__file__).resolve().parents[1]
SIZES = (8, 16, 24)
SEEDS = (0, 1, 2)
BASE = SimulationConfig(target_coverage_ml=2.0, sample_every_ml=0.05, step_barrier_ev=0)
RELATIVE_TOLERANCE = 0.10


def main(
    *, workers: int = 4, seeds: tuple[int, ...] = SEEDS, sizes: tuple[int, ...] = SIZES
) -> None:
    if any(size < 2 for size in sizes):
        raise ValueError("lattice sizes must be at least 2")
    output_root = artifact_root(ROOT)
    run_dir = output_root / "outputs" / "runs"
    figure_dir = output_root / "outputs" / "figures"
    roughness_means = []
    roughness_stds = []
    amplitude_means = []
    amplitude_stds = []
    detrended_means = []
    detrended_stds = []
    size_summaries = []
    for size in sizes:
        results = []
        runs = []
        configurations = [replace(BASE, lattice_size=size, seed=seed) for seed in seeds]
        timed_results = run_parallel(
            run_timed,
            configurations,
            workers=workers,
            description=f"generic convergence {size}x{size}",
        )
        for (result, elapsed), seed in zip(timed_results, seeds, strict=True):
            metrics = rheed_oscillation_metrics(result.coverage_ml, result.rheed_proxy)
            results.append(result)
            runs.append(
                {
                    "seed": seed,
                    "elapsed_s": elapsed,
                    "result_array_bytes": result_array_bytes(result),
                    "events": {
                        "deposited": result.deposited_events,
                        "selected_diffusion": result.selected_diffusion_events,
                        "equivalent_nearest_neighbor_hops": result.diffusion_events,
                        "long_hops": result.long_hop_events,
                        "desorbed": result.desorbed_events,
                    },
                    "trace": {
                        "coverage_ml": result.coverage_ml.tolist(),
                        "rheed_proxy": result.rheed_proxy.tolist(),
                    },
                    "oscillation_metrics": metrics.as_dict(),
                    "final": {
                        "rms_roughness_ml": float(result.roughness_ml[-1]),
                        "step_density": float(1.0 - result.rheed_proxy[-1]),
                        "mean_height_ml": float(result.final_heights.mean()),
                        "height_std_ml": float(result.final_heights.std()),
                        "minimum_height_ml": int(result.final_heights.min()),
                        "maximum_height_ml": int(result.final_heights.max()),
                        "occupied_site_fraction": float(np.mean(result.final_heights > 0)),
                    },
                }
            )
        roughness = np.array([result.roughness_ml[-1] for result in results])
        amplitudes = np.array([oscillation_amplitude(result.rheed_proxy) for result in results])
        detrended = np.array(
            [run_summary["oscillation_metrics"]["detrended_amplitude"] for run_summary in runs]
        )
        roughness_means.append(float(roughness.mean()))
        roughness_stds.append(float(roughness.std()))
        amplitude_means.append(float(amplitudes.mean()))
        amplitude_stds.append(float(amplitudes.std()))
        detrended_means.append(float(detrended.mean()))
        detrended_stds.append(float(detrended.std(ddof=1)))
        size_summaries.append(
            {
                "lattice_size": size,
                "seed_count": len(seeds),
                "elapsed_s": float(sum(run_summary["elapsed_s"] for run_summary in runs)),
                "runs": runs,
            }
        )

    successive_checks = []
    for index in range(1, len(sizes)):
        successive_checks.append(
            successive_size_check(
                sizes[index - 1],
                sizes[index],
                detrended_means[index - 1],
                detrended_means[index],
                detrended_stds[index - 1],
                detrended_stds[index],
                len(seeds),
                relative_tolerance=RELATIVE_TOLERANCE,
            )
        )

    summary = {
        "base_config": BASE.as_dict(),
        "lattice_sizes": sizes,
        "seeds": seeds,
        "effective_workers": min(resolve_workers(workers), len(seeds)),
        "roughness_mean_ml": roughness_means,
        "roughness_std_ml": roughness_stds,
        "proxy_amplitude_mean": amplitude_means,
        "proxy_amplitude_std": amplitude_stds,
        "principal_observable": {
            "name": "detrended_amplitude",
            "relative_tolerance": RELATIVE_TOLERANCE,
            "acceptance": (
                "absolute successive-size mean difference plus 1.96 pooled standard errors "
                "must be no more than 10% of the larger-size mean"
            ),
        },
        "detrended_amplitude_mean": detrended_means,
        "detrended_amplitude_std": detrended_stds,
        "successive_size_checks": successive_checks,
        "sizes": size_summaries,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "lattice_convergence.json").write_text(json.dumps(summary, indent=2) + "\n")

    figure, axes = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
    axes[0].errorbar(sizes, roughness_means, yerr=roughness_stds, marker="o", capsize=4)
    axes[0].set(xlabel="lattice side", ylabel="final RMS roughness (ML)")
    axes[1].errorbar(sizes, amplitude_means, yerr=amplitude_stds, marker="o", capsize=4)
    axes[1].set(xlabel="lattice side", ylabel="RHEED-proxy amplitude")
    figure.suptitle("Small-lattice sensitivity (mean +/- SD, 3 seeds)")
    figure.savefig(figure_dir / "lattice_convergence.png", dpi=160)
    plt.close(figure)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int)
    parser.add_argument("--seeds")
    parser.add_argument("--sizes")
    arguments = parser.parse_args()
    main(
        workers=resolve_workers(arguments.workers),
        seeds=parse_int_values(arguments.seeds, SEEDS),
        sizes=parse_int_values(arguments.sizes, SIZES),
    )
