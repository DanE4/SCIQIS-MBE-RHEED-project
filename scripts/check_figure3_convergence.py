"""Check one oscillation-scale Figure 3 window across practical lattice sizes."""

import json
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mbe_rheed_sim.analysis import (
    oscillation_amplitude,
    rheed_oscillation_metrics,
    run_summary,
    successive_size_check,
)
from mbe_rheed_sim.paper import figure3_config, figure3_parameters
from mbe_rheed_sim.workflows import (
    artifact_root,
    parse_workflow_args,
    run_parallel,
    run_timed,
)

ROOT = Path(__file__).resolve().parents[1]
RATIO = 0.82
DURATION_S = 4.0
SIZES = (8, 16, 32)
SEEDS = (0, 1, 2)
TIME_S = np.linspace(0.0, DURATION_S, 101)
RELATIVE_TOLERANCE = 0.10


def main(
    *, workers: int = 4, seeds: tuple[int, ...] = SEEDS, sizes: tuple[int, ...] = SIZES
) -> None:
    if len(sizes) < 2 or any(size < 2 for size in sizes):
        raise ValueError("at least two lattice sizes of 2 or greater are required")
    output_root = artifact_root(ROOT)
    run_dir = output_root / "outputs" / "runs"
    figure_dir = output_root / "outputs" / "figures"
    summaries = []
    parameters = figure3_parameters(RATIO)
    figure, axes = plt.subplots(1, len(sizes), figsize=(4 * len(sizes), 3.5), sharey=True)
    for axis, size in zip(axes, sizes, strict=True):
        results = []
        runs = []
        configurations = [
            figure3_config(RATIO, lattice_size=size, duration_s=DURATION_S, seed=seed)
            for seed in seeds
        ]
        timed_results = run_parallel(
            run_timed,
            configurations,
            workers=workers,
            description=f"Figure 3 convergence {size}x{size}",
        )
        for (result, elapsed), seed in zip(timed_results, seeds, strict=True):
            predicted_coverage = result.time_s * parameters.predicted_growth_rate_ml_s
            metrics = rheed_oscillation_metrics(predicted_coverage, result.rheed_proxy)
            results.append(result)
            runs.append(
                run_summary(result, seed, elapsed)
                | {
                    "trace": {
                        "time_s": result.time_s.tolist(),
                        "predicted_coverage_ml": predicted_coverage.tolist(),
                        "rheed_proxy": result.rheed_proxy.tolist(),
                    },
                    "oscillation_metrics": asdict(metrics),
                }
            )
        traces = np.vstack(
            [np.interp(TIME_S, result.time_s, result.rheed_proxy) for result in results]
        )
        roughness = np.array([result.roughness_ml[-1] for result in results])
        amplitudes = np.array([oscillation_amplitude(trace) for trace in traces])
        detrended = np.array(
            [run_summary["oscillation_metrics"]["detrended_amplitude"] for run_summary in runs]
        )
        mean = traces.mean(axis=0)
        std = traces.std(axis=0)
        summaries.append(
            {
                "lattice_size": size,
                "seed_count": len(seeds),
                "elapsed_s": float(sum(run_summary["elapsed_s"] for run_summary in runs)),
                "roughness_mean_ml": float(roughness.mean()),
                "roughness_std_ml": float(roughness.std()),
                "proxy_amplitude_mean": float(amplitudes.mean()),
                "proxy_amplitude_std": float(amplitudes.std()),
                "detrended_amplitude_mean": float(detrended.mean()),
                "detrended_amplitude_std": float(detrended.std(ddof=1)),
                "runs": runs,
            }
        )
        axis.plot(TIME_S, mean, color="tab:blue")
        axis.fill_between(
            TIME_S,
            np.clip(mean - std, 0, 1),
            np.clip(mean + std, 0, 1),
            color="tab:blue",
            alpha=0.22,
        )
        axis.set(title=f"{size}x{size}", xlabel="time (s)", ylim=(0, 1.03))
    axes[0].set_ylabel(r"$1-S_d$")
    figure.suptitle("Figure 3 ratio 0.82: 4 s size sensitivity (mean +/- SD, 3 seeds)")

    successive_checks = []
    for smaller, larger in pairwise(summaries):
        successive_checks.append(
            successive_size_check(
                smaller["lattice_size"],
                larger["lattice_size"],
                smaller["detrended_amplitude_mean"],
                larger["detrended_amplitude_mean"],
                smaller["detrended_amplitude_std"],
                larger["detrended_amplitude_std"],
                len(seeds),
                relative_tolerance=RELATIVE_TOLERANCE,
            )
        )

    output = {
        "nominal_ga_n_ratio": RATIO,
        "duration_s": DURATION_S,
        "seeds": seeds,
        "effective_workers": min(workers, len(seeds)),
        "paper_parameters": asdict(parameters),
        "principal_observable": {
            "name": "detrended_amplitude",
            "relative_tolerance": RELATIVE_TOLERANCE,
            "acceptance": (
                "absolute successive-size mean difference plus 1.96 pooled standard errors "
                "must be no more than 10% of the larger-size mean"
            ),
        },
        "successive_size_checks": successive_checks,
        "note": "pass --sizes 8,16,32,64 to include the expensive 64x64 point",
        "sizes": summaries,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "figure3_convergence.json").write_text(json.dumps(output, indent=2) + "\n")
    figure.savefig(figure_dir / "figure3_convergence.png", dpi=160)
    plt.close(figure)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main(**parse_workflow_args(seeds=SEEDS, sizes=SIZES))
