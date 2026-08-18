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
from mbe_rheed_sim.rheed import antiphase_grazing_angle_deg, specular_intensity
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
RELATIVE_TOLERANCE = 0.10
# The kinematic (00) intensity is recorded next to the proxy at the anti-phase condition,
# where layer filling actually modulates it. Same surfaces, same seeds, second observable.
GRAZING_ANGLE_DEG = antiphase_grazing_angle_deg()


def main(
    *,
    workers: int = 4,
    seeds: tuple[int, ...] = SEEDS,
    sizes: tuple[int, ...] = SIZES,
    duration_s: float = DURATION_S,
) -> None:
    if len(sizes) < 2 or any(size < 2 for size in sizes):
        raise ValueError("at least two lattice sizes of 2 or greater are required")
    # One common resampling grid, kept at ~25 points per second of the default window so a
    # longer run is not resolved more coarsely than the 4 s default it must be compared with.
    time_s = np.linspace(0.0, duration_s, round(25 * duration_s) + 1)
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
            figure3_config(RATIO, lattice_size=size, duration_s=duration_s, seed=seed)
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
            specular = specular_intensity(
                result.snapshots, grazing_angle_deg=GRAZING_ANGLE_DEG
            )
            specular_metrics = rheed_oscillation_metrics(predicted_coverage, specular)
            results.append(result)
            runs.append(
                run_summary(result, seed, elapsed)
                | {
                    "trace": {
                        "time_s": result.time_s.tolist(),
                        "predicted_coverage_ml": predicted_coverage.tolist(),
                        "rheed_proxy": result.rheed_proxy.tolist(),
                        "kinematic_specular": specular.tolist(),
                    },
                    "oscillation_metrics": asdict(metrics),
                    "kinematic_specular_metrics": asdict(specular_metrics),
                }
            )
        traces = np.vstack(
            [np.interp(time_s, result.time_s, result.rheed_proxy) for result in results]
        )
        roughness = np.array([result.roughness_ml[-1] for result in results])
        amplitudes = np.array([oscillation_amplitude(trace) for trace in traces])
        detrended = np.array(
            [run_summary["oscillation_metrics"]["detrended_amplitude"] for run_summary in runs]
        )
        specular_detrended = np.array(
            [
                run_summary["kinematic_specular_metrics"]["detrended_amplitude"]
                for run_summary in runs
            ]
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
                "specular_detrended_amplitude_mean": float(specular_detrended.mean()),
                "specular_detrended_amplitude_std": float(specular_detrended.std(ddof=1)),
                "runs": runs,
            }
        )
        axis.plot(time_s, mean, color="tab:blue")
        axis.fill_between(
            time_s,
            np.clip(mean - std, 0, 1),
            np.clip(mean + std, 0, 1),
            color="tab:blue",
            alpha=0.22,
        )
        axis.set(title=f"{size}x{size}", xlabel="time (s)", ylim=(0, 1.03))
    axes[0].set_ylabel(r"$1-S_d$")
    figure.suptitle(
        f"Figure 3 ratio {RATIO}: {duration_s:g} s size sensitivity "
        f"(mean +/- SD, {len(seeds)} seeds)"
    )

    def successive_checks_for(key: str) -> list[dict[str, int | float | bool]]:
        return [
            successive_size_check(
                smaller["lattice_size"],
                larger["lattice_size"],
                smaller[f"{key}_mean"],
                larger[f"{key}_mean"],
                smaller[f"{key}_std"],
                larger[f"{key}_std"],
                len(seeds),
                relative_tolerance=RELATIVE_TOLERANCE,
            )
            for smaller, larger in pairwise(summaries)
        ]

    successive_checks = successive_checks_for("detrended_amplitude")
    specular_successive_checks = successive_checks_for("specular_detrended_amplitude")

    output = {
        "nominal_ga_n_ratio": RATIO,
        "duration_s": duration_s,
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
        "kinematic_specular_successive_size_checks": specular_successive_checks,
        "secondary_observable": {
            "name": "specular_detrended_amplitude",
            "grazing_angle_deg": GRAZING_ANGLE_DEG,
            "definition": (
                "detrended amplitude of the kinematic (00) intensity at the anti-phase "
                "condition, computed from the same snapshots as the proxy; it is a "
                "single-scattering calculation, not dynamical RHEED"
            ),
        },
        "coverage_window_ml": duration_s * parameters.predicted_growth_rate_ml_s,
        "note": (
            "pass --sizes/--duration to widen the study; the 4 s default spans under one "
            "monolayer, so its detrended amplitude measures a partial period"
        ),
        "sizes": summaries,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "figure3_convergence.json").write_text(json.dumps(output, indent=2) + "\n")
    figure.savefig(figure_dir / "figure3_convergence.png", dpi=160)
    plt.close(figure)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main(**parse_workflow_args(seeds=SEEDS, sizes=SIZES, duration_s=DURATION_S))
