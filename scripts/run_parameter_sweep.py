"""Generate a small deterministic temperature/flux RHEED-amplitude map."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mbe_rheed_sim import SimulationConfig, run
from mbe_rheed_sim.analysis import (
    oscillation_amplitude,
    rheed_oscillation_metrics,
)
from mbe_rheed_sim.workflows import artifact_root, parse_int_values, resolve_workers, run_parallel

ROOT = Path(__file__).resolve().parents[1]
TEMPERATURES_K = (700.0, 850.0, 1_000.0)
FLUXES_ML_S = (0.25, 0.5, 0.75)
SEEDS = (0, 1, 2)
BASE = SimulationConfig(
    lattice_size=16,
    target_coverage_ml=2.0,
    sample_every_ml=0.05,
    max_isolated_hop_distance=3,
)


def main(*, workers: int = 4, seeds: tuple[int, ...] = SEEDS) -> None:
    output_root = artifact_root(ROOT)
    run_dir = output_root / "outputs" / "runs"
    figure_dir = output_root / "outputs" / "figures"
    processed_data = output_root / "data" / "processed" / "parameter_sweep.json"
    mean_amplitude = np.empty((len(TEMPERATURES_K), len(FLUXES_ML_S)))
    std_amplitude = np.empty_like(mean_amplitude)
    mean_detrended_amplitude = np.empty_like(mean_amplitude)
    std_detrended_amplitude = np.empty_like(mean_amplitude)
    oscillatory_fraction = np.empty_like(mean_amplitude)
    metrics_by_point = []
    configurations = [
        replace(
            BASE,
            temperature_k=temperature,
            deposition_flux_ml_s=flux,
            seed=seed,
        )
        for temperature in TEMPERATURES_K
        for flux in FLUXES_ML_S
        for seed in seeds
    ]
    results = run_parallel(
        run,
        configurations,
        workers=workers,
        description="temperature/flux parameter sweep",
    )
    coverage_ml = np.linspace(0.0, float(BASE.target_coverage_ml), 201)
    for temperature_index, temperature in enumerate(TEMPERATURES_K):
        temperature_metrics = []
        for flux_index, flux in enumerate(FLUXES_ML_S):
            point_index = temperature_index * len(FLUXES_ML_S) + flux_index
            point_results = results[point_index * len(seeds) : (point_index + 1) * len(seeds)]
            traces = np.vstack(
                [
                    np.interp(coverage_ml, result.coverage_ml, result.rheed_proxy)
                    for result in point_results
                ]
            )
            amplitudes = np.array([oscillation_amplitude(trace) for trace in traces])
            metrics = [rheed_oscillation_metrics(coverage_ml, trace) for trace in traces]
            detrended = np.array([metric.detrended_amplitude for metric in metrics])
            mean_amplitude[temperature_index, flux_index] = amplitudes.mean()
            std_amplitude[temperature_index, flux_index] = amplitudes.std()
            mean_detrended_amplitude[temperature_index, flux_index] = detrended.mean()
            std_detrended_amplitude[temperature_index, flux_index] = detrended.std(ddof=1)
            oscillatory_fraction[temperature_index, flux_index] = np.mean(
                [metric.is_oscillatory for metric in metrics]
            )
            temperature_metrics.append([metric.as_dict() for metric in metrics])
        metrics_by_point.append(temperature_metrics)

    summary = {
        "base_config": BASE.as_dict(),
        "temperatures_k": TEMPERATURES_K,
        "fluxes_ml_s": FLUXES_ML_S,
        "seeds": seeds,
        "effective_workers": min(resolve_workers(workers), len(configurations)),
        "amplitude_definition": "half of the 95th-minus-5th percentile proxy range",
        "mean_amplitude": mean_amplitude.tolist(),
        "std_amplitude": std_amplitude.tolist(),
        "principal_observable": {
            "name": "detrended_amplitude",
            "reason": "measures the periodic component after removing linear drift",
        },
        "mean_detrended_amplitude": mean_detrended_amplitude.tolist(),
        "std_detrended_amplitude": std_detrended_amplitude.tolist(),
        "oscillatory_fraction": oscillatory_fraction.tolist(),
        "oscillation_metrics_per_seed": metrics_by_point,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    processed_data.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(summary, indent=2) + "\n"
    (run_dir / "parameter_sweep.json").write_text(serialized)
    processed_data.write_text(serialized)

    figure, axis = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    image = axis.imshow(mean_amplitude, origin="lower", cmap="magma", aspect="auto")
    figure.colorbar(image, ax=axis, label="mean proxy amplitude")
    axis.set(
        xlabel="deposition flux (ML/s)",
        ylabel="temperature (K)",
        xticks=range(len(FLUXES_ML_S)),
        xticklabels=FLUXES_ML_S,
        yticks=range(len(TEMPERATURES_K)),
        yticklabels=TEMPERATURES_K,
        title="Small-lattice RHEED-proxy regime map (mean +/- SD, 3 seeds)",
    )
    for temperature_index in range(len(TEMPERATURES_K)):
        for flux_index in range(len(FLUXES_ML_S)):
            axis.text(
                flux_index,
                temperature_index,
                f"{mean_amplitude[temperature_index, flux_index]:.3f}\n"
                f"+/- {std_amplitude[temperature_index, flux_index]:.3f}",
                color="white",
                ha="center",
                va="center",
            )
    figure.savefig(figure_dir / "parameter_sweep.png", dpi=160)
    plt.close(figure)
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
