"""Regenerate the Figure 3 Ga/N comparison and morphology artifacts."""

import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from figure3_plots import (
    morphology_montage,
    morphology_sequence,
    plot_comparison,
    plot_metrics,
)

from mbe_rheed_sim import run
from mbe_rheed_sim.analysis import rheed_oscillation_metrics
from mbe_rheed_sim.paper import FIGURE3_NOMINAL_GA_N_RATIOS, figure3_config, figure3_parameters
from mbe_rheed_sim.workflows import (
    artifact_root,
    git_revision,
    parse_workflow_args,
    run_parallel,
)

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = ROOT / "data" / "reference" / "figure3_experimental_digitized.json"
SOURCE_PDF = ROOT / "nanomaterials-12-03052.pdf"
# The committed artifact under data/processed is generated from these defaults, and
# tests/test_publication_artifact.py checks it against them, so they are the shipped numbers.
# 128 is the largest size the convergence study covers and stays inside the event limit
# `figure3_config` sets; the seeds are the ensemble the mean and SD are taken over.
LATTICE_SIZE = 128
SEEDS = (2026, 2027, 2028, 2029, 2030)
TIME_GRID_S = np.linspace(0.0, 40.0, 401)
MORPHOLOGY_RATIO = 0.82


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_version() -> dict[str, str | bool]:
    source_digest = hashlib.sha256()
    for path in [Path(__file__), *sorted((ROOT / "src/mbe_rheed_sim").glob("*.py"))]:
        source_digest.update(str(path.relative_to(ROOT)).encode())
        source_digest.update(path.read_bytes())
    return {**git_revision(ROOT), "generation_source_sha256": source_digest.hexdigest()}


def _circular_difference(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    difference = abs(left - right) % 1.0
    return min(difference, 1.0 - difference)


def main(
    *,
    workers: int = 4,
    seeds: tuple[int, ...] = SEEDS,
    sizes: tuple[int, ...] = (LATTICE_SIZE,),
) -> None:
    if len(sizes) != 1:
        raise ValueError("the Ga/N comparison runs one lattice size; pass a single --sizes")
    (lattice_size,) = sizes
    output_root = artifact_root(ROOT)
    run_dir = output_root / "outputs" / "runs"
    figure_dir = output_root / "outputs" / "figures"
    processed_dir = output_root / "data" / "processed"
    run_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    reference_data = json.loads(REFERENCE_PATH.read_text())
    code_version = _git_version()
    references = {
        float(trace["nominal_ga_n_ratio"]): trace for trace in reference_data["traces"]
    }
    summaries = []
    traces = []
    morphology_result = None
    morphology_growth_rate = None
    # One representative seed per ratio, reused by the top-down montage below.
    montage_runs: list[dict[str, object]] = []

    configurations = [
        figure3_config(ratio, lattice_size=lattice_size, seed=seed)
        for ratio in FIGURE3_NOMINAL_GA_N_RATIOS
        for seed in seeds
    ]
    all_results = run_parallel(
        run,
        configurations,
        workers=workers,
        description="Figure 3 Ga/N ensemble",
    )

    for ratio_index, ratio in enumerate(FIGURE3_NOMINAL_GA_N_RATIOS):
        parameters = figure3_parameters(ratio)
        start = ratio_index * len(seeds)
        results = all_results[start : start + len(seeds)]
        proxy_traces = np.vstack(
            [np.interp(TIME_GRID_S, result.time_s, result.rheed_proxy) for result in results]
        )
        mean = proxy_traces.mean(axis=0)
        std = proxy_traces.std(axis=0)
        predicted_coverage = TIME_GRID_S * parameters.predicted_growth_rate_ml_s
        simulation_metrics = rheed_oscillation_metrics(predicted_coverage, mean)
        reference = references[ratio]
        reference_time = np.asarray(reference["time_s"])
        reference_signal = np.asarray(reference["rheed_panel_coordinate"])
        reference_metrics = rheed_oscillation_metrics(
            reference_time * parameters.predicted_growth_rate_ml_s, reference_signal
        )
        traces.append(
            {
                "nominal_ga_n_ratio": ratio,
                "paper_parameters": asdict(parameters),
                "simulation_config": asdict(results[0].config),
                "seeds": seeds,
                "time_s": TIME_GRID_S.tolist(),
                "predicted_coverage_ml": predicted_coverage.tolist(),
                "rheed_proxy_mean": mean.tolist(),
                "rheed_proxy_std": std.tolist(),
                "simulation_metrics": asdict(simulation_metrics),
                "reference_time_s": reference["time_s"],
                "reference_rheed_panel_coordinate": reference["rheed_panel_coordinate"],
                "reference_metrics": asdict(reference_metrics),
            }
        )
        ratio_label = f"{ratio:.2f}".replace(".", "")
        np.savez_compressed(
            run_dir / f"figure3_ratio_{ratio_label}.npz",
            config_json=json.dumps(asdict(results[0].config), sort_keys=True),
            seeds=np.asarray(seeds),
            code_version_json=json.dumps(code_version, sort_keys=True),
            time_s=TIME_GRID_S,
            predicted_coverage_ml=predicted_coverage,
            rheed_proxy_traces=proxy_traces,
            rheed_proxy_mean=mean,
            rheed_proxy_std=std,
            reference_time_s=reference_time,
            reference_rheed_panel_coordinate=reference_signal,
        )
        summaries.append(
            {
                "paper_parameters": asdict(parameters),
                "simulation_config": asdict(results[0].config),
                "seeds": seeds,
                "simulation_metrics": asdict(simulation_metrics),
                "reference_metrics": asdict(reference_metrics),
                "runs": [
                    {
                        "final_coverage_ml": float(result.coverage_ml[-1]),
                        "final_roughness_ml": float(result.roughness_ml[-1]),
                        "deposited_events": result.deposited_events,
                        "desorbed_events": result.desorbed_events,
                        "equivalent_diffusion_hops": result.diffusion_events,
                        "selected_diffusion_events": result.selected_diffusion_events,
                        "long_hop_events": result.long_hop_events,
                    }
                    for result in results
                ],
            }
        )
        montage_runs.append(
            {
                "nominal_ga_n_ratio": ratio,
                "result": results[0],
                "predicted_growth_rate_ml_s": parameters.predicted_growth_rate_ml_s,
            }
        )
        if ratio == MORPHOLOGY_RATIO:
            morphology_result = results[0]
            morphology_growth_rate = parameters.predicted_growth_rate_ml_s

    reference_baseline = traces[0]["reference_metrics"]["detrended_amplitude"]
    simulation_baseline = traces[0]["simulation_metrics"]["detrended_amplitude"]
    comparisons = []
    for trace in traces:
        reference_metrics = trace["reference_metrics"]
        simulation_metrics = trace["simulation_metrics"]
        reference_metrics["relative_detrended_amplitude"] = (
            reference_metrics["detrended_amplitude"] / reference_baseline
        )
        simulation_metrics["relative_detrended_amplitude"] = (
            simulation_metrics["detrended_amplitude"] / simulation_baseline
        )
        comparisons.append(
            {
                "nominal_ga_n_ratio": trace["nominal_ga_n_ratio"],
                "reference": reference_metrics,
                "simulation": simulation_metrics,
                "simulation_minus_reference_period_ml": (
                    simulation_metrics["period_ml"] - reference_metrics["period_ml"]
                    if simulation_metrics["period_ml"] is not None
                    and reference_metrics["period_ml"] is not None
                    else None
                ),
                "absolute_peak_phase_difference_ml": _circular_difference(
                    simulation_metrics["peak_phase_ml"], reference_metrics["peak_phase_ml"]
                ),
            }
        )

    if morphology_result is None or morphology_growth_rate is None:
        raise RuntimeError("representative morphology run was not produced")
    morphology = morphology_sequence(
        morphology_result, morphology_growth_rate, figure_dir, seeds[0], MORPHOLOGY_RATIO
    )
    montage = morphology_montage(montage_runs, figure_dir)
    plot_comparison(traces, figure_dir)
    plot_metrics(comparisons, figure_dir)

    provenance = {
        "generated_by": "scripts/reproduce_figure3.py",
        "code_version": code_version,
        "source_pdf_sha256": _sha256(SOURCE_PDF),
        "reference_json_sha256": _sha256(REFERENCE_PATH),
        "lattice_size": lattice_size,
        "seeds": seeds,
        "effective_workers": min(workers, len(configurations)),
        "classification": "qualitative finite-size comparison; amplitude not converged",
    }
    artifact = {
        "description": "Figure 3 Ga/N comparison and Figure 4-inspired morphology sequence",
        "provenance": provenance,
        "signal_definitions": {
            "reference": reference_data["classification"],
            "simulation": "raw morphology-derived normalized step-density proxy 1-S_d",
            "diagnostics": "linearly detrended signals used only for oscillation metrics",
        },
        "normalization": reference_data["extraction"],
        "reference_source": reference_data["source"],
        "traces": traces,
        "comparisons": comparisons,
        "morphology_sequence": morphology,
        "morphology_montage": montage,
        "figures": [
            "outputs/figures/figure3_comparison.png",
            "outputs/figures/figure3_metric_comparison.png",
            "outputs/figures/figure4_inspired_morphology.png",
            "outputs/figures/figure3_morphology_montage.png",
        ],
    }
    (run_dir / "figure3_comparison.json").write_text(json.dumps(artifact, indent=2) + "\n")
    (run_dir / "figure3_run_summaries.json").write_text(json.dumps(summaries, indent=2) + "\n")
    (processed_dir / "figure3_simulated_reduced.json").write_text(
        json.dumps(artifact, separators=(",", ":")) + "\n"
    )
    with (run_dir / "figure3_metric_comparison.csv").open("w", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=(
                "nominal_ga_n_ratio",
                "signal",
                "period_ml",
                "peak_phase_ml",
                "damping_rate_per_ml",
                "detrended_amplitude",
                "relative_detrended_amplitude",
            ),
        )
        writer.writeheader()
        for comparison in comparisons:
            for signal in ("reference", "simulation"):
                writer.writerow(
                    {
                        "nominal_ga_n_ratio": comparison["nominal_ga_n_ratio"],
                        "signal": signal,
                        **{
                            field: comparison[signal][field]
                            for field in writer.fieldnames
                            if field not in {"nominal_ga_n_ratio", "signal"}
                        },
                    }
                )
    print(json.dumps({"provenance": provenance, "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main(**parse_workflow_args(seeds=SEEDS, sizes=(LATTICE_SIZE,)))
