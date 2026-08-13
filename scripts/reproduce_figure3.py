"""Regenerate the Stage 5 Figure 3 comparison and morphology artifacts."""

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mbe_rheed_sim import run
from mbe_rheed_sim.analysis import rheed_oscillation_metrics
from mbe_rheed_sim.paper import FIGURE3_NOMINAL_GA_N_RATIOS, figure3_config, figure3_parameters
from mbe_rheed_sim.workflows import artifact_root, parse_int_values, resolve_workers, run_parallel

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = ROOT / "data" / "reference" / "figure3_experimental_digitized.json"
SOURCE_PDF = ROOT / "nanomaterials-12-03052.pdf"
LATTICE_SIZE = 7
SEEDS = (2026, 2027, 2028)
TIME_GRID_S = np.linspace(0.0, 40.0, 401)
MORPHOLOGY_RATIO = 0.82
MORPHOLOGY_TARGETS_ML = (0.0, 0.5, 1.0, 1.5, 2.0)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_version() -> dict[str, str | bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    source_digest = hashlib.sha256()
    source_paths = [Path(__file__), *sorted((ROOT / "src/mbe_rheed_sim").glob("*.py"))]
    for path in source_paths:
        source_digest.update(str(path.relative_to(ROOT)).encode())
        source_digest.update(path.read_bytes())
    return {
        "git_commit_at_generation": commit,
        "working_tree_dirty": dirty,
        "generation_source_sha256": source_digest.hexdigest(),
    }


def _circular_difference(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    difference = abs(left - right) % 1.0
    return min(difference, 1.0 - difference)


def _plot_comparison(traces: list[dict[str, object]], figure_dir: Path) -> None:
    figure, axes = plt.subplots(
        len(traces), 2, figsize=(10, 7.5), sharex=True, constrained_layout=True
    )
    for row, trace in enumerate(traces):
        ratio = float(trace["nominal_ga_n_ratio"])
        time = np.asarray(trace["time_s"])
        mean = np.asarray(trace["rheed_proxy_mean"])
        std = np.asarray(trace["rheed_proxy_std"])
        reference_time = np.asarray(trace["reference_time_s"])
        reference = np.asarray(trace["reference_rheed_panel_coordinate"])

        axes[row, 0].plot(reference_time, reference, color="tab:red", linewidth=1.5)
        axes[row, 0].set_ylim(0, 1)
        axes[row, 0].set_ylabel(f"Ga/N={ratio:.2f}\nfigure coordinate")
        axes[row, 1].fill_between(
            time,
            np.clip(mean - std, 0, 1),
            np.clip(mean + std, 0, 1),
            color="tab:blue",
            alpha=0.22,
            label="mean +/- 1 SD" if row == 0 else None,
        )
        axes[row, 1].plot(time, mean, color="tab:blue", linewidth=1.5)
        axes[row, 1].set_ylim(0, 1)
        axes[row, 1].set_ylabel(r"raw $1-S_d$")
    axes[0, 0].set_title("Figure-derived experimental RHEED\n(panel-coordinate normalized)")
    axes[0, 1].set_title("This model: morphology-derived proxy\n(7x7, 3-seed smoke ensemble)")
    axes[0, 1].legend(loc="lower right")
    for axis in axes[-1]:
        axis.set_xlabel("time (s)")
        axis.set_xlim(0, 40)
    figure.suptitle(
        "Figure 3 comparison — separate scales, shared time domain\n"
        "Reference is figure-derived; proxy amplitude is not finite-size converged"
    )
    figure.savefig(figure_dir / "figure3_publication_comparison.png", dpi=180)
    plt.close(figure)


def _plot_metrics(comparisons: list[dict[str, object]], figure_dir: Path) -> None:
    ratios = np.asarray([item["nominal_ga_n_ratio"] for item in comparisons])
    figure, axes = plt.subplots(2, 2, figsize=(9, 6.5), constrained_layout=True)
    specifications = (
        ("period_ml", "period (ML)"),
        ("peak_phase_ml", "peak phase (ML modulo 1)"),
        ("damping_rate_per_ml", "log-envelope slope (per ML)"),
        ("relative_detrended_amplitude", "relative amplitude (Ga/N=0.89 baseline)"),
    )
    for axis, (metric, label) in zip(axes.flat, specifications, strict=True):
        for source, color, marker in (
            ("reference", "tab:red", "o"),
            ("simulation", "tab:blue", "s"),
        ):
            values = [item[source][metric] for item in comparisons]
            axis.plot(ratios, values, color=color, marker=marker, label=source)
        axis.set(xlabel="nominal Ga/N ratio", ylabel=label)
        axis.grid(alpha=0.25)
    axes[0, 0].legend()
    figure.suptitle("Figure 3 diagnostics (reference is figure-derived, simulation is raw proxy)")
    figure.savefig(figure_dir / "figure3_metric_comparison.png", dpi=180)
    plt.close(figure)


def _morphology_sequence(
    result, growth_rate: float, figure_dir: Path, morphology_seed: int
) -> dict[str, object]:
    predicted = result.time_s * growth_rate
    indices = [int(np.argmin(np.abs(predicted - target))) for target in MORPHOLOGY_TARGETS_ML]
    frames = [
        {
            "target_predicted_coverage_ml": target,
            "time_s": float(result.time_s[index]),
            "predicted_coverage_ml": float(predicted[index]),
            "net_simulated_coverage_ml": float(result.coverage_ml[index]),
            "rheed_proxy": float(result.rheed_proxy[index]),
            "height_ml": result.snapshots[index].tolist(),
        }
        for target, index in zip(MORPHOLOGY_TARGETS_ML, indices, strict=True)
    ]

    figure, axes = plt.subplots(1, len(frames), figsize=(12, 2.8), constrained_layout=True)
    y, x = np.indices(result.final_heights.shape)
    cartesian_x = x + 0.5 * y
    cartesian_y = np.sqrt(3.0) / 2.0 * y
    maximum_height = max(max(max(row) for row in frame["height_ml"]) for frame in frames)
    for axis, frame in zip(axes, frames, strict=True):
        heights = np.asarray(frame["height_ml"])
        image = axis.scatter(
            cartesian_x,
            cartesian_y,
            c=heights,
            marker="h",
            s=190,
            cmap="viridis",
            vmin=0,
            vmax=max(1, maximum_height),
        )
        axis.set_title(
            f"target {frame['target_predicted_coverage_ml']:.1f} ML\n"
            f"actual {frame['predicted_coverage_ml']:.2f} ML"
        )
        axis.set_aspect("equal")
        axis.set_axis_off()
    figure.colorbar(image, ax=axes, shrink=0.7, label="column height (ML)")
    figure.suptitle(
        "Figure 4-inspired homoepitaxial morphology sequence — "
        f"Ga/N=0.82, seed {morphology_seed}; no strain"
    )
    figure.savefig(figure_dir / "figure4_inspired_morphology.png", dpi=180)
    plt.close(figure)
    return {
        "classification": "homoepitaxial layer-cycle illustration; not a strain/SK reproduction",
        "nominal_ga_n_ratio": MORPHOLOGY_RATIO,
        "seed": morphology_seed,
        "coordinate": "paper-predicted coverage = predicted growth rate times time",
        "frames": frames,
    }


def main(*, workers: int = 4, seeds: tuple[int, ...] = SEEDS) -> None:
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

    configurations = [
        figure3_config(ratio, lattice_size=LATTICE_SIZE, seed=seed)
        for ratio in FIGURE3_NOMINAL_GA_N_RATIOS
        for seed in seeds
    ]
    all_results = run_parallel(
        run,
        configurations,
        workers=workers,
        description="Figure 3 publication ensemble",
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
                "paper_parameters": parameters.as_dict(),
                "simulation_config": results[0].config.as_dict(),
                "seeds": seeds,
                "time_s": TIME_GRID_S.tolist(),
                "predicted_coverage_ml": predicted_coverage.tolist(),
                "rheed_proxy_mean": mean.tolist(),
                "rheed_proxy_std": std.tolist(),
                "simulation_metrics": simulation_metrics.as_dict(),
                "reference_time_s": reference["time_s"],
                "reference_rheed_panel_coordinate": reference["rheed_panel_coordinate"],
                "reference_metrics": reference_metrics.as_dict(),
            }
        )
        ratio_label = f"{ratio:.2f}".replace(".", "")
        np.savez_compressed(
            run_dir / f"figure3_ratio_{ratio_label}.npz",
            config_json=json.dumps(results[0].config.as_dict(), sort_keys=True),
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
                "paper_parameters": parameters.as_dict(),
                "simulation_config": results[0].config.as_dict(),
                "seeds": seeds,
                "simulation_metrics": simulation_metrics.as_dict(),
                "reference_metrics": reference_metrics.as_dict(),
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
    morphology = _morphology_sequence(
        morphology_result, morphology_growth_rate, figure_dir, seeds[0]
    )
    _plot_comparison(traces, figure_dir)
    _plot_metrics(comparisons, figure_dir)

    provenance = {
        "generated_by": "scripts/reproduce_figure3.py",
        "code_version": code_version,
        "source_pdf_sha256": _sha256(SOURCE_PDF),
        "reference_json_sha256": _sha256(REFERENCE_PATH),
        "lattice_size": LATTICE_SIZE,
        "seeds": seeds,
        "effective_workers": min(resolve_workers(workers), len(configurations)),
        "classification": "qualitative finite-size smoke comparison; amplitude not converged",
    }
    artifact = {
        "description": "Stage 5 Figure 3 comparison and Figure 4-inspired morphology sequence",
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
        "figures": [
            "outputs/figures/figure3_publication_comparison.png",
            "outputs/figures/figure3_metric_comparison.png",
            "outputs/figures/figure4_inspired_morphology.png",
        ],
    }
    (run_dir / "figure3_publication.json").write_text(json.dumps(artifact, indent=2) + "\n")
    (run_dir / "figure3_run_summaries.json").write_text(json.dumps(summaries, indent=2) + "\n")
    (processed_dir / "figure3_simulated_smoke.json").write_text(
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int)
    parser.add_argument("--seeds")
    arguments = parser.parse_args()
    main(
        workers=resolve_workers(arguments.workers),
        seeds=parse_int_values(arguments.seeds, SEEDS),
    )
