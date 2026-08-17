"""Static PNG figures for the Figure 3 comparison workflow.

Split out of `reproduce_figure3.py` so that script stays a data pipeline. These write files
with Matplotlib's Agg backend; the notebook's interactive Plotly equivalents live in
`mbe_rheed_notebook.figures`.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MORPHOLOGY_TARGETS_ML = (0.0, 0.5, 1.0, 1.5, 2.0)


def plot_comparison(traces: list[dict[str, object]], figure_dir: Path) -> None:
    """Reference panel coordinates beside the model proxy, on deliberately separate scales."""
    figure, axes = plt.subplots(
        len(traces), 2, figsize=(10, 7.5), sharex=True, constrained_layout=True
    )
    for row, trace in enumerate(traces):
        ratio = float(trace["nominal_ga_n_ratio"])
        time = np.asarray(trace["time_s"])
        mean = np.asarray(trace["rheed_proxy_mean"])
        std = np.asarray(trace["rheed_proxy_std"])

        axes[row, 0].plot(
            np.asarray(trace["reference_time_s"]),
            np.asarray(trace["reference_rheed_panel_coordinate"]),
            color="tab:red",
            linewidth=1.5,
        )
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
    axes[0, 1].set_title("This model: morphology-derived proxy\n(7x7, 3-seed ensemble)")
    axes[0, 1].legend(loc="lower right")
    for axis in axes[-1]:
        axis.set_xlabel("time (s)")
        axis.set_xlim(0, 40)
    figure.suptitle(
        "Figure 3 comparison — separate scales, shared time domain\n"
        "Reference is figure-derived; proxy amplitude is not finite-size converged"
    )
    figure.savefig(figure_dir / "figure3_comparison.png", dpi=180)
    plt.close(figure)


def plot_metrics(comparisons: list[dict[str, object]], figure_dir: Path) -> None:
    """Period, phase, damping, and relative amplitude versus Ga/N ratio."""
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
            axis.plot(
                ratios,
                [item[source][metric] for item in comparisons],
                color=color,
                marker=marker,
                label=source,
            )
        axis.set(xlabel="nominal Ga/N ratio", ylabel=label)
        axis.grid(alpha=0.25)
    axes[0, 0].legend()
    figure.suptitle("Figure 3 diagnostics (reference is figure-derived, simulation is raw proxy)")
    figure.savefig(figure_dir / "figure3_metric_comparison.png", dpi=180)
    plt.close(figure)


def morphology_sequence(
    result, growth_rate: float, figure_dir: Path, morphology_seed: int, ratio: float
) -> dict[str, object]:
    """Save the layer-cycle snapshot strip and return the frames as serializable data."""
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
    maximum_height = max(max(max(row) for row in frame["height_ml"]) for frame in frames)
    for axis, frame in zip(axes, frames, strict=True):
        image = axis.scatter(
            x + 0.5 * y,
            np.sqrt(3.0) / 2.0 * y,
            c=np.asarray(frame["height_ml"]),
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
        f"Ga/N={ratio:.2f}, seed {morphology_seed}; no strain"
    )
    figure.savefig(figure_dir / "figure4_inspired_morphology.png", dpi=180)
    plt.close(figure)
    return {
        "classification": "homoepitaxial layer-cycle illustration; not a strain/SK reproduction",
        "nominal_ga_n_ratio": ratio,
        "seed": morphology_seed,
        "coordinate": "paper-predicted coverage = predicted growth rate times time",
        "frames": frames,
    }
