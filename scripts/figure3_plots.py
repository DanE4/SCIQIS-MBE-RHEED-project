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
# Rows of the 2x3 montage: a half-filled layer beside a completed one.
MONTAGE_COVERAGES_ML = (0.5, 1.0)
# The two figures copied into assets/ by `make readme-figures` are fetched on every GitHub
# README load, where the column is under 900 px wide, so print resolution only costs bytes.
README_DPI = 120


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
    # Size and ensemble are workflow overrides, so read them off the traces being plotted.
    _size = traces[0]["simulation_config"]["lattice_size"]
    axes[0, 1].set_title(
        "This model: morphology-derived proxy\n"
        f"({_size}x{_size}, {len(traces[0]['seeds'])}-seed ensemble)"
    )
    axes[0, 1].legend(loc="lower right")
    for axis in axes[-1]:
        axis.set_xlabel("time (s)")
        axis.set_xlim(0, 40)
    figure.suptitle(
        "Figure 3 comparison - separate scales, shared time domain\n"
        "Reference is figure-derived; proxy amplitude is not finite-size converged"
    )
    figure.savefig(figure_dir / "figure3_comparison.png", dpi=README_DPI)
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
        "Figure 4-inspired homoepitaxial morphology sequence - "
        f"Ga/N={ratio:.2f}, seed {morphology_seed}; no strain"
    )
    figure.savefig(figure_dir / "figure4_inspired_morphology.png", dpi=README_DPI)
    plt.close(figure)
    return {
        "classification": "homoepitaxial layer-cycle illustration; not a strain/SK reproduction",
        "nominal_ga_n_ratio": ratio,
        "seed": morphology_seed,
        "coordinate": "paper-predicted coverage = predicted growth rate times time",
        "frames": frames,
    }


def morphology_montage(runs: list[dict[str, object]], figure_dir: Path) -> dict[str, object]:
    """Top-down 2x3 montage: rows are coverage, columns are the paper's Ga/N ratios.

    `runs` carries one representative single-seed result per ratio, already produced by the
    Figure 3 ensemble, so this costs no extra simulation. Panels are picked on the same
    paper-predicted coverage axis `morphology_sequence` uses.
    """
    ordered = sorted(runs, key=lambda entry: entry["nominal_ga_n_ratio"])
    panels = []
    for entry in ordered:
        result = entry["result"]
        predicted = result.time_s * entry["predicted_growth_rate_ml_s"]
        for target in MONTAGE_COVERAGES_ML:
            index = int(np.argmin(np.abs(predicted - target)))
            proxy = float(result.rheed_proxy[index])
            panels.append(
                {
                    "nominal_ga_n_ratio": entry["nominal_ga_n_ratio"],
                    "target_predicted_coverage_ml": target,
                    "predicted_coverage_ml": float(predicted[index]),
                    "net_simulated_coverage_ml": float(result.coverage_ml[index]),
                    "time_s": float(result.time_s[index]),
                    "roughness_ml": float(result.roughness_ml[index]),
                    "island_density_per_site": float(result.island_density_per_site[index]),
                    # The stored trace is the proxy itself, so S_d is its complement.
                    "step_density": 1.0 - proxy,
                    "rheed_proxy": proxy,
                    "height_ml": result.snapshots[index].tolist(),
                }
            )

    ratios = [entry["nominal_ga_n_ratio"] for entry in ordered]
    figure, axes = plt.subplots(
        len(MONTAGE_COVERAGES_ML),
        len(ratios),
        figsize=(11, 7.4),
        constrained_layout=True,
    )
    maximum_height = max(max(max(row) for row in panel["height_ml"]) for panel in panels)
    for panel in panels:
        row = MONTAGE_COVERAGES_ML.index(panel["target_predicted_coverage_ml"])
        column = ratios.index(panel["nominal_ga_n_ratio"])
        axis = axes[row, column]
        heights = np.asarray(panel["height_ml"])
        y, x = np.indices(heights.shape)
        image = axis.scatter(
            x + 0.5 * y,
            np.sqrt(3.0) / 2.0 * y,
            c=heights,
            marker="h",
            # One marker per site, deliberately overfilled so the map reads as a continuous
            # surface at any lattice size. Same convention as `morphology_sequence`, rescaled
            # from its (128 sites, s=190, 2.4 in) panel to this figure's wider ones.
            s=5.9e6 / len(heights) ** 2,
            cmap="viridis",
            vmin=0,
            vmax=max(1, maximum_height),
        )
        axis.set_title(
            f"Ga/N = {panel['nominal_ga_n_ratio']:.2f}, "
            f"{panel['predicted_coverage_ml']:.2f} ML",
            fontsize=10,
        )
        axis.set_xlabel(
            f"roughness {panel['roughness_ml']:.3f} ML\n"
            f"$S_d$ = {panel['step_density']:.3f}, "
            f"$1-S_d$ = {panel['rheed_proxy']:.3f}\n"
            f"island density {panel['island_density_per_site']:.4f} / site",
            fontsize=8,
        )
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_aspect("equal")
    for row, target in enumerate(MONTAGE_COVERAGES_ML):
        axes[row, 0].set_ylabel(f"target {target:.2f} ML", fontsize=11)
    figure.colorbar(image, ax=axes, shrink=0.6, label="column height (ML)")
    figure.suptitle(
        "Top-down homoepitaxial morphology across the Figure 3 Ga/N conditions\n"
        "Partial layers are stepped and rough; completed layers smooth out. "
        "No strain, no SK transition, no quantum dots."
    )
    figure.savefig(figure_dir / "figure3_morphology_montage.png", dpi=README_DPI)
    plt.close(figure)
    return {
        "classification": (
            "homoepitaxial coverage/flux morphology comparison; "
            "not a strain, Stranski-Krastanov or quantum-dot reproduction"
        ),
        "coordinate": "paper-predicted coverage = predicted growth rate times time",
        "nominal_ga_n_ratios": ratios,
        "target_coverages_ml": list(MONTAGE_COVERAGES_ML),
        # The picture is the PNG above, so the artifact keeps the per-panel numbers and drops
        # the six height fields behind them - 200 kB of committed JSON nothing reads back.
        "panels": [
            {key: value for key, value in panel.items() if key != "height_ml"}
            for panel in panels
        ],
    }
