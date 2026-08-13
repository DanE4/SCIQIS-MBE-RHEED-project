"""Check one oscillation-scale Figure 3 window across practical lattice sizes."""

import json
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mbe_rheed_sim import run
from mbe_rheed_sim.analysis import oscillation_amplitude
from mbe_rheed_sim.paper import figure3_config

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "outputs" / "runs"
FIGURE_DIR = ROOT / "outputs" / "figures"
RATIO = 0.82
DURATION_S = 4.0
SIZES = (8, 16, 32)
SEEDS = (0, 1, 2)
TIME_S = np.linspace(0.0, DURATION_S, 101)


def main() -> None:
    summaries = []
    figure, axes = plt.subplots(1, len(SIZES), figsize=(12, 3.5), sharey=True)
    for axis, size in zip(axes, SIZES, strict=True):
        started = perf_counter()
        results = [
            run(figure3_config(RATIO, lattice_size=size, duration_s=DURATION_S, seed=seed))
            for seed in SEEDS
        ]
        elapsed = perf_counter() - started
        traces = np.vstack(
            [np.interp(TIME_S, result.time_s, result.rheed_proxy) for result in results]
        )
        roughness = np.array([result.roughness_ml[-1] for result in results])
        amplitudes = np.array([oscillation_amplitude(trace) for trace in traces])
        mean = traces.mean(axis=0)
        std = traces.std(axis=0)
        summaries.append(
            {
                "lattice_size": size,
                "elapsed_s": elapsed,
                "roughness_mean_ml": float(roughness.mean()),
                "roughness_std_ml": float(roughness.std()),
                "proxy_amplitude_mean": float(amplitudes.mean()),
                "proxy_amplitude_std": float(amplitudes.std()),
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

    output = {
        "nominal_ga_n_ratio": RATIO,
        "duration_s": DURATION_S,
        "seeds": SEEDS,
        "note": "64x64 excluded from the default command due measured runtime; see docs/VALIDATION.md",
        "sizes": summaries,
    }
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "figure3_convergence.json").write_text(json.dumps(output, indent=2) + "\n")
    figure.savefig(FIGURE_DIR / "figure3_convergence.png", dpi=160)
    plt.close(figure)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
