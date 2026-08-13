"""Run the small-lattice Figure 3 simulation smoke comparison."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mbe_rheed_sim import run
from mbe_rheed_sim.paper import FIGURE3_NOMINAL_GA_N_RATIOS, figure3_config, figure3_parameters

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "outputs" / "runs"
FIGURE_DIR = ROOT / "outputs" / "figures"
LATTICE_SIZE = 7
SEEDS = (2026, 2027, 2028)
TIME_GRID_S = np.linspace(0.0, 40.0, 401)


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True, constrained_layout=True)
    summaries = []

    for axis, ratio in zip(axes, FIGURE3_NOMINAL_GA_N_RATIOS, strict=True):
        parameters = figure3_parameters(ratio)
        results = [
            run(figure3_config(ratio, lattice_size=LATTICE_SIZE, seed=seed))
            for seed in SEEDS
        ]
        traces = np.vstack(
            [np.interp(TIME_GRID_S, result.time_s, result.rheed_proxy) for result in results]
        )
        mean = traces.mean(axis=0)
        std = traces.std(axis=0)
        ratio_label = f"{ratio:.2f}".replace(".", "")
        np.savez_compressed(
            RUN_DIR / f"figure3_ratio_{ratio_label}.npz",
            time_s=TIME_GRID_S,
            rheed_proxy_traces=traces,
            rheed_proxy_mean=mean,
            rheed_proxy_std=std,
        )
        summaries.append(
            {
                "paper_parameters": parameters.as_dict(),
                "simulation_config": results[0].config.as_dict(),
                "seeds": SEEDS,
                "runs": [
                    {
                        "final_coverage_ml": float(result.coverage_ml[-1]),
                        "final_roughness_ml": float(result.roughness_ml[-1]),
                        "deposited_events": result.deposited_events,
                        "desorbed_events": result.desorbed_events,
                        "equivalent_diffusion_hops": result.diffusion_events,
                        "long_hop_events": result.long_hop_events,
                    }
                    for result in results
                ],
            }
        )
        axis.plot(TIME_GRID_S, mean, color="tab:blue", label="mean")
        axis.fill_between(
            TIME_GRID_S,
            np.clip(mean - std, 0, 1),
            np.clip(mean + std, 0, 1),
            color="tab:blue",
            alpha=0.22,
            label="+/- 1 SD",
        )
        axis.set(ylabel=r"$1-S_d$", ylim=(0, 1.03))
        axis.text(
            0.98,
            0.9,
            rf"$\phi_{{Ga}}/\phi_N={ratio:.2f}$",
            ha="right",
            va="top",
            transform=axis.transAxes,
        )

    axes[-1].set(xlabel="time (s)", xlim=(0, 40))
    axes[0].legend(loc="lower right")
    figure.suptitle(
        "Figure 3 parameterization: simulated step-density proxy\n"
        "7x7 smoke ensemble, 3 seeds"
    )
    figure.savefig(FIGURE_DIR / "figure3_simulated_smoke.png", dpi=160)
    plt.close(figure)
    (RUN_DIR / "figure3_simulated_smoke.json").write_text(
        json.dumps(summaries, indent=2) + "\n"
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
