"""Generate a small deterministic temperature/flux RHEED-amplitude map."""

import json
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mbe_rheed_sim import SimulationConfig
from mbe_rheed_sim.analysis import oscillation_amplitude, rheed_proxy_ensemble

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "outputs" / "runs"
FIGURE_DIR = ROOT / "outputs" / "figures"
TEMPERATURES_K = (700.0, 850.0, 1_000.0)
FLUXES_ML_S = (0.25, 0.5, 0.75)
SEEDS = (0, 1, 2)
BASE = SimulationConfig(
    lattice_size=7,
    target_coverage_ml=2.0,
    sample_every_ml=0.05,
    max_isolated_hop_distance=3,
)


def main() -> None:
    mean_amplitude = np.empty((len(TEMPERATURES_K), len(FLUXES_ML_S)))
    std_amplitude = np.empty_like(mean_amplitude)
    for temperature_index, temperature in enumerate(TEMPERATURES_K):
        for flux_index, flux in enumerate(FLUXES_ML_S):
            _, traces = rheed_proxy_ensemble(
                replace(BASE, temperature_k=temperature, deposition_flux_ml_s=flux),
                SEEDS,
            )
            amplitudes = np.array([oscillation_amplitude(trace) for trace in traces])
            mean_amplitude[temperature_index, flux_index] = amplitudes.mean()
            std_amplitude[temperature_index, flux_index] = amplitudes.std()

    summary = {
        "base_config": BASE.as_dict(),
        "temperatures_k": TEMPERATURES_K,
        "fluxes_ml_s": FLUXES_ML_S,
        "seeds": SEEDS,
        "amplitude_definition": "half of the 95th-minus-5th percentile proxy range",
        "mean_amplitude": mean_amplitude.tolist(),
        "std_amplitude": std_amplitude.tolist(),
    }
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "parameter_sweep.json").write_text(json.dumps(summary, indent=2) + "\n")

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
    figure.savefig(FIGURE_DIR / "parameter_sweep.png", dpi=160)
    plt.close(figure)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
