"""Regenerate the small deterministic baseline run and its summary figure."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mbe_rheed_sim import SimulationConfig, run

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "outputs" / "runs"
FIGURE_DIR = ROOT / "outputs" / "figures"

CONFIG = SimulationConfig(
    lattice_size=8,
    target_coverage_ml=1.0,
    temperature_k=800.0,
    deposition_flux_ml_s=0.5,
    attempt_frequency_hz=1_000.0,
    diffusion_barrier_ev=0.15,
    lateral_bond_energy_ev=0.05,
    sample_every_ml=0.125,
    seed=2026,
    max_events=100_000,
)
EXPECTED = {
    "deposited_events": 64,
    "diffusion_events": 777,
    "height_sha256": "7986adeaf41fe2b06b43b7b0d6aaecf7aec52128c8811820eb79ca25ef04a47a",
}


def main() -> None:
    result = run(CONFIG)
    height_sha256 = hashlib.sha256(result.final_heights.tobytes()).hexdigest()
    actual = {
        "deposited_events": result.deposited_events,
        "diffusion_events": result.diffusion_events,
        "height_sha256": height_sha256,
    }
    if actual != EXPECTED:
        raise RuntimeError(f"baseline changed: expected {EXPECTED}, got {actual}")

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    result.save_npz(RUN_DIR / "baseline.npz")

    summary = {
        "config": CONFIG.as_dict(),
        **actual,
        "simulation_time_s": float(result.time_s[-1]),
        "roughness_ml": float(result.roughness_ml[-1]),
        "island_density_per_site": float(result.island_density_per_site[-1]),
        "rheed_proxy": float(result.rheed_proxy[-1]),
    }
    (RUN_DIR / "baseline.json").write_text(json.dumps(summary, indent=2) + "\n")

    figure, axes = plt.subplots(1, 3, figsize=(11, 3.2), constrained_layout=True)
    image = axes[0].imshow(result.final_heights, origin="lower", cmap="viridis", vmin=0)
    figure.colorbar(image, ax=axes[0], label="height (ML)")
    axes[0].set(title="Final surface", xlabel="lattice x", ylabel="lattice y")
    axes[1].plot(result.coverage_ml, result.roughness_ml)
    axes[1].set(xlabel="coverage (ML)", ylabel="RMS roughness (ML)")
    axes[2].plot(result.coverage_ml, result.rheed_proxy, color="tab:red")
    axes[2].set(xlabel="coverage (ML)", ylabel=r"$1-S_d$ proxy", ylim=(0, 1.03))
    figure.savefig(FIGURE_DIR / "baseline.png", dpi=150)
    plt.close(figure)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

