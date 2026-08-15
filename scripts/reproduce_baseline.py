"""Regenerate the small deterministic baseline run and its summary figure."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mbe_rheed_sim import SimulationConfig, run
from mbe_rheed_sim.workflows import artifact_root, update_progress

ROOT = Path(__file__).resolve().parents[1]

CONFIG = SimulationConfig(
    lattice_size=8,
    target_coverage_ml=1.0,
    temperature_k=800.0,
    deposition_flux_ml_s=0.5,
    attempt_frequency_hz=1_000.0,
    diffusion_barrier_ev=0.15,
    lateral_bond_energy_ev=0.05,
    step_barrier_ev=0.05,
    desorption_barrier_ev=0.65,
    sample_every_ml=0.125,
    seed=2026,
    max_events=100_000,
)
EXPECTED = {
    "deposited_events": 67,
    "diffusion_events": 1_416,
    "desorbed_events": 3,
    "height_sha256": "be1e5258db1f41a314c8d281b229f8cec68c6856d66573c62854297d1be81679",
}


def main() -> None:
    update_progress(stage="deterministic baseline", completed=0, total=1, effective_workers=1)
    result = run(CONFIG)
    height_sha256 = hashlib.sha256(result.final_heights.tobytes()).hexdigest()
    actual = {
        "deposited_events": result.deposited_events,
        "diffusion_events": result.diffusion_events,
        "desorbed_events": result.desorbed_events,
        "height_sha256": height_sha256,
    }
    if actual != EXPECTED:
        raise RuntimeError(f"baseline changed: expected {EXPECTED}, got {actual}")

    output_root = artifact_root(ROOT)
    run_dir = output_root / "outputs" / "runs"
    figure_dir = output_root / "outputs" / "figures"
    run_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    result.save_npz(run_dir / "baseline.npz")

    summary = {
        "config": asdict(CONFIG),
        **actual,
        "simulation_time_s": float(result.time_s[-1]),
        "roughness_ml": float(result.roughness_ml[-1]),
        "island_density_per_site": float(result.island_density_per_site[-1]),
        "rheed_proxy": float(result.rheed_proxy[-1]),
    }
    (run_dir / "baseline.json").write_text(json.dumps(summary, indent=2) + "\n")

    figure, axes = plt.subplots(1, 3, figsize=(11, 3.2), constrained_layout=True)
    image = axes[0].imshow(result.final_heights, origin="lower", cmap="viridis", vmin=0)
    figure.colorbar(image, ax=axes[0], label="height (ML)")
    axes[0].set(title="Final surface", xlabel="lattice x", ylabel="lattice y")
    axes[1].plot(result.coverage_ml, result.roughness_ml)
    axes[1].set(xlabel="coverage (ML)", ylabel="RMS roughness (ML)")
    axes[2].plot(result.coverage_ml, result.rheed_proxy, color="tab:red")
    axes[2].set(xlabel="coverage (ML)", ylabel=r"$1-S_d$ proxy", ylim=(0, 1.03))
    figure.savefig(figure_dir / "baseline.png", dpi=150)
    plt.close(figure)

    update_progress(stage="deterministic baseline", completed=1, total=1)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
