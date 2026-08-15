"""Pre-compute the demonstration runs the notebook offers instead of simulating live.

These are committed so a talk or demo never waits on a KMC run. Regenerate with
`make gallery` after any change to the model.
"""

import json
from dataclasses import asdict, replace
from pathlib import Path

from mbe_rheed_sim import SimulationConfig, run
from mbe_rheed_sim.analysis import rheed_oscillation_metrics
from mbe_rheed_sim.paper import figure3_config, figure3_parameters
from mbe_rheed_sim.workflows import run_parallel

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "gallery"

# Each entry is one story the notebook can tell without running anything.
BASE = SimulationConfig(
    lattice_size=24,
    target_coverage_ml=3.0,
    sample_every_ml=0.05,
    seed=7,
    desorption_barrier_ev=1.2,  # suppress desorption so morphology is the only story
)
# Ga/N condition behind the paper entry. Written into the index so the notebook can reload
# this run into the live-run form in paper mode instead of guessing the ratio back.
PAPER_RATIO = 0.82

ENTRIES = {
    "gan-paper-082": {
        "title": "Layer-by-layer growth (GaN paper parameters)",
        "story": (
            "Temperature, effective Ga flux, and all four barriers come from the paper's "
            "fitted expressions for Ga/N = 0.82, not from the sliders. Adatoms are mobile "
            "enough to complete each layer before the next nucleates, so the surface returns "
            "to nearly flat every monolayer and the proxy oscillates with a ~1 ML period. "
            "This is the textbook RHEED oscillation the whole model is aiming at."
        ),
        "config": figure3_config(PAPER_RATIO, lattice_size=7, duration_s=40.0, seed=7),
        "figure3_ratio": PAPER_RATIO,
    },
    "island-growth": {
        "title": "Island growth (damped oscillations)",
        "story": (
            "With the teaching parameters the diffusion length is comparable to the lattice, "
            "so new layers start before the one below closes. The proxy still oscillates but "
            "irregularly and with a decaying envelope: real growth, not the ideal limit."
        ),
        "config": replace(BASE, temperature_k=900.0, step_barrier_ev=0.0),
    },
    "step-barrier-mounding": {
        "title": "Mounding from an Ehrlich-Schwoebel barrier",
        "story": (
            "The same conditions plus a large down-step barrier. Adatoms that land on top of "
            "an island cannot descend, so material piles into mounds. This is the roughest "
            "run in the set and its oscillation dies fastest."
        ),
        "config": replace(BASE, temperature_k=900.0, step_barrier_ev=0.25),
    },
    "too-cold": {
        "title": "Too cold: dense nucleation",
        "story": (
            "Dropping to 650 K shortens the diffusion length. Islands nucleate densely, layers "
            "never fully close, and the surface ends rougher than the 900 K run."
        ),
        "config": replace(BASE, temperature_k=650.0),
    },
    "too-fast": {
        "title": "Flux too high",
        "story": (
            "900 K with three times the flux. Adatoms are buried before they reach a step "
            "edge, so the surface roughens even though the temperature is unchanged. Compare "
            "with the cold run: here diffusion is fast but there is less time per layer."
        ),
        "config": replace(BASE, temperature_k=900.0, deposition_flux_ml_s=1.5),
    },
    "no-diffusion": {
        "title": "No diffusion: pure random deposition",
        "story": (
            "Switching diffusion off leaves uncorrelated random deposition, whose roughness "
            "grows as the square root of coverage and never oscillates at all. Note it is not "
            "the roughest run here: weak-but-nonzero diffusion builds islands and mounds that "
            "are rougher than random noise."
        ),
        "config": replace(BASE, attempt_frequency_hz=0.0),
    },
}

# The stories above make ordering claims. Fail the build rather than ship a false caption.
EXPECTED_ROUGHNESS_ORDER = (
    "gan-paper-082",
    "island-growth",
    "no-diffusion",
    "too-fast",
    "too-cold",
    "step-barrier-mounding",
)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    names = list(ENTRIES)
    results = run_parallel(
        run,
        [ENTRIES[name]["config"] for name in names],
        workers=len(names),
        description="pre-computed gallery",
    )
    index = {}
    for name, result in zip(names, results, strict=True):
        result.save_npz(OUTPUT / f"{name}.npz")
        entry = ENTRIES[name]
        # The paper entry runs on a physical clock, so record how coverage was derived.
        growth_rate = (
            figure3_parameters(PAPER_RATIO).predicted_growth_rate_ml_s
            if name == "gan-paper-082"
            else None
        )
        metrics = rheed_oscillation_metrics(
            result.time_s * growth_rate if growth_rate else result.coverage_ml,
            result.rheed_proxy,
        )
        index[name] = {
            "title": entry["title"],
            "story": entry["story"],
            "config": asdict(result.config),
            "predicted_growth_rate_ml_s": growth_rate,
            "figure3_ratio": entry.get("figure3_ratio"),
            "final_roughness_ml": float(result.roughness_ml[-1]),
            "oscillation_period_ml": metrics.period_ml,
            "is_oscillatory": metrics.is_oscillatory,
            "frames": len(result.snapshots),
            "bytes": (OUTPUT / f"{name}.npz").stat().st_size,
        }
        print(f"{name:20s} {index[name]['bytes'] / 1024:7.0f} KiB  {index[name]['frames']} frames")
    ordered = sorted(index, key=lambda name: index[name]["final_roughness_ml"])
    if tuple(ordered) != EXPECTED_ROUGHNESS_ORDER:
        raise RuntimeError(
            "gallery captions claim a roughness ordering the runs no longer produce:\n"
            f"  expected {EXPECTED_ROUGHNESS_ORDER}\n  measured {tuple(ordered)}"
        )
    if index["gan-paper-082"]["oscillation_period_ml"] is None or not (
        0.8 <= index["gan-paper-082"]["oscillation_period_ml"] <= 1.2
    ):
        raise RuntimeError("the layer-by-layer caption requires a ~1 ML period")
    if index["no-diffusion"]["is_oscillatory"]:
        raise RuntimeError("the random-deposition caption requires no oscillation")

    (OUTPUT / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(f"total {sum(item['bytes'] for item in index.values()) / 1024:.0f} KiB")


if __name__ == "__main__":
    main()
