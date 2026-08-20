"""Pre-compute the demonstration runs the notebook offers instead of simulating live.

These are committed so a talk or demo never waits on a KMC run, and so a laptop that cannot
afford a large lattice still gets one. Regenerate with `make gallery` after any change to the
model. `make gallery SIZES=96 WORKERS=6` trades statistics for a smaller build; above 128 the
paper entry needs more events than the limit `figure3_config` sets, and it stops rather than
quietly truncating.

One entry ignores that size. `stranski-krastanov` is always recorded at 256, because the ordered
phase the regime switch draws over it is only fully resolved there and nobody should have to wait
two minutes for a live run to see it.
"""

import json
from dataclasses import asdict, replace
from pathlib import Path

from mbe_rheed_sim import SimulationConfig, run
from mbe_rheed_sim.analysis import rheed_oscillation_metrics
from mbe_rheed_sim.kmc import SimulationResult
from mbe_rheed_sim.paper import figure3_config, figure3_parameters
from mbe_rheed_sim.workflows import (
    artifact_root,
    log_progress,
    parse_workflow_args,
    run_parallel,
    setup_logging,
)

ROOT = Path(__file__).resolve().parents[1]
# Every size here must be one the notebook's lattice dropdown offers, or the preset that
# reloads a stored run into the live form cannot express it (tests/test_notebook.py).
# 128 is the largest that keeps the paper entry inside the 10-million event safety limit
# `figure3_config` sets, and the whole set inside a few MB of committed data.
LATTICE_SIZE = 128

# Each entry is one story the notebook can tell without running anything.
BASE = SimulationConfig(
    lattice_size=LATTICE_SIZE,
    target_coverage_ml=3.0,
    sample_every_ml=0.05,
    seed=7,
    desorption_barrier_ev=1.2,  # suppress desorption so morphology is the only story
    # Events scale with the site count, so the 2-million default aborts these runs above
    # about a 48 lattice. 50 million is the largest limit the notebook form can express.
    max_events=50_000_000,
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
        "config": figure3_config(
            PAPER_RATIO, lattice_size=LATTICE_SIZE, duration_s=40.0, seed=7
        ),
        "figure3_ratio": PAPER_RATIO,
    },
    "island-growth": {
        "title": "Island growth (oscillation lost to island nucleation)",
        "story": (
            "The default parameters give a diffusion length of a few sites, far shorter than "
            "this lattice, so islands nucleate independently all over the surface and every "
            "new layer starts long before the one below closes. Averaged over that many "
            "uncorrelated patches the proxy stops oscillating altogether. The same parameters "
            "on a 24x24 lattice do oscillate, with a 0.8 ML period, and at 48x48 with 0.54 ML: "
            "neither is the ~1 ML layer-by-layer period, so that signal was the small surface "
            "completing layers in step, not real layer-by-layer growth. This is what a "
            "finite-size artifact looks like when you grow the lattice out of it."
        ),
        "config": replace(BASE, temperature_k=900.0, step_barrier_ev=0.0),
    },
    "step-barrier-mounding": {
        "title": "Mounding from an Ehrlich-Schwoebel barrier",
        "story": (
            "The same conditions plus a large down-step barrier. Adatoms that land on top of "
            "an island cannot descend, so material piles into mounds instead of filling the "
            "layer below. This is the roughest run in the set."
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
    "stranski-krastanov": {
        "title": "Stranski-Krastanov regime (256x256)",
        "story": (
            "The paper's Ga/N = 0.82 physics again, on the largest lattice the notebook offers "
            "and stopped at roughly the two monolayers that make up a wetting layer. On its own "
            "this is the layer-by-layer entry at higher resolution and a shorter clock. Its "
            "purpose is the switch it arrives with: the **Stranski-Krastanov regime** carries "
            "the run past its target coverage into the ordered phase strained GaN on AlN reaches "
            "once the wetting layer is complete. The model has no strain and no reconstruction, "
            "so those frames are prescribed rather than simulated - but 256 sites is where the "
            "ordered structure is fully resolved, which is why this run is recorded at that size."
        ),
        "config": figure3_config(PAPER_RATIO, lattice_size=256, duration_s=8.5, seed=7),
        "figure3_ratio": PAPER_RATIO,
        # Recorded at the size in the config above on purpose: below 256 the ordered phase loses
        # its detail, and the point of this entry is to have that size available without a
        # two-minute live run. So it ignores the build's `SIZES`.
        "fixed_size": True,
        # Loads with the regime switch already on, so the view it exists for is the one shown.
        "reconstruction": True,
    },
    "no-diffusion": {
        "title": "No diffusion: pure random deposition",
        "story": (
            "Switching diffusion off leaves uncorrelated random deposition, whose roughness "
            "grows as the square root of coverage and never oscillates at all. Note it is not "
            "the roughest run here: weak-but-nonzero diffusion builds islands and mounds that "
            "are rougher than random noise. Temperature, flux, barriers and seed match the "
            "island-growth entry, so the notebook can put the two side by side and change only "
            "the attempt frequency."
        ),
        "config": replace(
            BASE, temperature_k=900.0, step_barrier_ev=0.0, attempt_frequency_hz=0.0
        ),
    },
}

# The stories above make ordering claims. Fail the build rather than ship a false caption.
# Grouped, because that is how strong the claims are: the captions rank the cold and the
# fast run above the random one and below the mounded one, but say nothing about which of
# the two is rougher, and on a large lattice they land within noise of each other.
EXPECTED_ROUGHNESS_ORDER = (
    ("gan-paper-082",),
    ("island-growth",),
    ("no-diffusion",),
    ("too-cold", "too-fast"),
    ("step-barrier-mounding",),
)


def _entry_config(entry: dict, size: int) -> SimulationConfig:
    """One gallery configuration at the build's lattice size."""
    if entry.get("fixed_size"):
        # Opted out of the build size: the size it is recorded at is the one in its own config.
        return entry["config"]
    if entry.get("figure3_ratio"):
        return figure3_config(PAPER_RATIO, lattice_size=size, duration_s=40.0, seed=7)
    return replace(entry["config"], lattice_size=size)


def _configs(size: int) -> dict[str, SimulationConfig]:
    """The gallery configurations at one lattice size."""
    return {name: _entry_config(entry, size) for name, entry in ENTRIES.items()}


def _run_entry(item: tuple[str, SimulationConfig]) -> SimulationResult:
    """Run one entry, reporting its own percentage as it goes.

    Spawn workers start with no logging configuration, so without the `setup_logging` here
    the per-run lines would be dropped and a multi-minute build would look frozen.
    """
    name, config = item
    setup_logging()
    return run(config, on_progress=log_progress(name))


def _check_roughness_order(index: dict) -> None:
    """Reject the build if the measured roughness contradicts a caption."""
    ranked = {name for group in EXPECTED_ROUGHNESS_ORDER for name in group}
    measured = sorted(
        (name for name in index if name in ranked),
        key=lambda name: index[name]["final_roughness_ml"],
    )
    position = 0
    for group in EXPECTED_ROUGHNESS_ORDER:
        if set(measured[position : position + len(group)]) != set(group):
            raise RuntimeError(
                "gallery captions claim a roughness ordering the runs no longer produce:\n"
                f"  expected {EXPECTED_ROUGHNESS_ORDER}\n  measured {tuple(measured)}"
            )
        position += len(group)


def index_entry(name: str, result: SimulationResult, output: Path) -> dict:
    """Save one run and describe it the way the notebook's index expects.

    Separate from `main` so a single entry can be rebuilt on its own without the description
    drifting from the one a full `make gallery` would write.
    """
    result.save_npz(output / f"{name}.npz")
    entry = ENTRIES[name]
    # Paper-physics entries run on a physical clock, so record how coverage was derived.
    growth_rate = (
        figure3_parameters(PAPER_RATIO).predicted_growth_rate_ml_s
        if entry.get("figure3_ratio")
        else None
    )
    metrics = rheed_oscillation_metrics(
        result.time_s * growth_rate if growth_rate else result.coverage_ml,
        result.rheed_proxy,
    )
    return {
        "title": entry["title"],
        "story": entry["story"],
        "config": asdict(result.config),
        "predicted_growth_rate_ml_s": growth_rate,
        "figure3_ratio": entry.get("figure3_ratio"),
        "reconstruction": entry.get("reconstruction", False),
        "final_roughness_ml": float(result.roughness_ml[-1]),
        "oscillation_period_ml": metrics.period_ml,
        "is_oscillatory": metrics.is_oscillatory,
        "frames": len(result.snapshots),
        "bytes": (output / f"{name}.npz").stat().st_size,
    }


def main(*, workers: int = len(ENTRIES), sizes: tuple[int, ...] = (LATTICE_SIZE,)) -> None:
    setup_logging()
    if len(sizes) != 1:
        raise ValueError("the gallery is built at one lattice size")
    (size,) = sizes
    # Checked before the runs, not after: a size the notebook's dropdown cannot express
    # would only fail in tests/test_notebook.py, hours later.
    from mbe_rheed_notebook.controls import LATTICE_SIZES

    if size not in LATTICE_SIZES.values():
        raise ValueError(f"lattice size must be one the notebook offers: {sorted(LATTICE_SIZES.values())}")
    output = artifact_root(ROOT) / "data" / "gallery"
    output.mkdir(parents=True, exist_ok=True)
    configs = _configs(size)
    names = list(configs)
    results = run_parallel(
        _run_entry,
        [(name, configs[name]) for name in names],
        workers=workers,
        description=f"pre-computed gallery at {size}x{size}",
    )
    index = {}
    for name, result in zip(names, results, strict=True):
        index[name] = index_entry(name, result, output)
        print(f"{name:20s} {index[name]['bytes'] / 1024:7.0f} KiB  {index[name]['frames']} frames")
    _check_roughness_order(index)
    if index["gan-paper-082"]["oscillation_period_ml"] is None or not (
        0.8 <= index["gan-paper-082"]["oscillation_period_ml"] <= 1.2
    ):
        raise RuntimeError("the layer-by-layer caption requires a ~1 ML period")
    if index["no-diffusion"]["is_oscillatory"]:
        raise RuntimeError("the random-deposition caption requires no oscillation")
    if index["island-growth"]["is_oscillatory"]:
        raise RuntimeError(
            "the island-growth caption says the proxy stops oscillating once the lattice is "
            "much larger than the diffusion length; it oscillates at this size, so either "
            "rebuild larger or rewrite the caption"
        )

    (output / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(f"total {sum(item['bytes'] for item in index.values()) / 1024:.0f} KiB")


if __name__ == "__main__":
    main(**parse_workflow_args(sizes=(LATTICE_SIZE,)))
