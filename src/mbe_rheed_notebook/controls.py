"""Input widgets and the selection-to-SimulationConfig mapping for the growth experiment.

Every function here *builds* a marimo UI object and returns it. The notebook cell is what
assigns the result to a global name, which is what marimo requires for interactivity:
https://docs.marimo.io/guides/interactivity/
"""

import re
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import marimo as mo

from mbe_rheed_sim import SimulationConfig, run
from mbe_rheed_sim.kmc import SimulationResult
from mbe_rheed_sim.paper import figure3_config, figure3_parameters
from mbe_rheed_sim.rates import arrhenius_rate

HAND_TUNED = "Hand-tuned parameters"
FROM_PAPER = "GaN parameters from the paper"
PRE_COMPUTED = "Pre-computed demo"
SIMULATE_NOW = "Simulate now"
SAVED_RUN = "Saved run"
DEFAULT_PRESET = "Teaching defaults"

DEFAULT_PARAMETERS = {
    "experiment_mode": HAND_TUNED,
    "figure3_ratio": 0.82,
    "temperature_k": 800,
    "flux_ml_s": 0.5,
    "attempt_frequency_hz": 1_000.0,
    "barrier_ev": 0.15,
    "bond_energy_ev": 0.05,
    "step_barrier_ev": 0.05,
    "desorption_barrier_ev": 0.65,
    "size": 16,
    "stop_mode": "Coverage",
    "coverage_ml": 2.0,
    "duration_s": 4.0,
    "hop_distance": 1,
    "sample_every_ml": 0.05,
    "max_events": None,
    "seed": 7,
}

# (row label, form key, what the quantity does). The help text becomes a hover tooltip on the
# row label; every statement here must match the rate expressions in mbe_rheed_sim.rates.
# Sections listed in `_ADVANCED_SECTIONS` are folded away behind a <details> so the everyday
# form stays six rows tall.
_FIELDS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "Growth conditions": (
        (
            "Parameter source",
            "experiment_mode",
            (
                "Edit every growth condition by hand, or regenerate temperature, effective Ga flux "
                "and all four barriers from the paper's fit for one Ga/N ratio."
            ),
        ),
        (
            "Paper Ga/N condition",
            "figure3_ratio",
            (
                "Nominal Ga/N ratio of the paper's Figure 3. Ignored unless the parameter source is "
                "the paper."
            ),
        ),
        (
            "Temperature (K)",
            "temperature_k",
            (
                "Substrate temperature T. Every thermally activated rate carries exp(-E/kBT), so a "
                "small change in T moves diffusion by orders of magnitude."
            ),
        ),
        (
            "Deposition flux (ML/s)",
            "flux_ml_s",
            (
                "Beam flux F in monolayers per second. The total deposition rate is F times the "
                "number of lattice sites, independent of the current surface."
            ),
        ),
        (
            "Lattice size",
            "size",
            (
                "Edge length of the periodic lattice. Sites, and so the cost of one monolayer, "
                "scale as the square of this number."
            ),
        ),
        (
            "RNG seed",
            "seed",
            (
                "Seeds the random stream. The same seed and parameters reproduce a trajectory "
                "exactly; a different seed is a different sample of the same physics."
            ),
        ),
    ),
    "Energetics": (
        (
            "Attempt frequency",
            "attempt_frequency_hz",
            (
                "Prefactor nu shared by diffusion and desorption. 0 Hz freezes both, leaving pure "
                "random deposition."
            ),
        ),
        (
            "Diffusion barrier (eV)",
            "barrier_ev",
            (
                "E_diff, the barrier an isolated top particle crosses to hop to a neighbouring "
                "column. Larger values mean shorter diffusion lengths and rougher films."
            ),
        ),
        (
            "Lateral bond energy (eV)",
            "bond_energy_ev",
            (
                "E_b, added to the diffusion and desorption barriers once per lateral neighbour, so "
                "particles bound into island edges move and leave far less often."
            ),
        ),
        (
            "Down-step barrier (eV)",
            "step_barrier_ev",
            (
                "Ehrlich-Schwoebel barrier E_step, paid only when crossing one step downward. "
                "Larger values trap particles on top of islands and build mounds."
            ),
        ),
        (
            "Desorption barrier (eV)",
            "desorption_barrier_ev",
            (
                "E_des, the barrier to remove a top particle from the surface. Low values let the "
                "film lose material as fast as it gains it."
            ),
        ),
    ),
    "Numerical controls": (
        (
            "Stop by",
            "stop_mode",
            (
                "Whether the run ends at a target coverage or a target physical time. The target "
                "you did not pick is ignored."
            ),
        ),
        ("Target coverage (ML)", "coverage_ml", "Film thickness to grow, in monolayers."),
        ("Target time (s)", "duration_s", "Simulated physical time to grow for, in seconds."),
        (
            "Isolated-adatom hop limit",
            "hop_distance",
            (
                "1 is exact nearest-neighbour KMC. A larger limit lets an isolated adatom on open "
                "terrace cross several sites in one selected event, which is faster but "
                "approximate; the acceleration-validation workflow measures the difference."
            ),
        ),
        (
            "Sampling interval (ML)",
            "sample_every_ml",
            (
                "How much coverage passes between recorded frames. Smaller values give a smoother "
                "RHEED trace and use more memory."
            ),
        ),
        (
            "Maximum selected events",
            "max_events",
            (
                "Safety limit. The run raises rather than continuing forever if the target is not "
                "reached within this many selected events. `No limit` runs until the target is "
                "reached; interrupt the cell to stop it."
            ),
        ),
    ),
}

_ADVANCED_SECTIONS = ("Energetics", "Numerical controls")


# Sections flow into as many columns as the notebook is wide, each column capped so a single
# section does not stretch a slider across the whole page. Inline styles only: marimo's markdown
# renderer is not guaranteed to keep a <style> block.
_GRID = (
    "display:grid;grid-template-columns:repeat(auto-fit,minmax(21rem,30rem));"
    "gap:0 2.5rem;align-items:start"
)
_ROW_LABEL = "text-align:right;white-space:nowrap;padding:0 .6rem 0 0;vertical-align:middle"


def _layout() -> str:
    """Render `_FIELDS` as the `mo.md(...).batch(...)` template, help text as row tooltips.

    Raw HTML rather than markdown tables: marimo's markdown renderer leaves the body of a
    `<details>` unparsed, so the folded sections have to arrive as HTML anyway.
    """
    blocks = {}
    for section, rows in _FIELDS.items():
        body = "".join(
            # marimo's own tooltip: its RenderHTML wraps any element carrying data-tooltip and
            # styles it with a dotted underline. A plain title= is a bare browser tooltip that
            # takes a second to appear and shows no affordance.
            f'<tr><td style="{_ROW_LABEL}">{label} '
            f'<span data-tooltip="{help_text}" style="cursor:help">&#9432;</span></td>'
            f"<td>{{{key}}}</td></tr>"
            for label, key, help_text in rows
        )
        blocks[section] = (
            f'<div style="min-width:0"><h3>{section}</h3>'
            f'<table style="width:100%"><tbody>{body}</tbody></table></div>'
        )
    advanced = "".join(blocks.pop(section) for section in _ADVANCED_SECTIONS)
    return (
        f'<div style="{_GRID}">{"".join(blocks.values())}</div>'
        + "<details><summary><b>Advanced parameters</b></summary>"
        + f'<div style="{_GRID}">{advanced}</div></details>'
    )


_LAYOUT = _layout()


def _label(options: dict, value) -> str:
    """The dropdown label that carries `value`; dropdowns are keyed by label, not value."""
    for label, option in options.items():
        if option == value:
            return label
    raise KeyError(f"no option for {value!r}")


LATTICE_SIZES = {
    "7 x 7 — paper reduced": 7,
    "8 x 8 — baseline": 8,
    "12 x 12 — tiny": 12,
    "16 x 16 — fast interactive": 16,
    "24 x 24 — detailed interactive": 24,
    "32 x 32 — science check": 32,
    "48 x 48 — extended": 48,
    "64 x 64 — large": 64,
    "96 x 96 — expensive": 96,
    "128 x 128 — benchmark": 128,
    "256 x 256 — paper reference": 256,
}
HOP_DISTANCES = {
    "1 (exact nearest-neighbor KMC)": 1,
    "3 (accelerated)": 3,
    "5 (accelerated)": 5,
    "8 (accelerated)": 8,
    "16 (accelerated)": 16,
}
ATTEMPT_FREQUENCIES = {
    "0 Hz (diffusion frozen)": 0.0,
    "1e3 Hz (teaching)": 1_000.0,
    "1e6 Hz": 1e6,
    "1e9 Hz": 1e9,
    "1e13 Hz (atomistic)": 1e13,
}
EVENT_LIMITS = {
    "No limit": None,
    "2 million": 2_000_000,
    "10 million": 10_000_000,
    "50 million": 50_000_000,
}


def source_selector(gallery: dict) -> tuple[mo.ui.radio, mo.ui.dropdown]:
    """Radio for pre-computed vs live, plus the pre-computed run picker."""
    return (
        mo.ui.radio(
            options=[PRE_COMPUTED, SIMULATE_NOW, SAVED_RUN],
            value=PRE_COMPUTED,
            label="Result source",
        ),
        mo.ui.dropdown(
            {meta["title"]: name for name, meta in gallery.items()},
            value=next(iter(gallery.values()))["title"],
            label="Pre-computed run",
        ),
    )


def saved_browser(saved_dir: Path) -> mo.ui.file_browser:
    """Pick a `.npz` written by `save_result`.

    A file browser reads the directory when the user opens it, so a run saved during this
    session appears without any reactive refresh of the widget.
    """
    saved_dir.mkdir(parents=True, exist_ok=True)
    return mo.ui.file_browser(
        initial_path=saved_dir, filetypes=[".npz"], multiple=False, label="Saved run file"
    )


def save_controls() -> tuple[mo.ui.text, mo.ui.run_button]:
    """Name field and button for storing the active result. Read the button in another cell."""
    return (
        mo.ui.text(placeholder="my_run", label="Save as"),
        mo.ui.run_button(label="Save this result"),
    )


def save_result(result: SimulationResult, saved_dir: Path, name: str) -> str:
    """Write the active result to `saved_dir/<name>.npz` and report where it went."""
    stem = re.sub(r"[^A-Za-z0-9_-]", "_", name.strip()) or "run"
    path = saved_dir / f"{stem}.npz"
    result.save_npz(path)
    return f"Saved to `{path}`. Load it again with **Result source → {SAVED_RUN}**."


def saved_detail(config: SimulationConfig) -> str:
    return (
        f"Reloaded from disk, so nothing was simulated: "
        f"{config.lattice_size}x{config.lattice_size}, seed {config.seed}, "
        f"T = {config.temperature_k:.1f} K, flux = {config.deposition_flux_ml_s:.3f} ML/s, "
        f"max hop {config.max_isolated_hop_distance}."
    )


def preset_selector(gallery: dict) -> mo.ui.dropdown:
    """Pick a stored scenario to load into the live-run form, or start from the defaults."""
    return mo.ui.dropdown(
        {DEFAULT_PRESET: "", **{meta["title"]: name for name, meta in gallery.items()}},
        value=DEFAULT_PRESET,
        label="Start from",
    )


def preset_parameters(meta: dict) -> dict:
    """Form values that rebuild a stored gallery run exactly.

    `tests/test_notebook.py` asserts `build_run` maps the result back to the stored config,
    so a preset can never silently drift from the trajectory it claims to reproduce.
    """
    config = meta["config"]
    from_paper = meta.get("figure3_ratio") is not None
    stop_by_time = config["target_coverage_ml"] is None
    return DEFAULT_PARAMETERS | {
        "experiment_mode": FROM_PAPER if from_paper else HAND_TUNED,
        # The key is present but null for hand-tuned runs, so `.get` default never fires.
        "figure3_ratio": meta.get("figure3_ratio") or DEFAULT_PARAMETERS["figure3_ratio"],
        "temperature_k": config["temperature_k"],
        "flux_ml_s": config["deposition_flux_ml_s"],
        "attempt_frequency_hz": config["attempt_frequency_hz"],
        "barrier_ev": config["diffusion_barrier_ev"],
        "bond_energy_ev": config["lateral_bond_energy_ev"],
        "step_barrier_ev": config["step_barrier_ev"],
        "desorption_barrier_ev": config["desorption_barrier_ev"],
        "size": config["lattice_size"],
        "stop_mode": "Physical time" if stop_by_time else "Coverage",
        "coverage_ml": config["target_coverage_ml"] or DEFAULT_PARAMETERS["coverage_ml"],
        "duration_s": config["target_time_s"] or DEFAULT_PARAMETERS["duration_s"],
        "hop_distance": config["max_isolated_hop_distance"],
        "sample_every_ml": config["sample_every_ml"],
        "max_events": config["max_events"],
        "seed": config["seed"],
    }


def parameter_form(ratios: tuple[float, ...], on_change, values: dict | None = None) -> mo.ui.form:
    """The live-run parameter form, laid out by `_LAYOUT`, opened on `values`."""
    values = DEFAULT_PARAMETERS if values is None else values
    ratio_options = {f"Ga/N = {ratio:.2f}": ratio for ratio in ratios}
    sample_options = {str(value): value for value in (0.01, 0.025, 0.05, 0.1, 0.25)}
    return (
        mo.md(_LAYOUT)
        .batch(
            experiment_mode=mo.ui.radio(
                options=[HAND_TUNED, FROM_PAPER],
                value=values["experiment_mode"],
            ),
            figure3_ratio=mo.ui.dropdown(
                ratio_options,
                value=_label(ratio_options, values["figure3_ratio"]),
            ),
            temperature_k=mo.ui.slider(
                show_value=True,
                start=500,
                stop=1_200,
                step=10,
                value=values["temperature_k"],
            ),
            flux_ml_s=mo.ui.slider(
                show_value=True,
                start=0.05, stop=1.5, step=0.05, value=values["flux_ml_s"]
            ),
            attempt_frequency_hz=mo.ui.dropdown(
                ATTEMPT_FREQUENCIES,
                value=_label(ATTEMPT_FREQUENCIES, values["attempt_frequency_hz"]),
            ),
            barrier_ev=mo.ui.slider(
                show_value=True,
                start=0.05,
                stop=2.5,
                step=0.01,
                value=values["barrier_ev"],
            ),
            bond_energy_ev=mo.ui.slider(
                show_value=True,
                start=0.0,
                stop=0.6,
                step=0.01,
                value=values["bond_energy_ev"],
            ),
            step_barrier_ev=mo.ui.slider(
                show_value=True,
                start=0.0,
                stop=0.3,
                step=0.01,
                value=values["step_barrier_ev"],
            ),
            desorption_barrier_ev=mo.ui.slider(
                show_value=True,
                start=0.2,
                stop=3.0,
                step=0.05,
                value=values["desorption_barrier_ev"],
            ),
            size=mo.ui.dropdown(
                LATTICE_SIZES, value=_label(LATTICE_SIZES, values["size"])
            ),
            stop_mode=mo.ui.radio(
                options=["Coverage", "Physical time"],
                value=values["stop_mode"],
            ),
            coverage_ml=mo.ui.slider(
                show_value=True,
                start=0.25,
                stop=10.0,
                step=0.25,
                value=values["coverage_ml"],
            ),
            duration_s=mo.ui.slider(
                show_value=True,
                start=0.1, stop=40.0, step=0.1, value=values["duration_s"]
            ),
            hop_distance=mo.ui.dropdown(
                HOP_DISTANCES,
                value=_label(HOP_DISTANCES, values["hop_distance"]),
            ),
            sample_every_ml=mo.ui.dropdown(
                sample_options,
                value=_label(sample_options, values["sample_every_ml"]),
            ),
            max_events=mo.ui.dropdown(
                EVENT_LIMITS,
                value=_label(EVENT_LIMITS, values["max_events"]),
            ),
            seed=mo.ui.number(start=0, stop=10_000, value=values["seed"]),
        )
        .form(submit_button_label="Run simulation", bordered=True, on_change=on_change)
    )


# ponytail: order-of-magnitude gate only. Selected KMC events are deposition events times the
# isolated-adatom hops each one survives, divided by the hops an accelerated long jump collapses
# into one selection. Throughput is a single measured constant, and the exact and accelerated
# regimes straddle it by roughly an order of magnitude either way -- fine for a "this may take
# minutes" warning, not a scheduler. Re-measure SELECTED_EVENTS_PER_S on new hardware.
SELECTED_EVENTS_PER_S = 3e5


def estimated_runtime_s(config: SimulationConfig, coverage_ml: float) -> float:
    """Rough wall-clock estimate for one live run, used only to gate expensive launches."""
    hops_per_atom = (
        arrhenius_rate(
            config.attempt_frequency_hz, config.diffusion_barrier_ev, config.temperature_k
        )
        / config.deposition_flux_ml_s
    )
    selected_events = (
        config.lattice_size**2
        * coverage_ml
        * (1.0 + hops_per_atom / config.max_isolated_hop_distance**2)
    )
    return selected_events / SELECTED_EVENTS_PER_S


def _event_limit(parameters: dict) -> int | None:
    """`None` means the run has no event safety limit."""
    limit = parameters["max_events"]
    return None if limit is None else int(limit)


def build_run(parameters: dict) -> tuple[SimulationConfig, float, float | None, str, str]:
    """Map one form submission to (config, runtime estimate, growth rate, name, detail)."""
    size = int(parameters["size"])
    # SimulationConfig rejects a hop longer than half the periodic lattice.
    hop_distance = min(int(parameters["hop_distance"]), max(1, (size - 1) // 2))
    stop_by_time = parameters["stop_mode"] == "Physical time"

    if parameters["experiment_mode"] == FROM_PAPER:
        ratio = float(parameters["figure3_ratio"])
        paper = figure3_parameters(ratio)
        duration_s = (
            float(parameters["duration_s"])
            if stop_by_time
            else float(parameters["coverage_ml"]) / paper.predicted_growth_rate_ml_s
        )
        config = replace(
            figure3_config(
                ratio, lattice_size=size, duration_s=duration_s, seed=int(parameters["seed"])
            ),
            max_isolated_hop_distance=hop_distance,
            sample_every_ml=float(parameters["sample_every_ml"]),
            max_events=_event_limit(parameters),
        )
        coverage = duration_s * paper.predicted_growth_rate_ml_s
        name = f"GaN paper parameters (Ga/N = {ratio:.2f})"
        detail = (
            f"T = {paper.temperature_k:.2f} K; effective Ga flux = "
            f"{paper.effective_ga_flux_ml_s:.4f} ML/s; predicted growth rate = "
            f"{paper.predicted_growth_rate_ml_s:.4f} ML/s; {size}x{size}; "
            f"target {duration_s:.2f} s; seed {config.seed}."
        )
        growth_rate = paper.predicted_growth_rate_ml_s
    else:
        target_coverage_ml = None if stop_by_time else float(parameters["coverage_ml"])
        target_time_s = float(parameters["duration_s"]) if stop_by_time else None
        config = SimulationConfig(
            lattice_size=size,
            target_coverage_ml=target_coverage_ml,
            target_time_s=target_time_s,
            temperature_k=float(parameters["temperature_k"]),
            deposition_flux_ml_s=float(parameters["flux_ml_s"]),
            attempt_frequency_hz=float(parameters["attempt_frequency_hz"]),
            diffusion_barrier_ev=float(parameters["barrier_ev"]),
            lateral_bond_energy_ev=float(parameters["bond_energy_ev"]),
            step_barrier_ev=float(parameters["step_barrier_ev"]),
            desorption_barrier_ev=float(parameters["desorption_barrier_ev"]),
            max_isolated_hop_distance=hop_distance,
            sample_every_ml=float(parameters["sample_every_ml"]),
            seed=int(parameters["seed"]),
            max_events=_event_limit(parameters),
        )
        coverage = (
            target_coverage_ml
            if target_coverage_ml is not None
            else target_time_s * config.deposition_flux_ml_s
        )
        target = (
            f"{config.target_time_s:.2f} s"
            if target_time_s is not None
            else f"{config.target_coverage_ml:.2f} ML"
        )
        name = "Hand-tuned experiment"
        detail = (
            f"T = {config.temperature_k:.0f} K; flux = "
            f"{config.deposition_flux_ml_s:.3f} ML/s; {size}x{size}; "
            f"target {target}; seed {config.seed}."
        )
        growth_rate = None

    estimate = estimated_runtime_s(config, coverage)
    detail = (
        f"{detail} Estimated {estimate:.1f} s; hop limit {hop_distance}; sampled every "
        f"{config.sample_every_ml:g} ML; event limit "
        f"{'none' if config.max_events is None else format(config.max_events, ',')}."
    )
    return config, estimate, growth_rate, name, detail


def is_expensive(config: SimulationConfig, estimate_s: float) -> bool:
    return config.lattice_size >= 64 or estimate_s >= 20.0


def expensive_warning(estimate_s: float, override_button) -> mo.Html:
    return mo.vstack(
        [
            mo.callout(
                f"This run is estimated at roughly **{estimate_s:.0f} s** and is "
                "single-threaded. The estimate is order-of-magnitude only; the true cost "
                "depends strongly on the rate constants.",
                kind="warn",
            ),
            override_button,
        ]
    )


def run_with_progress(config: SimulationConfig, title: str) -> SimulationResult:
    """Run the KMC behind a marimo progress bar reporting elapsed time and an ETA."""
    steps = 100
    with mo.status.progress_bar(total=steps, title=title, remove_on_exit=True) as bar:
        state = {"shown": 0, "start": perf_counter()}

        def advance(fraction: float) -> None:
            elapsed = perf_counter() - state["start"]
            # ponytail: ETA assumes cost is uniform per unit coverage, which understates the
            # tail as the surface fills. Good enough to watch; not a benchmark.
            remaining = elapsed * (1.0 - fraction) / fraction if fraction > 0 else 0.0
            done = min(steps, int(fraction * steps))
            bar.update(
                increment=done - state["shown"],
                subtitle=f"{fraction:.0%} — {elapsed:.0f} s elapsed, ~{remaining:.0f} s left",
            )
            state["shown"] = done

        return run(config, on_progress=advance)


def gallery_detail(meta: dict, config: SimulationConfig) -> str:
    period = meta["oscillation_period_ml"]
    return (
        f"{meta['story']}\n\nStored run: {config.lattice_size}x{config.lattice_size}, "
        f"seed {config.seed}, {meta['frames']} recorded frames. Measured oscillation period "
        + (f"{period:.2f} ML." if period else "not resolved (no layer cycle).")
    )
