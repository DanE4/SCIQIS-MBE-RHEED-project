# MBE growth and RHEED oscillations

Kinetic Monte Carlo (KMC) simulation of molecular-beam-epitaxy growth. Deposition, thermally
activated surface diffusion, downward step barriers and desorption on a periodic solid-on-solid
lattice, with a step-density RHEED proxy plotted alongside the surface morphology.

The RHEED signal here is a step-density proxy, not a diffraction calculation. Scope and known
limitations: [`STATUS.md`](STATUS.md). Source paper: `nanomaterials-12-03052.pdf` (CC BY).

## Setup

Requires [Git](https://git-scm.com/), [uv](https://docs.astral.sh/uv/getting-started/installation/),
and optionally `make`. Python 3.12.0 is pinned in `.python-version`, dependencies in `uv.lock`;
uv installs both.

```bash
git clone <repo>
cd SCIQIS-MBE-RHEED-project
uv sync          # or: make setup, which refuses to change the lockfile
```

## Notebook

```bash
make notebook    # uv run marimo edit notebooks/mbe_rheed.py
```

Section **5** asks where results come from:

- **Pre-computed demo** (default) loads a stored trajectory from `data/gallery/` instantly. Good
  for presenting. Six runs: layer-by-layer growth with the paper's GaN parameters, island growth,
  step-barrier mounding, too-cold and too-high-flux roughening, pure random deposition.
  `make gallery` rebuilds them and fails if the measured roughness ordering or oscillation
  periods stop matching the captions.
- **Simulate now** runs the model live, with either hand-tuned parameters or the paper's fitted
  GaN values. Nothing runs until **Run simulation** is pressed. Runs estimated above 20 s, or on
  a 64x64-or-larger lattice, ask for a second confirmation.

One trajectory cannot use more than one core, since each event mutates the surface the next event
is sampled from. Cores only help across independent runs, which is what section **11. Batch
workflows** is for; it drives the same CLI as `make` and reloads promoted data in place.

## Commands

Every workflow name is a make target and a `scripts/run_workflow.py` argument:

| Command | What it does |
|---|---|
| `make reproduce` | 8x8, 1 ML, seed 2026 baseline; checks event counts and the final-height SHA-256 |
| `make figure3` | three seeds x three Ga/N ratios over 40 s vs. the published Figure 3 |
| `make figure3-parameters` | prints the Figure 3 flux conversion, fitted barriers, rate diagnostics |
| `make sweep` | 16x16 three-seed temperature/flux heatmap (notebook input) |
| `make convergence`, `make figure3-convergence` | finite-size sensitivity; add `SIZES=8,16,32,64` for the ~2 min 64x64 point |
| `make validate-acceleration`, `-science`, `-sweep` | accelerated vs. exact ensembles and model trends |
| `make benchmark-sizes` | sequential 64/128/256 runtime envelope |
| `make gallery` | rebuild the notebook demos |
| `make test`, `make check`, `make export` | pytest; Ruff + strict marimo check + notebook run; HTML export |

Output goes to `outputs/`, which is Git-ignored and can always be rebuilt. Committed notebook
inputs live in `data/processed/`, figure-derived reference curves in `data/reference/`.

`make digitize-figure3` re-extracts the reference curves from page 10 of the paper. It is the
only command that needs a non-Python tool (`pdftocairo` from Poppler), and it is not part of
setup, CI or normal reproduction. What it extracts are panel coordinates read off the figure, not
raw experimental values, and they stay separate from the model's own `1-S_d`.

## Execution details

**Backend.** The hot path runs through compiled Numba kernels. They repeat the reference NumPy
path's floating-point operations in the same order, so trajectories are bit-identical either way
and `MBE_KMC_BACKEND` only changes speed:

```bash
MBE_KMC_BACKEND=reference make test   # pure NumPy, no compiler
MBE_KMC_BACKEND=fast make figure3     # error out instead of falling back silently
```

Compilation happens once per machine (~1.2 s, cached in `src/mbe_rheed_sim/__pycache__`), so
timing one short run measures cache loading, not throughput.

**Workers.** Independent seeds and parameter points run in bounded `spawn` processes. Precedence:
`--workers`, then `MBE_WORKERS`, then `min(10, os.cpu_count() - 1)`. Convergence sizes run one
after another while the seeds within a size run concurrently, to keep memory down. Baseline
reproduction and the timing benchmark stay sequential, since running them in parallel would
defeat the point. On the M4 Pro I develop on, throughput saturates around 8 workers.

```bash
make figure3 WORKERS=4
make convergence WORKERS=1 SIZES=8,16 SEEDS=0,1   # debugging/CI fallback
```

**Provenance.** Each run writes to `outputs/batches/<UTC timestamp>-<workflow>/` with a manifest,
progress log, stdout/stderr, source revision, configuration and artifacts. Only a run that
finished cleanly takes the promotion lock and atomically replaces the canonical files, so a
failed or interrupted batch can never become a notebook input.

**Logs.** Stage and completed-count progress goes to **stderr**, so `make <workflow>` shows
progress while the JSON summary on stdout stays pipeable. `MBE_LOG_LEVEL=WARNING|DEBUG` turns the
volume down or up; `2>/dev/null | jq .` keeps only the JSON.

## Layout

```text
notebooks/mbe_rheed.py       narrative and reactive wiring only (~650 lines)
src/mbe_rheed_sim/           physics and reproducible execution (no marimo, no plotting)
src/mbe_rheed_notebook/      notebook widgets and figures (marimo + Plotly)
scripts/                     baseline, Figure 3, validation, sweep entry points
tests/                       scientific and software invariants
data/                        gallery demos, committed inputs, reference curves
```

A marimo notebook is one file and cannot be split, so everything except narrative and wiring is
imported from `src/mbe_rheed_notebook/`. That also makes the widget and figure logic testable
without launching a notebook (`tests/test_notebook.py`). Two marimo rules to remember when
editing: a `mo.ui` element only becomes interactive if a cell assigns it to a global, and a cell
may not reassign a variable another cell defines (hence the `_`-prefixed locals).

## Troubleshooting

- **`uv: command not found`**: install uv, rerun `uv sync`.
- **Python 3.12.0 missing**: `uv python install`.
- **Lockfile mismatch**: `uv lock`, check the dependency change, commit `uv.lock`.
- **Marimo token or port trouble**: close old tabs, then `make notebook MARIMO_PORT=2721`.
- **Import errors**: run commands from the repository root after `uv sync`.
- **Baseline hash changed**: don't just update it. Find out which model or dependency change moved
  it first.
