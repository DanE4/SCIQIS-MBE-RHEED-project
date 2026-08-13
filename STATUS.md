# Project status

## Project goal

Build a validated kinetic Monte Carlo model of molecular-beam epitaxy, connect simulated
surface morphology to a carefully labelled step-density RHEED proxy, reproduce the paper's
Figure 3 homoepitaxial behavior, and present the result as a polished interactive Marimo
experiment.

## Current state

- **Active stage:** Stage 4 - polished Marimo virtual experiment.
- **Estimated completion:** roughly 77% of the intended final project.
- **Working baseline:** deterministic deposition/diffusion/desorption KMC and Marimo notebook.
- **Main scientific gap:** Figure 3 periodicity is reproduced on a 7x7 smoke lattice, but the
  first paper-regime size study shows that its proxy amplitude is strongly finite-size affected;
  paper-scale convergence and experimental-curve normalization remain unresolved.
- **Main presentation gap:** the teaching flow, designed controls, playback, and synchronized
  Plotly view are working; accurate alternate display modes and the paper-results view remain.
- **Current computational limit:** local event-catalogue updates reduce a paper-derived 64x64,
  4 s run from 131 s to 39 s, but the practical three-seed check reaches 32x32 and is not
  converged in proxy amplitude. Full 40 s, 64x64 publication convergence remains Stage 5 work.

## Roadmap overview

- [x] Stage 0 - Reproducible project foundation
- [x] Stage 1 - Correct generic KMC event catalogue
- [x] Stage 2 - Reproduce Figure 3 homoepitaxial RHEED behavior
- [x] Stage 3 - Ensembles and parameter studies
- [ ] **Stage 4 - Polished Marimo + Plotly virtual experiment (current)**
- [ ] Stage 5 - Publication figures and paper-comparison view
- [ ] Stage 6 - Optional Three.js/AnyWidget `GrowthViewer`
- [ ] Stage 7 - Optional strain-driven GaN/AlN extension
- [ ] Stage 8 - Final validation and delivery

## Detailed staged plan

### Stage 0 - Reproducible project foundation (complete)

- [x] Pin Python 3.12.0 with `.python-version`.
- [x] Manage and lock dependencies with `uv`, `pyproject.toml`, and `uv.lock`.
- [x] Provide `make setup`, `make notebook`, `make test`, `make check`, `make reproduce`, and
  `make export`.
- [x] Add deterministic baseline JSON, NPZ, and PNG generation.
- [x] Add CI for locked sync, tests, Ruff, strict Marimo checking, notebook execution/export,
  and baseline reproduction.
- [x] Document setup, repository structure, commands, and troubleshooting.

**Exit criteria:** a new contributor can run `uv sync`, `make reproduce`, and `make notebook`
without local paths or undocumented dependencies. **Met.**

### Stage 1 - Correct generic KMC event catalogue (complete)

- [x] Implement seeded residence-time KMC on a periodic six-neighbor SOS lattice.
- [x] Implement deposition, surface diffusion, nucleation/island formation, and observables.
- [x] Allow single-step upward and downward nearest-neighbor hops; forbid multi-step jumps.
- [x] Apply
  `E_diff = E_diff^(0) + n E_b + m E_step`, with `m = 1` only for downward crossings.
- [x] Add Arrhenius desorption and enforce
  `film mass = deposited events - desorbed events`.
- [x] Expose temperature, flux, diffusion, step, and desorption controls using explicitly
  uncalibrated demonstration defaults.
- [x] Define the normalized step-density RHEED proxy as `I_proxy = 1 - S_d`, where `S_d` is
  the fraction of unique neighbor bonds with unequal endpoint heights.
- [x] Keep the proxy labelled as morphology-based, not as electron diffraction.

**Exit criteria:** focused invariants, seeded reproducibility, strict notebook checking, export,
and the canonical baseline all pass. **Met.**

### Stage 2 - Figure 3 homoepitaxial reproduction (complete at smoke scale)

Goal: establish `KMC growth -> step density -> 1-S_d -> Figure 3 comparison` without the
GaN/AlN strain model. Figure 3 is the main target because the homoepitaxial calculation omits
`E_str`.

#### 2A - Paper data and provenance

- [x] Record Figure 3 temperature, fluxes, Ga/N ratios, time axis, normalization limits, and
  initial conditions in `docs/PAPER_NOTES.md`.
- [x] Determine which experimental curves can be digitized or compared qualitatively under
  copyright and available-data constraints.
- [x] Record every fitted parameter equation, unit, validity range, and paper source.
- [x] Distinguish paper values, inferred values, demonstration values, and numerical controls.

#### 2B - GaN homoepitaxy parameterization

- [x] Implement the paper's Appendix A effective-flux conversion.
- [x] Implement the paper's Ga/N-ratio-dependent `E_diff^(0)`, `E_b`, `E_des^(0)`, and
  `E_step` expressions in tested source code.
- [x] Add named Figure 3 parameter sets separate from the fast demonstration preset.
- [x] Add `make figure3-parameters` to reproduce the conversion and rate diagnostics.
- [x] Verify rate magnitudes and competing deposition/diffusion/desorption timescales before
  running large simulations.
- [x] Document unavailable raw curves, arbitrary-unit normalization, seed, and initialization
  details instead of silently choosing them.
- [x] Implement conservative multiscale isolated-adatom acceleration with spatial/rate
  rescaling and an exact-mode fallback.
- [x] Validate accelerated ensemble means against exact KMC over 100 seeds with
  `make validate-acceleration`.
- [x] Connect the paper-derived parameters to an executable 40 s Figure 3 run preset.

#### 2C - Scientific reproduction

- [x] Reproduce simulated `1-S_d` traces for the reported Ga/N ratios with
  `make reproduce-figure3`.
- [x] Replace the rough generic regime for Figure 3 work with the paper-derived parameter sets;
  retain the generic baseline only as a software fingerprint.
- [x] Demonstrate the expected roughening near partial-layer coverage and smoothing near layer
  completion.
- [x] Compare phase, damping, and relative oscillation amplitude qualitatively with Figure 3:
  periodicity is reproduced; damping and amplitude remain finite-size/normalization limited.
- [x] Classify the result as a qualitative smoke reproduction, not quantitative agreement.

**Exit criteria:** at least one documented homoepitaxial configuration shows defensible
layer-by-layer oscillations, and the comparison can be rerun from one command. **Met at 7x7
smoke scale; publication convergence remains Stage 3 work.**

### Stage 3 - Ensembles and parameter studies

#### 3A - Runtime presets

- [x] Add and benchmark a 16x16 interactive preset (about 1.3 s for the current 2 ML demo).
- [x] Add a 64x64 publication candidate preset.
- [ ] Consider 128x128 only after measuring runtime and memory use.
- [x] Keep CI and baseline simulations small and fast.

#### 3B - Uncertainty and convergence

- [x] Add reusable interpolation of seeded RHEED-proxy ensembles.
- [x] Run three independent seeds per point in the 16x16 sweep.
- [x] Plot mean +/- standard deviation for the initial sweep and lattice-size morphology metric.
- [x] Plot three-seed mean +/- standard deviation traces for the Figure 3 smoke ensemble.
- [x] Run an initial 8x8/16x16/24x24 lattice-size and three-seed sensitivity check.
- [x] Run a three-seed, 4 s paper-derived size check through 32x32.
- [x] Benchmark the 64x64 paper-derived candidate over 0.5, 1, and 4 s.
- [x] Establish the 64x64 runtime ceiling and defer its full 40 s ensemble to publication work.
- [x] Repeat the diffusion-smoothing sanity check with the corrected event catalogue.

#### 3C - Parameter study

- [x] Choose `(T, F) -> RHEED oscillation amplitude` as the first sweep.
- [x] Define amplitude as half the 95th-minus-5th percentile proxy range.
- [x] Generate a reproducible 16x16, 3x3 heatmap with configuration and seed provenance via
  `make sweep`.
- [x] Let a selected heatmap point drive its morphology and RHEED views in Marimo.
- [x] Check the high-flux versus low-flux direction at 24x24 via `make validate-sweep`.
- [x] Record that the 700 K endpoint overlaps in uncertainty and that no monotonic temperature
  trend is supported.

**Exit criteria:** the reported trend survives multiple seeds and a documented lattice-size
check; the sweep regenerates without manual notebook interaction. **Met.**

### Stage 4 - Polished Marimo + Plotly virtual experiment

Use Marimo for reactive state/layout, Plotly for browser-interactive 3D and curves, and
Matplotlib for static publication outputs. Do not begin Three.js work in this stage.

#### 4A - Notebook narrative

- [x] **01 What is MBE?** Show Ga source -> beam -> substrate -> growing surface.
- [x] **02 What does an atom do?** Explain deposition, diffusion, attachment, and desorption.
- [x] **03 How does KMC work?** Show event rates, selected event, and residence-time advance.
- [x] **04 Grow a surface.** Add play/pause and coverage/time scrubbing.
- [x] **05 What does RHEED see?** Explain grazing incidence and the step-density relationship.
- [x] **06 Surface <-> RHEED.** Synchronize morphology and proxy trace.
- [x] **07 Experiment.** Present temperature, flux, barriers, size, and seed as a designed
  control panel rather than a raw dictionary.
- [x] **08 Parameter sweep.** Show the Stage 3 regime map and selected run.
- [ ] **09 Paper reproduction.** Present Figure 3 simulation/comparison results.
- [x] **10 Model limits.** State omitted physics and valid interpretation.

#### 4B - Interactive morphology

- [x] Replace the small Matplotlib 3D notebook view with a Plotly 3D height surface.
- [x] Support rotation, zoom, hover height, stable color limits, and a readable camera default.
- [ ] Add `Atoms | Height field | Step edges` display modes where they remain accurate and
  performant.
- [ ] Render the six-neighbor topology with hexagonal geometry where the lattice itself is
  shown; do not imply the rectangular array rendering is the physical metric geometry.
- [ ] Add morphology snapshots at 0.5 ML intervals.

#### 4C - Synchronized morphology and RHEED

- [x] Use one coverage/frame control for both surface and RHEED views.
- [x] Add a vertical current-coverage marker to the full RHEED trace.
- [x] Label the curve **normalized step-density RHEED proxy**.
- [ ] Show the flat-surface maximum versus partial-layer minimum mechanism visually.
- [x] Keep expensive simulations behind an explicit run action; frame scrubbing must reuse
  stored snapshots.

#### 4D - Visual and interaction validation

- [x] Verify first-load output, control changes, play/pause, scrubbing, and parameter selection.
- [x] Inspect desktop and narrow layouts.
- [x] Check browser console and notebook kernel for errors.
- [x] Confirm notebook execution and static HTML export remain automated.

**Exit criteria:** the notebook reads as one numerical experiment and physics explanation,
not a sequence of unrelated controls and plots.

### Stage 5 - Publication figures and paper-comparison view

- [ ] Keep Matplotlib for deterministic static figures and report-ready exports.
- [ ] Build a Figure 3-style panel comparing experimental RHEED and simulated `1-S_d` for the
  documented flux ratios.
- [ ] Build a Figure 4-inspired morphology sequence at selected coverages, while avoiding any
  claim of the strain-driven transition until strain exists.
- [ ] Include ensemble bands, units, parameter provenance, and model-limit captions.
- [ ] Extend the paper-derived ensemble to 64x64 and the full 40 s window after further
  performance work or access to suitable compute.
- [ ] Regenerate all main figures through `make reproduce` or a documented publication command.

**Exit criteria:** every final figure is traceable to a configuration, seed set, code version,
and generated data artifact.

### Stage 6 - Optional Three.js/AnyWidget `GrowthViewer`

This stage has a go/no-go gate. Start it only if the Stage 4 Plotly view cannot communicate the
atomic/hexagonal/event story well enough.

- [ ] Document the specific Plotly limitation that justifies a custom widget.
- [ ] Add `anywidget` and Three.js only after the KMC and Plotly interface are stable.
- [ ] Keep Python authoritative for positions, heights, event data, atom types, step edges,
  coverage, and simulation time; JavaScript owns rendering only.
- [ ] Put widget Python, JavaScript, and CSS in separate `src/mbe_rheed_sim/widgets/` files.
- [ ] Add orbit/zoom, spherical atoms, hexagonal substrate, and highlighted step edges.
- [ ] Add incoming deposition and hop animation only if event-history storage is bounded and
  does not compromise reproducibility.
- [ ] Add beam/detector geometry as explanatory visualization, not simulated diffraction.
- [ ] Verify widget fallback/export behavior in Marimo and CI.

K3D-jupyter is not planned as a central dependency. Reconsider it only if measured particle
counts make Plotly and the custom widget inadequate.

**Exit criteria:** the custom view adds clear explanatory value without moving scientific
logic into JavaScript or breaking notebook portability.

### Stage 7 - Optional strain-driven GaN/AlN extension

- [ ] Add `E_str` only after homoepitaxial validation is complete.
- [ ] Specify and validate the elastic/strain approximation before implementation.
- [ ] Investigate the reported 2D-to-3D transition near 2.25 ML.
- [ ] Attempt temperature-dependent morphology panels only with calibrated conditions.
- [ ] Keep full dynamical electron scattering and the complete multiscale paper model outside
  scope unless separately justified.

**Exit criteria:** any claimed Stranski-Krastanov behavior depends on an implemented, documented,
and tested strain model—not merely on mound formation in the generic KMC.

### Stage 8 - Final validation and delivery

- [ ] Run locked setup from a clean clone/worktree.
- [ ] Run the full test, lint, strict Marimo, execution, export, and reproduction suite.
- [ ] Verify deterministic baselines and ensemble artifact provenance.
- [ ] Review all scientific claims against implemented physics and cited sources.
- [ ] Confirm all notebook plots distinguish proxy, kinematic model, and experiment.
- [ ] Recheck desktop/narrow interaction and exported HTML.
- [ ] Update README, validation record, decisions, known limitations, and final status.

**Exit criteria:** a new contributor can reproduce the principal scientific result and launch
the interactive notebook using only the documented commands.

## Current validation record

- [x] `uv sync --locked`
- [x] `make test` - 15 tests pass
- [x] `make check` - Ruff, strict Marimo check, and notebook execution pass
- [x] `make reproduce` - deterministic fingerprint matches
- [x] `make export` - HTML export succeeds
- [x] Validate Figure 3-like oscillatory behavior at smoke scale
- [x] Validate corrected-model smoothing and step-barrier mounding trends
- [x] Run an initial generic-regime lattice-size sensitivity check
- [x] Re-inspect desktop and narrow layouts after the visual redesign
- [x] `make figure3-parameters` - Appendix A and Equation 8 values match hand-calculated checks
- [x] `make validate-acceleration` - 100-seed exact/accelerated observable comparison passes
- [x] `make reproduce-figure3` - three-seed 40 s bands for all three paper ratios generated
- [x] `make sweep` - 16x16, 3x3, three-seed amplitude map generated
- [x] `make validate-science` - five-seed smoothing/mounding ordering passes
- [x] `make validate-sweep` - 24x24 low/high-flux direction passes at all three temperatures
- [x] `make convergence` - 8x8/16x16/24x24, three-seed sensitivity artifacts generated
- [x] `make convergence-figure3` - 8x8/16x16/32x32, three-seed 4 s bands generated
- [x] Browser-check synchronized frame scrubbing plus responsive desktop/narrow rendering
  with zero console errors

The canonical 8x8, 1 ML software baseline records 67 deposition, 1,416 diffusion, and 3
desorption events. Its final-height SHA-256 is checked by `make reproduce`. This proves
repeatability, not scientific agreement.

## Scientific guardrails and known limitations

- The baseline is a single-species, periodic, solid-on-solid model with no overhangs.
- Six-neighbor connectivity is hexagonal; the current array image is not a hexagonal metric
  rendering.
- Energetic defaults are demonstration parameters, not calibrated GaN values.
- `1-S_d` is a normalized morphology proxy, not an electron-diffraction calculation.
- No strain, multiple species, reconstruction, or electron scattering is implemented.
- Optional isolated-adatom long-hop acceleration is implemented and validated only for the
  documented small-lattice ensemble observables; exact nearest-neighbor KMC remains available.
- Figure 3 homoepitaxial GaN is the near-term target because it does not require `E_str`.
- A fixed seed gives repeatability; uncertainty claims require seed ensembles.
- The generic reproducible baseline is too rough and does not show the target oscillation.
- The 7x7 Figure 3 amplitude is a finite-size smoke result, not a publication observable.

## Tooling decisions

- [x] Use Marimo for controls, reactive state, layouts, and the notebook application.
- [x] Keep Matplotlib for deterministic static/publication figures.
- [x] Add Plotly in Stage 4 for interactive 3D morphology and synchronized RHEED curves.
- [ ] Evaluate AnyWidget + Three.js only after Plotly and the KMC interface are stable.
- [x] Do not add K3D-jupyter without a measured rendering-scale need.

The requested `marimo-notebook` and `implement-paper` skills were unavailable in the official
Marimo skill repository; `marimo-pair` was installed and used as the fallback.

## Open decisions

- [x] Use `(T, F)` RHEED amplitude for the first ensemble map.
- [x] Retain 64x64 only as a publication candidate: measured runtime and non-convergence rule
  out promoting it to a validated preset yet.
- [ ] Decide whether Plotly is sufficient before approving the custom Three.js stage.

## Important commands

```bash
uv sync
make reproduce
make notebook
make test
make check
make validate-sweep
make convergence-figure3
make export
```

## Last meaningful update

2026-08-13 - Continued Stage 4. Added the source-to-film and atom-event teaching sequence,
replaced raw dictionary controls with labelled Marimo form tables, and linked responsive
Plotly surface/RHEED views to one stored-snapshot slider. Native timed playback advances the
same stored frame state and pauses cleanly. Desktop and 390 px layouts, form submission,
manual/timed scrubbing, and browser console output were verified; no custom widget is needed
yet.
