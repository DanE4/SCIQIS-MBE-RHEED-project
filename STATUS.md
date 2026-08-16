# Project status

## Project goal

Build a validated kinetic Monte Carlo model of molecular-beam epitaxy, connect simulated
surface morphology to a carefully labelled step-density RHEED proxy, reproduce the paper's
Figure 3 homoepitaxial behavior, and present the result as a polished interactive Marimo
experiment.

## Current state

- **Active stage:** Stage 5 smoke-scale publication workflow plus unresolved finite-size
  convergence before final delivery.
- **Estimated completion:** roughly 90% of the intended final project.
- **Working baseline:** deterministic deposition/diffusion/desorption KMC and Marimo notebook.
- **Main scientific gap:** finite-size convergence. Figure 3-like periodicity is reproduced at
  smoke scale, but proxy amplitude remains strongly lattice-size dependent and is not converged
  through 64x64. This prevents publication-quality amplitude claims.
- **Interpretation:** the imperfect/noisy morphology and RHEED proxy are expected at this
  stage. Current evidence does not isolate one obvious implementation bug. The remaining work
  is predominantly scale, convergence, oscillation analysis, quantitative comparison, and
  communication—not a missing fundamental KMC framework.
- **Presentation state:** the teaching flow, explicit experiment-mode selector, controls,
  playback, annotated synchronized layer-cycle view, hex-coordinate lattice view, separate-scale
  Figure 3 comparison, quantitative diagnostic table, and Figure 4-inspired morphology sequence
  work. The primary experiment form now exposes labelled lattice choices from 7x7 through
  256x256 and every `SimulationConfig` field relevant to manual generic or paper-derived runs,
  with expensive-run confirmation. Replacing smoke-scale results with a converged ensemble
  remains.
- **Current computational limit:** single-trajectory cost is no longer the binding constraint.
  Batched neighbourhood gathers, skipping empty sites in the local refresh, sampling from the
  occupied sites only, and compiled Numba kernels for refresh, rate-tree update and sampling
  together reduce the 0.1 s Figure 3 envelope from 1.49 to 0.029 s at 64x64, 6.06 to 0.124 s at
  128x128, and 24.9 to 0.512 s at 256x256, and a paper-derived 64x64, 4 s run from 36.2 to
  1.73 s. Every one of those runs is bit-identical to the trajectory it replaced. The
  three-seed Figure 3 convergence study now reaches 128x128 in 12 s of wall time, so finite-size
  convergence is a science question about how many seeds and sizes to average, not a runtime
  question. Bounded process workers add about 5.6x on top with eight workers on the M4 Pro.
  What now dominates is Python-level per-event overhead: the event loop, RNG draws and the
  per-call boundary into the kernels, at roughly 6.5 microseconds per event.
- **Size policy:** 16x16 is for responsive teaching, 64x64 is the current publication candidate,
  and 128x128/256x256 are benchmarks if runtime permits. Matching 256x256 exactly is not the
  objective; demonstrated convergence of reported observables is.

## What is already scientifically strong

The working framework includes residence-time/event-based KMC; deposition; nearest-neighbor
diffusion; upward/downward single-step motion; an Ehrlich-Schwoebel barrier; desorption; mass
accounting; seeded reproducibility; paper-derived Figure 3 parameterization and effective-flux
conversion; multiscale acceleration validated against exact-KMC ensembles; ensemble
uncertainty; parameter sweeps; convergence runners; synchronized Plotly/Marimo visualization;
and automated test, check, export, and reproduction commands.

## Why the current result does not yet look like the paper

1. **Finite-size/statistical effects:** smoke lattices exaggerate fluctuations, and the measured
   proxy signal changes strongly through 64x64. Three seeds are enough to expose
   sensitivity, not to establish publication convergence.
2. **Demonstration parameters:** the generic interactive model uses fast effective values and
   often produces rough/island-like growth. That is useful teaching behavior, not GaN
   calibration.
3. **Paper-derived Figure 3 parameters:** these correctly use the paper's effective-flux and
   activation-energy equations, but correct inputs alone do not remove finite-size,
   acceleration, seed, initialization, or normalization uncertainty.
4. **Model limitations:** `1-S_d` is morphology-derived, not electron diffraction. The model
   omits beam geometry, reconstruction, multiple scattering, and other experimental physics.
   Strain is irrelevant to this homoepitaxial target and must not be added as a patch.
5. **Visualization limitations:** the continuous height field is drawn in rectangular index
   space. The optional axial-coordinate view shows the six-neighbor geometry, but the final
   annotated morphology/RHEED cycle and paper comparison are unfinished.

## Simulation modes

### Generic interactive experiment

Purpose: teaching, fast interaction, parameter-dependence demonstrations, and a software/debug
baseline. It normally uses a 16x16 lattice and uncalibrated demonstration/effective parameters.
It may produce rough or island-like growth, is not expected to reproduce Figure 3
quantitatively, and must never be used as evidence of paper reproduction.

### Paper Figure 3 reproduction

Purpose: scientific comparison with the homoepitaxial GaN RHEED result. It uses paper-derived
effective-flux conversion and activation energies, the documented initial condition,
temperature and flux ratios, seed ensembles, and convergence studies. Its comparison targets
are phase, period, damping, and relative amplitude.

The Marimo UI exposes an obvious `Generic demonstration | Paper Figure 3 preset` selector.
Figure 3 mode loads the complete scientific preset through `figure3_config`, ignores generic
demonstration sliders, and reports the selected ratio, effective flux, predicted growth rate,
lattice, duration, and seed.

## Near-term priority order

Do not begin Stage 7 while these items remain unresolved:

1. Reduce the remaining local-rate refresh cost before attempting a 128x128, 4 s ensemble.
2. Finalize ensemble/statistical reporting at the accepted lattice size.
3. Replace the Stage 5 smoke traces without changing its comparison/provenance protocol.
4. Only then reconsider Three.js.
5. Only after homoepitaxial validation consider strain-driven GaN/AlN physics.

## Major unfinished milestones

### Meaningful RHEED oscillation metrics

**Why:** the current percentile half-range can be large for a monotonic decay and therefore
cannot support an oscillation or convergence claim.

**Current evidence:** Figure 3-like periodicity is visible at 7x7, while both the historical
range metric and detrended amplitude remain size-sensitive through 64x64.

**Implementation:** the reusable detrended, peak, period, spectral, damping, and phase
diagnostics now exist and are included in the Figure 3, sweep, and both convergence artifact
schemas. Detrended amplitude is the principal convergence observable.

**Validation:** known monotonic traces fail the oscillation classification and a damped
1-ML-period reference is recovered. The notebook exposes the actual morphology and proxy at
annotated 0/0.5/1/1.5/2 ML milestones without relabeling deviations as ideal behavior.

**Dependencies:** none; this is the first implementation task.

### Dedicated Figure 3 Marimo mode

**Why:** generic controls make it too easy to confuse a teaching run with a scientific preset.

**Current evidence:** the notebook has working generic controls, a read-only committed Figure 3
ensemble, and a top-level mode selector that launches a selected paper configuration.

**Implementation:** `notebooks/mbe_rheed.py` reuses `figure3_config` and paper parameter helpers.
The selected Ga/N condition, duration, acceleration, coverage axis, and provenance load
together; generic sliders cannot leak into the paper configuration.

**Validation:** a browser run loaded Ga/N = 0.82 at 1003.15 K, effective Ga flux 0.2450 ML/s,
predicted growth rate 0.2282 ML/s, 7x7, 40 s, and seed 7. The 1.0 ML shortcut synchronized both
views at 0.99 predicted ML with zero browser-console errors.

**Dependencies:** improved oscillation metrics for the scientific summary; the selector and
preset wiring may be developed independently.

### Larger-size Figure 3 convergence

**Why:** finite-size dependence currently prevents quantitative amplitude and morphology
claims.

**Current evidence:** 7x7 reproduces periodicity. Mean detrended amplitude over 4 s is 0.09309,
0.04111, 0.02765, and 0.03231 at 8x8, 16x16, 32x32, and 64x64. No successive-size pair passes
the documented criterion. After the rate-tree and cached-rate optimizations, the 0.1 s runtime
envelope is 1.5, 6.1, and 24.8 wall seconds at 64x64, 128x128, and 256x256, respectively.

**Implementation:** `scripts/check_figure3_convergence.py` now records full per-seed traces,
event counts, result-array footprint, oscillation metrics, final roughness/step density, and
morphology summaries through an opt-in 64x64 run. An updateable rate tree removes whole-lattice
cumulative scans at 128x128 and above; batched updates and cached discrete Arrhenius tables
remove additional bookkeeping. Further local-rate-refresh work is required before a 128x128
ensemble. Independent seeds now run in bounded spawn workers while lattice sizes remain
sequential, reducing wall time without pretending to improve per-run scaling.

**Validation:** successive sufficiently large lattices agree within a justified, documented
tolerance for the principal observable and morphology statistics, with ensemble uncertainty.

**Dependencies:** meaningful oscillation metrics and a decision on convergence tolerance.

### Publication Figure 3 comparison

**Why:** visible periodicity is only qualitative until simulation, uncertainty, reference data,
normalization, and provenance are compared without conflating physical quantities.

**Current evidence:** all three paper parameter sets produce committed three-seed 7x7 proxy
bands. The CC BY PDF supplies vector paths for a figure-derived experimental visual reference;
raw experimental values and the paper's arbitrary-unit normalization remain unavailable.

**Implementation:** notebook Section 09 and `make figure3` now provide separate reference
and simulation panels, uncertainty bands, period/phase/damping/relative-amplitude diagnostics,
a Figure 4-inspired no-strain morphology sequence, and provenance-rich JSON/CSV/NPZ artifacts.

**Validation:** every Ga/N condition reports phase, period, damping, and relative amplitude;
all curves have provenance and unambiguous labels; reference data use a documented legal and
technical acquisition/normalization method. The comparison is still qualitative because the
7x7 proxy amplitude is not finite-size converged.

**Dependencies:** the workflow and reference protocol are complete; accepted-size convergence
remains the sole scientific gate.

## Roadmap overview

- [x] Stage 0 - Reproducible project foundation
- [x] Stage 1 - Correct generic KMC event catalogue
- [x] Stage 2 - Reproduce Figure 3 homoepitaxial RHEED behavior
- [x] Stage 3 - Initial ensembles and parameter studies
- [x] Stage 4 - Polished Marimo + Plotly virtual experiment
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
- [x] Benchmark 128x128 and 256x256 after measuring the 64x64 single-run envelope: after the
  rate-tree optimization the 0.1 s envelope takes 1.5, 6.1, and 24.8 wall seconds at 64x64,
  128x128, and 256x256.
- [x] Keep CI and baseline simulations small and fast.

#### 3B - Uncertainty and convergence

- [x] Add reusable interpolation of seeded RHEED-proxy ensembles.
- [x] Run three independent seeds per point in the 16x16 sweep.
- [x] Plot mean +/- standard deviation for the initial sweep and lattice-size morphology metric.
- [x] Plot three-seed mean +/- standard deviation traces for the Figure 3 smoke ensemble.
- [x] Run an initial 8x8/16x16/24x24 lattice-size and three-seed sensitivity check.
- [x] Run the default three-seed, 4 s paper-derived size check through 32x32.
- [x] Run the opt-in three-seed, 4 s paper-derived 64x64 check; 32x32 -> 64x64 does not pass.
- [x] Benchmark the 64x64 paper-derived candidate over 0.5, 1, and 4 s.
- [x] Establish the 64x64 runtime ceiling and defer its full 40 s ensemble to publication work.
- [x] Repeat the diffusion-smoothing sanity check with the corrected event catalogue.
- [x] Record runtime, event count, relevant memory use, seed count, full proxy trace,
  period/amplitude, RMS roughness, step density, and final morphology statistics at every size.
- [x] Choose and justify a preliminary numerical successive-size convergence tolerance: the
  difference plus 1.96 pooled standard errors must fit inside 10% of the larger-size mean
  detrended amplitude. This is a finite-size numerical criterion, not experimental accuracy.
- [x] Add bounded spawn-based process parallelism for independent seeds and parameter points;
  preserve deterministic input ordering and keep each trajectory sequential.
- [x] Keep lattice sizes and timing benchmarks sequential where concurrent execution would
  increase memory pressure or invalidate timing evidence.

#### 3C - Parameter study

- [x] Choose `(T, F) -> proxy signal range` as the first sweep.
- [x] Define the compatibility metric as half the 95th-minus-5th percentile proxy range.
- [x] Add dedicated oscillation metrics: detrended amplitude, peak count, peak-to-trough
  amplitude, period and 1 ML deviation, spectral power near 1 cycle/ML, damping, and phase.
- [x] Propagate the dedicated metrics through sweep and convergence artifact schemas and select
  one justified principal convergence observable.
- [x] Generate a reproducible 16x16, 3x3 heatmap with configuration and seed provenance via
  `make sweep`.
- [x] Let a selected heatmap point drive its morphology and RHEED views in Marimo.
- [x] Check the high-flux versus low-flux direction at 24x24 via `make validate-sweep`.
- [x] Record that the 700 K endpoint overlaps in uncertainty and that no monotonic temperature
  trend is supported.

**Exit criteria:** the reported signal-range trend survives multiple seeds and a documented
lattice-size check; the sweep regenerates without manual notebook interaction. **Met.** The
percentile range alone does not establish an oscillation.

### Stage 4 - Polished Marimo + Plotly virtual experiment

Use Marimo for reactive state/layout, Plotly for browser-interactive 3D and curves, and
Matplotlib for static publication outputs. Do not begin Three.js work in this stage.

#### 4A - Notebook narrative

- [x] Add an explicit `Generic demonstration | Paper Figure 3 preset` selector; load the full
  paper configuration and provenance through the existing paper helpers.
- [x] Let both modes manually select lattice size, coverage/time stopping, acceleration,
  sampling, event limit, and seed; expose all generic kinetic parameters rather than locking
  the main panel to teaching-scale sliders.

- [x] **01 What is MBE?** Show Ga source -> beam -> substrate -> growing surface.
- [x] **02 What does an atom do?** Explain deposition, diffusion, attachment, and desorption.
- [x] **03 How does KMC work?** Show event rates, selected event, and residence-time advance.
- [x] **04 Grow a surface.** Add play/pause and coverage/time scrubbing.
- [x] **05 What does RHEED see?** Explain grazing incidence and the step-density relationship.
- [x] **06 Surface <-> RHEED.** Synchronize morphology and proxy trace.
- [x] **07 Experiment.** Present temperature, flux, barriers, size, and seed as a designed
  control panel rather than a raw dictionary.
- [x] **08 Parameter sweep.** Show the Stage 3 regime map and selected run.
- [x] **09 Paper reproduction smoke view.** Present the committed Figure 3 simulation ensemble.
- [x] Upgrade Section 09 to the quantitative paper-comparison requirements in Stage 5.
- [x] **10 Batch workflows.** Expose every simulation workflow with guarded execution,
  progress, cancellation, and artifact reload.
- [x] **11 Model limits.** State omitted physics and valid interpretation.

#### 4B - Interactive morphology

- [x] Replace the small Matplotlib 3D notebook view with a Plotly 3D height surface.
- [x] Support rotation, zoom, hover height, stable color limits, and a readable camera default.
- [ ] Add `Atoms | Height field | Step edges` display modes where they remain accurate and
  performant.
- [x] Add a Plotly axial-to-Cartesian hex-cell view for the six-neighbor topology; retain the
  rectangular height field with an explicit index-space label.
- [x] Add direct selection of stored morphology snapshots at 0.5 ML intervals.
- [x] Validate that the current hex-cell representation is sufficient for the Stage 4 lattice
  explanation; reserve atom columns and explicit step edges for a separately justified view.

#### 4C - Synchronized morphology and RHEED

- [x] Use one coverage/frame control for both surface and RHEED views.
- [x] Add a vertical current-coverage marker to the full RHEED trace.
- [x] Label the curve **normalized step-density RHEED proxy**.
- [x] Add an annotated synchronized sequence: 0 ML flat surface; 0.5 ML many islands/steps and
  proxy minimum; 1.0 ML completed layer and proxy maximum; 1.5 ML renewed roughening; 2.0 ML
  next layer completion. Show morphology and trace side by side with one coverage marker, and
  label these as ideal expectations while displaying the actual simulated values.
- [x] Mark the initial flat-surface proxy maximum and most stepped stored frame through 1 ML
  on the synchronized trace without claiming that the generic demonstration completes ideal
  layers.
- [x] Keep expensive simulations behind an explicit run action; frame scrubbing must reuse
  stored snapshots.

#### 4D - Visual and interaction validation

- [x] Verify first-load output, control changes, play/pause, scrubbing, and parameter selection.
- [x] Inspect desktop and narrow layouts.
- [x] Check browser console and notebook kernel for errors.
- [x] Confirm notebook execution and static HTML export remain automated.

#### 4E - Parallel batch workflows

- [x] Resolve worker count as CLI override, `MBE_WORKERS`, then default `min(10, cpu_count-1)`; reject values outside
  `1..os.cpu_count()`.
- [x] Run independent publication, sweep, acceleration, scientific-trend, and sweep-validation
  configurations in spawn-based workers with stable result ordering.
- [x] Run convergence seeds concurrently but lattice sizes sequentially; keep the deterministic
  baseline and 64/128/256 timing benchmark sequential.
- [x] Store every run under `outputs/batches/` with source revision, configuration, seed/size
  overrides, effective workers, elapsed time, JSON-line progress, and final state.
- [x] Promote canonical artifacts under a lock only after complete success; failed and
  interrupted smoke tests leave canonical artifacts unchanged.
- [x] Add notebook workflow, worker, seed, and size controls; gate the 64x64 convergence and
  128/256 benchmark, run asynchronously, show progress, support cancellation, and reload
  promoted publication/sweep data.
- [x] Verify exact equality between one- and multi-worker scientific outputs.
- [ ] Browser-check live batch launch, progress, cancellation, and data reload; the current
  validation environment has no connected in-app browser session.

**Exit criteria:** the notebook reads as one numerical experiment and physics explanation,
not a sequence of unrelated controls and plots.

### Stage 5 - Publication figures and paper-comparison view

- [x] Keep Matplotlib for deterministic static figures and report-ready exports.
- [x] Build a Figure 3-style panel for every reported Ga/N condition with simulation mean,
  ensemble uncertainty, and the experimental/reference curve when legally and technically
  available.
- [x] Document normalization and use identical or clearly related time/coverage domains.
- [x] Compare phase, period, damping, and relative amplitude explicitly.
- [x] Preserve configuration provenance for every curve and artifact.
- [x] Clearly distinguish experimental RHEED intensity, morphology-derived `1-S_d`, and every
  normalized/detrended signal; never imply that they are the same physical quantity.
- [x] Build a Figure 4-inspired morphology sequence at selected coverages, while avoiding any
  claim of the strain-driven transition until strain exists.
- [x] Include ensemble bands, units, parameter provenance, and model-limit captions.
- [ ] Extend the paper-derived ensemble to 64x64 and the full 40 s window after further
  performance work or access to suitable compute.
- [x] Benchmark 128x128 and the paper-reference 256x256; the optimized 0.1 s runtime envelope is
  documented, but no smaller size is accepted because convergence is not demonstrated.
- [x] Regenerate all main figures through the documented `make figure3` command.

**Exit criteria:** every final figure is traceable to a configuration, seed set, code version,
and generated data artifact. **Met for the smoke-scale publication artifacts; Stage 5 remains
open only because the full-window accepted-size ensemble is not computationally established.**

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

- [ ] Add `E_str` only after homoepitaxial Figure 3 convergence and quantitative validation are
  complete; imperfect current morphology is not justification to add strain.
- [ ] Specify and validate the elastic/strain approximation before implementation.
- [ ] Investigate the reported 2D-to-3D transition near 2.25 ML.
- [ ] Attempt temperature-dependent morphology panels only with calibrated conditions.
- [ ] Keep full dynamical electron scattering and the complete multiscale paper model outside
  scope unless separately justified.

**Exit criteria:** any claimed Stranski-Krastanov behavior depends on an implemented, documented,
and tested strain model—not merely on mound formation in the generic KMC.

This stage is specifically for GaN/AlN heteroepitaxy, Stranski-Krastanov growth, the 2D-to-3D
transition, quantum-dot formation, and critical thickness near the reported 2.25 ML regime.
Figure 3 is homoepitaxial GaN and does not require this physics.

### Stage 8 - Final validation and delivery

- [ ] Run locked setup from a clean clone/worktree.
- [x] Run the full test, lint, strict Marimo, execution, export, and reproduction suite.
- [x] Verify deterministic baselines and current sweep/convergence artifact provenance.
- [ ] Review all scientific claims against implemented physics and cited sources.
- [x] Confirm all notebook plots distinguish proxy, kinematic model, and experiment.
- [ ] Recheck desktop/narrow interaction and exported HTML.
- [x] Update README, validation record, decisions, known limitations, and final status.

**Exit criteria:** a new contributor can reproduce the principal scientific result and launch
the interactive notebook using only the documented commands.

## Current validation record

- [x] `uv sync --locked`
- [x] `make test` - 24 tests pass
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
- [x] `make figure3` - separate-scale Figure 3 comparison, diagnostics, morphology sequence,
  and provenance-rich JSON/CSV/NPZ artifacts generated
- [x] `make sweep` - 16x16, 3x3, three-seed amplitude map generated
- [x] `make validate-science` - five-seed smoothing/mounding ordering passes
- [x] `make validate-sweep` - 24x24 low/high-flux direction passes at all three temperatures
- [x] `make convergence` - 8x8/16x16/24x24, three-seed sensitivity artifacts generated
- [x] `make convergence-figure3` - 8x8/16x16/32x32, three-seed 4 s bands generated
- [x] `make figure3-convergence SIZES=8,16,32,64` - opt-in 64x64 point generated; 32x32 -> 64x64 fails
- [x] `make benchmark-sizes` - controlled 64x64/128x128/256x256 runtime envelope generated
- [x] Parallel acceptance - publication: 52.4 s with one worker versus 15.9 s with four
  (3.30x); Figure 3 convergence through 64x64: 159.1 s with one worker versus 51.9 s with
  three effective workers (3.07x); scientific arrays, event counts, metrics, and hashes match.
- [x] Artifact-safety acceptance - failed and interrupted smoke workflows remain in batch
  history and do not promote partial canonical artifacts.
- [x] Browser-check synchronized frame scrubbing plus responsive desktop/narrow rendering
  with zero console errors

The canonical 8x8, 1 ML software baseline records 67 deposition, 1,416 diffusion, and 3
desorption events. Its final-height SHA-256 is checked by `make reproduce`. This proves
repeatability, not scientific agreement.

## Scientific guardrails and known limitations

- The baseline is a single-species, periodic, solid-on-solid model with no overhangs.
- Six-neighbor connectivity is hexagonal; the continuous height-field view is rectangular
  index space, while the optional lattice view maps axial coordinates to hexagonal geometry.
- Energetic defaults are demonstration parameters, not calibrated GaN values.
- `1-S_d` is a normalized morphology proxy, not an electron-diffraction calculation.
- No strain, multiple species, reconstruction, or electron scattering is implemented.
- Optional isolated-adatom long-hop acceleration is implemented and validated only for the
  documented small-lattice ensemble observables; exact nearest-neighbor KMC remains available.
- Figure 3 homoepitaxial GaN is the near-term target because it does not require `E_str`.
- A fixed seed gives repeatability; uncertainty claims require seed ensembles.
- The generic reproducible baseline is too rough and does not show the target oscillation.
- The 7x7 Figure 3 amplitude is a finite-size smoke result, not a publication observable.
- The red Stage 5 reference is a CC BY figure-derived panel coordinate, not raw experimental
  detector data; its source, extraction, axis mapping, and limitations are recorded.

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
- [x] Select detrended amplitude as the principal convergence observable and require the
  successive-size difference plus uncertainty to fit within a 10% relative margin.
- [ ] Decide the minimum seed count/statistical interval for publication reporting after
  inspecting variance at larger sizes.
- [x] Resolve the legal/technical source and normalization protocol for the Figure 3
  experimental curves: vector extraction from the tracked CC BY PDF with plotted-axis mapping
  and an explicit figure-derived-data label.
- [x] Keep Plotly for the Stage 4 height/hex-cell explanation; do not approve Three.js without
  a concrete remaining communication or performance requirement.

## Important commands

```bash
uv sync
make reproduce
make notebook
make test
make check
make validate-sweep
make convergence-figure3
make figure3-convergence SIZES=8,16,32,64
make benchmark-sizes
make export
```

## Last meaningful update

2026-08-15 - Section 5 can now save the active result to `outputs/saved/` and reload one
through a third **Saved run** source, reusing `SimulationResult.save_npz`/`load_npz`.

2026-08-13 - Added bounded spawn-based ensemble parallelism, safe batch-history promotion,
JSON-line progress, worker/seed/size CLI overrides, and a guarded asynchronous Marimo batch
runner for all simulation workflows. One- versus multi-worker publication and 64x64
convergence outputs match exactly while wall time improves by 3.30x and 3.07x. The 64x64
result still fails the predeclared finite-size criterion, so parallelism improves turnaround
without changing that scientific limitation.
