# Project status

## Project goal

Build a validated, interactive KMC model of epitaxial growth and progressively connect its
surface morphology to RHEED intensity.

## Current phase

Initial vertical slice complete and ready for review.

## Working

- Primary-paper model, equations, figures, and limitations reviewed in `docs/PAPER_NOTES.md`.
- Additional RHEED/step-density/kinematic references recorded in `docs/REFERENCES.md`.
- Project scope and parameter provenance defined in `docs/SCIENCE_MODEL.md`.
- Official `marimo-pair` agent skill installed for live notebook inspection.
- Seeded residence-time KMC implements deposition and thermally activated diffusion.
- Coverage, roughness, island statistics, step density, snapshots, and serialization work.
- The Marimo notebook provides 2D/3D morphology, growth plots, and a RHEED proxy.
- Python 3.12.0, the uv environment, and all dependencies are locked for clean setup.
- `make reproduce` regenerates a deterministic 8x8, 1 ML baseline and checks its fingerprint.
- GitHub Actions tests, checks, executes/exports the notebook, and runs the baseline smoke test.
- Tests, Ruff, strict Marimo check, script execution, HTML export, and visual checks pass.

## In progress

- None. Awaiting review of the initial milestone.

## Next

1. Validate qualitative layer-by-layer trends across controlled parameter sweeps.
2. Add desorption and an Ehrlich-Schwoebel barrier only after baseline validation.
3. Add a kinematic diffraction model before making quantitative RHEED claims.
4. Consider strain and GaN/AlN calibration only as a separate advanced milestone.

## Scientific assumptions

- Baseline is a single-species, periodic, solid-on-solid lattice with no overhangs.
- The hexagonal neighbor topology follows the primary paper; energies are demo parameters.
- `1 - step density` is a morphology proxy, not a diffraction calculation.

## Validation status

All seven tests pass. A five-seed check at 2 ML gave mean final roughness 1.3135 ML for
deposition-only growth and 0.4056 ML with diffusion; diffusion was smoother for all five
seeds. The notebook passes strict checking, executes, exports, and has no kernel/browser errors.

## Known limitations

No desorption, step-edge barrier, strain, multiple species, reconstruction, or electron
scattering. The baseline is not quantitatively calibrated to GaN/AlN.

The requested `marimo-notebook` and `implement-paper` skills were not available in the
official Marimo skill repository; `marimo-pair` was installed and used as the fallback.

## Open questions

- Which experimental system and RHEED geometry should anchor quantitative validation?
- Should the next physics milestone prioritize desorption/step barriers or diffraction?

## Important commands

```bash
uv sync
make reproduce
make notebook
make test
make check
make export
```

## Last meaningful update

2026-08-13 - Added locked uv onboarding, deterministic reproduction, and clean-environment CI.
