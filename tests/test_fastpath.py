import numpy as np
import pytest

from mbe_rheed_sim import SimulationConfig, fastpath, run
from mbe_rheed_sim.paper import figure3_config

CASES = (
    # Below 128x128 the catalogue samples from cumulative rates, above it from the rate tree;
    # both selection paths have to agree with the reference implementation.
    SimulationConfig(
        lattice_size=24,
        target_coverage_ml=1.0,
        max_isolated_hop_distance=4,
        sample_every_ml=0.1,
        seed=5,
    ),
    figure3_config(0.82, lattice_size=64, duration_s=0.05, seed=0),
    figure3_config(0.82, lattice_size=128, duration_s=0.02, seed=1),
)


def _run_with_backend(config: SimulationConfig, backend: str, monkeypatch) -> dict:
    monkeypatch.setenv("MBE_KMC_BACKEND", backend)
    result = run(config)
    return {
        "final_heights": result.final_heights,
        "time_s": result.time_s,
        "rheed_proxy": result.rheed_proxy,
        "roughness_ml": result.roughness_ml,
        "snapshots": result.snapshots,
        "counts": (
            result.deposited_events,
            result.selected_diffusion_events,
            result.diffusion_events,
            result.long_hop_events,
            result.desorbed_events,
        ),
    }


@pytest.mark.parametrize("config", CASES, ids=lambda config: f"L{config.lattice_size}")
def test_compiled_backend_reproduces_the_reference_trajectory(config, monkeypatch) -> None:
    reference = _run_with_backend(config, "reference", monkeypatch)
    compiled = _run_with_backend(config, "fast", monkeypatch)
    assert compiled["counts"] == reference["counts"]
    for name, expected in reference.items():
        if name != "counts":
            # Bit-identical, not approximate: the kernels repeat the reference operations
            # in the reference order, so any drift here is a real divergence.
            assert np.array_equal(compiled[name], expected), name


def test_unknown_backend_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("MBE_KMC_BACKEND", "metal")
    with pytest.raises(ValueError):
        fastpath.enabled()
