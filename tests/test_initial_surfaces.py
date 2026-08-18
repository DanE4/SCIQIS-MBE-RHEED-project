import numpy as np
import pytest

from mbe_rheed_sim import SimulationConfig, run
from mbe_rheed_sim.lattice import INITIAL_SURFACES, initial_lattice


def _config(**overrides) -> SimulationConfig:
    return SimulationConfig(
        lattice_size=16,
        target_coverage_ml=0.5,
        temperature_k=900.0,
        seed=3,
        **overrides,
    )


@pytest.mark.parametrize("name", sorted(INITIAL_SURFACES))
def test_named_surfaces_are_deterministic_and_well_formed(name: str) -> None:
    first = initial_lattice(name, 32)
    assert first.dtype == np.int64
    assert first.shape == (32, 32)
    assert first.min() >= 0
    # A config stores the name, not the array, so the name has to pin the surface exactly.
    assert np.array_equal(first, initial_lattice(name, 32))


def test_unknown_surface_is_rejected_by_builder_and_config() -> None:
    with pytest.raises(ValueError, match="unknown initial surface"):
        initial_lattice("hexagonal-sponge", 8)
    with pytest.raises(ValueError, match="initial_surface must be one of"):
        _config(initial_surface="hexagonal-sponge")


def test_flat_stays_the_default_trajectory() -> None:
    # Adding the field must not move any existing run.
    default = run(_config())
    explicit = run(_config(initial_surface="flat"))
    assert np.array_equal(default.final_heights, explicit.final_heights)
    assert default.coverage_ml[0] == 0.0


@pytest.mark.parametrize("name", sorted(INITIAL_SURFACES))
def test_growth_starts_from_the_named_surface_and_conserves_mass(name: str) -> None:
    result = run(_config(initial_surface=name))
    start = initial_lattice(name, 16)
    assert np.array_equal(result.snapshots[0], start)
    assert result.coverage_ml[0] == pytest.approx(float(np.mean(start)))

    # Coverage counts monolayers deposited, so the surface must gain exactly the net atoms.
    net_atoms = result.deposited_events - result.desorbed_events
    assert int(result.final_heights.sum()) == int(start.sum()) + net_atoms
    assert result.final_heights.min() >= 0


def test_step_flow_start_grows_without_recreating_the_flat_case() -> None:
    stepped = run(_config(initial_surface="straight-step"))
    flat = run(_config(initial_surface="flat"))
    # The stepped substrate keeps its two steps, so it cannot land on the flat trajectory.
    assert not np.array_equal(stepped.final_heights, flat.final_heights)
    assert stepped.roughness_ml[0] > flat.roughness_ml[0]
