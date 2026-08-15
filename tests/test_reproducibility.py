import json
from pathlib import Path

import numpy as np

from mbe_rheed_sim import SimulationConfig, run
from mbe_rheed_sim.kmc import SimulationResult

GALLERY = Path(__file__).resolve().parents[1] / "data" / "gallery"


def test_seeded_runs_are_identical() -> None:
    config = SimulationConfig(lattice_size=5, target_coverage_ml=0.5, seed=42)
    first = run(config)
    second = run(config)
    assert np.array_equal(first.final_heights, second.final_heights)
    assert np.array_equal(first.time_s, second.time_s)
    assert first.diffusion_events == second.diffusion_events
    assert first.desorbed_events == second.desorbed_events


def test_result_serializes_without_pickle(tmp_path) -> None:
    result = run(SimulationConfig(lattice_size=3, target_coverage_ml=0.25, seed=1))
    path = tmp_path / "run.npz"
    result.save_npz(path)
    with np.load(path, allow_pickle=False) as saved:
        assert json.loads(str(saved["config_json"]))["seed"] == 1
        assert np.array_equal(saved["final_heights"], result.final_heights)

    restored = SimulationResult.load_npz(path)
    assert restored.config == result.config
    assert restored.deposited_events == result.deposited_events
    assert np.array_equal(restored.snapshots, result.snapshots)


def test_notebook_gallery_matches_its_index() -> None:
    """The notebook presents these without running anything, so they must stay loadable."""
    index = json.loads((GALLERY / "index.json").read_text())
    assert index, "gallery index is empty; run `make gallery`"
    for name, entry in index.items():
        result = SimulationResult.load_npz(GALLERY / f"{name}.npz")
        assert result.config.lattice_size == entry["config"]["lattice_size"]
        assert len(result.snapshots) == entry["frames"]
        assert result.roughness_ml[-1] == entry["final_roughness_ml"]
        assert entry["title"] and entry["story"]
