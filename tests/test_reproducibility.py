import json

import numpy as np

from mbe_rheed_sim import SimulationConfig, run


def test_seeded_runs_are_identical() -> None:
    config = SimulationConfig(lattice_size=5, target_coverage_ml=0.5, seed=42)
    first = run(config)
    second = run(config)
    assert np.array_equal(first.final_heights, second.final_heights)
    assert np.array_equal(first.time_s, second.time_s)
    assert first.diffusion_events == second.diffusion_events


def test_result_serializes_without_pickle(tmp_path) -> None:
    result = run(SimulationConfig(lattice_size=3, target_coverage_ml=0.25, seed=1))
    path = tmp_path / "run.npz"
    result.save_npz(path)
    with np.load(path, allow_pickle=False) as saved:
        assert json.loads(str(saved["config_json"]))["seed"] == 1
        assert np.array_equal(saved["final_heights"], result.final_heights)
