from dataclasses import replace
from multiprocessing import active_children

import numpy as np
import pytest

from mbe_rheed_sim import SimulationConfig, run
from mbe_rheed_sim.workflows import (
    parse_int_values,
    promote_artifacts,
    resolve_workers,
    run_parallel,
)


def test_worker_resolution_and_value_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MBE_WORKERS", "3")
    assert resolve_workers() == 3
    assert resolve_workers(1) == 1
    assert parse_int_values("3,1,2", (9,)) == (3, 1, 2)
    with pytest.raises(ValueError):
        resolve_workers(0)
    with pytest.raises(ValueError):
        parse_int_values("1,1", (9,))
    monkeypatch.setenv("MBE_WORKERS", "invalid")
    with pytest.raises(ValueError):
        resolve_workers()


def test_parallel_simulations_preserve_order_and_results() -> None:
    base = SimulationConfig(lattice_size=5, target_coverage_ml=0.2, sample_every_ml=0.1)
    configs = [replace(base, seed=seed) for seed in (3, 1, 2)]
    sequential = [run(config) for config in configs]
    parallel = run_parallel(run, configs, workers=2, description="test")

    for expected, actual in zip(sequential, parallel, strict=True):
        assert expected.config.seed == actual.config.seed
        assert expected.deposited_events == actual.deposited_events
        assert expected.diffusion_events == actual.diffusion_events
        np.testing.assert_array_equal(expected.final_heights, actual.final_heights)
        np.testing.assert_array_equal(expected.rheed_proxy, actual.rheed_proxy)


def test_parallel_failure_is_propagated() -> None:
    with pytest.raises(ValueError):
        run_parallel(int, ("1", "not-an-integer", "3"), workers=2, description="failure test")
    assert not active_children()


def test_promotion_keeps_history_and_replaces_canonical_atomically(tmp_path) -> None:
    source = tmp_path / "history" / "artifacts"
    project = tmp_path / "project"
    generated = source / "outputs" / "runs" / "result.json"
    generated.parent.mkdir(parents=True)
    generated.write_text('{"new": true}\n')
    canonical = project / "outputs" / "runs" / "result.json"
    canonical.parent.mkdir(parents=True)
    canonical.write_text('{"old": true}\n')

    promoted = promote_artifacts(source, project, tmp_path / "promotion.lock")

    assert promoted == ["outputs/runs/result.json"]
    assert canonical.read_text() == '{"new": true}\n'
    assert generated.read_text() == '{"new": true}\n'
