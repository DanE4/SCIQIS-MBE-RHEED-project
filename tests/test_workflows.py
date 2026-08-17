from dataclasses import replace
from multiprocessing import active_children

import numpy as np
import pytest

from mbe_rheed_sim import SimulationConfig, run
from mbe_rheed_sim.workflows import (
    log_progress,
    parse_int_values,
    parse_workflow_args,
    promote_artifacts,
    resolve_workers,
    run_parallel,
    update_progress,
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


def test_workflow_args_honour_and_reject_duration(monkeypatch: pytest.MonkeyPatch) -> None:
    """--duration must reach `main()`, default when absent, and refuse a non-physical window."""
    monkeypatch.setattr("sys.argv", ["script", "--duration", "40"])
    assert parse_workflow_args(workers=False, duration_s=4.0) == {"duration_s": 40.0}
    monkeypatch.setattr("sys.argv", ["script"])
    assert parse_workflow_args(workers=False, duration_s=4.0) == {"duration_s": 4.0}
    monkeypatch.setattr("sys.argv", ["script", "--duration", "0"])
    with pytest.raises(ValueError):
        parse_workflow_args(workers=False, duration_s=4.0)
    # A script that does not declare the override must not silently swallow the flag.
    monkeypatch.setattr("sys.argv", ["script", "--duration", "40"])
    with pytest.raises(SystemExit):
        parse_workflow_args(workers=False)


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


def test_progress_is_logged_even_without_a_batch_manifest(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A script run straight from the shell still reports, so the CLI is never silent."""
    monkeypatch.delenv("MBE_PROGRESS_FILE", raising=False)
    with caplog.at_level("INFO", logger="mbe"):
        update_progress(stage="demo", completed=2, total=5)
    assert "demo 2/5" in caplog.text


def test_log_progress_reports_once_per_decile(caplog: pytest.LogCaptureFixture) -> None:
    """One long trajectory is one work item, so it needs fraction logging, not a count."""
    report = log_progress("32x32")
    with caplog.at_level("INFO", logger="mbe"):
        for fraction in (0.0, 0.04, 0.09, 0.1, 0.15, 1.0):
            report(fraction)
    assert [record.getMessage() for record in caplog.records] == [
        "32x32   0%",
        "32x32  10%",
        "32x32 100%",
    ]


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
