"""Run one reproducible workflow with history, progress, and safe promotion."""

import argparse
import json
import os
import signal
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter, sleep

from mbe_rheed_sim.workflows import promote_artifacts, resolve_workers

ROOT = Path(__file__).resolve().parents[1]
BATCH_ROOT = ROOT / "outputs" / "batches"
WORKFLOWS = {
    "baseline": ("reproduce_baseline.py", False, False, False, ()),
    "publication": ("reproduce_figure3.py", True, True, False, ()),
    "sweep": ("run_parameter_sweep.py", True, True, False, ()),
    "convergence": ("check_lattice_convergence.py", True, True, True, ()),
    "figure3-convergence": ("check_figure3_convergence.py", True, True, True, ()),
    "figure3-convergence-64": (
        "check_figure3_convergence.py",
        True,
        True,
        True,
        ("--include-64",),
    ),
    "validate-acceleration": ("validate_acceleration.py", True, True, False, ()),
    "validate-science": ("validate_scientific_trends.py", True, True, False, ()),
    "validate-sweep": ("validate_sweep_lattice.py", True, True, False, ()),
    "benchmark-sizes": ("benchmark_large_lattices.py", False, False, True, ()),
}
WORKFLOW_PRESETS = {
    "baseline": "8x8, 1 ML, seed 2026",
    "publication": "7x7, 40 s, Ga/N 0.89/0.82/0.68, seeds 2026/2027/2028",
    "sweep": "16x16, 2 ML, T=700/850/1000 K, flux=0.25/0.5/0.75 ML/s, seeds 0/1/2",
    "convergence": "8/16/24, 2 ML, seeds 0/1/2",
    "figure3-convergence": "8/16/32, Ga/N 0.82, 4 s, seeds 0/1/2",
    "figure3-convergence-64": "8/16/32/64, Ga/N 0.82, 4 s, seeds 0/1/2",
    "validate-acceleration": "7x7, 0.5 ML, 100 exact/accelerated seed pairs",
    "validate-science": "8x8, 2 ML, three physics configurations, seeds 0-4",
    "validate-sweep": "24x24, 2 ML, three temperatures/two fluxes, seeds 0/1/2",
    "benchmark-sizes": "64/128/256, Ga/N 0.82, 0.1 s, seed 0; sequential",
}


def _write_manifest(path: Path, **fields: object) -> dict[str, object]:
    data = json.loads(path.read_text()) if path.exists() else {}
    data.update(fields)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(temporary, path)
    return data


def _revision() -> dict[str, str | bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"git_commit": commit, "working_tree_dirty": dirty}


def _emit_progress(manifest: dict[str, object]) -> None:
    fields = (
        "workflow",
        "status",
        "stage",
        "completed",
        "total",
        "effective_workers",
        "batch_directory",
    )
    line = json.dumps({key: manifest.get(key) for key in fields})
    print(line, flush=True)
    batch_directory = manifest.get("batch_directory")
    if isinstance(batch_directory, str):
        with (ROOT / batch_directory / "progress.jsonl").open("a") as stream:
            stream.write(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workflow", choices=WORKFLOWS)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--seeds")
    parser.add_argument("--sizes")
    parser.add_argument("--batch-id")
    arguments = parser.parse_args()
    script, supports_workers, supports_seeds, supports_sizes, fixed_arguments = WORKFLOWS[
        arguments.workflow
    ]
    if arguments.seeds and not supports_seeds:
        parser.error(f"{arguments.workflow} does not accept seed overrides")
    if arguments.sizes and not supports_sizes:
        parser.error(f"{arguments.workflow} does not accept size overrides")

    requested_workers = resolve_workers(arguments.workers)
    effective_workers = requested_workers if supports_workers else 1
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    batch_id = arguments.batch_id or f"{timestamp}-{arguments.workflow}"
    if Path(batch_id).name != batch_id or batch_id in {".", ".."}:
        parser.error("batch ID must be a single path-safe name")
    batch_dir = BATCH_ROOT / batch_id
    artifacts = batch_dir / "artifacts"
    manifest_path = batch_dir / "manifest.json"
    batch_dir.mkdir(parents=True)
    command = [sys.executable, str(ROOT / "scripts" / script), *fixed_arguments]
    if supports_workers:
        command.extend(("--workers", str(requested_workers)))
    if arguments.seeds:
        command.extend(("--seeds", arguments.seeds))
    if arguments.sizes:
        command.extend(("--sizes", arguments.sizes))
    started_at = datetime.now(UTC)
    manifest = _write_manifest(
        manifest_path,
        workflow=arguments.workflow,
        status="running",
        started_at=started_at.isoformat(),
        requested_workers=requested_workers,
        effective_workers=effective_workers,
        seeds=arguments.seeds or "canonical defaults",
        sizes=arguments.sizes or "canonical defaults",
        command=command,
        batch_directory=str(batch_dir.relative_to(ROOT)),
        artifact_directory=str(artifacts.relative_to(ROOT)),
        source_revision=_revision(),
        configuration=WORKFLOW_PRESETS[arguments.workflow],
        completed=0,
        total=None,
    )
    _emit_progress(manifest)
    environment = os.environ.copy()
    environment["MBE_ARTIFACT_ROOT"] = str(artifacts)
    environment["MBE_PROGRESS_FILE"] = str(manifest_path)
    started = perf_counter()
    process = None

    def interrupt(_signal_number, _frame) -> None:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
        _write_manifest(
            manifest_path,
            status="interrupted",
            finished_at=datetime.now(UTC).isoformat(),
            elapsed_s=perf_counter() - started,
        )
        os._exit(130)

    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    with (batch_dir / "stdout.log").open("w") as stdout, (
        batch_dir / "stderr.log"
    ).open("w") as stderr:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        last_progress = (
            manifest.get("stage"),
            manifest.get("completed"),
            manifest.get("total"),
        )
        while (return_code := process.poll()) is None:
            sleep(0.2)
            progress = json.loads(manifest_path.read_text())
            signature = (
                progress.get("stage"),
                progress.get("completed"),
                progress.get("total"),
            )
            if signature != last_progress:
                _emit_progress(progress)
                last_progress = signature
    if return_code != 0:
        error_tail = (batch_dir / "stderr.log").read_text()[-4_000:]
        manifest = _write_manifest(
            manifest_path,
            status="failed",
            return_code=return_code,
            error=error_tail,
            finished_at=datetime.now(UTC).isoformat(),
            elapsed_s=perf_counter() - started,
        )
        _emit_progress(manifest)
        raise SystemExit(return_code)

    promoted = promote_artifacts(artifacts, ROOT, BATCH_ROOT / ".promotion.lock")
    manifest = _write_manifest(
        manifest_path,
        status="succeeded",
        return_code=0,
        completed=json.loads(manifest_path.read_text()).get("total"),
        generated_artifacts=promoted,
        finished_at=datetime.now(UTC).isoformat(),
        elapsed_s=perf_counter() - started,
    )
    _emit_progress(manifest)


if __name__ == "__main__":
    main()
