"""Run one reproducible workflow with history, progress, and safe promotion."""

import argparse
import json
import os
import signal
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workflow", choices=WORKFLOWS)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--seeds")
    parser.add_argument("--sizes")
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
    batch_dir = BATCH_ROOT / f"{timestamp}-{arguments.workflow}"
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
    _write_manifest(
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
        completed=0,
        total=None,
    )
    environment = os.environ.copy()
    environment["MBE_ARTIFACT_ROOT"] = str(artifacts)
    environment["MBE_PROGRESS_FILE"] = str(manifest_path)
    started = perf_counter()
    process = None

    def interrupt(_signal_number, _frame) -> None:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait()
        _write_manifest(
            manifest_path,
            status="interrupted",
            finished_at=datetime.now(UTC).isoformat(),
            elapsed_s=perf_counter() - started,
        )
        raise SystemExit(130)

    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    with (batch_dir / "stdout.log").open("w") as stdout, (
        batch_dir / "stderr.log"
    ).open("w") as stderr:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr)
        return_code = process.wait()
    if return_code != 0:
        error_tail = (batch_dir / "stderr.log").read_text()[-4_000:]
        _write_manifest(
            manifest_path,
            status="failed",
            return_code=return_code,
            error=error_tail,
            finished_at=datetime.now(UTC).isoformat(),
            elapsed_s=perf_counter() - started,
        )
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
    print(json.dumps({"manifest": str(manifest_path.relative_to(ROOT)), **manifest}))


if __name__ == "__main__":
    main()
