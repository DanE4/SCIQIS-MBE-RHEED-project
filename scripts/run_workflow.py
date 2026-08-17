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

from mbe_rheed_sim.workflows import (
    NEW_PROCESS_GROUP,
    git_revision,
    merge_json,
    promote_artifacts,
    resolve_workers,
    terminate_process_group,
)

ROOT = Path(__file__).resolve().parents[1]
BATCH_ROOT = ROOT / "outputs" / "batches"
# workers/seeds/sizes/duration flag whether the underlying script accepts that override.
WORKFLOWS = {
    "baseline": {
        "script": "reproduce_baseline.py",
        "preset": "8x8, 1 ML, seed 2026",
    },
    "figure3": {
        "script": "reproduce_figure3.py",
        "workers": True,
        "seeds": True,
        "sizes": True,
        "preset": "7x7, 40 s, Ga/N 0.89/0.82/0.68, seeds 2026/2027/2028",
    },
    "sweep": {
        "script": "run_parameter_sweep.py",
        "workers": True,
        "seeds": True,
        "preset": "16x16, 2 ML, T=700/850/1000 K, flux=0.25/0.5/0.75 ML/s, seeds 0/1/2",
    },
    "convergence": {
        "script": "check_lattice_convergence.py",
        "workers": True,
        "seeds": True,
        "sizes": True,
        "preset": "8/16/24, 2 ML, seeds 0/1/2",
    },
    "figure3-convergence": {
        "script": "check_figure3_convergence.py",
        "workers": True,
        "seeds": True,
        "sizes": True,
        "duration": True,
        "preset": (
            "8/16/32, Ga/N 0.82, 4 s, seeds 0/1/2; the accepted study is "
            "--sizes 32,64,128 --duration 40 --seeds 0,1,2,3,4"
        ),
    },
    "validate-acceleration": {
        "script": "validate_acceleration.py",
        "workers": True,
        "seeds": True,
        "preset": "7x7, 0.5 ML, 100 exact/accelerated seed pairs",
    },
    "validate-science": {
        "script": "validate_scientific_trends.py",
        "workers": True,
        "seeds": True,
        "preset": "8x8, 2 ML, three physics configurations, seeds 0-4",
    },
    "validate-sweep": {
        "script": "validate_sweep_lattice.py",
        "workers": True,
        "seeds": True,
        "preset": "24x24, 2 ML, three temperatures/two fluxes, seeds 0/1/2",
    },
    "benchmark-sizes": {
        "script": "benchmark_large_lattices.py",
        "sizes": True,
        "preset": "64/128/256, Ga/N 0.82, 0.1 s, seed 0; sequential",
    },
}
PROGRESS_FIELDS = ("workflow", "status", "stage", "completed", "total", "effective_workers")


def _emit_progress(manifest: dict[str, object]) -> None:
    print(json.dumps({key: manifest.get(key) for key in PROGRESS_FIELDS}), flush=True)


def _echo_new_logs(path: Path, offset: int) -> int:
    """Copy whatever the child appended to its log since `offset` onto our own stderr.

    The workflow scripts log through `mbe_rheed_sim.workflows`, which writes to stderr; the
    child's stream is a file so failures keep a durable tail, so this is what makes those
    lines visible live on the command line. Reuses the existing poll loop, no reader thread.
    """
    if not path.exists():
        return offset
    with path.open() as stream:
        stream.seek(offset)
        if text := stream.read():
            sys.stderr.write(text)
            sys.stderr.flush()
        return stream.tell()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workflow", choices=WORKFLOWS)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--seeds")
    parser.add_argument("--sizes")
    parser.add_argument("--duration")
    parser.add_argument("--batch-id")
    arguments = parser.parse_args()
    workflow = WORKFLOWS[arguments.workflow]
    for name in ("seeds", "sizes", "duration"):
        if getattr(arguments, name) and not workflow.get(name):
            parser.error(f"{arguments.workflow} does not accept --{name}")

    requested_workers = resolve_workers(arguments.workers)
    effective_workers = requested_workers if workflow.get("workers") else 1
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    batch_id = arguments.batch_id or f"{timestamp}-{arguments.workflow}"
    if Path(batch_id).name != batch_id or batch_id in {".", ".."}:
        parser.error("batch ID must be a single path-safe name")
    batch_dir = BATCH_ROOT / batch_id
    artifacts = batch_dir / "artifacts"
    manifest_path = batch_dir / "manifest.json"
    batch_dir.mkdir(parents=True)
    command = [sys.executable, str(ROOT / "scripts" / workflow["script"])]
    if workflow.get("workers"):
        command.extend(("--workers", str(requested_workers)))
    if arguments.seeds:
        command.extend(("--seeds", arguments.seeds))
    if arguments.sizes:
        command.extend(("--sizes", arguments.sizes))
    if arguments.duration:
        command.extend(("--duration", arguments.duration))
    started_at = datetime.now(UTC)
    manifest = merge_json(
        manifest_path,
        workflow=arguments.workflow,
        status="running",
        started_at=started_at.isoformat(),
        requested_workers=requested_workers,
        effective_workers=effective_workers,
        seeds=arguments.seeds or "canonical defaults",
        sizes=arguments.sizes or "canonical defaults",
        duration_s=arguments.duration or "canonical default",
        command=command,
        batch_directory=str(batch_dir.relative_to(ROOT)),
        artifact_directory=str(artifacts.relative_to(ROOT)),
        source_revision=git_revision(ROOT),
        configuration=workflow["preset"],
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
        if process is not None:
            terminate_process_group(process)
        merge_json(
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
            **NEW_PROCESS_GROUP,
        )
        last_progress = (
            manifest.get("stage"),
            manifest.get("completed"),
            manifest.get("total"),
        )
        log_offset = 0
        while (return_code := process.poll()) is None:
            sleep(0.2)
            log_offset = _echo_new_logs(batch_dir / "stderr.log", log_offset)
            progress = json.loads(manifest_path.read_text())
            signature = (
                progress.get("stage"),
                progress.get("completed"),
                progress.get("total"),
            )
            if signature != last_progress:
                _emit_progress(progress)
                last_progress = signature
        # The child can write between the last poll and exiting, so flush the tail.
        _echo_new_logs(batch_dir / "stderr.log", log_offset)
    if return_code != 0:
        error_tail = (batch_dir / "stderr.log").read_text()[-4_000:]
        manifest = merge_json(
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
    manifest = merge_json(
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
