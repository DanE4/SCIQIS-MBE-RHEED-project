"""Bounded process execution and safe artifact handling for simulation workflows."""

import argparse
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter
from typing import cast

from mbe_rheed_sim.config import SimulationConfig
from mbe_rheed_sim.kmc import SimulationResult, run

if sys.platform == "win32":
    import msvcrt

    # msvcrt locks a byte range at the current file offset, so both calls seek to byte 0
    # first; a lock taken at an append position would not exclude anyone.
    def _lock_file(handle) -> None:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)

    def _unlock_file(handle) -> None:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)

    # Windows has no POSIX process group, and `start_new_session` is silently ignored
    # there. A new console process group is the equivalent, and a break event is the one
    # signal that reaches all of it, so workers still die with their supervisor.
    NEW_PROCESS_GROUP = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}

    def terminate_process_group(process: subprocess.Popen) -> None:
        """Stop a supervisor and the workers it started; pair with `**NEW_PROCESS_GROUP`."""
        if process.poll() is None:
            os.kill(process.pid, signal.CTRL_BREAK_EVENT)
else:
    import fcntl

    def _lock_file(handle) -> None:
        fcntl.flock(handle, fcntl.LOCK_EX)

    def _unlock_file(handle) -> None:
        fcntl.flock(handle, fcntl.LOCK_UN)

    NEW_PROCESS_GROUP = {"start_new_session": True}

    def terminate_process_group(process: subprocess.Popen) -> None:
        """Stop a supervisor and the workers it started; pair with `**NEW_PROCESS_GROUP`."""
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)


DEFAULT_WORKERS = min(10, max(1, (os.cpu_count() or 1) - 1))
WORKER_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

log = logging.getLogger("mbe")


def setup_logging() -> None:
    """Send progress to stderr, leaving stdout as the machine-readable JSON summary.

    `MBE_LOG_LEVEL=DEBUG` (or WARNING to quieten it) overrides the default.
    """
    logging.basicConfig(
        level=os.environ.get("MBE_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


def log_progress(label: str) -> Callable[[float], None]:
    """Build a `run(on_progress=...)` callback that logs one line per 10% of a trajectory.

    For a single long run, which is the case a completed-count cannot report on: one KMC
    trajectory is sequential, so it is one item that stays at 0/1 until it finishes.
    """
    reported = -1

    def report(fraction: float) -> None:
        nonlocal reported
        decile = int(fraction * 10)
        if decile > reported:
            reported = decile
            log.info("%s %3.0f%%", label, fraction * 100)

    return report


def resolve_workers(requested: int | None = None) -> int:
    """Resolve CLI, environment, then safe-default worker count."""
    raw = requested if requested is not None else os.environ.get("MBE_WORKERS", DEFAULT_WORKERS)
    try:
        workers = int(raw)
    except (TypeError, ValueError) as error:
        raise ValueError("workers must be an integer") from error
    available = os.cpu_count() or 1
    if not 1 <= workers <= available:
        raise ValueError(f"workers must be between 1 and {available}")
    return workers


def parse_int_values(value: str | None, defaults: Sequence[int]) -> tuple[int, ...]:
    """Parse a comma-separated positive integer override."""
    if value is None:
        return tuple(defaults)
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise ValueError("values must be comma-separated integers") from error
    if not values or any(item < 0 for item in values) or len(set(values)) != len(values):
        raise ValueError("values must be unique non-negative integers")
    return values


def parse_workflow_args(
    *,
    workers: bool = True,
    seeds: Sequence[int] | None = None,
    sizes: Sequence[int] | None = None,
    duration_s: float | None = None,
) -> dict[str, object]:
    """Parse the shared --workers/--seeds/--sizes/--duration CLI into `main()` keywords.

    Pass the script's canonical defaults for the overrides it accepts; omit the rest so the
    parser rejects a flag the script cannot honour instead of silently ignoring it.
    """
    setup_logging()
    parser = argparse.ArgumentParser()
    if workers:
        parser.add_argument("--workers", type=int)
    if seeds is not None:
        parser.add_argument("--seeds")
    if sizes is not None:
        parser.add_argument("--sizes")
    if duration_s is not None:
        parser.add_argument("--duration", type=float, default=duration_s)
    arguments = parser.parse_args()

    options: dict[str, object] = {}
    if workers:
        options["workers"] = resolve_workers(arguments.workers)
    if seeds is not None:
        options["seeds"] = parse_int_values(arguments.seeds, seeds)
    if sizes is not None:
        options["sizes"] = parse_int_values(arguments.sizes, sizes)
    if duration_s is not None:
        if not 0 < arguments.duration <= 1000:
            raise ValueError("duration must be greater than 0 and no more than 1000 s")
        options["duration_s"] = arguments.duration
    return options


def artifact_root(project_root: Path) -> Path:
    """Return the project root or the supervisor-provided mirrored artifact root."""
    return Path(os.environ.get("MBE_ARTIFACT_ROOT", project_root))


def merge_json(path: Path, **fields: object) -> dict[str, object]:
    """Atomically merge fields into a JSON file, creating it when absent."""
    data = json.loads(path.read_text()) if path.exists() else {}
    data.update(fields)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(temporary, path)
    return data


def update_progress(**fields: object) -> None:
    """Log progress, and merge it into the active batch manifest when one is configured.

    Every workflow already funnels its stage reports through here, so logging at this one
    point gives each script a command-line trace without touching any of them.
    """
    if (stage := fields.get("stage")) is not None:
        total, completed = fields.get("total"), fields.get("completed")
        log.info("%s %s", stage, f"{completed}/{total}" if total else "starting")
    path = os.environ.get("MBE_PROGRESS_FILE")
    if path is not None:
        merge_json(Path(path), **fields)


def git_revision(root: Path) -> dict[str, str | bool]:
    """Record the commit and whether the working tree had uncommitted changes."""
    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()

    return {
        "git_commit": git("rev-parse", "HEAD"),
        "working_tree_dirty": bool(git("status", "--porcelain")),
    }


def run_parallel[T, R](
    function: Callable[[T], R],
    items: Sequence[T],
    *,
    workers: int,
    description: str,
) -> list[R]:
    """Run independent work with spawn processes and return input-ordered results."""
    effective_workers = min(resolve_workers(workers), max(1, len(items)))
    update_progress(
        stage=description,
        completed=0,
        total=len(items),
        effective_workers=effective_workers,
    )
    if effective_workers == 1:
        results = []
        for completed, item in enumerate(items, start=1):
            results.append(function(item))
            update_progress(stage=description, completed=completed, total=len(items))
        return results

    for name in WORKER_ENVIRONMENT:
        os.environ[name] = "1"
    missing = object()
    results: list[R | object] = [missing] * len(items)
    executor = ProcessPoolExecutor(
        max_workers=effective_workers,
        mp_context=get_context("spawn"),
    )
    futures = {executor.submit(function, item): index for index, item in enumerate(items)}
    completed = 0
    try:
        for future in as_completed(futures):
            results[futures[future]] = future.result()
            completed += 1
            update_progress(stage=description, completed=completed, total=len(items))
    except BaseException:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    executor.shutdown(wait=True)
    return [cast(R, result) for result in results]


def run_timed(config: SimulationConfig) -> tuple[SimulationResult, float]:
    """Run one picklable simulation task and report worker-local wall time."""
    started = perf_counter()
    return run(config), perf_counter() - started


def promote_artifacts(source_root: Path, project_root: Path, lock_path: Path) -> list[str]:
    """Copy a complete mirrored artifact tree into canonical paths atomically per file."""
    files = sorted(path for path in source_root.rglob("*") if path.is_file())
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    promoted = []
    with lock_path.open("a+") as lock:
        _lock_file(lock)
        for source in files:
            relative = source.relative_to(source_root)
            destination = project_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.promoting")
            shutil.copy2(source, temporary)
            os.replace(temporary, destination)
            # as_posix keeps promoted paths identical across platforms, for artifact manifests.
            promoted.append(relative.as_posix())
        _unlock_file(lock)
    return promoted
