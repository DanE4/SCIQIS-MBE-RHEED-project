"""Launching and monitoring `scripts/run_workflow.py` from inside the notebook.

The notebook keeps only the reactive wiring (`mo.state` for the running process, `mo.ui.refresh`
to re-poll). Process control and manifest reading live here so they stay testable.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import marimo as mo

from mbe_rheed_sim.workflows import NEW_PROCESS_GROUP, terminate_process_group

ROOT = Path(__file__).resolve().parents[2]
EXPENSIVE_WORKFLOWS = frozenset({"benchmark-sizes"})

WORKFLOW_LABELS = {
    "Baseline reproduction": "baseline",
    "GaN Ga/N comparison": "figure3",
    "Temperature/flux sweep": "sweep",
    "Generic convergence": "convergence",
    "Figure 3 convergence": "figure3-convergence",
    "Acceleration validation": "validate-acceleration",
    "Scientific-trend validation": "validate-science",
    "Sweep lattice validation": "validate-sweep",
    "64/128/256 runtime benchmark": "benchmark-sizes",
}


def controls(on_change) -> mo.ui.form:
    """The batch launcher form."""
    return (
        mo.md("""
        | Batch setting | Selection |
        |---|---|
        | Workflow | {workflow} |
        | Worker processes | {workers} |
        | Seed override | {seeds} |
        | Lattice-size override | {sizes} |
        | Confirm expensive workflow | {confirm_expensive} |
    """)
        .batch(
            workflow=mo.ui.dropdown(
                WORKFLOW_LABELS, value="GaN Ga/N comparison", label="Workflow"
            ),
            workers=mo.ui.slider(1, 8, value=4, label="Worker processes"),
            seeds=mo.ui.text(
                value="", placeholder="blank = canonical; e.g. 0,1,2", label="Comma-separated seeds"
            ),
            sizes=mo.ui.text(
                value="",
                placeholder="blank = canonical; e.g. 8,16,32",
                label="Comma-separated lattice sizes",
            ),
            confirm_expensive=mo.ui.checkbox(
                value=False,
                label="I understand a 64x64 convergence or 128/256 benchmark may take minutes",
            ),
        )
        .form(submit_button_label="Launch batch workflow", bordered=True, on_change=on_change)
    )


def needs_confirmation(request: dict) -> bool:
    return request["workflow"] in EXPENSIVE_WORKFLOWS or "64" in request["sizes"]


def launch(request: dict) -> dict:
    """Start one workflow in its own process group and return the state to poll."""
    batch_id = f"notebook-{time.time_ns()}-{request['workflow']}"
    command = [
        sys.executable,
        str(ROOT / "scripts/run_workflow.py"),
        request["workflow"],
        "--workers",
        str(request["workers"]),
        "--batch-id",
        batch_id,
    ]
    for flag in ("seeds", "sizes"):
        if request[flag].strip():
            command.extend((f"--{flag}", request[flag].strip()))
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **NEW_PROCESS_GROUP,
    )
    return {
        "process": process,
        "workflow": request["workflow"],
        "manifest": ROOT / "outputs/batches" / batch_id / "manifest.json",
        "started": time.time(),
        "reloaded": False,
    }


def cancel(state: dict | None) -> None:
    """Terminate the whole process group so worker processes die with the supervisor."""
    if state is not None:
        terminate_process_group(state["process"])


def read_status(state: dict | None) -> tuple[dict, float, bool]:
    """Return (manifest, elapsed seconds, whether artifacts just became promotable)."""
    if state is None:
        return {"status": "idle", "completed": 0, "total": None}, 0.0, False
    manifest = state["manifest"]
    status = (
        json.loads(manifest.read_text())
        if manifest.exists()
        else {"status": "starting", "completed": 0, "total": None}
    )
    elapsed = float(status.get("elapsed_s", time.time() - state["started"]))
    finished = state["process"].poll() is not None
    just_promoted = finished and status["status"] == "succeeded" and not state["reloaded"]
    if just_promoted:
        state["reloaded"] = True
    return status, elapsed, just_promoted


def status_markdown(status: dict, elapsed: float) -> str:
    total = status.get("total")
    completed = status.get("completed", 0)
    progress = f"{completed}/{total}" if total is not None else str(completed)
    error = status.get("error")
    return (
        f"**Status:** `{status.get('status', 'unknown')}`  \n"
        f"**Stage:** {status.get('stage', 'waiting for worker')}  \n"
        f"**Progress:** {progress}  \n"
        f"**Workers:** {status.get('effective_workers', 'n/a')}  \n"
        f"**Elapsed:** {elapsed:.1f} s  \n"
        f"**History:** `{status.get('batch_directory', 'not created yet')}`"
        + (f"\n\n```text\n{error[-1500:]}\n```" if error else "")
    )
