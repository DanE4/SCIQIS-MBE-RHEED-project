.PHONY: setup notebook test check reproduce publication reproduce-figure3 digitize-figure3 figure3-parameters validate-acceleration validate-science validate-sweep convergence convergence-figure3 convergence-figure3-64 benchmark-sizes sweep export

NOTEBOOK := notebooks/mbe_rheed.py
MARIMO_PORT ?= 2718
WORKER_ARG = $(if $(WORKERS),--workers $(WORKERS),)
SEED_ARG = $(if $(SEEDS),--seeds $(SEEDS),)
SIZE_ARG = $(if $(SIZES),--sizes $(SIZES),)

setup:
	uv sync --locked

notebook:
	uv run marimo edit $(NOTEBOOK) --no-token --port $(MARIMO_PORT)

test:
	uv run pytest

check:
	uv run ruff check .
	uv run marimo check --strict $(NOTEBOOK)
	uv run python $(NOTEBOOK)

reproduce:
	uv run python scripts/run_workflow.py baseline $(WORKER_ARG)

figure3-parameters:
	uv run python scripts/inspect_figure3_parameters.py

reproduce-figure3:
	uv run python scripts/run_workflow.py publication $(WORKER_ARG) $(SEED_ARG)

publication: reproduce-figure3

digitize-figure3:
	uv run python scripts/extract_figure3_reference.py

validate-acceleration:
	uv run python scripts/run_workflow.py validate-acceleration $(WORKER_ARG) $(SEED_ARG)

validate-science:
	uv run python scripts/run_workflow.py validate-science $(WORKER_ARG) $(SEED_ARG)

validate-sweep:
	uv run python scripts/run_workflow.py validate-sweep $(WORKER_ARG) $(SEED_ARG)

convergence:
	uv run python scripts/run_workflow.py convergence $(WORKER_ARG) $(SEED_ARG) $(SIZE_ARG)

convergence-figure3:
	uv run python scripts/run_workflow.py figure3-convergence $(WORKER_ARG) $(SEED_ARG) $(SIZE_ARG)

convergence-figure3-64:
	uv run python scripts/run_workflow.py figure3-convergence-64 $(WORKER_ARG) $(SEED_ARG) $(SIZE_ARG)

benchmark-sizes:
	uv run python scripts/run_workflow.py benchmark-sizes $(WORKER_ARG) $(SIZE_ARG)

sweep:
	uv run python scripts/run_workflow.py sweep $(WORKER_ARG) $(SEED_ARG)

export:
	mkdir -p outputs
	uv run marimo export html $(NOTEBOOK) -o outputs/mbe_rheed.html -f
