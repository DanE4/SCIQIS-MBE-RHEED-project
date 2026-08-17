# Every workflow name accepted by scripts/run_workflow.py is a make target of the same name.
WORKFLOWS := baseline figure3 sweep convergence figure3-convergence \
	validate-acceleration validate-science validate-sweep benchmark-sizes

.PHONY: setup notebook test check export figure3-parameters digitize-figure3 \
	reproduce reproduce-figure3 convergence-figure3 gallery $(WORKFLOWS)

NOTEBOOK := notebooks/mbe_rheed.py
MARIMO_PORT ?= 2718
WORKER_ARG = $(if $(WORKERS),--workers $(WORKERS),)
SEED_ARG = $(if $(SEEDS),--seeds $(SEEDS),)
SIZE_ARG = $(if $(SIZES),--sizes $(SIZES),)
DURATION_ARG = $(if $(DURATION),--duration $(DURATION),)

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

$(WORKFLOWS):
	uv run python scripts/run_workflow.py $@ $(WORKER_ARG) $(SEED_ARG) $(SIZE_ARG) $(DURATION_ARG)

# Long-standing aliases kept so documented commands keep working.
reproduce: baseline
reproduce-figure3: figure3
convergence-figure3: figure3-convergence

gallery:
	uv run python scripts/build_gallery.py

figure3-parameters:
	uv run python scripts/inspect_figure3_parameters.py

digitize-figure3:
	uv run python scripts/extract_figure3_reference.py

export:
	mkdir -p outputs
	uv run marimo export html $(NOTEBOOK) -o outputs/mbe_rheed.html -f
