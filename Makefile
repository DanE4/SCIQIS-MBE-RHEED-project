.PHONY: setup notebook test check reproduce reproduce-figure3 figure3-parameters validate-acceleration validate-science convergence sweep export

NOTEBOOK := notebooks/mbe_rheed.py

setup:
	uv sync --locked

notebook:
	uv run marimo edit $(NOTEBOOK)

test:
	uv run pytest

check:
	uv run ruff check .
	uv run marimo check --strict $(NOTEBOOK)
	uv run python $(NOTEBOOK)

reproduce:
	uv run python scripts/reproduce_baseline.py

figure3-parameters:
	uv run python scripts/inspect_figure3_parameters.py

reproduce-figure3:
	uv run python scripts/reproduce_figure3.py

validate-acceleration:
	uv run python scripts/validate_acceleration.py

validate-science:
	uv run python scripts/validate_scientific_trends.py

convergence:
	uv run python scripts/check_lattice_convergence.py

sweep:
	uv run python scripts/run_parameter_sweep.py

export:
	mkdir -p outputs
	uv run marimo export html $(NOTEBOOK) -o outputs/mbe_rheed.html -f
