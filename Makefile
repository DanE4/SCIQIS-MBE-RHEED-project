.PHONY: setup notebook test check reproduce export

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

export:
	mkdir -p outputs
	uv run marimo export html $(NOTEBOOK) -o outputs/mbe_rheed.html -f

