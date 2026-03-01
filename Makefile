SOURCE = main.py

.PHONY: format lint test

all: format lint

format:
	uv run ruff format $(SOURCE)
	uv run ruff check --fix $(SOURCE)

lint:
	uv run ruff check $(SOURCE)
	uv run ty check $(SOURCE)
