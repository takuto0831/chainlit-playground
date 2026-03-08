TARGET ?= hello

SOURCE = src

.PHONY: format lint run setup-db teardown-db

all: format lint

format:
	uv run ruff format $(SOURCE)
	uv run ruff check --fix $(SOURCE)

lint:
	uv run ruff check $(SOURCE)
	uv run ty check $(SOURCE)

run:
	uv run uvicorn chainlit_playground.main:app --reload --factory --host localhost --port 8000

setup-db:
	docker compose -f sandbox/postgres/docker-compose.yaml up -d

teardown-db:
	docker compose -f sandbox/postgres/docker-compose.yaml stop
