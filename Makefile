.PHONY: help install lint format test build check-dist clean rules

help:
	@echo "Available targets:"
	@echo "  install     Install dev dependencies (poetry)"
	@echo "  lint        Run ruff + black checks"
	@echo "  format      Auto-format with ruff + black"
	@echo "  test        Run pytest"
	@echo "  build       Build sdist + wheel"
	@echo "  check-dist  Validate dist metadata (twine check)"
	@echo "  clean       Remove build/test artifacts"
	@echo "  rules       Regenerate RULES.md"

install:
	poetry install --with dev

lint:
	poetry run ruff check .
	poetry run black --check .

format:
	poetry run ruff check . --fix
	poetry run black .

test:
	poetry run pytest

build:
	poetry build

check-dist:
	poetry run twine check dist/*

clean:
	rm -rf dist build .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete

rules:
	poetry run python scripts/generate_rules.py
