.DEFAULT_GOAL := help

.PHONY: help install test lint format typecheck security check build clean clean-all

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Sync dependencies and install pre-commit hooks
	uv sync
	uv run pre-commit install

test: ## Run the test suite
	uv run pytest

lint: ## Lint and autofix with ruff
	uv run ruff check . --fix

format: ## Format code with ruff
	uv run ruff format .

typecheck: ## Type-check with ty
	uv run ty check

security: ## Security scan with bandit
	uv run bandit -c pyproject.toml -r tap/

check: lint typecheck security test ## Run all checks (lint, types, security, tests)

build: ## Build sdist and wheel
	uv build

clean: ## Remove caches and build artifacts
	rm -rf build dist wheels *.egg-info .eggs
	rm -rf .pytest_cache .ruff_cache .mypy_cache .ty_cache .coverage coverage.xml htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.py[co]' -delete

clean-all: clean ## Also remove the virtual environment
	rm -rf .venv
