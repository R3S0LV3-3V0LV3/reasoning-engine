.PHONY: format lint type test check
format:
	uv run ruff format .
lint:
	uv run ruff format --check .
	uv run ruff check .
type:
	uv run mypy --strict src tests
test:
	uv run pytest
check: lint type test

