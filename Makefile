.PHONY: bootstrap format lint type test check quality
bootstrap:
	uv sync --locked --all-groups --no-install-project
	uv pip install --python .venv/bin/python --no-deps . --no-build-isolation
format: bootstrap
	uv run --no-sync ruff format .
lint: bootstrap
	uv run --no-sync ruff format --check .
	uv run --no-sync ruff check .
type: bootstrap
	uv run --no-sync mypy --strict src tests
test: bootstrap
	uv run --no-sync pytest
check: lint type test
quality: check
