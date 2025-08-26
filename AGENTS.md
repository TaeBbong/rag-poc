# Repository Guidelines

## Project Structure & Module Organization
- `src/`: application code (pipelines, retrievers, API).
  - `src/ingest/`: loaders/cleaners and chunking.
  - `src/index/`: embeddings, vector store adapters.
  - `src/app.py` or `src/server/`: API entrypoint (FastAPI).
- `tests/`: unit and integration tests mirroring `src/` packages.
- `configs/`: YAML/TOML configs (e.g., `configs/local.yaml`).
- `data/`: sample corpora and local artifacts (git-ignored).
- `scripts/`: task CLIs (e.g., `scripts/ingest.py`, `scripts/bench.py`).

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt` (Windows: `.venv\\Scripts\\activate`).
- `uv run uvicorn src.app:app --reload`: start the local API.
- `python scripts/ingest.py --config configs/local.yaml`: ingest and index documents.
- `pytest -q` (filter with `-k <name>`): run tests.
- `ruff check . && black .`: lint and format.

## Coding Style & Naming Conventions
- Python, PEP 8; format with Black; lint with Ruff; sort imports with isort.
- Public functions typed; keep mypy clean in CI.
- Names: modules `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE_CASE`.
- Tests named `test_*.py`; mirror `src/` module paths under `tests/`.

## Testing Guidelines
- Framework: `pytest`; prefer small, deterministic tests with fixtures in `tests/conftest.py`.
- Coverage target: ≥ 80% lines. Use: `pytest --cov=src --cov-report=term-missing`.
- Mock I/O and network; add regression tests for every bug fix.

## Commit & Pull Request Guidelines
- Conventional Commits (e.g., `feat(index): add hybrid retriever`).
- One topic per PR with description, linked issue, and before/after evidence for behavior changes.
- Update docs/config examples when changing defaults; include tests for new behavior.

## Security & Configuration
- Secrets via env (.env), never committed. Common: `OPENAI_API_KEY`, `EMBEDDING_MODEL`, `VECTOR_DB_URL`.
- Config precedence: CLI flags > env vars > `configs/*.yaml`.

## Architecture Overview
- Flow: ingest → chunk → embed → index → retrieve → generate.
- Components (embeddings, vector store, LLM) are swappable; keep boundaries narrow and inject via config.
