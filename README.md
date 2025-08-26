# RAG POC (LlamaIndex + Milvus + Ollama)

This project provides a minimal Retrieval-Augmented Generation (RAG) service using LlamaIndex with Milvus as a vector store and Ollama as the local LLM and embedding provider. Open WebUI can be used as a chat interface alongside this service.

## Quickstart (uv)
1. Ensure prerequisites:
   - Ollama running locally at `http://localhost:11434`
   - Pull models: `ollama pull llama3.1` and `ollama pull nomic-embed-text`
   - Start Milvus: `docker compose -f docker-compose.milvus.yml up -d`

2. Create env and install deps:
   - `uv venv`
   - `uv pip install -e .[dev]`

3. Prepare some files in `data/` for ingestion.

4. Ingest and query:
   - `uv run scripts/ingest.py --config configs/local.yaml`
   - `uv run scripts/query.py "What does the README describe?"`

5. Run API:
   - `uv run uvicorn src.app:app --reload --port 8000`

Endpoints: `GET /health`, `POST /ingest`, `POST /query`.

## Configuration
Copy `.env.example` to `.env` or edit `configs/local.yaml`.

Key settings: models (`LLM_MODEL`, `EMBED_MODEL`), Milvus (`MILVUS_HOST`, `MILVUS_PORT`, `MILVUS_COLLECTION`, `MILVUS_DIM`), and ingest/query parameters.

## Troubleshooting
- ModuleNotFoundError: `milvus_lite`
  - Our integration uses an explicit `MilvusClient` to avoid `milvus_lite`. Ensure deps are up to date: `uv pip install -e .[dev]`.
  - Alternatively, install the optional package: `uv pip install milvus-lite`.
- Cannot connect to Milvus: verify `docker compose -f docker-compose.milvus.yml ps` shows all services healthy and `http://localhost:9091/healthz` returns OK.
