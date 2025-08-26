from __future__ import annotations

from typing import Optional

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_poc.config import load_settings
from rag_poc.logging_utils import configure_logging
# from rag_poc.pipeline import ingest as ingest_pipeline, query as query_pipeline
from rag_poc.pipeline import ingest as ingest_pipeline, query_sources_only as query_pipeline


app = FastAPI(title="RAG POC", version="0.1.0")

# Allow Open WebUI (localhost:3000) and general local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup_logging():
    configure_logging()


class IngestRequest(BaseModel):
    config: Optional[str] = None


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None
    config: Optional[str] = None


@app.get("/health", operation_id="rag_health", tags=["tools"])
async def health() -> dict:
    return {"status": "ok"}


@app.post("/ingest", operation_id="rag_ingest", tags=["tools"])
async def ingest(req: IngestRequest) -> dict:
    logging.getLogger("rag.api").info("/ingest called config=%s", req.config)
    settings = load_settings(req.config)
    return await ingest_pipeline(settings)


@app.post("/query", operation_id="rag_query", tags=["tools"])
async def query(req: QueryRequest) -> dict:
    logging.getLogger("rag.api").info(
        "/query called top_k=%s text=%r",
        req.top_k,
        (req.question[:80] + ("…" if len(req.question) > 80 else "")),
    )
    settings = load_settings(req.config)
    return await query_pipeline(settings, req.question, req.top_k)
