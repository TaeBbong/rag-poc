from __future__ import annotations

from typing import Optional
import logging

from pymilvus import MilvusClient  # noqa: F401

from .config import Settings


def get_vector_store(settings: Settings):
    # Prefer explicit client (avoids optional milvus_lite dependency paths)
    uri = f"http://{settings.milvus_host}:{settings.milvus_port}"
    logger = logging.getLogger("rag.milvus")
    logger.info(
        "Using Milvus at %s (collection=%s, dim=%s)",
        uri,
        settings.milvus_collection,
        settings.milvus_dim,
    )

    # Default to cosine distance with HNSW; adjust as needed.
    index_params = {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 8, "efConstruction": 200},
    }
    search_params = {"metric_type": "COSINE", "params": {"ef": 64}}

    # Lazy import to avoid accidental milvus_lite import paths on Windows
    try:
        from llama_index.vector_stores.milvus import MilvusVectorStore  # type: ignore
    except ModuleNotFoundError as e:
        if getattr(e, "name", "") == "milvus_lite":
            raise RuntimeError(
                "MilvusVectorStore attempted to import milvus_lite. This project runs in remote Milvus mode only.\n"
                "Please ensure llama-index-vector-stores-milvus and pymilvus are installed and up to date,"
                " and do not install milvus-lite on Windows."
            ) from e
        raise

    # Prefer explicit remote URI; avoid any local DB default
    return MilvusVectorStore(
        uri=uri,
        collection_name=settings.milvus_collection,
        dim=settings.milvus_dim,
        index_params=index_params,
        search_params=search_params,
        overwrite=False,
        text_key="text",
    )
