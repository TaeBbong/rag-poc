from __future__ import annotations

import logging

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

from .config import Settings


def get_llm(settings: Settings) -> Ollama:
    logging.getLogger("rag.llm").info(
        "Using LLM model=%s timeout=%ss base=%s",
        settings.llm_model,
        settings.llm_timeout,
        settings.ollama_base_url,
    )
    return Ollama(model=settings.llm_model, base_url=settings.ollama_base_url, request_timeout=180)


def get_embed_model(settings: Settings) -> OllamaEmbedding:
    logging.getLogger("rag.embed").info(
        "Using Embedding model=%s timeout=%ss base=%s",
        settings.embed_model,
        settings.embed_timeout,
        settings.ollama_base_url,
    )
    return OllamaEmbedding(model_name=settings.embed_model, base_url=settings.ollama_base_url)

