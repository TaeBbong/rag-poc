from __future__ import annotations

from typing import Dict, List

from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter

from .config import Settings
from .llm import get_llm, get_embed_model  # fallback
from .vector import get_vector_store


def ingest(settings: Settings) -> Dict[str, int]:
    docs = SimpleDirectoryReader(settings.data_dir, recursive=True).load_data()
    splitter = SentenceSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )

    vector_store = get_vector_store(settings)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    embed_model = get_embed_model(settings)
    index = VectorStoreIndex.from_documents(
        docs,
        storage_context=storage_context,
        transformations=[splitter],
        embed_model=embed_model,
    )

    return {"documents": len(docs)}


def query(settings: Settings, question: str, top_k: int | None = None) -> Dict:
    vector_store = get_vector_store(settings)
    embed_model = get_embed_model(settings)
    llm = get_llm(settings)

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )
    qe = index.as_query_engine(llm=llm, similarity_top_k=top_k or settings.top_k)
    resp = qe.query(question)

    sources: List[Dict] = []
    for sn in getattr(resp, "source_nodes", []) or []:
        sources.append(
            {
                "text": sn.get_text(),
                "score": sn.score,
                "metadata": sn.metadata or {},
            }
        )

    return {"answer": str(resp), "sources": sources}
