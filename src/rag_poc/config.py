from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv


@dataclass
class Settings:
    # Models / endpoints
    llm_model: str = "qwen3:4b"
    embed_model: str = "nomic-embed-text"
    ollama_base_url: str = "http://localhost:11434"

    # Milvus
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection: str = "rag_docs"
    milvus_dim: int = 768

    # Ingestion
    data_dir: str = "data"
    chunk_size: int = 1024
    chunk_overlap: int = 100

    # Query
    top_k: int = 5

    # Timeouts (seconds)
    llm_timeout: float = 120.0
    embed_timeout: float = 60.0


def _read_yaml(path: Optional[os.PathLike[str] | str]) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _flatten(cfg: Dict[str, Any]) -> Dict[str, Any]:
    # Map nested YAML to Settings fields
    out: Dict[str, Any] = {}
    if not cfg:
        return out
    out["llm_model"] = cfg.get("llm_model")
    out["embed_model"] = cfg.get("embed_model")
    out["ollama_base_url"] = cfg.get("ollama_base_url")

    milvus = cfg.get("milvus", {}) or {}
    out["milvus_host"] = milvus.get("host")
    out["milvus_port"] = milvus.get("port")
    out["milvus_collection"] = milvus.get("collection")
    out["milvus_dim"] = milvus.get("dim")

    ingest = cfg.get("ingest", {}) or {}
    out["data_dir"] = ingest.get("data_dir")
    out["chunk_size"] = ingest.get("chunk_size")
    out["chunk_overlap"] = ingest.get("chunk_overlap")

    query = cfg.get("query", {}) or {}
    out["top_k"] = query.get("top_k")

    # Timeouts can be top-level or under `timeouts:`
    timeouts = cfg.get("timeouts", {}) or {}
    out["llm_timeout"] = cfg.get("llm_timeout", timeouts.get("llm"))
    out["embed_timeout"] = cfg.get("embed_timeout", timeouts.get("embed"))

    return {k: v for k, v in out.items() if v is not None}


def load_settings(config_path: Optional[str] = None) -> Settings:
    """Load settings from .env, YAML, and env vars (env wins)."""
    load_dotenv(override=False)

    # Base from YAML
    yaml_cfg = _flatten(_read_yaml(config_path))

    # Env overrides
    def getenv(key: str, default: Any) -> Any:
        return os.getenv(key, default)

    s = Settings(
        llm_model=str(getenv("LLM_MODEL", yaml_cfg.get("llm_model", Settings.llm_model))),
        embed_model=str(
            getenv("EMBED_MODEL", yaml_cfg.get("embed_model", Settings.embed_model))
        ),
        ollama_base_url=str(
            getenv(
                "OLLAMA_BASE_URL", yaml_cfg.get("ollama_base_url", Settings.ollama_base_url)
            )
        ),
        milvus_host=str(
            getenv("MILVUS_HOST", yaml_cfg.get("milvus_host", Settings.milvus_host))
        ),
        milvus_port=int(
            getenv("MILVUS_PORT", yaml_cfg.get("milvus_port", Settings.milvus_port))
        ),
        milvus_collection=str(
            getenv(
                "MILVUS_COLLECTION",
                yaml_cfg.get("milvus_collection", Settings.milvus_collection),
            )
        ),
        milvus_dim=int(
            getenv("MILVUS_DIM", yaml_cfg.get("milvus_dim", Settings.milvus_dim))
        ),
        data_dir=str(getenv("DATA_DIR", yaml_cfg.get("data_dir", Settings.data_dir))),
        chunk_size=int(
            getenv("CHUNK_SIZE", yaml_cfg.get("chunk_size", Settings.chunk_size))
        ),
        chunk_overlap=int(
            getenv("CHUNK_OVERLAP", yaml_cfg.get("chunk_overlap", Settings.chunk_overlap))
        ),
        top_k=int(getenv("TOP_K", yaml_cfg.get("top_k", Settings.top_k))),
        llm_timeout=float(
            getenv("LLM_TIMEOUT", yaml_cfg.get("llm_timeout", Settings.llm_timeout))
        ),
        embed_timeout=float(
            getenv(
                "EMBED_TIMEOUT", yaml_cfg.get("embed_timeout", Settings.embed_timeout)
            )
        ),
    )
    return s
