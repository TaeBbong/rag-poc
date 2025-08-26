#!/usr/bin/env python
from __future__ import annotations

import typer
from rag_poc.logging_utils import configure_logging

from rag_poc.config import load_settings
from rag_poc.pipeline import ingest as run_ingest


app = typer.Typer(add_completion=False)


@app.command()
def main(config: str = typer.Option("configs/local.yaml", help="Path to YAML config")):
    configure_logging()
    typer.echo("[ingest] loading settings …")
    settings = load_settings(config)
    typer.echo("[ingest] reading documents and embedding → Milvus …")
    result = run_ingest(settings)
    typer.secho(f"Ingested {result['documents']} documents.", fg=typer.colors.GREEN)
    if "seconds" in result:
        typer.echo(f"[ingest] done in {result['seconds']}s")


if __name__ == "__main__":
    app()
