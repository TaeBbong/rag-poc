#!/usr/bin/env python
from __future__ import annotations

import typer
from rag_poc.logging_utils import configure_logging

from rag_poc.config import load_settings
from rag_poc.pipeline import query as run_query


app = typer.Typer(add_completion=False)


@app.command()
def main(
    question: str = typer.Argument(..., help="Your question"),
    top_k: int = typer.Option(None, help="Override top-k similarity"),
    config: str = typer.Option("configs/local.yaml", help="Path to YAML config"),
):
    configure_logging()
    typer.echo("[query] loading settings …")
    settings = load_settings(config)
    typer.echo("[query] retrieving and asking LLM …")
    result = run_query(settings, question, top_k)
    typer.secho("Answer:\n" + result["answer"], fg=typer.colors.CYAN)
    if result.get("sources"):
        typer.echo("\nSources:")
        for i, s in enumerate(result["sources"], 1):
            typer.echo(f"[{i}] score={s.get('score'):.3f}\n{s.get('text')[:400]}\n")
    if "seconds" in result:
        typer.echo(f"[query] LLM answered in {result['seconds']}s")


if __name__ == "__main__":
    app()
