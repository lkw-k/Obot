"""Typer CLI: serve, build, ask."""

import logging
from pathlib import Path
from typing import Annotated

import typer

from obot.config import ConfigError, load_settings, resolve_bots
from obot.core.builder import build_all
from obot.core.embedder import get_embedder
from obot.core.loader import LoadError
from obot.core.store import StoreLockedError, open_qdrant

STORAGE_DIR = Path("storage")

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    """OBot: a RAG chatbot per data file."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


@app.command()
def build(
    bot: Annotated[str | None, typer.Option(help="Build only this bot.")] = None,
) -> None:
    """Build bots whose data or config changed."""
    try:
        settings = load_settings()
        bots = resolve_bots(settings)
        if bot is not None:
            bots = [b for b in bots if b.name == bot]
            if not bots:
                _fail(f"unknown bot '{bot}'")
        if not bots:
            _fail("no bots: add JSON / JSONL / CSV files to data/ or set 'bots'")
        qdrant = open_qdrant(STORAGE_DIR / "qdrant")
        try:
            results = build_all(bots, settings, get_embedder, qdrant, STORAGE_DIR)
        finally:
            qdrant.close()
    except (ConfigError, LoadError, StoreLockedError) as e:
        _fail(str(e))
    for r in results:
        status = "skipped (unchanged)" if r.skipped else "built"
        typer.echo(f"{r.bot}: {status}, {r.record_count} records")


def _fail(message: str) -> None:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(1)
