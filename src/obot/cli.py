"""Typer CLI: serve, build, ask."""

import logging
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from obot.config import ConfigError, load_settings, resolve_bots
from obot.core.bot import Bot
from obot.core.builder import build_bot
from obot.core.embedder import get_embedder
from obot.core.loader import LoadError
from obot.core.store import StoreLockedError, open_qdrant
from obot.llm.base import LLMError, create_llm

STORAGE_DIR = Path("storage")

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    """OBot: a RAG chatbot per data file."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


@app.command()
def serve() -> None:
    """Run the API server; bots build in the background."""
    # Imported here so build / ask do not load FastAPI.
    import uvicorn

    from obot.api.app import create_app

    try:
        settings = load_settings()
        bots = resolve_bots(settings)
        llm = create_llm(settings.llm, settings.secrets)
        api = create_app(settings, bots, llm, get_embedder, STORAGE_DIR)
        uvicorn.run(api, host=settings.server.host, port=settings.server.port)
    except (ConfigError, StoreLockedError) as e:
        _fail(str(e))


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
            # One at a time, printing each result so earlier bots still show
            # when a later one fails.
            for b in bots:
                r = build_bot(b, settings, get_embedder, qdrant, STORAGE_DIR)
                status = "skipped (unchanged)" if r.skipped else "built"
                typer.echo(f"{r.bot}: {status}, {r.record_count} records")
        finally:
            qdrant.close()
    except (ConfigError, LoadError, StoreLockedError) as e:
        _fail(str(e))


@app.command()
def ask(
    name: Annotated[str, typer.Argument(help="Bot name.")],
    question: Annotated[str, typer.Argument(help="Question to ask.")],
) -> None:
    """Ask a bot from the terminal, building it first if needed."""
    try:
        settings = load_settings()
        bot = next((b for b in resolve_bots(settings) if b.name == name), None)
        if bot is None:
            _fail(f"unknown bot '{name}'")
        qdrant = open_qdrant(STORAGE_DIR / "qdrant")
        try:
            build_bot(bot, settings, get_embedder, qdrant, STORAGE_DIR)
            llm = create_llm(settings.llm, settings.secrets)
            result = Bot(
                name, qdrant, get_embedder(), llm, settings.retrieval.top_k
            ).ask(question)
        finally:
            qdrant.close()
    except (ConfigError, LoadError, StoreLockedError, LLMError) as e:
        _fail(str(e))
    typer.echo(result.answer)
    typer.echo("\nsources:")
    for hit in result.sources:
        typer.echo(f"  {hit.record_id} (score {hit.score:.3f})")


def _fail(message: str) -> NoReturn:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(1)
