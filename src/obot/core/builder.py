"""Build pipeline and change detection (spec §5.2)."""

import hashlib
import json
import logging
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient

from obot.config import BotConfig, Settings
from obot.core import store
from obot.core.chunker import chunk_text
from obot.core.embedder import EMBEDDING_MODEL, Embedder
from obot.core.loader import LoadError, load_file
from obot.core.schema import (
    FALLBACK_ID_FIELD,
    SCHEMA_VERSION,
    analyze_schema,
    save_schema,
)

logger = logging.getLogger(__name__)


@dataclass
class BuildResult:
    bot: str
    skipped: bool
    record_count: int


def build_bot(
    bot: BotConfig,
    settings: Settings,
    get_embedder: Callable[[], Embedder],
    qdrant: QdrantClient,
    storage_dir: str | Path = "storage",
    force: bool = False,
) -> BuildResult:
    """Build one bot, or skip it if nothing changed since the last build.

    get_embedder is called only when a build is needed, so skipped runs do not
    load the model. force=True rebuilds even when nothing changed.
    """
    for f in bot.files:
        if not Path(f).is_file():
            raise LoadError(f"{f}: file not found")
    bot_dir = Path(storage_dir) / bot.name
    manifest_path = bot_dir / "manifest.json"
    fingerprint = {
        "file_hashes": {_key(f): _sha256(f) for f in bot.files},
        "embedding_model": EMBEDDING_MODEL,
        "obot_schema_version": SCHEMA_VERSION,
        "build_config_hash": _build_config_hash(bot, settings),
    }
    previous = _read_manifest(manifest_path)
    if (
        not force
        and previous is not None
        and "record_count" in previous
        and all(previous.get(k) == v for k, v in fingerprint.items())
        and qdrant.collection_exists(bot.name)
    ):
        logger.info("%s: unchanged, build skipped", bot.name)
        return BuildResult(bot.name, True, previous["record_count"])

    # Remove first so a failed build is retried instead of skipped next time.
    manifest_path.unlink(missing_ok=True)
    logger.info("%s: building", bot.name)

    records = [r for f in bot.files for r in load_file(f)]
    if settings.schema_analysis.use_llm:
        logger.warning(
            "%s: schema_analysis.use_llm is not available yet; using rule-based roles",
            bot.name,
        )
    schema = analyze_schema(records)
    if schema.id_field == FALLBACK_ID_FIELD:
        if any(f.name == FALLBACK_ID_FIELD for f in schema.fields):
            # The record index would silently replace the field's own values.
            raise LoadError(
                f"{bot.name}: field '{FALLBACK_ID_FIELD}' is reserved but does not "
                "qualify as the id; rename it in the data"
            )
        ids: list[Any] = list(range(len(records)))
        fields = [FALLBACK_ID_FIELD] + [f.name for f in schema.fields]
    else:
        ids = [r[schema.id_field] for r in records]
        fields = [f.name for f in schema.fields]
    schema.columns = store.column_names(fields)

    embeddings = get_embedder().encode([chunk_text(r) for r in records])
    store.write_points(qdrant, bot.name, ids, records, embeddings)
    store.write_records(bot_dir / "records.db", schema, ids, records)
    save_schema(schema, bot_dir / "schema.json")

    manifest = {
        **fingerprint,
        "record_count": len(records),
        "built_at": datetime.now().isoformat(timespec="seconds"),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.info("%s: built %d records", bot.name, len(records))
    return BuildResult(bot.name, False, len(records))


class Locked:
    """Proxy that runs every method call of the wrapped object under one lock."""

    def __init__(self, obj: Any):
        self._obj = obj
        self._lock = threading.Lock()

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._obj, name)
        if not callable(attr):
            return attr

        def call(*args: Any, **kwargs: Any) -> Any:
            with self._lock:
                return attr(*args, **kwargs)

        return call


class BuildInProgressError(Exception):
    """The bot is already queued or building."""


class BuildManager:
    """Builds bots one at a time on a background thread and tracks their status."""

    def __init__(
        self,
        bots: list[BotConfig],
        settings: Settings,
        get_embedder: Callable[[], Embedder],
        qdrant: QdrantClient,
        storage_dir: str | Path = "storage",
    ):
        self.bots = {b.name: b for b in bots}
        self.settings = settings
        self.get_embedder = get_embedder
        self.qdrant = qdrant
        self.storage_dir = Path(storage_dir)
        self.status: dict[str, str] = {}
        self.errors: dict[str, str] = {}
        # Kept in memory: a rebuild removes manifest.json while it runs.
        self.record_counts: dict[str, int] = {}
        self._lock = threading.Lock()
        self._queue: queue.Queue[tuple[str, bool] | None] = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        with self._lock:
            for name in self.bots:
                self.status[name] = "building"
                self._queue.put((name, False))
        self._thread.start()

    def rebuild(self, name: str) -> None:
        with self._lock:
            if self.status.get(name) == "building":
                raise BuildInProgressError(f"bot '{name}' is already building")
            self.status[name] = "building"
            self.errors.pop(name, None)
            self._queue.put((name, True))

    def stop(self) -> None:
        # Drop queued builds so shutdown waits only for the one in progress.
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._queue.put(None)
        self._thread.join()

    def _run(self) -> None:
        while (item := self._queue.get()) is not None:
            name, force = item
            try:
                result = build_bot(
                    self.bots[name],
                    self.settings,
                    self.get_embedder,
                    self.qdrant,
                    self.storage_dir,
                    force=force,
                )
            except Exception as e:
                logger.exception("%s: build failed", name)
                with self._lock:
                    self.status[name] = "failed"
                    self.errors[name] = str(e)
            else:
                with self._lock:
                    self.status[name] = "ready"
                    self.record_counts[name] = result.record_count


def _key(path: Path) -> str:
    return Path(path).as_posix()


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _build_config_hash(bot: BotConfig, settings: Settings) -> str:
    config = {
        "schema_analysis": settings.schema_analysis.model_dump(),
        "files": [_key(f) for f in bot.files],
    }
    return hashlib.sha256(
        json.dumps(config, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _read_manifest(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("%s: unreadable manifest, rebuilding", path)
        return None
