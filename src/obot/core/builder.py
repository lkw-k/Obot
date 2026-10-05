"""Build pipeline and change detection (spec §5.2)."""

import hashlib
import json
import logging
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
) -> BuildResult:
    """Build one bot, or skip it if nothing changed since the last build.

    get_embedder is called only when a build is needed, so skipped runs do not
    load the model.
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
        previous is not None
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
