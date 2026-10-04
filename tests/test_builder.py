import json
import shutil
from pathlib import Path

import pytest

from obot.config import BotConfig, Settings
from obot.core.builder import build_all, build_bot
from obot.core.loader import LoadError, load_file
from obot.core.schema import load_schema
from obot.core.store import count_points, count_rows, open_qdrant

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
SAMPLES = ["products.json", "reviews.jsonl", "orders.csv"]


@pytest.fixture
def storage(tmp_path):
    return tmp_path / "storage"


@pytest.fixture
def qdrant(storage):
    client = open_qdrant(storage / "qdrant")
    yield client
    client.close()


@pytest.fixture
def bots(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    result = []
    for name in SAMPLES:
        shutil.copy(EXAMPLES / name, data / name)
        result.append(BotConfig(name=Path(name).stem, files=[data / name]))
    return result


def _build_all(bots, qdrant, storage, embedder, settings=None):
    return build_all(bots, settings or Settings(), lambda: embedder, qdrant, storage)


def test_samples_counts_match(bots, qdrant, storage, fake_embedder):
    results = _build_all(bots, qdrant, storage, fake_embedder)
    for bot, result in zip(bots, results, strict=True):
        expected = len(load_file(bot.files[0]))
        assert not result.skipped
        assert result.record_count == expected
        assert count_points(qdrant, bot.name) == expected
        assert count_rows(storage / bot.name / "records.db") == expected


def test_writes_schema_and_manifest(bots, qdrant, storage, fake_embedder):
    _build_all(bots, qdrant, storage, fake_embedder)
    schema = load_schema(storage / "orders" / "schema.json")
    assert schema.id_field == "_obot_id"
    assert set(schema.columns) == {"_obot_id"} | {f.name for f in schema.fields}
    manifest_path = storage / "orders" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["embedding_model"] == "BAAI/bge-m3"
    assert manifest["record_count"] == schema.record_count
    assert list(manifest["file_hashes"]) == [bots[2].files[0].as_posix()]


def test_rerun_skips_without_loading_model(bots, qdrant, storage, fake_embedder):
    _build_all(bots, qdrant, storage, fake_embedder)

    def no_model():
        raise AssertionError("embedder must not be loaded when skipping")

    results = build_all(bots, Settings(), no_model, qdrant, storage)
    assert all(r.skipped for r in results)
    assert [r.record_count for r in results] == [24, 30, 28]


def test_changed_file_triggers_rebuild(bots, qdrant, storage, fake_embedder):
    _build_all(bots, qdrant, storage, fake_embedder)
    reviews = bots[1].files[0]
    lines = reviews.read_text(encoding="utf-8").strip().splitlines()
    reviews.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    results = _build_all(bots, qdrant, storage, fake_embedder)
    assert [r.skipped for r in results] == [True, False, True]
    assert count_points(qdrant, "reviews") == results[1].record_count == 29


def test_changed_schema_config_triggers_rebuild(bots, qdrant, storage, fake_embedder):
    _build_all(bots, qdrant, storage, fake_embedder)
    settings = Settings.model_validate({"schema_analysis": {"use_llm": True}})
    results = _build_all(bots, qdrant, storage, fake_embedder, settings)
    assert not any(r.skipped for r in results)


def test_missing_collection_triggers_rebuild(bots, qdrant, storage, fake_embedder):
    _build_all(bots, qdrant, storage, fake_embedder)
    qdrant.delete_collection("products")
    result = build_bot(bots[0], Settings(), lambda: fake_embedder, qdrant, storage)
    assert not result.skipped
    assert count_points(qdrant, "products") == 24


def test_failed_build_leaves_no_manifest(bots, qdrant, storage, fake_embedder):
    _build_all(bots, qdrant, storage, fake_embedder)
    bots[0].files[0].write_text("not json", encoding="utf-8")
    with pytest.raises(LoadError):
        build_bot(bots[0], Settings(), lambda: fake_embedder, qdrant, storage)
    assert not (storage / "products" / "manifest.json").exists()


def test_multiple_files_are_concatenated(tmp_path, qdrant, storage, fake_embedder):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    a.write_text('{"id": 1}\n{"id": 2}\n', encoding="utf-8")
    b.write_text('{"id": 3}\n', encoding="utf-8")
    bot = BotConfig(name="both", files=[a, b])
    result = build_bot(bot, Settings(), lambda: fake_embedder, qdrant, storage)
    assert result.record_count == 3
    assert count_points(qdrant, "both") == 3
