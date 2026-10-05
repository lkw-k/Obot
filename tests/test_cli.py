import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from obot import cli
from obot.core.store import open_qdrant

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
runner = CliRunner()


@pytest.fixture
def project(tmp_path, monkeypatch, fake_embedder):
    monkeypatch.chdir(tmp_path)
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OBOT_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(cli, "get_embedder", lambda: fake_embedder)
    (tmp_path / "data").mkdir()
    shutil.copy(EXAMPLES / "orders.csv", tmp_path / "data" / "orders.csv")
    shutil.copy(EXAMPLES / "reviews.jsonl", tmp_path / "data" / "reviews.jsonl")
    return tmp_path


def test_build_then_skip(project):
    first = runner.invoke(cli.app, ["build"])
    assert first.exit_code == 0, first.output
    assert "orders: built, 28 records" in first.output
    assert "reviews: built, 30 records" in first.output
    second = runner.invoke(cli.app, ["build"])
    assert "orders: skipped (unchanged), 28 records" in second.output


def test_build_single_bot(project):
    result = runner.invoke(cli.app, ["build", "--bot", "orders"])
    assert result.exit_code == 0
    assert "orders: built" in result.output
    assert "reviews" not in result.output


def test_unknown_bot_fails(project):
    result = runner.invoke(cli.app, ["build", "--bot", "nope"])
    assert result.exit_code == 1
    assert "unknown bot 'nope'" in result.output


def test_locked_storage_fails_with_guidance(project):
    holder = open_qdrant(project / "storage" / "qdrant")
    try:
        result = runner.invoke(cli.app, ["build"])
    finally:
        holder.close()
    assert result.exit_code == 1
    assert "obot serve" in result.output


def test_earlier_results_print_before_a_later_failure(project):
    (project / "data" / "zzz.json").write_text("not json", encoding="utf-8")
    result = runner.invoke(cli.app, ["build"])
    assert result.exit_code == 1
    assert "orders: built, 28 records" in result.output
    assert "error: " in result.output and "zzz.json" in result.output


def test_missing_configured_file_fails_without_traceback(project):
    (project / "config.yaml").write_text(
        "bots:\n  - name: typo\n    files: [data/missing.csv]\n", encoding="utf-8"
    )
    result = runner.invoke(cli.app, ["build"])
    assert result.exit_code == 1
    assert "file not found" in result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)
