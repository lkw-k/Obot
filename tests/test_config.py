from pathlib import Path

import pytest

from obot.config import ConfigError, load_settings, normalize_bot_name, resolve_bots

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OBOT_API_KEY"):
        monkeypatch.delenv(key, raising=False)


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_example_config():
    settings = load_settings(ROOT / "config.example.yaml", env_file=None)
    assert settings.llm.provider == "ollama"
    assert settings.llm.model == "qwen2.5:7b"
    assert settings.retrieval.top_k == 5
    assert settings.routing.enabled is True
    assert settings.server.port == 8000
    assert settings.bots == []


def test_missing_config_uses_defaults(tmp_path):
    settings = load_settings(tmp_path / "nope.yaml", env_file=None)
    assert settings.llm.provider == "ollama"
    assert settings.server.host == "127.0.0.1"


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_missing_provider_key_raises(tmp_path, provider):
    cfg = _write(tmp_path / "config.yaml", f"llm:\n  provider: {provider}\n")
    with pytest.raises(ConfigError, match=f"{provider.upper()}_API_KEY"):
        load_settings(cfg, env_file=None)


def test_empty_key_in_env_file_counts_as_missing(tmp_path):
    cfg = _write(tmp_path / "config.yaml", "llm:\n  provider: anthropic\n")
    env = _write(tmp_path / ".env", "ANTHROPIC_API_KEY=\n")
    with pytest.raises(ConfigError):
        load_settings(cfg, env_file=env)


def test_provider_key_from_env_file(tmp_path):
    cfg = _write(tmp_path / "config.yaml", "llm:\n  provider: openai\n")
    env = _write(tmp_path / ".env", "OPENAI_API_KEY=sk-test\n")
    settings = load_settings(cfg, env_file=env)
    assert settings.secrets.openai_api_key == "sk-test"


def test_invalid_config_raises(tmp_path):
    cfg = _write(tmp_path / "config.yaml", "llm:\n  provider: gemini\n")
    with pytest.raises(ConfigError):
        load_settings(cfg, env_file=None)


def test_unknown_key_raises(tmp_path):
    cfg = _write(tmp_path / "config.yaml", "retrieval:\n  topk: 3\n")
    with pytest.raises(ConfigError):
        load_settings(cfg, env_file=None)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Products", "products"),
        ("my data.v2", "my_data_v2"),
        ("a-b_c", "a-b_c"),
        ("상품", "__"),
    ],
)
def test_normalize_bot_name(raw, expected):
    assert normalize_bot_name(raw) == expected


def test_discovers_supported_files(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    for name in ("Products.json", "logs.jsonl", "users.csv", "notes.txt", ".gitkeep"):
        (data / name).touch()
    bots = resolve_bots(load_settings(tmp_path / "none.yaml", env_file=None), data)
    assert sorted(b.name for b in bots) == ["logs", "products", "users"]


def test_discovered_name_collision_raises(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "products.json").touch()
    (data / "products.csv").touch()
    with pytest.raises(ConfigError, match="set names explicitly"):
        resolve_bots(load_settings(tmp_path / "none.yaml", env_file=None), data)


def test_explicit_bots_are_normalized(tmp_path):
    cfg = _write(
        tmp_path / "config.yaml",
        "bots:\n  - name: My Products\n    files: [data/products.json]\n",
    )
    bots = resolve_bots(load_settings(cfg, env_file=None), tmp_path / "data")
    assert [b.name for b in bots] == ["my_products"]
    assert bots[0].files == [Path("data/products.json")]
