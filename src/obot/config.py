"""Settings from config.yaml and .env; bot discovery and name normalization."""

import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

SUPPORTED_EXTENSIONS = (".json", ".jsonl", ".csv")

_PROVIDER_KEYS = {
    "anthropic": ("anthropic_api_key", "ANTHROPIC_API_KEY"),
    "openai": ("openai_api_key", "OPENAI_API_KEY"),
}


class ConfigError(Exception):
    """Invalid or incomplete configuration; the app must not start."""


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LLMConfig(_Section):
    provider: Literal["ollama", "anthropic", "openai"] = "ollama"
    model: str = "qwen2.5:7b"
    base_url: str | None = None
    temperature: float = 0.2


class SchemaAnalysisConfig(_Section):
    use_llm: bool = False


class RetrievalConfig(_Section):
    top_k: int = Field(default=5, ge=1)


class RoutingConfig(_Section):
    enabled: bool = True


class ServerConfig(_Section):
    host: str = "127.0.0.1"
    port: int = 8000


class BotConfig(_Section):
    name: str
    files: list[Path]


class Secrets(BaseSettings):
    """API keys from .env or the environment. Empty values count as missing."""

    model_config = SettingsConfigDict(
        env_file=".env", env_ignore_empty=True, extra="ignore"
    )

    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    obot_api_key: str | None = None


class Settings(_Section):
    llm: LLMConfig = LLMConfig()
    schema_analysis: SchemaAnalysisConfig = SchemaAnalysisConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    routing: RoutingConfig = RoutingConfig()
    server: ServerConfig = ServerConfig()
    bots: list[BotConfig] = []
    secrets: Secrets = Field(default_factory=Secrets, exclude=True)


def load_settings(
    config_path: str | Path = "config.yaml", env_file: str | Path | None = ".env"
) -> Settings:
    """Load settings; defaults when config.yaml is missing. Raises ConfigError."""
    path = Path(config_path)
    raw: dict = {}
    if path.is_file():
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as e:
            raise ConfigError(f"{path}: invalid YAML: {e}") from e
        if not isinstance(raw, dict):
            raise ConfigError(f"{path}: top level must be a mapping")

    try:
        settings = Settings.model_validate(
            {**raw, "secrets": Secrets(_env_file=env_file)}
        )
    except ValidationError as e:
        raise ConfigError(f"{path}: {e}") from e

    _check_provider_key(settings)
    return settings


def _check_provider_key(settings: Settings) -> None:
    provider = settings.llm.provider
    if provider not in _PROVIDER_KEYS:
        return
    attr, env_name = _PROVIDER_KEYS[provider]
    if not getattr(settings.secrets, attr):
        raise ConfigError(
            f"llm.provider is '{provider}' but {env_name} is not set in .env"
        )


def normalize_bot_name(name: str) -> str:
    """Lowercase; characters outside [a-z0-9_-] become '_'."""
    normalized = re.sub(r"[^a-z0-9_-]", "_", name.lower())
    if not normalized:
        raise ConfigError("bot name must not be empty")
    return normalized


def _discover_files(data_dir: Path) -> list[Path]:
    if not data_dir.is_dir():
        return []
    return sorted(
        p
        for p in data_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def resolve_bots(settings: Settings, data_dir: str | Path = "data") -> list[BotConfig]:
    """Bots from config, or one per supported file in data_dir.

    Raises ConfigError when two bots share a name after normalization.
    """
    if settings.bots:
        bots = [
            BotConfig(name=normalize_bot_name(b.name), files=b.files)
            for b in settings.bots
        ]
    else:
        bots = [
            BotConfig(name=normalize_bot_name(p.stem), files=[p])
            for p in _discover_files(Path(data_dir))
        ]

    sources: dict[str, list[Path]] = {}
    for bot in bots:
        if bot.name in sources:
            clashing = ", ".join(str(f) for f in sources[bot.name] + bot.files)
            raise ConfigError(
                f"bot name '{bot.name}' is used more than once ({clashing}); "
                "set names explicitly under 'bots' in config.yaml"
            )
        sources[bot.name] = bot.files
    return bots
