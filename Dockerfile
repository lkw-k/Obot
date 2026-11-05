FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# Dependencies first so code changes do not reinstall torch.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable


FROM python:3.11-slim-bookworm

ENV PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/app/.cache/huggingface \
    PYTHONUNBUFFERED=1
WORKDIR /app

RUN useradd --create-home --uid 1000 obot \
    && mkdir -p data storage .cache/huggingface \
    && chown -R obot:obot /app

COPY --from=builder --chown=obot:obot /app/.venv /app/.venv
COPY --chown=obot:obot docker/config.yaml ./config.yaml

USER obot
EXPOSE 8000
VOLUME ["/app/storage", "/app/.cache/huggingface"]
CMD ["obot", "serve"]
