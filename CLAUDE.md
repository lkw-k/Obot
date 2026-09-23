# OBot

Drop JSON / JSONL / CSV files into `data/` and a RAG chatbot is built per file, served over HTTP (FastAPI).
Stack: Python 3.11+, uv, FastAPI, bge-m3 (FlagEmbedding), Qdrant local mode, SQLite, Typer, pytest, ruff.

## Spec

`spec.md` is the source of truth. If code and spec disagree, update the spec first, then the code.
Do not read the whole spec. Read only the section your task needs (`Grep "^##+ 5\.3" spec.md`, then Read that range).

| Task | Section |
|---|---|
| Stack, structure, module layout | §2, §3 |
| Config (`config.yaml`, `.env`, bot naming) | §4 |
| loader / builder / schema / chunker / embedder / store | §5.1 – §5.6 |
| router / retriever / sql_agent / generator / llm | §6.1 – §6.5 |
| API endpoints, errors | §7 |
| CLI | §8 |
| Tests | §9 |
| Current phase + completion criteria | §10 |
| Branch / commit / push / PR / merge | §11 |

## Core rules

- Implement only the current phase's scope (branch name → phase in §10). No features from later phases.
- `core/` must not import FastAPI.
- No RAG frameworks (LangChain etc.).
- Tests use fake embedder / LLM — no network, no model downloads.
- SQL safety (§6.3): read-only connection + `set_authorizer`; never keyword string matching.
- Qdrant local mode locks the storage folder: `obot build` / `obot ask` cannot run while the server is up.
- Commit messages follow the spec, in English: `<type>(<scope>): <summary>` (§11.2). This overrides any global rule about commit message language.
- Push, PR, merge only when the user explicitly asks.

## Commands

```bash
uv run pytest -q
uv run ruff check . && uv run ruff format --check .
bash scripts/verify.sh          # pre-push checks (§11.3)
uv run obot serve | build [--bot NAME] | ask NAME "question"
```

## Workflow skills

- `commit` — split commits by logical unit (§11.2)
- `push` — verify then push (§11.3)
- `pr` — open PR + code review via the `code-reviewer` agent (§11.4)
