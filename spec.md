# OBot Development Spec (spec.md)

> This document is the source of truth for implementing OBot. If the code and this document disagree, update this document first, then align the code.

---

## 1. Overview

Drop JSON · JSONL · CSV files into the `data/` folder and start the server.
A RAG chatbot is built automatically for each file and can be called over HTTP.

**Goals**
- Minimal configuration: works once an LLM provider and API key are set
- Record-level chunking + hybrid search
- Aggregation questions answered via SQL
- Runs with `uv run`, no external database installation

**Out of scope (MVP)**
- Web UI
- File upload API at runtime
- Multi-user / account management
- Server-side conversation history (stateless)

---

## 2. Tech Stack

| Area | Choice |
|---|---|
| Language / packaging | Python 3.11+, uv |
| API server | FastAPI, Uvicorn |
| Embedding | `BAAI/bge-m3` (fixed), `FlagEmbedding.BGEM3FlagModel` |
| Vector store | Qdrant local mode (`QdrantClient(path=...)`) |
| Records / aggregation | SQLite (standard library `sqlite3`) |
| LLM | Official SDKs: `ollama`, `anthropic`, `openai` |
| Config / validation | Pydantic v2, pydantic-settings, PyYAML |
| CLI | Typer |
| Test / lint | pytest, ruff |

Do not use RAG frameworks such as LangChain.

Dev dependencies: `uv add --dev pytest ruff`. The PR workflow requires the GitHub CLI (`gh`) installed and authenticated (`gh auth login`).

---

## 3. Project Structure

```
obot/
├── src/obot/
│   ├── core/
│   │   ├── loader.py       # file → list of records
│   │   ├── schema.py       # field role analysis
│   │   ├── chunker.py      # record → text for embedding
│   │   ├── embedder.py     # bge-m3 dense + sparse
│   │   ├── store.py        # Qdrant + SQLite read/write
│   │   ├── retriever.py    # hybrid search, RRF
│   │   ├── router.py       # search / aggregate classification
│   │   ├── sql_agent.py    # Text-to-SQL, safe execution
│   │   ├── generator.py    # prompt building, answer generation
│   │   ├── builder.py      # build pipeline, change detection, bot status, background build thread
│   │   └── bot.py          # Bot object: ask(question, history)
│   ├── llm/
│   │   ├── base.py         # LLMClient interface, LLMError, provider factory
│   │   ├── ollama.py
│   │   ├── anthropic.py
│   │   └── openai.py
│   ├── api/
│   │   ├── app.py          # FastAPI app, lifespan, error handlers ({"error": {...}})
│   │   ├── routes.py
│   │   ├── schemas.py      # request/response models
│   │   └── auth.py
│   ├── config.py           # settings, bot discovery, name normalization / collision check
│   └── cli.py
├── data/                   # user data (git-ignored, only .gitkeep committed)
├── storage/                # build output (git-ignored)
├── examples/               # sample data
├── tests/                  # conftest.py: fake embedder / LLM
├── .github/workflows/test.yml
├── scripts/verify.sh       # pre-push verification (Section 11.3)
├── config.example.yaml
├── .env.example
└── pyproject.toml
```

**Principles**
- `core/` must not import FastAPI. It must work as plain Python functions without the API.
- Load the embedding model only once per process (singleton).
- Builds (background thread) and requests run concurrently, so guard the embedder and the Qdrant client each with a `threading.Lock`. Build bots one at a time, sequentially.

---

## 4. Configuration

### config.yaml

```yaml
llm:
  provider: ollama          # ollama | anthropic | openai
  model: qwen2.5:7b
  base_url: null            # custom Ollama address (default http://localhost:11434)
  temperature: 0.2

schema_analysis:
  use_llm: false            # if true, refine rule-based results with the LLM

retrieval:
  top_k: 5

routing:
  enabled: true             # if false, every question is handled as search

server:
  host: 127.0.0.1
  port: 8000

bots: []                    # if empty, auto-discover files in data/
# bots:
#   - name: products
#     files: [data/products.json]
```

### .env

```
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
OBOT_API_KEY=               # enables API authentication when set
```

### Rules
- If `config.yaml` is missing, run with defaults.
- If the API key for the selected provider is missing, exit at startup with a clear error message.
- If `bots` is empty, create one bot per supported file in `data/`, named after the file (without extension).
- Normalize bot names to `^[a-z0-9_-]+$` (lowercase, other characters → `_`).
- If two names collide after normalization (`products.json`, `products.csv`), exit at startup and tell the user to set names explicitly in `bots`.

---

## 5. Build Pipeline

```
load → change detection → schema analysis → chunking → embedding → storage
```

### 5.1 Loading (`loader.py`)

| Format | Handling |
|---|---|
| JSON | If the top level is an array of objects, use it. If the top level is an object with exactly one array field, use that array. Otherwise, error |
| JSONL | One line = one record; skip blank lines |
| CSV | Try encoding `utf-8-sig`, then `cp949`. Header required. All values are read as strings, so try converting int → float; empty strings become null |

- Normalize records to `dict[str, Any]`.
- Flatten nested objects with dot notation (`{"a": {"b": 1}}` → `{"a.b": 1}`).
- Store list values as JSON strings.
- Skip records that fail to parse and log a warning. If all records fail, the build fails.

### 5.2 Change Detection (`builder.py`)

Store the following in `storage/{bot}/manifest.json`:

```json
{
  "file_hashes": {"data/products.json": "sha256..."},
  "embedding_model": "BAAI/bge-m3",
  "obot_schema_version": 1,
  "build_config_hash": "sha256...",
  "record_count": 120,
  "built_at": "2026-09-23T12:00:00"
}
```

Skip the build if file hashes, embedding model, schema version, and the build config hash (`schema_analysis`, the bot's `files`) are all unchanged.

### 5.3 Schema Analysis (`schema.py`)

Assign exactly one role per field. Apply the first matching rule from the top.

| Role | Condition |
|---|---|
| `id` | All values unique and non-null, and the field name contains `id` or it is the first field |
| `number` | Values are int / float (excluding bool) |
| `date` | ≥ 90% of sampled values parse as ISO dates |
| `category` | String, unique values ≤ max(20, 5% of record count), average length ≤ 30 |
| `text` | Any other string |

- If there is no `id` field, use the record index as `_obot_id`.
- Save the result to `storage/{bot}/schema.json`.
- If `use_llm: true`, send field names, roles, and 3 sample records to the LLM to refine roles. If the response is invalid JSON, keep the rule-based result.

### 5.4 Chunking (`chunker.py`)

- One record = one chunk.
- Serialize every field as `field: value` lines for the embedding text. Exclude null and empty strings.

### 5.5 Embedding (`embedder.py`)

```python
model.encode(texts, return_dense=True, return_sparse=True, max_length=2048)
# dense_vecs: (N, 1024)
# lexical_weights: [{token_id(str): weight}, ...]
```

- Process in batches and log progress.
- Texts over `max_length` are truncated by the model; log a warning for truncated records.
- Convert sparse `token_id` keys to `int` before inserting into Qdrant.
- Define an `Embedder` interface so tests can inject a fake embedder.

### 5.6 Storage (`store.py`)

| Store | Path | Content |
|---|---|---|
| Qdrant | `storage/qdrant/` (collection = bot name) | named vector `dense` (1024, cosine) + sparse vector `sparse` |
| SQLite | `storage/{bot}/records.db` | table `records`, flattened fields = columns |

- Qdrant point id = `uuid5(NAMESPACE_URL, f"{bot}:{record_id}")`; keep the original id in the payload.
- Store the full original record in the payload.
- Normalize SQLite column names to letters, digits, and `_`; save the mapping to original field names in `schema.json`.
- SQLite column types follow roles: `number` → REAL (INTEGER if all integers), everything else → TEXT. Values that fail conversion become null.
- On rebuild, set the bot status to `building` first, then drop and recreate the collection and table.

**Qdrant local mode constraints (verified)**
- Only one process can open a storage folder. A second client raises `RuntimeError`.
  → `obot build` and `obot ask` cannot run while the server is running. Use `POST /bots/{name}/rebuild` and `POST /bots/{name}/chat` instead. If the folder is locked, the CLI prints this guidance and exits.
- qdrant-client warns about performance above 20,000 points per collection. State in the README that the MVP supports up to 20,000 records per bot.

---

## 6. Query Pipeline

```
question → routing → (search) hybrid search → answer generation
                   → (aggregate) Text-to-SQL → answer generation
```

### 6.1 Routing (`router.py`)

- Give the LLM the question and the field list; accept only JSON `{"route": "search" | "aggregate"}`.
- Aggregate means: counts, sums, averages, max/min, sorting, ranking, full lists matching conditions.
- On parse failure or `routing.enabled: false`, use `search`.

### 6.2 Hybrid Search (`retriever.py`)

1. Dense search, top `top_k × 4`
2. Sparse search, top `top_k × 4`
3. Combine with RRF: `score = Σ 1 / (60 + rank)`
4. Return the top `top_k`

Implement RRF manually (do not rely on Qdrant's built-in fusion).

### 6.3 Text-to-SQL (`sql_agent.py`)

- LLM input: table name, column names / types / roles, 3 sample rows, the question.
- LLM output: JSON `{"sql": "SELECT ..."}`.

**Safety rules (required)**
- Open SQLite read-only: `sqlite3.connect("file:...?mode=ro", uri=True)`
- The query must start with `SELECT` or `WITH`. Strip a trailing `;`.
- Multiple statements are rejected by `sqlite3` itself (`ProgrammingError`).
- Use `conn.set_authorizer()` to allow only `SQLITE_SELECT`, `SQLITE_READ`, `SQLITE_FUNCTION`.
  (Do not use keyword string matching: column names like `created_at` falsely match `CREATE`.)
- Limit results to 100 rows: `SELECT * FROM (<sql>) LIMIT 100`

**Failure handling**
- On execution error, retry once with the error message included.
- If the retry fails, fall back to the search route.

### 6.4 Answer Generation (`generator.py`)

System prompt rules:
- Answer only from the provided context.
- If the context lacks the answer, say you don't know.
- Answer in the same language as the question.
- Include the record id when referring to a record.

Input: system prompt + conversation history (last 10 messages) + context + question

### 6.5 LLM Interface (`llm/base.py`)

```python
class LLMClient(Protocol):
    def chat(self, system: str, messages: list[dict], json_mode: bool = False) -> str: ...
```

- The three providers implement only this interface.
- With `json_mode=True`, instruct the model to output JSON only, strip Markdown code fences from the response, then parse.
- Wrap network errors in `LLMError`.

---

## 7. API

### Common
- All responses are JSON.
- If `OBOT_API_KEY` is set, the `X-API-Key` header is required. Mismatch → 401 (except `/health`).
- Bot builds run in the background at startup. The server starts immediately without waiting for builds.

### Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | `{"status": "ok"}` |
| GET | `/bots` | Bot list and status |
| GET | `/bots/{name}` | Status, record count, schema |
| POST | `/bots/{name}/chat` | Question → answer |
| POST | `/bots/{name}/rebuild` | Force rebuild (returns 202) |

Bot status: `building` | `ready` | `failed`

### POST `/bots/{name}/chat`

Request
```json
{
  "message": "Which wireless earphones cost under 100,000 won?",
  "history": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
  "top_k": 5
}
```
- `history` and `top_k` are optional.

Response
```json
{
  "answer": "...",
  "route": "search",
  "sources": [{"id": "P-1023", "score": 0.031, "record": {"...": "..."}}],
  "sql": null
}
```
- When `route: "aggregate"`, `sources` is an empty array and `sql` is `{"query": "...", "rows": [...]}`.

### Errors

| Code | Case |
|---|---|
| 401 | API key mismatch |
| 404 | Unknown bot |
| 409 | Bot is `building` or `failed` (chat), or already `building` (rebuild) |
| 422 | Invalid request |
| 502 | LLM call failed |

Error body: `{"error": {"code": "BOT_NOT_READY", "message": "..."}}`

---

## 8. CLI

```bash
obot serve                    # run the server (includes auto-build)
obot build [--bot NAME]       # build only
obot ask NAME "question"      # ask from the terminal without the server
```

---

## 9. Testing

- Replace the embedding model and LLM with fakes so tests run without network access or model downloads.
- Required test targets:
  - loader: each JSON shape, JSONL, CSV encodings, CSV numeric conversion, nested flattening
  - schema: each role classification rule
  - chunker: serialization output, null exclusion
  - retriever: RRF calculation
  - sql_agent: write statements rejected, `created_at` column queried normally, trailing `;` handled
  - api: status codes (401, 404, 409)

---

## 10. Development Phases

Each phase must meet its completion criteria before moving on. One phase = one branch = one PR (see Section 11).

- [ ] **1. Project setup** `feat/setup`: uv project, folder structure, config loading, GitHub Actions (ruff + pytest), `scripts/verify.sh`
  - Done when: settings object is created from `config.example.yaml`, missing key raises an error, CI runs on PRs
- [ ] **2. Loading + schema analysis** `feat/loader`
  - Done when: `schema.json` is generated from the sample files in `examples/`, tests pass
- [ ] **3. Chunking + embedding + storage** `feat/embedding`
  - Done when: after building the samples, Qdrant point count = SQLite row count = record count; rerunning skips the build
- [ ] **4. Search + answer (dense only)** `feat/retrieval`
  - Done when: `obot ask` returns an answer with sources
- [ ] **5. API server** `feat/api`
  - Done when: all endpoints work in `/docs`; 409 is returned during a build
- [ ] **6. Hybrid search** `feat/hybrid`
  - Done when: sparse + RRF applied; results improve for exact product name / ID queries
- [ ] **7. Routing + Text-to-SQL** `feat/sql`
  - Done when: "how many" and "most expensive" questions are routed to aggregate; safety tests pass
- [ ] **8. Deployment prep** `chore/docker`
  - Done when: Dockerfile, API key auth, sample data, README demo GIF

---

## 11. Development Workflow

**Develop each phase on its own branch and merge into main through a PR.**

### 11.1 Branches

- Always branch from the latest `main`.
- Use the branch names from Section 10. Bug fixes: `fix/<topic>`, docs: `docs/<topic>`.
- **Implement only the scope of the current phase on its branch.** Do not build features from later phases in advance.
- If earlier-phase code must change, keep it minimal and explain why in the PR description.

```bash
git switch main && git pull
git switch -c feat/loader
```

### 11.2 Commits: split by file / function

After making changes, **do not commit everything at once.** Split commits by logical unit.

- One commit = one purpose. Group files by what they do, not by when they were edited.
- Stage selectively with `git add <file>` (or `git add -p` for partial changes). Never use `git add .` for a mixed set of changes.
- Commit implementation, tests, and config/docs separately when they serve different purposes.
- Before committing, review `git status` and `git diff --staged` to confirm only the intended changes are included.

Example — changes to loader, schema, tests, and config in one session:
```
feat(loader): add JSON/JSONL/CSV loading and nested flattening
feat(loader): add CSV numeric conversion and encoding fallback
feat(schema): add rule-based field role classification
test(loader): add tests for JSON shapes and CSV encodings
test(schema): add tests for each role rule
chore(config): add schema_analysis section to config.example.yaml
```

Commit message format: `<type>(<scope>): <summary>`
- type: `feat`, `fix`, `test`, `refactor`, `docs`, `chore`
- scope: module name (`loader`, `schema`, `embedder`, `api`, ...)
- summary: imperative, lowercase, no period

### 11.3 Push: only when asked, always verify first

- **Never push unless the user explicitly asks to push.** Committing locally is fine; pushing is not automatic.
- When the user asks to push, **run the verification steps below first.** Push only if every step passes.

**Pre-push verification (required)**

Run `bash scripts/verify.sh`. It performs steps 1–5 and prints only a short PASS/FAIL summary per step (full output only on failure).

1. `git branch --show-current` — the branch is not `main`.
2. `git status --porcelain` — no uncommitted changes.
3. `uv run ruff check . --quiet` and `uv run ruff format --check . --quiet` — no lint or format errors.
4. `uv run pytest -q` — all tests pass.
5. `git fetch origin` then `git diff --name-only origin/main...HEAD` — no `.env`, `data/`, or `storage/` files.

Then check manually:
6. `git log origin/main..HEAD --oneline` — the commits follow 11.2 and match the current phase.
7. `git diff --stat origin/main...HEAD` first; open the full diff only for files that need it. Confirm no secrets or debug code.

If any step fails: stop, report which step failed and why to the user, and fix it before pushing. Do not bypass verification.

```bash
git push -u origin feat/loader
```

### 11.4 Pull Requests: always include a code review

Open a PR **only when the user asks** (the branch must already be pushed via 11.3). When opening a PR, **always perform a code review** of the full diff (`git diff origin/main...HEAD`) and post the result on the PR.

**Steps**
1. Create the PR against `main` with the template below: `gh pr create --title "..." --body "..."`
2. Review the full diff against the checklist below.
3. Post the review as a PR comment: `gh pr comment <number> --body "..."`
   (GitHub does not allow approving your own PR, so use a comment.)
4. If the review finds problems, report them to the user and fix them before merging.

**Code review checklist**
- Spec compliance: matches this document and the completion criteria of the phase
- Scope: no features from other phases
- Correctness: edge cases, error handling, empty inputs
- Safety: SQL safety rules (6.3), no secrets, read-only access where required
- Structure: `core/` does not import FastAPI; one responsibility per module
- Tests: new logic is covered; tests use fakes for the model and LLM
- Readability: clear names, no dead or debug code, type hints present

**Review comment format**
```markdown
## Code Review
**Verdict:** ✅ Ready to merge / ⚠️ Changes needed

### Issues
- [severity: high/medium/low] file:line — description and suggested fix

### Suggestions (optional)
-

### Checklist
- [x] Spec compliance
- [x] Scope
- ...
```

**PR description template**
```markdown
## Phase
spec Section 10 — 2. Loading + schema analysis

## Changes
-

## Completion Criteria
- [ ] (paste the completion criteria of this phase from Section 10)
- [ ] ruff and pytest pass

## Notes / Reasons for changes outside scope
-
```

### 11.5 Merge

- **Merge only when the user asks**, after CI passes and the code review has no unresolved high-severity issues.
- Use **"Create a merge commit"** so the per-file commits from 11.2 are preserved in `main` history.
- Delete the branch after merging.

### 11.6 Example Instruction (Claude Code)

> Current branch is `feat/loader`. Implement only phase 2 of spec.md Section 10, satisfying its completion criteria and the tests in Section 9. Commit changes split by file/function following 11.2. Do not push until I ask.
