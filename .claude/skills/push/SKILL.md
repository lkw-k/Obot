---
name: push
description: Verify and push the current branch, following spec.md §11.3. Use ONLY when the user explicitly asks to push — never push on your own initiative.
---

# Push (spec §11.3)

Push only if every step passes. On any failure: stop, report which step failed and why, fix it, and re-run. Never bypass verification (no `--force`, no `--no-verify`).

## Automated checks (steps 1–5)

If `scripts/verify.sh` exists, run `bash scripts/verify.sh`.
Otherwise run the steps directly:

1. `git branch --show-current` — must not be `main`.
2. `git status --porcelain` — must be empty.
3. `uv run ruff check . --quiet` and `uv run ruff format --check . --quiet` — no errors.
4. `uv run pytest -q` — all tests pass.
5. `git fetch origin` then `git diff --name-only origin/main...HEAD` — no `.env`, `storage/`, or `data/` files other than `data/.gitkeep`.

## Manual checks

6. `git log origin/main..HEAD --oneline` — commits follow §11.2 and match the current phase (§10).
7. `git diff --stat origin/main...HEAD` first; open full diffs only for files that need it. Confirm no secrets or debug code.

## Push

```bash
git push -u origin <current-branch>
```
