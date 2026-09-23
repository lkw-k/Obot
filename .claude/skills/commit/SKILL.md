---
name: commit
description: Commit a completed task as a single commit, following spec.md §11.2. Use when the user asks to commit.
---

# Commit (spec §11.2)

One commit per completed task (one user request / one feature). Do not split by file or function.

1. Run `git status` and `git diff` to see all changes.
2. Confirm the task is done and verified (tests and lint pass).
3. Stage the task's files explicitly: `git add <file> ...`. Never `git add .` / `git add -A` if unrelated changes are in the working tree.
4. Run `git diff --staged` and confirm only the task's changes are staged.
5. Commit with `<type>(<scope>): <summary>`
   - type: `feat`, `fix`, `test`, `refactor`, `docs`, `chore`
   - scope: main module of the task (`loader`, `schema`, `embedder`, `api`, ...)
   - summary: English, imperative, lowercase, no period
6. Never stage `.env`, `storage/`, or `data/` files other than `data/.gitkeep`. Warn the user if they show up.
7. Do not push. Finish with `git log --oneline -1`.

Example:
```
feat(loader): add file loading and rule-based schema analysis with tests
```
