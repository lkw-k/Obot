---
name: commit
description: Commit working-tree changes split by logical unit, following spec.md §11.2. Use when the user asks to commit.
---

# Commit (spec §11.2)

1. Run `git status` and `git diff` to see all changes.
2. Group changes by purpose, not by edit time. One commit = one purpose.
   Implementation, tests, and config/docs go in separate commits when they serve different purposes.
3. For each group:
   - Stage selectively: `git add <file>` (or `git add -p <file>` for partial changes). Never `git add .` / `git add -A` on mixed changes.
   - Run `git diff --staged` and confirm only the intended changes are staged.
   - Commit with `<type>(<scope>): <summary>`
     - type: `feat`, `fix`, `test`, `refactor`, `docs`, `chore`
     - scope: module name (`loader`, `schema`, `embedder`, `api`, ...)
     - summary: English, imperative, lowercase, no period
4. Never stage `.env`, `data/`, or `storage/` files. Warn the user if they show up.
5. Do not push. Finish with `git log --oneline` for the new commits.

Example:
```
feat(loader): add JSON/JSONL/CSV loading and nested flattening
test(loader): add tests for JSON shapes and CSV encodings
chore(config): add schema_analysis section to config.example.yaml
```
