---
name: pr
description: Open a pull request against main and post a code review on it, following spec.md §11.4. Use ONLY when the user explicitly asks to open a PR.
---

# Pull Request + Code Review (spec §11.4)

## 1. Preconditions

- The branch must already be pushed (`git status -sb` shows it tracking `origin/<branch>` with nothing ahead). If not, stop and tell the user to run the `push` flow first.
- `gh auth status` must succeed.

## 2. Create the PR

Map the branch to its phase in spec §10 and copy that phase's completion criteria.

```bash
gh pr create --base main --title "<type>(<scope>): <summary>" --body "$(cat <<'EOF'
## Phase
spec Section 10 — <N. phase name>

## Changes
- <one line per logical change>

## Completion Criteria
- [ ] <completion criteria of this phase from Section 10>
- [ ] ruff and pytest pass

## Notes / Reasons for changes outside scope
- <or "None">
EOF
)"
```

## 3. Review

Delegate to the `code-reviewer` agent: pass the branch name, phase, and PR number.
It returns a review in the §11.4 comment format.

## 4. Post the review

```bash
gh pr comment <number> --body "<review from code-reviewer>"
```
(GitHub does not allow approving your own PR, so use a comment.)

## 5. Report

Give the user the PR URL and the verdict. If there are issues, list them and fix them before any merge.
Do not merge unless the user asks (§11.5: merge commit, delete branch after).
