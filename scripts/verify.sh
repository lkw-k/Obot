#!/usr/bin/env bash
# Pre-push verification (spec.md section 11.3, steps 1-5).
# Prints PASS/FAIL per step; full output only on failure.
set -u

cd "$(dirname "$0")/.." || exit 1

failed=0

run_step() {
    local label="$1"
    shift
    local out
    if out="$("$@" 2>&1)"; then
        echo "PASS  $label"
    else
        echo "FAIL  $label"
        echo "$out" | sed 's/^/      /'
        failed=1
    fi
}

check_branch() {
    local branch
    branch="$(git branch --show-current)"
    if [ "$branch" = "main" ]; then
        echo "on main; push from a feature branch"
        return 1
    fi
}

check_clean() {
    local status
    status="$(git status --porcelain)"
    if [ -n "$status" ]; then
        echo "$status"
        return 1
    fi
}

check_lint() {
    uv run ruff check . --quiet && uv run ruff format --check . --quiet
}

check_tests() {
    uv run pytest -q
}

check_forbidden_files() {
    git fetch origin --quiet || return 1
    local bad
    bad="$(git diff --name-only origin/main...HEAD \
        | grep -E '^(\.env$|storage/|data/)' \
        | grep -vx 'data/\.gitkeep')"
    if [ -n "$bad" ]; then
        echo "forbidden files in diff:"
        echo "$bad"
        return 1
    fi
}

run_step "1. branch is not main" check_branch
run_step "2. no uncommitted changes" check_clean
run_step "3. ruff lint + format" check_lint
run_step "4. pytest" check_tests
run_step "5. no .env / data / storage files" check_forbidden_files

if [ "$failed" -ne 0 ]; then
    echo "verify: FAILED"
    exit 1
fi
echo "verify: OK"
