---
name: lint-format-open-darts
description: Lint and format open-DARTS code using Ruff and pre-commit, matching CI behavior. Use when preparing commits, fixing code quality issues, or reproducing lint failures.
---

# Lint and Format

Use this skill to apply the same checks used in CI.

## Steps

1. Run pre-commit for changed files or the full tracked file set.
2. Apply safe Ruff fixes and formatting.
3. Re-run checks until clean.

## Primary commands

- Single file: `pre-commit run -v --files /absolute/path/to/file.py --show-diff-on-failure`
- Project lint pass: `pre-commit run --files $FILES --show-diff-on-failure --color always`

## Reference

Read `references/linting-and-formatting.md` for rules, hook configuration, and CI parity.
