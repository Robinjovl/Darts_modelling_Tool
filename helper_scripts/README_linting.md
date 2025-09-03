## Python Linting and Formatting with Pre-commit and Ruff

This document describes the updated linting and formatting setup for the DARTS project.

### Highlights
- Pre-commit hooks run automatically on `git commit` / `git push` to enforce formatting, linting, and basic validations.
- Ruff replaces black, isort, and flake8. Do not use `--unsafe-fixes` with Ruff.
- CI runs the same checks using `pre-commit`.
- No separate Python linting script is needed anymore.

---

## Purpose of `.pre-commit-config.yaml`
Defines the set of checks ("hooks") that run automatically on `git commit` and `git push`. Each hook references a repository and version, and may have per-hook options (e.g., file globs, arguments). This file is the single source of truth for which validations we run locally and in CI.

## Hooks configured in `.pre-commit-config.yaml`

Ruff hooks (from `astral-sh/ruff-pre-commit`):
- `ruff-check` (with `--fix`, `--show-fixes`): Lints Python and applies safe, non-breaking fixes. Avoid `--unsafe-fixes`.
- `ruff-format`: Formats Python code (Ruff formatter), analogous to Black but configured via Ruff.

General quality and hygiene (from `pre-commit/pre-commit-hooks`):
- `end-of-file-fixer`: Ensures files end with a single newline.
- `trailing-whitespace`: Removes stray trailing whitespace.
- `check-yaml`: Validates YAML syntax for selected files.
- `check-toml`: Validates TOML syntax (e.g., `pyproject.toml`).
- `mixed-line-ending`: Normalizes line endings; prevents mixed CRLF/LF.
- `detect-private-key`: Detects accidentally committed private keys.
- `check-added-large-files --maxkb=500`: Prevents committing very large files to the repo.
- `check-merge-conflict`: Detects unresolved merge conflict markers.

Notes:
- Hook environments are downloaded and cached automatically on first use by pre-commit (into `.cache/pre-commit`).
- The config schedules auto-updates quarterly (see `ci.autoupdate_schedule`). Maintainers can run `pre-commit autoupdate` to bump hook versions.

## Ruff configuration location
Ruff is configured in `pyproject.toml` under:
- `[tool.ruff]`, `[tool.ruff.lint]`, `[tool.ruff.format]`, `[tool.ruff.lint.mccabe]`

---

## Preparing your development environment
Choose one of the following:

Option A (recommended, includes project dev extras):
```bash
python -m pip install -e .[dev]
pre-commit install
```

Option B (manual, if you don't want all dev extras):
```bash
python -m pip install --upgrade ruff pre-commit
pre-commit install
```
---

## Local usage of pre-commit

Automatic, on commit or push:
```bash
git commit -m "..."   # hooks run automatically
git push -m "..."     # hooks run automatically
```

Manually, mirror the CI job's file selection (Python, YAML, TOML in selected paths):
```bash
# From repo root
FILES="$(
  {
    git ls-files -- 'darts' 'tests' 'helper_scripts' '.cicd' '.gitlab-ci.yml' '.pre-commit-config.yaml' 'pyproject.toml' 2>/dev/null || true
  } | grep -E '\\.(py|ya?ml|toml)$' || true
)"
pre-commit run --files $FILES --show-diff-on-failure --color always
```

Skip hooks for a single commit (use sparingly):
```bash
git commit --no-verify -m "commit message"
```

---

## Which files are checked

Local pre-commit (via hooks) uses the `files` patterns defined in `.pre-commit-config.yaml`:
- Ruff hooks (`ruff-check`, `ruff-format`): `^(darts|tests|helper_scripts|\.cicd)/.*\.py$`
- YAML check: `^((.cicd|helper_scripts)/.*\.ya?ml|\.gitlab-ci\.ya?ml)$`
- TOML check: `^[^/]+\.toml$`
- Other general hooks apply to all files unless restricted by the hook.

---

## CI Pipeline Integration
The pre-commit job runs in the `pre_commit` stage using `python:3.10`, installs `pre-commit`, selects target files (Python, YAML, TOML in `darts`, `tests`, `helper_scripts`, `.cicd`, plus `pyproject.toml` and `.gitlab-ci.yml`), and executes:

```bash
pre-commit run --files $FILES --show-diff-on-failure --color always
```

The job currently allows failure (`allow_failure: true`) to ease adoption; aim to fix issues proactively.


## Tips, recommendations, and gotchas
- Do not use `ruff --unsafe-fixes` (or `--unsafe-fixes` via pre-commit). Only safe fixes are enabled.
- To update hook versions: `pre-commit autoupdate` (we also auto-update quarterly).
- Hook environments are cached under `.cache/pre-commit`.
- If a new directory is added for Python code, update the `files` glob for Ruff in `.pre-commit-config.yaml`.
- In CI, failures in the pre-commit job currently do not fail the pipeline; treat them as warnings to be fixed.

---

## Troubleshooting
- "Command not found" for `pre-commit` or `ruff`: install via `python -m pip install -e .[dev]` or `python -m pip install pre-commit ruff` and `pre-commit install` in the project folder.
- Pre-commit keeps re-downloading hooks: check write access to `.cache/` and that your home or repo cache is not cleaned between runs.
- Formatting/linting behaves differently locally vs CI: ensure you are running the same hook versions (`pre-commit autoupdate`), and use the same file selection as CI when comparing runs (see examples above).
