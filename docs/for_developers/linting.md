## Python Linting and Formatting with Pre-commit and Ruff

This document describes the updated linting and formatting setup for the open-DARTS project.

### Highlights
- Pre-commit hooks run automatically on `git commit` / `git push` to enforce formatting, linting, and basic validations. Need to have pre-commit module installed.
Use `./helper_scripts/install_darts_deps.sh` on Linux or `helper_scripts\install_darts_deps.bat` for that (only for the initial open-DARTS installation, then use the script without '_deps').
If it is not installed, no checks would be performed locally and the CI/CD pipeline might fail, since it runs the same checks using the `pre-commit` stage.
- Ruff is used for linting and formatting checks. Avoid `--unsafe-fixes` usage with Ruff.
- There is an option  `--no-verify` to skip checks.

---

## Purpose of `.pre-commit-config.yaml`
Defines the set of checks ("hooks") that run automatically on `git commit` and `git push`. Each hook references a repository and version, and may have per-hook options (e.g., file globs, arguments).

## Hooks configured in `.pre-commit-config.yaml`

Ruff hooks (from `astral-sh/ruff-pre-commit`):
- `ruff-check` (with `--fix`, `--show-fixes`): Lints Python and applies safe, non-breaking fixes.
- `ruff-format`: Formats Python code (Ruff formatter)

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
Ruff is configured in `pyproject.toml` in the `[tool.ruff*]` sections.

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
- Python files in folders: `darts`, `tests`, `helper_scripts`, `.cicd`
- YAML files (`.gitlab-ci.yml`)
- TOML files (`pyproject.toml`)

---

## CI Pipeline Integration
The pre-commit job runs in the `pre_commit` stage using `python:3.10`, installs `pre-commit`, selects target files and executes:

```bash
pre-commit run --files $FILES --show-diff-on-failure --color always
```

The job currently allows failure (`allow_failure: true`) to ease adoption; aim to fix issues proactively.


## Tips:
- Do not use `ruff --unsafe-fixes` (or `--unsafe-fixes` via pre-commit). Only safe fixes are enabled.
- To update hook versions: `pre-commit autoupdate` (we also auto-update quarterly).
- Hook environments are cached under `.cache/pre-commit`.
- If a new directory is added for Python code, update the `files` glob for Ruff in `.pre-commit-config.yaml`.
- In CI/CD, failures in the pre-commit job currently do not fail the pipeline; treat them as warnings to be fixed.

---

## Troubleshooting
- "Command not found" for `pre-commit` or `ruff`: install via `python -m pip install -e .[dev]` or `python -m pip install pre-commit ruff` and `pre-commit install` in the project folder.
- Pre-commit keeps re-downloading hooks: check write access to `.cache/` and that your home or repo cache is not cleaned between runs.
- Formatting/linting behaves differently locally vs CI/CD: ensure you are running the same hook versions (`pre-commit autoupdate`), and use the same file selection as CI/CD when comparing runs (see examples above).
