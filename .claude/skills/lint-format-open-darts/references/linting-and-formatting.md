# Skill: Linting and Formatting

## Overview

open-DARTS uses **Ruff** for Python linting and formatting, orchestrated through
**pre-commit** hooks. Configuration lives in `pyproject.toml` (Ruff rules) and
`.pre-commit-config.yaml` (hook definitions).

---

## Tools

| Tool | Role | Config location |
|---|---|---|
| Ruff | Python linter + formatter | `pyproject.toml` `[tool.ruff*]` |
| pre-commit | Hook runner (auto on commit/push) | `.pre-commit-config.yaml` |

---

## Ruff Configuration (pyproject.toml)

```toml
[tool.ruff]
line-length = 88
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]
ignore = ["E203", "E501", "F403", "F405", "E722", "E731", "C901"]
unfixable = ["F841"]

[tool.ruff.format]
quote-style = "preserve"

[tool.ruff.lint.mccabe]
max-complexity = 10
```

### Enabled Rule Groups

| Code | Category |
|---|---|
| `E` | pycodestyle errors |
| `F` | pyflakes |
| `I` | isort (import sorting) |
| `UP` | pyupgrade (safe modernization) |
| `B` | flake8-bugbear (correctness) |

---

## Pre-commit Hooks

Configured hooks (`.pre-commit-config.yaml`):

**Ruff hooks** (applied to `darts/`, `tests/`, `helper_scripts/`, `.cicd/`):
- `ruff-check` — lint with `--fix --show-fixes` (safe auto-fixes only)
- `ruff-format` — format Python code

**General hygiene** (from `pre-commit/pre-commit-hooks`):
- `end-of-file-fixer` — ensure trailing newline
- `trailing-whitespace` — strip trailing whitespace
- `check-yaml` — validate YAML syntax
- `check-toml` — validate TOML syntax
- `mixed-line-ending` — detect mixed line endings; do not use it to make
  line-ending-only changes unless the user explicitly requests an EOL change
- `detect-private-key` — prevent accidentally committed keys
- `check-added-large-files` — block files > 500 KB
- `check-merge-conflict` — detect conflict markers

**Local repository policy hooks**:
- `check-line-endings` — require tracked `.py`, `.pyi`, and `.sh` repository
  blobs to use LF-only line endings

---

## Common Commands

### Install pre-commit (first time)

```bash
pip install pre-commit ruff
# or
pip install -e .[dev]

# Install hooks into local git repo
pre-commit install
```

### Lint a Single File

```bash
pre-commit run -v --files /absolute/path/to/file.py --show-diff-on-failure
```

### Lint All Project Files (mirrors CI)

```bash
FILES="$(
  {
    git ls-files -- 'darts' 'tests' 'helper_scripts' '.cicd' \
                    '.gitlab-ci.yml' '.pre-commit-config.yaml' 'pyproject.toml'
  } | grep -E '\.(py|ya?ml|toml)$'
)"
pre-commit run --files $FILES --show-diff-on-failure --color always
```

### Run Ruff Directly (without pre-commit)

```bash
# Lint
ruff check darts/ tests/ --fix --show-fixes

# Format
ruff format darts/ tests/

# Check formatting without modifying
ruff format --check darts/ tests/

# Lint a single file
ruff check /absolute/path/to/file.py --fix --show-fixes
ruff format /absolute/path/to/file.py
```

### Update Hook Versions

```bash
pre-commit autoupdate
```

---

## Important Rules

1. **Never use `--unsafe-fixes`** with Ruff. Only safe, non-breaking fixes are
   enabled.
2. **Do not use `--no-verify`** on commits unless absolutely necessary — CI will
   catch violations and the pipeline may fail.
3. **The `F841` rule** (unused variables) is marked `unfixable` — Ruff will
   report but not auto-remove them; fix manually.
4. Ruff is configured to **preserve existing quote style** (`quote-style =
   "preserve"`).
5. Write Python docstrings with opening and closing triple quotation marks on
   separate lines.
6. Document Python input and output arguments with `:param name:`, `:type name:`,
   `:return:`, and `:rtype:` fields.
7. New Python directories added to the project must be included in the `files`
   glob in `.pre-commit-config.yaml`.
8. Preserve existing file line endings. Do not manually normalize CRLF/LF style,
   do not make line-ending-only changes, and revert accidental EOL-only rewrites
   from formatters or hooks unless the user explicitly requested them.
9. `.gitattributes` and `check-line-endings` make LF mandatory for committed
   `.py`, `.pyi`, and `.sh` content. On Windows, this check reads Git blobs so
   checkout conversion does not create false failures.

---

## CI Integration

The CI pipeline runs a `pre_commit` stage (defined in `.cicd/jobs/pre-commit.yml`)
that executes the same checks on all Python, YAML, and TOML files. The job is
set to `allow_failure: false`, so linting failures block the pipeline.

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `pre-commit` or `ruff` not found | `pip install -e .[dev]` or `pip install pre-commit ruff`, then `pre-commit install` |
| Hook env keeps re-downloading | Check write access to `.cache/pre-commit` |
| Local vs CI differences | Run `pre-commit autoupdate` to align versions |
| Want to skip hooks once | `git commit --no-verify -m "message"` (use sparingly) |
