# Skill: Verify open-DARTS

## Overview

This skill defines the repository's verification rule:

- Run the smallest set of checks that actually covers the touched area.
- Escalate to broader checks when the change crosses Python/C++ boundaries,
  affects numerical behavior, or changes build/test infrastructure.
- Do not claim the task is complete unless the required checks passed or the
  remaining blockers are stated precisely.

---

## Conda Environment Policy

Use a prompt-defined conda environment when one is provided. Otherwise, create
and activate one session-level environment at the first build, install, test,
debug, docs, or lint step, then reuse it across all skills for the rest of the
session.

Default session environment:

```bash
conda create -y -n open-darts-session python=3.10
conda activate open-darts-session
```

Keep all commands for the session's verification work in the same environment.

---

## Classify the change first

Start from `git diff --name-only` and map the touched paths to validation:

| Touched area | Required validation |
|---|---|
| `docs/` only, without behavior impact | Use `docs-open-darts`; no runtime stack required |
| Python code under `darts/`, `models/`, `tests/`, `tutorials/` | Lint changed files, then run the nearest affected tests |
| `engines/`, `discretizer/`, `solvers/`, `thirdparty/`, `CMakeLists.txt` | Build/install validation plus relevant Python-facing tests |
| `pyproject.toml`, `setup.py`, install/build helper scripts | Editable install or wheel/build validation, plus at least one runtime smoke test |
| `.gitlab-ci.yml`, `.cicd/jobs/` | Reproduce the closest local commands and explain CI impact |
| Mixed Python + C++ or package boundary changes | Run build/install plus the broad regression stack |

If the change is broad and the right scope is unclear, run `project-overview`
first, then bias toward more validation rather than less.

---

## Required command groups

### 1. Linting

For Python file edits, run:

```bash
pre-commit run -v --files /absolute/path/to/file.py --show-diff-on-failure
```

If many Python files changed, run pre-commit on the changed file set instead of
one file at a time.

### 2. Build/install validation

Run build or install validation when the change affects compiled code, package
metadata, or build scripts.

Common commands:

```bash
./helper_scripts/install_darts.sh -e
./helper_scripts/build_darts_cmake.sh -c -w -p -m -j 8
```

Use `build-open-darts` if you need GPU, debug, Valgrind, or platform-specific
flags.

### 3. Runtime and regression validation

Use the nearest affected test first, then widen if the change is cross-cutting.

```bash
cd models && darts run_test_suite2.py LOG
cd tests/engines/src/interpolation && darts test_interpolation.py
cd discretizer/tests/compare_discretizers && darts main.py
```

Guidance:

- Changes in `darts/physics/`, `darts/models/`, `darts/reservoirs/`, or
  `models/` usually require the model regression suite.
- Interpolation code or bindings require the interpolator tests.
- Discretizer code or mesh/discretization interfaces require the discretizer
  tests.
- Build-system changes should still include at least one runtime smoke test
  after install/build validation.

### 4. CI behavior changes

For `.gitlab-ci.yml` or `.cicd/jobs/` changes:

- identify which stage, job, or matrix behavior changed
- reproduce the closest local build/test/docs command
- state what could not be reproduced locally

Use `gitlab-cicd-open-darts` for the pipeline-specific reasoning.

---

## Reporting rules

When you finish verification, report:

- which files or subsystems triggered the checks
- exactly which commands were run
- which checks were intentionally skipped and why
- any remaining blocker that prevents full verification

Do not say "verified" if the required check set was not executed.
