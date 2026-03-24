### Agent quick commands and conventions

- **Use absolute paths**
- **Run non-interactively** and print the command before execution.
- **Project-level commands** should be executed from the repo root.
- **Environment rule:** for build, install, test, debug, docs, or lint commands, use a prompt-defined conda environment when one is provided; otherwise create and activate one session-level conda environment at the first such step and reuse it across all skills for the rest of the session.
- **Skills source of truth:** `.agents/skills`
- **Claude mirror:** `.claude/skills` (keep synchronized with `helper_scripts/sync_agent_skills.py`)

### Project overview

- Core Python package: `darts/`
- C++ extensions and bindings: `engines/`, `discretizer/`, `solvers/`
- Example and regression models: `models/`
- Tests: `tests/`, `discretizer/tests/`
- Documentation: `docs/`
- CI/CD and release flow: `.gitlab-ci.yml`, `.cicd/jobs/`

### Mandatory skill usage

- Use `$project-overview` before non-trivial code changes, when touching multiple subsystems, or when the execution path is not obvious.
- Use `$model-workflow-open-darts` when configuring, running, analyzing, troubleshooting, post-processing, or plotting model workflows under `models/`, `tutorials/`, or other model-script locations.
- Use `$build-open-darts` when compilation, editable installs, wheel builds, GPU builds, or packaging changes are required, or when work touches `engines/`, `discretizer/`, `solvers/`, `thirdparty/`, `CMakeLists.txt`, `setup.py`, `pyproject.toml`, or build helper scripts.
- Use `$verify-open-darts` when changes affect runtime code, numerical behavior, tests, packaging, or build/test behavior. Do not mark the task complete until the required checks for the touched area pass or a concrete blocker is recorded.
- Use `$test-open-darts` when reproducing or validating simulation behavior, regression failures, interpolator behavior, discretizer behavior, or model changes.
- Use `$lint-format-open-darts` for Python file edits before handoff. **Always run `ruff check` on all modified Python files before presenting changes as complete or committing.** Pre-commit hooks enforce this — failing to lint wastes a commit cycle.
- Use `$docs-open-darts` when editing `docs/`, Sphinx configuration, docstrings used by docs, or documentation with behavior impact.
- Use `$gitlab-cicd-open-darts` when touching `.gitlab-ci.yml`, `.cicd/jobs/`, release/deploy rules, or CI artifact flow.
- Use `$debug-profile-open-darts` when investigating crashes, leaks, memory issues, slowdowns, or profiling data.
- Use `$release-readiness-open-darts` when preparing a release, reviewing a version bump or tag, or assessing whether user-visible compatibility, packaging, or deployment changes are safe to ship.

### Compatibility rules

- Preserve user-visible Python API and CLI behavior when practical.
- If exported constructor signatures, dataclass-like parameter ordering, CLI flags, file formats, or packaging requirements must change, call out the compatibility impact explicitly and add focused validation coverage.
- Treat supported Python versions, wheel availability, and optional solver/backend requirements as release-critical compatibility surfaces.

### Validate skill trees

```bash
python helper_scripts/sync_agent_skills.py --check
```

### Sync skill trees

```bash
python helper_scripts/sync_agent_skills.py
```

### Lint Python files

```bash
pre-commit run -v --files <absolute-path-to-file> --show-diff-on-failure
```

### Run a model

```bash
darts <script_path_name>
```

### Debug a model

```bash
gdb --args darts <script_path_name>
cuda-gdb --args darts <script_path_name>
```

### Build/install Python package locally (from repo root)

```bash
./helper_scripts/install_darts.sh -e
```

### Git rules

- Baseline branch is `development`
- Keep commits focused and include a brief but descriptive message.
