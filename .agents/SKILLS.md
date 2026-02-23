# open-DARTS Agent Skills

This folder contains structured skill documents for AI agents (GPT, Claude,
Gemini, Copilot, Codex, etc.) to automate common development workflows in the
open-DARTS project.

## Available Skills

| Skill | File | Description |
|---|---|---|
| **Project Overview** | [project-overview.md](project-overview.md) | Architecture, repository layout, key modules, conventions |
| **Build & Compile** | [build-and-compile.md](build.md) | CMake builds, Python packaging, ST/MT/GPU configurations |
| **Linting & Formatting** | [linting-and-formatting.md](linting-and-formatting.md) | Ruff, pre-commit hooks, code quality rules |
| **Testing** | [testing.md](testing.md) | Test suite, model regression tests, interpolator/discretizer tests |
| **Debugging & Profiling** | [debugging-and-profiling.md](debugging-and-profiling.md) | Debug builds, Valgrind, VTune, timers, sanitizers |
| **Documentation** | [documentation.md](documentation.md) | Sphinx build, API docs, adding new pages |
| **CI/CD Pipeline** | [ci-cd-pipeline.md](ci-cd-pipeline.md) | GitLab CI stages, trigger rules, deployment, local reproduction |

## How to Use

Point your AI agent/assistant at the relevant skill file when you need help with
a specific workflow. Each file is self-contained with commands, configuration
details, and troubleshooting tips.

For a quick overview of the project, start with `project-overview.md`.

## Quick Reference

```bash
# Build everything from scratch
./helper_scripts/build_darts_cmake.sh -c -w -m -j 8

# Install in editable mode (fast, if build/ exists)
./helper_scripts/install_darts.sh -e

# Lint a file
pre-commit run -v --files /absolute/path/to/file.py --show-diff-on-failure

# Run a model
darts /absolute/path/to/model-folder/main.py

# Run test suite
cd models && darts run_test_suite2.py LOG

# Build documentation
cd docs && make html
```
