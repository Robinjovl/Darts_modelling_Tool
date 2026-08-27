---
name: project-overview
description: Repository architecture and module map for open-DARTS. Use when planning code changes, locating where functionality lives, understanding Python/C++ boundaries, or choosing the right build and test workflow.
---

# Project Overview

Use this skill to orient quickly in the open-DARTS repository before implementing changes.
When the planned work leads to build, install, test, debug, docs, or lint commands, use the session conda environment. If the prompt defines one, use it. Otherwise, create and activate one session-level conda environment once, then reuse it across the rest of the session.

## Steps

1. Read `references/project-overview.md` to identify the relevant subsystem.
2. Map requested changes to concrete paths (`darts/`, `engines/`, `interpolation/`, `discretizer/`, `linear_solvers/`, `models/`, `tests/`, `docs/`).
3. Confirm the execution path (build, lint, tests) and the session conda environment plan before editing.

## Conventions

- Preserve existing file line endings. Do not make line-ending-only changes or
  run tools with the intent of changing EOL style unless the user explicitly
  requests an EOL change.
- Write Python docstrings with the opening and closing triple quotation marks on separate lines.
- Document Python input and output arguments with `:param name:`, `:type name:`, `:return:`, and `:rtype:` fields.

## Output

- State which modules and files are in scope.
- State which validation flow will be used after edits.
