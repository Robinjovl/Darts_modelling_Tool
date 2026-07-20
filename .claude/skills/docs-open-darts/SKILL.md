---
name: docs-open-darts
description: Build and maintain open-DARTS documentation with Sphinx and MyST, including API docs and GitLab Pages deployment behavior. Use when editing or validating docs.
---

# Documentation

Use this skill for docs authoring and publishing checks.
Use the session conda environment. If the prompt defines one, use it. Otherwise, create and activate one session-level conda environment once, then reuse it across the rest of the session.

## Steps

1. Confirm the session conda environment is active; create it once if needed.
2. Install docs dependencies.
3. Build docs locally and resolve warnings/errors.
4. Update toctree or autosummary entries when adding new pages/modules.

## Primary commands

- Install deps: `pip install .[docs]`
- Build docs: `cd docs && make html`

## Reference

Read `references/documentation.md` for structure, style, and CI deployment details.
