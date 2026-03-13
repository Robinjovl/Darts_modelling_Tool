---
name: docs-open-darts
description: Build and maintain open-DARTS documentation with Sphinx and MyST, including API docs and GitLab Pages deployment behavior. Use when editing or validating docs.
---

# Documentation

Use this skill for docs authoring and publishing checks.

## Steps

1. Install docs dependencies.
2. Build docs locally and resolve warnings/errors.
3. Update toctree or autosummary entries when adding new pages/modules.

## Primary commands

- Install deps: `pip install .[docs]`
- Build docs: `cd docs && make html`

## Reference

Read `references/documentation.md` for structure, style, and CI deployment details.
