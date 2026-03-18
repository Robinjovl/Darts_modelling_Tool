---
name: release-readiness-open-darts
description: Review release readiness for open-DARTS when preparing a version bump, tag, deployment change, or other user-visible compatibility-sensitive change that is about to ship. Trigger for release candidates, packaging metadata changes, supported-Python changes, wheel-matrix changes, or CI/deploy updates.
---

# Release Readiness

Use this skill for diff-based release review before tagging or approving release-critical changes.
Use the session conda environment for any local verification commands. If the prompt defines one, use it. Otherwise create and activate one session-level conda environment once, then reuse it across the rest of the session.

## Steps

1. Confirm the session conda environment is active if the review includes local verification commands; create it once if needed.
2. Choose the comparison base (usually the previous release tag).
3. Inspect the diff for compatibility, packaging, CI/deploy, and documentation impact.
4. Produce a release call with evidence, risk level, and an unblock checklist when needed.

## Primary commands

- List recent tags: `git tag --sort=-v:refname | head`
- Diff stats from previous tag: `git diff <previous-tag>...HEAD --stat`
- Commit summary: `git log --oneline <previous-tag>..HEAD`
- Release-critical files: `git diff <previous-tag>...HEAD -- pyproject.toml setup.py CMakeLists.txt .gitlab-ci.yml .cicd/jobs docs`

## Reference

Read `references/release-readiness.md` for the review checklist, risk criteria, and output format.
