# Skill: Release Readiness

## Overview

This skill reviews a concrete diff and answers one operational question:

Is the change safe to ship now?

The output should be a release call backed by evidence, not a generic summary.

---

## Conda Environment Policy

Pure git-diff review does not require a new environment, but any build, install,
test, docs, or debug validation that is part of the release review should run
from the session conda environment. If the prompt defines one, use it.
Otherwise, create and activate one session-level environment once, then reuse it
across all skills for the rest of the session.

Default session environment:

```bash
conda create -y -n open-darts-session python=3.11
conda activate open-darts-session
```

Use the same session environment for all supporting verification commands
referenced in the release call.

---

## When to use it

Use this skill when work is approaching release and any of these changed:

- version metadata in `pyproject.toml`, `setup.py`, or tag/deploy configuration
- supported Python versions or wheel/platform coverage
- compiled extension build behavior
- user-visible runtime behavior, CLI behavior, or model-facing defaults
- CI or deployment logic under `.gitlab-ci.yml` or `.cicd/jobs/`

For ordinary feature work that is not near release, use `verify-open-darts`
instead.

---

## Comparison base

Prefer the previous release tag as the baseline.

Useful commands:

```bash
git tag --sort=-v:refname | head -n 10
git diff <previous-tag>...HEAD --stat
git log --oneline <previous-tag>..HEAD
```

If there is no meaningful previous tag for the scope in question, compare
against `development` and state that assumption.

---

## Review checklist

### Packaging and install surface

Inspect:

- `pyproject.toml`
- `setup.py`
- `CMakeLists.txt`
- build/install helper scripts
- wheel rename or upload logic

Look for:

- supported Python version changes
- new required system or third-party dependencies
- changed wheel matrix or platform coverage
- install steps that differ from current docs

### Runtime compatibility

Inspect changed files under:

- `darts/`
- `engines/`
- `discretizer/`
- `linear_solvers/`
- `models/`

Look for:

- changed constructor signatures or defaults
- changed CLI behavior
- changed numerical behavior that may invalidate reference outputs
- removed or renamed public modules, functions, or scripts

### CI and deployment behavior

Inspect:

- `.gitlab-ci.yml`
- `.cicd/jobs/`
- docs deployment behavior
- PyPI/TestPyPI/Zenodo or registry publication jobs

Look for:

- missing artifacts for release jobs
- build/test matrix regressions
- release jobs that no longer match the package metadata
- documentation deployment changes that leave release docs stale

### Documentation and migration notes

Check whether docs or release-facing notes need updates when:

- installation requirements changed
- Python support changed
- user-visible behavior changed
- new optional dependencies or solvers are required

---

## Release call rules

Start from "safe to ship" and move away from that only when the diff contains
concrete evidence of a real release risk.

Use one of these calls:

- `GREEN`: safe to ship; no release-blocking issue found
- `YELLOW`: shippable, but follow-up or explicit release-note callout is needed
- `BLOCKED`: do not ship until the listed issues are resolved

If you use `BLOCKED`, include a specific unblock checklist tied to the evidence.

---

## Output format

Use this structure:

```text
Release readiness review
Release call: GREEN | YELLOW | BLOCKED
Scope summary:
- ...
Risks:
- ...
Required follow-ups:
- ...
Evidence:
- ...
```

Keep the evidence concrete. Point to exact files, commands, metadata changes, or
test gaps that justify the call.
