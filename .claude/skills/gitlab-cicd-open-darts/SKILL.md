---
name: gitlab-cicd-open-darts
description: Operate and troubleshoot the open-DARTS GitLab CI/CD pipeline, including stage flow, job matrix, trigger rules, and deployment jobs. Use when editing pipeline logic or reproducing CI behavior locally.
---

# GitLab CI/CD

Use this skill when pipeline behavior is part of the task.
Use the session conda environment. If the prompt defines one, use it. Otherwise, create and activate one session-level conda environment once, then reuse it across the rest of the session.

## Steps

1. Map requested changes to specific CI job files in `.cicd/jobs/`.
2. Validate stage/rule interactions and matrix dependencies.
3. Reproduce the relevant workflow locally from the session conda environment where possible.

## Key files

- Entry point: `.gitlab-ci.yml`
- Jobs: `.cicd/jobs/*.yml`

## Reference

Read `references/ci-cd-pipeline.md` for full matrix, triggers, artifacts, and deployment details.
