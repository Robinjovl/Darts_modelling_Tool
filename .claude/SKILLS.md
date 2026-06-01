# open-DARTS Skills Catalog

Project skills are stored **once**, in `.claude/skills/`, and serve every coding agent:

- **Claude Code** discovers them natively from `.claude/skills/`.
- **Codex** (and other AGENTS.md-aware tools) read the same files via `AGENTS.md`.

There is a single source of truth — no per-tool copies to keep in sync.

## Available skills

| Skill | Path | Purpose |
|---|---|---|
| `project-overview` | `.claude/skills/project-overview` | Repository architecture and module map |
| `build-open-darts` | `.claude/skills/build-open-darts` | CMake builds, wheel builds, editable installs, GPU/debug modes |
| `lint-format-open-darts` | `.claude/skills/lint-format-open-darts` | Ruff and pre-commit workflows |
| `test-open-darts` | `.claude/skills/test-open-darts` | Regression tests and component tests |
| `verify-open-darts` | `.claude/skills/verify-open-darts` | Mandatory pre-handoff validation stack |
| `model-workflow-open-darts` | `.claude/skills/model-workflow-open-darts` | Configure, run, analyze, and plot models |
| `debug-profile-open-darts` | `.claude/skills/debug-profile-open-darts` | Debug, Valgrind, VTune, timer diagnostics |
| `docs-open-darts` | `.claude/skills/docs-open-darts` | Sphinx and API docs maintenance |
| `gitlab-cicd-open-darts` | `.claude/skills/gitlab-cicd-open-darts` | GitLab CI/CD structure and troubleshooting |
| `release-readiness-open-darts` | `.claude/skills/release-readiness-open-darts` | Release / version-bump compatibility review |

## Maintenance

```bash
# Validate skill structure: frontmatter, name/directory match, and reference links
python helper_scripts/validate_skills.py
```

## Notes

- Each skill directory must include `SKILL.md` with YAML frontmatter containing
  `name` (matching the directory) and `description`.
- Detailed workflow material belongs in `references/` files and is loaded only
  when needed.
- Skills must instruct agents to preserve existing file line endings and avoid
  line-ending-only changes unless the user explicitly requests them.
