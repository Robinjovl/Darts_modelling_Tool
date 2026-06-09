# open-DARTS Skills Catalog

This repository stores project skills in `.agents/skills/` and mirrors them to
`.claude/skills/` for Claude Code compatibility.

## Skill Locations

- Source of truth: `.agents/skills/`
- Mirror for Claude Code: `.claude/skills/`

## Available Skills

| Skill | Path | Purpose |
|---|---|---|
| `project-overview` | `.agents/skills/project-overview` | Repository architecture and module mapping |
| `build-open-darts` | `.agents/skills/build-open-darts` | CMake builds, wheel builds, editable installs, GPU/debug modes |
| `lint-format-open-darts` | `.agents/skills/lint-format-open-darts` | Ruff and pre-commit workflows |
| `test-open-darts` | `.agents/skills/test-open-darts` | Regression tests and component tests |
| `debug-profile-open-darts` | `.agents/skills/debug-profile-open-darts` | Debug, Valgrind, VTune, timer diagnostics |
| `docs-open-darts` | `.agents/skills/docs-open-darts` | Sphinx and API docs maintenance |
| `gitlab-cicd-open-darts` | `.agents/skills/gitlab-cicd-open-darts` | GitLab CI/CD structure and troubleshooting |

## Maintenance Commands

```bash
# Sync source skills to Claude mirror
python helper_scripts/sync_agent_skills.py

# Validate skill structure + metadata + mirror consistency
python helper_scripts/validate_skills.py
```

## Notes

- Each skill directory must include `SKILL.md` with YAML frontmatter containing
  `name` and `description`.
- Detailed workflow material belongs in `references/` files and is loaded only
  when needed.
- Skills must instruct agents to preserve existing file line endings and avoid
  line-ending-only changes unless the user explicitly requests them.
