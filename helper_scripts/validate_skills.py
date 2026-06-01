#!/usr/bin/env python3
"""Validate the open-DARTS agent skills tree.

Every skill under the skills directory must provide a ``SKILL.md`` whose YAML
frontmatter declares a ``name`` matching the skill directory and a non-empty
``description``. Each ``references/*.md`` file referenced from a ``SKILL.md`` body
must exist, and declared skill names must be unique.

Skills live once under ``.claude/skills``: Claude Code discovers them natively, and
Codex (plus other AGENTS.md-based agents) reads the same tree through ``AGENTS.md``.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REFERENCE_RE = re.compile(r"`(references/[^`]+\.md)`")


def parse_frontmatter(text: str) -> dict[str, str] | None:
    """Return the leading YAML frontmatter as a flat dict, or None if malformed."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    fields: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            return fields
        key, sep, value = line.partition(":")
        if sep:
            fields[key.strip()] = value.strip()
    return None  # no closing delimiter


def validate_skill(skill_dir: Path) -> tuple[list[str], str]:
    """Return (errors, declared_name) for a single skill directory."""
    name = skill_dir.name
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.is_file():
        return [f"{name}: missing SKILL.md"], ""

    text = skill_md.read_text(encoding="utf-8")
    fields = parse_frontmatter(text)
    if fields is None:
        return [f"{name}: SKILL.md has no closing '---' YAML frontmatter"], ""

    errors: list[str] = []
    declared = fields.get("name", "")
    if not declared:
        errors.append(f"{name}: frontmatter is missing 'name'")
    elif declared != name:
        errors.append(f"{name}: frontmatter name '{declared}' does not match directory")

    if not fields.get("description"):
        errors.append(f"{name}: frontmatter is missing 'description'")

    for ref in REFERENCE_RE.findall(text):
        if not (skill_dir / ref).is_file():
            errors.append(f"{name}: referenced file '{ref}' does not exist")

    return errors, declared


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the agent skills tree.")
    parser.add_argument(
        "--skills-dir",
        default=".claude/skills",
        help="Directory containing skill folders (default: .claude/skills)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    skills_dir = Path(args.skills_dir).resolve()
    if not skills_dir.is_dir():
        print(f"Skills directory does not exist: {skills_dir}", file=sys.stderr)
        return 1

    skill_dirs = sorted(p for p in skills_dir.iterdir() if p.is_dir())
    if not skill_dirs:
        print(f"No skills found under {skills_dir}", file=sys.stderr)
        return 1

    errors: list[str] = []
    declared_names: dict[str, str] = {}
    for skill_dir in skill_dirs:
        skill_errors, declared = validate_skill(skill_dir)
        errors.extend(skill_errors)
        if declared:
            if declared in declared_names:
                errors.append(
                    f"{skill_dir.name}: duplicate skill name '{declared}' "
                    f"(also used by '{declared_names[declared]}')"
                )
            else:
                declared_names[declared] = skill_dir.name

    if errors:
        print("Skill validation failed:", file=sys.stderr)
        for err in errors:
            print(f"  {err}", file=sys.stderr)
        return 1

    print(f"Validated {len(skill_dirs)} skills under {skills_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
