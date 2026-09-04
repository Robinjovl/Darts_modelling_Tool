#!/usr/bin/env python3
"""Validate project skills: frontmatter, references, placeholders, mirror consistency."""

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SKILLS = REPO / ".agents" / "skills"
PLACEHOLDER = re.compile(r"^\s*-?\s*TODO\b", re.MULTILINE)


def check_skill(skill_dir: Path) -> list[str]:
    errors = []
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        return [f"{skill_dir.name}: missing SKILL.md"]
    text = skill_md.read_text(encoding="utf-8")
    match = re.match(r"---\n(.*?)\n---\n", text, re.DOTALL)
    if not match:
        return [f"{skill_dir.name}: SKILL.md has no YAML frontmatter"]
    fields = dict(
        line.split(":", 1) for line in match.group(1).splitlines() if ":" in line
    )
    name = fields.get("name", "").strip()
    if name != skill_dir.name:
        errors.append(f"{skill_dir.name}: frontmatter name {name!r} != directory name")
    if not fields.get("description", "").strip():
        errors.append(f"{skill_dir.name}: empty description")
    for rel in re.findall(r"`(references/[^`]+\.md)`", text):
        if not (skill_dir / rel).exists():
            errors.append(f"{skill_dir.name}: SKILL.md references missing {rel}")
    for md in sorted(skill_dir.rglob("*.md")):
        if PLACEHOLDER.search(md.read_text(encoding="utf-8")):
            errors.append(f"{skill_dir.name}: unfinished placeholder in {md.name}")
    for script in (
        (skill_dir / "scripts").glob("*") if (skill_dir / "scripts").is_dir() else []
    ):
        if script.suffix in {".py", ".sh"} and not script.stat().st_mode & 0o111:
            errors.append(f"{skill_dir.name}: {script.name} is not executable")
    return errors


def main() -> int:
    errors = []
    for skill_dir in sorted(p for p in SKILLS.iterdir() if p.is_dir()):
        errors.extend(check_skill(skill_dir))
    sync = subprocess.run(
        [
            sys.executable,
            str(REPO / "helper_scripts" / "sync_agent_skills.py"),
            "--check",
        ],
        capture_output=True,
        text=True,
    )
    if sync.returncode != 0:
        errors.append("mirror out of sync: " + (sync.stdout + sync.stderr).strip())
    for error in errors:
        print(f"ERROR: {error}")
    print("skills OK" if not errors else f"{len(errors)} problem(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
