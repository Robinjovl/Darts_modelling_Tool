# open-DARTS — Claude Code

This repository keeps **one** source of agent guidance. The shared, vendor-neutral
instructions live in [`AGENTS.md`](AGENTS.md) (the open AGENTS.md standard, also read
by Codex and other agents). Claude Code imports them below, so both toolchains follow
exactly one source of truth and nothing is duplicated.

Project skills live once in `.claude/skills/`: Claude Code discovers them natively,
and they are the same skills Codex reads through `AGENTS.md`. When a task matches a
skill, use `.claude/skills/<skill-name>/SKILL.md` and any file under its `references/`.

@AGENTS.md
