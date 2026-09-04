# workflows/evals — evaluation harness (developer tool)

Runs a skill-driven agent on hidden cases and scores it after exit (design section 10).

- `sandbox.py`: `bwrap` mount manifest (repository and env read-only; `truth/`, `cases/` and
  `scorer.py` masked; one writable run directory; scratch HOME with read-only credentials) and a
  canary that proves the masking from inside.
- `broker.py`: parent-side simulation broker over a Unix socket; the agent submits study specs,
  the broker runs `python -m workflows` into a parent-owned results directory.
- `controller.py`: `AgentRunner` protocol (`ClaudeCliRunner`, `ShellRunner`), accounting (wall;
  CPU and peak RSS from `/usr/bin/time` inside the sandbox because `bwrap --unshare-pid` hides
  descendants from the parent's rusage; CLI usage JSON with tokens and cost), `run.json`,
  post-exit re-hashing. The CLI runner binds the `claude`/`node` install prefixes read-only,
  works from the repository root so project skills are discovered, and allow-lists tools
  explicitly (headless runs cannot prompt; the sandbox is the boundary).
- The agent submits studies with `python -m workflows.evals.broker --spec ... --study ...`
  (socket from `WORKFLOWS_BROKER`); case prompts may use `{repo}`, `{python}`, `{run_dir}`.
- `scorer.py`: gates-based scoring against `truth/<case>/reference.json` (hidden).
- `report.py`: `python -m workflows.evals.report` tabulates `runs/` (pass, gate, turns, cost, wall)
  with per case/model/head aggregates for the acceptance rule below.

Network: `bwrap` shares the network namespace; outbound traffic is unrestricted unless a proxy
is deployed, and every run record says so.

Fast tests: `workflows/tests/test_evals.py` (skips when `bwrap` is absent).

Run one hidden case (parent side, writes `runs/<case>/<utc>-<model>.json`; add
`--runner codex --model <codex model>` to evaluate Codex CLI instead of Claude Code):

```bash
PYTHONPATH=<repo> <env>/bin/python -m workflows.evals.controller \
  --case workflows/evals/cases/opt-place-proxy-smoke.json --model claude-sonnet-4-6 --run-root <dir>
```

## Pre-registered protocol (smoke tier)

- Cases: `opt-place-proxy-smoke`, `ens-proxy-smoke`, `hm-proxy-twin-smoke` (Brugge proxy, seeds
  31 / 11 / 21). Each has a hidden `truth/<case>/reference.json` produced by a controller-side
  study through the same CLI, and a `tolerance` fixed before any agent run.
- Runner: `ClaudeCliRunner` with an exact model id, `--max-turns` and `--max-budget-usd` from the
  case budget, tools allow-listed, network unrestricted (recorded).
- Recorded per run (`runs/<case>/<utc>-<model>.json`): pass/fail with the gate values, tokens by
  kind, cost, turns, wall, CPU, peak RSS, broker requests (spec hashes), git head, result text.
- A skill or harness edit is accepted only if, over the three smoke cases with the same seeds and
  model, it does not lower the pass count and does not raise median cost or turns; ties are
  broken by fewer broker requests (fewer wasted studies). Paired runs (before/after) use the same
  case prompts; prompts are never edited to fit a model.
- Full tier (design Section 9: precision-adaptive ensemble references, ten hidden truths,
  exhaustive placement over all feasible cells) is not yet generated.
