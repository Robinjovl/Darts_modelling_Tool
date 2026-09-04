# workflows/evals — evaluation harness (developer tool)

Runs a skill-driven agent on hidden cases and scores it after exit (design section 10).

- `sandbox.py`: `bwrap` mount manifest (repository and env read-only; `truth/`, `cases/` and
  `scorer.py` masked; one writable run directory; scratch HOME with read-only credentials) and a
  canary that proves the masking from inside.
- `broker.py`: parent-side simulation broker over a Unix socket; the agent submits study specs,
  the broker runs `python -m workflows` into a parent-owned results directory.
- `controller.py`: `AgentRunner` protocol (`ClaudeCliRunner`, `ShellRunner`), parent-side
  accounting (wall, CPU, peak RSS, CLI usage JSON), `run.json`, post-exit re-hashing.
- `scorer.py`: gates-based scoring against `truth/<case>/reference.json` (hidden).

Network: `bwrap` shares the network namespace; outbound traffic is unrestricted unless a proxy
is deployed, and every run record says so.

Fast tests: `workflows/tests/test_evals.py` (skips when `bwrap` is absent).
