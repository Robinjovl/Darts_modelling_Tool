---
orphan: true
---

# Agentic workflows: user guide

Companion to the design page (`agentic_workflows_design.md`). Everything described here lives in
the repository: the study layer `workflows/`, three skills under `.agents/skills` (mirrored to
`.claude/skills`), and the evaluation harness `workflows/evals/`. Nothing is installed separately;
the package is imported through the checkout.

Prerequisites: an environment with open-DARTS built or installed (`$build-open-darts`, or
`helper_scripts/install_darts.sh -e`) and `pip install -r workflows/requirements.txt` into that
same interpreter; that interpreter is `<env>/bin/python` below. Headless runs need the `claude`
or `codex` CLI installed and logged in; the evaluator additionally needs `bubblewrap` (`bwrap`).

## 1. What the workflows do

| Workflow (`spec.workflow`) | Skill | Question it answers | Implemented drivers |
|---|---|---|---|
| `ensemble` | `ensemble-study-open-darts` | How uncertain is an output, and which inputs matter? | LHS, Sobol sequence, Morris screening, Saltelli variance decomposition; bootstrap P10/P50/P90 with 90 % intervals; Morris and Sobol indices |
| `hm-esmda` | `history-matching-open-darts` | Which inputs are consistent with observed well data? | ES-MDA over `dageo` with diagonal noise, full-cell log10-permeability field, Gaspari–Cohn localization on cell-to-well distance, identical-twin truths, validity gates |
| `optimize` | `optimization-open-darts` | Where should a well go? | Exhaustive discrete placement over feasible cells with a minimum-spacing rule, baseline regret, seeded candidate subsets |

Terms used throughout: P10/P50/P90 are statistical percentiles (P10 is the 10th percentile of
the value, not the reservoir-engineering exceedance convention; production rates are negative on
the proxy). An identical twin is a synthetic truth drawn from the declared prior, with its own
hidden noisy observations, generated once by `truth` so a calibration can be graded against a
known answer. Regret is the gap between the reference objective and a candidate's, normalized by
the reference-minus-baseline gap, so 0 at the optimum.

The shared substrate is described once in `workflows/docs/study-substrate.md`; what a user needs
at a glance: one study is one JSON spec (`name`, `workflow`, `model`, `parameters`, `seed_root`,
`observations`, `design`, `compute`; schema `workflows/spec.py:STUDY_SCHEMA`), every simulation
runs in its own process and is journaled, and a model joins through an adapter.
`workflows.adapters.brugge_proxy` covers `models/Uniform_Brugge` (431 cells, about 10 s per
member); another model needs its own `ModelAdapter` (`build`, `apply`, `run`, `observe`) first.

Not implemented, and stated as such inside the skills: adjoint-gradient history matching (an
assertive check exists under `workflows/evals/spikes/`, design Appendix C), DiWA, control
optimization, surrogate, robust and CO2 drivers, the full-field Brugge bed.

## 2. Running a workflow

Pin the interpreter; never rely on a bare `python` or `darts` from another environment.

```bash
cd <repo>
export PYTHONPATH=<repo>
PY=<env>/bin/python

$PY -m workflows estimate --spec study.json                  # planned simulations, wall, CPU-hours, disk
$PY -m workflows truth    --spec study.json --study runs/s1  # hm-esmda only, and before run: the twin truth
$PY -m workflows run      --spec study.json --study runs/s1  # runs; analyzes an ensemble on completion
$PY -m workflows analyze  --spec study.json --study runs/s1  # ensemble: re-analyze a finished study
```

Each command prints one JSON line. `run` resumes a study by default (members already recorded
`ok` in the journal are skipped; `--no-resume` restarts). Results live under `--study`:
`manifest.json`, `journal.jsonl`, `members/<id>/result.json`, and per workflow `analysis.json`,
`esmda_summary.json` with `params_step<k>.npy` and `data_step<k>.npy`, or
`optimize_summary.json` with `candidates.json` (table in the substrate reference).

Minimal specs (all against the Brugge proxy):

```json
{"name": "ens", "workflow": "ensemble", "seed_root": 11,
 "model": {"model_dir": "models/Uniform_Brugge", "adapter": "workflows.adapters.brugge_proxy"},
 "parameters": [{"name": "tm", "family": "LogScalarParam", "args": {"low": 0.5, "high": 2.0, "target": "tran_multiplier"}},
                {"name": "k", "family": "LogPermField", "args": {"n_components": 3, "range_m": 1500.0}}],
 "observations": {"wells": ["P1", "P2", "P3"], "quantities": ["oil_rate"], "report_times": [20, 40, 60]},
 "design": {"method": "lhs", "n": 8}}
```

```json
{"name": "hm", "workflow": "hm-esmda", "seed_root": 21,
 "model": {"model_dir": "models/Uniform_Brugge", "adapter": "workflows.adapters.brugge_proxy"},
 "parameters": [{"name": "k", "family": "LogPermField", "args": {"sigma_log10": 0.3, "range_m": 1500.0}}],
 "observations": {"wells": ["P1", "P5", "P10"], "quantities": ["oil_rate"], "report_times": [20, 40, 60, 80, 100],
                  "held_out_fraction": 0.2, "sigma_rel": 0.05},
 "design": {"ne": 6, "n_steps": 2, "localization_length_m": 1500.0}}
```

```json
{"name": "opt", "workflow": "optimize", "seed_root": 31,
 "model": {"model_dir": "models/Uniform_Brugge", "adapter": "workflows.adapters.brugge_proxy"},
 "parameters": [{"name": "I1", "family": "ScalarParam", "args": {"target": "well_xyz"}}],
 "observations": {"wells": ["P1", "P2", "P3"], "quantities": ["oil_rate"], "report_times": [20, 40, 60]},
 "design": {"driver": "exhaustive", "well": "I1", "objective": "cumulative_oil", "min_spacing_m": 300.0, "max_candidates": 4}}
```

Notes on these specs: `estimate` for `hm-esmda` counts assimilation runs only, `truth` adds one
simulation; in `optimize`, only `cumulative_oil` is implemented and the `parameters` entry names
the relocated well to satisfy the schema (the driver reads `design.well`). Sizes for production
use rather than smoke tests: `n_base` a power of two of at least 128 for Saltelli (LHS and Sobol use `n`), `ne` of 50 to 100
with localization for ES-MDA, and the full feasible set (omit `max_candidates`) for placement.
Always read the `estimate` line before `run`.

## 3. Running under different hosts

Claude Code and Codex CLI are verified below; any other host with function calling needs only
three abilities: write a file, run `python -m workflows <command>`, read the JSON line.

### 3.1 Claude Code

Skills are discovered from `.claude/skills` (the mirror of `.agents/skills`; keep it in sync with
`helper_scripts/sync_agent_skills.py`). `AGENTS.md` routes tasks to the three skills.

- Interactive: open the repository in Claude Code and ask in plain language ("history-match the
  Brugge proxy to P1, P5, P10 oil rates ..."); the skill loads by description, or invoke it
  explicitly with `$history-matching-open-darts`.
- Headless (what the evaluator uses):

```bash
claude -p "<task>" --output-format json --model claude-sonnet-4-6 \
  --max-turns 40 --max-budget-usd 5 --permission-mode default \
  --allowedTools "Bash,Read,Write,Edit,Glob,Grep,Skill" --setting-sources project
```

The JSON result carries `usage` (tokens by kind), `total_cost_usd`, `num_turns`, `duration_ms`
and the final text; the evaluator records the token counts, cost, turns and text from it and
measures wall, CPU and peak RSS itself inside the sandbox (`duration_ms` is not recorded).

### 3.2 Codex CLI

Codex reads `AGENTS.md` at the repository root and discovers the repository skills in
`.agents/skills` (Codex CLI 0.153, asked for its available skills from the repository root, lists
the three study skills next to its built-ins). No conversion is needed.

- Interactive: run `codex` from the repository root and ask as above.
- Headless: `codex exec --json -o last.txt -m <model> -C <repo> "<task>"` prints JSON events
  (a `turn.completed` event per user message with token counts; Codex reports no cost) and
  writes the final answer to `last.txt`. Choose Codex's own isolation with `--sandbox`, or run
  inside the evaluator's `bwrap` sandbox where `danger-full-access` is safe.
- Evaluator: `--runner codex --model <model>` selects `CodexCliRunner`, which binds
  `~/.codex/auth.json` and `config.toml` read-only into the scratch HOME and records tokens,
  turns (completed items, comparable with the Claude count), wall, CPU and peak RSS;
  `total_cost_usd` stays empty. Evidence: the placement smoke case passed through Codex with the
  same skills (design Appendix C.2).

### 3.3 A plain external LLM (no tools)

Paste the relevant `SKILL.md` and the references it names as the system prompt, plus
`workflows/docs/study-substrate.md`. Ask the model for the study spec and the report; a human
or a thin script runs the CLI commands and feeds the JSON lines back. The invariants still
apply: `estimate` before `run`, no simulations outside the executor, report the gate values.

## 4. What is evaluated (`workflows/evals/`)

The evaluator runs an agent on **hidden cases** and scores it after exit, so the agent never
sees the answer. Isolation is OS-level (`bwrap`; mount manifest in `workflows/evals/README.md`):
the repository, the environment and the CLI's install prefixes (`claude`/`node` or
`codex`/`node`) are read-only; the truths, cases and scorer are masked; the agent writes only
to its run directory and the broker's socket directory. The agent never runs simulations: it
submits a spec to the broker (`WORKFLOWS_BROKER`) with

```bash
python -m workflows.evals.broker --spec study.json --study <case> --command estimate|truth|run|analyze
```

and the parent runs the study into a results directory the agent sees read-only
(`WORKFLOWS_RESULTS`). After exit the results are re-hashed and compared with the record.

Cases (`workflows/evals/cases/*.json`, smoke tier, Brugge proxy):

| Case | Workflow | Reference (hidden) | Gate |
|---|---|---|---|
| `opt-place-proxy-smoke` | optimize | exhaustive optimum, seed 31, 4 candidates | normalized regret of the agent's best candidate ≤ 0.05 |
| `ens-proxy-smoke` | ensemble | LHS 8, seed 11 percentiles with bootstrap intervals | P10/P50/P90 per producer within 5 % plus the interval half-width |
| `hm-proxy-twin-smoke` | hm-esmda | twin truth, seed 21, `ne` 6, 2 steps | held-out normalized RMSE ≤ 1.25 × reference, 80 % coverage ≥ 0.6, spread ratio ≥ 0.1 |

Run one case and score it:

```bash
PYTHONPATH=<repo> <env>/bin/python -m workflows.evals.controller \
  --case workflows/evals/cases/opt-place-proxy-smoke.json --model claude-sonnet-4-6 --run-root /path/run1
PYTHONPATH=<repo> <env>/bin/python -m workflows.evals.report      # all records, with aggregates
```

Each run writes `workflows/evals/runs/<case>/<utc>-<model>.json` with: pass/fail and the gate
values, tokens by kind, cost, turns, wall, CPU and peak RSS (measured inside the sandbox), the
broker requests with spec hashes, the git head, the network statement, and the agent's final text.

## 5. The self-improvement loop

The loop changes skills or harness code, never the cases or their prompts.

1. Run the three smoke cases on the current head (one command each, about 2 to 5 minutes and
   USD 0.3 to 0.6 per run with `claude-sonnet-4-6`).
2. Read the record: a failed gate, extra broker requests (wasted studies), a high turn count,
   or a wrong statement in the final text each point at a skill sentence or a harness output.
3. Edit the skill (`.agents/skills/...`, then `helper_scripts/sync_agent_skills.py` and
   `helper_scripts/validate_skills.py`) or the substrate (`workflows/`, with tests).
4. Rerun the same cases with the same model and seeds; compare with `report`.
5. Accept only under the rule in `workflows/evals/README.md` (pass count not lower, median cost
   and turns not higher, ties by fewer broker requests); otherwise reject and keep the record.

Metrics, in order of authority: pass/fail with the gate value (regret; held-out RMSE, coverage,
spread; percentile checks), median cost and turns (the rule's comparators), broker requests
(the tie-breaker), tokens, wall, CPU, peak RSS.

Parameters you can vary, and what they test:

| Parameter | Where | What it varies |
|---|---|---|
| `--model` | controller | the agent (exact id; records are grouped by model) |
| `--runner` | controller | `claude` (`ClaudeCliRunner`) or `codex` (`CodexCliRunner`); any other host implements `AgentRunner.run(prompt, sandbox, env, budget)` |
| `budget.max_turns`, `budget.max_usd` | case file | how much the agent may spend before being cut off |
| `seeds`, `design` sizes (`n`, `ne`, `n_steps`, `max_candidates`) | case + reference | the study's difficulty and cost; changing them needs a new reference |
| `tolerance` | case file | gate strictness (fixed before any run in a tier) |
| `compute` (`max_workers`, `threads_per_member`, `walltime_s`) | study spec | throughput only; a thread change is journaled as a non-equivalent attempt |

Adding a case: write the spec, run it through the CLI in the parent to produce the reference,
store the resolved targets as `workflows/evals/truth/<case>/reference.json` (use
`workflows.evals.scorer.reference_from_study`), write `cases/<case>.json` with a prompt that
tells the agent to submit through the broker (placeholders `{repo}`, `{python}`, `{run_dir}`),
and confirm the reference study scores against itself.

Evidence to date: three rounds on the three cases with Claude Code and one cross-host round with
Codex; three harness gaps and two skill gaps were found by the loop itself and fixed. Run-by-run
numbers are in Appendix C.2 of the design page.
