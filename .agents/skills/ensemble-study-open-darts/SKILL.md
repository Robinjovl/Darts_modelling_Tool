---
name: ensemble-study-open-darts
description: Design, run, and analyze ensemble, sensitivity, and uncertainty-quantification studies of open-DARTS models with the in-repo workflows package (LHS, Sobol, Morris, Saltelli designs; isolated per-member runs; journaled resume; bootstrap percentiles and sensitivity indices). Use for P10/P50/P90, parametric sensitivity, screening, or any request to run many realizations of a model.
---

# Ensemble study

Use this skill when a question needs many simulations of one model under varied inputs.
Preserve existing file line endings; never make line-ending-only changes.

## Invariants (do not trade these for speed)

- Invoke only `PYTHONPATH=<repo> <env>/bin/python -m workflows ...` (see `workflows/docs/study-substrate.md`).
- One process per simulation, snapshotted inputs, lean outputs; the executor enforces this — do not
  loop `model.run()` in one process to "save time".
- Run `estimate` and state planned simulations, wall time, disk **before** `run`.
- Failed members are journaled and never resampled; report failures with their class.
- Report P10 as the 10th percentile (statistical convention) with its bootstrap interval.
- Never read `workflows/evals/truth/`.

## Steps

1. Read `references/experimental-design.md`; pick the design from the question (screening →
   Morris; variance decomposition → Saltelli; distribution/percentiles → LHS or Sobol).
2. Read `references/priors-and-parameterization.md`; choose families and ranges, prefer log
   scales for permeability and multipliers; keep the dimension small enough for the budget.
3. Write `study.json` (`workflow: "ensemble"`, `design: {"method", "n", "second_order"?}`),
   then `estimate`; adjust `n` (power of two for Sobol/Saltelli) or the parameter set.
4. `run`, then `analyze`; if interrupted, rerun `run` (resume is the default).
5. Report per `references/analysis-and-reporting.md`.

## Commands

```bash
PYTHONPATH=<repo> <env>/bin/python -m workflows estimate --spec study.json
PYTHONPATH=<repo> <env>/bin/python -m workflows run     --spec study.json --study runs/<name>
PYTHONPATH=<repo> <env>/bin/python -m workflows analyze --spec study.json --study runs/<name>
```

Only `models/Uniform_Brugge` has an adapter today (`workflows.adapters.brugge_proxy`); other
models need a `ModelAdapter` first (`workflows/adapter.py`).
