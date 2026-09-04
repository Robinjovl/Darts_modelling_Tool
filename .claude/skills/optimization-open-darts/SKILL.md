---
name: optimization-open-darts
description: Optimize open-DARTS well placement and controls (EOR, gas/CO2 injection, geothermal) with the in-repo workflows package; exhaustive placement (implemented), adjoint and derivative-free drivers, GBT surrogates, robust objectives, feasibility rules, regret gates, post-validation. Use for well placement, injection/production strategy, NPV or recovery maximization, or surrogate-assisted search.
---

# Optimization

Use this skill to maximize a simulated objective over placement or control decisions.
Preserve existing file line endings; never make line-ending-only changes.

## Invariants

- Invoke only `PYTHONPATH=<repo> <env>/bin/python -m workflows ...` (`workflows/docs/study-substrate.md`).
- Define the objective, constraints and feasibility rules before any run; report regret
  against the baseline (and the reference when one exists), not the raw objective alone.
- Every proposed optimum is re-simulated in the substrate (post-validation); surrogate values
  are never reported as results.
- Cumulative quantities are trapezoids of instantaneous rates at report times; make the report
  grid fine enough (≤ 1/10 of the horizon).
- One process per simulation; `estimate` before `run`; never read `workflows/evals/truth/`.

## Steps

1. Read `references/problem-formulation.md`; write down objective, decision variables,
   constraints, horizon, report times, budget.
2. Placement: read `references/well-placement.md`; `design: {"driver": "exhaustive", ...}` is
   implemented. Controls, gradients, surrogates, robust objectives: read `references/drivers.md`
   for status and protocol before promising anything.
3. Write `study.json` (`workflow: "optimize"`), `estimate`, `run`.
4. Report best decision, objective, regret of the baseline, feasibility summary, cost.

## Commands

```bash
PYTHONPATH=<repo> <env>/bin/python -m workflows estimate --spec study.json
PYTHONPATH=<repo> <env>/bin/python -m workflows run      --spec study.json --study runs/<name>
```
