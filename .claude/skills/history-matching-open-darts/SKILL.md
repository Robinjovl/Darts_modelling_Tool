---
name: history-matching-open-darts
description: History-match open-DARTS models to well data (rates, BHP, temperature) with ES-MDA (dageo) or adjoint gradients using the in-repo workflows package; twin-test truths, observation conventions, localization, and validity gates (held-out RMSE, coverage, spread). Use for calibration, data assimilation, inversion, or matching production history.
---

# History matching

Use this skill to calibrate uncertain model inputs against observed well data.
Preserve existing file line endings; never make line-ending-only changes.

## Invariants

- Invoke only `PYTHONPATH=<repo> <env>/bin/python -m workflows ...` (`workflows/docs/study-substrate.md`).
- Observations are instantaneous well values at report times; interval averages
  (`time_data_report`) are a different quantity and must not be mixed in.
- Declare noise (`sigma_abs`/`sigma_rel`) before running; hold out the last part of the history
  (`held_out_fraction`, default 0.2) and never tune against it.
- Judge success by the gates (held-out RMSE, 80 % coverage, spread ratio, chi² band), not by
  the training misfit alone. Recovery of the truth is secondary and only meaningful in the
  identifiable subspace.
- One process per simulation; `estimate` before `run`; never read `workflows/evals/truth/`.

## Steps

1. Read `references/method-selection.md` and choose ES-MDA (implemented) or adjoint (spike only).
2. Read `references/observations-and-gates.md`; define wells, quantities, report times, noise.
3. Twin test first when the real data is not yet available: `truth` generates a hidden truth
   from the prior with a disjoint seed, then `run` assimilates it.
4. Write `study.json` (`workflow: "hm-esmda"`), read `references/esmda.md` for `design`
   (`ne`, `n_steps`, `alphas`, `localization_length_m`, `bounds_sigma`), `estimate`, `run`.
5. Report per step: chi², held-out RMSE, coverage, spread ratio; state what remained
   unidentified.

## Commands

```bash
PYTHONPATH=<repo> <env>/bin/python -m workflows estimate --spec study.json
PYTHONPATH=<repo> <env>/bin/python -m workflows truth    --spec study.json --study runs/<name>
PYTHONPATH=<repo> <env>/bin/python -m workflows run      --spec study.json --study runs/<name>
```

Adjoint gradients: `references/adjoint-hm.md`; the reproducible check is
`workflows/evals/spikes/brugge_adjoint_check.py`.
