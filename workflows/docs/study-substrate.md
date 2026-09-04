# Study substrate (`workflows/`): the one reference the three study skills share

Everything a study needs lives in the repo-root package `workflows/` (no install step).

## Invocation (pinned; never bare `python` or `darts`)

```bash
cd <repo>
PYTHONPATH=<repo> <env>/bin/python -m workflows estimate --spec study.json
PYTHONPATH=<repo> <env>/bin/python -m workflows run      --spec study.json --study <dir>
PYTHONPATH=<repo> <env>/bin/python -m workflows truth    --spec study.json --study <dir>   # hm-esmda twins
PYTHONPATH=<repo> <env>/bin/python -m workflows analyze  --spec study.json --study <dir>   # ensemble
```

`<env>` is the session conda environment that has open-DARTS built or installed plus
`workflows/requirements.txt`. Each command prints one JSON line. `run` resumes a study by
default (journaled members are skipped); `--no-resume` restarts.

## Study spec (`study.json`, schema `workflows/spec.py:STUDY_SCHEMA`)

| Field | Meaning |
|---|---|
| `name`, `workflow` | `ensemble`, `hm-esmda`, `optimize` are implemented; `hm-adjoint` is reserved (spike only) |
| `model` | `{"model_dir", "adapter", "module"="model", "cls"="Model", "kwargs"}`; adapter = dotted module exposing `Adapter(ModelAdapter)` |
| `parameters` | list of `{"name", "family", "args"}`; families in `workflows/members.py`: `ScalarParam(low, high)`, `LogScalarParam`, `MultiplierField(n, sigma_log10, groups)`, `LogPermField(mean_log10, sigma_log10, range_m, n_components)`; `args.target` selects the realization key (`tran_multiplier`, `wi_multiplier`, `poro`, `bhp`, `permx`, `well_xyz`) |
| `seed_root` | one integer; all streams derive from it (`SeedLedger`: design, geology, observation_noise, assimilation, optimizer, surrogate, agent) |
| `observations` | `wells`, `quantities` (`oil_rate`, `wat_rate`, `gas_rate`, `bhp`, `bht`), `report_times` (days), `held_out_fraction`, `sigma_abs`/`sigma_rel`/`sigma_floor` |
| `design` | workflow-specific (see the skill); `reference` optional; `compute` = `platform`, `walltime_s`, `max_workers`, `threads_per_member`, `retries`, `memory_per_member_gb` |

Identities: `workflow_run_hash` (spec + engine fingerprint), `truth_generation_hash`
(adds provenance), `score_version`; all recorded in `manifest.json`.

## Adapters (`workflows/adapter.py`)

One `ModelAdapter` per model: `input_patterns`, `rebuild_scope` (which realization keys need a
fresh build), `member_seconds`, `build(paths, realization)`, `apply(model, realization)`,
`run(model, report_times)`, `observe(model, observation)`. Shipped: `workflows.adapters.brugge_proxy`
for `models/Uniform_Brugge` (431 cells, ~10 s per member, BHP-controlled wells). For any other model
write an adapter first and prove it with one member before designing a study.

## Execution model (invariants, not tunables)

- One fresh process per simulation (`IsolatedExecutor`): perm/relperm/PVT edits are not safe in-place.
- Inputs are snapshotted read-only once (`inputs/`) and staged per member by symlink; the member
  writes only inside `members/<id>/`.
- Lean outputs: no VTK, no spreadsheets, `store_well_time_data(save_output_files=False)`.
- Walltime kill + retries with distinct attempt ids; a retry with more threads is a
  non-equivalent attempt and is journaled as such.
- `journal.jsonl` is append-only; `manifest.json` holds identities, provenance, seeds, design.
- Observations: instantaneous well rows at report times in the simulator's sign convention
  (production rates are negative on the Brugge proxy; `cumulative_production` flips the sign);
  `time_data_report` averages over intervals and is a different quantity.
- Cost gate: run `estimate` and report planned simulations, wall time and disk before `run`.
- Launch studies through the CLI or a script file, never from stdin or an interactive session:
  the executor spawns workers that re-import the main module.
- Never read `workflows/evals/truth/` or any file named as a truth; results are compared by the evaluator.

## Outputs

| Workflow | Files under `--study` |
|---|---|
| all | `spec.json`, `manifest.json`, `journal.jsonl`, `inputs/`, `members/<id>/result.json` |
| ensemble | `design_points.npy`, `analysis.json` (percentiles with bootstrap CI, Morris/Sobol indices when designed) |
| hm-esmda | `truth/truth.json` (twin only), `esmda/step_<k>_{params,data}.npy`, `esmda_summary.json` (per-step chi², held-out RMSE, coverage, spread ratio) |
| optimize | `candidates.json`, `optimize_summary.json` (baseline, best, `regret_of_baseline`) |

Fast tests: `PYTHONPATH=<repo> <env>/bin/python -m unittest discover -s workflows/tests -t .`;
simulator-backed tests add `WORKFLOWS_RUN_DARTS=1`.
