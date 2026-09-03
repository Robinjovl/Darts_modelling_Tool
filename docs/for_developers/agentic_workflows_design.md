---
orphan: true
---

# Agentic workflow skills: ensemble modeling, history matching, optimization

Design draft, version 4 (2026-09-03). Baseline revision analyzed: `59bda10b` on branch
`avnovikov/skills_and_workflows`; the commit introducing this document supersedes that
baseline. Status: **draft for review, not accepted**. The page is kept out of the documentation
toctree until approval. Two review rounds and the owner's direction of 2026-09-03 (no separate
package; keep everything inside this repository, in the skills folders and a new `workflows/`
folder, with the smallest possible interface set) are recorded in Appendix E. No workflow
implementation exists; this document proposes capabilities.

## 1. Summary

This document proposes three agent skills (ensemble studies, history matching, optimization
incl. well placement) under `.agents/skills/`, as thin guides over one repository-local Python
package, `workflows/`, that turns a `DartsModel` into a **study**: a reproducible, journaled,
process-isolated fan-out of simulations with typed parameterizations, an observation contract,
validity gates and a cost estimate. Methods (adjoint, ES-MDA, DiWA, derivative-free and
surrogate optimizers) are drivers inside the same package. The evaluation controller, hidden
truths, scorer and feasibility spikes live in `workflows/evals/`. Nothing is packaged or
installed separately: the package is imported through the already mandatory
`PYTHONPATH=<repo>` and is tied to the open-darts build of the same checkout. The first
milestone (Section 12) is two CI models and three zero-agent vertical slices.

| Decision (v4) | Section |
|---|---|
| One repository-local package `workflows/` (about a dozen modules); no separate distribution, no `pyproject.toml`; dependencies in `workflows/requirements.txt` installed into the `skills` env; graduation to a distributable package deferred | 6.1 |
| Three skills (ensemble, history matching, optimization); the evaluation loop is a developer tool documented in `workflows/evals/README.md`, not a skill | 7 |
| ES-MDA contract: diagonal observation covariance only in the first implementation (stated); native full-covariance update later; adjoint history matching diagonal only | 6.3 |
| Evaluator: the agent submits specs to a parent-side broker that runs simulations and owns records; mount manifest with truths and scorer masked; canary; read-only credential bind; post-exit hash verification; network acknowledged as unrestricted unless a proxy is deployed | 10 |
| ES-MDA milestone specified (bounds, transforms, anisotropy, regions, collapse diagnostics, scaling); PCA projection experimental; twin test pre-registered | 6.3, 12 |
| Sobol: retry-then-preserve failure policy, no row replacement, base size 512; reference ensembles with intervals; P10 is the 10th percentile; hierarchical seed ledger; thread-varying retries are non-equivalent attempts | 6.2, 9 |
| Full-field Brugge is a local candidate bed, not CI; the fraternal twin is a hypothesis until a cross-model mapping exists | 4.2, 9 |

## 2. Scope

| Workflow | User request | Methods in scope |
|---|---|---|
| Ensemble modeling | run a set or ensemble of input parameters (porosity/permeability fields, relative permeability, other reservoir and fluid parameters); parametric, sensitivity and UQ studies | LHS and Sobol designs, Morris screening, Sobol indices (first and total order; second order optional and costed), scalable geostatistical priors with PCA/KL reduction |
| History matching | constrain model parameters to wellbore rates, BHP and temperature | built-in adjoint gradients (`OptModuleSettings`), DiWA proxy (MR!330 branch after rebase; private scaffold pending provenance review), ES-MDA via `dageo` (diagonal covariance) and a planned native update (full covariance) |
| Optimization | maximize/minimize a simulation-derived quantity, incl. well placement, gas-injection EOR and CO2 storage | adjoint gradients where the engine provides them (transmissibility, well index), parallel finite differences, CMA-ES/PSO/pattern search, surrogate-assisted search with full-physics validation, exhaustive search for small discrete spaces |

Out of scope: seismic observations, geomechanics adjoints, GPU ensemble scheduling (deferred
until measured), combined CO2-EOR (no compositional CO2 + hydrocarbon bed), any installable or
public API for the studies layer at this stage.

## 3. Principles

| # | Principle | Concrete consequence |
|---|---|---|
| P1 | Workflows before agents | every simulation, update step and gate is deterministic code in `workflows/`; the agent chooses strategy, reads results, writes the report |
| P2 | Tools are contracts | JSON specs validated against schemas; structured results and failure classes; expensive fan-outs require an estimate first |
| P3 | Externalized, idempotent state | study directory with `study.json`, `manifest.json`, `journal.jsonl`; identities `truth_generation_hash`, `workflow_run_hash`, `score_version`; a hierarchical seed ledger; commit and dependency lock recorded per run |
| P4 | Process isolation is structural | one OS process per simulation; walltime, kill, retry with distinct attempt identity; immutable input staging per worker |
| P5 | Trusted evaluation | during evaluation the agent never executes simulations itself; a parent-side broker runs them and owns the records; truths and scorer are unreachable; every hash is verified after exit |
| P6 | Reproducibility over convenience | pinned interpreter and `PYTHONPATH`, seed tree, per-worker directories, engine binary fingerprint and input hashes in the truth identity |
| P7 | Reservoir-engineering discipline as invariants | drivers enforce invariants (preserve declared dependence, validate convergence, hold out data, no truth peeking, baseline before optimization); numerical settings are heuristics with stated ranges |
| P8 | Measure everything | simulations, CPU and wall time, RSS, iterations, failure classes, attempt identities; tokens, turns, cost, permission denials |
| P9 | Smallest interface set | one package, one CLI entry point, one adapter protocol, one spec schema family; new modules only when a milestone needs them |

## 4. Verified facts

### 4.1 open-DARTS API and environment

| Topic | Fact | Evidence | Consequence |
|---|---|---|---|
| Cheap knobs | well controls (no reset), initial state, `mesh.set_res_tran`, `mesh.set_wells_tran`, `mesh.poro`/depth/volume in place, solver specs, timestepping; then `model.reset()` | `darts/models/darts_model.py`, `engines/src/pybind11/py_mesh_conn.cpp` | adapter "apply" path |
| Rebuild-only knobs | permeability to transmissibility/WI (WI baked at `add_perforation`), well topology, rel-perm/PVT, OBL axes, platform | `darts/reservoirs/*.py`, `darts/physics/base/physics.py` | fresh reservoir or physics per member |
| Brugge proxy construction | `set_reservoir()` regenerates the gmsh mesh on every `Model()` into a relative path and reads `Brugge_struct/` relatively; the constructor takes no arguments | `models/Uniform_Brugge/model.py`, `mesh_creator.py` | adapter refactor: input snapshot, absolute paths, mesh reuse, physics-only rebuild |
| CPG permeability update | `CPG_Reservoir.update_perm()` references an unset attribute and does not push transmissibilities to the mesh | `darts/reservoirs/cpg_reservoir.py` | forbidden; rebuild the reservoir |
| OBL cache | on-disk key hashes evaluator class and grid signature, not PVT/rel-perm content | `physics.py`, `tools/obl_cache.py` | `cache=False` when rel-perm/PVT vary; realization-scoped `cache_dir` otherwise |
| Observables, forward path | `store_well_time_data()` returns one row per accepted adaptive timestep; pkl/xlsx writing dominates its cost | `darts/models/output.py`; measured | `save_output_files=False`; rows matched by time with a tolerance |
| Observables, adjoint path | `objfun_assembly` compares instantaneous `engine.time_data` rates at Dirac-selected report times against the observation frame; `time_data_report` is interval-averaged; weights are pointwise (diagonal) | `opt_module_settings.py`; spike | adjoint observations are instantaneous truth rows at report times, production negated; adjoint history matching supports diagonal covariance only |
| Adjoint controls | gradients for transmissibility and well index (and MPFA flux multipliers); other modifiers by forward FD | `opt_module_settings.py` | control optimization is FD or derivative-free |
| Adjoint on the proxy | assertive check passes: Appendix C | `workflows/evals/spikes/` | eligible bed; angle gate stays mandatory |
| Adjoint traps | `engine_nc_nl_cpu` returns a silently wrong gradient; thermal assembly incomplete; `transmissibility_modifier` overwrites `tranD`; forward crash calls `sys.exit()` | engine sources, `opt_module_settings.py` | engine guard, thermal warning, `tranD` preservation, subprocess |
| MR!330 | open with conflicts; 17 commits and 87 changed files relative to `development`; `git merge-tree` at `59bda10b` reports seven conflicting paths (`CHANGELOG.md`, `darts_model.py`, `thmc_model.py`, three `nonlinear_solvers` files, `engine_base.cpp`); C++/pybind only; `engine_super_cpu` only; no Python glue | `git merge-tree`, GitLab MR !330 | fresh rebase and verification required before DiWA work; time-sensitive |
| Solver stats | `nonlinear_solver.stats` exposes total and wasted timesteps, Newton and linear iterations | `darts/nonlinear_solvers/base.py` | journaled |
| Headless accounting | `claude -p --output-format json` returns usage, cost, turns, duration, per-model usage, session id, permission denials; aliases resolve loosely; credentials live in `~/.claude/.credentials.json` | measured | pin exact model ids; bind credentials read-only into the sandbox |
| Sandboxing | `bwrap`, `podman`, `docker`, `unshare`, `systemd-run` present; unprivileged user namespaces work | measured (also by the reviewer) | OS-level isolation without root |
| Lint and docs tooling | Ruff and pre-commit exist outside the `skills` env (`mambaforge/bin`, 0.13.1; CI pins 0.12.11); Sphinx absent from `skills`; the pre-commit Ruff hooks now include `workflows/` | `.pre-commit-config.yaml` (this change) | lint the new tree; docs build needs the docs extra |
| Environment | no `pyyaml`, `dageo`, `xgboost`, `catboost`, `optuna`, `SALib`, `gstools`, `pymoo`, `cma`, `shap`, `zarr`, `pydantic`, `scikit-learn` in `skills`; PyPI reachable; 128 affinity cores, no cgroup CPU quota | measured | JSON specs; `workflows/requirements.txt`; executor sizing from affinity and memory |

### 4.2 Test beds

| Bed | Grid | Wells | Physics | Cost | Role |
|---|---|---|---|---|---|
| Uniform_Brugge (CI, coarse proxy) | 431 wedge cells, one 72 m layer, 611 interfaces, 30 well indices | 10 gas injectors (BHP 180 bar), 20 producers (BHP 150 bar) | three-phase black-oil PVT tables | lean member 8-12 s wall (median 10 s), 0.8 GB, 3 MB disk (2000 days) | primary MVP bed |
| GeoRising (CI) | 60x60x3, Van Essen permeability field | INJ/PRD doublet | IAPWS PT or PH | 2.4-3.6 s simulation, 18 report steps | second MVP bed: ES-MDA twin, doublet placement |
| Full-field Brugge (local candidate, not CI) | `cpg_sloping_fault` case `brugge_wbhp`: 139x48x9 corner-point, 43,847 active cells; inputs are an untracked local 8.8 MB directory `meshes/brugge` | `PRD_12`, `INJ_12` (layers 1-9); no 30-well schedule for this grid | dead-oil, geothermal, CO2-water variants via the case harness | dead-oil 39 s wall, 0.96 GB; geothermal with burden layers 84,150 blocks, 74 s, 1.27 GB (single local runs) | deferred until license, provenance, storage and CI availability are resolved |
| 3ph_bo (CI) | 10x10x3 | I1/P1 | black-oil | 5 s wall | smoke tier |

Environment notes (not part of the design): 128 cores at load 20-60, 1 TB RAM, two A100 (one
busy), `OMP_NUM_THREADS=1`.

### 4.3 Repository conventions

Skills live in `.agents/skills/<name>/` and are mirrored to `.claude/skills/` by
`helper_scripts/sync_agent_skills.py`; `validate_skills.py` is referenced but missing;
`AGENTS.md` holds the mandatory-usage table; the `verify-open-darts` skill triggers on
`darts/`, `models/`, `helper_scripts/`, packaging and CI files (to be extended to
`workflows/`). `*.txt`, `*.pkl`, `*.h5`, `*.npy`, `*.png`, `*.log` are gitignored.

```bash
PYTHONPATH=/oahu/data/avnovikov/open-darts-skills \
  /oahu/data/avnovikov/mambaforge/envs/skills/bin/python -m workflows <command> --spec study.json
```

## 5. Architecture overview

```
+---------------------------- Agent harnesses --------------------------------+
| .agents/skills (source of truth) . .claude/skills mirror . AGENTS.md table   |
| project settings (PostToolUse sync+validate) . AgentRunner: ClaudeCliRunner |
+----------------------------- Skills (3) ------------------------------------+
| ensemble-study-open-darts . history-matching-open-darts .                   |
| optimization-open-darts   (+ model-workflow-open-darts reference filled)    |
|   SKILL.md (invariants inline) -> one method reference -> thin CLI wrappers |
+------------------------------ workflows/ (in-repo package) -----------------+
| spec . adapter . members . executor . journal . gates . cost                |
| ensemble . esmda . adjoint . optimize . cli                                 |
| adapters/ (brugge_proxy, georising) . docs/study-substrate.md . tests/      |
| evals/ (controller, broker, scorer, cases, hidden truth, spikes, README)   |
+---------------------------- open-DARTS (existing) --------------------------+
| DartsModel . Output . reservoirs . OptModuleSettings . engine bindings      |
+-----------------------------------------------------------------------------+
```

## 6. The deterministic layer: `workflows/`

### 6.1 Placement, dependencies, verification

- `workflows/` is a plain package at the repository root (`workflows/__init__.py`), imported
  as `workflows.*` through the mandatory `PYTHONPATH=<repo>`; it is not matched by the wheel's
  `darts*` include and is not installed separately. It runs only against the open-darts build
  of the same checkout, so no version range is declared; the engine binary fingerprint is still
  recorded in every manifest. Graduation to a distributable package is deferred.
- Dependencies: numpy, scipy, pandas, h5py, jsonschema (present). Optional groups listed in
  `workflows/requirements.txt` with comments per workflow (`dageo`; `SALib`; `gstools`;
  `scikit-learn`, `xgboost`, `catboost`, `optuna`, `shap`; `cma`, `pymoo`), installed into
  the `skills` env with `pip install -r workflows/requirements.txt`; optional imports raise a
  `MissingDependency` error naming the package.
- Verification: the pre-commit Ruff hooks cover `workflows/` (this change); the
  `verify-open-darts` trigger list gains `workflows/`; `workflows/tests/` has a fast lane
  without the simulator and a darts-gated lane on the smoke beds, added to the existing model
  test job in `.cicd/jobs/` rather than a new pipeline.
- Model inputs stay where they are (`models/<bed>/`); adapters in `workflows/adapters/`
  reference them by absolute path from the checkout. Nothing is copied into the package.

### 6.2 Modules (the whole package)

| Module | Responsibility |
|---|---|
| `spec.py` | dataclasses and JSON schemas for `ModelRef`, parameters, observations, `StudySpec`, `RunResult`; the three identities; the seed ledger (`numpy.random.SeedSequence` tree with named substreams: design, geology, observation noise, assimilation perturbations, optimizer initialization, surrogate sampling, agent repetition; common random numbers share only intended substreams) |
| `adapter.py` | `ModelAdapter` protocol (`snapshot_inputs`, `stage`, `build`, `apply`, `run`, `observe`, `cost_hint`) with `PROTOCOL_VERSION`; input snapshot with hashes; per-worker staging by reflink, copy or read-only symlink; absolute paths; declared rebuild scope per parameter family |
| `members.py` | parameter families (`ScalarParam`, `LogScalarParam`, `MultiplierField` per region or connection for tran/WI and per cell for poro/depth/volume, `LogPermField` with declared bounds, fixed anisotropy ratio and porosity coupling, `RelPermParams`, `WellIndexParams`; `perm_to_tran_wi_reference()`), random fields on cell centroids (gstools; `dageo.RandomPermeability` only for 2-D smoke), and observations (`ObservationSpec`, extraction by time matching on both paths, `NoiseModel`, positions) |
| `executor.py` | isolated executor ported from agentic-clrm (spawn per task, walltime, kill, `/proc` diagnostics); worker count bounded by `os.sched_getaffinity`, cgroup quota when present and memory, load average as a secondary throttle; retries with a different thread count get a distinct attempt identity and are reported as non-equivalent |
| `journal.py` | study layout, atomic manifests, append-only journal; every record carries spec hash, input hash, binary fingerprint and attempt identity |
| `gates.py` | versioned pure functions: `held_out_residuals`, `predictive_coverage`, `spread_ratio`, `rank_diagnostics`, `chi2_band` (only with a declared covariance and effective degrees of freedom), `gradient_angle`, `regret`, `budget_exceeded`, `convergence_stall`; scientific tolerances separate from the replay tolerances of `cicd_tools.check_performance` |
| `cost.py` | `estimate(spec)`: startup and initialization per member, planned members x steps with multipliers: Saltelli `N x (D + 2)` (first and total order) or `N x (2D + 2)` (with second order), `N` a power of two; central FD `2 x dim` per gradient plus line-search evaluations as a range; ES-MDA `(n_steps + 1) x ne`; robust `candidates x ne`; expected failed attempts and post-validation; wall time as a range and CPU-hours separately |
| `ensemble.py` | LHS and Sobol designs, Morris trajectories, Saltelli matrices with an explicit second-order flag; fan-out; P10/P50/P90 (statistical percentiles) with bootstrap intervals, Morris and Sobol indices reported as conditional on screening; Sobol failure policy: retry the identical sample, preserve it as failed, then fail the conventional analysis or invoke a failure-aware estimator; never resample rows |
| `esmda.py` | ES-MDA over `dageo.esmda` with diagonal covariance (Section 6.3); planned native update with full or block-diagonal covariance and localization on the parameter-data covariance |
| `adjoint.py` | driver over `OptModuleSettings` in a subprocess: verified observation convention, per-family scaling, bounds, regularization, mandatory angle gate, `tranD` preservation, checkpoints, RML ensembles perturbing prior anchor and data, engine guard, thermal warning; DiWA Stage A after the MR!330 rebase |
| `optimize.py` | problem definition (fields used by the cases), objectives (cumulative production, recovery factor, gas utilization, NPV with declared assumptions, net energy, CO2 injected and pressure footprint for CO2-water physics), candidate mapping with feasibility checks before and after snapping and `mapped_design_hash`; drivers: exhaustive, central-difference controls with L-BFGS-B, adjoint well-index completion, CMA-ES/PSO/pattern search, surrogate-assisted (baselines, cross-validated selection among scikit-learn GPs and tree ensembles, untouched test set, uncertainty-aware acquisition, envelope clipping, re-validation); robust evaluation over a fixed realization subset with common random numbers; post-validation. Definitions: normalized regret `(J_ref - J) / max(J_ref - J_baseline, eps_abs, eps_rel x |J_ref|)` with an "uninformative benchmark" outcome below the floor; CVaR(0.2) as the mean of the worst 20 %; failed simulations infeasible with a declared policy; relaxed WI mapped to a discrete completion by thresholding and re-evaluation; injection compositions through an additive log-ratio transform |
| `cli.py` | `python -m workflows <estimate|run|analyze|hm-esmda|hm-adjoint|optimize|postvalidate|truth> --spec study.json [--resume]`; one JSON summary line on stdout; child output contained in the study log |
| `adapters/` | `brugge_proxy.py` (with the small `models/Uniform_Brugge/model.py` refactor: constructor arguments for input directory and mesh file, skip mesh generation when the staged mesh exists), `georising.py`; local-only beds later |
| `docs/study-substrate.md` | the canonical substrate reference linked from every skill |
| `tests/` | fast lane (no simulator: specs, seed ledger, gates, cost, Sobol policy, executor with a fake task) and darts-gated lane (smoke beds) |

### 6.3 ES-MDA contract and milestone

First implementation: `esmda.py` over `dageo.esmda` with a **diagonal observation covariance
only** (per-datum sigma; time correlation handled by decimation to report times), full-cell
log-permeability updating on the proxy (431 cells) or GeoRising (10,800 cells) with
Gaspari-Cohn localization on cell centroids (taper of shape n_parameters x n_data using each
datum's well location), `return_steps` off by default. Specified choices: bounds enforced after
every update through the declared transform; anisotropy as a fixed ratio; no hard-data
conditioning in the twin; Euclidean localization within one region; localization across faults
or disconnected regions deferred (needs a connectivity-aware distance); collapse diagnostics
(spread ratio, rank of ensemble anomalies) every step; observation scaling by per-datum sigma
across rates, pressure and temperature. Post-update PCA projection is an experimental
alternative only. Extension: a native update with full or block-diagonal covariance, localized
on the parameter-data covariance (no whitening), needed before any fraternal twin.

Pre-registered twin test (proxy, identical twin): H1 localized full-cell updating attains lower
held-out normalized RMSE than unlocalized coefficient-space updating at equal cost; H2 its 80 %
predictive interval coverage lies in [0.7, 0.9]; metrics: held-out normalized RMSE, coverage at
80 and 90 %, spread ratio, correlation with truth in the identifiable subspace; seeds: ledger
root 20260903, substreams 0-9 (ten truths); selection rule: adopt the formulation with better
held-out RMSE when bootstrap intervals separate, otherwise the simpler one; fixed before results.

## 7. Skills

Three self-contained skills. Each `SKILL.md` carries the indispensable invariants inline
(pinned invocation, one process per simulation, staged inputs, lean outputs, observation
conventions, cost gate, no truth peeking), links the canonical substrate reference
`workflows/docs/study-substrate.md`, and routes to exactly one method reference per task.
Skill `scripts/` are thin wrappers around `python -m workflows`. `validate_skills.py` checks
routing, links, frontmatter, unfinished placeholders and executable helpers; concision is a
review criterion. `AGENTS.md` gains one mandatory-usage line per skill. Filling
`model-workflow-open-darts/references/model-workflow.md` is a prerequisite.

| Skill | references/ | scripts/ |
|---|---|---|
| `ensemble-study-open-darts` | `experimental-design.md`, `priors-and-parameterization.md`, `analysis-and-reporting.md` | `estimate.py`, `design.py`, `run_ensemble.py`, `analyze.py` |
| `history-matching-open-darts` | `method-selection.md`, `observations-and-identifiability.md`, `adjoint-hm.md`, `esmda.md`, `diwa.md`, `validity-gates.md` | `make_truth.py`, `gradient_check.py`, `hm_adjoint.py`, `hm_esmda.py` |
| `optimization-open-darts` | `problem-formulation.md`, `well-placement.md`, `control-optimization.md`, `surrogate-assisted.md`, `robust-and-risk.md`, `reporting.md` | `optimize.py`, `postvalidate.py`, `exhaustive.py` |

## 8. Harness surfaces

Project settings with a `PostToolUse` hook running sync and validation on skill edits;
`helper_scripts/validate_skills.py`; the `AgentRunner` protocol in `workflows/evals/` with
`ClaudeCliRunner` first (headless CLI, exact model id, JSON output) so other harnesses reading
`.agents/skills` can be evaluated the same way; no subagents; Workflow scripts only for
proposing skill edits. The evaluation loop is documented in `workflows/evals/README.md`.

## 9. Test beds and ground truths

| Case | Bed | Parameters and controls | Observations / objective | Reference | Gates |
|---|---|---|---|---|---|
| ens-proxy | proxy | log-perm cell field (KL 10), 3 Corey parameters, per-region WI multiplier (D = 14) | cumulative oil, water cut, breakthrough | precision-adaptive ensemble (grown until P10/P90 interval half-widths fall below a third of the tolerance; nested high-sample reference with a Monte Carlo error budget) | statistics within tolerance including reference uncertainty; ranking agreement; budget |
| hm-proxy-twin | proxy | truth drawn from the declared prior with a seed disjoint from tuning and evaluation seeds; known per-well BHP schedule varied every 120 days | producer oil/water and injector gas rates at 120-day report times, declared diagonal noise, last 20 % held out; ten hidden truths | truth fields, T/WI references | held-out normalized residuals, predictive coverage, spread ratio, no degradation on withheld quantities; recovery in the identifiable subspace secondary |
| hm-georising | GeoRising | perm-field perturbation, PT/PH | PRD BHT, BHP, rates | ten truths | same |
| hm-fraternal (hypothesis) | proxy vs full field | requires a paired-realization or upscaling map, a well-coordinate and completion map, compatible controls, observation definitions and units, and a separation of upscaling error from structural discrepancy | same | full-field rates | not executable until the mapping exists |
| opt-place-proxy | proxy | one injector relocation | cumulative oil | exhaustive over feasible cells | feasibility, regret within tolerance, post-validation |
| opt-place-georising | GeoRising | producer placement | net energy | exhaustive 60x60 feasible | same |
| opt-controls-proxy | proxy | 20 producer BHP targets x 2 intervals (D = 40) | cumulative oil | repeated-seed FD/CMA-ES numerical reference | feasibility, regret vs baseline and reference |
| opt-wi-proxy | proxy | completion via WI of 10 producers | cumulative oil | repeated-seed adjoint reference | same plus angle gate |
| opt-robust-proxy | proxy | injector relocation over a 20-member ensemble | CVaR(0.2) cumulative oil | exhaustive with common random numbers | same |
| opt-surrogate-smoke | proxy | 2-D placement, database <= 300 | cumulative oil | exhaustive | re-validation, regret, budget |
| opt-gas-eor-proxy | proxy | gas injection rates and composition (simplex transform) | cumulative oil, gas utilization | repeated-seed reference | regret |
| opt-ccs-cpg (deferred with the bed) | full-field CO2-water variant | injection schedule | CO2 injected, pressure footprint | repeated-seed reference | feasibility, regret |

Truths are generated by the controller with `truth_generation_hash`, stored as CSV/JSON in
`workflows/evals/truth/` (hidden from the evaluated agent by mount masking, Section 10), never
referenced by skills; every case has smoke and full tiers.

## 10. Evaluation harness and improvement loop (`workflows/evals/`)

- Broker: inside the sandbox the agent prepares study specs and calls the CLI, which submits
  them over a Unix socket to a parent-side broker (`workflows/evals/broker.py`); the broker
  validates the spec, runs the simulations in its own processes, owns the journal and member
  results, and exposes results read-only (a parent-owned directory bind-mounted read-only into
  the sandbox). The agent's writable directory holds only its own notes, specs and reports.
- Mount manifest (`bwrap`): repository and conda environment read-only; `workflows/evals/truth/`
  and `workflows/evals/scorer.py` over-mounted with empty tmpfs after the repository bind;
  per-run writable directory; scratch `HOME` with a read-only bind of
  `~/.claude/.credentials.json` and a minimal settings file; minimal `/proc` and `/dev`; PID
  namespace unshared. A canary test at controller start reads a canary file in the masked
  paths from inside the sandbox and must fail.
- Network: `bwrap` cannot filter egress; outbound traffic is unrestricted unless an
  allow-listing proxy (API host only) is deployed through proxy environment variables; the
  deployed state is recorded in every result.
- Post-exit verification: the controller re-hashes the input snapshot, every submitted spec,
  every broker record and the results; manifests must chain (result to spec to inputs to
  binary fingerprint); mismatches fail the run. Simulation counts, CPU time, RSS and exit
  status come from the broker and the parent, never from the agent's directory.
- `AgentRunner` protocol: `run(case, seed, budget) -> transcript, usage, exit`;
  `ClaudeCliRunner` first.
- Pre-registration per benchmark version: primary metric per workflow, secondary metrics
  (simulations, tokens, wall time), non-inferiority margins, hidden holdout cases and seeds,
  8 to 10 paired runs per case and variant, a maximum number of tuning rounds, sequential or
  multiple-comparison control, and a new benchmark version whenever prompts, scorer logic or
  metric definitions change. Per-seed results are logged in `workflows/evals/TUNING_LOG.md`.

## 11. Cost model and budgets

Simulator costs (proxy lean member 8-12 s wall; about 70 usable cores; ranges, not points):

| Item | Simulations | Wall | CPU-hours |
|---|---|---|---|
| precision-adaptive ensemble reference (proxy, once) | 2000-4000 | 5-10 min | 6-11 |
| Morris D = 14, r = 10 | 150 | 20-40 s | 0.4 |
| Sobol first/total order, D = 8, N = 512 | 5120 | 12-15 min; with second order 9216, 22-27 min | 14 / 26 |
| ES-MDA ne = 100, 4 steps + posterior prediction (proxy) | 500 | 1-2 min | 1.4 |
| adjoint HM, 30 iterations (proxy) | 30 forward + 30 adjoint | 6-10 min serial | 0.2 |
| exhaustive placement, proxy / GeoRising | <= 431 / 3600 | 1-2 min / 3-5 min | 1.2 / 3.6 |
| FD controls D = 40, 20 iterations | 1600 gradient runs + line search (1800-2500) | 4-7 min | 5-7 |
| robust placement, 431 candidates x 20 realizations | 8620 | 20-30 min (full tier) | 24 |

Aggregate evaluation budget (example round): 10 cases x 10 truths generated once (proxy:
about 10 min); 4 public cases x 8 paired runs x 2 variants = 64 agent runs per round; each
run 200-1000 simulations (0.5-3 min of simulator wall) and 10-40 min of agent wall; with eight
concurrent runs a round takes 2-6 h; five rounds and the hidden holdout double it.

Token budget (assumptions to be replaced by measurements): 0.3-1.0 M tokens per agent run;
64 runs per round give 20-65 M tokens; a five-round campaign 100-330 M tokens. Cost in currency
is read from the CLI result per run. A smoke tier (two cases, three seeds) is mandatory before
any full round.

## 12. First milestone and implementation order

First milestone (zero-agent, deterministic, two CI beds: the Brugge proxy and GeoRising):

1. Brugge-proxy ensemble/UQ with the seed ledger, staged inputs and observation extraction.
2. ES-MDA identical twin (proxy or GeoRising) with diagonal covariance and full-cell localized
   updating; the pre-registered twin test of Section 6.3.
3. A small discrete well-placement problem with an exhaustive reference.

Deferred until the substrate passes its tests: the full-field Brugge bed, the fraternal twin,
DiWA, surrogate optimization, CO2 cases, large Saltelli studies, native full-covariance ES-MDA,
any packaging.

Order: (1) mathematical contracts, identities, metrics and cost formulas (this document);
(2) `workflows/` skeleton with `spec.py`, `journal.py`, `cost.py`, `requirements.txt`, the
fast test lane, Ruff and verify-trigger coverage; (3) input staging and the adapter lifecycle,
including the Brugge model refactor; (4) executor, broker, journal ownership, process
accounting; (5) deterministic integration tests on the smoke beds, including the assertive
adjoint check; (6) the evaluator behind OS-level isolation with canary and post-exit
verification; (7) the model-workflow reference and the three thin skills once CLI contracts
are stable; (8) hidden agent evaluations and improvement rounds.

## 13. Risks and open questions

- The localization formulation and the diagonal-covariance restriction are hypotheses until
  the twin test runs; the fraternal twin has no executable mapping yet.
- The layer is usable only inside this checkout by design at this stage; packaging is deferred.
- Outbound network from the sandbox is unrestricted without a proxy.
- MR!330 requires a fresh rebase; seven conflicting paths at the analyzed baseline.
- Private DiWA code, fine-model data and locally stored manuscripts need license and
  provenance review before any copy enters this repository; the full-field Brugge inputs are
  untracked and their storage mechanism is undecided.
- Thermal adjoint gradients are approximate; thermal beds default to ES-MDA.
- The documentation build could not be verified in the `skills` env.

## Appendix A: invariants and heuristics

Invariants: pinned interpreter; one process per simulation; staged immutable inputs; lean
member outputs; rows selected by time; instantaneous adjoint observation convention; angle gate
before trusting a new adjoint setup; engine-type guard; `cache=False` when rel-perm or PVT
vary; preserve declared dependence between parameters; log-space permeability sampling;
hold-out data and no truth peeking; baseline before optimization; feasibility before and after
snapping; full-physics re-validation of surrogate results; a declared observation covariance
before quoting chi-square; retry identity preserved; Sobol rows never replaced.

Heuristics with ranges: Morris before Sobol and full Sobol only at low dimension; ensemble
size by convergence check (100-300 at field scale); localization near the correlation range;
PCA energy by prediction sensitivity; multi-start counts; surrogate database sizes as case
observations.

## Appendix B: reference outlines

Per reference: What it is . When to use . Assumptions . Uncertainty treatment . Inputs .
Invocation . Outputs . Gates and stopping . Caveats . Cost model . Best practices, carrying the
rules of Appendix A relevant to the method; the ES-MDA reference states the diagonal-covariance
restriction and the choices of Section 6.3; the surrogate reference covers baselines,
cross-validated selection, uncertainty-aware acquisition and re-validation.

## Appendix C: prototype evidence

Assertive check `workflows/evals/spikes/brugge_adjoint_check.py` (with `brugge_model_adj.py`;
result `brugge_adjoint_check_result.json`): objective vanishes at the truth, stratified control
subset (near-well and far-field interfaces, well indices), step-size sweep, random full-vector
directional derivatives, assertions with non-zero exit on failure, provenance (repository head,
engine binary hashes, input hashes, versions, platform, threads), monotonic timers, contained
child output, atomic result write. The previous non-assertive version was reproduced exactly by
the reviewer.

| Check | Result |
|---|---|
| Objective at truth / at 2 % start | 1.4e-27 / 693.6 |
| Stratification of the 12-control subset | near-well/far-field (WI, far-field, near-well) |
| Median relative error by relative FD step | 0.001: 4.4e-05, 0.0001: 1.6e-05, 1e-05: 4.0e-05 |
| Best-step relative error (median / max) | 1.4e-05 / 1.1e-04 |
| Subset angle | 0.0009 degrees |
| Random full-vector directional derivatives (relative error) | 1.2e-05, 6.9e-05, 4.0e-06 |
| Timings (s) | truth_build_and_init_s 0.639, truth_forward_s 0.961, proxy_build_and_init_s 0.392, forward_s 0.641, adjoint_gradient_s 0.505 |
| Checks passed | objective_vanishes_at_truth, subset_angle_below_max, median_best_rel_err_below_max, directional_derivatives_agree, stratified_groups_present; overall passed = True |
| Provenance recorded | head 59bda10b, engine binaries 1 hashed, inputs 6 hashed, numpy 2.4.6, OMP 1 |

Other measurements (scratch runs, not committed): proxy lean member 8-12 s wall; full-field
Brugge dead-oil 39 s and geothermal 74 s wall; headless CLI JSON fields present.

## Appendix D: sources

Cited through `docs/references.bib`: {cite}`Tian2022DiWA`, {cite}`Tian2023PhD`,
{cite}`Emerick2013ESMDA`, {cite}`Gaspari1999`, {cite}`Chen2012RML`,
{cite}`Chen2017Localization`, {cite}`Oliver2008Inverse`, {cite}`Morris1991`,
{cite}`Campolongo2007`, {cite}`Saltelli2010`, {cite}`McKay1979LHS`, {cite}`Bangerth2006`,
{cite}`Isebor2014`, {cite}`VanEssen2009`, {cite}`Brouwer2004`, {cite}`Jansen2011Adjoint`,
{cite}`Zandvliet2008`, {cite}`Jones1998EGO`, {cite}`Novikov2026WellPlacement`,
{cite}`dageo2025`. Non-redistributed inputs: the GEOEN45444 reviewer manuscript (surrogate
methodology, case observation only), the agentic-clrm repository (executor, closed-loop report
2026-07-26), GitLab MR !330, the open-darts-skills tree at `59bda10b`, the Agent Skills
specification and Claude Code documentation.

```{bibliography}
:filter: docname in docnames
```

## Appendix E: planned dispositions

| Review finding or direction | Planned disposition |
|---|---|
| Round 1: ES-MDA localization inconsistent | full-cell updating with spatial localization; twin test |
| Round 1: truth isolation | OS-level isolation (superseded by the round-2 broker design) |
| Round 1: CO2 case | split gas-EOR / CCS; combined deferred |
| Round 1: worker staging | adapter lifecycle with snapshot and staging |
| Round 1: Saltelli, FD, hashes, gates, thresholds, seeds, rank histogram, geostatistics, surrogates, hard rules, tolerances, evidence, provenance, plan breadth | retained from version 2 |
| Round 2: package boundary incomplete | superseded by the owner's direction: no separate package; in-repo `workflows/` with `requirements.txt`, Ruff coverage (done) and tests in the existing job |
| Round 2: full covariance vs dageo | diagonal-only first implementation stated; native full-covariance update planned; adjoint diagonal only |
| Round 2: evaluator tamperable | broker, mount manifest with masking, canary, credential bind, post-exit hash verification, network acknowledgement |
| Round 2: ES-MDA milestone under-specified | scientific choices specified; PCA projection experimental; pre-registered hypotheses, metrics, seeds, selection rule |
| Round 2: Sobol failed members, power-of-two N | retry-then-preserve policy; N = 512 |
| Round 2: reference ensembles are estimates; P10 convention | precision-adaptive references with intervals; percentile convention stated |
| Round 2: seed ledger; thread-varying retry | hierarchical ledger; non-equivalent attempt identities |
| Round 2: full-field Brugge not CI; cell count; fraternal mapping | local candidate bed; 43,847 everywhere; fraternal twin as hypothesis with required mappings |
| Round 2: cost and budgets | corrected formulas, ranges, CPU-hours, aggregate simulation and token budgets |
| Round 2: optimization definitions; GP dependency | regret floor and uninformative outcome, CVaR tail, NPV assumptions, failure policy, WI mapping, composition transform; scikit-learn in the requirements file |
| Round 2: skills duplication and line gates; harness coupling | inline invariants, one canonical substrate reference in `workflows/docs`, validation without line counts, `AgentRunner` |
| Round 2: adjoint spike | assertive, provenance-rich, linted; moved to `workflows/evals/spikes/` |
| Round 2: provenance and MR fact | baseline wording, orphan page, bibliography entries, seven-conflict merge-tree fact, rebase required |
| Owner direction (2026-09-03) | no separate repository or package at this stage; everything inside this repository in the skills folders and `workflows/`; interfaces kept to one package, one CLI, one adapter protocol, one spec family; three skills; evals as a developer tool under `workflows/evals/` |
