# Dynamic solver reconfiguration plan (on-the-fly parameter update / solver switching)

Goal: let a running simulation (a) change parameters of the live linear solver, (b) swap the
solver entirely, and (c) drive both from a pluggable policy (hand-written or learned), with the
hard invariant that **no reconfiguration path ever reallocates or rebinds the Jacobian** (or the
structure-derived caches: `csr_expansion`, `scalar_csr_adapter`, thread partition, device mirrors).

Motivating case: `models/displaced_fault_reactivation` needs FS-CPR (tol 1e-8) in the
quasi-static phase and GMRES+ILU0 (tol 1e-12, 500 iters) during dynamic rupture. Today that is
expressed through the legacy `engine_pm_cpu::ls_params` vector, which the open-source build
cannot honor (the spec path only injects one solver). The general need: online tuning /
switching for convergence and wall-time, e.g. bandit- or RL-style controllers.

## What already exists (verified in code, 196cd0276)

1. `engine_base::set_linear_solver(shared_ptr, name)` (engine_base.h:192) on an initialized
   engine re-inits the new solver against the **existing** Jacobian:
   `linear_solver->init(Jacobian, params->max_i_linear, params->tolerance_linear)`.
   The Jacobian is untouched — mid-run switching is already allocation-safe w.r.t. the matrix.
2. `AdaptiveSolverSpec` + `DartsModel._maybe_switch_linear_solver` (darts_model.py:518) evaluate a
   `policy(SolverSwitchContext) -> int` after every timestep and rebuild+inject the selected
   candidate. Works end-to-end on the open-source CPU build.
3. The `linear_solver` interface has the right lifecycle split: `init` (bind structure once),
   `setup` (per Newton), `refresh` (value-only fast path), `stats()` (uniform result struct).

## What is broken or missing (verified)

- **Signals**: `SolverSwitchContext.linear_solver_error` is always 0 — the C++ engines maintain
  `linear_solver_error_last_dt` but it is not pybind-bound (py_engine_base.cpp binds only
  `n_newton_last_dt` / `n_linear_last_dt`). Worse, MGR setup failures return `-0 == 0` through
  `linsolv_mgr::solve` (mgr_linear_solver.cpp:6719-6726 + linsolv_mgr.cpp:1402), so even the C++
  counter misses them; the engine applies a zero Newton update believing the solve succeeded.
- **No parameter update on a live solver**: the only way to change any knob is to build a new
  solver object (new HYPRE handles, new workspace) and re-init. For per-timestep policy actions
  like "tighten tolerance", "grow kdim", "raise AMG strong_threshold" this is 10-100x more
  expensive than necessary and discards warm state (AMG hierarchy, ILU pattern analysis).
- **Engine overrides spec knobs**: engine init clobbers solver tolerance / max_iters with
  `params->max_i_linear` / `params->tolerance_linear`, so even a rebuilt spec cannot change these
  without also mutating `params` — an undocumented coupling that any update API must own.
- **Switching granularity**: policy runs only after a *finished* timestep. A failed timestep is
  retried with the *same* solver at smaller dt; the fallback solver only takes over on the next
  step. For rupture-style events you want retry-with-fallback within the same report step.
- **Thread-count trap**: `set_num_threads()` after Jacobian init silently corrupts assembly
  (frozen `row_thread_starts` partition, sparsity_pattern.cpp:95) — any "adaptive resource"
  policy must be blocked from this until repartitioning exists.
- **GPU seam**: GPU solvers are enum-selected at engine-factory time (`params.linear_type`),
  not registry-built; there is no injection path, so no mid-run switch/reconfigure on GPU until
  the GPU solvers move onto the registry.

## Phase 0 — fix the signal path (prerequisite, small diffs)

1. Bind `linear_solver_error_last_dt` in `py_engine_base.cpp` (one `def_readwrite` line).
2. Fix the MGR `-0` return: in `mgr::LinearSolver::solve` return a distinct negative code when
   `!m_lastResults.converged` regardless of iteration count (and/or check the `converged` flag in
   `linsolv_mgr::solve` instead of the sign).
3. Plumb `linear_solver->stats()` (already returns `{iterations, residual, converged}`) into a
   per-timestep, Python-visible record: engine accumulates a small POD array of per-Newton
   `solver_stats` + setup/solve wall times (both timer nodes already exist); expose read-only via
   pybind. Extend `SolverSwitchContext` with: `setup_time`, `solve_time`, `per_newton_linear_iters`,
   `final_residual`, `dt`, `wasted_newtons`. All are already measured — this is exposure, not
   new instrumentation.

## Phase 1 — `reconfigure()` on the C++ interface (parameter update without realloc)

Add to `linear_solver`:

```cpp
/** Apply a new configuration to a live solver without rebinding the matrix.
 *  Never reallocates or touches the bound matrix A_, the csr_expansion cache,
 *  or engine-owned vectors. Returns 0 on success, >0 if some fields could not
 *  be applied in place (caller may then fall back to a full rebuild). */
virtual int reconfigure(const solver_config &cfg);
```

with a three-tier per-field classification inside each solver:

| tier | examples | action |
|------|----------|--------|
| hot | tolerance, max_iterations, log level, adaptive-rebuild thresholds, local-correction alpha, BILU0 fallback strategy | store; effective next solve; zero cost |
| warm (invalidate hierarchies, keep bindings) | AMG strong_threshold / coarsen_type / relax, ILU fill, MGR level layout, CPR amg_max_iters | set `force_rebuild_` → next `setup()` rebuilds HYPRE hierarchies against the *same* IJ matrices; no Jacobian/adapter/workspace change |
| structural (identity change) | different registry solver, different block size | refuse (return >0) → caller uses the switch path |

Notes:
- The default base implementation handles the base-config fields (tolerance/max_iterations) —
  every solver already reads them per solve — and returns >0 for derived configs it does not
  recognize, so the fallback to rebuild is automatic and safe.
- GMRES `restart`/MGR `kdim` growth needs a Krylov-workspace regrow: allowed, done lazily at next
  solve; it is solver-internal memory, still zero impact on the Jacobian. Document it as "warm".
- pybind: `LinearSolver.reconfigure(config)`; spec side gets
  `LinearSolverSpec.reconfigure_live(built_solver, **field_changes)` that diffs the dataclass,
  produces the C++ config, and reports which tier was hit.
- Ownership stays as-is: the solver object identity is unchanged, so `linear_solver_external`,
  timers, and the engine raw pointer all remain valid — no interaction with the fragile
  keep_alive conventions.

## Phase 2 — one Python entry point: `model.update_solver(...)`

```python
model.update_solver(tolerance=1e-8)                # tier-hot reconfigure of the live solver
model.update_solver(pressure_amg=PressureAMGSpec(strong_threshold=0.7))   # tier-warm
model.update_solver(spec=SuperLUSolverSpec())      # full switch (set_linear_solver path)
```

Semantics:
- If `spec` is omitted: apply field updates to `self.solver` (the authoritative spec — so
  restarts and `_apply_solver` reproduce the current state) and call `reconfigure` on
  `self._linear_solver`. If reconfigure reports "structural", transparently rebuild+inject.
- If `spec` is given (different solver): build, `engine.set_linear_solver(...)` — the verified
  no-realloc seam. First `setup()` after a switch pays one full hierarchy build (same as one
  Newton-iteration setup) — document this so policies can budget it.
- `update_solver` also writes `params.tolerance_linear` / `params.max_i_linear` (and
  `data_ts.linear_*`) when those fields change, resolving the engine-override coupling in one
  place. Long term: make `data_ts.linear_* = None` mean "defer to spec".
- Explicitly reject thread-count changes here until the assembly repartition hook exists.

## Phase 3 — policy hooks able to carry an online learner

Extend the adaptive machinery (backwards-compatible):

1. `SolverSwitchContext` gains the Phase-0 fields — iterations, times, residuals, dt, phase tag.
   A `phase` string is settable by the model (`model.solver_phase = 'dynamic'`) so phase-driven
   policies (displaced fault: static→FS-CPR, dynamic→ILU0@1e-12) are one-line policies instead of
   engine-side ls_params machinery. Deprecate `engine_pm_cpu::ls_params` in favor of this.
2. Policy return type widens: `int` (switch candidate, as today) **or**
   `SolverAction(index=i, updates={...})` — switch and/or reconfigure in one decision. The
   dataclass keeps int-returning policies working.
3. New optional `on_timestep_failed(context) -> SolverAction | None` hook, evaluated **before the
   dt-cut retry**, so the retry itself can run on the fallback solver / tightened tolerance.
   This is the piece the displaced-fault dynamic mode actually needs.
4. For learning loops: context + action + measured reward (wall time per converged day, from the
   timer fields) is exactly a contextual-bandit interface. Ship one reference policy
   (`EpsilonGreedyTuner` over a small discrete action set: {reuse on/off, strong_threshold grid,
   candidate index}) as an example; keep the learner itself out of core.

## Phase 4 — invariants, tests, docs

- **No-realloc test**: init on SPE10-small; record `Jacobian` data pointer (pybind exposes the
  values buffer), solver workspace pointers, `csr_expansion` pointer. Reconfigure every hot+warm
  field, switch solvers twice (spec→spec), run 5 timesteps: pointers unchanged, results converge.
- **Failure-signal test**: force a singular pressure block → assert `linear_solver_error != 0`
  reaches the policy (guards both Phase-0 fixes).
- **Cost documentation**: microbench reconfigure (µs), warm rebuild (≈ one Newton setup), switch
  (init + first setup); put the numbers in the `specs.py` docstrings so policy authors can budget.
- **Determinism**: same run with a scripted action trace replays bit-identically at fixed thread
  count.

## Sequencing & effort

| step | size | risk | unlocks |
|------|------|------|---------|
| P0 signal fixes | ~40 lines C++/pybind | low | trustworthy policies (also fixes 2 review findings) |
| P1 reconfigure() | ~150 lines + per-solver classification | low-med | cheap in-place tuning |
| P2 update_solver | ~100 lines Python | low | single user-facing API |
| P3 policy hooks | ~120 lines Python | low | displaced-fault migration, learned tuners |
| P4 tests | ~1 day | — | guards the no-realloc invariant |

GPU: excluded until GPU solvers are registry-built (blocked on the AMGX-CPR block_csr port —
`engine_base_gpu.h:340` TODO). The API surface above is designed so GPU specs slot in later with
no user-visible change (`update_solver` just gains a working backend).
