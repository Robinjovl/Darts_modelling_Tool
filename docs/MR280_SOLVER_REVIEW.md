# MR280 linear-solver subsystem review — interfaces, performance, dynamic reconfiguration

*Review of 2026-07-06. Branch `xiaoming/add-mgr` @ `196cd027` (MR head), base `deb647c2` (development HEAD).
Benchmarks: full SPE10 (60×220×85 = 1.122M cells, 2-component dead-oil, 5 wells, 200 days) on oahu
(128 cores, 2× A100 80GB). MR build: open-source GPU (`ENABLE_BOS_SOLVERS=OFF, WITH_AMGX=ON, WITH_CUDSS=ON`);
baseline: development install (proprietary BOS + AMGX).*

## Verdict

The architecture is right: one C++ `linear_solver` interface, name-based registry, one Python
`LinearSolverSpec` API, runtime-block-size matrix layer with a clean setup/solve split, OpenMP genuinely
restored. What blocks the performance story today: the CPU default (FGMRES+CPR) is **4.6× slower** than the
proprietary baseline on SPE10-class problems, the GPU story has **no AMG preconditioner** (AMGX-CPR silently
redirects to BiCGStab+ILU0, 11× slower than the dev GPU stack), and every HYPRE-based preconditioner stage
runs **serial** regardless of thread count. Tuned MGR is genuinely competitive (+28% wall vs BOS CPR+AMG at
np16) and cuDSS is a real new capability. Two critical robustness defects need fixing before merge.

| dimension | assessment |
|---|---|
| Extensibility | Registry + spec dataclasses: good. Adding a solver still means 4-way field mirroring; GPU specs can't carry config at all. |
| Fragility | Two criticals (MGR −0 return, frozen thread partition). Ownership works by convention (keep_alive + spec attrs) — holds, but nothing enforces it. |
| Consistency | Spec↔C++ field mapping verified 1:1 (34 MGR fields). Mixed error conventions; dead knobs (spec tolerance on engine-resident solvers); MGR code island. |
| Robustness | Loud failures at build/lookup; silent failures at solve time (MGR setup, GMRES/BiCGStab success-on-non-convergence, AMGX status ignored). |
| Performance | Matrix layer right by design (structure once, O(nnz) value re-gather). Consumption flawed: CPR pays the adjoint chain every forward Newton; `refresh()` dead; MGR/HYPRE stages serial. |

## 1 · SPE10 benchmark (full 85 layers, 200 days, identical schedule)

Protocol: original development-branch `shared/spe10` model recovered from git, untouched; solver injected by a
parametrized runner (`/oahu/data/avnovikov/spe10_solver_bench/`). `first_ts=0.1, mult_ts=2.5, max_ts=1000,
tol_newton=1e0, tol_linear=1e-2, it_newton=10`, eta-constrained. One config per process, strictly sequential.

| config | wall run s | steps | NI | LI | LI/NI | assembly s | LS setup s | LS solve s | LS % sim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dev cpu_gmres_cpr_amg, np=1 | 194 | 16 | 41 | 184 | 4.5 | 74 | 48 | 66 | 59% |
| dev cpu_gmres_cpr_amg, np=16 | **174** | 16 | 41 | 184 | 4.5 | 66 | 45 | 58 | 60% |
| dev gpu_gmres_cpr_amgx_ilu | **17** | 17 | 43 | 227 | 5.3 | 2 | 5 | 3 | 51% |
| MR FGMRES+CPR (BoomerAMG), np=1 | 973 | 15 | 33 | 1317 | 39.9 | 64 | 192 | 714 | 93% |
| MR FGMRES+CPR (BoomerAMG), np=16 | 804 | 15 | 33 | 1317 | 39.9 | 12 | 197 | 590 | 98% |
| MR MGR tuned (BCSR-CPR), np=1 | 275 | 16 | 42 | 298 | 7.1 | 68 | 61 | 142 | 74% |
| MR MGR tuned (BCSR-CPR), np=16 | **222** | 16 | 42 | 298 | 7.1 | 11 | 62 | 143 | 93% |
| MR AMGX-CPR request (→ BiCGStab+ILU0) | 195 | 30 | 94 | 4466 | 47.5 | 23 | 3 | 145 | 76% |
| MR cuDSS direct (GPU) | 366 | 16 | 41 | 41 | 1.0 | 2 | 354 | 2 | 98% |

Key readings:

1. **Physics parity holds.** dev np1/np16, dev GPU, MR MGR, MR cuDSS: 16–17 steps, 41–43 NI — same
   nonlinear trajectory. cuDSS (exact solves, LI/NI=1.0) matches dev at exactly 41 NI — a strong
   correctness cross-check of assembly → block→scalar expansion → device solve.
2. **MGR is the real CPU replacement, not the default CPR.** Tuned MGR 222 s vs dev 174 s (+28%),
   iteration quality 7.1 vs 4.5 LI/NI. Default FGMRES+CPR: 804 s (4.6×), LI/NI=39.9.
3. **OpenMP restoration works — and is the only thing scaling.** MR assembly 64–68 → 11–12 s at np16
   (≈5.8×); dev GPU-build shows none (74→66 s). Solver stages do not scale: MGR solver time np-invariant
   (204 vs 206 s), CPR setup flat (HYPRE-OMP is an opt-in rebuild, off by default).
4. **GPU flagship regressed 11×.** dev AMGX-CPR: 17 s. MR: "In-tree AMGX-CPR is not yet
   block_csr_matrix-ready" → BiCGStab+ILU0: 195 s, 30 steps, 94 NI, 4466 LI.
5. **cuDSS: from "infeasible" to robustness option.** 2.244M unknowns factorize in ≈8.6 s (~64 GB device
   memory); prior "infeasible" verdict was GPU memory contention, not a hard limit.
6. **Conditions:** shared machine (background load ~30–45/128 cores; GPU0 shared with another job at
   20–77% util) — GPU timings if anything pessimistic.

## 2 · Findings (106 total, 9 areas; key ones re-verified by hand)

Status: **verified** = failure chain traced in code by hand; **empirical** = reproduced at runtime;
**reported** = agent finding, not independently re-verified.

### Critical

- **MGR setup failure returns 0 (success) with a zero solution** — failure paths set
  `iterations=0, converged=false`; `solve()` encodes failure as `-iterations` = −0 = 0; `linsolv_mgr` only
  checks `iters < 0`. Newton applies a zero update, silently stalls to `max_i_newton`; invisible to stats and
  adaptive policies. `mgr_linear_solver.cpp:6719–6726`, `linsolv_mgr.cpp:1402`. Fix: branch on `converged`. *(verified)*
- **Thread partition frozen at Jacobian init** — `row_thread_starts` sized once to `omp_get_max_threads()`;
  public `set_num_threads()` has no repartition hook. Fewer threads afterwards → unassembled tail rows (stale
  Jacobian, silent); more → OOB reads. `sparsity_pattern.cpp:95`, `py_globals.cpp:347`. *(verified)*

### Performance majors

- **CPR pays the adjoint machinery every forward Newton**: unconditional scalar transposes (As→AsT, Ap→ApT),
  4 HYPRE IJ refreshes, and on rebuild 2×BoomerAMGSetup + 2×ILUSetup (fwd + transpose). ~2× setup for
  forward-only runs; no guard. `linsolv_cpr.cpp:588,652,805`. *(verified)*
- **`refresh()` fast path dead from the engine**: all engines call `setup(Jacobian)` per Newton; reuse only via
  per-solver flags, off by default. `engine_base.cpp:3099` vs `linear_solver.hpp:104`. *(verified)*
- **AMGX-CPR request silently serves BiCGStab+ILU0** — `linsolv_bos_cpr_gpu::init` expects legacy
  `csr_matrix<N>` device layout, segfaults on `block_csr_matrix` (in-code TODO). `engine_base_gpu.h:340–353`. *(empirical)*
- **Preconditioner stages serial in default build** — HYPRE without OpenMP (opt-in `HYPRE_OPENMP=1` + clean
  rebuild); MGR/BCSR-CPR kernels unthreaded. Measured: MGR solver time np-invariant. *(empirical)*
- Plain-MGR mode rebuilds+destroys the entire MGR hierarchy + coarse AMG + Krylov object *inside every
  solve()*; default BCSR-CPR path tears down/rebuilds pressure IJ+AMG per Newton unless reuse flags on.
  `mgr_linear_solver.cpp:5574,3277`. *(reported)*
- PETScSolver rebuilds AIJ/KSP/GAMG every Newton; block→scalar structural pass is a pure-Python triple loop.
  `python_solvers.py:243,78`. *(reported)*

### Robustness / fragility majors

- **Success-on-failure family**: GMRES returns 0 at max-iters (observed live: LI = 2×cap reported converged);
  BiCGStab returns 0 on breakdown; AMGX statuses ignored; cuSOLVER reports singular as success. Only cuDSS
  propagates failure (−1 → timestep cut). *(empirical)*
- **`SolverSwitchContext.linear_solver_error` always 0** — C++ engines maintain `linear_solver_error_last_dt`
  but it is not pybind-bound. One-line fix. `py_engine_base.cpp:39–42`. *(verified)*
- dual_array dirty flags set on pointer access, not writes; `sync_to_device` can clobber newer device data;
  const_cast bridge edits bypass tracking; asserts only. `dual_array.hpp:133,152`. *(reported)*
- hypre wrapper leaks IJ matrix+vectors per setup (FS-CPR pressure stage); MGR strategy's coarse AMG never
  destroyed; `check_result` calls `exit(-1)` on HYPRE errors. `linsolv_hypre_amg.cpp:173,337`,
  `compositionalFlowStrategy.cpp:579`. *(reported)*
- Open-source engine `exit(1)`s if no solver injected (kills the Python process/kernel);
  platform gating by enum ordering. `engine_base.h:825`. *(verified)*

### Consistency / extensibility

- Spec↔C++ field mapping complete and correct (MGR 34 fields, CPR, GMRES, FS-CPR) — verified. *(pass)*
- Adding a solver: every config field declared 4× (C++ struct, pybind, dataclass, `_make_config`) + hand-copied
  13-case block-size switch per factory; config-type mismatch silently discards user config.
  `solver_factories.cpp:196,227`. *(reported)*
- GPU specs carry no configuration (enum-only); spec `tolerance`/`max_iterations` dead on engine-resident CPU
  solvers too (engine init overrides from `data_ts`). *(verified)*
- `FSCPRSolverSpec` missing from `darts/solvers/__init__.py` exports. *(verified)*
- CHANGELOG "deterministic across thread counts" overstated — `dot()` fixed-order combine is reproducible per
  thread count, not across counts. *(verified)*
- MGR subtree is a ported code island (GEOS-style CamelCase, `void*` init API, options plumbed through 3
  hand-maintained layers with drifted defaults, ~25 unconditional stdout lines per strategy rebuild). *(reported)*

## 3 · CPU solvers & OpenMP under the GPU build

| capability | status | evidence |
|---|---|---|
| Registry: gmres/cpr/mgr/superlu/fs_cpr | works | all build; GMRES+CPR and tuned MGR ran full SPE10 |
| PETSc / Pardiso (Python-resident) | present | petsc4py + pypardiso importable |
| AdaptiveSolverSpec mid-run switching | works | live SPE10 test: MGR 2 steps → policy switch → GMRES+CPR continued on same Jacobian |
| OpenMP assembly under GPU build | works | 5.8× at np16; dev build has none |
| OpenMP in preconditioner stages | gap | HYPRE-OMP opt-in; MGR kernels unthreaded |
| `set_num_threads` after init | unsafe | frozen-partition critical |
| Import-time default threads | surprise | no `OMP_NUM_THREADS` ⇒ max/2 at import; 0 on 1-core hosts |

## 4 · Dynamic reconfiguration plan (summary)

Full plan: `SOLVER_DYNAMIC_RECONFIG_PLAN.md` at repo root.
Verified foundation: `engine.set_linear_solver()` re-inits the new solver against the **existing** Jacobian —
no reallocation (`engine_base.h:192`); `AdaptiveSolverSpec` switches per-timestep end-to-end today.

- **P0 signals** (~40 lines): bind `linear_solver_error_last_dt`; fix MGR −0; expose per-timestep
  `solver_stats` + setup/solve times into `SolverSwitchContext`.
- **P1 `reconfigure()`** (~150 lines): 3-tier field classification — hot (tolerance/max_iters/thresholds:
  applied next solve), warm (AMG/ILU shape: `force_rebuild_`, hierarchies rebuilt against the same bound
  matrices), structural (refused → switch path). Never touches matrix binding / csr_expansion / engine vectors.
- **P2 `model.update_solver(...)`** (~100 lines): one entry point; in-place with transparent fallback to
  rebuild; owns the `params`/`data_ts` override coupling; keeps `self.solver` authoritative.
- **P3 policy hooks** (~120 lines): context + `SolverAction(index, updates)` + `on_timestep_failed` hook so the
  dt-cut retry runs on the fallback solver (displaced_fault dynamic mode); ls_params vector deprecated;
  context/action/reward = contextual-bandit-ready.
- **P4 tests**: pointer-identity no-realloc test; forced-failure signal test; cost microbench in docstrings.

GPU excluded until GPU solvers are registry-built (blocked on the AMGX-CPR block_csr port).

## 5 · Recommendations

Before merge: fix the two criticals; sweep the success-on-failure family; bind the error counter; reconsider
the CPU default for large models (auto-MGR above a size threshold or `AdaptiveSolverSpec([CPR, MGR])`);
document the GPU redirect loudly.

Follow-ups: guard the CPR adjoint chain behind an `enable_adjoint` flag (~2× setup win); wire `refresh()` into
the Newton loop; HYPRE-OMP by default + thread MGR kernels; port `linsolv_bos_cpr_gpu` to `block_csr_matrix`;
config-declaration codegen; FS-CPR export fix; MGR island normalization.

---
*Reproduction: `/oahu/data/avnovikov/spe10_solver_bench/` (`run_bench.py`, `drive_all.sh`, `analyze.py`,
`results/<config>/run.{json,log}`, `adaptive_switch_test.py`). MR build `196cd0276` + `WITH_AMGX=ON`;
dev build `deb647c2`.*
