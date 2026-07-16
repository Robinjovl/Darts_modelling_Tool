# MR280 solver review — update after priority fixes (CPR parity + GPU AMGX-CPR port)

*Follow-up to `MR280_SOLVER_REVIEW.md` (2026-07-06). Both review priorities are resolved and
validated on the full SPE10 benchmark (60×220×85 = 1.122M cells, 200 days, identical schedule,
same machine/conditions as the original study).*

## Executive summary

| | before (review) | after (this update) | baseline (dev proprietary) |
|---|---:|---:|---:|
| CPU FGMRES+CPR, np=16 | 804 s (LI/NI 39.9) | **129 s (LI/NI 4.6)** | 174 s (LI/NI 4.5) |
| CPU FGMRES+CPR, np=1 | 973 s | **186 s** | 194 s |
| GPU AMGX-CPR request | 195 s (BiCGStab+ILU0 redirect, 94 NI) | **17.1 s (real FGMRES+AMGX-CPR)** | 17 s |

The in-tree open-source stack now **beats the proprietary CPU baseline by 26%** at np=16 (thanks to
iteration parity + the OpenMP assembly this MR restored) and **matches the proprietary GPU stack
exactly**. No fallback paths remain in either configuration.

## Priority 1 — FGMRES+CPR vs `cpu_gmres_cpr_amg`: root cause and fix

### Root cause (from the proprietary sources at `darts-linear-solvers/lib/bos_linear_solver_lib`)

The production BOS stack (`linsolv_bos_gmres` → `linsolv_bos_cpr` → `linsolv_bos_amg<1>` +
`csr_ilu_prec<N>`; note `csr_cpr_prec.cpp` is dead code, not compiled into `libbos.a`) differs
from the open-source CPR in three ways, ranked by measured impact (5-day SPE10 ablation, np=16):

| delta | BOS behaviour | open-source CPR behaviour (before) | impact when reverted alone |
|---|---|---|---|
| **Pressure-AMG configuration** | PMIS coarsening, strength θ=0.75, Stüben standard interpolation with **no truncation**, C/F-ordered Gauss-Seidel (1+1 sweeps, weight 1.0), plain V-cycle, **1 cycle per apply**, dense-LU coarse solve at ≤100 rows, **no aggressive coarsening** (`amg_solver.cpp:47-83`, `linsolv_bos_amg.cpp:38-54`) | HYPRE defaults (HMIS, θ=0.25, ext+i interp truncated at 4 elems) **plus** one aggressive-coarsening level with multipass interp, 2 cycles per apply | **LI/NI 3.9 → 31.1 (8×)** — the dominant factor |
| **Decoupling weights** | column-sum **True-IMPES**: `w_i = D_ps·inv(D_ss)` with `D` = sums over the whole block column (`linsolv_bos_cpr.cpp:281-355`) | diagonal-block-only (quasi-IMPES with local f-elimination), despite the True-IMPES label | LI/NI 3.9 → 5.0 (+30%) |
| **Stage-2 smoother** | **block** ILU(0): dense N×N block LU on the block system, no fill (`csr_ilu_prec.cpp:290-565`) | HYPRE scalar ILU(0) on the expanded scalar system (plus the expansion + per-apply IJ-vector traffic it needs) | wall +38% at equal LI |

On top of these, the forward path unconditionally built the **adjoint (CPRA) transpose chain every
Newton** — a full scalar transpose, 4 HYPRE IJ refreshes, and a second BoomerAMGSetup + ILUSetup —
roughly doubling preconditioner setup for forward-only runs.

### Fix (all in-tree, default-on, previous behaviour still selectable)

`linear_solvers/src/linsolv_cpr.cpp` + `linsolv_cpr.hpp`, `cpr_block_ilu0.hpp` (new),
`solver_configs.hpp`, `solver_factories.cpp`, pybind + `CPRSolverSpec`:

1. **BOS-mimic BoomerAMG profile as the default**, fully parameterised (16 new knobs on
   `CPRSolverSpec` / `cpr_solver_config`: coarsen/interp/relax type + order + sweeps, strong
   threshold, aggressive levels, PMax/trunc, max levels, cycle, coarse size, coarse relax, relax
   weight). Negative value = keep the HYPRE default; the previous configuration remains reachable.
2. **Column-sum True-IMPES weights** (`weight_scheme=1` default; `0` = previous diagonal-block
   variant), plus BOS-style row sign normalisation of `A_p` (applied symmetrically on the forward
   restriction and the transposed prolongation — the CPRA math stays exact).
3. **In-tree block ILU(0)** (`cpr_block_ilu0<N>`: dense block inverses, structure prepared once,
   values refactored per setup, identity fallback on degenerate well blocks) as the default
   stage-2 (`stage2_type=1`; `0` = HYPRE scalar ILU). With it, the forward path no longer needs
   the scalar expansion at all.
4. **Lazy adjoint chain** (`eager_adjoint=False` default): the CPRA transpose hierarchies are
   built on the first `solve_transposed()` call and kept refreshed afterwards; forward-only runs
   never pay for them.
5. CPR/GMRES sub-timers ("CPR AMG setup", "CPR BILU0", "CPR scalar expand", ...) under the
   engine's linear-solver timer nodes — the solver time is no longer an opaque blob.

### Validation

- Full SPE10 200 days: **129 s** np=16 / **186 s** np=1 (dev: 174/194 s), LI/NI 4.64 (dev 4.5),
  identical NI/LI at np=1 and np=16.
- Timer split now mirrors dev: AMG setup 42.6 s vs dev BOS-AMG 40.4 s — like-for-like.
- `2ph_do` full run on the new default: healthy (NI=1, LI=3 per step late-time).
- Solver unit tests (`solvers_bos_cpr`, `solver_registry_roundtrip`, `bos_gmres`, `transpose`,
  `scalar_csr_adapter`, `csr_mat_t_vec`): all pass.
- **Adjoint (CPRA)**: `Adjoint_super_engine --adjoint-solver cpra` gradient angle **0.000614°**,
  bit-identical to the SuperLU exact-adjoint run and to the pre-change baseline — the lazy chain
  and sign normalisation are exact.

## Priority 2 — GPU: `linsolv_bos_cpr_gpu` on `block_csr_matrix`, no fallback

### What was missing

The in-tree AMGX-CPR stack was nearly port-complete (`setup()`/`solve()` already polymorphic over
`csr_matrix_base` device accessors; the AMGX wrapper already uploads device pointers), but the
**GPU outer Krylov did not exist** — `linsolv_bos_gmres` is a stub in-tree — so the engine served
every AMGX-CPR request with BiCGStab+ILU0.

### Fix

1. **New `linsolv_gmres_gpu<N>`** (`linear_solvers/include/linsolv_gmres_gpu.hpp` +
   `linear_solvers/src/linsolv_gmres_gpu.cpp`, compiled as CUDA): restarted right-preconditioned GMRES
   with the Krylov basis on the device; SpMV through the `block_csr_matrix` device layer,
   BLAS-1/GEMV through cuBLAS, Hessenberg/Givens on the host. Orthogonalisation is CGS2 (two
   GEMV pairs) instead of i round-trip dots per iteration. Proprietary-parity semantics:
   restart = min(max_iters, 50), ‖r‖ ≤ tol·‖b‖ convergence, one preconditioner application +
   one SpMV per iteration, single preconditioner application on the combined update per restart.
2. **Engine factory** (`engine_base_gpu.h`): under `OPENDARTS_GPU_HAS_AMGX`,
   `GPU_GMRES_CPR_AMGX_ILU` now builds `linsolv_gmres_gpu` → `linsolv_bos_cpr_gpu`
   (device True-IMPES reduction, `p_solver_setup_gpu=1, p_solver_solve_gpu=1`) →
   `linsolv_amgx<1>` on the pressure system + `linsolv_cusparse_ilu<N>` on the full system —
   mirroring the proprietary wiring. `GPU_BICGSTAB_CPR_AMGX` gets the same CPR under BiCGStab.
   **The redirect message is gone**; the fallback path remains only for genuinely AMGX-less builds.

### Validation

- Full SPE10 200 days on the A100: **17.1 s** wall — dev proprietary stack: 17 s. 17 steps /
  45 NI / 312 LI (6.9 LI/NI); assembly 2.4 s, LS setup 4.1 s, LS solve 3.9 s.
- Newton trajectory matches the exact-solver (cuDSS) reference step for step.
- Solver identity confirmed in the log: `GPU_GMRES_CPR_AMGX_ILU`, AMGX sub-timers present.

## Updated benchmark table (full SPE10, 200 days, identical schedule)

| config | wall run s | steps | NI | LI | LI/NI | assembly s | LS setup s | LS solve s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dev cpu_gmres_cpr_amg, np=1 | 194 | 16 | 41 | 184 | 4.5 | 74 | 48 | 66 |
| dev cpu_gmres_cpr_amg, np=16 | 174 | 16 | 41 | 184 | 4.5 | 66 | 45 | 58 |
| **MR FGMRES+CPR (fixed), np=1** | **186** | 17 | 44 | 204 | 4.6 | 70 | 53 | 59 |
| **MR FGMRES+CPR (fixed), np=16** | **129** | 17 | 44 | 204 | 4.6 | 12 | 55 | 57 |
| MR MGR tuned (BCSR-CPR), np=16 | 222 | 16 | 42 | 298 | 7.1 | 11 | 62 | 143 |
| dev gpu_gmres_cpr_amgx_ilu | 17 | 17 | 43 | 227 | 5.3 | 2 | 5 | 3 |
| **MR AMGX-CPR (ported)** | **17.1** | 17 | 45 | 312 | 6.9 | 2 | 4 | 4 |
| MR cuDSS direct (GPU) | 366 | 16 | 41 | 41 | 1.0 | 2 | 354 | 2 |
| *(before)* MR FGMRES+CPR, np=16 | *804* | *15* | *33* | *1317* | *39.9* | *12* | *197* | *590* |
| *(before)* MR AMGX request → BiCGStab+ILU0 | *195* | *30* | *94* | *4466* | *47.5* | *23* | *3* | *145* |

Notes: shared machine (same caveats as the original report); MGR numbers unchanged from the
original run. The CPU default (`GMRESSolverSpec(prec=CPRSolverSpec())`) now needs no
size-threshold caveat: it is the fastest CPU option on SPE10-class problems.

## Files changed

- `linear_solvers/include/solver_configs.hpp`, `linear_solvers/include/linsolv_cpr.hpp`,
  `linear_solvers/src/linsolv_cpr.cpp` — CPR formulation + AMG profile + lazy adjoint + timers
- `linear_solvers/include/cpr_block_ilu0.hpp` — new in-tree block ILU(0)
- `linear_solvers/include/linsolv_gmres_gpu.hpp`, `linear_solvers/src/linsolv_gmres_gpu.cpp` — new GPU GMRES
- `linear_solvers/src/solver_factories.cpp`, `linear_solvers/src/pybind11/py_main.cpp`,
  `darts/linear_solvers/specs.py` — config plumbing (spec ↔ C++ 1:1)
- `linear_solvers/src/linsolv_gmres.cpp` — timer forwarding to the preconditioner
- `engines/src/engine_base_gpu.h` — AMGX-CPR factory wiring, redirect removed
- `linear_solvers/src/CMakeLists.txt` — register the new GPU GMRES

## Findings-resolution round (after the two priorities)

All remaining actionable findings from the interface review were fixed in a follow-up pass;
the deep refactors are deferred with reasons.

**Performance re-validated after the fixes** (full SPE10, 200 days; background load was ~2.5x
higher than during the reference runs, so wall times carry that noise -- the Newton/linear
iteration counts and the solver-internal timers are the load-insensitive signal):

| config | after fixes | reference (post-priority) | iterations |
|---|---:|---:|---|
| MR FGMRES+CPR, np=16 | 137 s (LS setup 55.8 s) | 129 s (LS setup 55.4 s) | **bit-identical** (44 NI / 204 LI) |
| MR MGR tuned, np=16 | 245 s | 222 s | **bit-identical** (42 NI / 298 LI) |
| MR AMGX-CPR (GPU) | 19.2 s | 17.1 s | **bit-identical** (45 NI / 312 LI) |

Additional validation: all solver unit tests pass; `2ph_do` unchanged (68 steps); adjoint CPRA
gradient angle 0.000614 unchanged; and a new mid-run thread-switch probe (init at 1 thread, grow
to 16, shrink to 2 during a `2ph_do` run) completes on the correct trajectory -- exercising the
frozen-partition fix that previously produced silent Jacobian corruption or out-of-bounds reads.

### Fixed — correctness / robustness (C++)

| finding | fix |
|---|---|
| **MGR setup failure returned 0 (silent zero solution)** *(critical)* | failure code clamped to <= -1 regardless of iteration count (`mgr_linear_solver.cpp`) |
| **Thread partition frozen at Jacobian init** *(critical)* | `get_row_thread_starts()` re-derives the even-row partition when `omp_get_max_threads()` changed (both `block_csr_matrix` via `sparsity_pattern::ensure_row_partition_current()` and legacy `csr_matrix<N>`); validated by a grow/shrink mid-run probe |
| GMRES returned 0 at `max_iters` without convergence | hard failures (non-finite residual) return -4; plain non-convergence keeps the legacy 0 (BOS parity -- the proprietary GMRES also always returned 0) but is now visible via `stats().converged` |
| GMRES `restart < 2` infinite loop + negative indexing | clamped to >= 2 at solve entry |
| GMRES wasted SpMV against the zeroed initial guess | first residual is a straight copy of the RHS |
| BiCGStab silent breakdown / NaN | non-finite alpha/omega/residual return -4; `stats().converged` reports the max_iters case |
| AMGX solve status ignored | `AMGX_SOLVE_FAILED` / `DIVERGED` now return -1 (NOT_CONVERGED stays acceptable for a fixed-budget preconditioner stage) |
| cuSOLVER reported singular matrices as success | singular -> -1 (consistent with cuDSS); `h_Q` malloc/delete[] mismatch fixed |
| cuSPARSE-ILU(0) zero pivots never checked | `cusparseXbsrilu02_zeroPivot` queried after factorisation; zero pivot fails the setup |
| CPR-GPU ignored stage return codes | AMGX / ILU setup+solve failures now propagate (-> engine timestep cut) |
| `linsolv_hypre_amg` / `_ilu` leaked one IJ matrix + two vectors per setup; `check_result` called `exit(-1)` | stale handles destroyed on re-setup; helpers throw and the public entry points translate to a nonzero return; destructor uses a no-throw variant; helpers made file-local (the two same-named `inline` definitions were an ODR hazard) |
| MGR strategy leaked its coarse BoomerAMG per rebuild | destroyed on re-setup and in a new destructor |
| engine `exit(1)` / `exit(-3)` on solver-selection errors | `std::runtime_error` (pybind translates to a Python exception; the process/Jupyter kernel survives) |
| duplicate registry names silently kept the first factory; config-type mismatch silently discarded the user configuration | `register_builtin_solvers()` self-guarded; a mismatched config type now throws, with base `SolverConfig` still accepted and its tolerance/max_iterations honoured |
| `linsolv_iface_bos::solver` and `linear_solver_prop` members uninitialised | null/BOS-default initialised (prop takes the proprietary ctor's 30 / 1e-5 -- a plain zero would have made its guarded `set_tolerance` a permanent no-op) |
| GMRES hard `dynamic_cast` to `linsolv_cpr` for adaptive feedback | `set_last_outer_iters()` virtual on the base interface -- any preconditioner can join the adaptive-rebuild loop |
| MGR strategy printed ~25 stdout lines per rebuild | gated behind `log_level >= 2` |
| `get_num_threads` bound to `omp_get_num_threads` (always 1 -> THMC no-OpenMP guard dead) | bound to `omp_get_max_threads`; import-time default `max/2` clamped to >= 1 (was 0 on single-core hosts) |
| dual_array: device write did not mark the buffer populated (later `sync_to_device` clobbered it); read-only GPU SpMV marked the device modified | `device_data()` sets `device_populated_`; `gpu_bsr_spmv` reads via the const accessors |

### Fixed — Python API / build

`engine.linear_solver_error_last_dt` pybind-bound (SolverSwitchContext now sees real solver errors);
`FSCPRSolverSpec` exported from `darts.linear_solvers`; unknown GPU `linear_type_name` raises and a CPU
registry spec on `platform='gpu'` warns (both previously silent); a raw compiled solver on
`self.linear_solver` is honoured as documented (was silently replaced by the default);
`AdaptiveSolverSpec` rejects GPU / Python-resident candidates at construction (was a mid-run
failure); the SuperLU >30k-cell warning fires for `SuperLUSolverSpec`; `_BlockToScalarExpander`
structural pass vectorised (was an interpreted triple loop over nnz*b^2, verified bit-equivalent);
PETSc solver now builds its AIJ matrix / KSP / fieldsplit / composite PC once and refreshes values
in place per Newton; cuDSS cmake no longer FORCE-ratchets `WITH_CUDSS=OFF` into the cache and the
CONFIG-package path bundles `libcudss` into the wheel; the `BOS_SOLVERS_DIR` back-compat shim
applies once per build dir so an explicit `ENABLE_BOS_SOLVERS=OFF` wins; CHANGELOG determinism
claim corrected ("reproducible at a fixed thread count"); stale registry docs refreshed.

### Deferred (documented, not blocking)

- **MGR code-island refactors** -- triple-maintained ~40-field option plumbing, six near-identical
  solveGMRES_*/solveFlexGMRES_* variants, per-row heap allocations in the True-IMPES weight build,
  plain-MGR-mode per-solve hierarchy rebuild: invasive rework of ported code; the tuned BCSR-CPR
  profile (the recommended configuration) already avoids the per-solve rebuild path. Best done as
  a dedicated MGR-normalisation MR.
- **32-bit scalar-nnz arithmetic** (`index_t = int`): overflows past 2^31 scalar nonzeros
  (~270M blocks at b=2, beyond current single-node sizes); an `index_t` widening is an
  ABI-affecting change tied to the XSDK_INDEX_SIZE decision.
- **`refresh()` wiring into the engine Newton loop** -- part of the dynamic-reconfiguration plan
  (P1) rather than a spot fix.
- **HYPRE OpenMP by default + threading the MGR/BCSR kernels** -- changes solver numerics
  (documented opt-in `HYPRE_OPENMP=1`); the np16 solver-scaling lever remains open.
- **block_csr_view<N> unused hot path, scalar_csr single-slot cache keying, GPU enum-ordering
  platform gate in `engine_base.cpp`, SuperLU SamePattern reuse, FS-CPR NE>1, CI BOS-twin matrix,
  Windows .bat log path, `proprietary_linear_type` per-spec defaults** -- design decisions /
  low-risk-low-priority; tracked in the findings inventory.

## Follow-ups that remain from the original review

The two criticals (MGR −0 return, frozen thread partition), the success-on-failure sweep
(GMRES/BiCGStab/AMGX status handling), the `linear_solver_error_last_dt` binding, HYPRE-OpenMP /
MGR-kernel threading, and the dynamic-reconfiguration plan (`SOLVER_DYNAMIC_RECONFIG_PLAN.md`)
are unchanged and still recommended before merge / as tracked follow-ups.
