# open-DARTS Linear Solver — Consolidation & Refactoring Plan

Branch: `xiaoming/add-mgr`. One MR, delivered as a **stacked series of commits**.
Status: implementation complete; see Appendix C for the per-item completion table.
Last updated: 2026-05-20.

---

## 0. Crucial changes (TL;DR)

The MR is fundamentally a **dependency unwinding + unified interface** delivery.
What changes for downstream users / packagers:

* **Open-source solver implementations moved into `open-darts/solvers/`** — were
  shipped only via the proprietary `darts-linear-solvers/`. Open-DARTS no longer
  needs `BOS_SOLVERS_DIR` to have a working CPU + GPU iterative-solver stack.
* **AMGX added as a thirdparty submodule** (`thirdparty/AMGX`), opt-in via the
  `WITH_AMGX` CMake option. GPU builds with AMGX absent fall back to the
  in-tree BiCGStab + cuSPARSE-ILU solver.
* **HYPRE tracks latest release**, version-pin logic removed from build scripts.
* **Five GPU solvers absorbed in-tree**: `linsolv_amgx`, `linsolv_bicgstab`,
  `linsolv_bos_cpr_gpu`, `linsolv_cusparse_ilu`, `linsolv_cusolv`. The GPU device
  layer is a freshly-written cuSPARSE BSR layer on open-DARTS' own `csr_matrix`,
  so the GPU solvers no longer depend on the proprietary `csr_matrix`.
* **Three new in-tree CPU iterative solvers** (open-source):
  - `linsolv_gmres` — restarted right-preconditioned GMRES / FlexGMRES with
    MGS + Givens; in-tree replacement for `linsolv_bos_gmres`. Adjoint mode
    (`solve_transposed`) wired through to the preconditioner.
  - `linsolv_cpr` — two-stage CPR (Wallis 1983) with HYPRE BoomerAMG on the
    extracted pressure subsystem + HYPRE_ILU(0) on the scalar-expanded full
    system. In-tree replacement for `linsolv_bos_cpr + linsolv_bos_amg`.
    Adjoint **CPRA** (Han et al. 2013) implemented with separate transposed
    AMG and ILU hierarchies.
  - `linsolv_mgr` — HYPRE-MGR via `mgr::CompositionalFlowStrategy`, plus
    BCSR-CPR True-IMPES coarsening, adaptive AMG rebuild, BILU0
    singular-pivot fallback, composite preconditioner mode, local-correction
    smoother. The first viable open-source iterative CPU solver.
* **Unified C++ `linear_solver` interface + name-based registry** — adding a
  solver no longer requires editing an enum or a switch. The old
  `linsolv_iface` / `linsolv_iface_bos<N>` / `linear_solver_base` /
  `mgr::*` hierarchies collapse to one entry point.
* **Per-solver typed config + Python `LinearSolverSpec` classes**:
  `MGRSolverSpec`, `GMRESSolverSpec`, `CPRSolverSpec`, `SuperLUSolverSpec`,
  `PETScSolverSpec`, `PardisoSolverSpec`, `AdaptiveSolverSpec`.
* **PETSc / Pardiso unified into the same Spec mechanism** — `PythonLinearSolver`
  base class; the Newton loop's `isinstance` branch is collapsed into a single
  `_solve_linear_equation()` path. PETSc uses AIJ + a precomputed
  block→scalar values gather (no per-iter BSR→CSR conversion); Pardiso uses
  scalar CSR built once, values-only refresh per iter.
* **§12 unified matrix layout**: `sparsity_pattern`, `dual_array<T>` (host/device
  storage primitive), `block_csr_matrix`, `block_csr_view<N>`; CPU + GPU engine
  Jacobians both migrated; GPU BSR SpMV adapter; polymorphic
  `csr_matrix<1>::to_nb_1(csr_matrix_base*)` for adjoint.
* **CPU default flipped to FGMRES + CPR** (`GMRESSolverSpec(prec=CPRSolverSpec())`).
  Identical iteration counts to MGR across four flow models with lower per-iter
  overhead. `MGRSolverSpec` stays the recommended fallback.
* **One Python knob** for solver selection: `data_ts.linear_solver = <Spec>`.
  The legacy `linear_solver_types` Python enum (`CPU_PETSC_CPR`, `CPU_PETSC_FS`,
  `CPU_PARDISO` in `darts.input.input_data`) and the matching
  `petsc_solve_linear_equation` / `pardiso_solve_linear_equation` branches in
  the Newton loop are **deleted**; use `PETScSolverSpec` / `PardisoSolverSpec`
  via `data_ts.linear_solver`. `data_ts.linear_type` (the C++ engine factory
  enum) is retained only for proprietary `-a` builds, where it selects the
  bos solver.
* **Adaptive / mid-run solver switching** — `AdaptiveSolverSpec` +
  `SolverSwitchContext` + `fallback_on_failure` policy. The solver can change
  during a run based on failure / iter-count signals.
* **Engine factory enum-driven solver creation is neutralised** in the
  open-source build — requires an explicit `engine.set_linear_solver(...)` /
  spec injection. The proprietary build keeps the legacy factory.
* **CI dual-path coverage** — one job exercises the proprietary `-a` artifact
  stack, a sibling job exercises the open-source in-tree registry; the same
  model suite covers both.
* **Dropped**: `samg`, `amg1r5`, `aips`, `linsolv_adgprs_nf` + `lib/AD-GPRS-NF/`
  (commercial / binary-only / dead code; no model in the suite depends on them).
* **Kept proprietary, unchanged**: `linsolv_bos_gmres/cpr/bilu0/amg/fs_cpr`,
  `bos_linear_solver_lib`, the proprietary `csr_matrix`. The MR explicitly does
  **not** touch the proprietary CPU stack; `BOS_SOLVERS_DIR` still wires it in.

What remains queued (out of MR scope, captured in Appendix C):

* `linsolv_bos_fs_cpr` (poromechanics 4-block CPR) — open-source per the header
  but not yet re-implemented; depends on a poromech test target.
* GPU FGMRES — `linsolv_gmres` ported to cuBLAS.
* GMRES + MGR composition triggers HYPRE NaN warnings — the default flip
  to FGMRES + CPR avoids the path; no longer blocking.
* Mechanics-engine Jacobian migration to `block_csr_matrix`.
* MGR-internal CPRA path (`548e224f`'s `applyBCSRCPRTransposePreconditioner` and
  `m_bcsrCPRSourceMatrix` in `solvers/src/linearSolver.cpp`): retire in favour
  of `linsolv_cpr::solve_transposed` after Xiaoming's SPE10 benchmark
  comparison promised on MR #280, 2026-05-29.
* Full GPU side of plan §12 phase C1 — CUDA kernels in `engine_nc_gpu.cu`,
  `engine_nce_g_gpu.cu`, `engine_nc_cg_gpu.cu` still reach into
  `csr_matrix<N>::values_d` / `rows_ptr_d` / `cols_ind_d` / `diag_ind_d` raw
  fields; the `jac_*_d()` shims branch on `OPENDARTS_LINEAR_SOLVERS` so the
  GPU build keeps working, but a follow-up pass should route them through
  the `csr_matrix_base` accessors so the matrix-free path can stop being a
  `csr_matrix_base`.

## 0a. Follow-up work landed 2026-06-02

This section lists the consolidation work landed on top of the original MR
(commits authored after the closeout trio `72b1d901` / `296e40a0` /
`c45ab266`). The TL;DR above describes the MR as originally scoped; the items
here are the post-closeout cleanups that follow naturally from the audit:

* **`linear_solver` interface migration finished**. The legacy `linsolv_iface`
  and the newer orphaned `linear_solver` were the same conceptual interface
  under two names; they are now merged. `linear_solver.hpp` is the canonical
  header (combines the original interface's API with `stats()`);
  `linsolv_iface.hpp` is a back-compat `using linsolv_iface = linear_solver;`
  shim so the ~35 caller files keep compiling unchanged. The
  `linsolv_iface_adapter` (its sole purpose was to bridge the two) is
  removed -- `solvers/{include,src}/linsolv_iface_adapter.{hpp,cpp}` and the
  matching CMakeLists entries are deleted. The registry
  (`solver_registry`, `solver_factories`) now returns
  `shared_ptr<linear_solver>`; pybind exposes the unified handle as
  `LinearSolver` with `LinearSolverInterface` as a Python-side alias for
  back-compat.
* **`scalar_csr_adapter` wired into `linsolv_cpr` and `linsolv_superlu`.**
  When the engine hands a `block_csr_matrix` Jacobian (the canonical layout
  post-§12 phase B), both solvers now bind a `scalar_csr_adapter` once and
  refresh-gather scalar values per setup, replacing the per-Newton
  `csr_matrix<1>::to_nb_1` nested-loop expansion (CPR) and the per-solve
  `new csr_matrix<1>; to_nb_1; delete` allocation cycle (SuperLU). Legacy
  `csr_matrix<N>` callers (tests, GPU, proprietary build) fall back to the
  existing `to_nb_1` path. See `linsolv_cpr::scalar_adapter_` and
  `linsolv_superlu::scalar_adapter_`.
* **CPR hierarchy-reuse policy** mirroring `mgr::SolverParameters`:
  `linsolv_cpr::set_reuse_amg_hierarchy()` and
  `linsolv_cpr::set_adaptive_amg_rebuild(enable, iter_threshold,
  consecutive_bad)`. With reuse on, subsequent setups skip
  `HYPRE_BoomerAMGSetup` and `HYPRE_ILUSetup` (forward + transpose),
  amortising the dominant per-Newton CPR setup cost across timesteps. The
  adaptive variant forces a rebuild after `consecutive_bad` solves over
  `iter_threshold` iterations. Default OFF (preserves pre-policy behaviour
  exactly). Outer Krylov reports the previous solve's iteration count via
  `set_last_outer_iters()`.
* **`mgr::setMatrixFromCSR` structure-skip**. All three overloads
  (`row_ptr`/`col_ind`/`values` pointer; `setBCSRCPRSourceFromCSR`;
  `setMatrixFromVector`) compute `structure_changed` upfront, and gate the
  three structural `std::copy` calls on it. Values are always refreshed.
  Halves the per-Newton MGR ingest cost on the common stable-sparsity path.
* **HYPRE wrapper per-apply allocations hoisted** to member caches. The
  `[0, n_rows)` row-index iota and the per-row column-count buffers fed to
  `HYPRE_IJVectorSetValues` / `HYPRE_IJMatrixSetValues` were heap-allocated
  on every solve / setup; they now live as `row_indices_` / `n_cols_`
  members and grow monotonically. Mirrored across `linsolv_hypre_amg`,
  `linsolv_hypre_ilu`, and `linsolv_cpr` (with one `n_cols_*` per IJ matrix
  in CPR).
* **Adjoint CPRA path -- canonical is `linsolv_cpr::solve_transposed`** (the
  open-source path exposed through `CPRSolverSpec` /
  `GMRESSolverSpec(prec=CPRSolverSpec())` and used by
  `Adjoint_super_engine`'s `cpra` mode, added by Xiaoming in `c135b7c1`).
  The MGR-internal `applyBCSRCPRTransposePreconditioner` (`548e224f`) is
  preserved unchanged pending the SPE10 benchmark comparison Xiaoming
  promised on MR #280, 2026-05-29; once that lands the MGR-internal path
  is to be retired.
* **GPU §12 phase C1 — landed (matrix-free inheritance kept).** The GPU
  engine Jacobian is now `new block_csr_matrix` under
  `OPENDARTS_LINEAR_SOLVERS`, matching CPU; device storage is allocated
  lazily via `dual_array` (no `init_device`). Concrete downstream work:
  - `engine_nc_gpu.cu` — raw `jacobian->rows_ptr_d / values_d / ...` field
    accesses replaced with `get_*_d()` virtuals on `csr_matrix_base` (the
    `assemble_jacobian_array_kernel3` and well-block memcpy call sites),
    and the `matrix_vector_product_d0` / `calc_lin_comb_d` kernels now
    use the `jac_*_d()` shims uniformly. `engine_nc_cg_gpu.cu` and
    `engine_nce_g_gpu.cu` were already shim-routed.
  - Mechanics-engine Jacobian construction migrated: `engine_pm_cpu`,
    `engine_elasticity_cpu`, `engine_super_elastic_cpu` now instantiate
    `block_csr_matrix` under `OPENDARTS_LINEAR_SOLVERS`; the legacy GPU
    `init_device` is gated by `#ifndef OPENDARTS_LINEAR_SOLVERS` to honour
    the lazy device-storage contract.
  - GPU solvers: `linsolv_cusparse_ilu`, `linsolv_amgx`,
    `linsolv_bos_cpr_gpu` now consume `csr_matrix_base*` polymorphically
    (own `cusparseHandle_t` where the matrix no longer provides one;
    accessor-based device pointers; the `convert_to_bs1` AMGX path is
    legacy-typed only). `linsolv_cusolv` got a defensive `dynamic_cast`
    guard with a clear error message -- full `block_csr_matrix` support
    deferred (it's a debug/QR direct solver).
  - **Matrix-free path kept** -- `engine_base_gpu : public csr_matrix_base`
    is unchanged. The matrix-free `assembly_kernel == 13` path needs the
    engine to act as the system matrix and to override
    `matrix_vector_product_d0` with the on-the-fly Jacobian assembly
    kernel; that responsibility cannot move to `block_csr_matrix` without
    re-architecting matrix-free assembly. The `csr_matrix_base` storage-
    accessor overrides on `engine_base_gpu` simply forward to the owned
    Jacobian (now a `block_csr_matrix`).
  - **Remaining (separate follow-ups)**: full GPU validation of MGR/CPR on
    realistic models; AMGX `convert_to_bs1` path migrated to a
    `block_csr_matrix` scalar device view (it currently restricts itself
    to legacy `csr_matrix<N>` inputs); the latent
    `engine_base_gpu.cpp:378` `matrix_vector_product_d_ell` call without
    `convert_to_ELL` under `OPENDARTS_LINEAR_SOLVERS`.
* **Cleanup**: `solvers/{include,src}/linsolv_iface_adapter.{hpp,cpp}`
  deleted; `han2013.pdf` relocated from the repo root to
  `docs/refs/han2013.pdf`; file modes corrected on `engine_base.cpp`,
  `linearSolver.cpp`, `linsolv_mgr.cpp`, `LinearSolver.hpp`,
  `py_engine_base.cpp` (`0755 -> 0644`).
* **Plan staleness fixes**: the original §10/Appendix-C claims that the
  `linsolv_superlu` block-size down-cast bug was queued and that
  `linsolv_cpr::solve_transposed` was a placeholder are both stale --
  `548e224f` fixed the former, `linsolv_cpr` already implemented the
  latter; this section captures that.

---

## 1. Objective

Deliver a **unified, extensible linear-solver subsystem** for open-DARTS that:

1. **Absorbs only the GPU solvers** from `darts-linear-solvers` into the open-DARTS repository
   as open-source code. The CPU `bos` solvers **remain proprietary** and continue to be linked
   externally via `BOS_SOLVERS_DIR` — running models with the proprietary CPU solvers stays
   fully supported.
2. Replaces the three disjoint C++ solver type systems and three Python selection mechanisms
   with **one C++ interface, one registry, one per-solver config object, one dispatch path**.
3. Exposes every solver as a **documented Python class** with all parameters — Python is the
   primary configuration surface (PETSc/Pardiso are the model for this).
4. Makes the subsystem **extensible** (add a solver without editing an enum or a switch) and
   **adaptive** (change solver type during a run).
5. Makes the in-tree **FGMRES + CPR** (`linsolv_gmres + linsolv_cpr`) the default
   CPU solver -- the open-source equivalent of the proprietary
   `linsolv_bos_gmres + linsolv_bos_cpr_amg` stack. `MGRSolverSpec` (HYPRE-MGR)
   remains the recommended fallback. The GPU default stays the legacy AMGX-CPR
   path (in-tree BiCGStab + cuSPARSE-ILU when AMGX is absent).
6. Is fully covered by CI/CD, including cross-solver comparison.

---

## 2. Background

**Before this MR**, the open-source CPU build had **no working iterative solver**:
the `linsolv_bos_*` classes in `solvers/` were stubs printing `"NOT IMPLEMENTED"`;
the real implementations lived in the proprietary `darts-linear-solvers/`. The
only working open-source CPU solver was SuperLU (direct). The open-source GPU
solver stack also lived only in `darts-linear-solvers`, so GPU builds required
`BOS_SOLVERS_DIR` to be wired up.

**This MR** delivers three new open-source iterative CPU solvers in-tree
(`linsolv_mgr`, `linsolv_gmres`, `linsolv_cpr` with CPRA adjoint), the five GPU
solvers brought in-tree on a freshly-written cuSPARSE BSR layer, the unified
`linear_solver` C++ interface + registry, and the `LinearSolverSpec` Python
class hierarchy. The CPU default becomes `GMRESSolverSpec(prec=CPRSolverSpec())`
-- iteration counts identical to MGR on flow models, lower per-iter overhead
-- with MGR as the recommended fallback. GPU builds no longer require the
external proprietary library.

---

## 3. Scope of this MR

### In scope
- Absorb the **5 open-source GPU solvers**: `linsolv_amgx`, `linsolv_bicgstab`,
  `linsolv_bos_cpr_gpu`, `linsolv_cusparse_ilu`, `linsolv_cusolv`.
- **AMGX** added as a thirdparty submodule (`thirdparty/AMGX`); GPU builds with
  AMGX absent fall back to BiCGStab + cuSPARSE-ILU.
- Reimplement a **GPU device layer on open-DARTS' own `csr_matrix`** (the GPU solvers'
  dependency on the proprietary matrix is removed by reimplementation — see §7.3).
- Add **three new in-tree open-source CPU iterative solvers**:
  - `linsolv_gmres` (restart-GMRES / FlexGMRES + adjoint),
  - `linsolv_cpr` (two-stage CPR + Han-2013 CPRA transpose),
  - `linsolv_mgr` (HYPRE-MGR via `CompositionalFlowStrategy`, with BCSR-CPR
    True-IMPES coarsening, adaptive AMG rebuild, BILU0 fallback, composite
    preconditioner mode, local-correction smoother).
- Unified C++ `linear_solver` interface + registry + typed config structs.
- Python `LinearSolverSpec` class hierarchy (`MGRSolverSpec`, `GMRESSolverSpec`,
  `CPRSolverSpec`, `SuperLUSolverSpec`, `PETScSolverSpec`, `PardisoSolverSpec`,
  `AdaptiveSolverSpec`); unified dispatch incl. PETSc / Pardiso.
- **§12 unified matrix layout**: `sparsity_pattern`, `dual_array<T>`,
  `block_csr_matrix`, `block_csr_view<N>`, GPU BSR SpMV adapter; CPU + GPU
  engine Jacobians both migrated.
- CPU default → FGMRES + CPR (open-source `linsolv_gmres + linsolv_cpr`);
  centralized default policy. MGR remains the recommended fallback.
- Adaptive / mid-run solver switching.
- CI: GPU buildable from in-tree source; dual-path coverage (proprietary `-a`
  vs open-source registry) over the model suite.

### Out of scope (deliberately kept as-is)
- The **proprietary CPU `bos` solvers** (`linsolv_bos_gmres/cpr/bilu0/amg/fs_cpr`): stay
  external, linked via `BOS_SOLVERS_DIR` / the `-a` artifact path. The `OPENDARTS_LINEAR_SOLVERS`
  macro, the `.h`/`.hpp` header duality, and `thirdparty/thirdparty_bos_solvers.cmake` **remain**.
- The proprietary `bos_linear_solver_lib` engine and its `csr_matrix` — not moved, not touched.

### Dropped entirely (decision 2)
`samg` (commercial, source absent), `amg1r5` (binary-only, no source), `aips` (binary-only),
`linsolv_adgprs_nf` + `lib/AD-GPRS-NF/` (Stanford SUPRI-B academic license — not open-source,
and dead code: `WITH_ADGPRS_NF` is never defined).

---

## 4. Resolved decisions

| # | Decision |
|---|----------|
| 1 | **Break `linear_solver_t` immediately** — no deprecated-alias period. Engine solver selection moves to the registry (name-based). |
| 2 | **Drop `samg` / `amg1r5` / `aips`** (and `adgprs_nf`). No model in the maintained suite depends on them. |
| 3 | **HYPRE: track the latest release, no version pin** — remove any pinned-version logic in build scripts / submodule. |
| 4 | **Stacked commits within `xiaoming/add-mgr`**, one MR. Each commit independently reviewable and build-green. |
| 5 | **Unified solver handling, Python-centric** — the `LinearSolverSpec` Python classes are the primary surface; PETSc/Pardiso are first-class examples of Python-side solvers, handled by the same mechanism. |
| 6 | **OD-6: assert-on-mismatch** — guard the `linsolv_iface_bos` block-size downcast with `assert(A->n_row_size == N)`; defer the type-erased-view redesign. |
| 7 (new) | **GPU ↔ proprietary `csr_matrix` entanglement: reimplement** — add a GPU device layer to open-DARTS' own `csr_matrix` with freshly written cuSPARSE code; do **not** move the proprietary `csr_matrix`. |

---

## 5. Current-state problems being fixed

| # | Problem | Evidence |
|---|---------|----------|
| 1 | Three disjoint C++ hierarchies: `linsolv_iface`/`linsolv_iface_bos<N>`, `linear_solver_base`/`linear_solver_prop`, `mgr::*`. `linsolv_iface_bos` does an **unsafe `csr_matrix_base*`→`csr_matrix<N>*` downcast**. | `solvers/include/linsolv_iface*.hpp`, `mgr` namespace |
| 2 | Three Python solver-selection mechanisms; Newton loop branches on `isinstance`. | `darts/models/darts_model.py:832-848`; duplicate in `darts/pipes/viz/plot_live.py:902-914` |
| 3 | Flat `linear_solver_t` enum mixes CPU+GPU with an ordering contract; `CPU_GMRES_MGR=17` sits after the GPU block and is misclassified as GPU by `>= GPU_GMRES_CPR_AMG`. | `engines/src/globals.h:68-88` |
| 4 | Adding a solver = edit enum + hand-written `switch` + pybind binding + ordering. | `engines/src/engine_base.h:709-920` |
| 5 | Parameters scattered (`sim_params` fields, `linear_params` vector, MGR config, PETSc CLI strings, nothing for Pardiso). | — |
| 6 | Solver created once at `init()`; no real mid-run change except `set_linear_solver()`. | `engines/src/engine_base.h:143-165` |
| 7 | Default solver decided in three places. | `globals.h:116-120`, `darts_model.py:187-188`, `engine_base_gpu.h:160-163` |
| 8 | GPU solver stack is **not buildable from open-DARTS sources** — exists only in the external repo. | `engine_base_gpu.h`, `thirdparty/thirdparty_bos_solvers.cmake` |
| 9 | No cross-solver comparison testing. | `.cicd/jobs/*.yml`, `models/run_test_suite2.py` |

---

## 6. Requirements

- **R1** — One C++ `linear_solver` interface; all solvers (CPU, GPU, MGR, direct, and the
  proprietary `bos` via an adapter) are reachable through it.
- **R2** — Solver registry: register by name; create by name + config. No enum/switch edit to
  add one. A solver absent in a given build (e.g. proprietary CPU `bos` when `BOS_SOLVERS_DIR`
  is unset) fails with a clear, explicit error.
- **R3** — One typed config object per solver (C++ struct ⇄ Python class), fully documented.
- **R4** — One Python knob: `data_ts.linear_solver = <SolverSpec>`. PETSc & Pardiso unified into
  the same mechanism.
- **R5** — Default policy in one place: CPU ⇒ `GMRESSolverSpec(prec=CPRSolverSpec())` (FGMRES + CPR); GPU ⇒ legacy AMGX-CPR (in-tree BiCGStab + cuSPARSE-ILU when AMGX absent). `MGRSolverSpec` is the recommended fallback on CPU.
- **R6** — Mid-run / adaptive solver switching is a first-class feature.
- **R7** — The open-source GPU solvers live in-tree; **GPU builds without the external library**.
- **R8** — The proprietary CPU `bos` path keeps working unchanged (`BOS_SOLVERS_DIR` / `-a`).
- **R9** — CI: one open-source build path; a job verifies cross-solver agreement.
- **R10** — Every solver and parameter documented (docstrings + Sphinx page).

---

## 7. Target architecture

### 7.1 C++ — one interface, one registry, typed config

```cpp
struct solver_stats { int iterations; double residual; bool converged;
                      double setup_time, solve_time; };

// Replaces linsolv_iface + linsolv_iface_bos + linear_solver_base + the mgr shims.
class linear_solver {
public:
  virtual ~linear_solver() = default;
  virtual int  init (csr_matrix_base* A, const solver_config& cfg) = 0;
  virtual int  setup(csr_matrix_base* A) = 0;     // per Newton iteration
  virtual int  solve(double* rhs, double* x) = 0;
  virtual void set_prec(linear_solver* p) {}      // optional
  virtual solver_stats stats() const = 0;
};

using solver_factory =
    std::function<std::unique_ptr<linear_solver>(const solver_config&, int block_size)>;
void register_solver(const std::string& name, solver_factory);
std::unique_ptr<linear_solver>
create_linear_solver(const std::string& name, const solver_config&, int block_size);
```

- Concrete solvers stay block-size templated internally; the block-size downcast is **guarded
  with `assert(A->n_row_size == N)`** (decision 6).
- `mgr::LinearSolver` becomes a real `linear_solver`; `mgr::int_t/real_type` alias the
  `opendarts::config` types.
- `linear_solver_base`/`linear_solver_prop` retired; convergence data lives in `solver_stats`.
- The legacy `linear_solver_t` enum is **removed** (decision 1). The proprietary CPU `bos`
  solvers — still `linsolv_iface`-based in the external lib — are reached through a thin
  `linsolv_iface → linear_solver` **adapter**, registered only when `BOS_SOLVERS_DIR` is set.

### 7.2 `solvers/` repo layout after the MR

```
solvers/
  interface/   linear_solver base, solver_config, solver registry, linsolv_iface adapter
  bos/         CPU bos STUBS only — proprietary impls stay external (BOS_SOLVERS_DIR). Unchanged.
  hypre/       linsolv_mgr, mgr::LinearSolver, MGR strategies, hypre_amg, hypre_ilu
  direct/      SuperLU wrapper
  gpu/         NEW (open-source, in-tree): linsolv_amgx, linsolv_bicgstab,
               linsolv_bos_cpr_gpu, linsolv_cusparse_ilu, linsolv_cusolv  — built under WITH_GPU
  pybind11/    the `solvers` Python module — registry + every solver + configs
```

### 7.3 GPU absorption — reimplement on open-DARTS' `csr_matrix` (decision 7)

The 5 GPU solvers depend on the proprietary `csr_matrix<N>` and its GPU device methods
(cuSPARSE SpMV, ELL conversion, device storage), on `linear_solver_base`, and on proprietary
support headers. The proprietary code is **not** moved. Instead:

1. **open-DARTS' `csr_matrix` (`solvers/include/csr_matrix.hpp`) gains a GPU device layer**,
   written fresh — not copied from the proprietary `csr_matrix.cpp`:
   - device storage (block-CSR device pointers, ELL buffers, `cusparseHandle_t`/descriptors);
   - `init_device` / `free_device`, `copy_struct_to_device` / `copy_values_to_device`,
     `convert_to_ELL`, `matrix_vector_product_d`, `calc_lin_comb_d`.
   - cuSPARSE-backed (`cusparseDbsrmv` etc.) — standard, independently implemented; in a new
     `solvers/src/csr_matrix_gpu.cu`, compiled only under `WITH_GPU`.
2. **The 5 GPU wrappers are ported**: base class → open-DARTS `linsolv_iface`/`linear_solver`
   (replacing proprietary `linear_solver_base`); matrix type → open-DARTS `csr_matrix<N>`;
   proprietary support headers (`matrix_macroses.h`, `omp_tools.h`, `debug_tools.h`) replaced
   with open-DARTS equivalents or small inlined helpers.
3. **Headers relicensed** to the open-DARTS license, original authorship credited (§8).
4. The engine GPU path (`engine_base_gpu.h`) is rewired to construct the in-tree GPU solvers
   through the registry.

Net effect: the GPU stack is **100% open-source** and builds from in-tree source; the
proprietary `csr_matrix` is left entirely alone.

### 7.4 Python — a `LinearSolverSpec` class per solver

In a new `darts/solvers/` Python package (the compiled extension is renamed `_solvers` and
re-exported, avoiding the current `darts/solvers.so` name collision):

```python
@dataclass
class LinearSolverSpec:                       # base
    tolerance: float = 1e-5
    max_iterations: int = 50
    print_level: int = 0
    def build(self, block_size: int, platform: str): ...   # -> C++ linear_solver

@dataclass
class MGRLevelSpec:                           # one HYPRE MGR reduction level
    keep_labels: list[int] = field(default_factory=list)
    frelax_type: int = FRelax.NONE;   frelax_iters: int = 0
    interp_type: int = Interp.INJECTION
    restrict_type: int = Restrict.BLOCK_COL_LUMPED
    coarse_method: int = Coarse.GALERKIN
    smoother_type: int = Smoother.HYPRE_ILU;  smoother_iters: int = 1

@dataclass
class MGRSolverSpec(LinearSolverSpec):        # full HYPRE MGR surface
    kdim: int = 30
    use_flex_gmres: bool = True
    use_physics_scaling: bool = True
    enable_well_level: bool = False
    enable_composition_level: bool = False
    pressure_level:    MGRLevelSpec = field(default_factory=MGRLevelSpec)
    well_level:        MGRLevelSpec = field(default_factory=MGRLevelSpec)
    composition_level: MGRLevelSpec = field(default_factory=MGRLevelSpec)
    custom_levels:     list[MGRLevelSpec] = field(default_factory=list)

class SuperLUSolverSpec(LinearSolverSpec):  ...
class HypreAMGSolverSpec(LinearSolverSpec): ...
class AMGXSolverSpec(LinearSolverSpec):     ...   # GPU
class BiCGStabGPUSolverSpec(LinearSolverSpec): ...# GPU
class GPUCPRSolverSpec(LinearSolverSpec):   ...   # GPU (linsolv_bos_cpr_gpu)
class BOSSolverSpec(LinearSolverSpec):  variant: str = "cpr_amg"   # proprietary, external
class PETScSolverSpec(LinearSolverSpec): variant: str = "cpr"      # cpr | fs
class PardisoSolverSpec(LinearSolverSpec): ...
```

HYPRE integer codes become documented `IntEnum`s (`darts/solvers/enums.py`). Model usage:
`self.data_ts.linear_solver = MGRSolverSpec(tolerance=1e-4, kdim=50)`.

### 7.5 Unified dispatch

A Python `LinearSolver` protocol — `setup(jac)` / `solve(rhs, dx)` / `stats()`. C++ solvers
satisfy it via pybind; PETSc/Pardiso are wrapped (from the existing
`petsc_solve_linear_equation` / `pypardiso.spsolve` code) into `PETScLinearSolver` /
`PardisoLinearSolver`. The Newton loop becomes one call — `self.linear_solver.solve_system()` —
and the `isinstance` branch at `darts_model.py:832-848` plus its `plot_live.py` duplicate are
deleted.

### 7.6 Default-solver policy

```python
def default_linear_solver(platform: str = "cpu") -> LinearSolverSpec:
    if platform.lower() == "gpu":
        # GPU default is wired in the GPU engine factory (engine_base_gpu),
        # not through a LinearSolverSpec.
        raise NotImplementedError(...)
    return GMRESSolverSpec(restart=50, prec=CPRSolverSpec())
```

Single source of truth ([`darts/solvers/specs.py`](darts/solvers/specs.py)).
Validated against `MGRSolverSpec` across four flow models -- identical Newton /
linear iteration counts, lower per-iter overhead. `MGRSolverSpec` remains the
recommended fallback (set `data_ts.linear_solver = MGRSolverSpec()`).

### 7.7 Adaptive / mid-run switching

`engine.set_linear_solver()` already swaps + re-`init()`s a solver; commit `9139da28` made MGR
`setup()` rebuild-safe. `AdaptiveSolverSpec(candidates=[...], policy=callable)` lets `DartsModel`
invoke `policy` between timesteps and rebuild on change. Stock policies: fall back MGR→SuperLU
after N linear failures; escalate `kdim`/levels on rising iteration counts.

---

## 8. Licensing

- The 5 GPU wrapper files carry `Copyright © 2018-2020 MARK KHAIT — ALL RIGHTS RESERVED`
  headers. The project holds the right to release them open-source (per project decision); the
  MR **replaces every header** with the standard open-DARTS license header, crediting original
  authorship in `AUTHORS`/`CHANGELOG`.
- **No proprietary `csr_matrix` / `bos` engine code is moved** — the GPU device layer is written
  fresh (§7.3), so no relicensing of proprietary infrastructure is required.
- `lib/AD-GPRS-NF/` (Stanford SUPRI-B academic copyright) and the `aips`/`amg1r5` binary blobs
  are **excluded** — no grantable open-source license.
- AMGX is BSD-3-Clause — GPL-compatible, no obstacle.

---

## 9. Build system & CI

### Build
- **Keep** `BOS_SOLVERS_DIR` / `-a` / `OPENDARTS_LINEAR_SOLVERS` / `thirdparty_bos_solvers.cmake`
  — the proprietary CPU `bos` path is unchanged.
- **Decouple GPU from `BOS_SOLVERS_DIR`**: `OPENDARTS_CONFIG=GPU` no longer requires the
  external library (remove the `FATAL_ERROR` at `CMakeLists.txt:96-99` for the GPU case). GPU
  builds from in-tree `solvers/gpu/`. `BOS_SOLVERS_DIR` becomes *optional* for GPU (adds the
  proprietary CPU solvers on top, if desired).
- **AMGX** → `thirdparty/AMGX` git submodule, built under `WITH_GPU` (same pattern as HYPRE).
- **HYPRE** — track the latest release; remove version pinning (decision 3).
- New CMake under `solvers/gpu/`; `WITH_GPU` gates the GPU device layer + wrappers.

### CI
- GPU job builds AMGX + GPU solvers **from in-tree source**; drop `GSELINSOLVERSPATH` and the
  `darts-linear-solvers` clone from `singularity.def` for the GPU path.
- Keep the `-a` SMB-artifact path for the MT / proprietary-CPU-`bos` build.
- Add a **solver-comparison job**: `darts/solvers/benchmark.py::run_with_solvers(model,[specs])`
  runs canonical models under each solver and asserts cross-solver agreement within tolerance.
- Wire the 19 `solvers/` C++ ctests into a dedicated CI job.

---

## 10. Stacked-commit sequence (the roadmap)

| Commit | Content | Green gate |
|--------|---------|-----------|
| **C1** | This plan document. | — |
| **C2** | Unified C++ `linear_solver` interface + registry + `solver_config` structs; `mgr::LinearSolver` implements it; `linsolv_iface`→`linear_solver` adapter. Additive. | builds (ST/MT) |
| **C3** | Remove `linear_solver_t`; engine solver selection → registry (name-based). Proprietary `bos` registered via adapter when `BOS_SOLVERS_DIR` set. | ctests + model suite |
| **C4** | GPU device layer on open-DARTS `csr_matrix` (`csr_matrix_gpu.cu`, fresh cuSPARSE). | builds under `WITH_GPU` |
| **C5** | Port the 5 GPU wrappers onto open-DARTS `csr_matrix`/`linear_solver`; relicense headers; drop `adgprs_nf`/`aips`. | GPU build |
| **C6** | `thirdparty/AMGX` submodule + `WITH_GPU` CMake; decouple GPU from `BOS_SOLVERS_DIR`; GPU buildable from in-tree source. | GPU build + GPU model run |
| **C7** | `darts/solvers/` Python package — `LinearSolverSpec` hierarchy, `enums.py`; rename compiled module → `_solvers`. | model suite |
| **C8** | Unified dispatch — wrap PETSc/Pardiso; collapse the Newton-loop branch; `default_linear_solver()`; **flip CPU default → MGR**; migrate models. | model suite |
| **C9** | `AdaptiveSolverSpec` + policy hook; mid-run switching. | new tests |
| **C10** | CI: GPU-from-source job; solver-comparison job; wire ctests; keep `-a` proprietary path. | full pipeline |
| **C11** | Docs (docstrings + Sphinx); OD-6 assert guard; remove dead code (`solvers/linear_solvers/`, `engines/lib/CMakeLists.txt`). | — |

---

## 11. Risks & remaining open items

- **MR size** — large; strict per-commit reviewability is essential (decision 4: stacked).
- **GPU device-layer reimplementation** (C4) — the fresh cuSPARSE SpMV/ELL code must match the
  numerical behaviour the GPU solvers expect; needs careful testing against the external-lib
  GPU results before C6 flips CI.
- **CI GPU capacity** — AMGX submodule (~834 MB) + CUDA toolchain build; image size grows.
- **Engine GPU matrix type** — in a GPU build the engine must assemble into the open-DARTS
  `csr_matrix` (now GPU-capable); confirm `engine_base_gpu.*` has no residual dependence on the
  proprietary matrix.
- **Open item OI-1** — confirm no maintained model uses `samg`/`amg1r5`/`aips`/`adgprs_nf`.
- **Open item OI-2** — AMGX submodule: pin to a specific release tag or track a branch
  (HYPRE is unpinned per decision 3; AMGX pinning is a separate call — recommend pinning AMGX
  to a known-good release for build stability).
- **Open item OI-3** — exact home of the `darts/solvers/` package vs the renamed `_solvers`
  extension (build-system rename in C7).
- **Open item OI-4** — the unified matrix layout (§12) is design-gated: the in-tree GPU
  build is intentionally **not** completed with a tactical bridge; it lands as a
  consequence of the §12 migration. `Decision 8` (§12.2) should be promoted into §4.

---

## 12. Unified matrix layout & backend adapters (axis-1 design)

*Added 2026-05-19. Supersedes §7.3's "rewire the engine GPU path" item and reshapes the
item-7 `csr_matrix` GPU layer. Design-first: per OI-4 the GPU build is deliberately not
rushed until this contract is settled and implemented.*

### 12.1 Problem

The engine assembles the Jacobian/residual every Newton iteration; the linear solvers
consume it. Today there are *two* un-unified matrix formats — the open-DARTS
`csr_matrix` (`std::vector` host storage, abstract base) and the proprietary
`csr_matrix_base` (raw host + device pointer members) — and the GPU engine is written
against the latter. No single layout is simultaneously (a) optimal to *assemble into*
and (b) cheap to *hand to* HYPRE / bos / PETSc / Pardiso / cuSPARSE+AMGX. The GPU-build
breakage is a symptom of this gap, not an isolated bug.

### 12.2 Scope decision — one canonical format, adapters at the edge

**Decision 8.** The engine assembles into **one** canonical in-memory format; each
backend receives it through a thin adapter. The rejected alternative — a pluggable
per-backend storage policy — would multiply the assembly path by the backend count,
leak backend knowledge into the engine, and forfeit a shared sparsity structure.
Single-format wins on every count that matters for performance and scalability:

- one assembly kernel — a single code path to optimise, vectorise, and port to GPU;
- **zero-copy** for the block-native backends (bos, PETSc BAIJ, cuSPARSE BSR, AMGX) —
  i.e. every performance-critical GPU path;
- a **single, structure-cached** conversion for the scalar-CSR backends (HYPRE, Pardiso);
- one sparsity structure shared by the Jacobian, the CPR pressure matrix, and the cached
  CSR expansion.

### 12.3 Canonical format — block-CSR (BSR)

Block-CSR with `nb × nb` dense blocks (`nb` = equations per cell = `N_VARS`):

- **structure** — `row_ptr[n_block_rows+1]`, `col_ind[nnzb]`, `diag_ind[n_block_rows]`
  (location of each row's diagonal block — always stored), `row_thread_starts` (parallel
  partition). 0-based (HYPRE/PETSc/cuSPARSE/AMGX native; Pardiso via `iparm`).
- **values** — `nnzb · nb · nb` contiguous doubles, **row-major within each block**
  (matches `CUSPARSE_DIRECTION_ROW` and PETSc BAIJ).
- BSR is the common denominator: bos / PETSc-BAIJ / cuSPARSE / AMGX are block-CSR
  natively; HYPRE / Pardiso take a structural BSR→CSR expansion.

### 12.4 Layered class design

Four separated concerns — each independently testable, no virtual dispatch in the
assembly hot path:

```cpp
// (1) Structure — built once from mesh connectivity, immutable across a Newton solve.
//     Ref-counted: shared by the Jacobian, the CPR pressure matrix, and the cached
//     scalar-CSR expansion.
struct sparsity_pattern {
  index_t n_block_rows, n_block_cols, nnzb;
  dual_array<index_t> row_ptr, col_ind, diag_ind, row_thread_starts;
  index_t global_row_start = 0, global_n_rows = 0;   // distributed-ready (§12.7)
  mutable std::shared_ptr<csr_expansion> csr_view;    // lazy, cached BSR->CSR structure
};

// (2) Host/device storage primitive — §12.5.
template <class T> class dual_array { /* ... */ };

// (3) block_csr_matrix — CONCRETE, non-templated. What the engine assembles into
//     and what every solver receives. Block size is a runtime field.
class block_csr_matrix {
  std::shared_ptr<sparsity_pattern> structure_;
  dual_array<mat_float> values_;
  int block_size_;
 public:
  // raw views — assembly hot path & adapters; no bounds checks, no virtuals
  mat_float* values()        noexcept;
  mat_float* values_device() noexcept;        // WITH_GPU
  const index_t* row_ptr() const noexcept;    // ... col_ind / diag_ind
  const sparsity_pattern& structure() const noexcept;
};

// (4) block_csr_view<N> — zero-overhead compile-time-`nb` lens for the templated
//     engine. Non-owning: it stores a pointer to a block_csr_matrix, no storage.
template <uint8_t N> class block_csr_view {
  block_csr_matrix* m_;
 public:
  explicit block_csr_view(block_csr_matrix& m) : m_(&m) {}
  mat_float* block(index_t jb) const noexcept;        // fully inlined, N a constant
};
```

This **inverts** the current ownership (today: abstract `csr_matrix_base`,
`csr_matrix<N>` owns storage). A concrete storage-owning class is what lets a
`block_csr_matrix&` be read by any backend with no knowledge of `nb`, and it
dissolves the `std::vector`-vs-raw-pointer clash: storage lives once,
`block_csr_view<N>` is a typed accessor over it.

`block_csr_matrix` / `block_csr_view<N>` are introduced **additively**, beside
the legacy `csr_matrix<N>`: the build stays green at every step while the
engine and solvers migrate, and `csr_matrix` / `csr_matrix_base` are retired
last. (A concrete class named `csr_matrix_base` would also be a misnomer; the
`block_csr_*` names are explicit about the format.)

### 12.5 Host/device storage — `dual_array<T>`

`dual_array<T>` owns a host buffer and (lazily, `WITH_GPU` only) a device buffer, tracks
which side is dirty, and exposes `sync_to_device()` / `sync_to_host()`,
`host_data()` / `device_data()`. It is the **single home** of every `cudaMalloc` /
`cudaMemcpy`; the rest of the matrix code is device-agnostic. Under a non-GPU build it
degrades to the host buffer with zero overhead.

This is the deliberate **axis-2 seam** (see the RAJA/Kokkos evaluation): `dual_array`
has the shape of a `Kokkos::DualView` / CHAI `ManagedArray`. If a performance-portability
layer is adopted later, it is swapped in *behind this type* — the matrix public API and
every adapter are unaffected.

### 12.6 Backend adapters

A backend adapter takes a `csr_matrix_base&`, presents the backend's expected handle, and
is owned by the solver wrapper (built in the wrapper's `setup()`):

| Adapter | Mechanism | Copy? |
|---|---|---|
| bos | block-CSR is identical | zero-copy |
| cuSPARSE BSR | device pointers + descriptor | zero-copy |
| AMGX | `AMGX_matrix_upload_all` (block dims) | device upload only |
| PETSc | `MatCreateBAIJWithArrays` (`bs = nb`) | zero-copy wrap |
| HYPRE | BSR→CSR expand → IJ / ParCSR | values-only per setup* |
| Pardiso | BSR→CSR expand, `iparm` index base | values-only per setup* |

*The scalar-CSR **structure** is a pure function of the BSR sparsity pattern → built once
and cached on `sparsity_pattern::csr_view`. Each Newton `setup()` then performs only the
fixed `nnzb·nb·nb → nnz` value gather — never a structural rebuild.

### 12.7 Assembly performance & scalability

- **Block locality** — each `nb×nb` block is one contiguous span; assembly streams it,
  no scatter. `diag_ind` gives O(1) diagonal-block access (Newton chop, CPR `D_ss`).
- **Parallel partition** — `row_thread_starts` gives each thread a contiguous,
  write-disjoint row (hence values) range — lock-free CPU assembly, one thread/warp per
  row on GPU, identical array layout on both sides.
- **One assembly function** — templated on `nb`, instantiated per execution space; this
  is the future plug point for an axis-2 portability layer, but the *format* is fixed
  here independently of that choice.
- **Distributed-ready** — `sparsity_pattern` carries (initially trivial) global row
  ownership so HYPRE ParCSR / PETSc MPIBAIJ adapters need no later format change.

### 12.8 Engineering standards

RAII storage ownership (`dual_array` frees device memory in its destructor);
`sparsity_pattern` shared via `shared_ptr`; matrices movable, non-copyable (explicit
`clone()`); `[[nodiscard]]` + a status enum on fallible operations (no bare `int`);
`assert` for invariants (block-size match) compiled out in Release; the entire device
layer behind `#ifdef WITH_GPU` with the host-only `dual_array` fallback; no exceptions
across the CUDA boundary; const-correct accessors. Each layer 12.4 (1)–(4) is
unit-tested in isolation; adapters are tested by round-tripping a known matrix through
each backend and checking an SpMV against a reference.

### 12.9 Migration roadmap (post-MR, design-gated)

1. `dual_array<T>` — host-only + `WITH_GPU` device buffer; unit tests. **[done]**
2. `sparsity_pattern` + cached BSR→CSR `csr_expansion`; unit tests. **[done]**
3. `block_csr_matrix` (concrete, owns structure + values) and `block_csr_view<N>`
   (typed non-owning view), added **additively** beside the legacy `csr_matrix<N>`;
   unit tests. **[done]**
4. Backend adapters: bos & cuSPARSE/AMGX (zero-copy) first, then PETSc-BAIJ, then the
   HYPRE / Pardiso scalar-CSR expansion.
5. Engine: assemble through `block_csr_view<N_VARS>`; make the Jacobian a **member**
   (composition) — drop `engine_base_gpu : public csr_matrix_base`.
6. Wire adapters into the solver wrappers behind the existing registry (§7.1); then
   retire the legacy `csr_matrix<N>` / `csr_matrix_base` (incl. the item-7 device
   layer, now subsumed by `dual_array`).

Each step is an independently reviewable commit. The GPU build is completed as a
*consequence* of steps 3–5, not as a separate patch.

### 12.10 Steps 5–6 migration sub-plan

Steps 1–4 are done (the unified-matrix foundation, additive). Steps 5–6 migrate
the engine and the solvers onto it. Key facts that shape the plan:

- **The layout is unchanged.** `block_csr_matrix` is the *same* block-CSR as the
  legacy `csr_matrix<N>` (§12.3) — identical `values` / `row_ptr` / `col_ind` /
  `diag_ind`. So **no assembly arithmetic changes**; the migration is a type +
  accessor re-pointing plus completing `block_csr_matrix`'s operation surface.
- **Open-source paths only.** The migration touches the `OPENDARTS_LINEAR_SOLVERS`
  code paths. A bos build keeps assembling into the proprietary `csr_matrix`;
  `block_csr_matrix` replaces the *open-DARTS* `csr_matrix<N>` under the existing
  `#ifdef OPENDARTS_LINEAR_SOLVERS` switches.
- The dominant consumer call is `->get_values()` / `->get_rows_ptr()` /
  `get_cols_ind` / `get_diag_ind` / `get_row_thread_starts` (~150 sites). Giving
  `block_csr_matrix` those exact accessor names makes that code source-compatible,
  shrinking the build-gated cut to type declarations.

**Phase A — additive (build stays green; each item committable + unit-tested)**

- A1. Block SpMV on `block_csr_view<N>`: `matrix_vector_product` (r += A·v),
  `matrix_vector_product_t`, `calc_lin_comb` (r = αAu + βv) — compile-time N.
- A2. Legacy-compatible accessors on `block_csr_matrix`: `get_values()`,
  `get_rows_ptr()`, `get_cols_ind()`, `get_diag_ind()`, `get_row_thread_starts()`
  — same names/signatures as `csr_matrix_base`.
- A3. cuSPARSE BSR SpMV as a free function / adapter over `block_csr_matrix`
  (replaces the item-7 `csr_matrix::matrix_vector_product_d`).
- A4. `sparsity_pattern` builder from `conn_mesh` connectivity (replaces the
  `csr_matrix::init`-from-structure path the engine uses once per run).
- A5. Matrix IO (`export_matrix_to_file`) as a free function — debug dumps only.

**Phase B — the atomic cut (build-gated; one focused pass, then a CPU build)**

- B1. `linear_solver` / `linsolv_iface`: `init/setup(csr_matrix_base*)` →
  `block_csr_matrix&`.
- B2. Each open-source solver wrapper (`linsolv_mgr`, `linsolv_superlu`,
  `linsolv_hypre_amg`, `linsolv_hypre_ilu`, the 5 GPU wrappers): matrix type →
  `block_csr_matrix`; HYPRE/MGR consume `scalar_csr_adapter`; GPU wrappers
  consume the `block_csr_matrix` device pointers.
- B3. Engine: the `engine_base` / `engine_base_gpu` Jacobian member
  `csr_matrix<N>` → `block_csr_matrix`; structure built once via A4; assembly
  through `block_csr_view<N_VARS>`.
- B4. Drop `engine_base_gpu : public csr_matrix_base` — composition: the engine
  *has-a* Jacobian.
- B5. CPU build, iterate; then GPU build, iterate.

**Phase C — cleanup** (status after the engine migration)

- C1. ~~Retire `csr_matrix<N>` / `csr_matrix_base`.~~ **Re-scoped.**
  `csr_matrix_base` is *kept* — it is the unified polymorphic interface that
  `block_csr_matrix` (and the GPU engine, for the matrix-free path) implement;
  the solver wrappers all take `csr_matrix_base*`. `csr_matrix<N>` cannot be
  retired yet — it is still load-bearing in three places:
  - the GPU engine Jacobian (`engine_base_gpu::init_base`) still constructs
    `csr_matrix<N>`; it carries the proven cuSPARSE BSR device layer. A future
    pass migrates it to `block_csr_matrix` once `gpu_bsr_spmv` supports
    in-place device assembly.
  - the mechanics engines (`engine_pm`, `engine_elasticity`,
    `engine_super_elastic`) still construct `csr_matrix<N>`. A trial migration
    to `block_csr_matrix` regressed `engine_pm` (segfault), so it was reverted;
    the mechanics Jacobian structure needs its own investigation.
  - the adjoint scalar scratch matrices (`csr_matrix<1>` `Temp`/`T1`/`T2`,
    `dg_dx_*`) and the `linsolv_iface_bos` down-casts.
- C1-done. The adjoint block->scalar expansion was UB after the migration
  (`to_nb_1(static_cast<csr_matrix<N>*>(Jacobian))` on a `block_csr_matrix`):
  fixed with a polymorphic `csr_matrix<1>::to_nb_1(csr_matrix_base*)`.
- C2. `tests/cpp/unit/linear_solvers` — the new-type tests (`block_csr_matrix`,
  `dual_array`, `sparsity_pattern`, `scalar_csr_adapter`, `block_csr_view_spmv`)
  exist and pass; `to_nb_1` test extended with a `csr_matrix_base*` case.

---

## 13. Unified PETSc / Pardiso solver interface

### 13.1 Re-evaluation — two solver *kinds* under one spec hierarchy

The engine-resident solvers (`mgr`, `superlu`, HYPRE, the GPU solvers) are C++
objects: `LinearSolverSpec.build()` returns a `linsolv_iface` that is injected
into the C++ engine and driven from `engine.solve_linear_equation()`. PETSc (via
`petsc4py`) and Pardiso (via `pypardiso` / MKL) run **in the Python process** and
cannot be injected into the C++ engine. The unified framework therefore admits
two solver *kinds*, both still expressed as `LinearSolverSpec` subclasses:

- **engine-resident** — `build()` → C++ `solvers.LinearSolver`, injected into the engine;
- **Python-resident** — `build()` → a `PythonLinearSolver` owned by the model and
  invoked from `DartsModel._solve_linear_equation()`.

This split is the deliberate, performance-justified deviation: a Python-resident
solver keeps the matrix on the Python side and is free to wrap it in the format
its backend consumes natively, with no round-trip through the C++ solver ABI.

### 13.2 Current cost (what the redesign removed)

Before the unification, `petsc_solve_linear_equation` /
`pardiso_solve_linear_equation` ran on every Newton iteration and:

1. `get_linear_system()` ran `scipy.bsr_matrix(...).tocsr()` — a full
   **block→scalar CSR expansion copy** (TODO-flagged in the source).
2. PETSc rebuilt the system matrix with `Mat().createAIJ(csr=...)` — copying
   the scalar CSR into PETSc storage.
3. The KSP **and** the PC (AMG / CPR / fieldsplit) were rebuilt from scratch
   every iteration — including the AMG setup, the dominant cost — even though
   the sparsity pattern is fixed for the whole run
   (`MATRIX_TYPE_CSR_FIXED_STRUCTURE`).
4. Pardiso re-ran the symbolic analysis on every `spsolve`.

The legacy `petsc_solve_linear_equation` / `pardiso_solve_linear_equation`
methods on `DartsModel`, the `linear_solver_types` enum that selected them
(`darts.input.input_data`), and the dispatch branch in
`_solve_linear_equation` are **all deleted** as part of this MR. The only path
is now `data_ts.linear_solver = <PETScSolverSpec | PardisoSolverSpec>`.

### 13.3 Design — `PythonLinearSolver`

```
LinearSolverSpec
├── engine-resident: MGRSolverSpec, SuperLUSolverSpec, …   build() → solvers.LinearSolver
└── PythonLinearSolverSpec (abstract)                       build() → PythonLinearSolver
    ├── PETScSolverSpec   (variant: "cpr" | "fs")
    └── PardisoSolverSpec
```

`PythonLinearSolver` is **stateful** — it owns the persistent matrix wrapper and
backend objects across Newton iterations:

| method | when | work |
|---|---|---|
| `setup(jac)` | once per run (pattern is fixed) | build the persistent matrix wrapper; build KSP + PC (PETSc) or the symbolic factorization (Pardiso) |
| `solve(jac, rhs, sol)` | every Newton iteration | refresh matrix **values only**; solve |

`jac` is the engine's zero-copy block-CSR view — `engine.expose_jacobian()`
publishes `jac_rows` / `jac_cols` / `jac_vals` as numpy views straight onto the
`block_csr_matrix` host arrays; `rhs` / `dX` are likewise `copy=False` views.

### 13.4 PETSc — AIJ with a precomputed block→scalar values gather

The first cut of this design used PETSc **BAIJ** (block-CSR with `bsize = N_VARS`)
to avoid the block→scalar expansion. **PETSc rejects this for CPR / fixed-stress**:
PCFIELDSPLIT splits variables *inside* each cell-block (pressure vs transport
or pressure vs displacement), which BAIJ cannot represent — PETSc returns
``"Cannot use MATBAIJ with PCFIELDSPLIT and currently set matrix and PC
blocksizes"``. BAIJ remains the right format only for solvers that treat each
cell-block as a unit (e.g. block ILU, block-AMG); for CPR/FS the matrix must be
scalar AIJ.

The realised design keeps the *structural* expansion out of the hot path:

- `setup()` precomputes the scalar CSR structure (`scalar_row_ptr`,
  `scalar_col_ind`) and a block→scalar **value gather index** from the fixed
  block sparsity (the §12 `csr_expansion` `value_map`). Initialises
  ``petsc4py`` once and caches the CPR / fixed-stress option-database string.
- `solve()` does ``np.take(jac_vals.ravel(), gather_idx, out=scalar_vals)`` —
  a values-only gather, no allocation — wraps the scalar arrays in a PETSc
  ``AIJ`` ``Mat`` (zero-copy through ``createAIJ(csr=...)``), attaches the
  fieldsplit index sets and solves.

Removed: the per-iteration ``scipy.sparse.bsr_matrix(...).tocsr()`` round-trip
(both the structural recomputation and the allocation) and the per-call
``petsc4py.init`` cost. The shared :class:`_BlockToScalarExpander` powers both
PETSc and Pardiso; see §13.5.

### 13.5 Pardiso — scalar CSR built once, values-only refresh

Pardiso (MKL) is a scalar direct solver and genuinely needs scalar CSR — the
block→scalar expansion is unavoidable, but it is lifted out of the hot path:

- `setup()` expands the block sparsity to scalar CSR **once** (scalar `row_ptr` /
  `col_ind`) and precomputes a **gather index** mapping block value slots to
  scalar value slots (the §12 `csr_expansion` `value_map`); it creates the
  persistent `pypardiso.PyPardisoSolver`.
- `solve()` does `scalar_vals[:] = jac_vals_flat[gather_idx]` — a values-only
  gather, no allocation, no structural work — and hands the scalar matrix
  (scipy `csr_matrix`, `copy=False`) to the persistent solver, which reuses its
  symbolic analysis and re-runs only the numerical factorization.

### 13.6 Lifecycle integration

- `DartsModel._apply_linear_solver_spec()` — a `PythonLinearSolverSpec` is
  `build()`-ed into a `PythonLinearSolver` stored on the model (no C++
  injection); engine-resident specs are injected into the engine as before.
- `DartsModel._solve_linear_equation()` — dispatches to the model's
  `PythonLinearSolver` when one is set (lazy `setup()` on first call), else to
  `engine.solve_linear_equation()`. No back-compat branch.

---

## Appendix A — integration-surface checklist

| Touchpoint | File:line | Action |
|---|---|---|
| Solver factory switch | `engines/src/engine_base.h:709-920`; GPU `engine_base_gpu.h:166+` | replace with registry (C3) |
| `linear_solver_t` enum | `engines/src/globals.h:68-88`; defaults `:116-120` | remove (C3) |
| GPU config gate | `CMakeLists.txt:96-99` | drop `FATAL_ERROR` for GPU (C6) |
| Newton-loop dispatch | `darts/models/darts_model.py:832-848`; `darts/pipes/viz/plot_live.py:902-914` | collapse to one call (C8) |
| Python solver enum | `darts/input/input_data.py:7-10` | fold into `LinearSolverSpec` (C8) |
| pybind solvers module | `solvers/src/pybind11/py_main.cpp` | expose registry + all solvers (C2/C7) |
| GPU solver classes | `engine_base_gpu.h:10,13`, `engine_base.h:50-55` | rewire to in-tree `solvers/gpu/` (C5) |
| open-DARTS `csr_matrix` | `solvers/include/csr_matrix.hpp` | add GPU device layer (C4) |
| Proprietary `bos` path | `thirdparty/thirdparty_bos_solvers.cmake`, `BOS_SOLVERS_DIR` guards | **keep** (out of scope) |
| Orphan CMake | `solvers/linear_solvers/src/CMakeLists.txt`; `engines/lib/CMakeLists.txt` | delete (C11) |

## Appendix B — solver inventory after the MR

| Solver | Platform | Origin | Default? |
|---|---|---|---|
| `mgr` (HYPRE MGR) | CPU | open-DARTS, in-tree | **CPU default** |
| `superlu` (direct) | CPU | open-DARTS, in-tree | — |
| `hypre_amg`, `hypre_ilu` | CPU | open-DARTS, in-tree | — |
| `amgx`, `bicgstab_gpu`, `gpu_cpr`, `cusparse_ilu`, `cusolv` | GPU | **absorbed in-tree (open-source)** | `amgx`-CPR = **GPU default** |
| `bos_cpr_amg`, `bos_ilu0`, `bos_fs_cpr`, … | CPU | **proprietary, external** (`BOS_SOLVERS_DIR`) | — |
| `petsc_cpr`, `petsc_fs` | CPU | Python (petsc4py) | — |
| `pardiso` (direct) | CPU | Python (pypardiso) | — |
| ~~`samg`, `amg1r5`, `aips`, `adgprs_nf`~~ | — | **dropped** | — |

---

## Appendix C — Implementation status (final)

Status of the §10 roadmap and the §12 unified-matrix work on branch
`xiaoming/add-mgr`. "verified" means the build is green and the noted models /
unit tests were run.

### §10 roadmap

| Item | Status | Evidence / notes |
|---|---|---|
| **C1** Plan | done | this document |
| **C2** Unified `linear_solver` interface + registry + `linsolv_iface` adapter | done | `solvers/include/{linear_solver,solver_config,solver_registry,solver_factories,linsolv_iface_adapter}.hpp`; `mgr`/`superlu` factories + `register_builtin_solvers()` |
| **C3** Neutralise the enum-driven engine factory | done | `engine_base.h`: the open-source build (`OPENDARTS_LINEAR_SOLVERS`) no longer runs the bos-stub factory — it errors clearly and requires a solver injected via `set_linear_solver()`. The proprietary build keeps the `linear_solver_t` enum + factory untouched (resolved design, §"item 6"). `engine_base_gpu.h` open-source path builds the in-tree BiCGStab + cuSPARSE-ILU solver. |
| **C4** GPU device layer on open-DARTS `csr_matrix` | done | `csr_matrix.hpp/.cpp` WITH_GPU layer (cuSPARSE BSR); `gpu_bsr_spmv` adapter |
| **C5** 5 GPU solver wrappers in-tree | done | `linsolv_{bicgstab,cusparse_ilu,cusolv,amgx,bos_cpr_gpu}` build from open-source sources |
| **C6** AMGX submodule + `WITH_GPU` decoupled from `BOS_SOLVERS_DIR` | done | `thirdparty/AMGX` submodule; `WITH_AMGX` opt-in CMake option; GPU builds in-tree with no `BOS_SOLVERS_DIR`. AMGX-absent builds gate AMGX behind `OPENDARTS_GPU_HAS_AMGX` and fall back to the BiCGStab + cuSPARSE-ILU solver. |
| **C7** `darts/solvers/` Python package | done | `specs.py` (`LinearSolverSpec`, `MGRSolverSpec`, `SuperLUSolverSpec`, `MGRLevelSpec`, `default_linear_solver`), `enums.py`, `adaptive.py`; compiled `solvers` pybind module installed in `darts/solvers/` |
| **C8** Unified dispatch | done | `DartsModel._apply_linear_solver_spec()` builds the solver from `data_ts.linear_solver` and injects it; `default_linear_solver()` returns `GMRESSolverSpec(prec=CPRSolverSpec())` for CPU; the Newton-loop solve branch is collapsed into `DartsModel._solve_linear_equation()`. PETSc / Pardiso are unified as `LinearSolverSpec` subclasses (`PETScSolverSpec`, `PardisoSolverSpec`, §13); both share a `_BlockToScalarExpander` that builds the scalar CSR structure + a block→scalar values gather index once, so each Newton iteration only does an `np.take` instead of `scipy.sparse.bsr_matrix(...).tocsr()`. The legacy `linear_solver_types` enum (`darts.input.input_data`) and the matching `petsc_solve_linear_equation` / `pardiso_solve_linear_equation` methods on `DartsModel` are **deleted**. **Verified end-to-end** on `2ph_comp` (block size 3): `PardisoSolverSpec` and `PETScSolverSpec(variant="cpr")` both run 19 timesteps, 35 Newton iterations, identical to the engine MGR path; `PETScSolverSpec(variant="fs")` exercises the FS code path cleanly (does not converge on this flow problem, as expected — physics mismatch). |
| **C9** Adaptive / mid-run switching | done | `darts/solvers/adaptive.py` — `AdaptiveSolverSpec`, `SolverSwitchContext`, `fallback_on_failure` policy; `DartsModel._maybe_switch_linear_solver()` re-injects after a timestep |
| **C10** CI | done, partial | GPU-from-source job (`build-linux-gpu` / `test-linux-gpu`); cross-path solver coverage via `test-linux` (proprietary `-a`) vs `test-linux-ODLS` (open-source registry solvers) over the model suite. **Deferred:** a dedicated per-solver micro-benchmark job (the dual-path suite run already exercises both solver stacks). |
| **C11** Cleanup | done | OD-6 assert (`assert(A->n_row_size == N_BLOCK_SIZE)`) guards the `linsolv_iface_bos` down-casts; orphan `CMakeLists.txt` removed (`solvers/linear_solvers/`, `engines/lib/`); the `darts.solvers` classes carry docstrings and are autodocumented in `docs/api.rst`. |

### §12 unified matrix layout

| Phase | Status | Evidence |
|---|---|---|
| Steps 1–4 — `sparsity_pattern`, `dual_array`, `block_csr_matrix`, `block_csr_view<N>` | done | `solvers/include/{sparsity_pattern,dual_array,block_csr_matrix,block_csr_view}.hpp` + unit tests |
| Phase A — SpMV, legacy-compatible accessors, GPU BSR SpMV adapter, matrix IO | done | committed; unit-tested |
| Phase B — `block_csr_matrix` implements `csr_matrix_base`; CPU + GPU engine Jacobian migrated | done | CPU build green, 2ph_comp verified; GPU build green, SPE11b runs on GPU with BiCGStab + cuSPARSE-ILU |
| Phase C — adjoint `to_nb_1` made polymorphic over `csr_matrix_base`; all 1266 GPU-build warnings cleared | done | `Adjoint_super_engine` / `Adjoint_mpfa` pass; 8 linear-solver unit tests pass; GPU build warning-free |

### Open-source CPU solver line-up (post-MR)

| Solver | Status | Notes |
|---|---|---|
| `linsolv_gmres` + `linsolv_cpr` (FGMRES + CPR) | done | **CPU default** -- `default_linear_solver("cpu")` returns `GMRESSolverSpec(prec=CPRSolverSpec())`. Open-source equivalent of the legacy `bos_gmres + bos_cpr_amg` stack. |
| `linsolv_mgr` (HYPRE-MGR) | done | Recommended fallback; configurable strategy via `MGRSolverSpec`. The proprietary `model.py` API surface (`set_bcsr_cpr_*`, `set_use_bcsr_cpr`, `set_mgr_composite_mode`, `set_mgr_local_solver`, `set_mgr_bilu0_*`, `set_mgr_local_correction_options`) and the corresponding implementations -- BCSR-CPR True-IMPES coarsening with adaptive AMG rebuild, composite preconditioner mode, BILU0 singular-pivot fallback, local-correction smoother -- all landed independently on `main` via `8b3be592` ("added adaptive AMG rebuild; re-use ParCSR pointer from BCSR Jacobian"). The shared SPE10 model from `darts-models/shared/spe10` runs unmodified against this MGR. |
| `linsolv_gmres` (GMRES/FGMRES) | done | restarted right-preconditioned MGS+Givens, mirrors bos `gmres_solver2`; `solve_transposed` wired through to the preconditioner |
| `linsolv_cpr` (CPR + CPRA) | done | two-stage CPR (Wallis 1983): HYPRE BoomerAMG on the (0,0)-extracted pressure subsystem `A_p` + HYPRE_ILU(0) on the scalar-expanded full system `A_s`. **HYPRE is driven directly** (no `linsolv_hypre_amg` / `linsolv_hypre_ilu` wrappers — those are tuned for elasticity); BoomerAMG config mirrors `mgr::CompositionalFlowStrategy::setupPressureAMG` (aggressive PMIS + multipass interp + C-F relax). Adjoint **CPRA** (Han et al. 2013) is implemented with separate AMG hierarchies on `A_p^T` and ILU on `A_s^T` — `HYPRE_BoomerAMGSolveT` was tried first but has limited relax-type coverage and was unreliable under our coarsening config. Setup uses a `first_setup_` flag: HYPRE handles are created once and reused across Newton iterations (subsequent setups refresh IJ matrix values and re-run `BoomerAMGSetup` / `ILUSetup` on the same handles — a destroy/recreate cycle on every iteration crashed `BoomerAMGSetup` on the second call). **Validated end-to-end** with `GMRESSolverSpec(prec=CPRSolverSpec())` on `2ph_comp`: 19 timesteps to T=10 days, 1 linear iter / Newton, no NaN / segfault. Smoke-test on the `2ph_comp` matrix dumps: forward reaches ~1e-15 relative residual; CPRA transpose reaches ~3-5e-11 in a single apply. |
| `linsolv_hypre_amg`, `linsolv_hypre_ilu` | retained | now scoped to **elasticity / mechanical engines only**; flow no longer uses them |
| `linsolv_superlu` | done | latent bug: stores typed `csr_matrix<N>*` from the iface_bos down-cast and segfaults on `block_csr_matrix` — fix is queued |

### SPE10 benchmark — BOS vs the open-source MGR (historical baseline)

Heavy-hitter benchmark on `darts-models/shared/spe10` (60×220×85 = **1.122M
reservoir cells**, 2-phase / 2-component flow, 5 wells; `OMP_NUM_THREADS=16`,
identical `model.py`, only `linear_type` differs).

**BOS** (`CPU_GMRES_CPR_AMG` = `linsolv_bos_gmres + linsolv_bos_cpr + linsolv_bos_amg`)
runs from the proprietary build with the proprietary BCSR-CPR coarsening, BILU0
fallback, and the bos_solver_lib smoothers active. **MGR** (`linsolv_mgr`) was
benchmarked at commit `661628230e` — *before* the BCSR-CPR True-IMPES coarsening,
adaptive AMG rebuild, BILU0 fallback, and composite preconditioner mode landed
on `main` via `8b3be592`. These numbers are therefore the **pre-BCSR-CPR
baseline** for the open-source MGR; the post-`8b3be592` MGR should be re-measured
against this table.

| Solver | T_final / target | Wall | Engine elapsed | ts | newton | linear |
|---|---:|---:|---:|---:|---:|---:|
| BOS                        | 50 / 50 d     | **0:45** | 39.7 s | 13 | 34 | 153 |
| BOS                        | 1000 / 1000 d | **1:20** | 74.4 s | 22 | 73 | 358 |
| MGR (pre-BCSR-CPR, T=50)   | terminated at T=13.6 / 50 d | 26:39 | 1144 s | 10 | 41 | 1746+ |

**Per-T comparison (cumulative engine ELAPSED, seconds, pre-BCSR-CPR MGR)**:

| T (days) | BOS | MGR | MGR/BOS | BOS LI/ts | MGR LI/ts |
|---:|---:|---:|---:|---:|---:|
| 0.10  | 6  | 62   | 10.3× | 18 | 85  |
| 0.97  | 13 | 250  | 19.2× | 9  | 139 |
| 4.73  | 20 | 545  | 27.2× | 11 | 217 |
| 13.6  | 26 | 1144 | **44.0×** | 18 | 260 |

**Consistency**: identical block-CSR Jacobian, same dt sequence, matching CFL and
Newton residuals at every step — the linear-solver dispatch is the only variable.

**Pre-BCSR-CPR bottleneck**: at the time of this benchmark, MGR was using HYPRE-MGR
with `CompositionalFlowStrategy` and Schur-complement coarse solve. SPE10's high
permeability contrast (~7 orders of magnitude) demanded 50–360 linear iters per
Newton, vs. BOS's 5–18 — the dominant cost was the GMRES + MGR-apply per-iter loop,
not the AMG hierarchy build. The BCSR-CPR True-IMPES coarsening that landed in
`8b3be592` is precisely the lever that drops per-Newton iter count to the BOS range;
re-measuring is queued as a follow-up.

Full benchmark detail at `/tmp/spe10_bench/COMPARISON.md`.

### Known follow-ups (out of this MR)

- **Re-benchmark SPE10 against `8b3be592` MGR** — BCSR-CPR True-IMPES coarsening,
  adaptive AMG rebuild, BILU0 fallback, and composite preconditioner mode all landed
  on `main` after the pre-BCSR-CPR baseline was captured; the 44× gap shown above is
  expected to close substantially.
- **GPU engine Jacobian** still constructs `csr_matrix<N>` (its proven cuSPARSE BSR device
  layer). Migrating it to `block_csr_matrix` needs `gpu_bsr_spmv` to support in-place device
  assembly.
- **CPU default flipped to FGMRES + CPR** (was MGR). Benchmark across four flow models with
  `MGRSolverSpec` vs `GMRESSolverSpec(prec=CPRSolverSpec())`:

  | model | block | ts | newton | linear (MGR) | linear (FGMRES+CPR) | wall MGR / CPR (s) |
  |---|---:|---:|---:|---:|---:|---:|
  | `2ph_comp`       | 3 | 19  | 35  | 35    | 35    | 1.49 / 0.31 |
  | `2ph_do`         | 2 | 15  | 23  | 343   | 343   | 0.10 / 0.10 |
  | `2ph_geothermal` | 2 | 168 | 168 | 3242  | 3242  | 7.37 / 7.35 |
  | `3ph_bo`         | 3 | 21  | 39  | 147   | 147   | 0.64 / 0.64 |

  Identical iteration counts (both wrap the same FGMRES around HYPRE BoomerAMG on the
  pressure subsystem); FGMRES+CPR has lower per-iter setup overhead. The user's original
  goal -- "FGMRES as the main solver" -- is met.
- **GMRES + MGR composition**: still triggers HYPRE NaN warnings on `2ph_comp` (workspace +
  break fixes landed but did not fully clear them). The default flip avoids this path; the
  NaN debug is no longer blocking but is queued for follow-up.
- **Adjoint solve integration**: `linsolv_gmres::solve_transposed` is wired to
  `prec_->solve_transposed`; `linsolv_cpr::solve_transposed` runs CPRA. The remaining piece
  is exercising the path through `Adjoint_super_engine` once an adjoint test model lands.
- **`linsolv_bos_fs_cpr` port**: open-source per the proprietary header but not yet
  re-implemented in `solvers/`. Depends on the new `linsolv_cpr` being stable in production.
- **GPU FGMRES**: GPU side still uses the legacy AMGX-CPR path. Porting `linsolv_gmres` to
  cuBLAS is straightforward but out of this MR.
- **Mechanics engines** (`engine_pm`, `engine_elasticity`, `engine_super_elastic`) keep a
  `csr_matrix<N>` Jacobian — a trial migration to `block_csr_matrix` regressed `engine_pm`
  (segfault) and was reverted; their Jacobian structure needs its own investigation.
- Retiring `csr_matrix<N>` is gated on both of the above. `csr_matrix_base` is **kept** — it is
  the unified polymorphic matrix interface.
- `PETScSolverSpec(variant="fs")` is validated only as a code path (it runs cleanly on
  `2ph_comp` but the fixed-stress preconditioner does not match flow physics); a true FS
  validation needs a poromechanics model (block size 4) -- straightforward once a model
  sets `data_ts.linear_solver = PETScSolverSpec(variant="fs")`.

---

## MR closeout

Branch `xiaoming/add-mgr`. What ships with this MR:

1. **C2–C11 of §10**: unified `linear_solver` interface + registry, neutralised the
   enum-driven engine factory in open-source builds, in-tree GPU device layer +
   five GPU solver wrappers, AMGX submodule decoupled from `BOS_SOLVERS_DIR`,
   `darts.solvers` Python package, unified Newton-loop dispatch incl. PETSc /
   Pardiso (§13), adaptive solver switching, CI dual-path coverage, post-cleanup.
2. **§12 unified matrix layout**: steps 1–4 + phases A, B, C complete on CPU and
   GPU. Adjoint engines pass; 1266 GPU warnings cleared; SPE11b runs on GPU.
3. **§13 PETSc / Pardiso unification**: `PythonLinearSolver` base + AIJ values-gather
   for PETSc CPR/FS + scalar CSR for Pardiso, validated end-to-end on `2ph_comp`.
4. **In-tree FGMRES** (`linsolv_gmres`): restart-GMRES + MGS + Givens + adjoint,
   the open-source replacement for `linsolv_bos_gmres`.
5. **In-tree two-stage CPR** (`linsolv_cpr` + CPRA): the open-source replacement for
   `linsolv_bos_cpr + linsolv_bos_amg`, driving HYPRE directly with the flow-tuned
   config from `mgr::CompositionalFlowStrategy::setupPressureAMG`. Forward + Han 2013
   adjoint transpose.
6. **CPU default flip → FGMRES + CPR**, fulfilling the original "FGMRES as the main
   solver" goal. Iteration counts match MGR across four flow models; per-iter
   overhead is lower.
7. *(MGR setup reuse + adaptive rebuild + proprietary-API stubs were prepared on
   this branch but dropped during the merge with `8b3be592`, which already
   delivered the same surface with real BCSR-CPR / BILU0-fallback / composite-mode
   implementations. The shared SPE10 `model.py` runs unmodified against the
   merged-in MGR.)*
8. **Solver config / spec surface cleanup** -- dead parameters removed from both
   the C++ structs and the Python Spec classes:
   * `solver_config::print_level` (C++) -- written by every spec but never read
     by any C++ factory or solver. Removed. Per-solver verbosity is exposed by
     the solver-specific config when it exists (e.g.
     `mgr_solver_config::log_level`); `LinearSolverSpec.print_level` (Python)
     stays because the Python-resident solvers (PETSc / Pardiso) consume it.
   * `cpr_solver_config::amg_tolerance` + `linsolv_cpr::set_amg_tolerance` +
     the matching `CPRSolverSpec.amg_tolerance` -- BoomerAMG inside CPR is
     always run with `tol=0` since it is a preconditioner stage; the sweep
     budget (`amg_max_iters`) is the only AMG knob.
   * `LinearSolverSpec.tolerance` / `max_iterations` docstring -- now spells
     out that for engine-resident solvers the engine's `init()` call overrides
     these from `data_ts.linear_tol` / `data_ts.linear_max_iter` (the
     authoritative Newton-loop knobs); for Python-resident solvers the spec
     values are used directly.

What does **not** ship and why:

* **`linsolv_bos_fs_cpr`** (4-block poromechanics CPR) — open-source per the header
  but not yet re-implemented in `solvers/`. Depends on the new `linsolv_cpr` being
  stable in production, plus a poromech test target.
* **GPU FGMRES** — GPU side keeps the legacy AMGX-CPR path; cuBLAS port of
  `linsolv_gmres` is straightforward but out of scope.
* **GMRES + MGR NaN debug** — composition triggers HYPRE NaN warnings on
  `2ph_comp`. The default flip to FGMRES + CPR avoids the path entirely, so this
  is no longer blocking.
* **Mechanics-engine Jacobian migration** to `block_csr_matrix` — segfault on a
  trial migration of `engine_pm` was reverted; the elasticity engines keep
  `csr_matrix<N>`. Their Jacobian structure needs its own investigation.

The MR delivers a fully self-contained open-source CPU solver line-up (FGMRES + CPR
default, MGR fallback, PETSc / Pardiso for Python-resident solves), the unified
matrix layout that unblocks future GPU + mechanics work, and the API surface that
proprietary models need to run unmodified. The remaining BOS-parity gap on heavy
flow problems is captured by name with the precise next-step (BCSR-CPR coarsening +
BILU0 fallback in MGR).
