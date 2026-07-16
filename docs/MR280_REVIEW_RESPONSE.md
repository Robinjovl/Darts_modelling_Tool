# MR280 — review response

Brief answers to the nine review points. Items marked **[done]** were changed in code; the rest are
analysis/answers with a recommendation.

---

## 1. Breaking changes in `CHANGELOG.md` **[done]**

Added a dedicated **"Breaking changes (!280)"** block to `CHANGELOG.md`. The ones that can silently
change user results:

| Change | Impact |
|---|---|
| **Default linear solver changed** — CPU: in-tree **FGMRES+CPR**; GPU: **AMGX-CPR** | Different LI/NI and timings; at loose tolerances the Newton path can differ. **Re-baseline references.** |
| `set_sim_params(tol_linear=, it_linear=)` **removed** | `TypeError` — see §2 |
| **`data_ts.linear_tol` / `linear_max_iter` / `linear_type` / `linear_print_level` removed** | Assignments become **silently inert** — the loudest trap here. Move them to `self.linear_solver` (see §3) |
| `set_solver_params()` removed, `data_ts.linear_solver` retired | Solver config now lives in `set_solver()` + `self.linear_solver` |
| `linear_solver_t` enum **renumbered** (`CPU_GMRES_MGR` moved before the GPU block) | Only breaks code hardcoding *integer* enum values; symbolic names are safe |
| GPU builds **always** build AMGX (`--amgx` removed, `WITH_AMGX=ON`) | GPU build requires the `thirdparty/AMGX` submodule (auto-initialised) |
| PETSc/Pardiso now report failure on a non-finite solve | Previously-silent NaN solves now cut the timestep |

---

## 2. `set_sim_params` — nonlinear only **[done]**

`tol_linear` / `it_linear` are **removed** from `set_sim_params()`; it now covers **time-stepping +
Newton only**. Linear settings live on the solver:

```python
def set_solver(self):
    self.set_sim_params(first_ts=..., tol_newton=1e-3, it_newton=20)  # time-stepping + Newton
    super().set_solver()                                              # platform default spec
    self.linear_solver.tolerance = 1e-6                                      # linear knobs
    self.linear_solver.max_iterations = 50
```

`super().set_solver()` materialises the platform-default spec (FGMRES+CPR on CPU, AMGX-CPR on GPU)
and is idempotent, so a model that names its own solver (`self.linear_solver = MGRSolverSpec(tolerance=…)`)
simply keeps it. All **37 shipped models were migrated** this way.

---

## 3. Unify `data_ts` / `sim_params` / `solver` **[done]**

There were **three overlapping surfaces** for one concern — that is the root of the confusion:

| Surface | Was | **Now** |
|---|---|---|
| `data_ts` (`DataTS`) | time-stepping, Newton, **and** `linear_tol` / `linear_max_iter` / `linear_type` / `linear_print_level` | **time-stepping + Newton only** — all four `linear_*` fields **removed** |
| `self.linear_solver` (`LinearSolverSpec`) | solver + preconditioner choice, but its `tolerance` / `max_iterations` were **ignored** for engine-resident solvers (the engine re-applied `data_ts.linear_*` at `init()`) | **the single owner of every linear setting**: solver, preconditioner, `tolerance`, `max_iterations`, `print_level` |
| `params` (`sim_params`, C++) | user-facing-ish mirror fed by `copy_data_ts_to_sim_params()` | **C++ mirror, not a user API** — time-stepping/Newton from `data_ts`, linear from `self.linear_solver` |

**One owner per concern.** `DartsModel._apply_solver()` now mirrors the spec into `sim_params`
(`_sync_solver_to_sim_params()`: `tolerance` → `tolerance_linear`, `max_iterations` →
`max_i_linear`, `print_level` → `linear_print_level`) after `set_solver()` and before
`engine.init()`, so the spec is authoritative for **every** backend — engine-resident (MGR / GMRES /
CPR / SuperLU) and Python-resident (PETSc / Pardiso) alike. Previously the spec's `tolerance` was
silently overridden by `data_ts.linear_tol`, which is exactly the duplication this removes.

Defaults are unchanged (`DataTS.linear_tol` 1e-5 / `linear_max_iter` 50 == `LinearSolverSpec`
`tolerance` 1e-5 / `max_iterations` 50), so models that set nothing behave identically. Mechanics /
THMC models (no `data_ts`; they drive `engine.ls_params` + `params` directly) and proprietary builds
are explicitly left to the engine factory — `set_solver()` records whether the spec was *chosen* or
merely defaulted (`_solver_is_default()`, identity-compared, so `super().set_solver()` followed by
`self.linear_solver = <Spec>` still counts as chosen), and `_apply_solver()` never acts on a spec the model
did not ask for.

**Migration:** assignments to `data_ts.linear_*` are now **silently inert** — grep for them. Models
driven by case files can keep the knob in their own input data and apply it in `set_solver()`, which
is what `cpg_sloping_fault` now does (`idata.sim.linear_tol` → `self.linear_solver.tolerance`).

### Side-effect worth flagging: some spec tolerances were previously *decorative*

The old precedence meant a spec's `tolerance` / `max_iterations` were **overwritten at
`engine.init()`** by `params.tolerance_linear` / `max_i_linear` — which, for a model that never
called `set_sim_params()`, were just the `sim_params` C++ defaults (**1e-5 / 50**, `globals.h:117`).
Concretely, `engine_pm_cpu.cpp:298` does:

```cpp
linear_solver_external->init(Jacobian, params->max_i_linear, params->tolerance_linear);
```

So the five mechanics models that declare `GMRESSolverSpec(prec=FSCPRSolverSpec(...), tolerance=1e-8,
max_iterations=200)` — `1ph_1comp_poroelastic_analytics`, `1ph_1comp_poroelastic_convergence`,
`SPE10_mech`, `displaced_fault_reactivation`, `THM_vs_geomech_proxy` — were in fact **solving at
1e-5 / 50**. Their declared 1e-8 never took effect.

Once the spec became authoritative it did — and 1e-8 is **not reachable** by FS-CPR on these systems,
so the extra iterations are wasted. Measured on `SPE10_mech` (same 900 s wall-clock budget, neither
run to completion — so read the *per-Newton* columns, not the totals):

| spec | avg linear iters / Newton | solves hitting the iteration cap |
|---|---|---|
| `tolerance=1e-8, max_iterations=200` (honoured) | **99** | **22 of 48** (stall at `lin 201`) |
| `tolerance=1e-5, max_iterations=50` (what it really ran) | **41** | 0 |

**Resolution:** the five specs were pinned to `1e-5 / 50` — the configuration they have always
actually run — so behaviour is **unchanged** and no re-baselining is needed. (The pin rests on
behaviour preservation, not on a performance claim: the two configurations were not run to the same
simulated end time, so their cut / Newton-failure counts are not directly comparable.) This is the
strongest single piece of evidence for why the duplication had to go: two numbers, and nobody could
tell which one was live.

---

## 4. Do we need the empty/abstract `linsolv_bos_*`?

**No — they are dead stubs, not an extension point.** `linsolv_bos_{gmres,amg,bilu0,cpr,fs_cpr}`
print `"NOT IMPLEMENTED: …"` and return; they do nothing:

```
linear_solvers/src/linsolv_bos_gmres.cpp:45   std::cout << "NOT IMPLEMENTED: linsolv_bos_gmres::setup(...)"
linear_solvers/src/linsolv_bos_amg.cpp:68     std::cout << "NOT IMPLEMENTED: linsolv_bos_amg::setup"
```

They exist only so the **enum-driven engine factory** (`case sim_params::CPU_GMRES_CPR_AMG: new
linsolv_bos_gmres…`) still *compiles and links* in the open-source build. The real BOS solvers come
from the **external** `bos_solvers` library (`ENABLE_BOS_SOLVERS` / `GSELINSOLVERSPATH`), which
brings its own classes — the in-tree stubs are never used to support third-party BOS.

**Recommendation:** delete them and `#ifdef`-guard the BOS branches of the factory out of the
open-source build (the open-source path already bypasses the enum entirely via the registry +
external-solver injection). **Exception:** `linsolv_bos_cpr_gpu.cu` is **not** a stub (1302 lines,
it is the real GPU CPR backing AMGX-CPR) — keep it, but it should be **renamed** (it is no longer
"bos").

---

## 5. `csr_matrix_base` / `csr_matrix` / `block_csr_matrix` — interfaces

```
csr_matrix_base                 (abstract) — the polymorphic contract the SOLVERS see
   ├── csr_matrix<N_BLOCK_SIZE>  (legacy)  — compile-time block size, owns its own device layer
   └── block_csr_matrix          (unified) — runtime block size, dual_array storage
```

| Type | Role | Why it lives there |
|---|---|---|
| **`csr_matrix_base`** | The **only** type solvers depend on: `n_rows`, `get_rows_ptr/cols_ind/values/diag_ind`, SpMV (`matrix_vector_product*`), and the device mirrors (`get_*_d`, `matrix_vector_product_d*`, `..._t_d*`). | Solvers must work with *either* backend, so the contract is one abstract base — not templated on block size. Device entry points are here (not in the derived class) so a GPU solver can drive **any** matrix through a `csr_matrix_base*`. |
| **`csr_matrix<N>`** | Legacy, block size fixed at compile time; ships its own raw `*_d` device pointers. | Kept for the proprietary/BOS path, which is templated on `N`. New code should not use it. |
| **`block_csr_matrix`** | The unified engine Jacobian: **runtime** block size, structure shared via `sparsity_pattern`, values in a `dual_array` (host+device mirror). | Runtime block size removes the `N`-template explosion; the shared structure lets scalar views be cached (see §6). |

Design rule: **capability on the base, storage in the derived class.** Anything a solver may call
polymorphically (including transposed device SpMV) is a `csr_matrix_base` virtual, defaulted to
"unsupported" so only the backends that can serve it opt in.

---

## 6. `sparsity_pattern` vs `csr_expansion`; contiguity; conversion

- **`sparsity_pattern`** (`linear_solvers/include/sparsity_pattern.hpp`) — the **block** (BCSR) structure:
  `row_ptr`, `col_ind`, `diag_ind` over *blocks*. Shared (`shared_ptr`) by all matrices with the same
  sparsity, so the structure is stored **once**.
- **`csr_expansion`** (`linear_solvers/include/csr_expansion.hpp`) — the **scalar (point) CSR expansion** of
  that block pattern: `csr_expansion(const sparsity_pattern&, int block_size)`. It is a pure function
  of (block pattern, block size), so it is computed **once** and **cached on the `sparsity_pattern`**.

**Storage is contiguous in both forms:**

- **BCSR values**: one contiguous array, `nnz_blocks × b × b`, block-major, each block row-major.
- **Scalar CSR values**: one contiguous array, `nnz_blocks × b × b` (same count) in point ordering.

**Conversion:** structure ↔ structure is done once (`csr_expansion`); per Newton iteration only the
**values** are gathered block→scalar through a precomputed index map (a permutation gather — no
allocation, no structure rebuild). The reverse (scalar→block) is never needed: nothing writes back a
scalar matrix.

---

## 7. What is a "view"? What is `dual_array<T>`?

- **`dual_array<T>`** — a **host+device mirror of one contiguous buffer** with dirty tracking. It owns
  `std::vector<T>` on host and a `cudaMalloc`'d buffer on device, plus `host_modified_` /
  `device_modified_` / `device_populated_` flags and `sync_to_device()` / `sync_to_host()`. The
  const accessors are read-only; the **non-const** accessors record a write. It is what makes
  `block_csr_matrix` usable from CPU and GPU without the caller hand-managing `cudaMemcpy`.
- **"view"** — a **non-owning** window onto someone else's storage. Two in the codebase:
  - the **scalar-CSR view** of a block matrix (`build_scalar_csr_device()` / `scalar_csr_adapter`) —
    presents BCSR data as point CSR for backends that cannot ingest blocks (HYPRE, Pardiso, PETSc,
    AMGX-bs1). It does **not** copy the structure (that is cached on the `sparsity_pattern`).
  - `block_csr_view` — a lightweight handle over an existing pattern + values.

Rule of thumb: **`dual_array` owns, a view borrows.**

---

## 8. HYPRE conversion overhead — do we use HYPRE BCSR?

**No — HYPRE has no block-CSR ingest on this path.** We build an `HYPRE_IJMatrix` with
`HYPRE_IJMatrixSetObjectType(A_ij, HYPRE_PARCSR)`; ParCSR is **point (scalar) CSR**. The wrapper is
even explicitly specialised for scalar only:

```
linear_solvers/src/linsolv_hypre_amg.cpp:302  template<> void linsolv_hypre_amg<1>::csr_matrix_to_hypre_ij(...)
                              :307  // NOTE: This function works only for N_BLOCK_SIZE = 1
```

So **every block matrix must be expanded to scalar CSR before HYPRE** — that is exactly why
`csr_expansion` exists.

**The overhead has two parts:**

1. **Structure expansion — negligible.** Done once and cached on the `sparsity_pattern` (§6).
2. **Per-`setup()` cost — this is the real one.** `csr_matrix_to_hypre_ij()` calls
   `HYPRE_IJMatrixCreate` → `SetValues` → `Assemble` **on every setup (i.e. every Newton iteration)**,
   which rebuilds HYPRE's internal ParCSR each time, plus the block→scalar value gather. Row-index /
   row-degree buffers are already cached; the IJ matrix itself is not.

**Recommendation (perf, not correctness):** create the `HYPRE_IJMatrix` **once** and only refresh
values on subsequent setups (`HYPRE_IJMatrixInitialize` + `SetValues` on the existing object, keeping
the assembled ParCSR), i.e. the same "structure once, values per iteration" policy used everywhere
else. This is the concrete content of the deferred `refresh()` item — see §9.

---

## 9. Documentation / plans / comments

Current state and what is still open:

- **Reports** (kept): `docs/MR280_SOLVER_REVIEW.md`, `docs/MR280_SOLVER_REVIEW_UPDATE.md` — the
  benchmark + review record. `docs/GPU_ADJOINT_CPRA_PLAN.md` — GPU adjoint/CPRA design + status.
- **This file** answers the review points and is the migration reference for §1–§2.
- **Deferred / open** (explicitly *not* in this MR):
  1. **`linear_solver::refresh()` is still dead** — the engine always calls `setup()`. Not a
     correctness bug (setup() is right, and CPR already does a value-only refresh internally), but the
     HYPRE per-iteration IJ rebuild in §8 is the concrete win. Needs an opt-in policy + a full setup on
     the first Newton iteration of each step + reference re-validation — not a mechanical swap.
  2. **Delete the `linsolv_bos_*` stubs** and rename `linsolv_bos_cpr_gpu` (§4).
  3. **GPU solver registry** — the GPU factory is still `linear_type`-enum driven; porting it to the
     registry is what would let the enum be retired entirely.
  4. **Thermal adjoint is incomplete** (energy-row terms) — warned at runtime; see
     `GPU_ADJOINT_CPRA_PLAN.md`.
