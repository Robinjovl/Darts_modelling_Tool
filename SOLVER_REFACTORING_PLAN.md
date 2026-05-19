# open-DARTS Linear Solver — Consolidation & Refactoring Plan

Branch: `xiaoming/add-mgr`. One MR, delivered as a **stacked series of commits**.
Status: planning complete, implementation starting. Last updated: 2026-05-18.

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
5. Makes **HYPRE/MGR the default CPU solver**; the GPU default stays the legacy AMGX-CPR path.
6. Is fully covered by CI/CD, including cross-solver comparison.

---

## 2. Background

The open-source CPU build currently has **no working iterative solver**: the `linsolv_bos_*`
classes in `solvers/` are stubs printing `"NOT IMPLEMENTED"`; the real implementations are
proprietary and external. The only working open-source CPU solvers are SuperLU (direct) and the
new HYPRE MGR solver — repaired in commit `9139da28` ("solved the problem of Jacobian
updating"). MGR is the first viable open-source iterative CPU solver and becomes the new CPU
default. The GPU solver stack also lives only in `darts-linear-solvers`; its open-source GPU
portion is brought in-tree by this MR so that **GPU builds no longer require the external
library**.

---

## 3. Scope of this MR

### In scope
- Absorb the **5 open-source GPU solvers**: `linsolv_amgx`, `linsolv_bicgstab`,
  `linsolv_bos_cpr_gpu`, `linsolv_cusparse_ilu`, `linsolv_cusolv`.
- Reimplement a **GPU device layer on open-DARTS' own `csr_matrix`** (the GPU solvers'
  dependency on the proprietary matrix is removed by reimplementation — see §7.3).
- Unified C++ `linear_solver` interface + registry + typed config structs.
- Python `LinearSolverSpec` class hierarchy; unified dispatch (incl. PETSc/Pardiso).
- CPU default → MGR; centralized default policy.
- Adaptive / mid-run solver switching.
- CI: GPU buildable from in-tree source; solver-comparison job.

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
- **R5** — Default policy in one place: CPU ⇒ MGR; GPU ⇒ legacy AMGX-CPR.
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
def default_linear_solver(platform: str) -> LinearSolverSpec:
    return MGRSolverSpec() if platform == "cpu" else AMGXSolverSpec()
```

Single source of truth. Remove the `CPU_SUPERLU` default at `globals.h:116-120`, the override
at `darts_model.py:187-188`, and the coercion at `engine_base_gpu.h:160-163`.

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
| **C8** Unified dispatch | done, partial | `DartsModel._apply_linear_solver_spec()` builds the solver from `data_ts.linear_solver` and injects it; `default_linear_solver()` returns `MGRSolverSpec` for CPU; the duplicated Newton-loop solve branch is collapsed into `DartsModel._solve_linear_equation()` (used by the model loop and the live-plotting loop). **Deferred:** wrapping PETSc / Pardiso as `LinearSolverSpec` subclasses — they remain selected through `data_ts.linear_type`; the conversion touches the working solver-setup path and is not verifiable without `petsc4py` / `pypardiso` and their models. |
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

### Known follow-ups (out of this MR)

- **GPU engine Jacobian** still constructs `csr_matrix<N>` (its proven cuSPARSE BSR device
  layer). Migrating it to `block_csr_matrix` needs `gpu_bsr_spmv` to support in-place device
  assembly.
- **Mechanics engines** (`engine_pm`, `engine_elasticity`, `engine_super_elastic`) keep a
  `csr_matrix<N>` Jacobian — a trial migration to `block_csr_matrix` regressed `engine_pm`
  (segfault) and was reverted; their Jacobian structure needs its own investigation.
- Retiring `csr_matrix<N>` is gated on both of the above. `csr_matrix_base` is **kept** — it is
  the unified polymorphic matrix interface.
- **C8 PETSc / Pardiso** `LinearSolverSpec` subclasses (see C8 row).
