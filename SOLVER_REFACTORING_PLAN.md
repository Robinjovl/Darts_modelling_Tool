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

## Appendix C — Implementation progress (2026-05-18)

Done and verified on `xiaoming/add-mgr` (uncommitted working tree):

| Commit | Status |
|---|---|
| Plan (this document) | done |
| C2 — unified `linear_solver` interface + registry + `linsolv_iface` adapter | done, builds |
| Solver registration — `mgr`/`superlu` factories, `register_builtin_solvers()` | done, builds |
| pybind exposure — registry + config classes in the (renamed) `_solvers` module | done, builds + runs |
| `darts/solvers/` package — `LinearSolverSpec` classes, HYPRE enums, `default_linear_solver()` | done, builds + runs |

New files: `solvers/include/{linear_solver,solver_config,solver_configs,solver_registry,solver_factories,linsolv_iface_adapter}.hpp`, `solvers/src/{solver_registry,solver_factories,linsolv_iface_adapter}.cpp`, `darts/solvers/{__init__,enums,specs}.py`. The pybind module was renamed `solvers` → `_solvers` and installed into `darts/solvers/` (with `libopendarts_solvers.so` alongside it).

### Remaining — item 6 (enum removal / engine rewiring): implementation notes

Removing `sim_params::linear_solver_t` is **one atomic refactor** — there is no
compiling intermediate state, so the following must change together and be
driven through the compile-build-fix loop:

- `engines/src/globals.h` — delete `linear_solver_t`, the `linear_type` field,
  the `linear_solver_params` class, and the ctor defaults.
- `engines/src/engine_base.h` — delete the factory `switch` (~lines 716-919);
  `init_base` uses only an injected solver (clear error if none); replace the
  `linear_type >= GPU_GMRES_CPR_AMG` GPU test with the already-computed
  `is_gpu_engine` (engine-name based, line ~688). Migrate `set_linear_solver`
  and the `linear_solver` member from `linsolv_iface` to the unified
  `linear_solver` interface.
- `engines/src/engine_base.cpp` — same GPU-discriminator replacement at the
  device-Jacobian copy guard; migrate the solve path from `get_n_iters()` to
  `stats()`.
- `engines/src/engine_base_gpu.h` — delete its factory + the `linear_type == 0`
  coercion.
- `engines/src/pybind11/py_globals.cpp` — delete the `linear_solver_t` /
  `linear_solver_params` bindings.
- Proprietary `bos` path: when `BOS_SOLVERS_DIR` is set, register the external
  bos solvers in the registry (via the `linsolv_iface` adapter), `#ifdef`-gated,
  so the proprietary build still has solvers once the factory is gone.
- `darts/models/darts_model.py`, `thmc_model.py`, `pipes/viz/plot_live.py` and
  every model — build the solver from `data_ts.linear_solver` (a
  `LinearSolverSpec`) and inject it; collapse the Newton-loop `isinstance`
  dispatch; PETSc/Pardiso become `LinearSolverSpec` subclasses.

This is the recommended starting point for continuing the MR.

### Item 6 — resolved design (decision, 2026-05-18)

A structural conflict was found: the solver registry (`solvers/` tree) is
compiled **only in the open-source build** — `add_subdirectory(solvers)` is
gated by `if(NOT DEFINED BOS_SOLVERS_DIR)`. In the proprietary build the tree
is skipped and `linear_solvers` is the imported external `.a`, so there is no
registry there. Deleting the engine factory outright would break the
proprietary build.

**Resolved: `#ifdef`-gate the two paths** (refines decision 1).

- Proprietary build (`!OPENDARTS_LINEAR_SOLVERS`): factory switch +
  `linear_solver_t` enum **kept, untouched**. Because `WITH_GPU` implies
  `BOS_SOLVERS_DIR` implies `!OPENDARTS_LINEAR_SOLVERS`, the GPU factory, the
  `linear_type >= GPU_GMRES_CPR_AMG` discriminator, and `engine_base_gpu.h`
  are all proprietary-only and need **no change**.
- Open-source build (`OPENDARTS_LINEAR_SOLVERS` defined): `engine_base::init_base`
  no longer runs the factory; the linear solver must be injected from Python
  (built via the registry from a `LinearSolverSpec`). `linear_solver_t` stays
  in `globals.h` (proprietary needs it) but is unused on the open-source path.

So item 6 narrows to: `engine_base.h` (gate the open-source factory section to
require an injected solver); `darts_model.py` (build the solver from
`data_ts.linear_solver` and inject it; `default_linear_solver()` → MGR);
collapse the Newton-loop `isinstance` dispatch; migrate models.

**Remaining design call:** the registry's `create_linear_solver` currently
returns the new `linear_solver` type, but `engine_base::set_linear_solver`
takes `linsolv_iface`. Simplest fix — have the registry/`create_linear_solver`
return the `linsolv_iface`-based solver for engine injection (the `linsolv_mgr`
/ `linsolv_superlu` objects already are `linsolv_iface`); the new
`linear_solver` interface + `linsolv_iface_adapter` then remain as the
Python-facing / future-solver abstraction. To be settled at the start of the
item-6 implementation.
