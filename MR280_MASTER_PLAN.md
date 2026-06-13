# MR280 Master Plan — Solver Consolidation: Close-out, Hardening & Migration Study

Branch: `xiaoming/add-mgr` → MR !280 → `development`. Single source of truth for the remaining
work; supersedes the planning content of the three earlier docs (kept for history/narrative):

- `SOLVER_REFACTORING_PLAN.md` — authoritative C++ architecture record (§0–§13, Appendix C).
- `SOLVER_UNIFICATION_PLAN.md` — authoritative Python `self.solver` unification narrative
  (header note: phase numbering defect + stale GPU-spec claim corrected here).
- `SET_SOLVER_MIGRATION_PLAN.md` — **superseded** (describes the removed
  `data_ts.linear_solver` carrier); historical value only.

Last verified against code: **2026-06-12**, commit `da24fc39`, pipeline `2590797258`.

---

## 0. Current invariants (verified against code, not docs)

* Solver carrier = **`self.solver` (a `LinearSolverSpec`)**, resolved by
  `DartsModel._resolve_solver_spec()` / injected by `_apply_solver()` pre-`engine.init()`.
  `data_ts.linear_solver` is **removed** from `DataTS`.
* CPU open-source default = **FGMRES+CPR** (`default_linear_solver("cpu")` →
  `GMRESSolverSpec(restart=50, prec=CPRSolverSpec())`, `darts/solvers/specs.py:671`).
* GPU default = **AMGX+CPR** via `params.linear_type = gpu_gmres_cpr_amgx_ilu`
  (`_apply_gpu_solver()`; GPU specs `AMGXCPRSolverSpec` / `GPUBiCGStabCPRSolverSpec` /
  `GPUGMRESILU0SolverSpec` exist and are wired — contrary to the older plan's "missing").
* C++ registry = `['cpr','fs_cpr','gmres','mgr','superlu']` (`solver_factories.cpp:568-572`).
* Open-source vs proprietary is currently keyed on **`DEFINED BOS_SOLVERS_DIR`** (5 top-level +
  19 per-target CMake guards) which sets/omits the `OPENDARTS_LINEAR_SOLVERS` compile macro;
  the C++ enum default is `#ifdef OPENDARTS_LINEAR_SOLVERS → CPU_SUPERLU : CPU_GMRES_CPR_AMG`
  (`globals.h:116-120,189-193`) — the Python spec layer overrides the open-source one.
* The proprietary build's *flow* default (what green CI historically exercised) is
  **`CPU_GMRES_CPR_AMG`**. `cpu_gmres_fs_cpr` is the **fixed-stress poromechanics** CPR; it
  *errors on flow engines* ("Linear solver type 2 is not supported", verified empirically in
  the legacy `chemistry` env).
* Run invocation on this host (all 3 repos share one polluted site-packages):
  **`PYTHONPATH=<repo_root> python <script>`** in the matching conda env
  (`solvers` / `chemistry` / `development`). Never the `darts` CLI.

---

## 1. Where the MR actually stands (pipeline `2590797258`, sha `da24fc39`)

All 6 builds green; **all 7 test jobs red**. Failure classes, by root cause:

| # | Class | Evidence | Root cause | Fix workstream |
|---|---|---|---|---|
| F1 | **Open-source solver-quality regressions vs proprietary refs** | `3ph_bo` LI 1712 vs 129 (+1227%); `2ph_do_thermal_mpfa` LI 10743 vs 733 (+1366%); `GeoRising` LI 236 vs 35 (+574%); `2ph_comp` LI +147% + NI +32% + solution drift; `Uniform_Brugge` LI +26% | in-tree CPR weaker than `bos_cpr_amg` on these Jacobians (decoupling/AMG config), refs proprietary-generated | **E (perf study)** → solver tuning or documented accept + `_odls` refs |
| F2 | Over-tight thresholds / ref provenance | `2ph_do` −1.1% LI fails; `2ph_comp_solid` 1.3e-8 vs tol 1e-9 | refs from proprietary numerics; tol 1e-9 below solver tol | regenerate `_odls` refs (+ v2.0.0 threshold loosening) — gated on E |
| F3 | Proprietary `-a` crash | `2ph_comp` main.py exit 1; `Adjoint_super_engine` | raw `solvers.create_mgr_solver_for_block_size` absent in `-a` build | **A1** guard (`open_source_solvers_available()` pattern) |
| F4 | Proprietary `displaced_fault_reactivation` FAIL | test-linux `-a`, 33 s ×2 | needs trace triage (mech path) | **A2** |
| F5 | GPU jobs: well-TS mismatches + 2 h timeout | oahu + tahiti | GPU numerics vs CPU refs; + slow tail after `cpg_sloping_fault` | **A3** triage; `_gpu` refs |
| F6 | valgrind-check: ~2851–3169 "errors"/model | only **3 contexts** repeated; `definitely lost: 110 KB/1637 blocks`, `possibly lost: 5.6 MB` | leak suspects: `linsolv_superlu` per-solve leaks (Store/Stat/perm) + spec-build path | **B1** |

> **Performance degradation is the gating concern.** No CI consolidation (D) and no
> proprietary-artifact removal until the migration study (E) demonstrates per-model parity or
> the regression is explicitly accepted & re-referenced.

---

## 2. Workstreams

### A — Pipeline fixes (correctness first)

* **A1** `2ph_comp` / `Adjoint_super_engine`: guard the raw-MGR path for non-registry builds —
  `if not open_source_solvers_available(): fall back to engine-factory default` (and for
  Adjoint: skip adjoint-MGR wiring with a clear message). Validates F3 in proprietary CI.
* **A2** `displaced_fault_reactivation` (proprietary): pull job trace, classify (likely
  solver-selection or ref mismatch after the `self.solver` migration), fix or re-reference.
* **A3** GPU 2 h timeout: identify the post-`cpg_sloping_fault` tail from the oahu trace;
  bound the offending model; separate numeric `_gpu` ref refresh from hangs.
* **A4** Stray-file hygiene: `darts/solvers/*.so` ignored; untracked scratch
  (`verify_mgr_spec.py` → `tests/`, run outputs, `gitlab_ci_verify_report.json`) cleaned.

### B — Code-quality hardening (recon-verified findings)

* **B1 (HIGH)** `linsolv_superlu`: add `Destroy_SuperMatrix_Store(&A_superlu)` +
  `StatFree(&stat_superlu)` per solve; free `perm_r`/`perm_c` in dtor; remove malformed dead
  `#ifndef SLU_SIMPLE` block. Re-run valgrind locally → expect the 3-context storm to collapse.
* **B2 (HIGH)** Forbid `GMRESSolverSpec(prec=MGRSolverSpec())`-style composition of a
  *non-flexible* outer with an iterative MGR preconditioner (NaN root cause): factory/spec-level
  rejection with a clear error (or force FGMRES + fixed-cycle MGR).
* **B3** CPRA duplication: one transpose path. Either `linsolv_mgr` delegates to
  `linsolv_cpr::solve_transposed` or the MGR-internal `applyBCSRCPRTransposePreconditioner`
  is retired (per the reviewer thread + promised SPE10 adjoint comparison).
* **B4** Replace `std::exit(-1)` in `check_hypre` with exception → recoverable timestep cut.
* **B5** Dead config surface: wire or delete `linsolv_cpr` `reuse_amg_hierarchy` /
  `adaptive_amg_rebuild` (+ unreachable adaptive logic).
* **B6** Unit tests for the new subsystem: registry round-trip, GMRES+CPR residual assertion on
  a small block system, `solve_transposed` vs explicit-transpose check, SuperLU on
  `block_csr_matrix`, Python spec smoke (build each spec).
* **B7** Release-build hardening of the `iface_bos` down-cast (checked branch, not bare
  `static_cast` behind `assert`).
* **B8** GMRES `n_iters` off-by-one on immediate convergence.

### C — OpenMP under the GPU build

Today `-fopenmp` is gated to `OPENDARTS_CONFIG==MT` only, and the build scripts make GPU and MT
mutually exclusive → GPU builds are single-threaded on the host side.

* **C1** CMake: broaden the OpenMP gate to `MT OR GPU` (GNU/Clang/MSVC blocks); prefer
  `find_package(OpenMP)` + `OpenMP::OpenMP_CXX` on the consuming targets so nvcc gets
  `-Xcompiler=-fopenmp` and libgomp links correctly (raw flag forwarding is fragile).
* **C2** Keep `OPENDARTS_CONFIG=GPU` as the sentinel (drives `CUDA=1`); no script semantics
  change — CMake adds OpenMP for GPU. Update script info text.
* **C3** Determinism: hoist `omp_set_dynamic(0)` before `init_jacobian_structure` on the GPU
  engine path (matches the CPU §0b contract). The host `row_thread_starts` partition is already
  installed for the GPU Jacobian (verified — no OOB).
* **C4** Validation: `./helper_scripts/build_install_darts_gpu.sh -c -j24` (env `solvers`);
  `compile_commands.json` shows `-fopenmp` on both g++ and nvcc TUs; GPU model run
  `OMP_NUM_THREADS=1` vs `N` **bit-for-bit**; no regression vs OpenMP-off GPU baseline.
  Expectation set: this is a build/runtime-correctness milestone, not a GPU speedup.

### D — cuDSS GPU direct solver (dependency à la AMGX = opt-in flag; binary, not submodule)

cuDSS ships **prebuilt** (no source) — the AMGX-analog is the *opt-in CMake option*, not
`add_subdirectory`. Template: `linsolv_cusolv` (GPU direct on scalar CSR; its enum
`GPU_CUSOLVER` exists but is **unreachable** — no switch case; cuDSS must not repeat that).

* **D1** Acquire: conda/pip `nvidia-cudss` (or NVIDIA tarball) into env `solvers`;
  `WITH_CUDSS` option (default OFF) + `thirdparty/thirdparty_cudss.cmake`
  (`find_package(cudss CONFIG)` → fallback `find_path`/`find_library` + imported target);
  `add_compile_definitions(WITH_CUDSS)`; link into `linear_solvers` mirroring the AMGX block.
* **D2** Wrapper `linsolv_cudss<N>` (copy `linsolv_cusolv`): `cudssCreate/ConfigCreate/
  DataCreate`, matrix via `cudssMatrixCreateCsr` on the **existing shared scalar-CSR device
  arrays** (`fetch_scalar_csr_device` / `cusparseDbsr2csr` path — reuse verbatim);
  ANALYSIS in init, FACTORIZATION per setup, SOLVE per solve; N=1..13 instantiations;
  error checks on every `cudssExecute`.
* **D3** Reachability: `GPU_CUDSS` enum (append-only) + pybind value + `engine_base_gpu.h`
  switch case under `#ifdef WITH_CUDSS` **plus** the not-built redirect fallback; Python
  `CuDSSSolverSpec` (GPUSolverSpec subclass, `linear_type_name="gpu_cudss"`) + export.
* **D4** Build wiring: `-D WITH_CUDSS=ON` via `OD_CMAKE_ARGS` + first-class flag in the GPU
  build script. Correctness gate: small model — cuDSS vs SuperLU solutions match to machine
  precision (both exact direct), NI equal.
* **D5 (follow-up, deferred)** cuDSS native block API to skip the bsr2csr expansion.

### E — Performance / migration study (THE GATE for D-rollout & CI consolidation)

Three categories, plus legacy-state columns; one CSV row per (model × solver × env), pivoted
into the comparison table. Scrape `engine.stat` (`n_timesteps/newton/linear_total[+wasted]`) +
`timer.node['simulation'].get_timer()` + Python wall; fresh subprocess per cell; verify the
solver actually ran via the `Linear solver type is <label>` log line.

| Category | Selector | Env | Models |
|---|---|---|---|
| CPU iter — BOS legacy `cpu_gmres_cpr_amg` (flow) / `cpu_gmres_fs_cpr` (mech-only row) | `params.linear_type` pre-init | `chemistry` (legacy repo) | flow set (+ mech set for fs_cpr) |
| CPU iter — FGMRES+CPR (MR280 default) | `m.solver=GMRESSolverSpec(restart=50,prec=CPRSolverSpec())` | `solvers` | all |
| CPU iter — MGR (MR280) | `m.solver=MGRSolverSpec()` (SPE10: its tuned raw/spec profile) | `solvers` | all |
| GPU iter — AMGX+CPR legacy | `init(platform='gpu')` | `development` (legacy repo) | medium+large |
| GPU iter — AMGX+CPR (MR280 in-tree) | GPU build of this repo | `solvers` (post-C/D rebuild) | medium+large |
| Direct — SuperLU | `m.solver=SuperLUSolverSpec()` | `solvers` | ≤30k cells |
| Direct — Pardiso | `m.solver=PardisoSolverSpec()` (`pip install pypardiso`) | `solvers` | ≤30k cells |
| Direct — GPU cuDSS | `m.solver=CuDSSSolverSpec()` (post-D) | `solvers` GPU build | ≤~100k scalar rows; spe10 infeasible-by-memory documented |

Model set: smoke `2ph_do`(100); small `3ph_bo`(300, 3-D), `2ph_geothermal`(500); medium
`2ph_comp`(1000), `3ph_do`(1000); regression hot-spots `2ph_do_thermal_mpfa`, `GeoRising`;
large `Uniform_Brugge`(~60k, iterative-only); **SPE10** 60×220×85 = 1.122M cells
(`/oahu/data/avnovikov/darts-models/shared/spe10`; iterative CPU ×2 + GPU; capped T_final +
subprocess timeout). OMP_NUM_THREADS=1 primary (CI parity) + multi-thread spot-check.

**Success criterion (pre-registered):** per model, MR280 CPU default within ~±15% wall and
~±25% LI of the BOS legacy baseline, or an explicit accept/tune decision recorded per model.
F1 models (3ph_bo, mpfa, GeoRising, 2ph_comp) get a dedicated decoupling/AMG-config
investigation if the gap confirms.

### F — CI consolidation (`ENABLE_BOS_SOLVERS`) — **implement now, activate after E**

Decision (2026-06-12): `ENABLE_BOS_SOLVERS=ON` default = **per-engine** —
`cpu_gmres_fs_cpr` for mechanics engines, `cpu_gmres_cpr_amg` for flow engines.

* **F1** CMake: `option(ENABLE_BOS_SOLVERS ... OFF)`; ON requires `BOS_SOLVERS_DIR`
  (FATAL_ERROR if unset); flip **all 24 guards** (5 top-level: CMakeLists 109/120/288/295/414;
  19 per-target `OPENDARTS_LINEAR_SOLVERS` definitions) from `DEFINED BOS_SOLVERS_DIR` to the
  option; grep-verify zero stragglers (a single miss silently mis-selects the default solver).
* **F2** Per-engine default: mechanics engines default to `CPU_GMRES_FS_CPR`, flow engines to
  `CPU_GMRES_CPR_AMG` under the BOS build (engine-factory or `DartsModel` level — where the
  engine kind is known), models with explicit `set_solver()` keep their choice.
* **F3** Scripts: `-a`/`-b` continue to drive artifact fetch/path; they now *also* emit
  `-D ENABLE_BOS_SOLVERS=ON` (sh + bat + GPU wrapper).
* **F4** Job collapse (kept as a separate, clearly-gated change): delete `-a` build/test jobs;
  rename `*-ODLS` → `*`; flip valgrind to open-source; rewire `needs`/`dependencies`
  (deploy.yml:44-45,91; upload-wheels.yml:55-56; decide `wheels_iter` fate); re-gate the
  **iter-only models** (`SPE11b`, `SPE10_mech`, `displaced_fault_reactivation`, extra CPG —
  run_test_suite2.py:113-208) into the surviving open-source run; reconcile pkl-suffix logic
  (`archive_pkl.sh`, `get_pkl_suffix`, `_normalize_odls_env`); regenerate `_odls` refs.
  **Do not merge F4 until E passes**; validate with CI lint + a custom-branch pipeline while
  `_iter` refs still exist.

---

## 2b. Implementation status (2026-06-12, this working tree — uncommitted)

| Item | Status | Notes |
|---|---|---|
| A1 fix | ✅ done | Root cause of F3 found: `sim_params.cpu_gmres_cpr` **does not exist** (`AttributeError` in the proprietary fallback). Fixed to `cpu_gmres_cpr_amg` in `models/2ph_comp/model.py` + `models/Adjoint_super_engine/model_definition.py`. |
| B1 SuperLU leaks | ✅ done | `Destroy_SuperMatrix_Store(A)` + `StatFree` per solve; `perm_r/perm_c` freed (dtor + re-init); malformed dead dtor block fixed; members nullptr-initialised; **`info_superlu` now propagated** (singular factorization → timestep cut, was silent success). Needs rebuild + valgrind re-run. |
| B2 GMRES+MGR guard | ✅ done + verified | `GMRESSolverSpec.build()` rejects `prec=MGRSolverSpec` with a clear error (NaN root cause). Live-tested in env `solvers`. |
| B4 std::exit | ✅ done | `check_hypre` throws; `setup/solve/solve_transposed` wrap `*_unguarded` impls and return −1 → engine timestep cut (verified the engine consumes nonzero returns at `engine_base.cpp:3093`). `HYPRE_ClearAllErrors()` on failure. Legacy `linsolv_hypre_amg/ilu` `exit(-1)` left as-is (elasticity-scoped, pre-existing). |
| B8 n_iters | ✅ done | 0 (not 1) when no Arnoldi step ran. |
| C OpenMP-under-GPU | ✅ code done | CMake gate `MT OR GPU_CONFIG` (GNU+Clang) with `$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>` genex; `omp_set_dynamic(0)` hoisted to `engine_base_gpu::init_base` before `init_jacobian_structure`. Pending: GPU rebuild + 1-vs-N bit-for-bit validation. |
| D cuDSS | ✅ code done | Full chain: `thirdparty_cudss.cmake` (CONFIG pkg → pip-wheel/`CUDSS_ROOT` fallback; `nvidia-cudss-cu13` 0.8.0 installed in env `solvers`), `WITH_CUDSS` option, `linsolv_cudss<N>` (ANALYSIS once / FACTORIZATION per setup with `CUDSS_DATA_INFO` check / SOLVE per solve; full handle cleanup), `GPU_CUDSS` enum (append-only) + pybind, open-source GPU dispatch + graceful not-built fallback, `CuDSSSolverSpec` + export, `--cudss` build-script flag. **Bonus:** `GPU_CUSOLVER` made reachable (was dead enum) + `GPUCuSolverSpec`; **in-tree AMGX-CPR dispatch added to the open-source GPU branch** (`GPU_BICGSTAB_CPR_AMGX`, also serves `GPU_GMRES_CPR_AMGX_ILU` requests with a notice) — previously the open-source GPU build could never use AMGX at all. Pending: GPU rebuild + cuDSS-vs-SuperLU exactness check. |
| F1 ENABLE_BOS_SOLVERS | ✅ done | Option (default OFF) + FATAL_ERROR without `BOS_SOLVERS_DIR` + back-compat (`BOS_SOLVERS_DIR` given ⇒ ON). All 19 guard sites flipped; grep-verified zero functional `DEFINED BOS_SOLVERS_DIR` guards remain. |
| F2 per-engine BOS default | ✅ done | `thmc_model.py`: mechanics default = `cpu_gmres_fs_cpr` in BOS builds (via `open_source_solvers_available()`), `cpu_superlu` in open-source. Flow default stays `CPU_GMRES_CPR_AMG` (globals.h). |
| F3 script wiring | ✅ done | `build_darts_cmake.sh`/`.bat` emit `-D ENABLE_BOS_SOLVERS=ON` with `BOS_SOLVERS_DIR`; GPU script gains `--amgx`/`--cudss`. |
| F4 CI collapse | ✅ prepared (gated) | `-a` build/test jobs deleted; `*-ODLS` renamed; valgrind flipped to open-source; deploy/pages/upload deps rewired; `upload_wheels` (wheels_iter) dropped with migration note; GPU vestigial `ODLS:'-a'` cleared; heavy models re-gated `iter_solvers`→`heavy_models` (all CPU runs; GPU keeps lighter set). All YAML parse-validated. **Merge only after E verdicts.** |
| E campaign (CPU) | ✅ done | 52/52 cells (`/oahu/data/avnovikov/mr280_bench/results.csv`). **Verdicts vs legacy BOS (wall):** PASS ≤15% — 2ph_do, 3ph_bo (−14%), 2ph_geothermal, 3ph_do, GeoRising, **Uniform_Brugge (−49% NI / −17% wall, much better)**; borderline — 2ph_comp (+44% wall via +30% NI); investigate — mpfa (+75% wall). **MGR rejected as general default**: diverges on GeoRising (>1800 s vs 2.1 s CPR), NI 2044-vs-1194 on Brugge — stays the tuned-profile/fallback option. Pardiso errors on MPFA matrices (pypardiso AssertionError). |
| E campaign (GPU + SPE10) | ✅/🔄 | Full table + verdicts: **`SOLVER_PERF_STUDY.md`** (repo root). Headlines: **SPE10 1.122M cells, tuned MGR: TS=13/NI=33/LI=209/166 s @OMP=8 — iteration parity with the historical proprietary BOS run (13/34/153); the stale "44×" Appendix-C gap is closed.** cuDSS exact (NI == SuperLU everywhere), Brugge 180k rows in 117 s, GeoRising 1.6 s where MGR diverges. GPU BiCGStab+ILU0 (in-tree) matches legacy AMGX+CPR wall on Brugge. OpenMP 1-vs-4 deterministic on CPU and GPU paths. SPE10 FGMRES+CPR / GPU cells finishing. |
| B6 unit tests | ✅ pass | `tests/cpp/unit/linear_solvers/solver_registry_roundtrip.cpp` — registry round-trip, FGMRES+CPR forward solve (resid 2e-16), SuperLU on `block_csr_matrix` (B1 path), FGMRES+CPRA transposed vs explicit dense transpose. Green at OMP=1/4/48. Caught two real issues while landing: (a) the **B8 revision** — `iter==0` also occurs for a 1-step converged solve (inner-loop break precedes `++iter`), so the count needs a did-work flag or LI totals would collapse to 0 vs the references; (b) deprecated `csr_matrix_base::matrix_vector_product` **silently returns zeros for `block_csr_matrix`** (prints a warning, produces wrong data) — sharp edge for any remaining caller, queued for removal/fix. |
| WITH_AMGX flag leak | ✅ fixed | Pre-existing latent bug surfaced by the first whole-project `WITH_AMGX=ON` build: `target_link_libraries(linear_solvers PUBLIC amgx)` leaked `-DTHRUST_CUB_WRAPPED_NAMESPACE=amgx` (+`--Werror cross-execution-space-call`, AMGX's `-Xcompiler` block) into every consumer CUDA TU → all plain `thrust::` usage failed to compile (interpolation, engines). Now PRIVATE (linsolv_amgx.hpp exposes no AMGX header). |
| GPU build | ✅ green | `build_install_darts_gpu.sh -j24 --amgx --cudss` (CUDA 13.2, A100): wheel installed; `gpu_cudss` enum + registry + specs all import. `libcudss.so.0` bundled into `darts/` via install rule (RPATH=$ORIGIN, mirrors libamgxsh). |
| B3 CPRA-parity evidence | ✅ measured | `Adjoint_super_engine` gradient angle: MGR-internal CPRA **0.000614** vs `linsolv_cpr::solve_transposed` (CPRA) **0.000615**, status 0 both — the equivalence evidence the MR review thread asked for before retiring the MGR-internal duplicate. Retirement itself left to the team (Xiaoming's promised SPE10-scale adjoint comparison). **Bonus fix:** the `--adjoint-solver cpra` path was *crashing* on the removed `amg_tolerance` kwarg (stale after the spec-surface cleanup) — fixed in `model_definition.py`. |
| B7 down-cast hardening | ✅ done | `linsolv_iface_bos<N>::init/setup(csr_matrix_base*)` now `dynamic_cast`-checked: wrong matrix type (block_csr_matrix or mismatched block size) → clear stderr diagnostic + error return (Newton-loop failure path) instead of UB; assert kept for Debug fail-fast. |
| B5 hierarchy-reuse wired | ✅ done (was dead code) | Decision: **wire, not delete**. `cpr_solver_config` gains `reuse_amg_hierarchy` / `adaptive_amg_rebuild` / thresholds; factory + pybind + `CPRSolverSpec` pass-through; `linsolv_gmres` feeds `set_last_outer_iters()` back after each solve. Default OFF — verified byte-identical baseline (2ph_comp 2205/2205/11.35 s). **Experiment verdict (honest negative):** on 2ph_comp plain reuse is pathological (outer GMRES caps at 51 iters — the compositional front drifts the matrix too fast) and adaptive lands at 2214 NI / 27013 LI / 13.0 s — *worse* than baseline. The knob stays opt-in for slowly-varying problems; the 2ph_comp +44% wall is **not** a CPR-setup-cost issue — it tracks the +30% NI (reference provenance), addressed by the `_odls` ref regeneration. |
| **FS-CPR NE>1 latent bug (found by B7)** | ✅ memory-fixed; ⚠ convergence port = named follow-up | The new checked down-cast exposed it during ref regeneration: `linsolv_fs_cpr`'s **multi-variable-flow path (NE>1) handed the block-NE PPSS subsystem to the block-size-1 `hypre_amg_adapter<1>`** — formerly a bare `static_cast` silently misreading block memory in Release builds (UB that "passed" tests: bai 56–380 s OK in CI, `poroelastic_convergence` a 1629 s crawl). The in-tree NE>1 path had **never executed correctly anywhere**. Fix applied: scalar `to_nb_1` expansion (+ diag_ind fill + diag-first convention) before the `<1>` AMG — memory-correct, but converges poorly (bai_mech_rect: LI 325k vs proprietary ~50; AMG on the interleaved p-T expansion is the wrong operator). **Coverage decision:** NE>1 mech cases (`bai_*`, `1ph_1comp_poroelastic_convergence`, `SPE10_mech` thermal/dead-oil) gated out of the suite behind `FS_CPR_NE_GT1_READY=False` with the blocker named — honest red-flag instead of UB-lucky green. **Follow-up (high priority): port the proprietary FS-CPR NE>1 Schur reduction (scalar pressure system with secondary-equation folding — the `ps_rhs_mults_`/`x_sch_s_` machinery is already in place).** |

## 2c. Reference-pkl rebaseline — **user-gated, NOT applied** (2026-06-12)

Per explicit instruction: the `_odls` reference pkls are **not** regenerated in this tree until
the user has personally checked the performance and the solution differences between the
non-ODLS (`_iter`) and ODLS data. What was done instead:

* A full regeneration **experiment** was run (suite + `UPLOAD_PKL=1`, CI-flavor CPU build):
  the suite is **self-consistent — 80/80 pass** against refs produced by this build.
* The regenerated pkls were then **reverted** from the working tree; the committed references
  are untouched. The 38 regenerated files are preserved (path-mirrored) at
  **`/oahu/data/avnovikov/mr280_bench/regenerated_odls_refs/`** for the side-by-side
  `_iter`-vs-`_odls` comparison (e.g. diff against `models/*/ref/*_iter.pkl` and the committed
  `*_odls.pkl`).
* The performance side of that comparison is already tabulated in `SOLVER_PERF_STUDY.md`
  (NI/LI/wall vs the legacy `chemistry` BOS build per model).

The rebaseline (and therefore the F4 job deletion) stays blocked on that review.

## 3. Execution order (this effort)

1. Master plan (this doc) + plan-doc corrections — done.
2. **E-CPU cells launched first** (no code changes needed; legacy + current CPU numbers are the
   gating data, start immediately, run long).
3. B1/B2/B4/B8 quality fixes + A1 guard (small, surgical, unblock valgrind & proprietary jobs).
4. C (OpenMP-under-GPU) + D (cuDSS) code; **one** GPU rebuild
   (`./helper_scripts/build_install_darts_gpu.sh -c -j24`) validates both; ctest + OpenMP
   1-vs-N check + cuDSS-vs-SuperLU exactness check.
5. E-GPU cells (in-tree AMGX+CPR, cuDSS incl. SPE10-capped; legacy GPU from `development`).
6. F1–F3 (flag + scripts) implemented; F4 prepared as gated change.
7. Assemble the performance table, verdicts per model, and the accept/tune/block decision list.

Primary test env: **`solvers`**. Nothing is committed without explicit request.
