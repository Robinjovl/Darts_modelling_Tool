# #.#.# [Future]
- Solvers ([!280](https://gitlab.com/open-darts/open-darts/-/merge_requests/280)):
  - Open-source linear-solver stack moved in-tree (FGMRES+CPR default, MGR with BCSR-CPR/True-IMPES, SuperLU; AMGX + GPU wrappers; unified `LinearSolverSpec` API via `self.linear_solver`)
  - New: NVIDIA **cuDSS** GPU sparse direct solver (`CuDSSSolverSpec`; `WITH_CUDSS` is ON by default and consumed only by GPU builds -- disable with `-D WITH_CUDSS=OFF`). The prebuilt library is located via the `nvidia-cudss-cu*` wheel, `CUDSS_ROOT` or a CMake config package; if it is not found the GPU stack is built without cuDSS (warning, not error). cuSOLVER QR direct solver (`GPUCuSolverSpec`) made selectable
  - **OpenMP enabled in the GPU build.** The host-side assembly and Krylov kernels now run multi-threaded in a CUDA-enabled build (previously they were serial there), so a single GPU wheel also delivers multi-core CPU performance -- which matters for CPU-only runs on a GPU-capable machine and for models that have no GPU engine at all (mechanics/poromechanics are CPU-only). **Reproducibility**: the block SpMV is parallelised over block rows, so the decomposition is write-disjoint and each output block keeps its serial accumulation order -- it is bit-identical at *any* thread count. The dot/norm reductions instead sum per-thread partials into private slots that are combined in fixed thread-index order: this removes run-to-run variance at a *fixed* thread count, which a bare `#pragma omp parallel for reduction(+:s)` does not, because it combines partials in nondeterministic completion order and floating-point addition is not associative (the ULP difference then amplifies through the Krylov iteration). Across *different* thread counts the partials are grouped differently, so reduction results still differ at ULP level -- inherent to any parallel reduction, within the solver tolerance, and the reason reference comparisons must fix the thread count
  - New CMake option **`ENABLE_BOS_SOLVERS`** (default OFF) replaces the implicit `BOS_SOLVERS_DIR` switch for linking the proprietary BOS solvers; mechanics models default to `cpu_gmres_fs_cpr` in BOS builds
  - CI consolidated: the open-source lane is now the unsuffixed default (`build-linux` / `test-linux`, `build-windows` / `test-windows`), replacing the former `-ODLS` twins; the proprietary lane is retained as explicit `-BOS` jobs (`ENABLE_BOS_SOLVERS=ON`, `-a` build flag), gated by the `RUN_BOS_JOBS` CI variable
  - Fixes: SuperLU per-solve resource leaks + silent acceptance of singular factorizations; HYPRE failures in CPR now propagate as timestep cuts instead of `std::exit`; `GMRESSolverSpec(prec=MGRSolverSpec())` rejected (unsound composition); AMGX thirdparty compile flags no longer leak into consumer CUDA code; first unit-test coverage for the solver registry / FGMRES+CPR / CPRA / SuperLU-on-block-CSR
- **Breaking changes** ([!280](https://gitlab.com/open-darts/open-darts/-/merge_requests/280)) — these can change results or break existing models:
  - **Renamed: the `solvers` library/package is now `linear_solvers`, and `DartsModel.solver` is now `DartsModel.linear_solver`.** The C++ library folder `solvers/` moved to `linear_solvers/` (CMake target `opendarts_solvers` → `opendarts_linear_solvers`, shared library `libopendarts_solvers` → `libopendarts_linear_solvers`); the Python package `darts.solvers` is now `darts.linear_solvers` (compiled extension `solvers` → `linear_solvers`); the pip extra `open-darts[solvers]` is now `open-darts[linear_solvers]`. Update imports (`from darts.linear_solvers import GMRESSolverSpec, ...`) and model code (`self.linear_solver = <Spec>` in `set_solver()`).
  - **`DartsModel.linear_solver` now holds a runtime `LinearSolver` instance** (`darts.linear_solvers.LinearSolver`), mirroring `DartsModel.nonlinear_solver` holding a `NewtonSolver` ([!327](https://gitlab.com/open-darts/open-darts/-/merge_requests/327)). Assignment is normalizing -- `self.linear_solver = GMRESSolverSpec(...)` still works and is auto-wrapped; the declarative spec is `linear_solver.spec` (tune the default via `self.linear_solver.spec.tolerance = ...` after `super().set_solver()`), and after `init()` the live backend is `linear_solver.handle` (engine-injected C++ solver) or `linear_solver.python_solver` (PETSc/Pardiso). Code that read spec fields off the attribute directly (`model.linear_solver.tolerance`) must go through `.spec`. The compiled raw-handle class previously exposed as `darts.linear_solvers.LinearSolver` is shadowed by the runtime class; it stays reachable as `darts.linear_solvers.linear_solvers.LinearSolver` (alias `LinearSolverInterface`).
  - **Default solvers are now declared explicitly in `DartsModel.set_solver()`, and the default-solver helper functions are removed.** `darts.linear_solvers.default_linear_solver()` and `default_linear_solver_spec()` are gone (mirroring [!327](https://gitlab.com/open-darts/open-darts/-/merge_requests/327)'s removal of `default_nonlinear_solver()` / `default_nonlinear_spec()`); `set_solver()` now constructs both the nonlinear and the per-platform linear default with **every parameter stated explicitly**, so the effective configuration of a model that does not override it is readable in one place instead of being hidden in the spec dataclass defaults. The constructed defaults are unchanged (CPU: FGMRES restart=50 + CPR/BoomerAMG; GPU: AMGX-CPR) -- verified field-by-field against the removed helpers. `LinearSolver` correspondingly loses its lazy platform resolution (`resolve_spec()`) and the `LinearSolver(tolerance=...)` keyword sugar: pass a spec.
  - **The default linear solver changed.** Open-source **FGMRES+CPR** (in-tree) is now the default on CPU, replacing the proprietary BOS CPR+AMG. On GPU the default is **AMGX-CPR** (`gpu_gmres_cpr_amgx_ilu`). Iteration counts, timings and (at loose tolerances) the Newton trajectory can differ from previous runs — **re-baseline performance/reference data**. Pin the old behaviour explicitly with `self.linear_solver = ...SolverSpec(...)` (or `ENABLE_BOS_SOLVERS=ON` + `proprietary_linear_type`).
  - **`set_sim_params()` no longer accepts linear-solver parameters.** `tol_linear` and `it_linear` are **removed** from the signature. For one deprecation cycle they are still accepted: they emit a `DeprecationWarning` and are mapped onto `self.linear_solver.spec` (`.tolerance` / `.max_iterations`) by `_migrate_legacy_solver_kwargs()` — see the integration note below. They are removed outright next release. (Only an *unknown* keyword raises `TypeError`; a raw-handle `linear_solver` with no spec raises `RuntimeError`.) Note also that `set_sim_params()` now **always** emits a `DeprecationWarning`, even when only timestep arguments are passed. `set_sim_params()` configures time-stepping only. The linear solver is configured on `self.linear_solver` (a `LinearSolverSpec`) inside `set_solver()`:\
    {- Before: self.set_sim_params(..., tol_newton=1e-3, tol_linear=1e-6, it_linear=50) -}\
    {+ Now:    self.set_sim_params(...)                          # time-stepping only
               super().set_solver()                              # default solvers
               self.nonlinear_solver.spec.tolerance = 1e-3       # nonlinear knobs
               self.linear_solver.spec.tolerance = 1e-6          # linear knobs
               self.linear_solver.spec.max_iterations = 50 +}\
    (Equivalently, name the solver outright: `self.linear_solver = GMRESSolverSpec(tolerance=1e-6, max_iterations=50, prec=CPRSolverSpec())`.)
  - **`DataTS` no longer carries linear-solver settings.** `data_ts.linear_tol`, `linear_max_iter`, `linear_type` and `linear_print_level` are **removed**: `self.linear_solver` is the single owner of the linear solver (its choice, `tolerance`, `max_iterations`, `print_level`), and `DartsModel._apply_solver()` mirrors these into `sim_params` before `engine.init()`. Assignments to `data_ts.linear_*` are now silently inert — grep for them. Case-driven models can keep the value in their own input data (e.g. `idata.sim.linear_tol`) and apply it in `set_solver()`, as `cpg_sloping_fault` does.
  - **A solver spec's `tolerance` / `max_iterations` are now actually applied.** Previously the engine overwrote them at `init()` with `sim_params` (`tolerance_linear` / `max_i_linear`, defaults **1e-5 / 50**), so for engine-resident solvers a spec's values were **decorative**. They are authoritative now. If your model declares a spec tolerance it never really used, it will take effect — check it is one the preconditioner can actually reach. (The five shipped mechanics models declared `tolerance=1e-8, max_iterations=200` while really solving at 1e-5/50; they were **pinned to 1e-5/50 so their behaviour is unchanged**. FS-CPR does not reach 1e-8 on those systems anyway — on `SPE10_mech` 22 of 48 solves exhaust the 200-iteration cap, averaging 99 linear iterations per Newton step versus 41 at 1e-5/50.)
  - **`set_solver_params()` removed** and `data_ts.linear_solver` retired — the single per-model home for solver configuration is now `set_solver()` + `self.linear_solver`.
  - **Integration with the nonlinear-solver refactoring ([!327](https://gitlab.com/open-darts/open-darts/-/merge_requests/327)).** After both merge: `set_sim_params()` is **timestep-only** — the nonlinear keywords (!327) and the linear ones (!280) are all removed, and for one deprecation cycle both families are mapped onto the corresponding spec with a `DeprecationWarning` by `_migrate_legacy_solver_kwargs()`. `DataTS` keeps only `dt_*`/`eta`; `copy_data_ts_to_sim_params()` is a retained no-op (the C++ `sim_params` timestep/Newton fields no longer exist). `DartsModel._solve_linear_equation()` keeps !280's spec-driven routing (Python-resident PETSc/Pardiso vs the engine solver) but now returns !327's `(rc, n_iters, residual)` contract, so the `PythonLinearSolver` backends report iterations and residual too. Mechanics/THMC models, which drive the linear solver through `params.linear_type`/`engine.ls_params` rather than a spec, are identified by the new `DartsModel.linear_solver_from_engine_factory` class flag (previously discriminated by `data_ts is None`, which !327's lazy `data_ts` property made always false).
  - **`sim_params.linear_solver_t` enum values were renumbered**: a new `CPU_GMRES_MGR` was inserted at the end of the CPU block (before `GPU_GMRES_CPR_AMG`), so that all CPU methods precede the GPU block (several engines classify GPU-vs-CPU by `linear_type >= GPU_GMRES_CPR_AMG`), and `GPU_GMRES_CPR_NF` was removed from the GPU block. **Net effect on the integer values relative to the 1.5.1 / `development` numbering:** the `+1` from inserting `CPU_GMRES_MGR` and the `-1` from removing `GPU_GMRES_CPR_NF` cancel for the GPU members that came *after* `GPU_GMRES_CPR_NF`, so those integers are unchanged, while the GPU members *before* it shifted by `+1` — i.e. **some but not all `GPU_*` integers changed**. Code referring to solvers by **name** (`sim_params.cpu_superlu`, …) is unaffected; code that hardcodes the **integer** value of a `GPU_*` member should switch to the named member. **One name was also removed:** `GPU_GMRES_CPR_NF` / `gpu_gmres_cpr_nf` (a GPU CPR-NF variant exposed on `development`) is gone with no replacement — a model naming it hits `AttributeError` even though it used a name, not an integer. (New names added: `CPU_GMRES_MGR`, `GPU_CUDSS`.) The corresponding solver code is now gone as well: the `linsolv_adgprs_nf` construction in `engine_super_mp_cpu` / `engine_super_elastic_cpu` (guarded by a `WITH_ADGPRS_NF` macro that no build defined, and referencing the already-deleted `GPU_GMRES_CPR_NF` enumerator, so it was unreachable in every configuration) has been removed, together with its only input **`sim_params.global_actnum`** -- a read/write Python attribute that nothing read once the NF path was gone. Setting it had no effect; it now raises `AttributeError`.
  - **GPU builds now always build AMGX.** `WITH_AMGX` defaults to `ON` and the `--amgx` flag was removed from `build_install_darts_gpu.sh` (AMGX backs the default GPU solver). The `thirdparty/AMGX` submodule is initialised automatically; a GPU build without it fails loudly.
  - **Python-resident solvers (PETSc / Pardiso) now report failure.** A non-finite solution returns a non-zero status (recorded in `NonlinearSolver.status.linear_solver_rc`), so the Newton loop cuts the timestep instead of silently accepting a NaN update. Models that previously "converged" through such a solve will now cut and may take a different path.
  - **Static interpolators and the `itor_mode` / `mode` parameter are removed.** `'static'` interpolation is gone from the Python layer: `PhysicsBase.init_physics()`, `PhysicsBase.set_interpolators()`, `PhysicsBase.create_interpolator()`, `Initialize()`, the chemistry/EoS physics constructors and `DartsModel.init()` no longer accept `itor_mode` (or `mode`), and `PhysicsBase.STATIC_GRID_N_POINTS` is deleted. Interpolation is always adaptive (unbounded grid, cells enumerated on demand). Rationale: the static interpolator classes were **never exposed to Python** (their `expose_class` calls had been commented out in `interpolation/pybind11/py_interpolator_exposer.hpp`, so any `'static'` request already failed with "No compiled OBL interpolator template"), and static dense storage needs a *bounded* OBL grid, which the OBL API no longer expresses since user bounds / point counts were dropped. The now-unreachable C++ templates are deleted as well: `linear_static_cpu_interpolator`, `multilinear_static_cpu_interpolator` and `multilinear_static_gpu_interpolator` (headers, `.tpp`s, their `CMakeLists.txt` entries and the dead pybind branch). The count of interpolator classes exposed to Python is unchanged at 471. Migration: delete the argument from the call — `itor_mode='adaptive'` was the only working value and is now the sole behaviour. Existing OBL cache files remain valid (the cache signature keeps its historical `_adaptive_` token).
  - **`PropertyContainer(rate_ann_mat=...)` removed (dead code).** The rate-annihilation matrix and the `nelem` attribute derived from it were write-only: nothing in the Python or C++ layers ever read `property_container.rate_ann_mat` / `.nelem`. The keyword argument is gone from `darts.physics.base.property_container.PropertyContainer` and `darts.physics.chemistry.property_container.PropertyContainer`; passing it now raises `TypeError`. Drop the argument from the call — there is no replacement, because it had no effect.
  - **`DataTS` (`data_ts`) is now a validated, serializable timestepping structure** — the timestepping analogue of the nonlinear/linear specs. It gains `validate()` (called at `init()`; raises on `dt_first<=0`, `dt_min<=0`, `dt_max<dt_min`, `dt_mult<1`, `runtime<=0`), `to_dict()`, and it now owns `runtime` (previously an orphan `self.runtime` attribute that raised `AttributeError` if `run()` was called with no `days` before any `set_sim_params`). `DartsModel.runtime` is a property single-sourced on `data_ts.runtime`. New `DartsModel.print_config()` dumps the effective timestepping + nonlinear + linear configuration in one place. This change is **behaviour-neutral** (SPE10 timestep trajectory bit-identical). Note a **known, deliberately-unchanged discrepancy**: `data_ts.dt_min` defaults to `1e-12` when a model constructs `DataTS` directly, but `set_sim_params(min_ts=)` defaults to `1e-15`, so the two configuration paths floor the divergence-abort timestep at different values; this is now observable via `print_config()`, and unifying it is deferred (it would shift the abort floor for models that ride it and require re-baselining their references).

# 1.5.1 [14-07-2026]
- DARTS-flash integration: pre-defined DARTS-flash Mixtures ([!324](https://gitlab.com/open-darts/open-darts/-/merge_requests/324)); bumped `open-darts-flash` dependency to 0.13.0.
  - EoS-based Physics structure:
    - New `EoSPhysics` (`darts.physics.eos_physics`, inherited from `PhysicsBase`) with methods to specify the mixture (`dartsflash.Mixture` type) and check consistency, keeping the API for setting up mixture/EoS completely on the darts-flash side: `set_mixture(mixture, region=...)` attaches a configured `Mixture` and checks that its components match the physics components and that its `FlashType` is compatible with the physics `StateSpecification`; `init_physics()` asserts a mixture was attached.
    - `EoSPhysics` methods to evaluate properties (density, enthalpy, fugacity) from the flash: `EoSPhysics.get_density_ev_from_flash(phase_idx)` / `get_enthalpy_ev_from_flash(phase_idx)` / `get_fugacity_ev_from_flash(phase_idx)` return the corresponding evaluator wired to the attached flash.
    - New `IAPWSPhysics` (`darts.physics.iapws_physics`, inherited from `EoSPhysics`): specific IAPWS (`dartsflash.IAPWS`) implementation of `EoSPhysics` for single-component H2O; checks consistency and keeps the API for setting up the mixture on the darts-flash side.
  - Changes to property evaluators:
    - `EoSDensity` and `EoSEnthalpy` can now take `flash_ev` and `phase_idx` arguments (instead of a direct `eos` object) to evaluate the specified property using internal darts-flash `EoSResults` (v0.13.0) logic; new `EoSFugacity` evaluator with the same interface. **Breaking:** `EoSDensity` no longer takes the `Mw` argument (molar weights come from the EoS `comp_data`), and the direct-EoS path is now keyword-based: `EoSDensity(eos=..., root_flag=...)`.
    - Option to evaluate Rachford-Rice with the darts-flash solver (`RR_EqConvex2`) in the `ConstantK` flash via the `use_dartsflash` argument, turned off by default. K-values keep their existing `Ki = yi/xi` definition (y-phase returned as phase 0); they are inverted internally to darts-flash's phase-0-as-reference convention (`Ki = xi1/xi0`) when `use_dartsflash` is enabled.
    - Added `PropertyContainer.check_properties()` to check consistency of input properties (density/enthalpy/conductivity evaluators for all phases, viscosity/diffusion/relperm for mobile phases, kinetic/energy-source entries); it is called for every property region in `PhysicsBase.init_physics()`. `PropertyContainer.energy_source_ev` is now a dict of evaluators (was a single optional evaluator).
- Nonlinear solver refactoring (`nonlinear_refactoring` branch):
  - **New single input source for the nonlinear solve: the `darts.nonlinear_solvers` package** (`base.py` — enums, sub-specs, base spec and runtime base classes; `newton.py` — the Newton specs and `NewtonSolver`; re-exported from the package root). Declarative spec dataclasses `NonlinearSolverSpec` / `NewtonSpec` (with `ChopSpec`, `OBLBoundsSpec`) are the declarative input (stdlib dataclasses; dict/JSON-serializable and Pydantic-forward-compatible); **`DartsModel.nonlinear_solver` holds the runtime `NewtonSolver` instance** built from a spec — assigned in the `set_solver()` hook (same hook as the linear solver spec of [!280](https://gitlab.com/open-darts/open-darts/-/merge_requests/280)) or inline in the constructor. The solver is constructed *detached* (no model needed) and `bind()`s to the model in `init()`; the input spec stays retrievable as `model.nonlinear_solver.spec` / `.to_spec()` and serialized via `.spec.to_dict()` (uses `dataclasses.asdict`, delegating to `model_dump()` if a spec is later a Pydantic model — the MR300 `darts/api/autospec.py` tracing convention). Timestep control (`dt_first`/`dt_min`/`dt_mult`/`dt_max`/`eta`) lives ONLY in the `data_ts` structure (not in `sim_params` anymore).
  - **Breaking (model API):** `set_sim_params()` no longer accepts any nonlinear-solver argument (`tol_newton`, `it_newton`, `newton_type`, `newton_params`, `line_search`, `coupled_well_res_norm_method` are removed); it now sets only the timestep and linear-solver parameters (`first_ts`/`mult_ts`/`min_ts`/`max_ts`/`runtime`/`tol_linear`/`it_linear`). Specify the nonlinear solver via `self.nonlinear_solver = NewtonSolver(tolerance=…, max_iterations=…, chop=ChopSpec(mode=…, factor=…))` (accepts a `NewtonSpec` positionally or its keyword arguments), and tune the default via `self.nonlinear_solver.spec.<field>`. All in-repo models were migrated accordingly.
  - `run_timestep()` moved from `DartsModel` to `darts.nonlinear_solvers.NewtonSolver`; the model method delegates, so existing overrides keep working. A per-iteration `on_iteration()` hook replaces the forked Newton loop in `plot_live`.
  - **Staged nonlinear iteration**: every iteration is decomposed into explicit `pre_iteration()` (user routines), `update()` (the spec-assembled dX-correction pipeline — composition correction, global/local chop, OBL-bounds constraints, thermal-variable correction — followed by the plain Newton step) and `post_iteration()` (user routines) solver methods. The C++ `apply_newton_update` composite was split into the self-guarded kernels `correct_composition` / `correct_chop_global` / `correct_chop_local` / `correct_obl_axes` / `correct_thermal` / `apply_update`, all bound to Python (the legacy composite remains, behavior-identical).
  - **Single source of truth**: `DataTS` (`model.data_ts`) holds only the timestep controls (`dt_*`, `eta`) and the transitional linear settings; the nonlinear-solver settings live solely on `model.nonlinear_solver.spec` (the `data_ts.newton_tol`/`newton_max_iter`/`newton_tol_stationary`/`newton_tol_wel_mult`/`coupled_well_res_norm_method` aliases are removed — read/write `nonlinear_solver.spec.tolerance`/`max_iterations`/`stationary_point_tolerance`/`well_tolerance_multiplier`/`coupled_well_res_norm_method`). The `DartsModel._get_nonlinear()` helper is gone: `model.nonlinear_solver` is the bound solver instance itself (its `.status`, `.stats`, `.spec` are read directly).
  - **Divergence fallbacks**: `NonlinearSolverSpec.fallbacks` is an ordered list of `FallbackSpec` tried by `NonlinearSolver.solve_timestep()` when the primary solve fails, before the driver cuts the timestep — each fallback retries the same dt with another solver spec and/or additional built-in or user-defined pre/post routines (`pre_routines`/`post_routines`, signature `f(solver, dt, t, iteration)`).
  - Python/C++ separation: the convergence decision, statistics, stationary-point detection and the iteration/timestep log lines are Python-side; the C++ engine keeps only the cell-looping kernels (assembly + OBL interpolation, residual norms, linear solve, dX corrections, timestep state commit/rollback).
  - **Breaking (C++ engine interface):** `engine.post_newtonloop(dt, t, converged)` now requires the Python convergence verdict; the engine members `n_newton_last_dt`, `n_linear_last_dt`, `newton_residual_last_dt`, `well_residual_last_dt`, `linear_solver_error_last_dt` and `engine.stat` (class `sim_stat`) are removed — use `NonlinearSolver.status` (per-timestep) and `NonlinearSolver.stats` (cumulative) instead, plus `engine.get_last_linear_iters()` / `engine.get_last_linear_residual()` for the last linear solve.
  - **Breaking (`sim_params`):** all nonlinear and timestep control fields removed — `first_ts`/`max_ts`/`mult_ts`/`min_ts`, `tolerance_newton`/`max_i_newton`/`min_i_newton`, `newton_type`/`newton_params`, `nonlinear_norm_type`, `log_transform`, `line_search`, `stationary_point_tolerance`, `well_tolerance_coefficient`, `obl_min_fac`, `tot_newt_count`, `interface_avg_tmult`. What remains is the linear-solver + physics rump, deleted entirely once [!280](https://gitlab.com/open-darts/open-darts/-/merge_requests/280) is merged. The kernel-level controls now live on the engine (`newton_chop_mode`, `newton_chop_factor`, `log_transform`, `residual_norm_type`) and are synced from the spec before every timestep solve.
  - `set_sim_params()` (timestep + linear only, see above) and `set_sim_params_data_ts()` remain as deprecated shims writing into `data_ts`. For one deprecation cycle `set_sim_params()` still accepts the removed nonlinear keyword arguments (`tol_newton`/`it_newton`/`newton_type`/`newton_params`/`coupled_well_res_norm_method`): they emit a `DeprecationWarning` and are mapped onto `model.nonlinear_solver.spec` (unknown keywords still raise `TypeError`). The historic nonlinear *attribute* names are gone from `DataTS` — use `model.nonlinear_solver.spec.*`; only the timestep controls and the transitional linear settings remain plain attributes on it until [!280](https://gitlab.com/open-darts/open-darts/-/merge_requests/280).
  - **`MechanicsNewtonSolver`** (`darts.nonlinear_solvers.mechanics`) drives the geomechanics engines' Newton loop (deviatoric per-component residual, per-component convergence, the C++ `apply_newton_update` composite) through overridable hooks, so the mechanics models share one driver instead of each copying the loop.
  - Verified bit-identical (timestep / Newton / linear counts and solution hash) on 2ph_do, 2ph_comp, 3ph_bo and 2ph_geothermal; mechanics validated via the poroelastic convergence studies and the Mandel analytic run. Fake-engine unit tests live in `tests/test_nonlinear_solver.py`.
- Migration guide (`nonlinear_refactoring`) — the nonlinear-solver settings moved off `sim_params`/`data_ts` onto `DartsModel.nonlinear_solver` (a `NewtonSolver` instance holding a `NewtonSpec`). Removed symbols raise `TypeError`/`AttributeError` (except `set_sim_params()`'s nonlinear keywords, kept one release with a `DeprecationWarning`).
  - **Configure the nonlinear solver** (in a `set_solver()` override or the model constructor):
    ```
    # Before
    self.set_sim_params(first_ts=1e-3, tol_newton=1e-4, it_newton=15,
                        newton_type=sim_params.newton_local_chop, newton_params=[0.2])
    # After
    from darts.nonlinear_solvers import NewtonSolver, ChopSpec
    self.nonlinear_solver = NewtonSolver(tolerance=1e-4, max_iterations=15,
                                         chop=ChopSpec(mode='local', factor=0.2))
    self.set_sim_params(first_ts=1e-3, tol_linear=1e-6, it_linear=200)  # timestep + linear only
    ```
  - **Read/write nonlinear settings** — the `data_ts.newton_*` aliases are gone; use `model.nonlinear_solver.spec.*`:
    ```
    data_ts.newton_tol                    -> nonlinear_solver.spec.tolerance
    data_ts.newton_max_iter               -> nonlinear_solver.spec.max_iterations
    data_ts.newton_tol_stationary         -> nonlinear_solver.spec.stationary_point_tolerance
    data_ts.newton_tol_wel_mult           -> nonlinear_solver.spec.well_tolerance_multiplier
    data_ts.coupled_well_res_norm_method  -> nonlinear_solver.spec.coupled_well_res_norm_method
    ```
  - **Statistics / per-timestep status** — `engine.stat` and the `engine.*_last_dt` members are removed:
    ```
    engine.stat.n_newton_total     -> model.nonlinear_solver.stats.n_newton_total  (same field names)
    engine.n_newton_last_dt        -> model.nonlinear_solver.status.n_newton
    engine.n_linear_last_dt        -> model.nonlinear_solver.status.n_linear
    engine.newton_residual_last_dt -> model.nonlinear_solver.status.newton_residual
    # last linear solve: engine.get_last_linear_iters() / engine.get_last_linear_residual()  (new)
    ```
  - **Removed `sim_params` fields** (`first_ts`/`max_ts`/`mult_ts`/`min_ts`, `tolerance_newton`/`max_i_newton`/`min_i_newton`, `newton_type`/`newton_params`, `nonlinear_norm_type`, `line_search`, `stationary_point_tolerance`, `well_tolerance_coefficient`, `log_transform`, `tot_newt_count`, `interface_avg_tmult`, `obl_min_fac`) — timestep controls live on `model.data_ts` (`dt_first`/`dt_min`/`dt_mult`/`dt_max`/`eta`), nonlinear controls on the `NewtonSpec`.
  - **Custom Newton loops**: models overriding `run_timestep` or copying the loop should instead use `NewtonSolver`; geomechanics models use `MechanicsNewtonSolver` and override its residual/convergence hooks (`compute_mech_residual`, `check_early_break`, `finalize_convergence`) rather than reimplementing the loop.
  - **Constraining the Newton state (OBL axis bounds)**: pass `NewtonSolver(..., obl_bounds=OBLBoundsSpec(mode='obl_axes', axis_min=[...], axis_max=[...]))` (or set `nonlinear_solver.spec.obl_bounds`). `axis_min`/`axis_max` are per-state-variable bounds in `[p, z_1, ..., z_{nc-1}, (T)]` order (length `n_vars`; `None` leaves an axis unbounded — composition axes are usually `None`, the composition correction already projects `z` onto the simplex). Each Newton iteration then clamps the update `dX` per cell/variable so the post-update state stays strictly inside the box — used to hold the trajectory in a physically/numerically valid window on the (otherwise unbounded) adaptive OBL grid. Default (`mode=None`) applies no constraint.
  - **Out-of-tree C++ engine subclasses**: `post_newtonloop(dt, t)` gained a third argument, `post_newtonloop(dt, t, converged)` (the Python-side convergence verdict); the nonlinear loop now lives in Python (`NewtonSolver.run_timestep`), and the engine keeps only the per-iteration kernels (`assemble_linear_system`, `calc_newton_residual`/`calc_well_residual`, `correct_composition`/`correct_chop_*`/`correct_obl_axes`/`correct_thermal`, `apply_update`, `solve_linear_equation`, `post_newtonloop`).
- Breaking changes ([!313](https://gitlab.com/open-darts/open-darts/-/merge_requests/313)):
  - **OBL grid API: the legacy bounds arguments are removed. Physics classes now take `axes_step` (required, per-axis cell size) and `axes_origin` (optional, per-axis grid origin) only.** Removed everywhere: `n_points`, `min_p`/`max_p`, `min_z`/`max_z`, `min_t`/`max_t`, `min_e`/`max_e`, `axes_min`/`axes_max`, `n_axes_points`, and `PhysicsBase.determine_obl_bounds()`. `epsilon_z` became a keyword argument (default `1e-9`). The grid is unbounded, so there is no `axes_max` or point count. Passing any removed argument raises `TypeError`. Affects `Compositional`, `Geothermal`, `ElementBasedReactiveFlow`, `Poroelasticity` and `PhysicsBase.create_interpolator` (see the migration guide below).
  - Physics instance fields renamed: read `physics.axes_origin` where you read `physics.axes_min`; `physics.axes_max`, `physics.n_axes_points`, `physics.PT_axes_min` and `physics.PT_axes_max` are gone (`physics.axes_step` gives the per-axis cell size; the P-T window lives in `physics.thermal_var_axes_origin` / `physics.thermal_var_axes_step`).
  - `InputData` OBL fields renamed (`idata.obl`): `n_points`, `min_p`/`max_p`, `min_z`/`max_z`, `min_t`/`max_t`, `min_e`/`max_e` become `p_step`/`p_origin`, `z_step`/`z_origin`, `t_step`/`t_origin`, `e_step`/`e_origin` (`idata.obl.epsilon_z` and `idata.obl.zero` kept), with the same `(max - min)/(n_points - 1)` conversion.
  - Shipped wheels and CI now build the `MINIMAL` interpolator profile, which does **not** include the linear adaptive interpolator. Models requesting `algorithm='linear'` must switch to `'multilinear'`, or build from source with `-D OPENDARTS_INTERPOLATOR_PROFILE=FULL`.
  - Live PH-diagram: `plot_live` no longer auto-derives its axes from the (now unbounded) OBL grid — set `LivePlotConfig.p_bounds`, `h_bounds` and `n_points` before enabling `enable_ph_diagram`.
  - Output / post-processing: `output.py` drops OBL-window single-precision state clipping and the P-T dead-operator guard, and the `body_path.txt` / output-header format changed from `n_points min max` to per-axis `origin step` with space-joined multi-index hypercube keys — update any parser of those files.
  - The `darts.engines.uint128` Python binding and the 128-bit-index interpolator instantiations were removed (superseded by the multi-index storage); out-of-tree C++ engine subclasses overriding `get_n_ops()` must change the return type to `uint16_t` and be recompiled.
  - The index-type template parameter was dropped from all interpolator classes — storage is keyed on the signed multi-index (`cell_key_t`, int32 per axis), never on a packed integer, and the linear family's vertex enumeration is now int32-native (a vertex converts to a cache key by a plain element copy; the standard-triangulation Kuhn walk from the shared hypercube corner is unchanged, and the static dense index uses `uint64_t`): `multilinear_adaptive_{cpu,gpu}_interpolator<index_t, value_t, N_DIMS, N_OPS>` become `<value_t, N_DIMS, N_OPS>`; `linear_{adaptive,static}_cpu_interpolator<index_t, N_DIMS, N_OPS>` become `<N_DIMS, N_OPS>`. Exposed Python class names lose the index-type letter accordingly:\
  {- Before: darts.interpolators.multilinear_adaptive_cpu_interpolator_l_d_2_10 (and the _i_ uint32 GPU twin) -}\
  {+ Now:    darts.interpolators.multilinear_adaptive_cpu_interpolator_d_2_10 (one class per precision/dims/ops) +}\
  `PhysicsBase.create_interpolator` resolves the new names and still falls back to the legacy `_i_`/`_l_` names when running against an older compiled module, so models using `create_interpolator`/predefined physics need no change; only code instantiating `darts.interpolators.*_i_*`/`*_l_*` classes by name must switch. Cache files are unaffected (the cache signature never contained the class name).
- Migration guide ([!313](https://gitlab.com/open-darts/open-darts/-/merge_requests/313)) — OBL grid API. Per axis `i` (pressure, each composition, thermal):
  ```
  # axes_min[i], axes_max[i], n_points  ->  axes_origin[i], axes_step[i]
  axes_origin[i] = axes_min[i]                             # grid floor; composition axes use epsilon_z (not 0)
  axes_step[i]   = (axes_max[i] - axes_min[i]) / (n_points - 1)
  ```
  `axes_step`/`axes_origin` length = `1` pressure `+ (nc-1)` compositions `[+ 1` thermal `]`; for `extrapolation_flag=True` all composition steps must be equal. The formula above reproduces the previous grid; but since the grid is now unbounded, `axes_origin` is only the index-0 anchor, so a better choice is to set its pressure and thermal entries to the model's **initial conditions** (keep the `epsilon_z` floor on composition axes) — the initial state then lands exactly on a grid node and the grid grows around the operating point. It does not default to the initial state (physics is built before the initial conditions are set); `axes_origin` otherwise falls back to a fixed unit floor (1 bar / `epsilon_z` / 273.15 K). Worked examples:\
  {- Before (isothermal, nc=3): Compositional(components, phases, timer, n_points=200, min_p=1, max_p=300, min_z=0., max_z=1., epsilon_z=eps, extrapolation_flag=True) -}\
  {+ Now:    Compositional(components, phases, timer, axes_step=[(300-1)/199, (1-3*eps)/199, (1-3*eps)/199], axes_origin=[1.0, eps, eps], epsilon_z=eps, extrapolation_flag=True) +}
  \
  {- Before (P-T thermal, nc=1): Compositional(..., n_points=400, min_p=0, max_p=1000, min_t=273.15, max_t=473.15, epsilon_z=eps, state_spec=...PT) -}\
  {+ Now:    Compositional(..., axes_step=[2.5, 0.5], axes_origin=[0.0, 273.15], epsilon_z=eps, state_spec=...PT) +}
  \
  {- Before (Geothermal P-H): Geothermal(timer, n_points=256, min_p=1, max_p=351, min_e=1000, max_e=10000) -}\
  {+ Now:    Geothermal(timer, axes_step=[(351-1)/255, (10000-1000)/255], axes_origin=[1.0, 1000.0]) +}
  \
  The Geothermal P-T `ThermalVarOperator` window (formerly the hardcoded `PT_axes_min`/`PT_axes_max`) is now the optional `thermal_var_axes_step` / `thermal_var_axes_origin` (defaults `[p_step, 1.0]` / `[p_origin, 273.15]`).
- Breaking changes ([!318](https://gitlab.com/open-darts/open-darts/-/merge_requests/318)):
  - **The `Geothermal` physics/engine is removed.** Single-component-water geothermal simulation (IAPWS-97, `[P, enthalpy]` state) is superseded by the compositional `PhysicsBase` engine driven by a DARTSFlash IAPWS-95 PT-flash. The primary unknowns change from `[P, enthalpy]` to `[P, temperature]`, so `engine.X` layout, the OBL grid axes (P-H → P-T), and any code reading the thermal variable change accordingly. The bundled former-geothermal models (`GeoRising`, `CoaxWell`, `cpg_sloping_fault`, `fracture_network`) were migrated; see the migration guide below.
  - Unused engines removed: all `engine_nc*` except `engine_nc_nl`.
  - `PropertyBase` was folded into `PropertyContainer`, and `operators_base.py` was merged into `operator_evaluator.py`. Import `OperatorsBase`, `WellCtrlOperators`, `ThermalVarOperator` and `PropertyOperators` from `darts.physics.base.operator_evaluator`; the `darts.physics.base.operators_base` and `darts.physics.base.property_base` modules no longer exist.
  - The built-in IAPWS-IF97 property evaluators are removed together with the `Geothermal` physics they served: the `darts.physics.properties.iapws` subpackage (`iapws_property.py`, `iapws_property_vec.py`, `custom_rock_property.py`) no longer exists. Migrated models take water/steam properties from the DARTSFlash IAPWS-95 EoS instead (`EoSDensity`/`EoSEnthalpy` on the `IAPWS` mixture); the dead `compute_temperature` helpers built on `_Backward1_T_Ph_vec` were dropped from the models. The external `iapws` pip dependency is kept — `models/chemistry/carbonated_water` still uses its viscosity correlation directly.
- Migration guide ([!318](https://gitlab.com/open-darts/open-darts/-/merge_requests/318)) — `Geothermal` → compositional `PhysicsBase` (`state_spec=PT`). A single-component-water geothermal model becomes a compositional model whose property evaluators are wired explicitly around an IAPWS-95 PT-flash:\
  {- Before: self.physics = Geothermal(idata, timer) -}\
  {+ Now:    self.physics = PhysicsBase(components, phases, timer, state_spec=PhysicsBase.StateSpecification.PT, axes_step=[p_step, t_step], axes_origin=[p_origin, t_origin], epsilon_z=eps) — with a hand-wired PropertyContainer (below) +}
  ```python
  from darts.physics.base.physics import PhysicsBase
  from darts.physics.base.property_container import PropertyContainer
  from dartsflash.mixtures import DARTSFlash, CompData, EoS, IAPWS
  from darts.physics.properties.eos_properties import EoSDensity, EoSEnthalpy
  from darts.physics.properties.basic import ConstFunc, PhaseRelPerm
  from darts.physics.properties.viscosity import MaoDuan2009

  components, phases, eps = ["H2O"], ["V", "L"], 1e-12       # 'V','L' (vapor, liquid) replace legacy 'steam','water'
  comp_data = CompData(components=components, setprops=True)
  pc = PropertyContainer(phases_name=phases, components_name=components, Mw=comp_data.Mw, eps_z=eps)

  flash = IAPWS(iapws_ideal=True, ice_phase=False)           # IAPWS-95 EoS
  flash.init_flash(flash_type=DARTSFlash.FlashType.PTFlash)
  pc.flash_ev = flash
  pc.density_ev      = {"V": EoSDensity(flash.eos["IAPWS"], comp_data.Mw, EoS.RootFlag.MAX),
                        "L": EoSDensity(flash.eos["IAPWS"], comp_data.Mw, EoS.RootFlag.MIN)}
  pc.enthalpy_ev     = {"V": EoSEnthalpy(flash.eos["IAPWS"], EoS.RootFlag.MAX),
                        "L": EoSEnthalpy(flash.eos["IAPWS"], EoS.RootFlag.MIN)}
  pc.viscosity_ev    = {"V": ConstFunc(0.01), "L": MaoDuan2009(components)}   # liquid µ must stay T/P-dependent
  pc.rel_perm_ev     = {"V": PhaseRelPerm("gas", swc=0.0), "L": PhaseRelPerm("oil", swc=0.0)}
  pc.conductivity_ev = {"V": ConstFunc(0.0), "L": ConstFunc(172.8)}           # kJ/m/day/K

  self.physics = PhysicsBase(components, phases, timer,
                             state_spec=PhysicsBase.StateSpecification.PT,
                             axes_step=[p_step, t_step], axes_origin=[p_origin, t_origin], epsilon_z=eps)
  self.physics.add_property_region(pc)
  ```
  Notes:
  - **State layout** changes `[P, enthalpy]` → `[P, temperature]`: update any `engine.X` slicing, and switch the OBL input fields `idata.obl.e_step`/`e_origin` (enthalpy axis) to `t_step`/`t_origin` (temperature axis).
  - **Keep the temperature `axes_origin` at ≥ `273.15` K** (the IAPWS liquid floor). With `ice_phase=False` the PT-flash returns NaN below it; because the OBL grid is unbounded it would otherwise sample the sub-freezing region and fail (singular CPR / timestep collapse).
  - **Liquid viscosity** must be `MaoDuan2009(components)` (T/P-dependent), not a constant — a constant `µ` rescales well rates by `µ_ref/µ_const` under BHP control.
  - IAPWS-97 → IAPWS-95 is a property-model change: well BHT/BHP shift by ≲ 0.2 %, so **regenerate reference solutions** for migrated models.
- Add hysteresis support for OBL-based compositional simulations through per-cell history variables, including Killough scanning-curve handling; the feature is disabled by default and enabled only when history variables are explicitly declared in the physics setup ([!310](https://gitlab.com/open-darts/open-darts/-/merge_requests/310)).
- Output:
  - output which was using `vtk` module, has been changed to use `meshio` (struct reservoir, cpg reservoir) and darts/tools/vtk_io.py (writing vtp files with dynamic results along well trajectories)
- Package:
  - removed `vtk` dependency ([!314](https://gitlab.com/open-darts/open-darts/-/merge_requests/314))
  - added "viz" option to install `vtk` and `pyvista`; added "all" option to install "viz" and "solvers" groups. Usage pip install open-darts[viz].
- Switched to Python 3.11 by default (CI/CD pipelines, ReadTheDocs build, `ruff` lint target, and the recommended developer environment); Python 3.10–3.13 remain supported and tested.
- OBL, interpolation and supporting-point cache ([!313](https://gitlab.com/open-darts/open-darts/-/merge_requests/313)):
  - The adaptive OBL interpolators are now **unbounded**: hypercubes are keyed on a signed multi-index instead of a packed integer bounded by `(axes_min, axes_max)`, so the grid is defined only by a per-axis origin and step and grows on demand wherever the solver lands. Out-of-window queries return bit-exact linear extrapolation with no clamping. The engines correspondingly drop OBL-window state clipping; Newton now clips only to the physical simplex `[0, 1] ± sim_eps`. (Requires recompiling the C++/pybind interpolators and engines.)
  - Rewrote the boundary-extrapolation support-point selection for robustness (rank-revealing modified-Gram–Schmidt selection with an `np.linalg.lstsq` fallback and a warning on singular supports, instead of silently least-squaring a singular system). `OperatorsBase.dz` may now be a scalar or a per-axis vector (stored as a NumPy array); uniform grids reproduce prior results exactly.
  - New self-contained on-disk OBL cache format (`DRTSFC03`): a single page-aligned, memory-mapped open-addressing hash arena plus trailing delta/epoch frames, replacing the pickled supporting-point dictionary. Cold caches load in O(1) with no ~1e9-entry rebuild, and the resident map is file-backed / demand-paged instead of a multi-hundred-GB anonymous-RAM image. The `obl_point_data_<md5>.pkl` filename is kept for cache discovery; old pickles still load (legacy integer-keyed adaptive caches are detected, ignored and regenerated) and are rewritten as `DRTSFC03` on the next flush. Cross-platform: Linux/macOS via POSIX `mmap` and Windows via Win32 file mapping — the arena path is selected on interpolator capability, never on the OS. Set `OBL_CACHE_BUILD_MMAP=1` to force the memory-bounded arena builder.
  - Bounded the in-memory OBL hypercube cache to cap peak RAM on large adaptive runs (`self.physics.hypercube_cap`; LRU on CPU, clear-on-overflow on GPU; `0`/unset keeps the legacy unbounded behaviour). Together with the mmap arena this removes the two dominant OBL RAM terms that could OOM large chemistry runs.
  - Record the evaluation epoch (batch/Newton index) at which each supporting point was first materialized; readable offline via `PhysicsBase.load_point_epochs()` for OBL-space-growth analysis.
  - Added a per-axis `int32` cell-index overflow guard with throttled reporting and saturation in both Debug and Release builds; the check is branch-light and keeps aggregation off the interpolation hot path.
- Parallel operator evaluation and output ([!313](https://gitlab.com/open-darts/open-darts/-/merge_requests/313)):
  - Extended parallel operator evaluation from the reservoir operators only to all default physics evaluators (reservoir / property / well / well-control / thermal-var) through a single fixed-size `SharedEvaluatorPool`, so the total worker-process count stays at `n_workers` regardless of how many evaluators are wrapped.
  - Output property interpolators are built and evaluated separately from the simulation property operators (`output_property_operators` / `output_property_itor`), so requesting extra output properties no longer overwrites the operators the simulation uses; output batches can run on the shared pool.
- Chemistry ([!313](https://gitlab.com/open-darts/open-darts/-/merge_requests/313)):
  - The Reaktoro flash reuses a persistent solver/options/conditions object and warm-starts each OBL point from the previous converged speciation (falling back to a cold solve on failure), materially speeding up Reaktoro-driven flash at identical results.
  - Reaktoro kinetic saturation ratios are resolved by name to the stable carbonate phases (`Calcite` / `Dolomite` / `Magnesite`) instead of the first formula match (which picked `Aragonite` / ordered dolomite in `supcrtbl`). This is a correctness fix and **changes computed SR values (and hence kinetic rates and results)** for reactive carbonate models; pass `mineral_sr_species={formula: species_name}` to override the mapping.
- Build system ([!313](https://gitlab.com/open-darts/open-darts/-/merge_requests/313)):
  - New CMake variable `OPENDARTS_MAX_DIMS` (default `8`) drives the interpolator `MAX_DIMS`, the engine `MAX_NC`, and the linear-solver block-size instantiation range from a single knob; set `-D OPENDARTS_MAX_DIMS=<N>` to build isothermal cases with up to `N` components, or thermal cases with up to `N - 1` components because the thermal state adds one interpolation axis. `MAX_NC` is no longer silently defaulted; compiling an engine translation unit without `-DMAX_NC` is now a hard error.
  - New CMake variable `OPENDARTS_INTERPOLATOR_PROFILE` (`MINIMAL` | `FULL`, default `FULL`) to cut per-translation-unit compiler memory. CI and the shipped wheels build `MINIMAL`, which compiles only the multilinear adaptive interpolator (see the breaking-changes note about `algorithm='linear'`).
  - CI test jobs install the built package with the `test` extra instead of installing `pytest` separately; `helper_scripts/build_darts_cmake.{sh,bat}` and the GPU wrapper honor `-t` by installing `open-darts[test]`.
  - Linear-solver explicit template instantiations (`superlu`, `bos_cpr`, `bos_bilu0`, `bos_gmres`) are generated from `OPENDARTS_MAX_DIMS` (with a floor at block size 13) instead of hand-written lists.
  - Operator indices were widened from `uint8_t` to `uint16_t` across the engines and interpolators, raising the supported operator/component count (e.g. up to ~272 operators for 30 components, 3 phases, thermal). Geomechanics / super-elastic engines are capped at `NC <= 3` (`MAX_NC_MECH=3`), independent of `OPENDARTS_MAX_DIMS`.
- Tests ([!313](https://gitlab.com/open-darts/open-darts/-/merge_requests/313)):
  - Added coverage for unbounded adaptive CPU interpolation, axes-step-only construction, cache round-trips, GPU smoke tests, convergence / linearity preservation, parallel evaluator consistency, mmap cache arena behaviour, and explicit hash-collision handling.
- Models, tools and diagnostics ([!313](https://gitlab.com/open-darts/open-darts/-/merge_requests/313)):
  - `DartsModel` verbosity is now a single integer level (`self.verbose`, 0–3: silent / default / +timers / +per-worker evaluator output); booleans are still accepted. Level >= 2 prints the timer breakdown to the DARTS log after every `run()`.
  - `models/chemistry/carbonated_water` is now a working reactive-transport example (batched OBL initialization through the parallel pool, cross-run persistent good-step counter, PHREEQC dilution-fallback Newton budget).
  - Fixed two GPU teardown crashes triggered by a partially-initialized model (uninitialized device pointers passed to `cudaFree`; `PhysicsBase.__del__` raising when `__init__` failed before `self.cache` was set).

# 1.5.0 [27-05-2026]
- Fluid heat capacity is added into the input data for THM models ([!270](https://gitlab.com/open-darts/open-darts/-/merge_requests/270))
- Support using the OBL method to calculate DFM well phase velocities. Direct method is still the default method since it is safer in terms of stability ([!287](https://gitlab.com/open-darts/open-darts/-/merge_requests/287))
- Add `x_mass` (mass composition of each phase) as a new property to `PropertyContainer` of the super engine because it is needed for evaluation of phase velocities in DFM wells using the OBL method ([!287](https://gitlab.com/open-darts/open-darts/-/merge_requests/287))
- Make DFM velocity calculation independent of the order of the phases specified by the user, so now the order of the phases does not affect the performance of DFM wells, but the user needs to specify `"G"` and `"L"` as names of gas and liquid phases for two-phase flow and `"G"`, `"L_a"`, and `"L_b"` as names of gas and two liquid phases for three-phase flow ([!287](https://gitlab.com/open-darts/open-darts/-/merge_requests/287)).
- Correct derivative of averaged density of the liquid phase for three-phase flow of gas and two liquid phases ([!287](https://gitlab.com/open-darts/open-darts/-/merge_requests/287))
- Improve storage and visualization of properties of DFM wells ([!287](https://gitlab.com/open-darts/open-darts/-/merge_requests/287)):
  - Streamline storage and visualization of DFM well properties
  - Support storage and visualization of multiple DFM wells
  - Support storage of DFM well output in `.vtp` files to be visualized in ParaView
- Support live plotting ([!287](https://gitlab.com/open-darts/open-darts/-/merge_requests/287)):
  - Live (real-time) plots for solver properties (time step size and number of Newton iterations) and tracking the state of a block on the PH diagram
  - Live (real-time) plots for profiles of DFM well properties
  - Save live-plot snapshots and monitor a reservoir block
- Align depth of perforated well segments with reservoir blocks ([!287](https://gitlab.com/open-darts/open-darts/-/merge_requests/287))
- Fix BHT calculation for PH formulation in the method `store_bhp_bht` in `output.py` ([!287](https://gitlab.com/open-darts/open-darts/-/merge_requests/287))
- Store the arrays `time`, `n_newton_iters`, and `time_step_size` in the class `DartsModel` ([!287](https://gitlab.com/open-darts/open-darts/-/merge_requests/287))
- Implement `engine_base::apply_thermal_var_correction` to improve the issue related to sharp enthalpy updates from the Newton-Raphson solver for the pressure-enthalpy (PH) formulation ([!289](https://gitlab.com/open-darts/open-darts/-/merge_requests/289))
- Support well controls (rate and WHP) for DFM wells consistent with EPM wells. WHP is controlled for DFM wells and BHP is controlled for EPM wells ([!292](https://gitlab.com/open-darts/open-darts/-/merge_requests/292))
- Support total (mass, molar, volumetric, and advective heat) rate control for both EPM and DFM wells. If well rate is controlled and phase is not specified, total rate will be applied ([!292](https://gitlab.com/open-darts/open-darts/-/merge_requests/292))
- Fix the issue in the derivative of wellhead equation for rate control of EPM wells ([!292](https://gitlab.com/open-darts/open-darts/-/merge_requests/292))
- Fix bugs when using DFM wells for three-phase (gas + two liquid phases) fluid flow ([!292](https://gitlab.com/open-darts/open-darts/-/merge_requests/292))
- Enable pipe flow calculations in DARTS-well for systems containing immobile phases ([!292](https://gitlab.com/open-darts/open-darts/-/merge_requests/292))
- Enable line search in `darts_model.py` to work with models containing DFM wells ([!292](https://gitlab.com/open-darts/open-darts/-/merge_requests/292))
- Fix output perforation rates when `ms_epm` is `False` by considering the gravity component in `output.py` + store `mesh.grav_coef` in the h5 file + expose `mesh.grav_ceof` to Python ([!292](https://gitlab.com/open-darts/open-darts/-/merge_requests/292))
- Unstructured reservoir [!298](https://gitlab.com/open-darts/open-darts/-/merge_requests/298):
	- fixed the order in store_depth_all_cells (could affect the initialization by gradient)
	- vtk output is fixed for 3D meshes (order)
	- separate vtk files for matrix and fracture data
	- reservoir cache is fixed
- Improve CI model tests by comparing generated well time-series reference files (`well_time_data.pkl`) and by extending performance-reference checks to include well primary variables in addition to reservoir primary variables ([!312](https://gitlab.com/open-darts/open-darts/-/merge_requests/312)).
- Extracted interpolators into a standalone `darts.interpolators` Python module / shared library, decoupled from `darts.engines` at link time (header-only coupling via `interpolation_config.h`). Template instantiations split across multiple translation units to enable parallel compilation and cut per-TU memory (full build down to ~6 min on multi-core; valgrind job pre-builds at `-j NT/2` to avoid OOM). Interpolator tests moved to `tests/interpolators/`. Breaking change: interpolator types are no longer exposed under `darts.engines` — import from `darts.interpolators` ([!301](https://gitlab.com/open-darts/open-darts/-/merge_requests/301))
- Parallel operator update ([!297](https://gitlab.com/open-darts/open-darts/-/merge_requests/297)): adaptive interpolators rewritten as a three-phase OpenMP update (discover / materialize / interpolate) governed by `OMP_NUM_THREADS`; new `evaluate_batch` interface and `ParallelEvaluator` that evaluates missing supporting points across a multiprocessing pool. Enable per model with `init(parallel_evaluation=True, n_workers=...)`; the default `DartsModel.get_evaluator_factory` (`ModelEvaluatorFactory`) needs no per-model code and works under both `fork` and `spawn`. `Chem_benchmark_new` runs on the parallel path in CI. See `docs/for_developers/parallel_operators.md`.
- Physics / Wells:
  - Generalize the potential-energy contribution in the energy equation to multi-component systems
  - Add `Pipe` options to enable/disable the profile parameter and drift velocity in the DFM closure
  - Unify EPM and DFM well-control operators into a single `WellCtrlOperators` class
- Examples / Models:
  - Add DFM-well validation scenarios against OLGA: `1ph_1comp_thermal_dfm_well_vs_olga`, `2ph_2comp_isothermal_dfm_vertical_well_vs_olga`, `2ph_2comp_isothermal_dfm_inclined_well_vs_olga`
  - Add `2ph_dead_oil_coupled_well_reservoir` and a constant-rate gaseous-phase injection example
  - Add a live-plotting example to the coupled well–reservoir model
- Dependencies: bump minimum `open-darts-flash` to `>=0.12.1`
- Breaking changes:
  - Rock thermal conductivity was renamed in the input data for geomechanical models:
  \
  {- Before: idata.rock.conductivity -}\
  {+ Now:    idata.rock.thermal_conductivity +}
  \
  - Equilibrium initialization function name was changed from version 1.3.2:
  \
  {- Before: init.solve() -}\
  {+ Now:    init.solve_up_and_downwards() +}
  \
  - Geothermal `PropertyContainer` field `saturation` renamed to `sat` to match the super-engine container:
  \
  {- Before: property_container.saturation -}\
  {+ Now:    property_container.sat +}
  \
  - In well outputs, `saturation` renamed to `volume_fraction`; well-side fields now use the same `pressure` and `temperature` names as the reservoir side
  - DFM example scenarios were renamed:
  \
  {- Before: two_phase_isothermal_dfm_well_flow, single_phase_thermal_dfm_well_flow, coupled_dfm_well_reservoir -}\
  {+ Now:    2ph_2comp_isothermal_dfm_vertical_well_vs_dwell, 1ph_1comp_thermal_dfm_well_vs_dwell, 2ph_1comp_coupled_dfm_well_reservoir +}
  \

# 1.4.0 [17-02-2026]
- OBL and operators:
  - Extrapolation of operators at supporting points with negative last compositions for consistent interpolation in hypercubes at the edge of the compositional domain - current logic works only for equal compositional axes across all dimensions. ([!204](https://gitlab.com/open-darts/open-darts/-/merge_requests/204))
  - Consistent composition bounds using min_z/max_z (e.g., 0 to 1), epsilon (for min_axis_z/max_axis_z: eps_z, 1-(nc-1)*eps_z) and sim_eps (min_axis_z + sim_eps, max_axis_z - sim_eps) - current logic only fully verified with equal compositional axes across all dimensions. ([!204](https://gitlab.com/open-darts/open-darts/-/merge_requests/204))
  - Change operators by splitting density out of `GRAD_OP` and introducing new `DENS_OP`. ([!237](https://gitlab.com/open-darts/open-darts/-/merge_requests/237))
  - Change operators by splitting `FLUX_OP` operator into two operators and introducing `SAT_OP`. ([!234](https://gitlab.com/open-darts/open-darts/-/merge_requests/234))
  - More robust OBL cache saving using atomic writes. [!238](https://gitlab.com/open-darts/open-darts/-/merge_requests/238)
  - Always use `WellOperators` for wells (in the past, `ReservoirOperators` was used for wells for thermal scenarios) ([!284](https://gitlab.com/open-darts/open-darts/-/merge_requests/284)).

- Physics:
  - Added the Drift-Flux Model (DFM) as a new multi-segment well model. ([!230](https://gitlab.com/open-darts/open-darts/-/merge_requests/230))
  - Added potential energy to the energy conservation equation. This feature is off by default ([!246](https://gitlab.com/open-darts/open-darts/-/merge_requests/246), [!263](https://gitlab.com/open-darts/open-darts/-/merge_requests/263))
  - Chemistry: Built-in interfaces to third-party geochemical flashes (PHREEQC and Reaktoro), databases (`phreeqc.dat`, `pitzer.dat`, `supcrtbl.dat`) and reaction kinetics model (`KineticRate`) for carbonate minerals (`PalandriKharaka.json`). New Element-based physics for reactive flow and transport in `ElementBasedReactiveFlow` supporting built-in thirdparty solvers and databases. [!238](https://gitlab.com/open-darts/open-darts/-/merge_requests/238)

- Reservoirs:
  - Add StructRadialReservoir and UnstructRadialReservoir classes, derived from (Un/)StructuredReservoir classes. Implementation in darts/models/ccs and darts/models/dfm_well ([!169](https://gitlab.com/open-darts/open-darts/-/merge_requests/169))
  - Easy setting of rock properties by regions is [implemented](https://gitlab.com/open-darts/open-darts/-/merge_requests/251) for CPG reservoir (ROCKNUM)

- Output:
  - Cell centroids are now included under static variables in the output file `reservoir_solution.h5`. ([!282](https://gitlab.com/open-darts/open-darts/-/merge_requests/282))
  - Reduced runtime by saving well output after DartsModel.run(). ([!228](https://gitlab.com/open-darts/open-darts/-/merge_requests/228))
  - Reduced well output evaluation time by using vectorized interpolators. In the past two for loops were used over time steps + over connection ids (wellheads and perforations). Now, they are removed. In addition, all operators now are evaluated using interpolators.
  - Added Output.output_to_plt() method for StructReservoir classes using xarray interface. ([!169](https://gitlab.com/open-darts/open-darts/-/merge_requests/169))
  - `CFL_max` is added to the H5 well output. [!238](https://gitlab.com/open-darts/open-darts/-/merge_requests/238)
  - Strain rate and additional output to vtk for poroelastic model: [link](https://gitlab.com/open-darts/open-darts/-/merge_requests/274)
  - Custom arrays output support for [fracture data](https://gitlab.com/open-darts/open-darts/-/merge_requests/241)

- Solvers:
  - An option to use PARDISO linear solver is [added](https://gitlab.com/open-darts/open-darts/-/merge_requests/247)
  - An optional use of PETSc linear solver for Geothermal and Poromechanical physics is [added](https://gitlab.com/open-darts/open-darts/-/merge_requests/235)

- Models:
  - Added a simple example of a thermal model with foam (models/3ph_comp_w_foam)
  - fracture_network model: 3D meshes are added, perforation dpeth range is supported, supported meshes without fractures [link](https://gitlab.com/open-darts/open-darts/-/merge_requests/271); related [fix](https://gitlab.com/open-darts/open-darts/-/merge_requests/243)

- Tools:
  - added a [script](https://gitlab.com/open-darts/open-darts/-/merge_requests/271) to convert meshes from paraview format to gmsh format

- Build system and CI/CD:
  - Python 3.13 is [supported](https://gitlab.com/open-darts/open-darts/-/merge_requests/261) and Python 3.9 support is [no longer supported](https://gitlab.com/open-darts/open-darts/-/merge_requests/248)
  - Switched to ubuntu2018 docker image and conda environments in the [pipelines](https://gitlab.com/open-darts/open-darts/-/merge_requests/267)
  - Support -e --with-deps -j arguments in installation scripts. [!238](https://gitlab.com/open-darts/open-darts/-/merge_requests/238)
  - CI/CD jobs moved from `helper_scripts/ci_jobs` to `.cicd/jobs`, splitted for platforms, improved job rules
  - Added pre-commit/linting in the pipelines, switched to ruff-based formatting [link](https://gitlab.com/open-darts/open-darts/-/merge_requests/233) and added [gitingest](https://gitlab.com/open-darts/open-darts/-/merge_requests/239)
  - Added [Valgrind](https://gitlab.com/open-darts/open-darts/-/merge_requests/201) checks in the pipelines
  - Added an additional job for GPU Linux platform based on an apptainer [image](https://gitlab.com/open-darts/open-darts/-/merge_requests/232)

- Breaking changes:
  - Input arguments to facilitate consistent compositional axes and extrapolation:
  \
  {- Before: Compositional(..., min_z=zero/10, max_z=1-zero/10) -}\
  {+ Now:    Compositional(..., min_z=0, max_z=1, epsilon_z=zero/10, sim_eps_multiplier=10, extrapolation_flag=True) +}
  \
  {- Before: PropertyContainer(..., min_z=zero) -}\
  {+ Now:    PropertyContainer(..., eps_z=epsilon) +}
  \
  {- Before: OperatorsBase(...) -}\
  {+ Now:    OperatorsBase(..., extrapolation_flag=True, dz: float) +}
  - Rename an input argument of the method `add_well`:\
  {- Before: self.reservoir.add_well(..., wellbore_diameter) -}\
  {+ Now:    self.reservoir.add_well(..., well_diameter) +}
  - Rename input arguments of the method `add_perforation`:\
  {- Before: self.reservoir.add_perforation(..., cell_index, well_radius, multi_segment) -}\
  {+ Now:    self.reservoir.add_perforation(..., res_cell_idx, well_diameter, ms_epm) +}
  - Plotting methods for structured data `plot_xarray()` has become `output_to_plt()` and has options to plot from solution file, xarray dataset or engine.X:\
  {- Before: self.output.plot_xarray(xarray_data, output_properties, timestep, ...) -}\
  {+ Now:    self.output.output_to_plt(sol_filepath, xarray_data, output_properties, timestep, ) +}

# 1.3.2 [03-07-2025]
- Porosity-permeability relationship: permporo_mult_ev
  - Introduce permporo_mult_ev, turn PORO_OP into MULT_OP, use harmonic mean for facial averaging of multiplier
- EoSDensity and EoSEnthalpy API changes:
  - Update darts-flash enthalpy call in EoSEnthalpy class: eos.H_PT() to eos.H()
  - Add optional arguments to pass ions and lumped ions to EoSDensity and EoSEnthalpy constructors
- Contact mechanics generalization:
  - Fixed logic in displaced_fault model: call self.reservoir_depletion() anyway since it is needed to check hash for objects for example porosity remove 'self.unstr_discr', 'self.pm' from checking hash as they are not available at check hash stage fix hash comparison (it was not comparing hash of the current data)
  - added run_tests() to be able to run tests within a displaced_fault model, locally.
  - Code refactoring (continues [link](https://gitlab.com/open-darts/open-darts/-/merge_requests/188)):
     1. get_normal_to_bound_face, get_parametrized_fault_props, write_fault_props are moved to the base class
     2. code piece is put into a function 'init_fractures' of the base class
- New feauture: pass property array dictionaries directly to output_to_vtk()

# 1.3.1 [10-06-2025]
- Add IPhreeqc to thirdparty dependencies, added phreeqc_dissolution model with CO2 injection and dissolution-precipitation kinetics [link](https://gitlab.com/open-darts/open-darts/-/merge_requests/173)
- Apply_rhs_flux supported for GPU platform [link](https://gitlab.com/open-darts/open-darts/-/merge_requests/173)
- Added .pvd file generation for unstructured meshes for Time in ParaView [link](https://gitlab.com/open-darts/open-darts/-/merge_requests/216)
- Reconstruct Darcy velocity for water and steam phases in Geothermal engine [link](https://gitlab.com/open-darts/open-darts/-/merge_requests/209)
- Switched to Python 3.10 by default in CI/CD pipelines
- well_indexD set to 0 by default in cpg_model [link](https://gitlab.com/open-darts/open-darts/-/merge_requests/213)

# 1.3.0 [16-05-2025]
- Unify well controls:
  - A single C++ class well_control_iface class has been generalized to handle bhp and different rate controls (inj/prod) for all engines
  - An instance of well_control_iface is created for every ms_well object upon init_rate_parameters()
  - Well controls and constraints are accessed through setters, with the aid of the PhysicsBase.set_well_controls() method
  - The set of WellControlOperators calculates BHP, BHT and each phase's volumetric, mass and molar rates
  - An enum of class well_control_iface specifies the control/constraint type: -1) BHP, 0) MOLAR_RATE, 1) MASS_RATE, 2) VOLUMETRIC_RATE, 3) ADVECTIVE_HEAT_RATE; default is BHP
  - API of set_well_controls(): control_type: well_control_iface.WellControlType, is_inj: bool, target: float,
                                phase_name: str, inj_composition: list, inj_temp: float, is_control: bool
  - A WellInitOperator translates PT-based controls to the given state specification (PT, PH, ...)
- Unify set_initial_conditions():
  - 1) Uniform or array -> specify constant or array of values for each variable to self.physics.set_initial_conditions_from_array()
  - 2) Depth table -> specify depth table with depths and initial distributions of unknowns over depth to self.physics.set_initial_conditions_from_depth_table()
  - DartsModel.set_uniform_conditions(): forces user to overload this method in Model() and set initial conditions according to 1) or 2)
  - PhysicsBase.set_initial_conditions_from_array() to set initial conditions uniformly/with array
  - PhysicsBase.set_initial_conditions_from_depth_table() to interpolate/calculate properties based on depth table
  - On the C++ level, there is an array initial_state that lives in the conn_mesh class, which is to be filled with the initial state of all the primary variables in PhysicsBase.set_initial_conditions_from_*()
  - The method set_initial_conditions() initializes the mesh for **RESERVOIR BLOCKS ONLY**. Well cell initialization is done inside the engine
- PhysicsBase class constructor contains StateSpecification enum to define state variables
  - Unifies P (isothermal), PT (pressure-temperature) and PH (pressure-enthalpy)
  - Compositional constructor contains state_spec variable rather than thermal (bool)
  - Geothermal is PH by default (as before)
  - All engines have a WellInitOperator to translate PT-based well controls to the given state specification
- DartsModel.output() class now contains all output related functions of open-DARTS. Please refer to this [example](https://gitlab.com/open-darts/open-darts/-/blob/main/tutorials/output_and_restart.py?ref_type=heads) as a reference.
  - The output folder is defined in DartsModel.set_output(output_folder=output) instead of in DartsModel.init()
  - The default name for save file 'solution.h5' is changed to 'reservoir_solution.h5' and now only contains reservoir blocks instead of the entire solution vector.
  - DartsModel.output.output_properties()'s new default behavior is to return primary variables if the properties list is not defined. If defined, only the listed primary (states) and/or secondary (properties) variables are returned as a dictionary.
  - DartsModel.output.store_well_time_data() is added to store well time data. The reported well time data include BHP, BHT, phases molar rates, phases mass rates, phases volumetric rates, components molar rates, components mass rates, and phases heat
  rates over time. The rates are reported for each perforation as well. Total rates for each well are reported in two different ways: summation of perforations rates and rates calculated at wellhead. New naming scheme of the new well time data is explained in [documentation](https://gitlab.com/open-darts/open-darts/-/blob/development/docs/technical_reference/well_time_data_guide.md?ref_type=heads).
  - DartsModel.output.plot_well_time_data() is added for saving the figures of well time data.
  - C++ rate calculations will be replaced in the near future with python based rate calculations in the DartsModel.output(). Currently, both are available. However, it is recomended that you start transitioning to the new python rates.
- Save data to *.hdf5 with compression and/or single precision in order to manage file size.
- Evaluate properties from engine or saved data.
- New option: DartsModel.set_output(all_phase_props=True), creates an extensive list of phase properties in the DartsModel.physics.
- Build system:
	- The GPU configuration is supported in cmake. [Details](https://gitlab.com/open-darts/open-darts/-/merge_requests/179)
	- C++ standard was changed to C++20. [Details](https://gitlab.com/open-darts/open-darts/-/merge_requests/182)
	  - a custom dockerfile usage with gcc-13 compiler for open-DARTS compilation in gitlab pipelines (Linux only).
	  - libstc++ added to the wheels (for Linux).
	  - DARTS command line interface (CLI) added.
	- The required cmake version changed to 3.26.
- Jacobian, Newton and Timestepping:
	- The BCSR jacobain matrix class exposed to Python. [Changes](https://gitlab.com/open-darts/open-darts/-/merge_requests/155/diffs?commit_id=766b876deb925ffeba1d6971402c93b80256b5a7)
	- Row-wise jacobian scaling for Geothermal engine. Turned off by default [Changes](https://gitlab.com/open-darts/open-darts/-/merge_requests/155/diffs?file=611eddccefa3651138d3f99d5c57b29072c40eed#611eddccefa3651138d3f99d5c57b29072c40eed_414_492)
	- Line search for Newton iterations. Turned off by default, can be enabled with model.params.line_search = True. [Changes](https://gitlab.com/open-darts/open-darts/-/merge_requests/155/diffs?file=4111fc0a9161b7a085b99b1cd0ac668b08efcff7#4111fc0a9161b7a085b99b1cd0ac668b08efcff7_585_613)
    - An optional timestep control by the variable change from the previous newton iteration [added](https://gitlab.com/open-darts/open-darts/-/merge_requests/197) with a new DataTS Python class grouping the simulation parameters
- Fluxes storage in arrays. Turned off by default, can be enabled by a call `engine.enable_flux_output()`. [Changes](https://gitlab.com/open-darts/open-darts/-/merge_requests/155/diffs?commit_id=23fd4fdfb9cbcbce7f0f72e0874c83bccae0be73)
- Unstructured mesh
  - A mesh generation script [added](https://gitlab.com/open-darts/open-darts/-/merge_requests/184)
  - An option to pass .geo file [added](https://gitlab.com/open-darts/open-darts/-/merge_requests/188/diffs?commit_id=009d0f65538d905c31ea23184ed0a33c930bed97), and .msh file will be generated using gmsh command line
- Geomechanics:
	- Convergence tests for geomechanical models [added](https://gitlab.com/open-darts/open-darts/-/merge_requests/184)
	- Displaced fault reactivation model [added] (https://gitlab.com/open-darts/open-darts/-/merge_requests/188)
	- pvd file creation (with Time in addition to Timestep index)[added] (https://gitlab.com/open-darts/open-darts/-/merge_requests/188)
	- SPE10_mech model is enabled in pipelines.
- CPG Reservoir:
  - Fault transmissibility multiplier application fixed.
  - A test case with a fault transmissibility multiplacator [added](https://gitlab.com/open-darts/open-darts/-/blob/development/models/cpg_sloping_fault/main.py?ref_type=heads#L239).
  - The heat capacity and rock conduction storage to vtk file in the cpg model fixed.
  - Description [added](https://gitlab.com/open-darts/open-darts/-/blob/development/models/cpg_sloping_fault/README.md?ref_type=heads)
  - A case with inactive cells [added] (https://gitlab.com/open-darts/open-darts/-/merge_requests/206) to tests
  - Reading permeability when PERMX is defined and PERMY is not defined, [fixed](https://gitlab.com/open-darts/open-darts/-/merge_requests/206)
- CPG and Struct reservoirs:
  - Fixed a check for perforations in the inactive block.
  - Save the \*.pvd file in the associated output directory, together with \*.vtk/\*.vtu files [Changes](https://gitlab.com/open-darts/open-darts/-/merge_requests/200)
- Struct Reservoir:
	- No-flow boundary conditions in reconstruction of velocities on structured grid. [Changes](https://gitlab.com/open-darts/open-darts/-/merge_requests/155/diffs?commit_id=cbdd7d97937c1c44b4ed5c947f46357dc72fa2cf)
	- Enable the visualization of large volume from mesh.vts for StructReservoir [added](https://gitlab.com/open-darts/open-darts/-/merge_requests/198)
- Mesh processing for DFN model with the Python discretizer [optimized](https://gitlab.com/open-darts/open-darts/-/merge_requests/194).
- Breaking changes:\
	- For open-DARTS wheels installed from pip or gitlab pipelines running command was changed. This is neccessary to ensure the libstdc++ provided with open-DARTS is loaded first. This doesn't impact Windows runs. However, DARTS command line interface (CLI) can also be used on Windows. [Details](https://gitlab.com/open-darts/open-darts/-/merge_requests/182).\
	{- Before, python main.py -}\
	{+ Now, darts main.py +}
	- Physics (P, PT, PH):\
	{- Before: self.physics = Compositional(..., thermal=True/False) -}\
	{+ Now:    state_spec = Compositional.StateSpecification.P/PT/PH +}\
	{+ Compositional(..., state_spec=state_spec) +}
	- Initialization:\
	{- Before: self.initial_values = {self.physics.vars[0]: 50, self.physics.vars[1]: 0.1} -}\
	{+ Now:    def set_initial_conditions(self): +}\
	{+ input_distribution = {self.physics.vars[0]: 50, self.physics.vars[1]: 0.1 +}\
    {+ self.physics.set_initial_conditions_from_array(mesh=self.reservoir.mesh, input_distribution=input_distribution) +}
	- Well controls:\
	{- Before: well.control = self.physics.new_bhp_prod(50) -}\
	{+ Now:    BHP: self.physics.set_well_controls(wctrl=well.control, control_type=well_control_iface.BHP, is_inj=False, target=50) +}\
	{- Before: well.control = self.physics.new_rate_inj(100, inj_stream, phase_index) -} \
	{+ Now:    MOLAR_RATE, MASS_RATE, VOLUMETRIC_RATE: self.physics.set_well_controls(wctrl=well.control, control_type=well_control_iface.MOLAR_RATE, is_inj=False, target=100, phase_name="phase_name", inj_composition=inj_comp, inj_temp=inj_temp) +}
	The `inj_stream` is split into `inj_comp` and `inj_temp` for thermal models. The integer parameter `phase_index` was replaced with the string `phase_name` which is the same as specified in physics.
	To keep the consistency with the previous model, for Geothermal physics `VOLUMETRIC_RATE` should be used, and for Compositional MOLAR_RATE should be used.
	Each of the different control types can also be uniformly applied to both Geothermal and Compositional models.

	- Output (hdf5 and well rates):\
	{- Before: m.init(output_folder=...) -}\
			   {- m.run(100)-}\
			   {- time_data_df = pd.DataFrame.from_dict(n.physics.engine.time_data)) -}\
	{+ Now:    m.init() +}\
			   {+ m.set_output(output_folder=...) +}\
			   {+ m.run(100)+}\
			   {+ time_data_dict = m.output.store_well_time_data()+}\
               {+ time_data_df = pd.DataFrame.from_dict(time_data_dict)+}\
			   {+ m.output.plot_well_time_data(phase_volumetric_rates=True)+}\

		{- Before: m.save_data_to_h5('solution') -}\
		{+ Now: m.output.save_data_to_h5('reservoir') +}
	- Timesteps: any model should have a `set_sim_params(...)` call after physics initialization and before `model.init()`, where a DataTS instance is created.

# 1.2.2 [27-01-2025]
- Remove rock thermal operators; linear rock compressibility is ignored in rock thermal terms
- Temperature and pressure operators are added
- Add enthalpy operators to elastic super engine
- Fix in operator indexing in mechanical models
- GPU-version tests added in CI/CD
- Keep only rates, super, pze_gra and pz_cap_gra interpolators
- Breaking changes (only if InputData was used in a model):
	- The well class is modified and renamed to WellData which now contains a dictionary of Well objects
	- WellControlsConst is replaced by more generic 'WellControl's list from InputData

# 1.2.1 [25-11-2024]
- Remove "IAPWS" suffix from Geothermal PropertyContainer and predefined GeothermalIAPWS physics for backward compatibility.
- Few small changes:
	- CI/CD job updates
 	- empty list of output properties if none have been specified
	- remove SolidFlash class
	- fix to FluidFlower mesh generation
	- Mesh tags and boundary conditions are added to the input_data

# 1.2.0 [15-11-2024]
- Python 3.7-3.8 support dropped.
- Changes to physics/engines:
	- Correct approximation of diffusion fluxes (including energy)
	- Simple mechanical dispersion term
	- Predefined physics Geothermal, GeothermalPH, DeadOil and BlackOil
	- Phase velocities in structured reservoir
- Changes to OBL interpolation:
	- Barycentric linear interpolation with Delaunay triangulation
	- Support of higher-dimensional linear interpolation (requires re-compilation)
	- Tests
- Changes to linear solvers:
	- Hypre added as a submodule
	- FS_CPR preconditioner by default for mechanical tests (in configuration with iter. solvers)
	- Tests
- Changes to CPG model:
	- well perforations definition by X,Y,Z coordinates in addition to I,J,K indices
	- setting rock thermal properties based on the porosity
	- output wells and center points to separate vtk files
	- added extracted energy plot
	- added 2 options for physics: geothermal and dead oil

# 1.1.4 [07-09-2024]
- Generalize super engine physics for solid formulation[(See details)](https://gitlab.com/open-darts/open-darts/-/merge_requests/112)
- Python 3.12 support
- Recover the well control and constraints in case of switching between BHP and rate control
- Fix local build script for Windows: don't turn off OpenMP if '-b' is specified

# 1.1.3 [13-07-2024]
- Support for *.h5
	- Save well states of well blocks to 'output/well_data.h5': [save well data](https://gitlab.com/open-darts/open-darts/-/blob/v1.1.3/darts/models/darts_model.py#L450)
	- Save states of all reservoir blocks to 'output/solution.h5': [save solution data](https://gitlab.com/open-darts/open-darts/-/blob/v1.1.3/darts/models/darts_model.py#L463)
- Breaking changes:\
	{- Before, input states in the `DartsModel output_properties()` function were read from engine.X}\
	{+ Now, input states in the `DartsModel output_properties()` are read from a user specified time-step in 'solution.h5'}\
	{- Before, the properties_list in the `DartsModel output_properties()` function contained variable names and property names'}\
	{+ Now, the 'properties_list' contains only the property names}\
    {- Before `DartsModel output_to_vtk()` function returned a single *.vtk file}\
    {+ Now, by default `DartsModel output_to_vtk()` returns a *.vtk file for every timestep contained in 'solution.h5'}
- Two examples of models that use the new output_functions: [example](https://gitlab.com/open-darts/darts-models/-/tree/development/publications/24_geothermal_chapter/basic_1d.py) 1, [example](https://gitlab.com/open-darts/darts-models/-/tree/development/publications/24_geothermal_chapter/basic_3d.py) 2
- Molar and phase volumetric well rate calculators.
- Support of restarts.
- Molar and phase volumetric well rate calculators.
- Support of restarts.


# 1.1.2 [12-06-2024]
- Thermo-hydro-mechanical-compositional (THMC) modeling:
 	- Coupled Multi-Point Stress and Multi-Point Flux Approximations
 	- Fully implicit thermo-poroelasticity resolved with collocated FVM and coupled with compositional multiphase transport
	- Tests to compare to Mandel, Terzaghi, two-layer Terzaghi and Bai analytics [(link))](https://gitlab.com/open-darts/open-darts/-/tree/development/models/1ph_1comp_poroelastic_analytics)
	- [Convergence test](https://gitlab.com/open-darts/open-darts/-/tree/development/models/1ph_1comp_poroelastic_convergence)
	- Interface to block-partitioned preconditioner
- Improved performance of discretization (C++)
- [InputData class](https://gitlab.com/open-darts/open-darts/-/blob/development/darts/input/input_data.py) added and used in THM tests
- C++ standard is changed from 14 to 20
- Discretizer binary type changed from shared to static library
- Enable linking to external library (iterative solvers) compiled in debug mode if compiling openDARTS in debug mode.
- Improve documentation on multi-thread version.
- Add `opmcpg` as a main dependency.

# 1.1.1 [15-03-2024]

# 1.1.0 [16-02-2024]
- Migrated to cmake build system [(See details)](https://gitlab.com/open-darts/open-darts/-/merge_requests/58). We kept the old Visual Studio projects, but they will be removed later.
- Well rates in SuperEngine ("Compositional") are defined in reservoir conditions now, the units are kmol/day
- VTK output unified for all the reservoir classes [(See details)](https://gitlab.com/open-darts/open-darts/-/merge_requests/79)
- Discrete Fracture mesh generation tool and model added [(See details)](https://gitlab.com/open-darts/open-darts/-/merge_requests/79)
- Windows build script supports optional arguments  [(See details)](https://gitlab.com/open-darts/open-darts/-/merge_requests/82)
- Removed python exposures [(See details)](https://gitlab.com/open-darts/open-darts/-/merge_requests/74)
- Breaking changes:
    - The function `darts_model run_python()` is renamed to `run()`.
	- Changes in unstructured mesh processing. It is required to specify tags in mesh file for all control elements now  (matrix, fractures and optionally boundary faces).
	- Changes in vtk output:\
	    {- Before: input properties were saved to the first timestep vtk file -}\
        {+ Now: input properties saved to the separate "mesh.vtk" file +}\
		{- Struct/CPG: DartsModel.export_vtk(file_name, local_cell_data, global_cell_data, vars_data_dtype, export_grid_data) -}\
		{- Unstruct: UnstructReservoir.output_to_vtk(output_directory, output_filename, property_data, ith_step) -}\
        {+ DartsModel.output_to_vtk(ith_step, output_directory, output_properties) +}
	- No need to call `super().set_physics(physics)`, `super().set_reservoir()` and `super().set_physics()` in user's model: \
	    {- reservoir = UnstructReservoir(...) -}\
		{- super().set_reservoir(reservoir)-}\
        {+ self.reservoir = UnstructReservoir(...) +}
	- Discretization moved from the reservoir constructor to `init_reservoir()` method, which is called in `DartsModel.init()`.
	Thus, reservoir.mesh is not available right after the self.reservoir initialization. If it is needed, one can call `self.reservoir.init_reservoir()`. For example:
	```python
	self.reservoir = StrucReservoir(...)
	self.reservoir.init_reservoir()
	volume = np.array(reservoir.mesh.volume, copy=False)
	# set boundary volume
	m.init()
	```
	- Adding additional properties to the report changed. Examples:\
	{- from darts.physics.super.operator_evaluator import PropertyOperators -}\
	{- props = [('Brine saturation', 'sat', 1), ('Gas saturation', 'sat', 0)] -} \
	{- physics.add_property_operators(PropertyOperators(props, property_container)) -}\
	{+ property_container.output_props = {'Brine saturation': lambda: property_container.sat[1], 'Gas saturation': lambda: property_container.sat[0]} +}
	- `property_container` and `property_container` are now dictionaries with a key=region index. Examples:
	{- sat = self.physics.property_operators.property_container.compute_saturation_full(state) -}\
	{- properties = self.physics.vars + self.physics.property_operators.props_name -}\
	{+ sat = self.physics.property_containers[0].compute_saturation_full(state)  # 0 - is a region index +}\
	{+ properties = self.physics.vars + self.physics.property_operators[0].props_name +}
	- The engine object moved to physics (as before, in version 1.0.4):\
	{- m.engine -}\
    {+ m.physics.engine +}

	- Wells object moved to Reservoir :\
	{- m.wells -}\
    {+ m.reservoir.wells +}

# 1.0.5 [20-11-2023]
- Adjoints with MPFA (C++ discretizer for the unstructured grid)
- MPFA for the heat conduction (C++ discretizer for the unstructured grid)
- Fast and accurate version of CPG discretizer
	- Transmissibility computation at fault cells (NNC)
	- Set boundary volume takes into account cells inactivity
	- Fault transmissibility multiplier
	- Initialization with arrays dictionary
	- Added over- and underburden layers generation
	- Faster grid initialization and vtk export
	- Added export to grdecl files
- Interface for the direct control of RHS from Python
- Initial project documentation (readthedocs)
- Maximum number of equations increased to 8
- Fully functioning darts-flash
- Simple well index computation for the unstructured reservoir (Python discretizer)
- Generic structure for different engines and reservoirs
	physics_base and reservoir_base classes added
- vtk and shapely added to the requirements
- fixes for GPU version, CUDA12 and updated AMGX compatibility
- Folders reorganized
- Breaking changes:
    - Reservoir classes moved:
        - Before: from darts.models.reservoirs.struct_reservoir import StructReservoir
        - Now:    from darts.reservoirs.struct_reservoir import StructReservoir
    - Changes in base darts model:
        - Before: self.mesh
        - Now:    self.reservoir.mesh
	- Changes in add_perforation() function of reservoir classes:
		- Before: three arguments: i, j, k
        - Now:    tuple of IJK indices: (i,j,k) as one argument
		- Before: default values: well_index=-1, well_indexD=-1
        - Now:    default values: well_index=None, well_indexD=None
	- set_initial_conditions in DartsModel class using self.initial_values dictionary
		- self.initial_values = {'pressure': 200, 'w': 0.001}
	- Physics initialization:
		- Before: self.physics = Geothermal(...); self.physics.init_physics()
		- Now:    physics = Geothermal(...); super().set_physics(physics)
	- Reservoir initialization:
		- Before: self.reservoir = StructReservoir(...)
		- Now:    reservoir = StructReservoir(...); super().set_reservoir(reservoir)
	- Added method set_wells to the DartsModel class (function with the same name in user's model should be renamed)
		- Before: set_boundary_conditions() # well controls were here
		- Now:	rename it to set_wells() and added return super().set_wells() in the end of this method
	- The 'platform' parameter moved from the engine constructor to set_physics:
		- Before: Geothermal(..., platform='gpu')
		- Now:	Geothermal(...); super().set_physics(physics, platform='gpu')

# 1.0.4 [11-09-2023]
Small fixes.

# 1.0.3 [11-09-2023]
- Folders reorganized.
- Breaking changes: physics creation changed:\
	{- Before: -}
	```python
	self.physics = Geothermal(...)
	```
	{+ Now: +}
	```python
	from darts.physics.geothermal.property_container import PropertyContainer
	property_container = PropertyContainer()
	self.physics = Geothermal(...)
	self.physics.add_property_region(property_container)
	self.physics.init_physics()
	```

# 1.0.2 [30-06-2023]
- Wheels creation for Python 3.11 added.

# 1.0.0 [16-06-2023]
- Folders reorganized.
- Breaking changes: import paths changed:\
    {- Before: -}
	```python
	from darts.models.physics.geothermal import Geothermal
	from darts.models.physics.iapws.iapws_property_vec import _Backward1_T_Ph_vec
	```
    {+ Now: +}
	```python
	from darts.physics.geothermal.physics import Geothermal\
	from darts.physics.properties.iapws.iapws_property_vec import _Backward1_T_Ph_vec
	```
- Stop wheels creation for Python 3.6

# 0.1.4 [13-04-2023]
- Added heat losses from wellbore. It works by default for all thermal models.
- Added capability for connection arbitrary well segments for modeling closed-loop wells. 
- Adedd CoaxWell model which models closed loop well with surrounded reservoir. 
- Added poromechanics tests.

# 0.1.3 [06-03-2023]
Initial release.
