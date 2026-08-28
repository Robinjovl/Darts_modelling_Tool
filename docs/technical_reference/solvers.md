# Nonlinear and linear solvers

Every model configures its solvers in one place: the `set_solver()` hook of
`DartsModel`. This page lists what can be prescribed there, the defaults, and the
solvers available on CPU and GPU with their most important parameters.

## Where solvers are prescribed

`set_solver()` is called at the start of `reset()` — after the reservoir/mesh and the
engine object exist, but before `engine.init()`. Override it in a model to declare:

* time-stepping, via `self.linear_solver.set_sim_params(...)` (time-stepping
  parameters **only**);
* the nonlinear solver, via `self.nonlinear_solver = NewtonSolver(...)`;
* the linear solver, via `self.linear_solver.spec = <LinearSolverSpec>`.

`nonlinear_solver` holds a *runtime* `NewtonSolver`, which a model replaces outright to
change the Newton driver — so it is (re)bound to the model on every `init()`/`reset()`.
`linear_solver` is a composed `darts.linear_solvers.LinearSolver` instance, created once
by `DartsModel.__init__` **with the model already attached** and never reassigned — so
there is no bind step. It owns the declarative `spec` as well as every method that binds a
model to its linear solver (`update_solver()`, `get_linear_system()`, the deprecated
`set_sim_params()` family, ...). Because it exists from construction, it is safe to touch
before `init()` — from a subclass `__init__` or a driver script. Assign a **spec** to
`.spec` to choose a solver:

```python
self.linear_solver.spec = MGRSolverSpec(tolerance=1e-4)
```

Assigning a spec is **build-safe**: in proprietary and GPU builds no C++ backend is
constructed from it, and the engine factory selects the backend instead.

There are two idiomatic ways to override. Replace a solver outright:

```python
def set_solver(self):
    super().set_solver()
    self.nonlinear_solver = NewtonSolver(tolerance=1e-4, chop=ChopSpec(mode='global'))
    self.linear_solver.spec = MGRSolverSpec(tolerance=1e-4)
```

or keep the defaults and tune them through `.spec`:

```python
def set_solver(self):
    self.linear_solver.set_sim_params(first_ts=..., max_ts=...)   # time-stepping
    super().set_solver()                            # default solvers
    self.nonlinear_solver.spec.tolerance = 1e-4
    self.linear_solver.spec.tolerance = 1e-6
```

The default implementation is idempotent and lazy: it keeps whatever a subclass already
assigned and only materializes a default for what is still unset. After `init()`, the
live backend is reachable as `linear_solver.handle` (engine-injected C++ solver) or
`linear_solver.python_solver` (PETSc / Pardiso).

## Defaults

| Platform | Default linear solver |
|---|---|
| CPU | `GMRESSolverSpec(restart=50, prec=CPRSolverSpec())` — FGMRES around two-stage CPR (HYPRE BoomerAMG on pressure + ILU(0) on the full system) |
| GPU | `AMGXCPRSolverSpec()` — GMRES + AMGX-CPR (AMGX on the pressure subsystem + ILU on the full system) |

Both defaults use `tolerance=1e-5`, `max_iterations=50`, `print_level=0`. `set_solver()`
spells out every default field explicitly, so the effective configuration of a model that
does not override it is readable in one place rather than hidden in dataclass defaults.

The nonlinear default is a `NewtonSolver` with:

| Parameter | Default | Meaning |
|---|---|---|
| `tolerance` | `1e-3` | reservoir-block residual tolerance |
| `well_tolerance_multiplier` | `100.0` | well tolerance = `tolerance × this` |
| `max_iterations` | `20` | max Newton iterations per timestep |
| `stationary_point_tolerance` | `1e-3` | residual-stagnation detection |
| `norm` | `Norm.L2` | residual norm |
| `coupled_well_res_norm_method` | `1` | DFM coupled well-residual norm (`1` or `2`) |
| `chop` | `ChopSpec(mode='local', factor=0.1, log_transform=False)` | update limiting; `mode` is `'local'`, `'global'` or `None`, `factor` is the max composition change per iteration |
| `obl_bounds` | `OBLBoundsSpec(mode=None)` | optional per-axis clamping to the OBL box; `mode='obl_axes'` with `axis_min` / `axis_max` |

`NewtonSolver` also takes `pre_routines`, `post_routines` and `fallbacks` (all empty by
default) for user procedures around each nonlinear iteration.

## Linear solvers — CPU

All specs inherit `tolerance` (`1e-5`), `max_iterations` (`50`) and `print_level` (`0`).

| Spec | Type | Most important parameters |
|---|---|---|
| `GMRESSolverSpec` | restarted FGMRES, right-preconditioned | `restart` (Krylov subspace dimension, dataclass default `30`; the model default passes `50`), `prec` (a preconditioner spec, e.g. `CPRSolverSpec()`) |
| `CPRSolverSpec` | two-stage CPR preconditioner | `weight_scheme` (`1` = True-IMPES), `amg_max_iters` (V-cycles on the pressure stage), `ilu_fill_level` (`0` = ILU(0) second stage), `stage2_type`, plus the full BoomerAMG configuration (`amg_coarsen_type=8` PMIS, `amg_interp_type=8` extended+i, `amg_relax_type=3`, `amg_strong_threshold=0.75`, `amg_max_coarse_size=100`, …) and hierarchy reuse (`reuse_amg_hierarchy`, `adaptive_amg_rebuild`, `adaptive_iter_threshold`, `adaptive_consecutive_bad`) |
| `MGRSolverSpec` | HYPRE MGR multigrid reduction | `kdim` (Krylov dimension, `30`), `use_flex_gmres`, `use_physics_scaling`, level controls (`enable_well_level`, `enable_composition_level`, `pressure_level`, `custom_levels`), `use_bcsr_cpr`, `bilu0`, `local_correction`, `pressure_amg` |
| `FSCPRSolverSpec` | FS-CPR poromechanics preconditioner | `u_amg_max_iters`, `p_amg_max_iters`, `force_amg_asymmetric`, and the problem layout `n_res` / `n_fracs` / `n_wells` / `p_var` / `z_var` / `u_var` / `nc` |
| `SuperLUSolverSpec` | sparse direct | none beyond the inherited fields (a direct solve takes no tolerance) |
| `PETScSolverSpec` | PETSc (`petsc4py`) Krylov, Python-resident | `variant` (default `'cpr'`); builds a scalar PETSc `AIJ` matrix |
| `PardisoSolverSpec` | Pardiso (`pypardiso` / Intel MKL) direct, Python-resident | none beyond the inherited fields |

## Linear solvers — GPU

GPU specs name the backend that the GPU engine factory builds; the AMG configuration
lives in the engine factory / AMGX JSON rather than in Python. The Python knobs are the
inherited three plus local Schur elimination.

| Spec | Type | Most important parameters |
|---|---|---|
| `AMGXCPRSolverSpec` | GMRES + AMGX-CPR (**GPU default**) | `tolerance`, `max_iterations`, `print_level`, `schur_elim_count` / `schur_elim_rows` / `schur_elim_cols` |
| `GPUBiCGStabCPRSolverSpec` | BiCGStab + AMGX-CPR | as above |
| `GPUGMRESILU0SolverSpec` | GMRES + cuSPARSE-ILU(0), single-stage fallback when the GPU build has no AMGX | as above |
| `CuDSSSolverSpec` | NVIDIA cuDSS sparse **direct** solver | as above; `WITH_CUDSS` is ON by default, but the GPU build silently omits cuDSS if the prebuilt library is not found |
| `GPUCuSolverSpec` | cuSOLVER QR sparse direct (legacy) | as above |

`schur_elim_count` (`0` = off) statically condenses that many cell-local equations before
the solve; `schur_elim_rows` / `schur_elim_cols` name the eliminated equation rows and
unknown columns and must have `schur_elim_count` entries.

## Composing solvers

Two specs wrap another solver rather than being one:

* `SchurEliminationSpec(inner=..., elim_rows=..., elim_cols=..., pivot_eps=0.0)` — exact
  local block-Schur elimination of cell-local (e.g. mineral) equations around an inner
  solver, on CPU.
* `AdaptiveSolverSpec(candidates=[...], policy=..., on_timestep_failed=...)` — switches
  between candidate solvers during a run; `candidates[0]` is used first, and the policy
  decides per timestep from the previous step's state. At least one candidate is required.

## Build availability

The open-source solver stack is in-tree and needs no proprietary library. The proprietary
BOS backends remain optional behind the CMake switch `ENABLE_BOS_SOLVERS` (default `OFF`).
For fine control, a model may build a raw C++ solver object directly into
`self.linear_solver.handle` (for example `linear_solvers.create_mgr_solver_for_block_size(...)`);
this is valid only in the open-source build, so guard it with
`self.linear_solver.open_source_solvers_available()` and name it via
`self.linear_solver.label` for the log.
