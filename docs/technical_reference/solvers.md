# Nonlinear and linear solvers

Every model configures its solvers in one place: the `set_solver()` hook of
`DartsModel`. This page lists what can be prescribed there, the defaults, and the
solvers available on CPU and GPU with their most important parameters.

## Where solvers are prescribed

`set_solver()` is called at the start of `reset()` — after the reservoir/mesh and the
engine object exist, but before `engine.init()`. Override it in a model to declare:

* time-stepping, via `self.ts_control.dt_first` / `.dt_mult` / `.dt_max` / `.runtime`
  etc.;
* the nonlinear solver, via `self.nonlinear_solver = NewtonSolver(...)`;
* the linear solver, via `self.linear_solver.spec = <LinearSolverSpec>`.

`nonlinear_solver` holds a *runtime* `NewtonSolver`, which a model replaces outright to
change the Newton driver — so it is (re)bound to the model on every `init()`/`reset()`.
`linear_solver` is a composed `darts.linear_solvers.LinearSolver` instance, created once
by `DartsModel.__init__` **with the model already attached** and never reassigned — so
there is no bind step. It owns the declarative `spec` as well as every method that binds a
model to its linear solver (`update_solver()`, `get_linear_system()`, ...). Because it
exists from construction, it is safe to touch
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
    self.ts_control.dt_first = ...                  # time-stepping
    self.ts_control.dt_max = ...
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
| `on_linear_nonconvergence` | `'accept'` | policy for a linear solve that exhausted its budget on a usable iterate: `'accept'` the inexact-Newton step or `'cut'` the timestep (see below) |
| `obl_bounds` | `OBLBoundsSpec(mode=None)` | optional per-axis clamping to the OBL box; `mode='obl_axes'` with `axis_min` / `axis_max` |

`NewtonSolver` also takes `pre_routines`, `post_routines` and `fallbacks` (all empty by
default) for user procedures around each nonlinear iteration.

### Non-convergence policy of the linear solve

Every linear solver reports one of three outcomes per solve (the unified
`linear_solver::solve()` convention): **converged**, **not converged but usable** (the
iteration budget ran out on a finite, non-regressing residual), or **hard failure**
(non-finite or growing residual, backend error). A hard failure always aborts the Newton
iteration and cuts the timestep. What happens on a *usable but non-converged* solve is a
nonlinear-solver policy, not the linear solver's decision:

```python
self.nonlinear_solver.spec.on_linear_nonconvergence = 'accept'   # default
self.nonlinear_solver.spec.on_linear_nonconvergence = 'cut'
```

`'accept'` applies the inexact-Newton step and lets the Newton residual gate decide —
the classical inexact-Newton treatment and the historical behaviour of the default
FGMRES+CPR solver. `'cut'` treats it as a failed solve and cuts the timestep — the
historical behaviour of MGR. With one policy in force, different linear solvers run the
same Newton trajectory on the same problem, which makes solver comparisons
like-for-like. Non-converged solves are marked ` NC` in the per-iteration log line and
counted in `nonlinear_solver.status.n_linear_nonconverged`.

## Solvers and preconditioners

Every spec plays one of three roles, given in the `Role` column of the tables below:

* a **solver** is assigned to `self.linear_solver` and drives the solve to a tolerance;
* a **preconditioner** is passed as `prec=` to a solver and is applied once per Krylov
  iteration — it never converges anything by itself;
* a **wrapper** composes another solver rather than replacing it (see *Composing solvers*).

## Linear solvers — CPU

All specs inherit `tolerance` (`1e-5`), `max_iterations` (`50`) and `print_level` (`0`).

| Spec | Role | Library | Type | Most important parameters |
|---|---|---|---|---|
| `GMRESSolverSpec` | solver | open-DARTS | restarted FGMRES, right-preconditioned | `restart` (Krylov subspace dimension, dataclass default `30`; the model default passes `50`), `prec` (a preconditioner spec, e.g. `CPRSolverSpec()`) |
| `CPRSolverSpec` | preconditioner | open-DARTS + HYPRE | two-stage CPR | `weight_scheme` (`1` = True-IMPES), `amg_max_iters` (V-cycles on the pressure stage), `stage2_type` (`1` = in-tree block ILU(0), the default; `0` = HYPRE scalar ILU(k)), `ilu_fill_level` (fill level for the `stage2_type=0` path only — inert at the default), plus the full BoomerAMG configuration (`amg_coarsen_type=8` PMIS, `amg_interp_type=8` extended+i, `amg_relax_type=3`, `amg_strong_threshold=0.75`, `amg_max_coarse_size=100`, …) and hierarchy reuse (`reuse_amg_hierarchy`, `adaptive_amg_rebuild`, `adaptive_iter_threshold`, `adaptive_consecutive_bad`) |
| `MGRSolverSpec` | preconditioner, assigned as a solver | HYPRE | MGR multigrid reduction + its own FlexGMRES/GMRES | `kdim` (Krylov dimension, `30`), `use_flex_gmres`, `use_physics_scaling`, level controls (`enable_well_level`, `enable_composition_level`, `pressure_level`, `custom_levels`), `use_bcsr_cpr`, `bilu0`, `local_correction`, `pressure_amg` |
| `FSCPRSolverSpec` | preconditioner | HYPRE | FS-CPR poromechanics preconditioner | `u_amg_max_iters`, `p_amg_max_iters`, `stage_growth_cap` (divergence guard, default `1e12` -- see *FS-CPR flow stage* below), `p_stage_type` (flow-block strategy when `NE = N_VARS - 3 > 1`: `2` = block-diagonal-decoupled systems AMG, the default; `1` = nested block CPR; `0` = raw systems AMG, diagnostic only — see *FS-CPR flow stage* below), `force_amg_asymmetric`, and the problem layout `n_res` / `n_fracs` / `n_wells` / `p_var` / `z_var` / `u_var` / `nc` |
| `SuperLUSolverSpec` | solver / preconditioner | SuperLU | sparse direct | none beyond the inherited fields (a direct solve takes no tolerance) |
| `PETScSolverSpec` | solver | PETSc | Krylov, Python-resident | `variant` (default `'cpr'`); builds a scalar PETSc `AIJ` matrix |
| `PardisoSolverSpec` | solver | Pardiso | direct, Python-resident | none beyond the inherited fields |

> **`MGRSolverSpec` is the exception to the solver/preconditioner split.** MGR is a
> preconditioner exactly like CPR — HYPRE attaches it with
> `HYPRE_ParCSRFlexGMRESSetPrecond(gmres, HYPRE_MGRSolve, HYPRE_MGRSetup, mgr_precond)`, and
> `use_mgr=False` runs the Krylov solver *without* it. But the spec names the whole **bundle**:
> FlexGMRES (or GMRES) *plus* MGR as its preconditioner. So it is assigned directly to
> `self.linear_solver`, and it cannot be passed as `prec=` —
> `GMRESSolverSpec(prec=MGRSolverSpec(...))` would nest one Krylov solver inside another,
> making the inner operator *varying*, which breaks the outer non-flexible GMRES and diverges
> into HYPRE NaNs. It raises `ValueError`. Hence the asymmetry in the names: `CPRSolverSpec`
> is a bare preconditioner that *needs* a `GMRESSolverSpec` around it, while `MGRSolverSpec`
> already contains its Krylov driver — despite both ending in `SolverSpec`.

### FS-CPR flow stage (`p_stage_type`)

FS-CPR splits the poromechanics Jacobian into a displacement block `U` (3
components per node) and a flow block, each preconditioned by its own
BoomerAMG inside the outer GMRES. When the model has a single flow equation
(`NE = N_VARS - 3 == 1`, poroelasticity) the flow block is a scalar pressure
system and a plain BoomerAMG is the right tool.

When `NE > 1` — thermoporoelasticity, or multiphase flow coupled to mechanics —
the flow block is a *coupled system* of component mass balances plus, when
thermal, the energy balance. There is no pressure equation among those rows.
BoomerAMG relaxes **point-wise** (hybrid Gauss–Seidel), so handing it those
rows unmodified means smoothing equations whose cell-local coupling is
dominated by the off-diagonal: `∂R_mass/∂z` is the accumulation term while
`∂R_mass/∂p` is only compressibility-small. On `SPE10_mech/dead_oil` that ratio
is ~700 and the resulting Gauss–Seidel spectral radius is ~400, i.e. the
smoother *amplifies* error and the solve makes no progress at all.
`HYPRE_BoomerAMGSetNumFunctions` cannot rescue this: it changes interpolation,
not the smoother. The block has to be decoupled first.

`p_stage_type` selects how:

| Value | Flow stage | Notes |
|---|---|---|
| `2` (default) | block-diagonal (ABF / quasi-IMPES) scaling of the flow block, then a systems BoomerAMG | Every flow unknown keeps multigrid treatment. Best on the thermal cases. |
| `1` | nested block CPR (`linsolv_cpr<NE>`) | Mirrors the proprietary FS-CPR. BoomerAMG on a true-IMPES-decoupled scalar *pressure* matrix, block ILU(0) carrying the rest. |
| `0` | systems BoomerAMG on the raw block | Pre-decoupling behaviour; diagnostic only, diverges on advective flow. |

Both `1` and `2` converge on every mechanics case in the test suite. `2` is the
default because demoting temperature to an ILU(0) second stage costs a large
factor on diffusive problems, while the extra cost of `2` on advective ones is
small:

| Case | `0` raw | `1` nested CPR | `2` decoupled AMG |
|---|---|---|---|
| `bai` (thermoporoelastic, 3 meshes) | 738 | 9352 | 757 |
| `SPE10_mech/single_phase_thermal` | 363 | 370 | 421 |
| `SPE10_mech/dead_oil` | *no progress* | 1695 | 1997 |
| `SPE10_mech/dead_oil_thermal` | *no progress* | 1610 | 1900 |

(total linear iterations, with identical timestep and Newton counts
throughout; `NE == 1` cases are unaffected by this field and are bit-identical
across all three.)

`2` is not the cheapest on any single row, but it is the only setting that runs
every case, and its cost against the best available option is bounded and
small. `0` cannot run multiphase flow at all. `1` is catastrophic on `bai` --
12x -- because it hands the energy balance to an ILU(0) second stage instead of
multigrid. Models that are known to be diffusive can set `p_stage_type=0`, and
models that are known to be advective can set `1`, but neither is safe as a
default.


### When FS-CPR cannot work at all

> **Historical note.** Between 2025-11-25 and the fix, FS-CPR appeared to fail on
> highly-resolved heterogeneous meshes. That was not a preconditioner problem: an
> unqualified `abs()` in the discretizer was truncating MPFA projection distances
> and SVD pivots to integers (see the CHANGELOG entry), which destroyed the
> M-matrix structure of the pressure block. With that fixed, FS-CPR solves
> `SPE10_mech/data_20_40_40` in ~14 linear iterations per Newton. The guidance
> below applies to operators that are genuinely outside classical AMG's reach --
> check the discretization first.

FS-CPR's stages are single BoomerAMG V-cycles, and a V-cycle is a contraction
only when the operator is close enough to an M-matrix for its point smoother to
converge. An MPFA pressure block on a strongly heterogeneous full-tensor field
need not be: on `SPE10_mech/data_20_40_40` only 41.7% of pressure rows are
diagonally dominant (99.8% on the 32x smaller `data_10_10_10`), 808 of 32004
rows carry a negative diagonal, and `cond_1(A_pp)` is 1.3e13 against 4.0e9.

The V-cycle then diverges, and it does so **silently**: HYPRE returns no error
and the vector is finite, just enormous — one apply amplifies the RHS by ~1.6e17.
FGMRES stalls at relative residual 1.0, reports "budget exhausted, residual did
not regress", and the default `on_linear_nonconvergence='accept'` policy applies
the step, so the Newton loop spins indefinitely.

`stage_growth_cap` catches this: a stage apply that is non-finite, or that
exceeds the cap times its own input in max-norm, fails the solve so the
timestep is cut. Healthy models peak at 106.5x amplification (measured over
1067 applies), so the `1e12` default has ten orders of headroom and still
catches the pathology five orders below it.

This makes such a failure diagnosable in under a second; it does not make the
operator solvable. If you hit it, the flow block is outside what a classical
AMG V-cycle can precondition — retuning BoomerAMG does not help (strength
threshold 0.25/0.5/0.75/0.9, `RelaxType` 3 + `RelaxOrder` 1, three
interpolation variants and the full CPR profile all still stall), raising the
V-cycle budget does not help, and neither does the proprietary BOS AMG: linked
from its prebuilt library and run against the same matrices it stalls
identically (8–18 iterations per solve on `data_10_10_10`, relative residual
~1.0 after 1001 iterations on `data_20_40_40`). Even a two-level hierarchy with
an *exact* coarse solve amplifies by 1.4e7, and ILUT either hits singular
pivots or diverges — on such an operator use a direct solver for the flow
stage or for the whole system.


## Linear solvers — GPU

GPU specs name the backend that the GPU engine factory builds; the AMG configuration
lives in the engine factory / AMGX JSON rather than in Python. The Python knobs are the
inherited three plus local Schur elimination.

| Spec | Role | Library | Type | Most important parameters |
|---|---|---|---|---|
| `AMGXCPRSolverSpec` | solver | AMGX + open-DARTS | GMRES + AMGX-CPR (**GPU default**) | `tolerance`, `max_iterations`, `print_level`, `schur_elim_count` / `schur_elim_rows` / `schur_elim_cols` |
| `GPUBiCGStabCPRSolverSpec` | solver | AMGX + open-DARTS | BiCGStab + AMGX-CPR | as above |
| `GPUGMRESILU0SolverSpec` | solver | cuSPARSE + open-DARTS | GMRES + cuSPARSE-ILU(0), single-stage fallback when the GPU build has no AMGX | as above |
| `CuDSSSolverSpec` | solver | cuDSS | sparse **direct** solver | as above; `WITH_CUDSS` is ON by default, but the GPU build silently omits cuDSS if the prebuilt library is not found |
| `GPUCuSolverSpec` | solver | cuSOLVER | QR sparse direct; NVIDIA deprecates this API in favour of cuDSS | as above |

`schur_elim_count` (`0` = off) statically condenses that many cell-local equations before
the solve; `schur_elim_rows` / `schur_elim_cols` name the eliminated equation rows and
unknown columns and must have `schur_elim_count` entries.


`GMRESSolverSpec` is the only spec exposing a `prec` field, so it is the composition point
on CPU. `MGRSolverSpec` is the exception to the pattern — see the note under the CPU table.

## Backends

Solver components are either **in-tree** (no external dependency) or built on a
**third-party** library. HYPRE is the only third-party library the CPU default requires, and
only for its pressure-stage AMG; the rest are independently optional.

**Every Krylov driver except MGR's is open-DARTS's own.** `GMRESSolverSpec` is in-tree even
  when preconditioned by HYPRE BoomerAMG, and the GPU GMRES/BiCGStab are in-tree cuBLAS code
  even when preconditioned by AMGX. Only `MGRSolverSpec` runs a HYPRE Krylov solver
  (`HYPRE_ParCSRFlexGMRESCreate` / `HYPRE_ParCSRGMRESCreate`).

**"GMRES" has three different implementations**: in-tree CPU (`linsolv_gmres.cpp`), in-tree
  GPU (`linsolv_gmres_gpu.cpp`, cuBLAS) and HYPRE's. They are not interchangeable and are not
  selected independently of the spec.

## Composing solvers

Two specs wrap another solver rather than being one:

* `SchurEliminationSpec(inner=..., elim_rows=..., elim_cols=..., pivot_eps=0.0)` — exact
  local block-Schur elimination of cell-local (e.g. mineral) equations around an inner
  solver, on CPU.
* `AdaptiveSolverSpec(candidates=[...], policy=..., on_timestep_failed=...)` — switches
  between candidate solvers during a run; `candidates[0]` is used first, and the policy
  decides per timestep from the previous step's state. At least one candidate is required.
  Candidates must be **engine-resident CPU registry specs** (`MGRSolverSpec`,
  `GMRESSolverSpec`, `CPRSolverSpec`, `SuperLUSolverSpec`, …): switching rebuilds the
  candidate through the open-source registry and injects it into the live engine, which a
  `GPUSolverSpec` (enum-selected by the engine factory) and a Python-resident solver
  (`PETScSolverSpec`, `PardisoSolverSpec`, owned by the model) cannot support. Both raise
  `TypeError` at construction rather than hours into a run. To retune the *current* solver
  instead of replacing it, call `model.linear_solver.update_solver(...)`, which reconfigures
  the injected solver in place without touching the Jacobian.

So the two families are separated in both directions: HYPRE's MGR cannot serve as a
preconditioner inside an in-tree Krylov driver (the `ValueError` above), and no in-tree
component can be substituted into MGR's own cycle. Within a run you may switch between
registry solvers or retune one, but you cannot hand the solve back and forth between a
registry solver and a GPU or Python-resident one.

## Build availability

The open-source solver stack is in-tree and needs no proprietary library. The proprietary
BOS backends remain optional behind the CMake switch `ENABLE_BOS_SOLVERS` (default `OFF`).
For fine control, a model may build a raw C++ solver object directly into
`self.linear_solver.handle` (for example `linear_solvers.create_mgr_solver_for_block_size(...)`);
this is valid only in the open-source build, so guard it with
`self.linear_solver.open_source_solvers_available()` and name it via
`self.linear_solver.label` for the log.
