> ℹ️ **Consolidated into `MR280_MASTER_PLAN.md` (2026-06-12).** This doc remains the authoritative
> narrative for the Python `self.solver` unification, but note two defects corrected in the master
> plan: (1) the phase numbering below is out of order with a **duplicated "Phase 5"** (the two
> Phase-5 blocks are the same topic — eliminating redundant solver-spec APIs — and Phase 4/GPU is
> physically interleaved); the master plan carries the clean 1→6 ordering. (2) The "Missing: all GPU
> specs" claim and Phase 4 ("needs GPU CI") are **stale** — `AMGXCPRSolverSpec`,
> `GPUBiCGStabCPRSolverSpec`, `GPUGMRESILU0SolverSpec` already exist and are wired via
> `_apply_gpu_solver()`; only end-to-end GPU validation remains.

> ✅ **Update (2026-06-13b) — model-level `params.linear_type` eliminated (Phase-5 close-out).**
> `self.solver = <LinearSolverSpec>` is now the **single interface** a model uses to declare its
> linear solver — no model writes `params.linear_type` / `data_ts.linear_type` as a selector anymore.
> The cross-build fallback moved *into the spec*: `LinearSolverSpec` gained
> **`proprietary_linear_type`** (a `darts.engines.linear_solver_t` enum), and the base
> `_apply_solver()` writes it to `params.linear_type` on the proprietary path (no open-source
> registry) — see `darts/solvers/specs.py` + `darts/models/darts_model.py`. Precedence is unchanged:
> open-source CPU → the spec builds + injects (`params.linear_type` ignored); proprietary → the spec's
> `proprietary_linear_type`; GPU → `_apply_gpu_solver()` translates the GPU spec's `linear_type_name`.
> Migrated: `2ph_comp`, `Adjoint_super_engine` (MGR → `proprietary_linear_type=cpu_gmres_cpr_amg`);
> `reaktoro_kinetics` (`SuperLUSolverSpec(proprietary_linear_type=cpu_superlu)`); `SPE10_mech`,
> `THM_vs_geomech_proxy`, `1ph_1comp_poroelastic_{analytics,convergence}` (GMRES+FS-CPR →
> `proprietary_linear_type=cpu_gmres_fs_cpr`, **mech_discretizer only**; the `pm_discretizer` branch
> keeps `engine.ls_params[-1].linear_type`, the mechanics multi-stage backend the plan retains).
> **Verified**: 2ph_comp bit-for-bit (1009/2234/4188, `verify_mgr_spec.py` PASS); Adjoint angle
> 0.000614°; poroelastic init + FS-CPR injection OK; and a forced registry-absent unit test confirms
> `_apply_solver` sets `params.linear_type = spec.proprietary_linear_type` on the proprietary path.
> **Only remaining model-level `params.linear_type`**: `darts/models/thmc_model.py` — the THMC base's
> pure engine-factory/`ls_params` mechanics path (no `self.solver` spec), which is the documented
> internal backend, not a redundant user-facing interface. (`params.linear_type` itself is **kept** as
> the internal backend the spec/GPU feed — only its model-level *use as a selector* is gone.)

> ✅ **Update (2026-06-13) — universal `set_solver` pattern + Phase-3 cleanup done.** The model
> suite now uses ONE consistent shape: **`set_sim_params(...)` is called from inside `set_solver()`**
> (which the base `reset()` runs before `engine.init`), not from `__init__`. The transitional
> scaffolding on the two MGR demonstrators was removed:
> - **`2ph_comp`** and **`Adjoint_super_engine`**: deleted the `use_bcsr_cpr_pressureguard_thr10_profile`
>   / `use_bcsr_cpr_levelaware_pressureguard_thr10_profile` helpers and the `set_sim_params` override;
>   folded time-stepping + the `MGRSolverSpec` into `set_solver()`; hardcoded the forward
>   `reduction_type=TRUE_IMPES`. **Verified bit-for-bit**: 2ph_comp `verify_mgr_spec.py` PASS
>   (TS=1009/NI=2234/LI=4188, `max|dX|=0`); Adjoint gradient angle 0.000614°, status 0.
> - **`BaseModels`**: moved `set_sim_params` `__init__`→`set_solver` (TS=20/NI=38/LI=167 unchanged).
> - **`chemistry/reaktoro_kinetics`**: migrated the last `params.linear_type = cpu_superlu` legacy
>   carrier to `self.solver = SuperLUSolverSpec()` in `set_solver()` (restores the intended direct
>   solve, which the base FGMRES+CPR default had been shadowing). Construct/`set_solver` smoke-verified;
>   a full run is blocked locally by an unrelated 8-dim OBL interpolator-template/index overflow.
> - **The ~24 flow models already followed the pattern** (set_sim_params in `set_solver`); mechanics
>   / THMC models (`SPE10_mech`, `THM_vs_geomech_proxy`, poroelastic) configure `params`/`ls_params`
>   directly and have no `set_sim_params` to move.
>
> **Documented exception — the `dfm_well` family (7 models).** These keep `set_sim_params` in
> `__init__` because `set_wells()` builds `RampUpRate(..., self.data_ts.dt_first, ...)`, and
> `set_wells()` runs inside `init()` *before* `reset()`/`set_solver()` — so `data_ts` must already
> exist at well-construction time. Moving `set_sim_params` into `set_solver()` would raise
> `AttributeError` on `self.data_ts`. A guarding comment was added to the 5 models with this hard
> dependency. Achieving the pattern there would require decoupling the ramp-up-rate setup from
> `data_ts.dt_first` (a deeper well-setup refactor, out of scope).

# Solver-Specification Unification Plan

**Goal:** one user-facing way to specify a linear solver — `self.solver = <LinearSolverSpec>` —
covering every solver and every backend, eliminating the four redundant mechanisms
(`params.linear_type` enum, `data_ts.linear_solver`, `engine.ls_params`, raw `self.solver` object).

> Decision (after evaluation): the **survivor is `LinearSolverSpec`**, exposed via the
> attribute `self.solver`. The raw C++ object path is *removed*, not kept — it is the
> weakest mechanism. "`self.solver` is a Python class that holds params and builds the
> solver" is exactly what `LinearSolverSpec` already is.

---

## Ground truth (recon)

Four mechanisms today; the C++ injection hook `engine.set_linear_solver(shared_ptr<linsolv_iface>, name)`
(`engine_base.h:192` → `linear_solver_external`) is honored by **3 of 4** backends:

| Backend | honors injected solver | what `spec.build()` must produce |
|---|---|---|
| CPU open-source | yes | build via `solvers.create_linear_solver(name, config, block_size)` → inject |
| CPU proprietary | yes (but no registry) | set `params.linear_type` enum (registry absent) |
| GPU (open) | partial — factory selects from enum; spec layer bypassed | enum-translation now; C++ wiring later |
| mechanics (super_elastic / pm) | yes (uses `engine.N_VARS`, not `physics.n_vars`) | inject built solver |

**Spec/registry inventory:** registry = `['cpr','fs_cpr','gmres','mgr','superlu']`; Python specs add
`PETSc`/`Pardiso` (Python-resident) + `AdaptiveSolverSpec`. **Missing:** standalone `ILU0`, all GPU specs.

**MGR coverage gap (the caveat):** 33 raw `set_*` knobs used by 2ph_comp/Adjoint; **16 already map to
`mgr_solver_config`, 18 do not** and need new C++ config fields (`composite_mode`, `local_solver`,
`bilu0_*`, `local_correction_*`, `pressure_amg_*`, all `bcsr_cpr_*`, `transpose_apply`, `forward_source`).
**→ closing the caveat requires a C++ rebuild, not just Python.**

---

## Target API (already agreed)

```python
def set_solver(self):
    self.set_sim_params(...)                      # time-stepping/Newton (unchanged, separate concern)
    self.solver = MGRSolverSpec(kdim=150, composite_mode=1, bilu0=BILU0Spec(...),
                                bcsr_cpr=BCSRCPRSpec(reduction=Reduction.TrueIMPES, ...),
                                pressure_level=MGRLevelSpec(...))
    # or SuperLUSolverSpec() / GMRESSolverSpec(prec=CPRSolverSpec()) / FSCPRSolverSpec()
    # / AMGXCPRSolverSpec() (GPU) / AdaptiveSolverSpec([fs_cpr, gmres+ilu0]) (displaced_fault)
```

`self.solver: LinearSolverSpec` is the single source of truth; default = `default_linear_solver(platform)`.

---

## Phases

### Phase 1 — Close the MGR caveat (C++ + Python + rebuild)  ✅ **DONE + verified**
- ✅ Added nested config structs `mgr_bilu0_config`, `mgr_local_correction_config`,
  `mgr_bcsr_cpr_config`, `mgr_pressure_amg_config` + 8 new optional fields
  (`scaling_type`, `composite_mode`, `local_solver`, `use_bcsr_cpr`, `bilu0`,
  `local_correction`, `bcsr_cpr`, `pressure_amg`) to `mgr_solver_config`
  (`solvers/include/solver_configs.hpp`). Defaults match `linsolv_mgr`'s constructor;
  applied only when set (so a bare config keeps out-of-the-box MGR behaviour).
- ✅ Wired them in `build_mgr` (`solvers/src/solver_factories.cpp`) in the exact order the
  composite models build them by hand (composite_mode / local_solver before `use_bcsr_cpr`,
  which auto-promotes the local solver). `transpose_apply`/`forward_source` are `optional<bool>`,
  applied only when set, preserving `linsolv_mgr`'s auto-derivation.
- ✅ pybind: bound the 4 nested structs + 8 new `MGRSolverConfig` fields (`py_main.cpp`).
- ✅ Added 5 enums (`CompositeMode`, `LocalPreconditioner`, `LocalFallback`, `BCSRCPRReduction`,
  `ScalingType`) to `darts/solvers/enums.py`; extended `MGRSolverSpec` with the 8 fields +
  nested `BILU0Spec`/`LocalCorrectionSpec`/`BCSRCPRSpec`/`PressureAMGSpec` (`darts/solvers/specs.py`);
  re-exported from `darts/solvers/__init__.py`.
- ✅ **Verified** (`models/2ph_comp/verify_mgr_spec.py`): the full `MGRSolverSpec` reproduces the
  raw-object MGR build **bit-for-bit** — TS=1009, NI=2234, LI=4188 identical, `max|dX| = 0`.
  *(Still TODO: `Adjoint_super_engine` gradient-angle check — it shares the same raw MGR build,
  so the same spec applies; defer to Phase 3 migration.)*

### Phase 2 — `ILU0SolverSpec` (C++ factory)
- `make_ilu0_solver` + `ilu0_solver_config` + pybind + registry name `'ilu0'` (HYPRE_ILU fill=0 shim,
  mirroring CPR's internal ILU). Python `ILU0SolverSpec`. Unblocks displaced_fault's dynamic phase.

### Phase 3 — Base mechanism + model migration (Python)  ✅ **DONE** (incl. universal-pattern cleanup, see top note)
Design via a 3-architect panel + lead synthesis (min-risk / clean-end-state / runtime-correctness).
Chosen: a single `_apply_solver(stage)` hook called at two sites in `reset()` —
`stage="pre"` (before `engine.init`) builds+injects every `LinearSolverSpec`; `stage="post"` (after)
injects only a legacy **raw** C++ object (the transition shim, removed in Phase 5). One
`_resolve_solver_spec()` precedence rule: `self.solver` spec > `data_ts.linear_solver` (legacy) > None.

**DONE + verified:**
- ✅ **Step 0 (base mechanism)** in `darts/models/darts_model.py`: added `_resolve_solver_spec()`,
  `_block_size()`, `_apply_solver(stage)`; rewrote `reset()` (two-site hook) and the default `set_solver()`
  (sets `self.solver = default_linear_solver("cpu")`; leaves `self.solver=None` on GPU/proprietary so the
  engine factory drives); `_maybe_switch_linear_solver` now uses the shared resolver; removed
  `_apply_linear_solver_spec` / `_apply_set_solver`. **Lazy default** restored in `_apply_solver` (when no
  explicit spec but `data_ts` exists → `default_linear_solver("cpu")`) so models that override `set_solver()`
  only for time-stepping (no `super()`) still get FGMRES+CPR. Gate: `verify_mgr_spec.py` unchanged still
  TS=1009/NI=2234/LI=4188, `max|dX|=0` (base rewrite perturbs nothing).
- ✅ **Step 4 (2ph_comp raw→`MGRSolverSpec`)**: `self.solver = MGRSolverSpec(...)` on the pre-init path;
  bit-for-bit vs the raw build (TS=1009/NI=2234/LI=4188, `max|dX|=0`); real `Model` re-confirmed.
- ✅ **Steps 2–3 (`data_ts.linear_solver` → `self.solver`)**: `1ph_1comp_poroelastic_analytics`,
  `1ph_1comp_poroelastic_convergence`, `SPE10_mech` (FS-CPR); `fracture_network`, `carbonated_water`
  (SuperLU). Same pre-init timing → behaviourally identical (carrier swap proven neutral).
- ✅ Editable-install fix: `site-packages/darts` was a frozen copy (build's `pip install`); swapped to a
  symlink → repo, so edits are live (`darts.sitecopy.bak` kept).

- ✅ **Step 5 — Adjoint_super_engine** raw→`MGRSolverSpec` (forward solver): migrated; `_attach` now only
  wires the adjoint solver + corrects the forward solver's `n_reservoir_blocks` post-init (it is 58
  pre-init / 50 post-init for this model — the spec captured the pre-init value, so `_attach` re-injects
  the corrected count, mirroring the old raw `_attach`). **Verified bit-for-bit**: adjoint-vs-numerical
  gradient angle = 0.000614° (== baseline), forward MGR `reservoir_blocks=50`, status=0.

### Phase 5 — Eliminate the redundant solver-specification APIs  ✅ **DONE + verified**
- ✅ **Raw shim removed**: no model builds a raw `self.solver` object anymore (2ph_comp, Adjoint migrated).
  Collapsed `_apply_solver(stage)` to a single pre-init `_apply_solver()`; dropped the `stage="post"` call
  from `reset()` and the raw-object / pre-init-placeholder branch.
- ✅ **`data_ts.linear_solver` carrier retired**: migrated the last users (`displaced_fault_reactivation`,
  `THM_vs_geomech_proxy`) `data_ts.linear_solver` → `self.solver`; removed the legacy fallback from
  `_resolve_solver_spec` (now: `self.solver` if a spec, else `None`) and deleted the `linear_solver` field
  from `DataTS`. Updated all docstrings / usage-example comments (PETSc/Pardiso) to `self.solver`.
- ✅ **Kept as internal backends** (per plan): `params.linear_type` / `data_ts.linear_type` (proprietary +
  GPU factory selectors) and `engine.ls_params` (mechanics multi-stage). The spec is the single
  *user-facing* API; these remain the backend the spec/engine feed.
- ✅ **Verified neutral**: 2ph_comp (LI=4188), 2ph_do (88/88), fracture_network (287/287), Adjoint
  (angle 0.000614°) unchanged after the collapse + carrier retirement.

**End state:** `self.solver: LinearSolverSpec` is the single user-facing way to pick a linear solver;
default via `default_linear_solver()`; `None` on GPU/proprietary (factory drives via `params.linear_type`).

**Note — displaced_fault / THM_vs_geomech_proxy** are not runnable locally (proprietary `pm_discretizer`
mechanics, missing meshes/pyvista); their `data_ts.linear_solver`→`self.solver` swap is the carrier-neutral
change proven on `fracture_network`, but their runtime needs CI / a GPU-less mechanics path to confirm.

### Phase 4 — GPU specs (C++, not locally verifiable — needs GPU CI)
- `AMGXCPRSolverSpec`, `BiCGStabCuSparseILUSpec`; either (a) GPU-engine wiring so `engine_base_gpu`
  consults the injected solver, or (b) interim: the GPU spec sets `params.linear_type`. `default_linear_solver("gpu")` returns a spec.

### Phase 5 — Remove the redundant user-facing APIs
- Delete `data_ts.linear_solver`, `data_ts.linear_type`, model-level `params.linear_type`,
  `engine.ls_params` from the API. Keep the C++ enum / `ls_params` as **internal backends only**
  (proprietary factory, GPU factory, mech multi-stage), fed by `spec.build()`.
- Proprietary parity: confirm the enum-translation path for all spec types.

### Phase 6 — Verify (CPU open + proprietary + GPU + mech) + docs.

---

## Risks / caveats
- **C++ + rebuild** for Phases 1, 2, 4 (and GPU is not locally verifiable — needs the GPU CI runner).
- **Proprietary build** can't be built locally; the enum-translation path needs CI validation.
- AdaptiveSolverSpec regime-driven switching (displaced_fault dynamic) needs a richer `SolverSwitchContext`
  (dt/time/inertia) — Phase 3 sub-task.
- This is a multi-step effort; each phase is independently shippable and verified before the next.
