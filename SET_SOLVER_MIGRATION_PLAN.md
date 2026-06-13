> ⚠️ **SUPERSEDED (2026-06-12).** This doc's API prescriptions describe code that no longer
> exists: the carrier `data_ts.linear_solver` was **removed** from `DataTS`, and
> `_apply_linear_solver_spec()` / `_apply_set_solver()` were replaced by a single
> `_apply_solver()`. The live carrier is **`self.solver` (a `LinearSolverSpec`)** — see
> `SOLVER_UNIFICATION_PLAN.md` and the consolidated **`MR280_MASTER_PLAN.md`**. Retained only
> for the §1 pipeline-failure root-cause history and the `fracture_network → SuperLUSolverSpec`
> fix. Do **not** follow §2–§4 as current guidance.

# `set_solver()` Migration Plan

Make `set_solver()` the **single, build-aware** mechanism every model uses to choose
its linear solver, and convert all `models/` to it. This unblocks the test failures in
pipeline [2580261915](https://gitlab.com/open-darts/open-darts/-/pipelines/2580261915)
(all builds green, all test jobs red).

---

## 1. What pipeline 2580261915 actually fails on (verified from job traces/artifacts)

`sha 885dcda9` — builds **all green** (the `row_thread_starts` test fix + CPR true-IMPES landed).
Test jobs:

| Job | Build | Symptom |
|---|---|---|
| `test-linux` / `test-windows` | proprietary (`ODLS=-a`) | 84/87: **`2ph_comp` (model.py + main.py)** and **`Adjoint_super_engine`** FAIL |
| `test-linux-ODLS` / `test-windows-ODLS` | open-source (`ODLS=''`) | **2 h timeout — hangs at `Fracture network tests:`**; also `2ph_comp` well-time-series mismatch |
| `test-linux-oahu-gpu` / `test-linux-tahiti-image-gpu` | open-source GPU | **2 h timeout — hangs at `Fracture network tests:`** |

Two build-config-specific failure classes (root causes confirmed by local reproduction in env `solvers`):

1. **`2ph_comp` + `Adjoint_super_engine` crash in the proprietary build.**
   `models/2ph_comp/model.py:set_solver()` and `Adjoint_super_engine/model_definition.py`
   call `solvers.create_mgr_solver_for_block_size(...)`. That compiled API exists **only in the
   open-source build** (`darts.solvers._have_compiled_solvers`); the proprietary `-a` build links
   the prebuilt `darts-linear-solvers` and has no `solvers.so`, so the call raises
   `AttributeError: module 'darts.solvers' has no attribute 'create_mgr_solver_for_block_size'`.
   These models run fine in the open-source build.

2. **`fracture_network` hangs in every open-source build** (ODLS Linux + ODLS Windows + both GPU),
   passes in proprietary. It is a **Geothermal DFM** model using the *default* solver. Local
   reproduction of `case_1` (3 report steps):

   | Solver | Result |
   |---|---|
   | default FGMRES+CPR | **hangs** (never converges → timestep collapse) |
   | **SuperLU (direct)** | **3.5 s**, `lin 1 (0.0e+00)`, clean Newton |
   | MGR | "linear solver solve failed" → min-timestep abort |

   The matrix is small (4 fracture segments + matrix cells); a **direct solve is the right fit**
   — exactly what the commented `# self.params.linear_type = sim_params.cpu_superlu` in
   `model.py` anticipated.

   (Aside, already fixed: `2ph_do_thermal_mpfa` hung the same way before the CPR true-IMPES
   commit `885dcda9`; the default now converges it. `2ph_comp`'s ODLS well-time-series mismatch
   is a *numeric* difference vs a proprietary-generated reference pkl — tracked separately, see §6.)

---

## 2. Target design — the `set_solver()` contract

`set_solver()` is the one place a model declares its linear solver. Two valid forms:

- **Spec form (preferred, build-safe):** set `self.data_ts.linear_solver = <LinearSolverSpec>`
  (`SuperLUSolverSpec`, `GMRESSolverSpec(prec=CPRSolverSpec())`, `MGRSolverSpec`, `FSCPRSolverSpec`, …).
  Built and injected by `DartsModel._apply_linear_solver_spec()`, which is **already guarded** to
  the open-source CPU registry — in proprietary / GPU builds it is a no-op and the engine factory
  picks the solver from `params.linear_type`. So a Spec is safe to set in *any* build.
- **Raw-object form (low-level, open-source-only):** build `self.solver` directly
  (`solvers.create_mgr_solver_for_block_size(...)` + `set_mgr_*`), as `2ph_comp` / `Adjoint`
  need for fine MGR control. Injected by the base `_apply_set_solver()`. **Must be guarded** by a
  build-availability check.

Base scaffolding already in place (this session, `darts/models/darts_model.py`):
`self.solver` / `self.solver_label`, the no-op `set_solver()` hook, and `_apply_set_solver()`
(called at the end of `reset()`, builds `self.solver` only if `None`, injects it).

**To add (Phase 0):** a guard helper so raw-object models degrade gracefully:

```python
# darts/models/darts_model.py
@staticmethod
def open_source_solvers_available() -> bool:
    """True when the compiled darts.solvers registry (MGR/registry API) is present
    (open-source build). False in proprietary -a builds, where create_mgr_solver_* etc.
    do not exist and the engine factory selects the solver from params.linear_type."""
    try:
        from darts import solvers
        return bool(getattr(solvers, "_have_compiled_solvers", False))
    except Exception:
        return False
```

---

## 3. Migration phases

### Phase 0 — Base interface hardening (done + small additions)
- [x] `set_solver()` hook, `_apply_set_solver()` injection, `self.solver` / `self.solver_label`.
- [ ] Add `DartsModel.open_source_solvers_available()` (above).
- [ ] `_apply_set_solver()`: if `self.solver` was built but `not open_source_solvers_available()`,
      skip injection defensively (belt-and-suspenders).

### Phase 1 — Fix the pipeline breakers (small, high-value, mostly verified)
- [ ] **`fracture_network`** — add `set_solver()`:
      ```python
      def set_solver(self):
          self.data_ts.linear_solver = SuperLUSolverSpec()
      ```
      ✅ verified: `case_1` hang → 3.5 s. (Spec form ⇒ proprietary build ignores it and keeps
      passing via the engine factory; open-source build uses the direct solver.)
- [ ] **`2ph_comp` / `Adjoint_super_engine`** — guard the raw MGR build:
      ```python
      def set_solver(self):
          if not self.open_source_solvers_available():
              self.solver = None          # proprietary build → fall back to engine factory
              self.params.linear_type = sim_params.cpu_gmres_cpr   # proprietary-supported default
              return
          ... existing create_mgr_solver_for_block_size(...) ...
      ```
      Fixes the proprietary crash (no-op there → factory default). ⚠ needs **proprietary-build CI
      validation** — the `-a` build can't be built locally here. Same guard at the top of
      `set_adjoint_solver()`.

### Phase 2 — Migrate models that already pick a non-default solver
Move the assignment out of `__init__` / `set_sim_params` into `set_solver()`, expressed as a Spec:
- `1ph_1comp_poroelastic_analytics`, `1ph_1comp_poroelastic_convergence`, `SPE10_mech`
  — already build `GMRESSolverSpec(prec=FSCPRSolverSpec…)`; just relocate into `set_solver()`.
- `displaced_fault_reactivation` (superlu / fs_cpr / ilu0 adaptive) → `set_solver()` with
  `AdaptiveSolverSpec` or the chosen `Spec`.
- `THM_vs_geomech_proxy` (`cpu_gmres_fs_cpr`) → `GMRESSolverSpec(prec=FSCPRSolverSpec())`.
- `chemistry/reaktoro_kinetics` (`cpu_superlu`) → `SuperLUSolverSpec()`.
- `cpg_sloping_fault` and others → `set_solver()` (default or Spec).

### Phase 3 — Migrate default-solver models + retire `linear_type`
- Models content with the default need **no** `set_solver()` (base no-op).
- Replace remaining legacy `params.linear_type = <enum>` assignments with the equivalent Spec.
- Once every model is on `set_solver()`/`data_ts.linear_solver`, the `linear_type` enum plumbing
  becomes back-compat-only (per the MR's stated direction).

### Phase 4 — Validation
- Per migrated model: `python main.py` in env `solvers` (open-source) → converges / matches ref.
- Full `models/run_test_suite2.py` locally in the open-source build.
- CI: `test-linux-ODLS` + GPU green (no fracture hang); `test-linux` green (2ph_comp/Adjoint fall back).

---

## 4. Per-model action table (the 23 models that touch the solver)

| Model | Now | Action |
|---|---|---|
| `fracture_network` | default → **hangs (open-src)** | `set_solver()` → `SuperLUSolverSpec()` ✅ |
| `2ph_comp` | raw MGR (`set_solver`) | keep + **build guard** |
| `Adjoint_super_engine` | raw MGR fwd+adj (`set_solver`/`set_adjoint_solver`) | keep + **build guard** |
| `2ph_do_thermal_mpfa` | default | OK now (CPR true-IMPES); leave default |
| `1ph_1comp_poroelastic_analytics` | `data_ts.linear_solver=GMRESSolverSpec` in `init` | relocate to `set_solver()` |
| `1ph_1comp_poroelastic_convergence` | same | relocate to `set_solver()` |
| `SPE10_mech` | same | relocate to `set_solver()` |
| `displaced_fault_reactivation` | `linear_type` superlu/fs_cpr/ilu0 | `set_solver()` + Adaptive/Spec |
| `THM_vs_geomech_proxy` | `cpu_gmres_fs_cpr` | `set_solver()` → `GMRESSolverSpec(prec=FSCPRSolverSpec())` |
| `chemistry/reaktoro_kinetics` | `cpu_superlu` | `set_solver()` → `SuperLUSolverSpec()` |
| `cpg_sloping_fault`, `CCS`, `2ph_*`, `3ph_*`, `Uniform_Brugge`, `chemistry/carbonated_water`, `2ph_constant_k` | default / legacy `linear_type` in `main.py` | default (no `set_solver`) unless a run shows non-convergence |

(Run each candidate in the open-source build; any that hangs/stalls like `fracture_network`
gets an explicit `SuperLUSolverSpec` or tuned Spec via `set_solver()`.)

---

## 5. Risks & notes
- **Proprietary fallback is unverifiable locally** (no `-a` build here) → Phase-1 `2ph_comp`/`Adjoint`
  guard must be checked in CI. Also confirm `cpu_gmres_cpr` (or chosen fallback) is the right
  proprietary-supported `linear_type`.
- **SuperLU memory** scales poorly with size — only assign it to small models (`fracture_network`
  `case_1` is tiny; large overburden cases may need an iterative Spec instead — gate by cell count).
- The proprietary test jobs are slated for removal (Release v2.0.0 milestone); the guard keeps them
  green in the interim without blocking the open-source path.

## 6. Out of scope (separate follow-ups)
- `2ph_comp` ODLS **well-time-series mismatch**: reference pkls were generated with the proprietary
  solver; open-source numerics differ. Fix by regenerating the open-source reference pkl
  (`_odls` suffix) or loosening the comparison threshold — not a `set_solver()` change.
- GPU numeric mismatches on other models (separate investigation; GPU artifacts expire).
