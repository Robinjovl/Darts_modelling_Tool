# MR !305 — Deep review and refactoring plan (v3)

**Scope**: `sajjad/ipr` → `development`, 92 files, +7 902 / −2 916.
**Branch under refactoring**: `sajjad/ipr_refactored` (12 commits, phases M0–M3 landed).
**Version history**: v1/v2 were produced from a multi-agent audit of the diff and the
GitLab discussions. **v3 incorporates a second, independent external review** and
corrects one materially wrong conclusion in v2 (§2).

---

## 1. Verdict

**Request changes** — unchanged from v2, and now reinforced. The MR mixes four
different maturity levels in one changeset: verified numerical corrections,
incomplete simulator infrastructure, experimental correlations, and OLGA-specific
workflow plus visualization code. The green pipeline demonstrates broad
non-regression coverage; it does not validate the missing Jacobians, multi-region
behaviour, multiwell ordering, interface semantics, or the performance of
per-Newton Python callbacks.

The v2 recommendation (phased refactoring on one branch) and the external
recommendation (split into ~9 focused MRs) are complementary, not competing: the
phases already landed map almost one-to-one onto the proposed split, and §8 states
the delivery mapping.

---

## 2. `2ph_comp_solid` — a marginal tolerance, not a regression in either direction

This model has now been misdiagnosed twice, in opposite directions, and the
resolution is worth recording carefully.

- **v2** called it a *pre-existing stale reference*. The conclusion (not
  MR-caused) was right; the stated reason was a guess.
- **The external review** asked for a comparison against `development` rather
  than a waiver. That was the right instinct.
- **v3 (first draft)** made that comparison, found `development` PASS and MR head
  FAIL, and concluded the MR caused it. **That was wrong** — the two trees were
  built with different `OPENDARTS_CONFIG` values, so it was never a valid A/B.

**Established, each verified directly:**

1. **A tree containing none of MR !305 fails with byte-identical numbers.**
   `open-darts-pt-constraints` (development lineage, `0c0f11c9`) produces exactly
   `L2 1.32E-08 / max 3.74E-08 / max abs 2.07E-06` — the same digits as the MR
   branch. `open-darts-minerals` likewise. All four trees share the *same*
   reference file (`5e71a004`, identical md5), so the reference is not the
   variable.
2. **The well-control fix is a provable no-op for this model.** The entire
   behavioural change is `ctrl_state_col_offset = (is_bhp_ctrl ? 0 :
   well_state_offset) * n_block_size_sq`, and `well_state_offset = is_inj ? 0 : 1`
   (`engines/src/well_controls.cpp:9`). The offset is non-zero **only for a
   rate-controlled producer**. `2ph_comp_solid` has a `MOLAR_RATE` **injector**
   and a **BHP** producer (`models/2ph_comp_solid/model.py:119,123`) — offset zero
   in both wells, so the fix writes to identical columns.
3. **The divergence appears at Newton iteration 0, in the reservoir**: the
   `KineticBasic` operator values differ by ~1 ULP (7.9e-19 on a 1.7e-5 scale).
   A well-control row cannot write reservoir operator values.
4. **The check is marginal and flips in both directions.** `development` passes
   it at 55 % of the 1e-9 bound; other trees fail at 1320 % of it. Conversely
   `Chem_benchmark_new` — same control topology, also kinetic — **fails on
   `development` and passes on the MR branch** (`L2 1.13E-09` against the same
   1e-09 tolerance).

**Conclusion**: the `2ph_comp_solid` well-pressure L2 tolerance is too tight for a
stiff kinetic model. It sits within a factor of about two of its pass/fail
boundary and is decided by build environment, not by source changes. **No
reference was regenerated**, because there is nothing from this MR to baseline
away.

**Required action** — not a re-baseline: make the check robust. Either widen the
well-pressure L2 tolerance for kinetic models to reflect the achievable
reproducibility, or make the comparison build-invariant. Until then this model
and `Chem_benchmark_new` will keep flipping between environments and will keep
costing reviewers time — as they did here, twice.

**Methodological lesson, recorded because it cost two wrong calls:** comparing
two trees only establishes something if everything except the variable under test
is identical. Reverting `darts/` to the MR head (v2) held the C++ build constant
but changed nothing relevant; comparing against a differently-configured
`development` build (v3 draft) changed two things at once. The decisive
experiment was the third one: a tree on the development lineage, built the same
way, containing none of the MR.

## 3. Findings the external review caught that v2 missed

| # | Finding | Evidence | Status |
|---|---|---|---|
| E1 | `2ph_comp_solid` must be compared against `development`, not waived | §2 — the instinct was right; the resulting diagnosis (MR-caused) was **refuted** by a zero-MR-code tree failing identically | Partially accepted: investigate, yes; MR-caused, no |
| E2 | IPR uses `property_containers[0]`, wrong for multi-region models | `darts/pipes/linear_dfm_well_ipr.py:373,388,394` | Confirmed. v2 flagged this only for the heat hook |
| E3 | The ramp schedule is evaluated at the old time, forcing a tiny first step | `darts/pipes/ramp_up_rate.py:599` asserts `first_ts_size < 0.01 s`; its own message says *"because during this time step I set the rate to zero!"* | Confirmed — a workaround, not a constraint |
| E4 | Scalar SciPy solve per interface in a face loop | `fsolve` per turbulent face, per Newton iteration | Confirmed — and M2 **relocated** it into `darts/pipes/drift_flux.py:276-281` without fixing it |
| E5 | The Jacobian stencil must be **declared before matrix allocation** | Python can only write blocks already in the BCSR pattern | Confirmed gap: v2's design caches CSR positions but has no pattern lifecycle or version |
| E6 | Additive contribution vs constraint replacement must be distinct typed operations | — | Partially addressed by M3's `DirichletPin` modes, but not formalized in the contract |
| E7 | Public API should separate **selector** from **law** | `FaceBoundary(selector=…, law=…)` | v2's items conflate the two (index arrays + behaviour) |
| E8 | Read-only diagnostics need a typed observer, not a residual-stage hook | — | M3 added `DartsModel.after_assembly()`; it is untyped and should become `NonlinearIterationObserver` |
| E9 | Restart/serialization belongs in the contribution contract | — | Absent from v2 entirely |
| E10 | Scaling matrix (10², 10⁴, 10⁶ contributions) + an assertion that Python callback count does not grow per cell/face | — | Absent from v2's validation matrix |
| E11 | `CHISHOLM` is offered as an option but raises | choke `:1252-1256` | Minor; now model-side after the M0 eviction |
| E12 | Split the MR into ~9 focused MRs | — | Adopted as the delivery strategy, §8 |

**Status since**: E2, E3 and E4 landed in M3.5, and E5–E9 landed in M4 — see §7
for what each one delivered, for the two defects found in M4's adjoint guard, and
for E7's remaining half (now a concrete deliverable: a batched
`connection_fluxes()`, which is also the prerequisite for M5). E1 stands as the
standing tolerance decision (§2, §10.1); E10 is measured and open — the
per-contribution scaling numbers are in `docs/for_developers/conditions_lowering.md`
§1 and its callback-count assertion is gate **G5** in §7's M6 — and E11/E12 are
unchanged (model-side after M0; §8).

---

## 4. Findings in v2 that the external review did not report

These are retained and remain the highest-severity items; several are already fixed
on `sajjad/ipr_refactored`.

| # | Finding | Severity |
|---|---|---|
| V1 | The lateral-heat hook subtracts heat from the **wellhead block's control-equation row**, not an energy equation — a kJ-scale term lands in a Kelvin-scale constraint | **Critical** — fixed in M1a |
| V2 | `apply(ctx)` runs *after* `assemble_linear_system()`, so any state-affecting condition leaves residual, Jacobian and convergence test evaluated at the previous iterate (measured on SPE11b: −1 linear iteration, 3e-7 shift) | **Critical design gap** — fixed in M3 via `project_state()` |
| V3 | Choke `RECOVERY='ON'` selects the unphysical low-pressure root and never applies the critical-flow cap | **Critical** (physics, not maturity) |
| V4 | B&G/Bai closures evaluate flow-direction-signed terms in a geometry-fixed frame — wrong for deviated producers, masked by the vertical-only benchmarks | Major — restricted to vertical in M1a |
| V5 | All three closure families crash or NaN on density inversion (ρG ≥ ρL) — the regime `bai_2023` explicitly targets | Major — fixed in M1a |
| V6 | B&G `C0` is unbounded by the `C0·s_G ≤ 1` constraint the velocity reconstruction requires → ±Inf velocities | Major — fixed in M1a |
| V7 | `Cmax` extrapolation degenerates inside its own advertised range (a2 < a1 above 4/3) | Minor — clamped in M1a |
| V8 | `save_dfm_well_props` silently drops requested `z`/`vG`/`vL`/rate columns | Major regression — fixed in M1a |
| V9 | `ms_well.num_segments` is uninitialized for EPM wells (nondeterministic garbage) | Major — fixed in M1a |
| V10 | `mech_discretizer.cpp:145` reads `mech_tangen.b` where `.a` is intended (counting pass disagrees with assembly pass) | Major — fixed in M2 |
| V11 | `bai` poroelastic case: `AQUIFER(t_init+50)` vs `pz_bounds` filled with `t_init` — a 50 K disagreement between two arrays the engine both reads | Open — surfaced by M2's coherence check |
| V12 | `run_test_suite2.py:264` runs `subprocess.run(["python", …])`; the main.py stage silently tests whatever interpreter is first on PATH | Open — CI defect, `sys.executable` is the fix |
| V13 | `reaktoro_kinetics` is broken at HEAD **and** at the merge base (its `Flash` override never creates `_solver`/`_conds`/`_state`); it is in no CI list | Open — pre-existing |
| V14 | `2ph_constant_k` cannot run as shipped (4 independent blockers) and its BHP pseudo-well applies **no dt**, giving an effective 1/dt productivity index | Open — documented in M3 |
| V15 | SPE11b is not run-to-run deterministic (16 OpenMP threads set in its constructor, ~2e-7 spread) | Open |

---

## 5. Convergent findings (independent agreement)

Both reviews independently reached the same conclusions on: the four-channel
taxonomy (wells and constitutive/operator sources stay **outside** the condition
API); `model.conditions.add(...)` as the public surface; the mechanics solver
bypassing `apply_rhs_flux`; three incompatible dt conventions; the lateral-heat
Jacobian being absent; the `well_layers_props` branch failing with a `TypeError`
on placeholder strings; the choke belonging outside core until validated; the
Ramp/UpstreamRamp duplication being unnecessary; `Pipe` needing typed closures;
IPR belonging on the well perforation as a flow law; `boundary_volumes` deserving
a real face BC or an explicitly named aquifer; and the GPU Jacobian never being
propagated from host writes.

Independent agreement on the architecture from two separate analyses is the
strongest signal in this document that the target design is right.

---

## 6. Architecture (v3)

Unchanged in its **two-layer split and four sanctioned channels**:

- **W — Wells**: controls, perforations, constraints. A separate public subsystem.
  A model using a "well" merely to fake a face boundary migrates to a face
  boundary; a physical well stays a well.
- **P — Physics operators**: constitutive sources inside property/operator
  evaluators (`evaluate_mass_source`, kinetic rates × `kin_factor`). Adjoint- and
  GPU-correct; deliberately outside the condition API.
- **D — Declarative boundary types**: `a·p + b·f = r` compiled into the
  discretizer. Types change stencils, transmissibilities and Jacobian sparsity, so
  they can never be a runtime hook.
- **R — Runtime conditions**: everything else, as closed vectorized item types.

**v3 changes to layer R**, from §3:

1. **Selector / law separation** (E7). An item becomes
   `CellSource(selector=…, law=PrescribedRate(…))` rather than an index array plus
   behaviour. Selectors resolve to indices at compile time; laws are reusable and
   independently testable.
2. **A stencil-declaration phase before Jacobian allocation** (E5). `compile()`
   gains a preceding `declare_stencil()` stage that reports every (row, column)
   block an item will touch, *before* the engine allocates its matrix, so a
   condition can introduce a coupling that does not yet exist. This is what
   removes the need for the IPR hook's zero-WI "fake perforation": today the
   pattern is smuggled in through a dummy perforation because Python cannot add a
   block afterwards. The compiled positions gain a **pattern version**; a
   re-initialized engine invalidates them instead of silently writing stale
   offsets.
3. **Additive vs replacement as distinct operations** (E6). `contribute()` and
   `replace_row()` are separate, and the assembler rejects an additive
   contribution to a row a constraint has claimed.
4. **A typed `NonlinearIterationObserver`** (E8), read-only, for diagnostics and
   policy checks — replacing the untyped `after_assembly` hook M3 introduced for
   the PHREEQC dilution policing.
5. **Restart/serialization in the contract** (E9): an item declares whether its
   internal state must survive a restart, and how it serializes.

**Backend lowering** is unchanged and remains the point of the design: face
boundaries lower to TPFA/MPFA/THM coefficients and native assembly; cell sources
lower to packed CPU/GPU kernels; connection fluxes lower to predeclared block
scatters; well laws lower through the well assembler. The Python `BlockCSRView`
path is a **small CPU prototyping fallback** for declared stencils, not the
general implementation — and unsupported backend contributions must fail at
initialization, never silently omit Jacobian terms.

---

## 7. Plan (v3)

### Landed on `sajjad/ipr_refactored`

| Phase | Content |
|---|---|
| **M0** | Choke evicted to model-side; skill-doc change reverted; unused plotters adopted |
| **M1a** | 13 correctness fixes (closure guards, IPR perturbation, heat control-row, `save_results`, Sutherland clamp, C++ tolerance/ghost/`num_segments`) |
| **M1b** | 116 tests incl. FD verification of the well-control row (~1e-8) and the IPR Jacobian (≤3e-6) |
| **M1c** | Conditions layer + `apply_rhs_flux` three-stage + mechanics solver + lifecycle |
| **M2** | RampUpRate strategy; Layer-D BC spec + S11 fix; viz renderer (−74%); `DriftFluxClosure` (Pipe 2276→1538); `ConstantStateBC`; shared OLGA base |
| **M3** | `DirichletPin` + `project_state()`; all remaining `set_rhs_flux`/`apply_rhs_flux` overriders migrated (**zero remain**) |
| **M3.5** | Region-aware pipe properties (E2); batched Colebrook solve (E4); ramp integrated over the step (E3) |
| **M4** | Stencil declaration + pattern versioning (E5); additive/replacement typing (E6); selectors (E7); typed observer (E8); restart contract (E9); typed mechanics BC declaration; placement-policy check wired into CI; the last two legacy hooks migrated and `set_rhs_flux`/`rhs_flux_hooks` removed; the layer documented in the technical reference |

Verification: 98/101 model suite, 259 unit tests, bit-identical A/B on every
migrated model and both refactors of the numerical core. The later phases carry
their own evidence, recorded with each commit: M3.5 moves the thirteen DFM cases
by at most 1.4e-11 against their existing references (and regenerates exactly one
model, for the ramp correction, §M3.5); M4's mechanics migration reproduces 780
compiled discretizer arrays bit-identically across the twenty-five CI
poromechanics cases, and its stencil declaration is proved on a twin of the
vertical OLGA benchmark with the dummy perforation deleted — identical row,
column and diagonal arrays, bit-identical simulation.

### M3.5 — corrections arising from this review (landed, except item 1)

1. **Do NOT regenerate `2ph_comp_solid`** (§2) — instead make its well-pressure
   L2 check robust to build environment. **Open**, and carried as standing issue
   §10.1: it is a test-tolerance decision for the maintainers, not a source
   change. The blast-radius audit of the well-control fix is complete and its
   "passes at development, fails here" class is **empty**: the fix moves
   rate-controlled *producers* only, and every such model
   (`2ph_geothermal_mass_flux`, `GeoRising` PT/PH, `SPE10_mech`,
   `cpg_sloping_fault` geothermal wrate) passes on both sides.
2. **Region-aware properties in the pipe hooks** (E2) — **landed**. Both the IPR
   hook and the lateral-heat hook resolve the property container from the
   connected block's operator region instead of `property_containers[0]`, with a
   multi-region test that would fail vacuously on a single-region model.
3. **Vectorize the Colebrook friction solve** (E4) — **landed**. One vectorized
   Newton iteration in `1/sqrt(f)` for all interfaces replaces the per-face
   `fsolve`, in the Colebrook closure, the Wang friction model and the
   Bhagwat-Ghajar profile parameter. Bit-identity with `fsolve` is not reachable
   *because `fsolve` is the inaccurate side* (up to 541 ULP from its own reported
   root, against ≤ 4 ULP for the batched solve); the thirteen DFM cases shift by
   ≤ 1.4e-11 and keep their references. ~6.5× faster at the 21 interfaces of the
   CI models, ~140× at 1e4.
4. **Fix the ramp schedule's time evaluation** (E3) — **landed**. The schedule is
   integrated exactly over `[t_n, t_{n+1}]` and the `first_ts_size < 0.01 s`
   assertion is gone. This is a physics correction and it moves one model:
   `2ph_1comp_coupled_dfm_well_reservoir`, the only CI case with a non-zero ramp
   period, was under-injecting by 3.2 % over the tested window; its Linux
   references were regenerated for that reason and its **Windows references still
   need a CI regeneration run**.

### M4 — contract hardening (landed)

The conditions layer is typed end to end. What each item of §3 asked for, and
what shipped:

| Item | Delivered |
|---|---|
| **E5** — declare the stencil before allocation | `ConditionItem.declare_stencil()` reports the `(row, col)` couplings an item will write, and the framework adds the missing ones as zero-transmissibility connections in the only valid window: between `mesh.add_wells()` (which assigns the well block indices) and `conn_mesh::reverse_and_sort()` (which freezes the list, in place, and cannot run twice). Existence is decided per pair — well connections enumerated, reservoir pairs matched by one vectorized scan — and the result is re-checked against the frozen arrays by `verify_stencil()`, so a missing *or* duplicated coupling fails loudly. A wellhead coupling is refused (that row carries the well-control equations). A perforation-free well is accepted when an item declares its coupling and reports it through `declared_well_names()`: **the zero-WI "fake" perforation is no longer needed**, proved on a twin of the vertical OLGA benchmark with the perforation deleted. Compiled CSR positions carry a `pattern_identity()` and are re-resolved after a matrix reallocation (reachable today through restart) instead of writing at stale offsets. |
| **E6** — additive vs replacement, typed | `ConditionItem.contribution` (`additive` / `replacement` / `none`) plus `written_rows()`; `compile()` refuses two claims on one row and an additive contribution to a claimed row, naming both items and the `(block, equation)`. `DirichletPin(mode="row")` is the shipped replacement case; `mode="state"` declares `none`, which is the right answer rather than a loophole — it projects the state and does not contribute to the system, and SPE11b pins cells that also carry a source. |
| **E7** — selector separated from law | `Selector` (`BlockIndices`, `Where`, `NamedRegion`) resolves a block set at bind time and is accepted anywhere an index array is (cell sets and either member of a connection). Strictly additive: every existing call that passes indices is unchanged. **Partial**: the *law* half is still the item subclass, not a composable law object — `CellSource(selector=…, law=PrescribedRate(…))` remains a target of the M5 lowering, where the law is what the backend kernel consumes. |
| **E8** — typed observer | `NonlinearIterationObserver`, registered on `model.conditions`, called once per Newton iteration after every item has contributed. Read-only is *enforced*, not documented: the context handed to `observe()` has non-writable `rhs` / `X` / `Xn` views and, since the M7 hardening, non-writable views of **all four** block-CSR arrays (`jac_vals`, `jac_rows`, `jac_cols`, `jac_diags`) — as first shipped only `jac_vals` was frozen and the structural arrays passed through writable (third-review finding R4, §11). The untyped `DartsModel.after_assembly(dt, t)` runs last and is deprecated. |
| **E9** — restart in the contract | `carries_restart_state` + `save_restart_state()` / `load_restart_state()`, serialized into a `<restart file>.conditions.json` sidecar written by `DartsModel.save_restart_state()`. A stateful item with a missing or mismatched sidecar **refuses** the restart rather than silently continuing from constructor defaults. |

Also landed in M4: the mechanics boundary conditions are declared through the
typed spec (`FaceBoundary(flow=…, mech=…, temp=…)`) in all five in-repo
poromechanics models, with the legacy `{an, bn, at, bt, rn, rt}` dict schema kept
for one cycle as a warning adapter; and the placement-policy check runs as a
whole-tree pre-commit hook, enforcing the two mechanically decidable clauses of
§6 (no module in the package without an in-repo consumer, no upward import from
`models/`) with a justified allowlist that cannot rot.

The layer is also no longer documented only in docstrings: `docs/technical_reference/conditions.md`
states the four channels, the contribution contract, the item types, the
declaration stage, the observer, the migration from the legacy paths, and the
limits (mechanics engines, the adjoint, the state-mode projection).

Finally, the two contributions that were deliberately held on the legacy path
while the framework changed underneath them — `LinearDFMWellIPRHook` and
`SemiAnalyticalWellLateralHeatTransferHook` — are condition items registered on
`model.conditions` in `set_wells()`, and **`set_rhs_flux` and `rhs_flux_hooks`
are removed from `DartsModel`**: `apply_rhs_flux()` is the conditions stage plus
the observer stage and nothing else, and is the framework's entry point rather
than an extension point. The two spellings are gone rather than adapted, and each
of the three ways they can still appear fails **loudly**: `rhs_flux_hooks.append()`
raises `AttributeError` at the call, and both a surviving `set_rhs_flux()` override
and a self-assigned `rhs_flux_hooks` list are refused at `init()` — the two cases
that would otherwise run on while silently dropping a source term. Nothing in
this repository registers a Python-side contribution any other way.

**Two defects in M4's adjoint guard, found while assessing M5** (details in
`docs/for_developers/conditions_lowering.md` §3):

1. **The guard is evaluated before the flag it reads can be set.**
   `ConditionSet.compile()` checks `engine.opt_history_matching` once, at the end
   of `DartsModel.init()`. `OptModuleSettings.make_opt_step_adjoint_method()` sets
   that flag *after* `init()` has returned and then calls `reset()`, which does not
   re-run `compile()`. In the driver's own call order the check therefore never
   fires, and a model with an opaque item history-matches to completion with a
   silently incomplete gradient. Fix: re-evaluate the check where the flag can be
   observed — in `reset()`, or at the first `apply_rhs_flux()` of a run. This is a
   correctness hole, not an optimization, and should be closed before M5 starts.
2. **`declare_stencil()` shifts the gradient vector's layout.** A declared-but-absent
   coupling is added as a zero-transmissibility connection, so `mesh->n_conns` grows
   by two and `n_interfaces = mesh->n_conns / 2` by one. The transmissibility
   gradient and the positional `col_dT_du` mapping built in `opt_module_settings.py`
   are both indexed by interface order, so a stencil-declaring item mis-indexes the
   gradient even when it contributes nothing. M5's adjoint work must exclude
   declared connections from that numbering or renumber `col_dT_du`.

Related, and free: `CellSource` with a constant or `f(t)` rate and no `d_rates` is
**provably adjoint-transparent** — it contributes nothing to `dg_dx_T`, `dg_dx_n`
or `dg_dT_general`, and the adjoint linearizes about the stored trajectory, which
already contains its effect. So does `ConstantStateBC`, which contributes nothing
at all. Both could set `adjoint_transparent = True` today (guarded by "the rate is
not a control variable and no stencil was declared") instead of blocking history
matching. Every other item type needs the native lowering first: the backward loop
rebuilds the Jacobian from C++ alone, with no per-step Python hook, so **native
lowering is a prerequisite for adjoint support rather than a parallel workstream**.

### M5 — backend lowering and well laws (assessed; partly landing)

**Assessed, with measurements, in `docs/for_developers/conditions_lowering.md`.**
The assessment changes the plan: most of the lowering §6 promises is not worth
building.

- **The Python cost is not where it was assumed to be.** On the heaviest CI case
  that uses the layer (`2ph_2comp_isothermal_dfm_vertical_well_vs_olga`,
  `ipr_volumetric`), `ConditionSet.apply` is 0.74 s of a 7.60 s run — **9.7 %**,
  against 47 % in OBL well point generation and 42 % in the Python drift-flux
  velocity evaluation.
- **The largest available win needs no C++.** `CellSource.apply`'s Jacobian path
  is a per-cell Python loop: 2626 ms per Newton iteration at 10⁶ contributions,
  against 18 ms for the vectorized RHS path. Replacing it with a flat-index numpy
  scatter measures **62 ms — 42×** — and is the fix for E10's "callback count must
  not grow per cell", which today is violated inside a shipped item type.
- **`InterfaceFlux` is the structural case**: a flat ~13.8 µs per connection
  (2.8 s per iteration at 2·10⁵), because `apply()` calls the subclass once per
  connection. Its compiled data is also the one that does **not** survive a trip to
  C++ — `connections` and `_positions` are Python lists of tuples, not arrays.
  Flattening them and adding a batched `connection_fluxes()` is the prerequisite
  for any lowering **and** is E7's remaining law half, deliverable before a kernel
  exists.
- **State dependence goes through the operators, not through a callback.** The
  engine already has a native, GPU-capable, adjoint-visible state-dependent source:
  `KIN_OP` scaled by `mesh.kin_factor`, i.e. channel P. A callback from C++ into
  Python is rejected — it keeps the GIL in the assembly, cannot exist in a CUDA
  kernel, and still leaves the adjoint replay unable to reproduce the term.
- **GPU needs staged arrays and a device scatter**, not a sync: `Jacobian->values_d`
  is what the linear solver reads, and the host copy is written only under
  `print_linear_system`. The existing `jac_wells` → `jac_wells_d` →
  `copy_data_within_device` path in `engine_super_gpu::assemble_jacobian_array` is
  the template.
- **Verdicts.** Do now: the `CellSource` scatter; the `InterfaceFlux` flattening +
  batched law; the E10 scaling tests; the adjoint-guard fix below. Design now,
  build on demand: native CPU and then GPU cell-source passes. **Do not build:** a
  generic native `InterfaceFlux` kernel (the flux law is user code; the one that
  matters here is a well flow law and belongs in the well assembler) and any
  C++→Python per-contribution callback.

**Landing natively, concurrently**: the linear IPR as a real perforation flow law
(`perforation_flow_law`, `ms_well::set_perforation_flow_law`,
`engine_base::build_perforation_flow_laws`, `engine_super_cpu::add_perforation_flow_law`)
— a POD law struct snapshotted at `init()`, resolved against the frozen connection
arrays with a loud failure on any mismatch, assembled with analytic derivatives
instead of the Python hook's `2 · n_vars` finite differences, and refusing both an
engine that cannot assemble it and an active `opt_history_matching`. That refusal
pattern is the one the rest of M5 should copy.

Still true of M4 and still M5's job: the Python `BlockCSRView` path is the
implementation rather than the fallback §6 describes; Jacobian contributions are
CPU-only; and the per-iteration cost is Python per item.

### M6 — inclusion gate (written down; audited, enforcement pending)

The gate is now eight checkable criteria a reviewer can tick — equation tests
against external numbers (G1), an independent public integration case with the
reference data in the tree (G2), finite-difference derivative verification (G3),
CPU ST/MT and GPU results **or an enforced restriction plus its reason** (G4),
scaling at ~10²/10⁴/10⁶ with the E10 callback-count assertion (G5), documented
applicability and failure modes (G6), a stable non-model-specific API (G7), and an
in-repo consumer (G8). "Not applicable" is an allowed answer; "not yet" means the
feature stays model-side. Stated in full, with the evidence each one demands, in
`docs/for_developers/conditions_lowering.md` §4, where it is also audited honestly
against the three features it governs. The audit's verdicts are candid, but the
gate's conclusion was **not enforced**: drift-flux and lateral heat fail gates
and remain in core with CI variants (third-review finding R7, §11). The gate
section is therefore labelled an assessment with the inclusion decision pending,
and `conditions_lowering.md` §4.4 records the maintainer decision required per
feature:

- **Choke** (outside the repository): fails every gate today, but its best
  evidence is recoverable — the deleted `tests/pipes/test_choke_models.py`
  (`aa899c8c^`) held eight genuine equation tests (Perkins critical-pressure ratio
  against the isentropic limit, pure-liquid against the incompressible orifice
  relation, the A30 stationary-point identity, SINTEF-HEM's incompressible limit,
  Rathjen–Straub surface tension against a CO₂ reference). Re-admission = restore
  those, fix `RECOVERY='ON'` (V3), resolve `CHISHOLM` (E11), re-express the API as
  a condition item or well law, add a CI model and an FD Jacobian test. G2 is the
  expensive one and is what decides whether it belongs in `darts/` at all.
- **The three drift-flux correlations**: G7 and G8 met (registry, per-closure
  keyword validation, standalone use, three CI variants); G6 much improved by M1a.
  **G1 and G2 are the real gap**: the 55 tests in
  `tests/pipes/test_drift_flux_closures.py` are algebra, guard and
  self-consistency tests, and **not one compares a closure against a number from
  the paper it implements**; the `vs_olga` CI variants compare against their own
  regenerated pickles, with no OLGA data in the tree. G3 is missing but cheap —
  `Pipe` has both an `"OBL"` and a `"numerical"` derivative path and nothing
  compares them.
- **Lateral heat**: closest to the gate. G1, G3 (PT), G6 (its docstring is the
  model of what G6 should look like) and G8 met. Fails on G7 —
  `well_layers_props` is an advertised argument that raises `NotImplementedError`,
  so either implement Willhite's U or delete the parameter — and on G2; G5 and the
  PH-lag convergence cost are unmeasured.

---

## 8. Delivery — mapping the phases onto focused MRs

The landed phases already decompose cleanly:

| MR | Content | Source |
|---|---|---|
| 1 | Well-control Jacobian correction **+ complete reference regeneration** | MR head + §2 |
| 2 | Pipe-state and extrapolation corrections | M1a |
| 3 | Small API/property fixes (Sutherland clamp, `add_perforation` validation, CHANGELOG) | M1a |
| 4 | Unified condition architecture | M1c + M4 |
| 5 | Pipe rate-boundary refactoring | M2 |
| 6 | Pipe closure strategies **+ vectorized friction** | M2 + M3.5 |
| 7 | Linear IPR as a well flow law | M5 |
| 8 | Well heat transfer, completed | M5 + M6 |
| 9 | Choke, only after the maturity gate | M6 |

Model migrations (M3) ride with MR 4. Visualization and OLGA workflow code should
be judged against the §6 placement policy independently of all of the above.

---

## 9. Validation matrix (v3)

Retained from v2: residual conservation/sign/unit/timestep tests; duplicate-selector
and conflicting-constraint tests; finite-difference Jacobian comparison for linear
IPR, heat transfer, face boundaries, pipe rate boundary and choke regimes; TPFA,
MPFA, mechanics/THM, PT and PH coverage; absent phases, zero flow, reverse flow and
composition boundaries; multiple wells with different creation orders.

**Added in v3** (E10, E2):

- Multi-region property resolution — a model with more than one operator region
  exercising every `property_containers[…]` lookup in the pipes package.
- Scaling at ~10², 10⁴ and 10⁶ contributions.
- An assertion that the number of Python callbacks does **not** grow per cell or
  face once native assembly is in place.
- A guard that no scalar SciPy solve occurs in a face-level production loop.
- CPU ST / CPU MT / GPU parity for every item type.

---

## 10. Standing issues for maintainer decision

1. The `2ph_comp_solid` / `Chem_benchmark_new` well-pressure L2 tolerance, which
   is decided by build environment rather than by source (§2). The blast-radius
   audit is done and found no affected model.
2. The `bai` 50 K Dirichlet incoherence between `mesh.bc` and `pz_bounds` (V11).
3. `2ph_constant_k`'s no-dt BHP pseudo-well — an effective 1/dt productivity index
   whose strength depends on timestep size (V14). Reproduced verbatim during
   migration; changing it is a physics decision needing its own re-baseline.
4. `reaktoro_kinetics` broken at HEAD and at the merge base, in no CI list (V13).
5. SPE11b's non-determinism from 16 OpenMP threads set in its constructor (V15).
6. `run_test_suite2.py`'s bare `python` subprocess (V12).
7. The heat-map `BoundaryNorm` crash when `n_cmap_bins_mu > n_cmap_bins_rho`, and
   the Colebrook array/scalar disagreement at exactly `Re == 2400` — both
   pre-existing, both preserved deliberately rather than silently changed.

---

## 11. Third independent review (2026-08-22) — findings R1–R8 and their disposition

A third independent review (`IPR_REFACTORED_CRITICAL_REVIEW.md`, of head
`19790f8f`) verified most of this document's resolutions, reproduced its open
items, and found eight defects in the refactoring itself. Its reproductions were
independently confirmed. The findings, and what the M7 hardening wave did about
each:

| ID | Finding | Disposition |
|---|---|---|
| **R1** | **Critical.** The native perforation flow law (`ipr_engine`) is assembled into residual/Jacobian, but both rate-reporting paths — modern `output.py` (WI-based transmissibility) and the legacy C++ `ms_well` time data (`p_diff * wi`) — know nothing about it. With the law's required zero geometric WI, every exported perforation and summed-well rate is exactly `0.0` while the simulation carries physically active IPR flow. | **Fixed in M7.** Rate reporting is made law-aware so exported perforation and summed-well rates equal the assembled law flux, with end-to-end rate/output tests. See CHANGELOG. |
| **R2** | The `ipr` env's installed package is stale MR-head code (`6f482ef6`); the repo-local engine binary embeds `f8d999aa` yet exposes post-M5 API — a dirty, non-reproducible build. A passing repo-root run is not evidence the installation works. | **Deferred — release step, owner: maintainer.** After the M7 source fixes land: clean-build exact HEAD, install into `ipr`, verify module/binary provenance from outside the checkout, and rerun both matrices without `PYTHONPATH` shadowing. The bare-`python` subprocess (V12, §10.6) belongs to the same step. |
| **R3** | Duplicate `CellSource` cells: the residual used advanced-index subtraction (last write wins) while the Jacobian used `np.add.at` (accumulates), so the advertised analytic Jacobian was not the derivative of the assembled residual. | **Fixed in M7.** Residual scatter accumulates (`np.add.at` semantics); duplicates now accumulate in **both** residual and Jacobian, with duplicate-cell tests. |
| **R4** | The "read-only" observer context froze only `jac_vals`; `jac_rows`/`jac_cols`/`jac_diags` passed through writable, so an observer could corrupt CSR structure through an engine-backed view. | **Fixed in M7.** The read-only twin freezes all four block-CSR arrays; structural-write tests added. §M4's E8 row above is corrected accordingly. |
| **R5** | `flow_law` is advertised on `ReservoirBase.add_perforation()` but only `StructReservoir` accepts it; `UnstructReservoir` and `CPG_Reservoir` raise an unexpected-keyword `TypeError`. | **In M7 scope (concurrent task): implement consistently or explicitly narrow.** If not closed in this wave it remains a merge blocker: either shared attachment in every concrete family with cross-family tests, or a documented capability/refusal narrowing to `StructReservoir`. |
| **R6** | An observer-only set on a mechanics model passes `compile()` (deliberate early return) but `apply_rhs_flux()` rejects the whole set at first assembly — compile and runtime disagree. | **Fixed in M7**, in the direction the reviewer called coherent once R4 is fixed: observer-only sets are permitted through the mechanics runtime path; contributing items on mechanics models are still refused at compile. The conditions reference documents the exemption. |
| **R7** | The M6 inclusion gate is documented but not enforced: its own audit records drift-flux and lateral heat failing gates, yet both remain in core with CI variants. | **Re-labelled in M7, decision escalated.** `conditions_lowering.md` §4 is now explicitly an assessment with the inclusion decision pending; §4.4 lists, per feature, the unmet gates, the evidence that would close each, and the cheap in-repo closures (paper-pinned closure values, the OBL-vs-numerical derivative test, the `well_layers_props` deletion, scaling points). Closing the gates or evicting the features is recorded as a maintainer decision — this wave neither softened the gate nor evicted anything. |
| **R8** | 117 generated PDFs under `models/dfm_well/*/output/` (5.5 MB), two run logs and a 187 KB generated mesh are tracked; commit `e7ac4d03` changes 117 output files alongside 13 source files, obscuring review. | **Handled in M7.** Every path was verified to be regenerated by `main.py`/test runs and consumed by no comparison; the exact 120-path removal list is `scratchpad_m7_artifact_removal.txt` (the `git rm` is executed centrally), and `.gitignore` now blocks `models/dfm_well/*/output/`, `log.txt` and the generated `transfinite.msh` from returning. The delivery-splitting half of R8 remains §8's plan (E12), still not delivered. |

The third review also confirmed as still open: V11 (bai 50 K), V12 (bare
`python`; folded into R2's release step), V13, V14, V15, E1 (reproduced with the
same `1.32e-08` L2), E3's Windows references, E12, and §10.7's two inherited
defects — all remain maintainer decisions as listed in §10.
