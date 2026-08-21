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

## 2. Correction to v2 — `2ph_comp_solid` is an MR-caused regression

v2 recorded `2ph_comp_solid` as a **pre-existing stale reference**. **That was
wrong**, and the external review was right to insist the comparison be made
against `development` rather than waived.

Measured:

| Tree | Result |
|---|---|
| `development` @ `89679e42` (MR base) | **PASS** |
| MR head @ `6f482ef6` | **FAIL** — normalized L2 1.32e-08 (tol 1e-09); normalized max 3.74e-08 (tol 1e-07, passes); max abs 2.07e-06 bar |
| `sajjad/ipr_refactored` @ M3 | FAIL, identical numbers |

The v2 error was methodological: I established that the failure is not caused by
the M0–M3 refactoring (reverting the entire `darts/` tree to the MR head still
fails) and then wrongly generalized that to "pre-existing on the development
lineage". Reverting to the *MR head* says nothing about *development*.

**Mechanism** — the model is well-rate-controlled (`well_control_iface.MOLAR_RATE`,
`models/2ph_comp_solid/model.py:119`), which is precisely the case the MR's
well-control Jacobian block-placement fix changes. That fix is *correct*: the
finite-difference test added in M1b (`tests/test_well_control_jacobian_fd.py`)
reproduces the assembled wellhead row to ~1e-8 relative for BHP, molar/mass total
and phase rate, injector and producer. It legitimately shifts the Newton path.

**The actual defect is scope**: the MR regenerated 32 reference files for
`GeoRising` and `2ph_geothermal_mass_flux` for exactly this reason, but never
enumerated the full blast radius of the fix. `2ph_comp_solid` was missed.

**Required action** — regenerate `models/2ph_comp_solid/ref/perf_*.pkl` (all four
suffixes; Windows variants via the CI `UPLOAD_PKL` pipeline) *and* record the
measured drift in the CHANGELOG next to the existing entry, as was done for the
other two models. Before doing so, enumerate every model with a rate-controlled
well and check each — the same gap may hide elsewhere. This is a deliberate
re-baseline of a verified fix, not a tolerance waiver.

---

## 3. Findings the external review caught that v2 missed

| # | Finding | Evidence | Status |
|---|---|---|---|
| E1 | `2ph_comp_solid` must be compared against `development` | §2 — confirmed MR-caused | **Correction accepted** |
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

Verification: 98/101 model suite, 259 unit tests, bit-identical A/B on every
migrated model and both refactors of the numerical core.

### M3.5 — corrections arising from this review (new, do first)

1. **Regenerate `2ph_comp_solid` references** and document the drift (§2); first
   enumerate every rate-controlled-well model and check for further gaps.
2. **Region-aware properties in the IPR hook** (E2) — resolve the property
   container from the connected block's region instead of `[0]`. Same audit for
   every other `property_containers[0]` in `darts/pipes/`.
3. **Vectorize the Colebrook friction solve** (E4) — replace the per-face
   `fsolve` loop with a batched Newton/Halley iteration on the Colebrook
   residual, or an explicit correlation, proving bit-comparability on the CI
   models. This is a prerequisite for any scaling claim about `Pipe`.
4. **Fix the ramp schedule's time evaluation** (E3) — evaluate at `t_{n+1}` or
   integrate over `[t_n, t_{n+1}]`, then delete the `first_ts_size < 0.01 s`
   assertion.

### M4 — contract hardening (revised)

Adds E5–E9 to the previously planned removal work: the stencil-declaration phase
with pattern versioning; selector/law separation; additive vs replacement typing;
the observer type; the restart contract. Then remove `set_rhs_flux` and
`rhs_flux_hooks` (now that `models/` is clean), retire the legacy BC dict schema,
and switch on the placement-policy CI check.

### M5 — backend lowering and well laws

IPR becomes `well.add_perforation(..., flow_law=LinearIPR(...))` assembled
engine-side — **with no dummy zero-WI perforation**, because M4's stencil
declaration makes it unnecessary — gaining exact derivatives, GPU and adjoint
support. Semi-analytic lateral heat becomes a well segment heat law returning rate
*and* derivatives. Native CPU/GPU kernels for the cell-source and connection-flux
item types. Adjoint support for declared items; until then the bind-time guard
stands.

### M6 — inclusion gate

A feature enters core only with: equation/reference tests, an independent public
integration case, derivative verification, CPU ST/MT and GPU results, scaling
measurements, applicability and failure-mode documentation, and a stable
non-model-specific API. Applies to the choke on re-admission (restoring the
deleted 310-line equation tests), to the new drift-flux correlations, and to the
lateral-heat model.

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

1. `2ph_comp_solid` reference regeneration and a full audit of the well-control
   fix's blast radius (§2).
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
