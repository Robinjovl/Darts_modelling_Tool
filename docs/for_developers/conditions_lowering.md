# Lowering the conditions layer, and the inclusion gate

This page is the design record for the two phases of the MR !305 refactoring that
have **not** landed: **M5**, moving the closed condition item types off the
per-Newton Python callback, and **M6**, the gate a feature must pass to enter the
package. It is a plan, not a description of shipped behaviour — for what the
layer does today see [Conditions](../technical_reference/conditions.md).

It is deliberately grounded in measurements and in named functions rather than in
principle, because the main conclusion is that **most of the lowering is not worth
doing**, and that conclusion only holds up with numbers behind it.

---

## 1. What the per-Newton Python cost actually is

Three measurements, all on this tree.

**(a) In a real model.** `models/dfm_well/2ph_2comp_isothermal_dfm_vertical_well_vs_olga`
in its `ipr_volumetric` variant — the CI case with the heaviest condition item in
the repository, `LinearDFMWellIPRHook`, which finite-differences its flux over
`2 * n_vars` perturbed property evaluations per connection per iteration:

| | time | share |
|---|---|---|
| whole run | 7.60 s | 100 % |
| `ConditionSet.apply` (517 Newton iterations) | 0.74 s | **9.7 %** |
| OBL well interpolation → point generation | 3.58 s | 47 % |
| `dfm_well_velocity_calculation` (Python drift flux) | 3.20 s | 42 % |

1.43 ms per Newton iteration. The conditions layer is not the bottleneck of the
models that use it; the OBL supporting-point generation and the Python drift-flux
closure are, by four to five times.

**(b) `CellSource`, scaled.** Synthetic: one item over *n* blocks of a fake
block-CSR matrix, `apply()` timed in a loop.

| n contributions | RHS only | with `d_rates` | vectorized scatter (proposed) |
|---|---|---|---|
| 10² | 0.003 ms | 0.264 ms | 0.005 ms |
| 10⁴ | 0.158 ms | 26.6 ms | 0.506 ms |
| 10⁶ | 18.1 ms | **2626 ms** | **62.0 ms** |

The RHS path is vectorized and costs ~18 ns per contribution — fine at any size
this simulator reaches. The Jacobian path is a **per-cell Python loop**
(`CellSource.apply`, `for k, pos in enumerate(self._diag_pos): ctx.jac.add_block(...)`)
and costs ~2.6 µs per contribution, 145× more. Review item E10 asks for an
assertion that the Python callback count does not grow per cell; today it grows
per cell *inside a shipped item type*.

**(c) `InterfaceFlux`, scaled.** Same method, a linear two-block exchanger:

| n connections | per iteration | per connection |
|---|---|---|
| 10² | 1.34 ms | 13.4 µs |
| 10⁴ | 138 ms | 13.8 µs |
| 2·10⁵ | 2805 ms | 14.0 µs |

Flat per connection, because `InterfaceFlux.apply` is a Python loop that calls the
subclass's `connection_flux()` once per connection and allocates three small numpy
arrays each time. This is the one item type whose cost is structural.

**What this says.** The 42× available from rewriting eight lines of numpy in
`CellSource.apply` is larger than anything native lowering would add on top of it,
and it needs no C++, no ABI, no GPU staging and no new failure mode. That is the
first thing to do and, for `CellSource`, very likely the last.

---

## 2. Native lowering (M5)

### 2.1 Where a native contribution would go

The CPU flow assembly is `engine_super_cpu<NC, NP, THERMAL>::assemble_jacobian_array`
(`engines/src/engine_super_cpu.tpp`). It fills `RHS` and the block-CSR `Jac` in one
pass over blocks, then ends with

```cpp
for (ms_well *w : wells)
  if (w->control.get_well_control_type() > ... NONE)
    w->add_to_jacobian(dt, X, jac_well_head, RHS);
```

A packed cell-source pass belongs **exactly there** — after the block loop, before
the return — for three reasons: the diagonal blocks it writes are already
initialized by then; it is outside the OpenMP region, so it needs no thread
partition of its own; and it is the same place, and the same shape, as the one
existing non-Darcy contribution the engine already assembles.

The proposed C++ surface mirrors the well one:

* `engine_base`: `std::vector<index_t> cell_source_blocks;` and
  `std::vector<value_t> cell_source_rates;` (`n_cells * n_vars`, equation-major),
  plus `cell_source_ders` (`n_cells * n_vars * n_vars`) when derivatives are given;
  `void engine_base::apply_cell_sources(value_t dt, csr_matrix_base *jacobian, std::vector<value_t> &RHS);`
* `virtual bool supports_cell_sources() const { return false; }`, overridden `true`
  by the engines that implement it, and checked at `init_base()` so an engine that
  cannot assemble a declared source **throws instead of dropping it**.

That refusal pattern is not hypothetical: `engine_base::build_perforation_flow_laws()`
(landing with the native perforation flow law, §2.6) already does exactly this —
`supports_perforation_flow_laws()`, a loud throw naming the engine, a duplicate-pair
check and a count check against the frozen connection arrays. The cell-source pass
should be a copy of that structure, not a new one.

Python side: `CellSource.bind()` already computes everything this needs. It would
additionally publish the arrays through a pybind setter and set a flag; `apply()`
becomes "refresh the rate array if the rate is a callable, otherwise do nothing".

### 2.2 Does the compiled item data survive the trip to C++?

This was the stated design intent of the closed types. **For `CellSource` it holds
up; for `InterfaceFlux` it does not, yet.**

`CellSource` after `bind()` holds `self.cells` (`int64`, contiguous),
`self._rhs_idx` (`int64`, contiguous), `self._diag_pos` (`int64`, contiguous) and,
when constant, `rates` / `d_rates` as C-contiguous `float64` arrays of exactly the
shapes the C++ pass wants. Nothing has to be repacked; a pybind `py::array_t`
binding takes them as-is. The one gap is that a *callable* rate is re-evaluated
every iteration, which §2.3 is about.

`InterfaceFlux` after `bind()` holds `self.connections` as a **Python list of
tuples** and `self._positions` as a **Python list of 4-tuples of Python ints**.
Neither is handable to C++ and neither is usable by a vectorized numpy path
either. Before any lowering, these must become `(n_conn, 2)` and `(n_conn, 4)`
int64 arrays, and `connection_flux(t, state_row, state_col)` must gain a batched
sibling — `connection_fluxes(t, states_row, states_col)` returning
`(n_conn, n_vars)` and two `(n_conn, n_vars, n_vars)` arrays — with the scalar form
kept as a default implementation that loops, so existing subclasses keep working.

That batching is worth doing **on its own merits**: it removes the per-connection
Python call from the measurement in §1(c) without any C++ at all, and it is the
prerequisite for both the numpy and the native paths. It is also the missing half
of review item E7 — the batched signature *is* the law object a backend kernel
consumes, and it can exist before the kernel does.

`PipeSourceTerm` and `SegmentSource` need no work of their own: the first is a
single block, the second is a `CellSource` with a resolved index range.

### 2.3 The hard case: a state-dependent rate

`CellSource(rates=f(t, states))` is a Python callable. Three options, and only one
of them is the answer.

**Option A — lower only constant and time-dependent rates.** `f(t)` is evaluated
once per timestep on the host and the resulting `(n_cells, n_vars)` array is
handed to C++ (or to the device); `f(t, states)` stays on the Python path. This is
cheap, it is honest, and the item already knows which case it is in
(`_callable_takes_states`). It covers every `CellSource` in this repository and,
by inspection of the migrations, every legacy `set_rhs_flux` override that was
converted. **Do this one.**

**Option B — express the state dependence as an OBL operator.** The engine already
has a native, GPU-capable, adjoint-visible channel for a state-dependent volumetric
source: the kinetic operator. In the assembly,

```cpp
RHS[i * N_VARS + c] += (PV[i] + RV[i]) * dt * op_vals_arr[i * N_OPS + KIN_OP + c] * kin_fac[i];
Jac[diag_idx + ...] += (PV[i] + RV[i]) * dt * op_ders_arr[(i * N_OPS + KIN_OP + c) * N_VARS + v] * kin_fac[i];
```

`KIN_OP` carries the state dependence through the interpolator (so it is
differentiated exactly, runs on GPU, and is seen by the adjoint) and
`mesh.kin_factor` is a per-cell host multiplier that doubles as a selector — zero
outside the chosen cells. A rate of the separable form `q(t, x) = a(t, cell) · K(x)`
maps onto it exactly. This is not a new mechanism to build; it is channel **P** of
the four-channel taxonomy, and the right answer for a term of that shape is to
tell the author to put it there rather than to build a second machine.

Its real limits, which must be documented rather than discovered: `KIN_OP` is
scaled by `(PV + RV)`, so an absolute rate needs `kin_factor = q / (PV + RV)`; the
term is applied to reservoir blocks only (`if (i < n_res_blocks)`), so it cannot
carry a well-segment source; the operator slot is shared with actual chemistry;
and on GPU `mesh_kin_factor_d` is copied **once**, in `engine_super_gpu::init`, so
a per-timestep host update to `kin_factor` never reaches the device. That last one
is a one-line fix (re-copy in the assembly when a dirty flag is set) and should be
made before Option B is recommended to anyone on GPU.

**Option C — call back into Python from C++.** Rejected. It would keep the GIL in
the assembly loop, so it cannot be parallel on CPU and cannot exist at all inside a
CUDA kernel; it would reintroduce, in a worse place, exactly the per-contribution
Python cost the lowering exists to remove; and it would leave the adjoint replay
(§3) still unable to reproduce the contribution, because the backward loop runs
without a Python interpreter frame it could call from. A native kernel that calls
Python once per cell is a Python loop with extra steps.

### 2.4 What GPU support additionally requires

The RHS half already works: `DartsModel.apply_rhs_flux()` writes the host `RHS`,
which `engine_base_gpu::assemble_linear_system` had copied to the host after the
kernel, and the Newton loop pushes it back (`copy_data_to_device(engine.RHS,
engine.get_RHS_d())` in `darts/nonlinear_solvers/newton.py`). That round trip is
the reason RHS-only items are allowed on GPU today.

The Jacobian half does not, and the reason is structural: **the device matrix is
the authority.** `Jacobian->values_d` is written by the assembly kernel and read
directly by the linear solver (`linear_solver->solve(RHS_d, dX_d)`); the host
`Jacobian->values` is copied back only under `print_linear_system`. A host-side
block write is therefore not "not synced" — it is written to a buffer nothing
reads.

Two things are needed, in this order:

1. **Staged arrays.** `cell_source_blocks`, `cell_source_rates`,
   `cell_source_ders` and the precomputed `diag_ind` offsets allocated on the
   device at `init()` and refreshed per timestep only when a callable produced new
   values. A rate that is constant over the run is uploaded once.
2. **A device scatter.** The template already exists and is worth copying rather
   than inventing: `engine_super_gpu::assemble_jacobian_array` computes the well
   head blocks **on the host**, packs them into `jac_wells`, uploads them
   (`copy_data_to_device(jac_wells, jac_wells_d)`) and then scatters them into
   `values_d` with `copy_data_within_device`. A cell-source pass is the same
   pattern with `+=` instead of overwrite, which means a small kernel rather than
   a memcpy — `n_cells * n_vars * n_vars` independent adds, no reduction, no
   atomics needed as long as the item's cell set is unique (which `bind()` can
   check once).

Nothing about this requires exposing the device Jacobian to Python, and it should
not: the Python side stays a producer of packed arrays.

### 2.5 Ordering, and what is not worth doing

| # | Step | Cost | Value | Verdict |
|---|---|---|---|---|
| 1 | Vectorize the diagonal scatter in `CellSource.apply` (flat index array built in `bind()`, one `jac_vals[idx] += ...`) | ~10 lines, pure numpy | 42× at 10⁶ contributions; removes the per-cell Python loop E10 objects to | **Do first** |
| 2 | Flatten `InterfaceFlux._positions` / `.connections` to int64 arrays and add the batched `connection_fluxes()` | small, additive, scalar form kept | removes the per-connection Python call; unblocks everything else; delivers E7's law half | **Do** |
| 3 | The E10 validation matrix as tests: the three scaling points, and an assertion that callback count is O(1) in cells and faces | small | this is what stops step 1 regressing | **Do** |
| 4 | Fix the adjoint guard's evaluation time (§3.1) | one line | closes a guard that does not currently fire in the real driver | **Do — it is a correctness hole, not an optimization** |
| 5 | Native CPU `apply_cell_sources` for constant / `f(t)` rates | moderate C++, pybind, capability query | 62 ms → ~1 ms at 10⁶; ~0 on every model in this repository | **Only when a model needs 10⁵+ sources.** After step 1 there is no in-repo case that justifies it |
| 6 | GPU staging + scatter for the same | larger; device lifetime, refresh policy | unlocks GPU for Jacobian-carrying sources | **Only with step 5, and only with a GPU model that needs it** |
| 7 | Native `InterfaceFlux` kernel | large: the flux law itself is user code | after step 2, the residue is the user's own numpy | **Not worth doing.** The generic case cannot be lowered without lowering the user's law with it; a specific law (linear IPR) is better lowered as a well flow law — §2.6 |
| 8 | A callback into C++ from Python per contribution (Option C) | — | — | **Do not do** |

The honest summary: steps 1–4 are worth doing now and cost days; steps 5–6 are
worth *designing* now and building when a model demands them; step 7 should not be
built as a generic facility at all.

### 2.6 The one piece that is genuinely native: the perforation flow law

Landing concurrently (`ms_well::set_perforation_flow_law`, `perforation_flow_law`,
`engine_base::build_perforation_flow_laws`, `engine_super_cpu::add_perforation_flow_law`)
is the native form of the linear IPR — a POD law struct attached to a perforation,
snapshotted at `init()`, resolved to its two mesh connections with a loud failure
on a mismatch, and assembled inside the connection loop with analytic derivatives
from the operators rather than the Python hook's `2 * n_vars` finite differences.

That is the right shape, and it is the shape the rest of M5 should follow: a value
type describing the law, a capability query on the engine, a resolution pass in
`init_base()` that verifies against the frozen arrays, and a refusal — never a
silent omission — when the engine or the driver cannot honour it. It also resolves
step 7 above: the interesting `InterfaceFlux` in this repository is a well flow
law, and it is better served as one.

---

## 3. The adjoint gap

### 3.1 The guard does not currently fire

`ConditionSet.compile()` refuses a non-`adjoint_transparent` item when
`engine.opt_history_matching` is set. `compile()` runs once, at the end of
`DartsModel.init()`. The optimization driver sets the flag in
`OptModuleSettings.make_opt_step_adjoint_method()` — **after** `init()` has already
returned — and then calls `DartsModel.reset()`, which bumps the pattern version and
drops the cached CSR view but **does not re-run `compile()`**.

So in the driver's own flow the guard is evaluated while the flag is still false,
and a model that registers an opaque item and then history-matches it runs to
completion with a silently incomplete gradient. The guard only bites if a user sets
the flag before `init()`.

The fix is to re-evaluate the adjoint check where the flag can be observed — in
`reset()`, or at the first `apply_rhs_flux()` of a run — rather than only at
compile time. This is a correctness hole in M4 and should be closed before M5
starts, independently of any lowering.

### 3.2 What differentiating an item would actually require

`engine_base::calc_adjoint_gradient_dirac_all()` walks the stored trajectory
backwards. At each stored step it re-evaluates the operators at `X_t[n]`, calls
`assemble_jacobian_array(dt, X, Jacobian, RHS)` to rebuild `∂g/∂x`, calls
`adjoint_gradient_assembly(...)` to fill `dg_dx_T` (its transpose), `dg_dx_n`
(`∂g/∂x_{n-1}`) and `dg_dT_general` (`∂g/∂θ`), solves `dg_dx_T λ = dj/dx`, and
accumulates `λᵀ ∂g/∂θ`.

Nothing in that loop touches Python. So an item is differentiated properly only
when all three of these hold:

1. **`dg_dx_T` contains its blocks.** The backward loop rebuilds the Jacobian from
   C++ alone, so the item's four (or one) blocks are simply absent and λ is the
   solution of the wrong linearized system. There is no way to supply them from
   Python: the loop has no per-step hook, and adding one would be Option C in a
   worse place. **Native lowering is therefore a prerequisite for adjoint support,
   not a parallel workstream.**
2. **`dg_dx_n` is right.** No shipped item depends on `x_{n-1}`, so its
   contribution is zero and the existing matrix is already correct. This one is
   free — but it must be *declared*, because a future stateful item would break it
   silently.
3. **`dg_dT_general` has a column for the item's parameters.** The gradient vector
   is indexed by mesh interfaces (`n_interfaces = mesh->n_conns / 2` in
   `engine_base::init_adjoint_structure`'s caller), with the well-head entries
   erased afterwards. An item whose parameter is a control — a productivity index,
   a `UA`, a rate — has no column at all, so `∂J/∂θ` for that parameter is
   identically zero rather than wrong-but-present.

There is a fourth hazard that is specific to M4 and easy to miss: **`declare_stencil()`
changes `n_interfaces`.** A declared-but-absent coupling is added to the mesh as a
zero-transmissibility connection, which increases `mesh->n_conns` by two and hence
`n_interfaces` by one. The transmissibility gradient vector, and the `col_dT_du`
mapping built in `opt_module_settings.py` against the reservoir connection list,
are both positional. A stencil-declaring item under history matching therefore
shifts the gradient layout even if the item itself contributes nothing. Any M5
adjoint work must either exclude declared connections from the interface numbering
or renumber `col_dT_du` — and until it does, this is an independent reason the
guard of §3.1 must actually fire.

### 3.3 Which item types can realistically get there

| Item | Can it be adjoint-transparent? |
|---|---|
| `CellSource`, constant or `f(t)`, no `d_rates` | **Already, in the strict sense** — its residual contribution is independent of `x`, `x_{n-1}` and (unless the rate is itself a control) of `θ`, so it contributes nothing to `dg_dx_T`, `dg_dx_n` or `dg_dT_general`, and the adjoint linearizes about the stored trajectory, which already includes its effect. This is provable per item and is the one case where `adjoint_transparent = True` can be set today, guarded by "the rate is not a control variable and no stencil was declared". Worth doing: it is the difference between "history matching works with a prescribed aquifer influx" and "it refuses". |
| `CellSource` with `d_rates`, `InterfaceFlux` | Only after §2 lowers them into `assemble_jacobian_array`, and then only with a matching addition to `adjoint_gradient_assembly`. Mechanical once lowered; impossible before. |
| `PipeSourceTerm` | Same as the first row, plus the ramp's own time integration. The rate is a per-timestep constant, so the same argument applies. |
| `DirichletPin(mode="state")` | **No, and it should stay refused.** The projection is invisible to the assembled system by construction, so the adjoint's forward linearization does not describe the forward map that was actually run. `mode="row"` is differentiable in principle (a claimed row is a linear constraint) but nothing needs it yet. |
| `ConstantStateBC` | Trivially — it contributes nothing to anything. It should declare `adjoint_transparent = True` rather than inherit the default. |
| The perforation flow law (§2.6) | Its own C++ already refuses under `opt_history_matching`, which is the correct interim state. It is the best candidate for the first real adjoint extension, because it is native and its derivatives are analytic. |

---

## 4. The inclusion gate (M6) — an assessment, with the inclusion decision pending

> **Status: ASSESSMENT, not an applied decision.** This section defines the gate
> and audits three features against it. The audit finds that two of the three —
> the drift-flux correlations and the lateral-heat model — currently **fail**
> gates while remaining in `darts/` with CI variants. The gate's own rule
> ("not yet" means model-side) has therefore **not been enforced**: enforcing it
> would mean either closing the recorded gaps or evicting shipped, CI-covered
> features, and that trade-off is a maintainer decision, not something this
> branch decides unilaterally. §4.4 states exactly what is unmet per feature and
> what evidence would close it. Until that decision is recorded, read
> §§4.1–4.3 as audit results, not as passed inclusion decisions.

A feature enters `darts/` when a reviewer can tick every box. "Not applicable" is
an allowed answer; **"not yet" is not** — it means the feature stays model-side.

| # | Gate | Evidence a reviewer must be able to open |
|---|---|---|
| **G1** | **Equation tests.** The closed-form content of the model is checked against hand-computed values, published limiting cases, or an independent implementation — not against its own previous output. | A test file under `tests/`, with the source of each expected number named in the test |
| **G2** | **Independent public integration case.** A model in `models/` that reproduces a published or otherwise externally-owned result, with the reference data *in the repository*. A CI case that compares against its own regenerated pickle satisfies non-regression, not validation. | The model directory, the reference data file, and the comparison in `main.py` or the test |
| **G3** | **Derivative verification.** Every analytic derivative the feature contributes is compared against finite differences, at a stated tolerance, in a test. A feature that lags a derivative says so and shows the convergence cost. | A test asserting a relative error bound |
| **G4** | **Platform results.** CPU single-thread, CPU multi-thread and GPU, or an *enforced* restriction (`requires_platform`, a `supports_*()` capability query, a raise at `init()`) plus the reason. Silence is a fail. | CI job coverage, or the refusing code path plus a test that it refuses |
| **G5** | **Scaling.** Timings at ~10², 10⁴ and 10⁶ contributions (or the feature's equivalent size axis), plus an assertion that the Python callback count does not grow with the problem size. | A benchmark or test recording the numbers |
| **G6** | **Applicability and failure modes, documented.** Where the correlation or model is valid, what it does outside that range, and what it does at each degenerate input (zero flow, absent phase, reverse flow, density inversion, vanishing denominator). | A module docstring or a page under `docs/`, and a guard test per named failure mode |
| **G7** | **Stable, non-model-specific API.** No parameter that exists only for one model; no `NotImplementedError` on an advertised path; one way to construct it; the four-channel placement (W / P / D / R) stated and correct. | The public signature, and a consumer that is not the model it was written for |
| **G8** | **An in-repo consumer.** The package does not carry code nothing in the repository calls. | The consumer, and the placement-policy check passing |

### 4.1 The choke boundary

Currently **outside** the repository: `darts/pipes/upstream_pressure_node_with_choke.py`
(3480 lines) was removed in M0 as unconsumed and untestable against the current
physics, and its 310-line equation-test file was deleted before that, in `aa899c8c`.

| Gate | Status |
|---|---|
| G1 | **Was partly met, then deleted.** The removed `tests/pipes/test_choke_models.py` held eight equation tests of real quality — the Perkins critical-pressure ratio against the isentropic limit, the pure-liquid limit against the incompressible orifice relation, the A30 root as a stationary point of A28, the SINTEF-HEM incompressible limit, Rathjen–Straub surface tension against a CO₂ reference. This is the single most recoverable item in this whole section: it is in git history. |
| G2 | **Missing.** No integration case at all; the choke model directory was `1ph_1comp_thermal_dfm_well_vs_olga_..._and_choke`, which is not public reference data. |
| G3 | **Missing.** No finite-difference check of the choke's contribution to the Jacobian. |
| G4 | **Missing.** CPU-only by construction; never stated, never enforced. |
| G5 | **Missing**, though a choke is O(1) contributions and this should be a one-line "not applicable, one contribution per choke". |
| G6 | **Missing and load-bearing.** V3 of the review: `RECOVERY='ON'` selects the unphysical low-pressure root and never applies the critical-flow cap. E11: `CHISHOLM` is offered as an option and raises. |
| G7 | **Missing.** The API is a `Pipe`-specific class, not a condition item or a well law. |
| G8 | Fails today by definition — that is why it was evicted. |

**To re-admit it**, in order: restore `tests/pipes/test_choke_models.py` from
`aa899c8c^` and make it pass against the current physics API (G1); fix the
`RECOVERY='ON'` root selection and either implement or delete `CHISHOLM` (G6);
express the model as a condition item or a well flow law rather than a `Pipe`
attachment (G7); add a CI model with the choke in the loop (G8) and a
finite-difference Jacobian test (G3); state and enforce CPU-only (G4). G2 — an
independent public case — is the expensive one and is what decides whether this
belongs in `darts/` at all.

### 4.2 The three drift-flux correlations

`tang_2019`, `bhagwat_ghajar_2014` and `bai_2023`, in `darts/pipes/drift_flux.py`,
with CI variants of `2ph_2comp_isothermal_dfm_vertical_well_vs_olga`.

| Gate | Status |
|---|---|
| G1 | **Partial.** `tests/pipes/test_drift_flux_closures.py` has 55 tests, and they are good tests — but they are *algebra, guard and self-consistency* tests: the C₀·s_G clamp, the density-inversion fallback, the Cmax extrapolation clamp, the inclination rejection, Bai's four documented deltas from Bhagwat–Ghajar. **Not one of them compares a closure against a number from the paper it implements.** For three correlations whose entire content is empirical fits, that is the gap that matters. |
| G2 | **Missing.** The `vs_olga` models are named for a comparison made outside this repository; there is no OLGA data in the tree, and CI compares each variant against its own `ref/perf_lin_*_<variant>.pkl`. That proves the variants do not drift; it does not validate them. |
| G3 | **Missing, and cheap to fix.** `Pipe` has two differentiation paths — `diff_method="OBL"` (chain rule through the property interpolator, the default) and `diff_method="numerical"`. No test and no model sets `diff_method`, so the two paths are never compared. One test per closure asserting that `eval_phase_vels_and_ders` agrees between them to a stated tolerance would satisfy G3 outright. |
| G4 | **Partial.** DFM wells are CPU-only and that is enforced by the engine, but it is enforced as a property of DFM wells, not declared per closure. No multi-thread result is recorded. |
| G5 | **Missing.** M3.5 measured the friction solve (~6.5× at 21 interfaces, ~140× at 10⁴), which is the right kind of number but only for one component of one closure. |
| G6 | **Partial, and much improved by M1a.** The inclination restriction (V4), the density-inversion guard (V5), the C₀ bound (V6) and the Cmax clamp (V7) are implemented, tested and warned about. What is not documented is the *validity envelope of each correlation as published* — the pipe diameters, fluids, flow regimes and inclinations the fits were made from. |
| G7 | **Met.** `make_drift_flux_closure()` with a registry, per-closure keyword validation, `default_friction_model`, and closures usable standalone (`test_bare_closure_matches_pipe`). This is the strongest part of the feature. |
| G8 | **Met** — three CI variants. |

**To close it**: a data table per correlation, taken from the source paper (or its
digitized figures) with the citation in the test, asserting the closure reproduces
it within the paper's own stated scatter (G1/G2); the OBL-vs-numerical derivative
agreement test (G3); the published validity envelope in the module docstring, with
a warning or refusal outside it (G6); and one scaling point at 10⁴ interfaces (G5).
Until G1/G2 are done, these are three *implementations of published correlations*
with no evidence they reproduce them — which is exactly the maturity level the
review objects to shipping in core.

### 4.3 The semi-analytical lateral heat model

`SemiAnalyticalWellLateralHeatTransfer` and its item
`SemiAnalyticalWellLateralHeatTransferHook`, in
`darts/pipes/add_lateral_heat_exchange.py`, with the `lateral_heat` CI variant.

| Gate | Status |
|---|---|
| G1 | **Met.** `tests/pipes/test_lateral_heat_exchange.py` checks the Chiu–Thakur rate against a hand computation, the three Zhang time-function branches, the Ramey early-time warning, and the constructor's exclusivity rules. |
| G2 | **Missing.** No comparison against Ramey's own worked example, against Chiu–Thakur's finite-radius solution, or against a published wellbore-temperature dataset. |
| G3 | **Met for PT.** `test_conductance_matches_finite_difference_of_evaluate` covers every time function. **Absent for PH**, where the item sets `provides_jacobian = False` and lags the temperature — correct behaviour, but its convergence cost is unmeasured. |
| G4 | **Partial.** CPU-only, enforced (the item writes the Jacobian, `compile()` refuses off-CPU). No MT or GPU result; the GPU restriction is real and enforced, so this can be a documented "not applicable" once written down. |
| G5 | **Missing.** One contribution per well segment — an O(n_segments) numpy path, so the number will be small, but it is not recorded. |
| G6 | **Met, and it is the model of what G6 should look like.** The module docstring states Ramey's ≳7-day validity, names Chiu–Thakur's early-time correction, says the Zhang option reproduces the instantaneous OGS rate but *not* the full time-convolution superposition, and lists what is not implemented (WHAP, pressure drop, Willhite U). |
| G7 | **Partial.** `well_layers_props` is an advertised constructor argument that raises `NotImplementedError` — a G7 fail by the letter of the gate. Either implement Willhite's U or remove the parameter. |
| G8 | **Met** — the `lateral_heat` CI variant. |

**To close it**: remove or implement `well_layers_props` (G7); one published
verification case — Ramey's worked example is short enough to be an equation test
and would satisfy G1 more strongly and G2 partially (G2); a PH convergence-cost
measurement next to the PT one (G3); a segment-count scaling point (G5); and write
down that GPU is out of scope because the item writes the Jacobian (G4).

Of the three features, this is the closest to the gate: it fails on two items, one
of which is a deletion.

### 4.4 Maintainer decision required

The audit above records "not yet" for gates of two features that are in core
today. Per the gate's own rule they would stay model-side; per the shipped tree
they are in `darts/` with CI variants. Neither state is wrong by itself — the
contradiction is. The decision, per feature:

**Drift-flux correlations (`darts/pipes/drift_flux.py` — `tang_2019`,
`bhagwat_ghajar_2014`, `bai_2023`; CI variants on
`2ph_2comp_isothermal_dfm_vertical_well_vs_olga`)**

*Unmet:* G1 (no test compares any closure against a number from its source
paper), G2 (no externally-owned reference data in the tree; the `vs_olga` CI
variants compare against their own regenerated pickles), G3 (the `"OBL"` and
`"numerical"` derivative paths of `Pipe` are never compared), G5 (no scaling
measurement beyond the friction-solve component). *Partial:* G4 (CPU-only is
enforced for DFM wells as a whole, not declared per closure; no multi-thread
result recorded), G6 (guards implemented and tested, but the published validity
envelope of each fit is not documented).

*Decide:* keep in core **conditional on closing G1/G3 (cheap, see below)** with
G2 tracked as follow-up work, or move the closures model-side until G1/G2 exist.

**Lateral heat (`darts/pipes/add_lateral_heat_exchange.py` —
`SemiAnalyticalWellLateralHeatTransfer(Hook)`; `lateral_heat` CI variant)**

*Unmet:* G2 (no published verification case — no Ramey worked example, no
Chiu–Thakur finite-radius comparison, no wellbore-temperature dataset), G5 (no
segment-count scaling point). *Partial:* G3 (met for PT; the PH lagged-Jacobian
convergence cost is unmeasured), G4 (the GPU restriction is real and enforced
but not written down as a "not applicable"), G7 (`well_layers_props` is an
advertised constructor argument that raises `NotImplementedError`).

*Decide:* keep in core **conditional on the G7 deletion and the G2 equation-test
closure (both cheap, see below)**, or move it model-side.

**Choke** — already outside the repository; §4.1 is its re-admission
checklist. No decision needed now.

**What can be closed cheaply, from inside this repository** (actionable, no new
machinery — each item is a test or a docstring):

1. *Drift-flux G1:* the closures' own docstrings and the v3 review already cite
   concrete published numbers (the Bhagwat–Ghajar `C0` bound and flow-pattern
   coefficients, Tang's parameter sets, Bai's four documented deltas from
   Bhagwat–Ghajar). Pin each closure against those citable values in
   `tests/pipes/test_drift_flux_closures.py`, naming the paper, table/equation
   and tolerance per assertion. Full digitized-figure tables would be better
   still, but the pinned-constants form already converts "no number from any
   paper" into a named external anchor.
2. *Drift-flux G3:* one test per closure asserting
   `eval_phase_vels_and_ders` agrees between `diff_method="OBL"` and
   `diff_method="numerical"` to a stated tolerance. Both paths exist; nothing
   compares them today.
3. *Drift-flux G5 / lateral-heat G5:* one benchmark-style test each recording
   timings at the feature's size axis (10⁴ interfaces for the closures,
   segment count for lateral heat), plus the E10 assertion that the Python
   callback count is O(1) in problem size — the same three-point method §1
   already used, so the harness exists.
4. *Drift-flux G6:* copy each paper's stated validity envelope (diameters,
   fluids, inclinations, flow regimes) into the module docstring, with a
   warning outside it — the lateral-heat docstring is the in-repo model of
   what this looks like.
5. *Lateral-heat G7:* delete the `well_layers_props` parameter (or implement
   Willhite's U; deletion is the cheap branch).
6. *Lateral-heat G2/G1:* Ramey's worked example is short enough to be an
   equation test; add it with the citation.
7. *Lateral-heat G3 (PH):* measure the PH lagged-temperature convergence cost
   next to the existing PT finite-difference test and record the iteration
   delta.
8. *Lateral-heat G4:* one sentence in the docstring stating GPU is out of scope
   because the item writes the Jacobian, referencing the enforced `compile()`
   refusal.

*Not cheap, and the real decision drivers:* drift-flux G2 (externally-owned
integration data in the tree — OLGA exports or published profiles with
redistribution rights) and lateral-heat G2 beyond the Ramey example. If those
are judged indispensable for core, the features go model-side until the data
exists.

---

## 5. Summary

* The conditions layer's per-Newton Python cost is real but is **not** the
  bottleneck of any model that uses it (9.7 % against 47 % and 42 % elsewhere).
* The largest single win available — 42× at 10⁶ contributions — is a numpy rewrite
  of `CellSource.apply`'s diagonal scatter, not a native kernel.
* `CellSource`'s compiled data is already C++-ready; `InterfaceFlux`'s is not, and
  flattening it plus a batched flux signature is both the prerequisite for lowering
  and the delivery of E7's remaining half.
* State dependence should go through the OBL operators (channel P), not through a
  callback from C++ into Python.
* Native lowering is a **prerequisite** for adjoint support, and the adjoint guard
  that stands in for it does not currently fire in the optimization driver's own
  call order — that is the one item in this document that is a bug rather than a
  plan.
* Of the three features the M6 gate governs, the lateral-heat model is close, the
  drift-flux correlations lack any validation against the papers they implement,
  and the choke is outside the repository with its best evidence recoverable from
  git history. The gate itself is an **assessment whose inclusion decision is
  pending** (§4.4): drift-flux and lateral heat remain in core while failing
  gates, and resolving that — close the gaps or move them model-side — is an
  explicit maintainer decision.
