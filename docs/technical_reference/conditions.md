# Conditions

`model.conditions` is the unified Python-side layer for source terms, interface
fluxes and prescribed states: everything a model needs to add to the assembled
system that is neither a well nor a constitutive property. It is the **only**
such channel: the ad-hoc `DartsModel.set_rhs_flux()` override and the untyped
`model.rhs_flux_hooks` list it replaced are **removed** — the base class defines
neither name, and a model that still carries one is refused at `init()` rather
than left to run without its source term (see {ref}`conditions-migration`).

The classes live in `darts.models.conditions`; the generated API reference is
in the *Conditions* section of the API page.

```python
from darts.models.conditions import CellSource

class Model(CICDModel):
    def set_boundary_conditions(self):
        # a constant CO2 stream injected into cell 0, in the residual units of
        # each equation: kmol/day per component, kJ/day for energy
        self.conditions.add(CellSource(cells=[0], rates=self.injection_rates))

    def injection_rates(self, t: float) -> np.ndarray:
        rates = np.zeros((1, self.physics.n_vars))
        rates[0, 1] = 19631.9        # kmol/day of CO2, positive INTO the cell
        rates[0, -1] = -2000 * 19631.9   # kJ/day carried by that stream
        return rates
```

---

## Where a term belongs: the four channels

A term added to the discrete system enters through exactly one of four
channels. Picking the right one is the first design decision, and the
conditions layer is deliberately only the last of them.

| Channel | What it carries | Where it lives |
|---|---|---|
| **W — Wells** | Physical wells: controls, perforations, well constraints | `reservoir.add_well()` / `add_perforation()`, `physics.set_well_controls()` |
| **P — Physics operators** | Constitutive sources: mass/energy source terms of the fluid, kinetic reaction rates | `PropertyContainer` evaluators (`energy_source_ev`, `evaluate_mass_source`, kinetic rates) |
| **D — Declarative boundary types** | The `a·p + b·f = r` Robin *type* of a boundary face | `darts.reservoirs.boundary_spec` (mech/MPFA family), reservoir discretization |
| **R — Runtime conditions** | Everything else, evaluated per Newton iteration | `model.conditions` — this page |

**Wells and constitutive sources are outside this API on purpose.**

* A well is a subsystem, not a source: it owns control equations, its own
  unknowns, an ordering in the block layout, and rate/BHP switching. Faking one
  with a Python source loses all of that. Conversely, a "well" that exists only
  to impose a face boundary should become a face boundary, not a condition item.
* A constitutive source belongs in the operators because that is where it is
  differentiated. Operator contributions are interpolated by the OBL machinery,
  run on CPU **and** GPU, and are seen by the adjoint. A Python contribution is
  none of those things (see {ref}`conditions-limits`). If a term is a function of the
  local state through the fluid model, it is a property, not a condition.
* A boundary **type** changes the discretization stencil, the transmissibilities
  and the Jacobian sparsity, so it can never be a post-assembly hook; it is
  compiled into the discretizer before the matrix exists.

What is left for layer R is what genuinely cannot be any of the other three: a
prescribed rate in a set of cells, a coupling between two blocks the
discretization does not connect, a pinned state, a declared far field.

---

## The contribution contract

`ConditionItem` is the base class and its docstring is the normative version of
this section.

**When it runs.** `ConditionSet.apply()` is called from
`DartsModel.apply_rhs_flux()`, which the nonlinear solver calls after **every**
engine assembly — i.e. once per Newton iteration, on every timestep, including
the ones that are later cut. Contributions are not cached between iterations: an
item re-derives them from the context it is handed, while model-dependent data
(resolved indices, CSR positions) is resolved once, at bind time.

**What it adds.** The engine residual is

```
R = acc(X) - acc(Xn) - dt * (inflow - outflow)
```

so a source `q_c` (per day, **positive into the block**) in block `b` is applied as

```python
ctx.rhs[b * n_vars + c] -= q_c * dt                 # residual, dt-scaled
ctx.jac.add_block(diag_pos(b), -dt * dq_c_dX)       # matching Jacobian block
```

Contributions are therefore **dt-scaled adds in residual units**, with the
engine's sign convention. The framework applies the scaling and the sign for
the shipped item types; a custom item does it in `apply()`.

**Additive or row-claiming.** Every item declares which one it is, in
`ConditionItem.contribution`:

| `contribution` | Meaning |
|---|---|
| `"additive"` (default) | The item **adds** to the rows it touches. Several additive items may share a row and their contributions sum. |
| `"replacement"` | The item **claims** the rows it touches and overwrites them with a constraint. A claimed row admits exactly one claimant and no additive contribution. |
| `"none"` | The item writes neither residual nor Jacobian (a declarative item, or a projection of the state), and takes part in no row conflict. |

The rows an item writes are reported by `written_rows()`, and
`ConditionSet.compile()` rejects a conflict at `init()` time, naming both items
and the offending `(block, equation)`. Without the check the failure is silent:
whichever item happens to run last wins.

**Jacobian writes are CPU-only.** `ctx.jac` is a `BlockCSRView` over
`engine.jac_rows/jac_cols/jac_diags/jac_vals`, which only the CPU flow engines
with an exposed block-CSR matrix provide — a GPU platform and the direct-solver
configurations (e.g. SuperLU) do not. An item that writes the Jacobian sets
`provides_jacobian = True`, and `compile()` then verifies the platform and the
engine **up front** rather than letting the item discover it mid-run. RHS-only
items work on the GPU platform: the solver pushes the host residual back to the
device after the conditions stage.

**Declared item attributes:**

| Attribute | Default | Meaning |
|---|---|---|
| `provides_jacobian` | `False` | The item writes analytic Jacobian blocks (CPU-only; validated at compile time). |
| `requires_platform` | `None` | Restrict the item to `"cpu"` or `"gpu"`. |
| `adjoint_transparent` | `False` | The contribution is correctly differentiated by the adjoint. Items are opaque unless proven otherwise; the history-matching driver refuses to run with an opaque item. |
| `contribution` | `"additive"` | Additive / replacement / none, as above. |
| `carries_restart_state` | `False` | The item owns internal state that must survive a restart (see {ref}`conditions-restart`). |

**Lifecycle of one item:**

| Stage | Method | When |
|---|---|---|
| Declaration | `declare_stencil(model)` | Once, inside `init()`, **before** the mesh connection list is frozen and long before the matrix is allocated |
| Binding | `bind(model)` | Once, in `ConditionSet.compile()` at the end of `init()`, when the engine and its matrix exist |
| Row report | `written_rows(model)` | Once, after `bind()`, for the conflict check |
| Timestep | `on_timestep_start` / `on_timestep_converged` / `on_timestep_failed` | Once per timestep attempt / acceptance / failure |
| Pre-assembly | `project_state(t)` | Only when a model's own Newton loop calls `conditions.project_state(t)` |
| Per-iteration | `apply(ctx)` | After every engine assembly |
| Restart | `save_restart_state()` / `load_restart_state()` | On `save_restart_state()` / `load_restart_data()` |

**Where to register.** `self.conditions.add(item)` is normally called from
`set_boundary_conditions()`. One exception matters: an item that **declares a
Jacobian stencil** must already be registered when `init()` reaches the well
stage, which runs *before* `set_boundary_conditions()` — register such an item
in `set_wells()` (as the DFM-well models do) or in the constructor. Items that
declare nothing may be registered in either place.

**What `compile()` refuses.** At the end of `init()` the set is validated
against the model, and each of these is an error rather than a silent
mis-simulation:

0. the model still carries a removed legacy channel — a `set_rhs_flux()`
   override, or a `rhs_flux_hooks` attribute of its own. Nothing calls or reads
   either, so the contribution would simply vanish; this check runs even when
   the model registers nothing, which is exactly the case that would otherwise
   go unnoticed;
1. the model overrides `apply_rhs_flux()` — it is the framework's entry point, not an extension point, and the registered items would never run;
2. a Jacobian-providing item on a non-CPU platform, or on an engine that does
   not expose its block-CSR matrix;
3. any item on a mechanics model (see {ref}`conditions-limits`);
4. the adjoint/history-matching driver is active and an item is not
   `adjoint_transparent`;
5. an item requiring a platform the model does not run on;
6. two claims on one row, or an additive contribution to a claimed row.

---

## Item types

### `CellSource` — rates in a set of blocks

The workhorse. Rates are given per cell and per equation, positive into the
cell, either as a constant array, a callable `f(t)`, or a callable
`f(t, states)` receiving the current `(n_cells, n_vars)` state of the selected
cells.

```python
from darts.models.conditions import CellSource

class Model(CICDModel):
    def set_boundary_conditions(self):
        self.inflow_cells = np.array([0, 1, 2])
        self.conditions.add(CellSource(cells=self.inflow_cells,
                                       rates=self.mass_flux_rates))

    def mass_flux_rates(self, t: float) -> np.ndarray:
        """(n_cells, n_vars), positive INTO the cell; kmol/day, kJ/day for energy."""
        rates = np.zeros((len(self.inflow_cells), self.physics.n_vars))
        rates[:, self.inflow_var_idx] = -self.outflow
        return rates
```

Pass `d_rates` to add the analytic diagonal Jacobian block of a state-dependent
rate — `d_rates[k, c, v] = ∂q_c/∂x_v` of cell `k`, as an array or a callable
`f(t, states)`. The item then declares `provides_jacobian = True` and is
CPU-only:

```python
self.conditions.add(CellSource(cells=self.leak_cells,
                               rates=self.leak_rates,      # f(t, states)
                               d_rates=self.d_leak_rates)) # f(t, states)
```

Without `d_rates` a state-dependent rate is simply lagged by one Newton
iteration; it converges, but it costs iterations.

### `SegmentSource` — rates in the body segments of a well

A `CellSource` anchored on a well name instead of block indices. It resolves
`well_head_idx` at bind time and applies to segments `1 … num_segments - 1`
only: segment 0 is the wellhead block, whose rows carry the well-control
equations and must never receive a source term. Rate arrays therefore have
`num_segments - 1` rows.

```python
self.conditions.add(SegmentSource(well_name='I1', rates=self.segment_rates))
```

### `PipeSourceTerm` — the injection source of a DFM pipe

Turns the current rate of a pipe source/sink (a `RampUpRate` or subclass, which
`Pipe` updates once per timestep) into the residual contribution of its segment
block. This is the unified replacement for the hand-rolled `set_rhs_flux`
overrides of the DFM-well models:

```python
# in set_wells(), after the Pipe carrying 'RampUpRate1' has been built
self.conditions.add(PipeSourceTerm(well_name='I1', source_sink_name='RampUpRate1'))
```

It is RHS-only, exactly as the legacy path was: for a ramp-up schedule the rate
does not depend on the state, so the analytic Jacobian contribution is zero.

### `InterfaceFlux` — a two-block coupling with analytic derivatives

A subclass supplies the flux of one connection and its two derivative blocks;
the framework owns the CSR positions, the state gather, the residual scatter,
the `dt` scaling and the four-block Jacobian pattern:

```
rhs[row] -= flux * dt        rhs[col] += flux * dt
jac[row, row] -= d_row * dt  jac[row, col] -= d_col * dt
jac[col, row] += d_row * dt  jac[col, col] += d_col * dt
```

```python
from darts.models.conditions import InterfaceFlux

class LinearHeatExchanger(InterfaceFlux):
    """Heat exchanged between two blocks, proportional to their temperature difference."""

    def __init__(self, connections, UA: float, temperature_var: int):
        super().__init__(connections)
        self.UA = UA                          # kJ/day/K
        self.temperature_var = temperature_var

    def connection_flux(self, t, state_row, state_col):
        n_vars = state_row.size
        v = self.temperature_var
        flux = np.zeros(n_vars)
        d_row = np.zeros((n_vars, n_vars))
        d_col = np.zeros((n_vars, n_vars))
        flux[-1] = self.UA * (state_col[v] - state_row[v])   # into row_block
        d_row[-1, v] = -self.UA
        d_col[-1, v] = self.UA
        return flux, d_row, d_col

class Model(CICDModel):
    def set_wells(self):
        ...
        # declares its stencil, so it is registered here, not in
        # set_boundary_conditions()
        self.conditions.add(LinearHeatExchanger([(2, 12)], UA=1.0e3,
                                                temperature_var=self.physics.n_vars - 1))
```

The two blocks do **not** have to be neighbours in the mesh — see
{ref}`conditions-stencil`.

### `ConstantStateBC` — a declared open / constant-state far field

Many models express an open boundary by giving the outermost cells an enormous
volume, so their state stays numerically constant and the boundary acts as an
infinite-acting aquifer. That mechanism is a *mesh* property, so this item
contributes nothing to the system (`contribution = "none"`); what it adds is a
place that sees the trick and checks it.

The check is worth having because the failure mode is silent. The engine caches
`PV = volume * poro` **once**, in `engine.init()`, which `DartsModel.init()`
runs after `set_boundary_conditions()`; a boundary volume written any later is
ignored and the model quietly runs with a closed boundary.

```python
def set_reservoir(self):
    self.reservoir = StructReservoir(...)
    self.reservoir.boundary_volumes['xy_plus'] = 1e20     # before init()

def set_boundary_conditions(self):
    self.conditions.add(ConstantStateBC(faces='xy_plus'))  # now it is checked
```

`faces=None` means "whichever faces the reservoir has a boundary volume for",
and then at least one is required.

### `DirichletPin` — a prescribed state value

Pins `(cell, equation)` pairs to a value — a scalar, one value per cell, or a
callable `f(t)`. It has two modes, and the difference is not cosmetic.

`mode="state"` (the default) is a **projection**: it overwrites the entry of the
state vector and touches neither the residual nor the Jacobian. It needs no
Jacobian, so it works on GPU and with direct solvers, and it reproduces exactly
what models used to hand-roll. `mode="row"` is the assembly-consistent
formulation: the block-CSR row of the pinned equation is replaced by the
constraint (`X - value`), which the linear solver and the preconditioner then
see. It is CPU-only, it claims its rows (`contribution = "replacement"`), and
because it changes the assembled system it changes results — it is strictly
opt-in.

```python
self.conditions.add(DirichletPin(cells=self.pinned_cells,
                                 equation=self.physics.n_vars - 1,   # temperature
                                 values=self.geothermal_temperature,  # f(t)
                                 mode='state'))
```

In state mode the pin is written at two points, both idempotent:
`project_state(t)` **before** the assembly — the load-bearing one, since it is
what makes the residual, the Jacobian and the convergence test see the pinned
state — and `apply()` after it, which is the only one the framework drives by
itself. The stock nonlinear solver has no pre-assembly stage, so a model that
needs the first must call `self.conditions.project_state(t)` at the top of its
own Newton loop, exactly where it used to call its hand-rolled pinning
function. Calling only the second gives a weaker condition than the legacy code
(measured on SPE11b: same timestep and Newton counts, one linear iteration
fewer, a 3e-7 relative shift in the final state).

### Selectors — the "where" half, separated from the "what"

A selector describes a **set of blocks** and resolves it once, at bind time.
Selectors are accepted anywhere an index array is accepted (`CellSource`,
`DirichletPin`, and either member of an `InterfaceFlux` connection), so passing
plain indices keeps working unchanged.

| Selector | Selects |
|---|---|
| `BlockIndices(indices)` | An explicit list of block indices (the identity selector) |
| `Where(predicate, centroids=None)` | Reservoir blocks whose centroid satisfies `f(x, y, z)` |
| `NamedRegion(name)` | A named cell group of the reservoir, or an operator region tag of the physics |

```python
from darts.models.conditions import CellSource, Where

self.conditions.add(CellSource(cells=Where(lambda x, y, z: z > 1000.0),
                               rates=self.aquifer_rates))
```

---

(conditions-stencil)=
## Declaring the Jacobian stencil

A Python contribution can only write Jacobian blocks that **exist** in the
block-CSR pattern, and that pattern is built by
`engine.init_jacobian_structure()` from the sorted two-way mesh connection
arrays. Until M4 a coupling that the mesh did not already contain could not be
written at all, and the linear-IPR hook worked around it by requiring a
zero-transmissibility "fake" perforation whose only purpose was to smuggle its
four blocks into the pattern.

An item now declares the couplings it will write:

```python
def declare_stencil(self, model):
    """(row_block, col_block) pairs; off-diagonal only, both directions implied."""
    return [(well_body_block, reservoir_block)]
```

and the framework adds every declared-but-absent pair to the mesh as a
zero-transmissibility connection before the pattern is built. Points worth
knowing:

* **The window is narrow.** The connection list is open only between
  `mesh.add_wells()` (which is what assigns the well block indices) and
  `conn_mesh::reverse_and_sort()` (which freezes it, in place, and cannot be run
  twice). `DartsModel._init_wells_with_declared_stencil()` reproduces
  `ReservoirBase.init_wells()` with the declaration stage inserted at exactly
  that point — and only when something actually declares a stencil; with no
  declarations it delegates to the reservoir verbatim.
* **`declare_stencil()` runs before the engine exists**, so an implementation
  may use only mesh and well information (block indices, well head/body indices,
  perforations) and must not touch `physics.engine`.
* **Diagonal blocks need not be declared** — they always exist. A declared
  self-coupling is refused, as is a coupling that would attach a column to a
  **wellhead** row: that row carries the well-control equations and must keep
  exactly its diagonal and the well-body column.
* **A declared connection contributes nothing on its own.** It carries
  `trans = 0` and `transD = 0` and is not a DFM connection, and every flux term
  in the assembly is multiplied by one of those or by a phase velocity it does
  not carry.
* **The result is verified, not assumed.** After the connection arrays are
  frozen, `verify_stencil()` checks that every declared coupling appears exactly
  once in each direction, which fails loudly both when a needed coupling is
  missing and when a duplicate was created.
* **A well no longer needs a dummy perforation** to be coupled to the
  reservoir: a perforation-free well is accepted when a registered item declares
  its coupling and reports the well through `declared_well_names()`.

Compiled CSR positions are stamped with a **pattern identity**
(`pattern_identity(model)`: engine identity, a version counter bumped by every
`DartsModel.reset()`, and the array sizes as a cross-check). `engine.init()`
reallocates the matrix, so a re-initialized model — the restart flow, or any
manual re-init — would otherwise write at stale offsets; instead the item is
re-bound. An item that caches positions calls `stamp_pattern(model)` at the end
of its `bind()`, and the framework calls `rebind_if_stale(ctx)` before every
`apply()`.

---

## Observers

A `NonlinearIterationObserver` is the typed way to **see** every assembled
system and contribute nothing to it: policing what the property evaluators did
during the assembly, gathering diagnostics, or raising to force a timestep cut.

```python
from darts.models.conditions import AssemblyContext, NonlinearIterationObserver

class MaxPressureObserver(NonlinearIterationObserver):
    def __init__(self):
        self.max_pressure = -np.inf

    def observe(self, ctx: AssemblyContext):
        pressure = ctx.X[: ctx.n_res_blocks * ctx.n_vars : ctx.n_vars]
        self.max_pressure = max(self.max_pressure, float(pressure.max()))

# in the model:
self.observer = self.conditions.add(MaxPressureObserver())
```

Observers run at the end of the conditions stage, after every item has
contributed, so what they see is the final system of that iteration. The
read-only rule is **enforced, not trusted**: the context handed to `observe()`
is a twin whose `rhs`, `X`, `Xn` and `jac.jac_vals` are non-writable numpy
views, so a write raises `ValueError: assignment destination is read-only`. An
observer that needs to change the system is not an observer — it is a
`ConditionItem`. Observers also receive `on_timestep_start` /
`on_timestep_converged` / `on_timestep_failed`.

An exception raised in `observe()` propagates out of the Newton loop; a model
that raises deliberately is responsible for catching it and turning it into a
timestep cut.

The untyped `DartsModel.after_assembly(dt, t)` hook still runs, last, and is
deprecated: it receives no context, nothing stops it from mutating the system,
and nothing in its signature says it must not.

---

(conditions-restart)=
## Restart

The restart file carries reservoir block data and the OBL history columns —
nothing else. An item that owns internal state therefore declares it:

```python
class MyItem(ConditionItem):
    carries_restart_state = True

    def save_restart_state(self) -> dict:
        return {"cumulative": float(self.cumulative)}   # JSON-serializable

    def load_restart_state(self, state: dict):
        self.cumulative = state["cumulative"]
```

`DartsModel.save_restart_state(path)` writes a sidecar (`<path>.conditions.json`)
holding every stateful item's state, and the restart **refuses** to continue if
that sidecar is missing or describes a different set of items. Continuing would
restart the item from its constructor defaults against a two-year-old reservoir
state, which is the one outcome the contract must not allow.

On the restart path `init()` does not initialize the engine, so `compile()`
records that binding was deferred and `load_restart_data()` re-runs it once the
engine is up.

---

(conditions-limits)=
## Limits

These are properties of the current implementation, not of the design, and each
is enforced rather than left to be discovered:

* **Mechanics models reject per-iteration items.** The pm/mech engines rescale
  equation rows *inside* assembly, so a post-assembly RHS/Jacobian write would be
  applied with the wrong scaling. `compile()` refuses any item on a model driven
  by `MechanicsNewtonSolver`, and `apply_rhs_flux()` raises as well. Mechanics
  boundary conditions go through the declarative spec
  (`darts.reservoirs.boundary_spec`) instead, whose value half is owned and
  driven by the reservoir.
* **The adjoint does not see Python contributions.** The
  optimization/history-matching driver replays the assembly to build gradients
  and knows nothing about this layer, so a gradient computed with an opaque item
  registered would be missing its terms. Items are `adjoint_transparent = False`
  by default and `compile()` refuses to run with an opaque item when
  `engine.opt_history_matching` is set. A term that must be differentiated
  belongs in channel P.
* **A state-mode `DirichletPin` is a projection, not a constraint.** The pinned
  equation is still assembled and its accumulation term is still computed — and
  then discarded, because the next projection overwrites whatever the Newton
  update did to that entry. The constraint is invisible to the linear solver and
  to any preconditioner; they see an unconstrained system and the pin is
  re-imposed behind their back. `mode="row"` is the honest formulation, at the
  cost of being CPU-only and of changing results.
* **Jacobian contributions are CPU-only**, and the Python `BlockCSRView` path is
  a small prototyping fallback, not the general implementation. Native
  CPU/GPU lowering of the item types is planned; unsupported backend
  contributions fail at initialization rather than silently omitting Jacobian
  terms.
* **The per-iteration cost is Python.** An item is called once per Newton
  iteration and should stay vectorized over its cells; the item types shipped
  here are. A per-cell Python loop in `apply()` will show up in the timers.

---

(conditions-migration)=
## Migrating from `set_rhs_flux` / `rhs_flux_hooks`

Both legacy paths are **gone** from `DartsModel` — not deprecated, removed.
`apply_rhs_flux()` is the conditions stage plus the observer stage and nothing
else, the base class defines neither `set_rhs_flux` nor `rhs_flux_hooks`, and
nothing consults a hook list. They were untyped — no declared stencil, no
platform validation, no row-conflict check, no restart contract — and a
`set_rhs_flux` override allocated and returned a whole `n_blocks * n_vars`
vector on every Newton iteration in order to write a handful of entries.

Removing them must not turn a working model into a quietly wrong one, so each
of the three ways the old spellings can still appear fails **loudly**:

| Old spelling | What happens now |
|---|---|
| `self.rhs_flux_hooks.append(item)` | `AttributeError` at the call — there is no such attribute |
| a `set_rhs_flux()` override | `ConditionSet.compile()` refuses at `init()`, naming the migration and the sign flip — nothing would call the override |
| a self-assigned `self.rhs_flux_hooks = []` then `.append(...)` | the same refusal at `init()` — nothing would read the list |

The second and third are the ones worth the check: they would otherwise succeed
and the model would run on without its source term. The refusal is raised before
the registered-item count is even looked at, because the model in danger is
precisely the one that registers nothing.

The translations below are mechanical.

**`set_rhs_flux()` → `CellSource`.** The legacy return value was added as
`rhs += rhs_flux * dt`, so a *positive* legacy entry removed mass from the cell.
`CellSource` rates are positive **into** the cell — the sign flips:

```python
# Before
def set_rhs_flux(self, t=None):
    nv, nb = self.physics.n_vars, self.reservoir.mesh.n_blocks
    rhs_flux = np.zeros(nb * nv)
    rhs_flux_var = rhs_flux[self.inflow_var_idx:self.reservoir.mesh.n_res_blocks * nv:nv]
    rhs_flux_var[self.inflow_cells] = self.outflow
    return rhs_flux

# After
def set_boundary_conditions(self):
    self.conditions.add(CellSource(cells=self.inflow_cells, rates=self.mass_flux_rates))

def mass_flux_rates(self, t: float) -> np.ndarray:
    rates = np.zeros((len(self.inflow_cells), self.physics.n_vars))
    rates[:, self.inflow_var_idx] = -self.outflow      # positive INTO the cell
    return rates
```

**A DFM-well source override → `PipeSourceTerm`.** The overrides of the
DFM-well models all had one body: read the pipe source/sink's current
component/energy rates at the specific potential energy of the receiving block,
and write them negated at `(n_res_blocks + segment_idx) * n_vars`. That is
exactly `PipeSourceTerm`, so the whole method collapses to one registration in
`set_wells()`:

```python
# Before
def set_rhs_flux(self, t=None):
    source = self.wells['I1'].source_sinks['RampUpRate1']
    block = self.reservoir.mesh.n_res_blocks + source.segment_idx
    rates = source.get_component_energy_rates(self.physics,
                                              self.reservoir.mesh.cell_spe[block])
    rhs_flux = np.zeros(self.reservoir.mesh.n_blocks * self.physics.n_vars)
    start = block * self.physics.n_vars
    rhs_flux[start:start + self.physics.n_vars] = -rates
    return rhs_flux

# After
self.conditions.add(PipeSourceTerm(well_name='I1', source_sink_name='RampUpRate1'))
```

If the override did something the source/sink object does not — the choke
boundary node, for instance, whose rates carry the *discharge* enthalpy the
valve produces rather than the constant upstream one
`get_component_energy_rates()` returns — subclass `PipeSourceTerm` and override
`apply()`. Such a term is model-specific, so under the placement policy it
belongs in the model's own directory, not in `darts/`.

**A hand-rolled state pin → `DirichletPin`.** A model that overwrote entries of
`engine.X` before each assembly registers a `DirichletPin(mode='state')` and
calls `self.conditions.project_state(t)` from its Newton loop where it used to
call its own pinning function. The item owns the GPU host round-trip that such
code had to perform by hand.

**A `rhs_flux_hooks` entry → a `ConditionItem`.** A legacy hook object exposed
`apply(dt, t)` and resolved everything itself. As an item:

| Legacy hook | Condition item |
|---|---|
| `model.rhs_flux_hooks.append(hook)` | `model.conditions.add(item)` — in `set_wells()` if it declares a stencil |
| index/CSR resolution inside `apply` | `bind(model)`, once, at the end of `init()` |
| a zero-well-index perforation to get the blocks into the pattern | `declare_stencil(model)` |
| `apply(dt, t)`, reading `engine.RHS`/`jac_vals` itself | `apply(ctx)`, with `ctx.rhs`, `ctx.jac`, `ctx.X`, `ctx.Xn`, `ctx.dt`, `ctx.t` |
| implicit assumptions about platform and adjoint | `provides_jacobian`, `requires_platform`, `adjoint_transparent` |
| — | `written_rows()` + `contribution` (the row-conflict check) |
| — | `carries_restart_state` + `save_restart_state()` / `load_restart_state()` |

**Two worked in-repo migrations.** The DFM-well hooks are now purpose-built
condition items, and they are the closest models to a custom item:
`LinearDFMWellIPRHook` (`darts/pipes/linear_dfm_well_ipr.py`) is additive,
`provides_jacobian = True`, declares its well-segment-to-cell coupling through
`declare_stencil()` — which is what retired its zero-well-index perforation — and
reports `written_rows()`; `SemiAnalyticalWellLateralHeatTransferHook`
(`darts/pipes/add_lateral_heat_exchange.py`) is additive, declares no stencil
(every block it writes is a well-segment diagonal), and sets `provides_jacobian`
per state specification: `True` for PT, where it adds the analytic
`∂R_energy/∂T`, `False` for PH, where the temperature is lagged. Both are
registered in `set_wells()`.

**A post-assembly check → `NonlinearIterationObserver`.** A model that overrode
`apply_rhs_flux()` (or `after_assembly()`) only to inspect the assembled system
registers an observer instead. Overriding `apply_rhs_flux()` while registering
conditions is refused at `init()`, because the items would silently never run.
