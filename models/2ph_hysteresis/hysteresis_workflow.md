# Hysteresis Implementation Workflow

This document summarizes the hysteresis implementation on top of the
`development` branch, focusing on the code path for `sg_max`, Killough-Land
history feedback, GPU history interpolation, and the `2ph_hysteresis` model.

## Scope

The implementation touches these subsystems:

- Python physics framework: `darts/physics/base/physics_base.py`
- Compositional physics: `darts/physics/super/physics.py`
- Compositional property evaluation: `darts/physics/super/property_container.py`
- Hysteresis properties: `darts/physics/properties/hysteresis.py`
- Base model lifecycle and restart/output: `darts/models/darts_model.py`,
  `darts/models/output.py`
- C++ engines and bindings: `engines/src/`, `interpolation/pybind11/`
- Example model: `models/2ph_hysteresis/`
- Regression runner: `models/run_test_suite2.py`

The current `models/2ph_hysteresis/` folder is intentionally minimal and should
only keep:

- `LookupTable.txt`
- `main.py`
- `model.py`

The standalone hysteresis Python test and temporary smoke/restart scripts were
removed; the model is expected to run through the normal model test workflow.

## Function Call Structure

The hysteresis implementation is easiest to understand as five connected call
chains:

1. Model construction declares `sg_max`.
2. Physics initialization builds extended OBL interpolators.
3. Engine assembly evaluates operators on `[X | Xhis]`.
4. A converged timestep updates `sg_max`.
5. Output/restart stores and restores `[X | Xhis]`.

### A. Model Construction

Entry point:

```text
models/2ph_hysteresis/main.py
└── main()
    └── parse_cli_args(sys.argv[1:])
    └── run_case(config, platform)
        └── build_model(config, platform)
            └── Model(hys=config.hysteresis)
            └── Model.setup_case(...)
                └── Model.set_reservoir(...)
                └── Model.set_physics(...)
```

Inside `Model.set_physics()`:

```text
Model.set_physics(...)
├── create HistoryField("sg_max") if self.hys is True
├── Compositional(..., history_fields=[HistoryField(...)] )
├── create PropertyContainer(...)
├── property_container.rel_perm_ev["V"] = KilloughRelPermTable(...)
├── property_container.rel_perm_ev["Aq"] = KilloughRelPermTable(...)
├── property_container.capillary_pressure_ev["V"] = KilloughCapillaryPressureTable(...)
├── property_container.capillary_pressure_ev["Aq"] = KilloughCapillaryPressureTable(...)
└── physics.add_property_region(property_container, region)
```

The key point is that `sg_max` is not hidden inside a custom physics subclass.
It enters the framework through the generic `history_fields` argument.

### B. Physics Initialization and Interpolator Setup

After model construction, `main.py` calls:

```text
build_model(...)
└── model.init(platform=platform)
```

The initialization path is:

```text
DartsModel.init(...)
├── reservoir.init_reservoir(...)
├── Model.set_wells()
├── physics.init_physics(...)
│   ├── physics.determine_obl_bounds(...)
│   ├── physics.set_engine(discr_type, platform)
│   ├── engine.n_his_runtime = physics.n_his
│   ├── physics.set_operators()
│   └── physics.set_interpolators(...)
│       ├── physics.get_interpolator_axes()
│       │   └── returns primary axes + sg_max axis
│       ├── physics.create_interpolator(reservoir_operators, extended_axes)
│       ├── physics.create_interpolator(property_operators, extended_axes)
│       ├── physics.create_interpolator(well_operators, extended_axes)
│       ├── physics.create_interpolator(well_ctrl_operators, extended_axes)
│       └── physics.create_interpolator(thermal_var_operator, primary_PT_axes)
├── reservoir.init_wells()
├── physics.init_wells(...)
│   └── broadcasts HistoryField.default to well.Xhis_well_default
├── set_initial_conditions()
│   └── physics.set_initial_conditions_from_array(...)
│       └── physics.populate_mesh_history_defaults(mesh)
├── reset()
│   └── engine.init(...)
│       └── allocates X, Xhis, Xop, op_ders_arr_ext
└── initialize_history_fields()
    └── physics.set_engine_history_array("sg_max", default)
```

Important ownership boundaries:

- Python owns the declaration and update rule for `sg_max`.
- C++ owns storage and interpolation-time assembly of `[X | Xhis]`.
- Newton unknowns remain primary-width; `sg_max` is outside Newton.

### C. Runtime Operator Evaluation

During a timestep, `DartsModel.run()` repeatedly calls `run_timestep()`:

```text
DartsModel.run(days, ...)
└── while t < stop_time:
    └── run_timestep(dt, t, verbose)
        └── engine.assemble_linear_system(dt)
            ├── well.check_constraints(...)
            ├── evaluate operators and derivatives
            ├── assemble_jacobian_array(...)
            └── solve/update Newton system
```

For a history-aware engine, operator evaluation follows this logic:

```text
engine.assemble_linear_system(dt)
└── if engine.get_n_his() > 0:
    ├── engine.build_Xop()
    │   ├── reservoir cells: Xop = [X | Xhis]
    │   └── boundary cells: Xop = [mesh.pz_bounds | mesh.Xhis_bounds]
    ├── interpolator.evaluate_with_derivatives(
    │       Xop,
    │       block_idxs,
    │       op_vals_arr,
    │       op_ders_arr_ext,
    │   )
    └── engine.project_xop_ders()
        └── keep d(operators)/dX, discard d(operators)/dXhis
```

For a non-history engine, the old path is preserved:

```text
engine.assemble_linear_system(dt)
└── if engine.get_n_his() == 0:
    └── interpolator.evaluate_with_derivatives(
            X,
            block_idxs,
            op_vals_arr,
            op_ders_arr,
        )
```

### D. Property Evaluation at an OBL Point

When an interpolator evaluates operators, it calls the Python evaluator stack:

```text
ReservoirOperators.evaluate(state)
└── PropertyContainer.evaluate(state)
    ├── get_state(state)
    │   └── reads primary variables from the front of state
    ├── run_flash(...)
    ├── compute_saturation(...)
    ├── extract history tail
    │   └── history_values = {"sg_max": state[-1]}
    ├── capillary_pressure_ev.evaluate(...)
    │   └── if HistoryAwareCapPressure:
    │       └── evaluate(sat, sg_max=...)
    └── rel_perm_ev[phase].evaluate(...)
        └── if HistoryAwareRelPerm:
            └── evaluate(sat, sg_max=...)
```

For gas relative permeability:

```text
KilloughRelPermTable.evaluate(sat, sg_max)
├── if phase is water:
│   └── evaluate_drainage(sat)
├── if gas sat >= sg_max:
│   └── evaluate_drainage(sat)
└── otherwise:
    └── evaluate_scanning(sat, sg_max)
        ├── _scan_cache lookup
        ├── _make_scanning_interp(sg_max) if needed
        └── interpolate scanning curve
```

For capillary pressure:

```text
KilloughCapillaryPressureTable.evaluate(sat, sg_max)
├── gas phase: return 0
├── convert wetting sat to gas sat
├── evaluate drainage Pc
├── if sg >= sg_max: return drainage Pc
└── blend drainage and imbibition Pc using Killough weighting
```

### E. Timestep-End History Update

`sg_max` is updated only after a timestep has converged:

```text
DartsModel.run(...)
└── if run_timestep(...) converged:
    ├── advance time
    └── after_converged_timestep()
        └── Model.after_converged_timestep()
            ├── update_injection_schedule()
            └── super().after_converged_timestep()
                └── update_history_fields_after_timestep()
                    └── Model.update_history_fields_after_timestep()
```

The model-specific update is:

```text
Model.update_history_fields_after_timestep()
├── output.output_properties(["sat_V"], engine=True)
│   └── reads current gas saturation from engine state
├── physics.get_engine_history_array("sg_max")
├── for each region:
│   ├── land_model = KilloughLandModel(swc, sgrmax)
│   └── for each block in region:
│       └── sg_max[block] = land_model.update_sg_max(
│               sg[block],
│               sg_max[block],
│           )
└── physics.set_engine_history_array("sg_max", sg_max)
```

The core algebra inside `KilloughLandModel.update_sg_max()` is:

```text
update_sg_max(sg, sg_max_old)
├── if sg >= sg_max_old:
│   └── return sg
├── sgr_old = residual_gas_saturation(sg_max_old)
├── if sg < sgr_old:
│   ├── sg_max_new = dissolution_feedback_sgmax(sg)
│   └── return clip(sg_max_new)
└── return sg_max_old
```

This is where dissolution feedback enters the implementation.

### F. GPU History Interpolation

The GPU path mirrors the CPU history path, but uses device buffers:

```text
engine_base_gpu::assemble_linear_system(dt)
└── evaluate_operators_d()
    ├── if get_n_his() > 0:
    │   ├── build_Xop()
    │   ├── copy_data_to_device(Xop, Xop_d)
    │   ├── evaluate_with_derivatives_d(
    │   │       Xop_d,
    │   │       block_idxs_d,
    │   │       op_vals_arr_d,
    │   │       op_ders_arr_ext_d,
    │   │   )
    │   ├── copy_data_to_host(op_ders_arr_ext, op_ders_arr_ext_d)
    │   ├── project_xop_ders()
    │   └── copy_data_to_device(op_ders_arr, op_ders_arr_d)
    └── else:
        └── evaluate_with_derivatives_d(
                X_d,
                block_idxs_d,
                op_vals_arr_d,
                op_ders_arr_d,
            )
```

The important structure is the same as CPU:

```text
extended interpolation state on GPU -> primary-width Newton derivatives
```

### G. Output and Restart

Reservoir output writes extended state:

```text
Output.save_data_to_h5(kind="reservoir")
└── save_specific_data(reservoir_solution.h5)
    ├── dataset width = physics.n_state
    ├── physics.get_engine_interpolator_state()
    │   └── returns flattened [X | Xhis]
    └── write dynamic/X and variable_names including sg_max
```

Well output writes primary state only:

```text
Output.save_data_to_h5(kind="well")
└── save_specific_data(well_data.h5)
    └── dataset width = physics.n_vars
```

Restart restores both primary and history state:

```text
DartsModel.load_restart_data(reservoir_filepath)
├── output.read_specific_data(...)
├── split columns:
│   ├── names in physics.vars -> initial_values
│   └── names in physics.history_fields -> history_values
├── physics.set_initial_conditions_from_array(initial_values)
├── reset()
│   └── engine.init(...) allocates Xhis
└── for each history column:
    └── physics.set_engine_history_array(label, values)
```

## 1. Declare History Variables

`HistoryField` is introduced in `darts/physics/base/physics_base.py`.

It describes one auxiliary OBL interpolation axis:

- `label`: name used in state/output, for example `sg_max`
- `axis_min`, `axis_max`: OBL bounds for the history axis
- `n_axis_points`: number of supporting points on this axis
- `default`: initial and fallback value

History variables are part of the OBL interpolation state, but they are not
Newton unknowns. In other words, interpolators see:

```text
[primary Newton state | history state]
```

while the Newton linear system is assembled only for the primary state.

## 2. Extend PhysicsBase State Handling

`PhysicsBase` now stores `history_fields` and exposes:

- `n_his`: number of history variables
- `n_state`: `n_vars + n_his`
- `get_interpolator_axes()`: primary OBL axes extended with history axes
- `get_interpolator_state_labels()`: primary variable labels plus history labels
- `get_engine_history_array(label, n_blocks=None)`: read one history field from
  `engine.Xhis`
- `set_engine_history_array(label, values, n_blocks=None)`: write one history
  field into `engine.Xhis`
- `get_engine_interpolator_state(n_blocks=None)`: return flattened `[X | Xhis]`
  for output/property evaluation

During `init_physics()`, the Python physics object sets:

```python
engine.n_his_runtime = self.n_his
```

before `engine.init()` allocates buffers.

## 3. Build Interpolators on Extended Axes

`PhysicsBase.set_interpolators()` uses `get_interpolator_axes()` for:

- reservoir accumulation/flux interpolators
- property interpolators
- well interpolators
- well-control interpolators

The thermal-variable interpolator remains on the primary PT axes, because it is
used for primary thermal state conversion rather than history-aware flow
operators.

`create_interpolator()` accepts an explicit `n_axes_points` argument so the same
method can construct both primary-width and extended-width interpolators.

## 4. Pass History Labels into Property Containers

When a property region is added through `PhysicsBase.add_property_region()`, the
physics object propagates:

```python
property_container.n_his = self.n_his
property_container.history_labels = [h.label for h in self.history_fields]
```

This lets `PropertyContainer.evaluate()` split incoming OBL state into:

- primary variables: pressure, composition, optional temperature/enthalpy
- trailing history variables: for example `sg_max`

For thermal states, this is important because temperature/enthalpy is no longer
blindly read from the final state entry when history variables are appended.

## 5. Dispatch Only to History-Aware Evaluators

`darts/physics/properties/hysteresis.py` defines two abstract base classes:

- `HistoryAwareRelPerm`
- `HistoryAwareCapPressure`

`PropertyContainer.evaluate()` forwards history values only when an evaluator
inherits from these classes. Plain evaluators keep the legacy call shape:

```python
kr_ev.evaluate(sat)
```

History-aware evaluators receive keyword arguments:

```python
kr_ev.evaluate(sat, sg_max=..., ...)
```

This keeps non-hysteretic models compatible while allowing more history fields
later.

## 6. Implement Killough-Land Hysteresis

`KilloughLandModel` stores the Land trapping model:

```text
C = 1 / sgrmax - 1 / (1 - swc)
sgr = sg_max / (1 + C * sg_max)
```

It provides:

- `residual_gas_saturation(sg_max)`
- `dissolution_feedback_sgmax(sgr_new)`
- `update_sg_max(sg, sg_max)`

The important latest fix is `dissolution_feedback_sgmax()`. When gas saturation
drops below the old trapped residual value, the model no longer simply clips
`sg_max` to the old `sgr`. Instead, it applies the inverse Land relation:

```text
sg_max_new = sgr_new / (1 - C * sgr_new)
```

so dissolution feedback updates the historical maximum consistently with the new
trapped-gas state.

## 7. Add Relative Permeability and Capillary Pressure Scanning Curves

The hysteresis module contains:

- `KilloughRelPermCorey`
- `KilloughRelPermTable`
- `KilloughCapillaryPressureTable`

For the non-wetting phase:

- if `sat >= sg_max`, use the primary drainage curve
- if `sat < sg_max`, use a Killough scanning curve

For the wetting phase, the implementation keeps the drainage curve behavior.

The table-driven implementation reads sections from `LookupTable.txt`, supports
wetting/non-wetting saturation axes, and caches generated scanning interpolators
with a bounded LRU cache to avoid unbounded memory growth.

## 8. Store History in the C++ Engine

`engine_base` adds runtime history support:

- `n_his_runtime`: number of history fields set from Python
- `Xhis`: per-cell history values
- `Xop`: extended OBL state `[X | Xhis]`
- `op_ders_arr_ext`: derivative scratch sized by `n_state`
- `newton_to_obl`: mapping from Newton unknown order to OBL primary-axis order

The central method is:

```cpp
engine_base::build_Xop()
```

For reservoir cells it writes:

```text
Xop[cell] = [X[cell] | Xhis[cell]]
```

For boundary cells it writes:

```text
Xop[boundary] = [mesh->pz_bounds[boundary] | mesh->Xhis_bounds[boundary]]
```

If no boundary history values are provided, the engine falls back to zero.

## 9. Drop History Derivatives Before Newton Assembly

History variables are not Newton unknowns, so derivatives with respect to
history axes must not enter the linear system.

After evaluating interpolators on the extended state, the engine calls:

```cpp
engine_base::project_xop_ders()
```

This copies only the first `n_vars` derivative columns from the extended
derivative buffer into the standard Newton derivative buffer.

Conceptually:

```text
d operators / d [X | Xhis]  ->  d operators / d X
```

The `d operators / d Xhis` columns are discarded.

## 10. Route CPU, MPFA, and GPU Paths Through Xop

All relevant assembly paths check whether history fields are active:

```cpp
if (get_n_his() > 0)
{
    build_Xop();
    evaluate_with_derivatives(Xop, ..., op_ders_arr_ext);
    project_xop_ders();
}
else
{
    evaluate_with_derivatives(X, ..., op_ders_arr);
}
```

The same logic is applied to:

- base CPU engine path
- MPFA / nonlinear MPFA engines
- super MPFA engine
- GPU engine path

For GPU, the implementation allocates and uses:

- `Xop_d`
- `op_ders_arr_ext_d`

The GPU interpolation path copies/builds the extended state, evaluates operators
on device using the extended state, copies/project derivatives as needed, and
then continues assembly with the primary-width Newton derivative buffer.

## 11. Propagate History Defaults to Boundaries and Wells

History defaults must be available wherever OBL operators are evaluated.

For boundary cells:

- `conn_mesh` adds `Xhis_bounds`
- `PhysicsBase.populate_mesh_history_defaults()` writes field defaults into it

For wells:

- `ms_well` adds `Xhis_well_default`
- `well_control_iface` adds `Xhis_well_default`
- `PhysicsBase.init_wells()` broadcasts history defaults into wells, controls,
  and constraints

Well rate/control evaluation appends these defaults to the well state so well
interpolators see the same extended state layout as reservoir interpolators.

## 12. Initialize and Update History in DartsModel

`DartsModel.init()` seeds history after `reset()`:

```python
self.initialize_history_fields()
```

The base model also adds:

- `after_converged_timestep()`
- `update_history_fields_after_timestep()`

The default update hook is a no-op. Hysteretic models override it to update
history after each converged timestep.

The `models/2ph_hysteresis/model.py` override reads current gas saturation,
updates `sg_max` region by region using `KilloughLandModel.update_sg_max()`, and
writes the result back with:

```python
self.physics.set_engine_history_array("sg_max", sg_max, n_blocks=...)
```

## 13. Preserve History in Output and Restart

Reservoir HDF5 output stores extended state:

```text
[pressure, composition, ..., sg_max]
```

Well HDF5 output remains primary-width and does not include history fields.

`Output.configure_h5_output()` chooses the dataset width:

- reservoir output: `physics.n_state`
- well output: `physics.n_vars`

`Output.save_specific_data()` writes `[X | Xhis]` for reservoir output by using
`physics.get_engine_interpolator_state()`.

`DartsModel.load_restart_data()` splits loaded columns into:

- primary variables, restored through `set_initial_conditions_from_array()`
- history variables, restored through `set_engine_history_array()`

This ensures `sg_max` survives restart.

## 14. Use the Minimal 2ph Hysteresis Model

The example model lives in `models/2ph_hysteresis/`.

`model.py` defines:

- two components: `H2O`, `CO2`
- two phases: aqueous and vapor
- structured 1D reservoir
- table-driven Killough relative permeability
- table-driven Killough capillary pressure
- optional `sg_max` history field when hysteresis is enabled

`main.py` provides:

- `CaseConfig`
- `build_model()`
- `run_case()`
- CLI platform selection: `cpu` or `gpu`
- optional Sg snapshot plotting through existing CLI flags

`LookupTable.txt` contains the drainage/imbibition lookup data.

The temporary helper scripts were removed:

- `smoke_hys.py`
- `restart_hys.py`

The pure Python hysteresis test was also removed:

- `tests/hysteresis/test_hysteresis.py`

## 15. Test-Suite Integration

`models/run_test_suite2.py` includes `2ph_hysteresis` in the default accepted
model list. It should not be special-cased behind a `debug_only_hysteresis`
switch. The intended behavior is that hysteresis participates in the normal
model test flow.

## 16. Validation Notes

Minimum validation for this workflow:

1. Lint changed Python files:

   ```bash
   pre-commit run -v --files <changed-python-files> --show-diff-on-failure
   ```

2. Compile edited Python entry points:

   ```bash
   python -m py_compile models/2ph_hysteresis/main.py models/2ph_hysteresis/model.py
   ```

3. Rebuild/install after C++ history interpolation changes:

   ```bash
   ./helper_scripts/install_darts.sh -e
   ```

4. Run a focused hysteresis model smoke or the model regression suite:

   ```bash
   cd models
   darts run_test_suite2.py LOG
   ```

5. For GPU-specific validation, use the GPU build/install workflow, then run:

   ```bash
   cd models/2ph_hysteresis
   python main.py gpu
   ```

If the current environment cannot run GPU or lacks a fresh compiled extension,
record that blocker explicitly rather than marking GPU runtime verification as
complete.
