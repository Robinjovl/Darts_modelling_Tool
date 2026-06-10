# SPE11b — schema-first JSON port

This directory is a minimal working JSON port of the SPE11b CO2-storage
benchmark (`models/SPE11b/`).  The goal is to demonstrate that the
schema-first `darts.api` pipeline can drive an SPE11b-class thermal,
heterogeneous, two-phase CO2-H2O simulation from a single
[`SPE11b.json`](./SPE11b.json) plus a thin custom-plugin sidecar.

## Files

| File | Purpose |
|---|---|
| [`SPE11b.json`](./SPE11b.json) | The schema-first model config (reservoir + thermal physics + 7 property regions + 2 wells + initial conditions + sim params + output). |
| [`spe11b_plugins.py`](./spe11b_plugins.py) | Custom evaluators registered as DARTS plugins: DartsFlash VLAq PT-flash, PR-EoS density / enthalpy, Garcia2001 aqueous density, Fenghour1998 / Islam2012 viscosities, modified Brooks-Corey rel-perm. |
| [`bake_arrays.py`](./bake_arrays.py) | One-shot Python script that runs the original `FluidFlowerStruct` from `models/SPE11b/` to materialise the SPE11b geometry as flat per-cell JSON arrays under `arrays/`.  Also prints the well-cell indices and boundary-volume multiplier needed by the JSON. |
| [`run_spe11b_json.py`](./run_spe11b_json.py) | Driver that loads `SPE11b.json`, builds the model via `ModelBuilder`, and steps through the 3-segment SPE11b injection schedule (I1 only → I1+I2 → both off). |
| [`arrays/`](./arrays/) | Baked per-cell arrays — `permx/permy/permz/poro/depth/hcap/rcond/op_num.json` (length 1008 = nx·ny·nz) + `initial_conditions.json` (two-point depth table). |

## What's covered vs. dropped

| SPE11b feature | Status in JSON port |
|---|---|
| Thermal compositional PT physics (CO2-H2O, V/Aq phases) | ✅ via builtin `physics/Compositional@v1` + the SPE11b plugin sidecar |
| 7-facies heterogeneous reservoir with anisotropic perm, hcap, rcond | ✅ baked into per-cell JSON arrays referenced via `DataRef` |
| 7 property regions with per-region rel-perm | ✅ `physics.property_regions[]` with `op_num` per cell |
| DartsFlash NegativeFlash (PR + Aq activity) PT flash | ✅ `flash/SPE11bVLAq@v1` plugin |
| Hydrostatic + geothermal initial conditions | ✅ via new `initial_conditions.by_depth_table` schema variant |
| `5e9·(1200/nz)` boundary-volume multiplier on left/right faces | ✅ via new `reservoir.boundary_volumes` schema field |
| Mass-rate CO2 injection (3024 kg/day) with thermal injection (283.15 K) | ✅ standard `MASS_RATE` well control |
| 3-segment injection schedule (I1 only 0–25y, I1+I2 25–50y, both off 50–) | ✅ driven from `run_spe11b_json.py` (re-applies controls between segments) |
| `newton_global_chop` Newton type | ✅ added to schema enum |
| Capillary pressure (per-region Brooks-Corey) | ⚠️ omitted — schema has no `capillary_pressure_ev` slot yet; impact at the 50 yr horizon is minor compared to dispersion |
| Dispersion (`init_dispersion` + `reconstruct_velocities`) | ❌ omitted — benchmark-only |
| RHS-flux injection (`apply_rhs_flux`, `set_top_bot_temp`) | ❌ omitted — benchmark-only; the JSON port uses the standard `MASS_RATE` well-control path (also supported by the Python baseline) |
| Custom `run_timestep` loop with stationary-point detection / line search | ❌ omitted — uses the stock DartsModel timestepper. `line_search` and `newton_tol_stationary` remain available via `sim_params`. |
| Per-axis OBL `n_axes_points[0] = 1001` override | ❌ omitted — uniform `n_points = 1001` across all OBL axes |
| Mass-per-component auditing, restart, custom plotting | ❌ omitted — post-processing concerns |

## How to run

```bash
# from /data/avnovikov/open-darts-api
source ~/miniconda3/etc/profile.d/conda.sh && conda activate agent

# 1) Bake the per-cell arrays (only needed once, or after editing
#    layer_props / nx / nz).  Writes to models/SPE11b_json/arrays/.
python models/SPE11b_json/bake_arrays.py

# 2) Drive the 3-segment SPE11b injection schedule.  Defaults run the
#    full 25y / 25y / 40y horizon; shorten via CLI flags for smoke tests.
python models/SPE11b_json/run_spe11b_json.py \
    --segment1 25 --segment2 25 --segment3 40
```

Outputs (`reservoir_solution.h5`, well time data, well log) go to
`output_json/` under the current working directory.

## Schema additions delivered alongside this port

To make this port possible, the following narrow extensions were added to
`darts.api` and the underlying configs.  Each is broadly useful beyond
SPE11b:

* `StructReservoirConfig.boundary_volumes` — dict of per-face volume
  multipliers (`yz_minus`, `yz_plus`, `xy_minus`, `xy_plus`,
  `xz_minus`, `xz_plus`).  Applied after `from_config()` via the
  existing `set_boundary_volume()` API.
* `StructReservoirConfig.op_num` and `actnum` widened to accept
  `DataRef` for large per-cell index arrays; the builder resolves them
  next to the float-typed reservoir fields.
* `InitialConditionsConfig.by_depth_table` (new
  `InitialConditionsByDepthTable` submodel) routed to
  `Physics.set_initial_conditions_from_depth_table` via a new
  `DartsModel._set_initial_conditions_from_depth_table` helper.
* `SimParamsConfig.newton_type` enum extended to include
  `newton_std`, `newton_global_chop`, and `newton_inflection_point`
  alongside the previously supported `newton_local_chop` and `default`.
* `PropertyContainerConfig.temperature` widened to `float | None` so
  JSON property containers can switch into thermal mode by passing
  `null` (mirroring the legacy Python kwarg semantics).

## Limitations

* The 3-segment schedule lives in the driver, not the JSON.  When the
  schema-first transport gains time-keyed `wells[].schedule`
  application in `JsonModel.set_well_controls`, the driver collapses
  back to a one-shot `python -m darts.api.run_json_model --json …`.
* The Python baseline routinely runs at `Nt=30, Dt=3 yr` (90-year
  horizon).  The driver defaults reproduce that horizon
  (`25 + 25 + 40 = 90 yr`); shorten via CLI for smoke tests.
* Numerical results are expected to differ from the Python baseline
  by:
  * a few-bar pressure / few-K temperature drift near top/bottom
    boundaries because the JSON port does not pin top/bot temperatures
    inside the Newton loop;
  * slightly reduced CO2 plume smearing due to the absence of dispersion.
  These differences are intentional for the *minimal* port.
