# Model workflow reference

## Scope

Model scripts under `models/<case>/` (`model.py` defines `Model(DartsModel)`, `main.py` runs it),
`tutorials/`, and study drivers in `workflows/` (ensemble, history matching, optimization use
their own skills). Regression references live next to each model as `.pkl` files.

## Configuration

- `Model.__init__` sets reservoir (`set_reservoir`), physics (`set_physics`), wells
  (`set_wells`), initial conditions and well controls; timestep and solver settings come from
  the model's data/params objects (`set_sim_params` where present).
- Keep default construction bit-identical when adding options: new keyword arguments default to
  the old behaviour (e.g. `Uniform_Brugge/model.py` `perm`, `well_coords`, `regenerate_mesh`).
- OBL grids and caches: `obl_point_data_*.pkl` keys ignore PVT/relperm content; delete the cache
  when physics inputs change.

## Running

- Canonical run path from the model directory: `darts main.py` (uses the pinned `darts` launcher of
  the session environment; never a bare `python` from another environment).
- Typical `main.py`: `redirect_darts_output('run.log')`, `m = Model()`, `m.init()`,
  `m.set_output()`, `m.run(days)`, then `m.print_timers()` / `m.print_stat()`.
- Regression suite: `cd models && darts run_test_suite2.py LOG` (all models, compares to
  `.pkl` references); a single model runs in its own directory.
- Many members or parameter sweeps: use the `workflows` package (one process per simulation,
  journaled), not a Python loop over `Model()`.

## Analysis

- Well time series: `m.output.store_well_time_data(save_output_files=False)` returns a dict with
  `time` and `well_<name>_<quantity>` keys (`volumetric_rate_<phase>_at_wh`, `BHP`, `BHT`).
- `engine.time_data` holds instantaneous values per timestep; `time_data_report` interval
  averages between report times: do not mix them.
- Solver statistics: `m.nonlinear_solver.stats` (Newton/linear iterations, wasted steps, cuts).
- Cell arrays: `m.output.output_properties(...)` and the HDF5 written by `set_output()`.

## Troubleshooting

- Timestep cuts and wasted Newton steps: check OBL axis bounds (`physics.obl_axis_min/max`),
  initial state inside the box, and `newton_global_chop`.
- Crash on `init`: mesh/well cell lookup (`reservoir.find_cell_index`), missing input files,
  stale OBL cache.
- Slow runs: `OMP_NUM_THREADS` set per process; one process per simulation; lean outputs
  (no VTK, no xlsx) for studies.
- Different results between runs: engine fingerprint (`workflows/journal.py:engine_fingerprint`)
  and interpreter path first; the `solvers`-style env shadowing trap is common.

## Post-processing

- VTK output: `m.output.output_to_vtk(...)` writes `.vts/.vtu` under the output directory;
  view with ParaView (local install noted in `docs/`).
- Keep post-processing out of study members; process `members/<id>/result.json` afterwards.

## Output and Plotting

- Well curves: `m.output.plot_well_time_data(...)` or `darts.tools.plot_darts`
  (`plot_oil_rate_darts`, `plot_water_rate_darts`, `plot_gas_rate_darts`, `plot_bhp_darts`,
  `plot_watercut_darts`) on the well time-data DataFrame.
- Save figures to files in headless sessions (`matplotlib.use("Agg")`); never call `plt.show()`
  in automated runs.
