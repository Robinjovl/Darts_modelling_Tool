
## Description
This geomechanical poroelastic model with contact mechanics is single-phase and single-component.
The model contains a single displaced fault, where the slip condition is evaluated using a constant friction coefficient or dynamic friction coefficient evaluated by slip-weakening law or RSF.
The numerical slip solution is compared against the analytical solution. The fault slip is triggered by stress changes induced by pore pressure change,
whereas the pore pressure can be changes by setting uniform depletion or adding a production well.
Geomechanics in this model is based on the quasi-static formulation before the slip, and can be changed to dynamic formulation during the slip.

## Notes
During the first run, the script creates a file 'cached_preprocessing.pkl' which contains processed mesh data. For next run, the script check that file existance and if it exists,
the mesh processing will be skipped and the data from the file will be used instead to reduce the initialization time.
If the reservoir geometry was changed, the file 'cached_preprocessing.pkl' should be manually deleted.

## Linear solvers per stage
`config['linear_solver'] = {'quasi_static': <name>, 'dynamic': <name>}` selects the linear solver of the
quasi-static stage and the one of the fully dynamic stage, switched at the rupture by
`model.linear_solver.update_solver(spec=...)` (rebuilt against the existing Jacobian). 'dynamic' defaults to the
quasi-static choice rebuilt at tolerance 1e-12 -- the rebuild matters for FS-CPR, whose displacement-block AMG is
set up once and would otherwise stay tuned to the stiffness matrix during the inertial stage (5x more GMRES
iterations); `'inplace'` keeps the pre-existing behaviour (same solver, tolerance tightened only). Names
(`Model.make_stage_solver_spec`): `fs_cpr` (GMRES + FS-CPR, default), `superlu`, `pardiso`, `cudss` (NVIDIA cuDSS
direct solver on the GPU), `gpu_gmres_ilu0` /
`gpu_gmres_ilu0_sp` (GPU GMRES + cuSPARSE block-ILU(0), double / single precision factors; converges only on the
mass-dominated dynamic Jacobians) and `gpu_cusolver`; a ready `LinearSolverSpec` is accepted too. The GPU solvers
need a CUDA build of open-DARTS and run on the CPU-assembled Jacobian (see docs/technical_reference/solvers.md).
`dynamic_benchmark.py --qs_solver ... --dyn_solver ...` runs the mixed well-depletion case and stores per-stage
timers in `dynamic_benchmark.json`.

## Full quasi-static / dynamic / quasi-static run (`run_full.py`)
`python run_full.py --mesh meshes/new_setup_fine.geo --qs_solver cudss --dyn_solver cudss --threads 1 --vtk_every 10`
runs the well-depletion, slip-weakening case through the quasi-static production stage, the fully dynamic
co-seismic stage (entered when the quasi-static Newton loop fails at nucleation) and, once the rupture arrests
(slipping area below `arrest_area_fraction` = 0.5 % of its peak after at least `min_dynamic_steps` = 100
dynamic steps), back to quasi-static stepping with the quasi-static solver re-injected and the timestep
restarting from `dt_after_arrest` (1e-3 days); production continues to the end of the schedule (`--days`).
If the dynamic stage stops making progress (the timestep is cut to the `1e-8` s floor and the step still fails --
the contact chatter of the arrest tail, which backward Euler cannot resolve because it is itself the fallback
scheme), the driver ends the co-seismic stage the same way as a natural arrest and continues quasi-statically;
such escapes are counted as `n_dynamic_stalls` in `run_full.json`.
Output: VTK files (3D + fault, every `vtk_every`-th dynamic step), `fault_video.mp4` (fault profiles;
`--fps`, `--stride`; a GIF when no ffmpeg is found -- `FFMPEG_PATH` or PATH) and `run_full.json` with the per-step
rupture diagnostics of `dynamic_benchmark.Diagnostics`. `--threads` (default 8) sets the OpenMP threads of the
assembly, which is bit-identical to the single-threaded one since the race fixes of this release.
`python compare_resolutions.py --runs fine=<out> ultra=<out>` compares the rupture propagation of two runs
(slipping area, maximum slip / slip rate and rupture-front extent against the time since nucleation, slip
profiles at chosen times, side-by-side animation `rupture_comparison.mp4`). Meshes: `new_setup_fine.geo`
(115 872 wedge cells) and `new_setup_ultra_fine.geo` (2 063 112 wedge cells, ~8.3M unknowns; use
`--qs_solver cudss_hybrid --dyn_solver cudss_hybrid`, cuDSS with part of the LU factors in host memory;
with `DARTS_CUDSS_DEVICE_LIMIT_GB=50` on one 80 GB A100 this costs ~390 s per dynamic step, one Newton
iteration each, against 15 s on the 459 k mesh `new_setup_lc50.geo`, whose factors fit on the device).

A run without `run_full.json` -- still in progress, or interrupted -- is handled too: the time series is
then derived from the fault VTK snapshots themselves (the co-seismic stage is identified by the snapshot
spacing). That derived series follows the same history but measures differently: its slip rate is a finite
difference between snapshots and so under-reads the instantaneous peak (6.7 vs 9.2 m/s at `vtk_every=25` on
the 116k run), and its slipping area counts the cells above a 0.01 m/s rate threshold where the engine uses
the contact state (the two agree at full rupture, not in the quiet nucleation phase). Mixing the two across
the runs of one comparison is therefore not like-for-like: pass `--derive_all` to put every run on the
derived series whenever any of them is still going. A run that has not nucleated yet is left out.

`main.plot_profiles_compare([out_1, out_2], ['DARTS: backward Euler', 'DARTS: Bathe'], 'fault_video_compare.mp4',
fps=5, snapshot_times=(0.15, 0.25))` overlays several runs in the layout of `plot_profiles` (slip, Coulomb /
shear / effective normal stress, friction coefficient and pressure against depth, with its fixed stress limits
and reservoir bands, one colour per run) through the co-seismic stage, aligned on the start of each run's
dynamic stage, and writes a PNG of the nearest frame for each of `snapshot_times` (seconds since that start) --
e.g. two time-integration schemes, or two meshes, of the same case.

## Output
This test outputs two groups of VTK files: one for the domain and one for the fault.

The domain output (3D) contains:
- p - pressure, bars
- u_x, u_y, u_z - displacements, m
- stress - effective stresses tensor, bars (negative values are compressive)
- tot_stress - total stresses tensor, bars; tot_stress = stress - p
- porosity - porosity, dimensionless (0..1)

The fault output (2D) contains:
- g_local - slip value, where 'X' stands for the normal to the fault direction, 'Y' and 'Z' tangential direction
- f_local - traction, with the same magnitudes direction names
- phi - slip cretiria
