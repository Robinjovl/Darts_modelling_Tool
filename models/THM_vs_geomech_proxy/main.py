from model import Model, fmt_e, fmt
from reservoir import file_sha256
from darts.tools.memory import print_allocated_memory
from darts.tools.cicd_tools import compare_vtk_with_ref, save_vtk_ref
import json
import numpy as np
import os
import shutil
import time
from datetime import datetime
from darts.engines import redirect_darts_output, timer_node
from plot_vtk import plot_vtk_pyvista

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def is_struct_like_case(case):
    # struct-like cases end in NX_NY_NZ (e.g. '17_17_15', 'zero_rate_17_17_15') and are
    # generated on the fly; named cases (case_*, no_damage_zone*) ship a committed mesh
    # and must not take the mesh-generation path (which reads idata.other.nx/ny/nz).
    parts = os.path.basename(case).split('_')
    return len(parts) >= 3 and all(p.isdigit() for p in parts[-3:])


def is_unstructured_generated_case(case):
    # Named cases whose mesh is built by a dedicated gmsh script rather than by
    # the structured box path in reservoir.py. Listing them here is what lets a
    # single generate_mesh flag cover every case that can regenerate its mesh.
    base = os.path.basename(case)
    return (base == 'case_5' or base.startswith('case_5_mesh_')
            or base.startswith('no_damage_zone'))


def supports_mesh_generation(case):
    # True when the mesh of `case` can be (re)generated from the scripts in this
    # folder. case_1..case_4 are not included: they share meshes/case_1/mesh.msh,
    # a committed tetrahedral mesh with no generator in the repo (generate_box_3d
    # produces hexahedra), so they keep loading the committed file.
    return is_struct_like_case(case) or is_unstructured_generated_case(case)


def mesh_key_file(mesh_filename):
    # .cache/ is git-ignored
    return os.path.join(os.path.dirname(mesh_filename), '.cache', 'mesh_key.json')


def mesh_up_to_date(mesh_files, key):
    """True when mesh_files were generated from `key` and have not changed since."""
    try:
        with open(mesh_key_file(mesh_files[0])) as f:
            stored = json.load(f)
        return stored['key'] == key and all(
            stored['files'][os.path.basename(fn)] == file_sha256(fn) for fn in mesh_files)
    except (OSError, ValueError, KeyError):
        return False


def store_mesh_key(mesh_files, key):
    key_file = mesh_key_file(mesh_files[0])
    os.makedirs(os.path.dirname(key_file), exist_ok=True)
    with open(key_file, 'w') as f:
        json.dump({'key': key, 'files': {os.path.basename(fn): file_sha256(fn) for fn in mesh_files}},
                  f, indent=1)


def generate_unstructured_mesh(case, idata=None, physics_type='single_phase_thermal',
                               wells_type=None, bulk_mesh_size=None, force=False):
    """
    (Re)generate the mesh of a named unstructured case. The mesh on disk is kept
    when it was generated from the same input: generator arguments, well
    coordinates, generator source and gmsh version (stored in .cache/mesh_key.json).

    Structured NX_NY_NZ cases are meshed inside the reservoir instead - see
    UnstructReservoirCustom.field_reservoir.

    :param case: case name, e.g. 'case_5' or 'no_damage_zone'.
    :param idata: input data of the case; read via set_case.set_input_data when None.
    :param physics_type: physics used to build idata when it is not supplied.
    :param wells_type: well configuration used to build idata when it is not supplied.
    :param bulk_mesh_size: far-field Gmsh size [m]; the well size keeps the 1:20 ratio.
        None takes it from a case_5_mesh_<size>m case name, 200 m otherwise.
    :param force: regenerate even when the mesh is up to date.
    :return: the mesh file, or None when the case has no generator.
    """
    base = os.path.basename(case)
    if not is_unstructured_generated_case(base):
        return None

    import gmsh
    from set_case import set_input_data
    if idata is None:
        idata = set_input_data(base, physics_type=physics_type, wells_type=wells_type)
    # several cases share one mesh (no_damage_zone_heter_mech_prop reuses
    # no_damage_zone), so the folder comes from idata rather than the case name
    mesh_dir = getattr(idata.other, 'mesh_dir', None) or base
    mesh_filename = os.path.join(BASE_DIR, 'meshes', mesh_dir, 'mesh.msh')
    os.makedirs(os.path.dirname(mesh_filename), exist_ok=True)

    if base.startswith('no_damage_zone'):
        import gen_fault_msh_no_damage_zone as generator
        # the fault-tagged twin is what fault.py reads when post-processing
        mesh_files = [mesh_filename, os.path.join(os.path.dirname(mesh_filename), 'mesh_fault.msh')]
        key = {'generator_sha256': file_sha256(generator.__file__), 'gmsh': gmsh.__version__}
        if not force and mesh_up_to_date(mesh_files, key):
            print(f'Mesh of {mesh_dir} is up to date, not regenerated')
            return mesh_filename
        print(f'Generating mesh for {mesh_dir}')
        generator.gen_fault_msh_no_damage_zone(fault=False)
        generator.gen_fault_msh_no_damage_zone(fault=True)
        store_mesh_key(mesh_files, key)
        return mesh_filename

    import gen_fault_msh as generator
    if bulk_mesh_size is None:
        bulk_mesh_size = case_5_mesh_size(base)
    well_mesh_size = bulk_mesh_size / 20.0
    # generate_3d_fault_mesh reads only the well coordinates from idata
    key = {'generator_sha256': file_sha256(generator.__file__), 'gmsh': gmsh.__version__,
           'bulk_mesh_size': float(bulk_mesh_size), 'well_mesh_size': float(well_mesh_size),
           'prod_well_coords': [float(c) for c in idata.other.prod_well_coords],
           'inj_well_coords': [float(c) for c in idata.other.inj_well_coords]}
    if not force and mesh_up_to_date([mesh_filename], key):
        print(f'Mesh of {mesh_dir} is up to date (bulk={bulk_mesh_size:g} m, well={well_mesh_size:g} m), '
              f'not regenerated')
        return mesh_filename
    print(f'Generating {mesh_dir}: bulk={bulk_mesh_size:g} m, well={well_mesh_size:g} m')
    generator.generate_3d_fault_mesh(idata, msh_filename=mesh_filename,
                                     bulk_mesh_size=bulk_mesh_size,
                                     well_mesh_size=well_mesh_size)
    store_mesh_key([mesh_filename], key)
    return mesh_filename


def case_5_mesh_name(mesh_size):
    """
    Return the case/output folder name for a case_5 mesh resolution.

    :param mesh_size: Far-field characteristic Gmsh size in metres.
    :type mesh_size: float
    :return: Case name unique to the requested mesh resolution.
    :rtype: str
    """
    size = float(mesh_size)
    if size <= 0.0:
        raise ValueError('case_5 mesh sizes must be positive')
    size_label = f'{size:g}'
    if float(size_label) != size:  # keep the name lossless: case_5_mesh_size() parses it back
        size_label = repr(size)
    size_label = size_label.replace('.', 'p')
    return f'case_5_mesh_{size_label}m'


def case_5_mesh_size(case, default=200.0):
    """
    Far-field Gmsh size [m] encoded in a case_5_mesh_name() case name, e.g. 300.0
    for 'case_5_mesh_300m'; `default` for any other case name.
    """
    base = os.path.basename(case)
    if base.startswith('case_5_mesh_') and base.endswith('m'):
        return float(base[len('case_5_mesh_'):-1].replace('p', '.'))
    return default


def plot_results(model, time_data_dict, out_dir):
    """Plot well data for thermal/isothermal and single/doublet cases."""
    import matplotlib.pyplot as plt
    import pandas as pd

    time_data = pd.DataFrame.from_dict(time_data_dict)
    if time_data.empty or 'time' not in time_data:
        print('Skipping well plots: no well time data are available')
        return []

    # The first stored row belongs to the long-time geomechanical
    # equilibrium solve (typically time=1e8 days).
    time_data = time_data.iloc[1:].reset_index(drop=True)
    if time_data.empty:
        print('Skipping well plots: no transient well time data are available')
        return []
    time_data['Time (years)'] = time_data['time'] / 365.25

    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d')
    saved_plots = []

    def save_plot(plot_name):
        filename = f'well_time_data_{plot_name}_{timestamp}.png'
        filepath = os.path.join(out_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=200)
        plt.close()
        saved_plots.append(filepath)

    plt.rc('font', size=12)
    well_names = [well.name for well in model.reservoir.wells]
    production_wells = [name for name in well_names if name.upper().startswith('PRD')]

    bhp_keys = [
        (well_name, f'well_{well_name}_BHP') for well_name in well_names
        if f'well_{well_name}_BHP' in time_data
    ]
    if bhp_keys:
        _, ax = plt.subplots()
        for well_name, key in bhp_keys:
            ax.plot(time_data['Time (years)'], time_data[key], label=well_name)
        ax.set(xlabel='Years', ylabel='BHP [bar]', title='Well BHP')
        ax.set_xlim(left=1.0)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend()
        save_plot('well_bhp')

    # Temperature is available only for thermal physics.
    bht_keys = [
        (well_name, f'well_{well_name}_BHT') for well_name in well_names
        if f'well_{well_name}_BHT' in time_data
    ]
    if model.thermal and bht_keys:
        _, ax = plt.subplots()
        for well_name, key in bht_keys:
            ax.plot(time_data['Time (years)'], time_data[key], label=well_name)
        ax.set(xlabel='Years', ylabel='BHT [K]', title='Well Temperature')
        ax.set_xlim(left=1.0)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend()
        save_plot('well_temperature')

    # Plot wellhead water-rate magnitude for all available wells together.
    rate_suffix = '_volumetric_rate_wat_at_wh'
    rate_keys = [
        (well_name, f'well_{well_name}{rate_suffix}') for well_name in well_names
        if f'well_{well_name}{rate_suffix}' in time_data
    ]
    if rate_keys:
        _, ax = plt.subplots()
        for well_name, key in rate_keys:
            ax.plot(
                time_data['Time (years)'], time_data[key].abs(), label=well_name
            )
        ax.set(xlabel='Years', ylabel='Water rate magnitude [m3/day]',
               title='Well Water Rate')
        ax.set_xlim(left=1.0)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend()
        save_plot('well_water_rate')

    # Cumulative extracted energy is meaningful only for a thermal case with
    # at least one production well. Advective heat rate is stored in kJ/day;
    # integration over days and 1e-12 convert it to PJ.
    if model.thermal and production_wells:
        heat_suffix = '_advective_heat_rate_wat_at_wh'
        heat_keys = [
            f'well_{name}{heat_suffix}' for name in production_wells
            if f'well_{name}{heat_suffix}' in time_data
        ]
        if heat_keys:
            energy_data = pd.DataFrame({
                'Time (years)': time_data['Time (years)']
            })
            heat_rate = time_data[heat_keys].sum(axis=1).abs().to_numpy()
            times = time_data['time'].to_numpy()
            dt = np.diff(times, prepend=times[0])
            energy_data['energy'] = np.cumsum(heat_rate * dt) * 1.e-12
            ax = energy_data.plot(x='Time (years)', y='energy', legend=False)
            ax.set(xlabel='Years', ylabel='Extracted energy [PJ]',
                   title='Cumulative Extracted Energy')
            ax.set_xlim(left=1.0)
            ax.grid(True, linestyle='--', alpha=0.3)
            save_plot('extracted_energy')

    print('Well time-data plots:', *saved_plots, sep='\n  ')
    return saved_plots


def run_python(m, days=0, restart_dt=0, init_step = False,
               save_well_data_after_run=True, max_init_cuts=1):
    if days:
        runtime = days
    else:
        runtime = m.ts_control.runtime

    mult_dt = m.ts_control.dt_mult
    max_dt = m.ts_control.dt_max
    m.e = m.physics.engine

    # get current engine time
    t = m.e.t

    # same logic as in engine.run
    if np.fabs(t) < 1e-15:
        dt = m.ts_control.dt_first
    elif restart_dt > 0:
        dt = restart_dt
    else:
        dt = m.ts_control.dt_max

    # evaluate end time
    runtime += t
    ts = 0
    n_init_cuts = 0

    while t < runtime:
        if init_step:   new_time = t
        else:           new_time = t + dt

        if not init_step:
            m.timer.node["update"].start()
            # store boundaries taken at previous time step
            m.reservoir.update(dt=dt, time=new_time)
            # evaluate and assign transient boundaries or sources / sinks
            # m.reservoir.update_boundary(time=new_time, idata=m.idata)
            # update transient boundaries or sources / sinks
            m.reservoir.update_trans(dt, m.physics.engine.X)
            m.timer.node["update"].stop()

        converged = m.nonlinear_solver.run_timestep(dt, t)
        if converged:
            t += dt
            ts = ts + 1
            print("# %d \tT = %f\tDT = %f\tNI = %d\tLI=%d"
                  % (ts, t, dt, m.nonlinear_solver.status.n_newton, m.nonlinear_solver.status.n_linear))

            # Use the configured multiplier for successful growth as well as
            # failed-step reduction instead of a separate hard-coded factor.
            dt *= mult_dt
            if dt > max_dt:
                dt = max_dt

            if t + dt > runtime:
                dt = runtime - t
        else:
            if init_step:
                # The equilibrium step is static (find_equilibrium): a smaller dt leaves the
                # mechanical system and the state unchanged, so a retry repeats the same failed
                # solve and cutting dt would loop forever.
                n_init_cuts += 1
                if n_init_cuts >= max_init_cuts:
                    raise RuntimeError(
                        'Geomechanical equilibrium did not converge after %d attempt(s); cutting dt does '
                        'not change this static problem. Check the linear solver output above (e.g. '
                        '"displacement stage diverged" from FS-CPR).' % n_init_cuts)
            new_time -= dt
            dt /= mult_dt
            print("Cut timestep to %.5e" % dt)

    # update current engine time
    m.e.t = runtime
    if save_well_data_after_run:
        # save well data at every converged time step
        m.output.save_data_to_h5(kind="well")

    stats = m.nonlinear_solver.stats
    print("TS = %d(%d), NI = %d(%d), LI = %d(%d)" % (stats.n_timesteps_total, stats.n_timesteps_wasted,
                                                     stats.n_newton_total, stats.n_newton_wasted,
                                                     stats.n_linear_total, stats.n_linear_wasted))
def run(model_folder, physics_type, uniform_props=False, wells_type=None,
        decouple_geomech=False, generate_mesh=False, report_step = 90., sim_time = 90., plot_vtk_timesteps=[],
        clear_output_dir=False, solver_type='fs_cpr', save_well_time_data=True, cache_discretization=None):
    '''
    :param model_folder: output folder for mesh, vtk results and figures
    :param physics_type: 'single_phase', 'single_phase_thermal'
    :param uniform_props: if False then set other values for perm and porosity out of the reservoir
    :param wells_type: 'prod', 'inj', 'doublet'
    :param decouple_geomech: turn off mechanics->porosity (so pressure and flow) influence
    :param generate_mesh: if True, mesh will be generated (unless it is up to date), otherwise it will be loaded from the model_folder/meshes
    :param solver_type: 'superlu', 'fs_cpr', or 'by_env_var' (see Model.set_solver_params)
    :param save_well_time_data: if True, write the well time-series files (pkl/xlsx).
    :param cache_discretization: reuse the discretization of an earlier run with the same input;
        None: on unless the environment variable DARTS_DISCR_CACHE=0
    :return:
    '''

    t_wall_start = time.time()

    try:
        # if compiled with OpenMP, set to run with 1 thread, as mech tests are not working in the multithread version yet
        from darts.engines import set_num_threads
        set_num_threads(1)
    except:
        pass

    # Structured NX_NY_NZ cases are meshed inside the reservoir; the named
    # unstructured ones need their gmsh script run first, so one generate_mesh
    # flag covers both.
    if generate_mesh:
        generate_unstructured_mesh(model_folder, physics_type=physics_type,
                                   wells_type=wells_type)

    m = Model(model_folder=model_folder, physics_type=physics_type, uniform_props=uniform_props, wells_type=wells_type,
              decouple_geomech=decouple_geomech, generate_mesh=generate_mesh, solver_type=solver_type,
              cache_discretization=cache_discretization)

    m.timer.node["model.init()"] = timer_node()
    m.timer.node["model.init()"].start()
    m.init()
    m.timer.node["model.init()"].stop()

    n_vars = m.physics.engine.get_n_vars()
    n_cells = m.reservoir.mesh.n_blocks
    est_mem_gb = 16 * n_vars * n_cells / 1024**2  # 16 KB per cell per variable (for THM)
    print(f"Estimated memory requirement: 16 KB * {n_vars} vars * {n_cells} cells = {est_mem_gb:.2f} GB")

    #m.restart = False
    #m.set_output()

    #redirect_darts_output('log.txt')
    m.timer.node["update"] = timer_node()
    # Properties for writing to vtk format:
    m.output_directory = get_output_directory(model_folder, physics_type, wells_type)

    if clear_output_dir and os.path.exists(m.output_directory):
        try:
            shutil.rmtree(m.output_directory)
        except:
            pass

    splitter = '-' * 100 + '\n'

    # Set up darts output to evaluate secondary properties (e.g. viscosity) from the
    # primary variables via the property interpolator. all_phase_props=True registers the
    # phase properties (incl. 'mu_<phase>') in output.properties and builds property_itor.
    # THMCModel.init() does not set some attributes that DartsModel.init() sets but which
    # set_output() reads (self.restart, self.has_dfm_well), so set them explicitly here.
    m.restart = False
    from darts.engines import ms_well
    m.has_dfm_well = any(well.ms_type == ms_well.MS_Type.DFM for well in m.reservoir.wells)
    m.set_output(output_folder=m.output_directory, all_phase_props=True, save_initial=False)

    # Preserve the transient first timestep while equilibrium initialization
    # temporarily replaces it with its intentionally very large timestep.
    transient_first_ts = m.ts_control.dt_first

    # For geomechanics equilibrium intialization, we initially run the simulation for a long time
    # to get the equilibrium, then store that initial displacements internally.
    # Further-timestep displacements will be relative to the initial ones.
    print(splitter + 'compute initialization ...\n' + splitter)
    m.reservoir.set_equilibrium(zero_conduction=True)
    m.physics.engine.find_equilibrium = True
    dt_init = 1.e+8 # days
    m.ts_control.dt_first = dt_init
    run_python(m, dt_init, init_step=True)
    m.reinit(zero_conduction=True)
    m.physics.engine.find_equilibrium = False
    print(splitter + 'initialization completed\n' + splitter)

    max_dt = report_step
    m.max_dt = max_dt
    m.ts_control.dt_max = max_dt

    first_ts = min(transient_first_ts, report_step)
    m.ts_control.dt_first = first_ts
    m.set_boundary_conditions_after_initialization()

    if m.decouple_geomech:
        m.reservoir.decouple_geomech()

    m.reservoir.create_vtk_wells(output_directory=m.output_directory)


    visc_key = f'mu_{m.physics.phases[0]}'  # state-dependent viscosity property key, e.g. 'mu_wat'

    m.timer.node["run_python"] = timer_node()
    m.timer.node["run_python"].start()

    m.time_steps = []
    data = []
    # Run over all reporting time-steps:
    ith_step = 0
    t_wall_tsteps_start = time.time()
    while m.physics.engine.t < sim_time:
        run_python(m=m, days=report_step)
        # compute state-dependent viscosity [cP] from the current engine state (primary vars).
        # engine=True evaluates the property interpolator at the live engine.X; the returned
        # array is shaped (n_timesteps=1, n_res_blocks), so [0] picks the single current step.
        # It is indexed by reservoir block id, which matches the cell_ids used in write_to_vtk.
        _, prop_arr = m.output.output_properties(output_properties=[visc_key], engine=True)
        viscosity = prop_arr[visc_key][0]
        m.reservoir.write_to_vtk(m.output_directory, ith_step + 1, m.physics.engine, viscosity=viscosity)
        ith_step += 1
        m.time_steps.append(m.physics.engine.t)
        data.append(m.get_performance_data(is_last_ts=(m.physics.engine.t >= sim_time)))

        # estimation should be based on elapsed for only the timesteps time, without time spent on initilization
        elapsed = time.time() - t_wall_start
        elapsed_tsteps = time.time() - t_wall_tsteps_start
        progress = m.physics.engine.t / sim_time
        estimated = elapsed_tsteps / progress + (t_wall_tsteps_start - t_wall_start) if progress > 0 else 0
        remaining = estimated - elapsed
        print(f"Wall time: elapsed={elapsed:.1f}s, estimated={estimated:.1f}s, remaining={remaining:.1f}s")

    m.timer.node["run_python"].stop()

    total_elapsed = time.time() - t_wall_start
    print(f"Total elapsed: {total_elapsed:.1f}s")

    print('Timers:')
    m.print_timers()
    #m.print_stat()
    if save_well_time_data:
        time_data_dict = m.output.store_well_time_data(save_output_files=True)
        plot_results(
            model=m,
            time_data_dict=time_data_dict,
            out_dir=os.path.join(m.output_directory, 'well_time_data_plots'),
        )
    print('Output folder:', m.output_directory, 'Timesteps:', ith_step, 't=', m.physics.engine.t, 'days')
    print_allocated_memory()

    #m.output.plot_well_time_data(phase_volumetric_rates=True)

    for tstep_to_plot in plot_vtk_timesteps:
        plot_vtk_pyvista(m.output_directory, tstep_to_plot=tstep_to_plot, idata=m.idata,
                         use_mesh_bounds=m.idata.other.use_mesh_bounds_in_plot)

    return m, data


# ---------------------------------------------------------------------------
# Reference solution (vtk) for the test suite
# ---------------------------------------------------------------------------
# The well time-series pkl files used by most models carry only well rates and
# BHP, so they say nothing about the mechanical response (displacements and
# stresses), which is what this model is about. As in
# models/displaced_fault_reactivation, the reference is therefore the vtk
# solution of the last reported timestep, compared field by field.

# Properties compared with the reference. Displacements and effective-stress
# change carry the mechanical response; pressure/temperature carry the flow part
# it is coupled to. The remaining vtk arrays are either constant model input
# (perm, E, poisson, poro) or algebraic combinations of the compared ones
# (tot_stress, eff_stress, strain), so they are left out to keep the reference
# files small.
REF_PROPS = ['pressure', 'temperature', 'ux', 'uy', 'uz', 'delta_eff_stress']
REF_TIMESTEP = 1  # reported timestep to compare (the last one in the test configuration)

# Cases whose reference solution is not added to the repo due to large pkl files.
# The test_suite checks only that these cases run. A reference generated locally with UPLOAD_PKL=1 is
# still compared, so the check can be done by hand.
CASES_WITHOUT_REF = ['case_1']


def get_output_directory(model_folder, physics_type, wells_type):
    """Output folder of a run, also used to locate its reference solution."""
    return os.path.join('results', 'sol_cpp_' + physics_type + '_' + wells_type + '_' + model_folder)


def run_test(args: list = [], platform='cpu'):
    import time
    if len(args) < 2:
        print('Not enough arguments for run_test:', args)
        return 1, 0.0
    case = args[0]
    physics_type = args[1]
    # references are (re)generated by running with UPLOAD_PKL=1
    overwrite = os.getenv('UPLOAD_PKL') == '1'
    thermal = physics_type == 'single_phase_thermal'
    wells_type = 'doublet' if thermal else 'inj'
    # every case that owns a generator regenerates its mesh; case_1..case_4 share
    # the committed meshes/case_1/mesh.msh and have none, so they load it as is
    generate_mesh = supports_mesh_generation(case)
    t0 = time.time()
    try:
        run(
            model_folder=case,
            physics_type=physics_type,
            wells_type=wells_type,
            decouple_geomech=True,
            generate_mesh=generate_mesh,
            sim_time=30.0,
            report_step=30.0,
            clear_output_dir=True,
            solver_type='by_env_var',
            save_well_time_data=True,
            cache_discretization=False,  # tests exercise the discretizer, and a cache would only cost disk
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return 1, time.time() - t0

    vtk_fname = 'solution' + str(REF_TIMESTEP) + '.vtu'
    vtk_cur_fname = os.path.join(get_output_directory(case, physics_type, wells_type), vtk_fname)
    # ref/<case folder>/<solution file>: the leading 'results' of the output folder is
    # dropped - references are committed input, not output
    vtk_ref_fname = os.path.join('ref', os.path.basename(os.path.dirname(vtk_cur_fname)), vtk_fname)
    if overwrite:
        # a zero test time makes the suite report SAVED instead of OK
        return save_vtk_ref(vtk_cur_fname, vtk_ref_fname, props=REF_PROPS), 0.0
    if case in CASES_WITHOUT_REF and not os.path.exists(vtk_ref_fname):
        print('NO REFERENCE VTK FILE', os.path.abspath(vtk_ref_fname) + ';',
              'checked only that the case runs.')
        return 0, time.time() - t0
    return compare_vtk_with_ref(vtk_cur_fname, vtk_ref_fname, props=REF_PROPS, abs_tolerance=1e-7), time.time() - t0


if __name__ == '__main__':
    try:
        # if compiled with OpenMP, set to run with 1 thread, as mech tests are not working in the multithread version yet
        from darts.engines import set_num_threads
        set_num_threads(1)
    except:
        pass

    # turns off mechanical to flow(pressure in this case) impact which is generally is quite small for field-scale applications. It typically improves the convergence.
    decouple_geomech = True
    #decouple_geomech = False

    cases = []

    # nx ny nz
    # cases += ['17_17_15']  # for debugging
    #cases += ['41_41_66']
    #cases += ['71_71_66']
    #cases += ['83_83_90']  # for isothermal (single well)
    #cases += ['97_97_90'] # for thermal (doublet)
    #cases += ['71_1_66']  # 1 layer by Y; it is not correct to use this as it corresponds to plane-strain solution

    generate_mesh=True  # struct-like mesh generation
    #generate_mesh=False  # skips mesh generation (uses a mesh from previous run), use if nothing mesh related was changed

    #cases += ['case_1']
    #cases += ['case_2']
    #cases += ['case_3']
    #cases += ['case_4']
    cases += ['case_5']
    #cases += ['no_damage_zone']
    #cases += ['no_damage_zone_heter_mech_prop']
    # cases += ['zero_rate_17_17_15']

    # Run the same coupled thermal-geomechanical case on multiple meshes.
    # well size keeps the same 1:20 ratio used by the 200 m reference mesh.
    case_5_mesh_study = True
    case_5_mesh_sizes = [600.0]
    if case_5_mesh_study:
        cases = [case_5_mesh_name(mesh_size) for mesh_size in case_5_mesh_sizes]

    # thermal = False
    thermal = True

    if not thermal:
        physics_type = 'single_phase'
    else:
        physics_type = 'single_phase_thermal'

    if not thermal:
        wells_type = 'inj'
    else:
        wells_type = 'doublet'

    if not thermal:
        n_years = 1
    else:
        n_years = 60

    sim_time = 365.25 * n_years
    report_step = 365.25 / 2

    # short run
    #sim_time = 30 # days
    #report_step = sim_time  # days

    # The mesh-size study needs one mesh per resolution, so it drives the
    # generator directly; every other case is meshed by run() below.
    if case_5_mesh_study and generate_mesh:
        for mesh_size in case_5_mesh_sizes:
            generate_unstructured_mesh(case_5_mesh_name(mesh_size),
                                       physics_type=physics_type,
                                       wells_type=wells_type,
                                       bulk_mesh_size=mesh_size)

    for case in cases:
        os.system("title thm_proxy: " + case + " PID=" + str(os.getpid())) # set the window title

        # struct-like NX_NY_NZ cases are meshed in the reservoir, case_5 and
        # no_damage_zone by their own gmsh script (run() dispatches); case_1..case_4
        # have no generator and keep loading the committed meshes/case_1/mesh.msh.
        case_generate_mesh = generate_mesh and supports_mesh_generation(case) and not case_5_mesh_study

        run(model_folder=case, physics_type=physics_type, generate_mesh=case_generate_mesh,
            wells_type=wells_type, decouple_geomech=decouple_geomech,
            report_step=report_step, sim_time=sim_time,
            plot_vtk_timesteps=[0, -1]) # plot initial and last timesteps
