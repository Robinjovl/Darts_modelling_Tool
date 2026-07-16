from model import Model, fmt_e, fmt
from darts.tools.memory import print_allocated_memory
import numpy as np
import os
import shutil
import time
from darts.engines import redirect_darts_output, timer_node
from plot_vtk import plot_vtk_pyvista

def run_python(m, days=0, restart_dt=0, init_step = False):
    if days:
        runtime = days
    else:
        runtime = m.runtime

    mult_dt = m.params.mult_ts
    max_dt = m.params.max_ts
    m.e = m.physics.engine

    # get current engine time
    t = m.e.t

    # same logic as in engine.run
    if np.fabs(t) < 1e-15:
        dt = m.params.first_ts
    elif restart_dt > 0:
        dt = restart_dt
    else:
        dt = m.params.max_ts

    # evaluate end time
    runtime += t
    ts = 0

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

        converged = run_timestep_python(m, dt, t)
        if converged:
            t += dt
            ts = ts + 1
            print("# %d \tT = %f\tDT = %f\tNI = %d\tLI=%d"
                  % (ts, t, dt, m.e.n_newton_last_dt, m.e.n_linear_last_dt))

            dt *= 1.5
            if dt > max_dt:
                dt = max_dt

            if t + dt > runtime:
                dt = runtime - t
        else:
            new_time -= dt
            dt /= mult_dt
            print("Cut timestep to %.5e" % dt)

    # update current engine time
    m.e.t = runtime

    print("TS = %d(%d), NI = %d(%d), LI = %d(%d)" % (m.e.stat.n_timesteps_total, m.e.stat.n_timesteps_wasted,
                                                     m.e.stat.n_newton_total, m.e.stat.n_newton_wasted,
                                                     m.e.stat.n_linear_total, m.e.stat.n_linear_wasted))
def run_timestep_python(m, dt, t):
    self = m
    max_newt = self.params.max_i_newton
    self.e.n_linear_last_dt = 0
    well_tolerance_coefficient = 1e2
    self.timer.node['simulation'].start()
    for i in range(max_newt + 1):
        self.e.assemble_linear_system(dt)
        res = self.e.calc_newton_dev()#self.e.calc_newton_residual()
        self.e.dev_p = res[0]
        self.e.dev_u = res[1]
        dev_e = 0
        if self.reservoir.thermoporoelasticity:
            self.e.dev_e = res[2]
            dev_e = res[2]

        self.e.newton_residual_last_dt = np.sqrt(self.e.dev_u ** 2 + self.e.dev_p ** 2 + dev_e ** 2)        #self.e.newton_residual_last_dt = self.e.calc_newton_residual()
        self.e.well_residual_last_dt = self.e.calc_well_residual()
        print(str(i) + ': ' + 'rp = ' + fmt_e(self.e.dev_p) + '\t' + 'ru = ' + fmt_e(self.e.dev_u) + '\t' + \
                    're = ' + fmt_e(dev_e) + '\t' + 'rwell = ' + fmt_e(self.e.well_residual_last_dt) + '\t' + 'CFL = ' + fmt_e(self.e.CFL_max))

        self.e.n_newton_last_dt = i
        #  check tolerance if it converges
        if ((self.e.dev_p < self.params.tolerance_newton and self.e.dev_u < self.params.tolerance_newton and dev_e < self.params.tolerance_newton
           and self.e.well_residual_last_dt < well_tolerance_coefficient * self.params.tolerance_newton )
              or self.e.n_newton_last_dt == self.params.max_i_newton):
            if (i > 0):  # min_i_newton
                if i < max_newt:
                    converged = 1
                else:
                    converged = 0
                break

        r_code = self.e.solve_linear_equation()
        self.timer.node["newton update"].start()
        self.e.apply_newton_update(dt)
        self.timer.node["newton update"].stop()
        if i < max_newt:
            converged = 1

    # End of newton loop
    converged = self.e.post_newtonloop(dt, t, converged)
    self.timer.node['simulation'].stop()
    return converged

def run(model_folder, physics_type, uniform_props=False, wells_type=None,
        decouple_geomech=False, generate_mesh=False, report_step = 90., sim_time = 90., plot_vtk_timesteps=[],
        clear_output_dir=False, solver_type='fs_cpr'):
    '''
    :param model_folder: output folder for mesh, vtk results and figures
    :param physics_type: 'single_phase', 'single_phase_thermal'
    :param uniform_props: if False then set other values for perm and porosity out of the reservoir
    :param wells_type: 'prod', 'inj', 'doublet'
    :param decouple_geomech: turn off mechanics->porosity (so pressure and flow) influence
    :param generate_mesh: if True, mesh will be generated, otherwise it will be loaded from the model_folder/meshes
    :param solver_type: 'superlu', 'fs_cpr', or 'by_env_var' (see Model.set_solver_params)
    :return:
    '''

    t_wall_start = time.time()

    try:
        # if compiled with OpenMP, set to run with 1 thread, as mech tests are not working in the multithread version yet
        from darts.engines import set_num_threads
        set_num_threads(1)
    except:
        pass

    m = Model(model_folder=model_folder, physics_type=physics_type, uniform_props=uniform_props, wells_type=wells_type,
              decouple_geomech=decouple_geomech, generate_mesh=generate_mesh, solver_type=solver_type)

    m.timer.node["model.init()"] = timer_node()
    m.timer.node["model.init()"].start()
    m.init()
    m.timer.node["model.init()"].stop()

    n_vars = m.physics.n_vars + 3 # 3 displs
    n_cells = m.reservoir.mesh.n_blocks
    est_mem_gb = 16 * n_vars * n_cells / 1024**2  # 16 KB per cell per variable (for THM)
    print(f"Estimated memory requirement: 16 KB * {n_vars} vars * {n_cells} cells = {est_mem_gb:.2f} GB")

    #m.restart = False
    #m.set_output()

    #redirect_darts_output('log.txt')
    m.timer.node["update"] = timer_node()
    # Properties for writing to vtk format:
    m.output_directory = os.path.join('results', 'sol_cpp_' + physics_type + '_' + wells_type + '_' + model_folder)

    if clear_output_dir and os.path.exists(m.output_directory):
        try:
            shutil.rmtree(m.output_directory)
        except:
            pass

    splitter = '-' * 100 + '\n'

    # For geomechanics quilibrium intialization, we initially run the simulation for a long time
    # to get the equilibrium, then store that initial displacements internally.
    # Further-timestep displacements will be relative to the initial ones.
    print(splitter + 'compute initialization ...\n' + splitter)
    m.reservoir.set_equilibrium(zero_conduction=True)
    m.physics.engine.find_equilibrium = True
    dt_init = 1.e+8 # days
    m.params.first_ts = dt_init
    run_python(m, dt_init, init_step=True)
    m.reinit(zero_conduction=True)
    m.physics.engine.find_equilibrium = False
    print(splitter + 'initialization completed\n' + splitter)

    max_dt = report_step
    m.max_dt = max_dt
    m.params.max_ts = max_dt
    first_ts = report_step
    m.params.first_ts = first_ts
    m.set_boundary_conditions_after_initialization()

    if m.decouple_geomech:
        m.reservoir.decouple_geomech()

    m.reservoir.create_vtk_wells(output_directory=m.output_directory)

    # Set up darts output to evaluate secondary properties (e.g. viscosity) from the
    # primary variables via the property interpolator. all_phase_props=True registers the
    # phase properties (incl. 'mu_<phase>') in output.properties and builds property_itor.
    # THMCModel.init() does not set some attributes that DartsModel.init() sets but which
    # set_output() reads (self.restart, self.has_dfm_well), so set them explicitly here.
    m.restart = False
    from darts.engines import ms_well
    m.has_dfm_well = any(well.ms_type == ms_well.MS_Type.DFM for well in m.reservoir.wells)
    m.set_output(output_folder=m.output_directory, all_phase_props=True, save_initial=False)
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
    m.output.store_well_time_data(save_output_files=True)
    print('Output folder:', m.output_directory, 'Timesteps:', ith_step, 't=', m.physics.engine.t, 'days')
    print_allocated_memory()

    #time_data_dict = m.output.store_well_time_data(save_output_files=True)
    #m.output.plot_well_time_data(phase_volumetric_rates=True)

    for tstep_to_plot in plot_vtk_timesteps:
        plot_vtk_pyvista(m.output_directory, tstep_to_plot=tstep_to_plot, idata=m.idata,
                         use_mesh_bounds=m.idata.other.use_mesh_bounds_in_plot)

    return m, data


def run_test(args: list = [], platform='cpu'):
    import time
    if len(args) < 2:
        print('Not enough arguments for run_test:', args)
        return 1, 0.0
    case = args[0]
    physics_type = args[1]
    thermal = physics_type == 'single_phase_thermal'
    wells_type = 'doublet' if thermal else 'inj'
    # structured NX_NY_NZ cases need mesh generation; named cases have a committed mesh
    generate_mesh = case not in ('case_1', 'case_2', 'case_3', 'case_4')
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
        )
        return 0, time.time() - t0
    except Exception as e:
        import traceback
        traceback.print_exc()
        return 1, time.time() - t0


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
    #cases += ['17_17_15']  # for debugging
    #cases += ['41_41_66']
    #cases += ['71_71_66']
    #cases += ['83_83_90']  # for isothermal (single well)
    #cases += ['97_97_90'] # for thermal (doublet)
    #cases += ['71_1_66']  # 1 layer by Y; it is not correct to use this as it corresponds to plane-strain solution

    #generate_mesh=True
    generate_mesh=False  # skips mesh generation (uses a mesh from previous run), use if nothing mesh related was changed

    #cases += ['case_1']
    #cases += ['case_2']
    #cases += ['case_3']
    #cases += ['case_4']
    cases += ['case_5']

    #thermal = False
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
        n_years = 30

    sim_time = 365.25 * n_years
    report_step = 365.25 / 4

    # short run
    #sim_time = 30 # days
    #report_step = sim_time  # days

    if 'case_5' in cases:
        from gen_fault_msh import generate_3d_fault_mesh
        generate_3d_fault_mesh()

    for case in cases:
        os.system("title thm_proxy: " + case + " PID=" + str(os.getpid())) # set the window title
    
        run(model_folder=case, physics_type=physics_type, generate_mesh=generate_mesh,
            wells_type=wells_type, decouple_geomech=decouple_geomech,
            report_step=report_step, sim_time=sim_time,
            plot_vtk_timesteps=[0, -1]) # plot initial and last timesteps
