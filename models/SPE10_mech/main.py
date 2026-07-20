from model import Model
import numpy as np
import os
from darts.engines import redirect_darts_output, timer_node

try:
    # if compiled with OpenMP, set to run with 1 thread, as mech tests are not working in the multithread version yet
    from darts.engines import set_num_threads
    set_num_threads(1)
except:
    pass

def run_python(m, days=0, restart_dt=0, init_step = False):
    if days:
        runtime = days
    else:
        runtime = m.runtime

    mult_dt = m.data_ts.dt_mult
    max_dt = m.data_ts.dt_max
    m.e = m.physics.engine

    # get current engine time
    t = m.e.t

    # same logic as in engine.run
    if np.fabs(t) < 1e-15:
        dt = m.data_ts.dt_first
    elif restart_dt > 0:
        dt = restart_dt
    else:
        dt = m.data_ts.dt_max

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
                  % (ts, t, dt, m.nonlinear_solver.status.n_newton, m.nonlinear_solver.status.n_linear))

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

    stats = m.nonlinear_solver.stats
    print("TS = %d(%d), NI = %d(%d), LI = %d(%d)" % (stats.n_timesteps_total, stats.n_timesteps_wasted,
                                                     stats.n_newton_total, stats.n_newton_wasted,
                                                     stats.n_linear_total, stats.n_linear_wasted))
def run_timestep_python(m, dt, t):
    self = m
    max_newt = self.nonlinear_solver.spec.max_iterations
    solver = self.nonlinear_solver
    status = solver.status
    status.reset()
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

        status.newton_residual = np.sqrt(self.e.dev_u ** 2 + self.e.dev_p ** 2 + dev_e ** 2)        #status.newton_residual = self.e.calc_newton_residual()
        status.well_residual = self.e.calc_well_residual()
        print(str(i) + ': ' + 'rp = ' + str(self.e.dev_p) + '\t' + 'ru = ' + str(self.e.dev_u) + '\t' + \
                    're = ' + str(dev_e) + '\t' + 'rwell = ' + str(status.well_residual) + '\t' + 'CFL = ' + str(self.e.CFL_max))

        status.n_newton = i
        #  check tolerance if it converges
        if ((self.e.dev_p < self.nonlinear_solver.spec.tolerance and self.e.dev_u < self.nonlinear_solver.spec.tolerance and dev_e < self.nonlinear_solver.spec.tolerance
           and status.well_residual < well_tolerance_coefficient * self.nonlinear_solver.spec.tolerance )
              or status.n_newton == self.nonlinear_solver.spec.max_iterations):
            if (i > 0):  # min_i_newton
                if i < max_newt:
                    converged = 1
                else:
                    converged = 0
                break

        r_code = self.e.solve_linear_equation()
        status.linear_solver_rc = r_code
        if r_code != 0:
            # failed linear solve: do NOT apply a stale update; fail the timestep
            converged = 0
            break
        status.n_linear += self.e.get_last_linear_iters()
        self.timer.node["newton update"].start()
        self.e.apply_newton_update(dt)
        self.timer.node["newton update"].stop()
        if i < max_newt:
            converged = 1

    # End of newton loop
    # NOTE: the old C++ pm/super_elastic post_newtonloop did not veto `converged`
    # (its residual re-check only selected a failure message), so the Python
    # verdict is passed through unchanged.
    converged = self.e.post_newtonloop(dt, t, converged)
    solver.stats.update(converged, status)
    self.timer.node['simulation'].stop()
    return converged

def run(model_folder, physics_type, is_finalize=True):
    m = Model(model_folder=model_folder, physics_type=physics_type, uniform_props=False)
    m.params.finalize_mpi = is_finalize
    m.init()
    #redirect_darts_output('log.txt')
    m.timer.node["update"] = timer_node()
    # Properties for writing to vtk format:
    m.output_directory = 'sol_cpp_' + physics_type + model_folder.split('data')[-1]

    # intialization:
    m.reservoir.set_equilibrium(zero_conduction=True)
    m.physics.engine.find_equilibrium = True
    dt_init = 1.e+8
    m.data_ts.dt_first = dt_init
    run_python(m, dt_init, init_step=True)
    m.reinit(zero_conduction=True)
    m.physics.engine.find_equilibrium = False

    size_report_step = 1
    max_dt = size_report_step
    m.max_dt = max_dt
    m.data_ts.dt_max = max_dt
    first_ts = size_report_step
    m.data_ts.dt_first = first_ts
    m.set_boundary_conditions_after_initialization()

    sim_time = 20 # days
    m.time_steps = []
    data = []
    # Run over all reporting time-steps:
    ith_step = 0
    while m.physics.engine.t < sim_time:
        run_python(m=m, days=size_report_step)
        m.reservoir.write_to_vtk(m.output_directory, ith_step + 1, m.physics.engine)
        ith_step += 1
        m.time_steps.append(m.physics.engine.t)
        data.append(m.get_performance_data(is_last_ts=(m.physics.engine.t >= sim_time)))

    m.print_timers()
    m.print_stat()

    return m, data

def test(mesh_type, physics_type, overwrite='0'):
    '''
    :param overwrite: write pkl file even if it exists
    :return: tuple (bool failed, float64 time)
    '''
    print('mesh_type:' + mesh_type, 'physics_type:' + physics_type, 'overwrite: ' + overwrite, sep=', ')
    import platform

    try:
        # if compiled with OpenMP, set to run with 1 thread, as mech tests are not working in the multithread version yet
        from darts.engines import set_num_threads
        set_num_threads(1)
    except:
        pass

    m, data = run(mesh_type, physics_type)

    pkl_suffix = ''
    if os.getenv('ODLS') != None and os.getenv('ODLS') == '-a':
        pkl_suffix = '_iter'
    file_name = os.path.join('ref', 'perf_' + platform.system().lower()[:3] + pkl_suffix +
                             '_' + mesh_type + '_' + physics_type + '.pkl')
    failed = 0

    is_plk_exist = os.path.isfile(file_name)
    if is_plk_exist:
        ref_data = m.load_performance_data(file_name=file_name)

    for ith_step, dt in enumerate(m.time_steps):
        if is_plk_exist:
            sol_data_step = data[ith_step]
            ref_data_step = ref_data[ith_step]
            failed += m.check_performance_data(ref_data_step, sol_data_step, failed, plot=False,
                                             png_suffix=mesh_type+'_'+physics_type+'_'+str(ith_step))

    if not is_plk_exist or overwrite == '1':
        m.save_performance_data(data=data, file_name=file_name)
        return False, 0.0

    if is_plk_exist:
        return (failed > 0), data[-1]['simulation time']
    else:
        return False, -1.0

def run_test(args: list = [], platform='cpu'):
    if len(args) == 3:
        return test(mesh_type=args[0], physics_type=args[1], overwrite=args[2])
    else:
        print('Wrong number of arguments provided to the run_test:', args)
        return 1, 0.0

if __name__ == '__main__':
    try:
        # if compiled with OpenMP, set to run with 1 thread, as mech tests are not working in the multithread version yet
        from darts.engines import set_num_threads
        set_num_threads(1)
    except:
        pass

    #test_all = False
    test_all = True
    physics_list = ['single_phase', 'single_phase_thermal', 'dead_oil', 'dead_oil_thermal']
    meshes_list = ['data_10_10_10', 'data_20_40_40']
    if test_all:
        is_finalize = False
        for physics in physics_list:
            for mesh in meshes_list:
                if physics == physics_list[-1] and mesh == meshes_list[-1]:
                    is_finalize = True
                run(model_folder=mesh, physics_type=physics, is_finalize=is_finalize)

    #run(model_folder='data_10_10_10', physics_type='single_phase')
    #run(model_folder='data_10_10_10', physics_type='single_phase_thermal')
    #run(model_folder='data_10_10_10', physics_type='dead_oil')
    #run(model_folder='data_10_10_10', physics_type='dead_oil_thermal')

    #run(model_folder='data_20_40_40', physics_type='single_phase')
    #run(model_folder='data_20_40_40', physics_type='single_phase_thermal')
    #run(model_folder='data_20_40_40', physics_type='dead_oil')
    #run(model_folder='data_20_40_40', physics_type='dead_oil_thermal')

    # runs and checks with the reference
    #test(mesh_type='data_10_10_10', physics_type='single_phase')
    #test(mesh_type='data_10_10_10', physics_type='single_phase_thermal')
    #test(mesh_type='data_10_10_10', physics_type='dead_oil_thermal')
