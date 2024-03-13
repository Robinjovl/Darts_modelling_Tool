from model import Model
import numpy as np
from darts.engines import redirect_darts_output, timer_node

def run_python(m, days=0, restart_dt=0, init_step = False):
    if days:
        runtime = days
    else:
        runtime = m.runtime

    mult_dt = m.params.mult_ts
    max_dt = m.params.max_ts
    m.e = m.engine

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
            m.reservoir.update_trans(dt, m.engine.X)
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
        self.e.run_single_newton_iteration(dt)
        res = self.e.calc_newton_dev()#self.e.calc_newton_residual()
        self.e.dev_p = res[0]
        self.e.dev_u = res[1]
        if len(res) > 2 and res[2] == res[2]:       self.e.dev_g = res[2]
        else:                                       self.e.dev_g = 0.0

        self.e.newton_residual_last_dt = np.sqrt(self.e.dev_u ** 2 + self.e.dev_p ** 2 + self.e.dev_g ** 2)
        #self.e.newton_residual_last_dt = self.e.calc_newton_residual()
        self.e.well_residual_last_dt = self.e.calc_well_residual()
        print(str(i) + ': ' + 'rp = ' + str(self.e.dev_p) + '\t' + 'ru = ' + str(self.e.dev_u) + '\t' + \
                    'rg = ' + str(self.e.dev_g) + '\t' + 'rwell = ' + str(self.e.well_residual_last_dt) + '\t' + 'CFL = ' + str(self.e.CFL_max))

        self.e.n_newton_last_dt = i
        #  check tolerance if it converges
        if ((self.e.dev_p < self.params.tolerance_newton and self.e.dev_u < self.params.tolerance_newton and self.e.dev_g < self.params.tolerance_newton
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

def run(model_folder, physics_type):
    m = Model(model_folder=model_folder, physics_type=physics_type)
    m.init()
    redirect_darts_output('log.txt')
    m.timer.node["update"] = timer_node()
    # Properties for writing to vtk format:
    m.output_directory = 'sol_cpp' + model_folder.split('data')[-1]

    # intialization: may use find_equilibrium and zero_fluid_transmissibilities=True
    m.reservoir.set_equilibrium(zero_fluid_transmissibilities=False)
    # m.engine.find_equilibrium = True
    dt_init = 1.e+8
    m.params.first_ts = dt_init
    run_python(m, dt_init, init_step=True)
    m.reinit()
    # m.engine.find_equilibrium = False
    m.reservoir.turn_off_equilibrium(zero_fluid_transmissibilities=False)
    m.engine.t = 0.0

    size_report_step = 100
    max_dt = size_report_step
    m.max_dt = max_dt
    m.params.max_ts = max_dt
    first_ts = size_report_step
    m.params.first_ts = first_ts

    # Run over all reporting time-steps:
    ith_step = 0
    while m.engine.t < 2000:
        run_python(m=m, days=size_report_step)
        m.reservoir.write_to_vtk(m.output_directory, ith_step + 1, m.engine)
        ith_step += 1

    m.print_timers()
    m.print_stat()

run(model_folder='meshes/data_10_10_10', physics_type='single_phase')
# run(model_folder='meshes/data_10_10_10', physics_type='dead_oil')