from model import Model
from darts.engines import *
import numpy as np
import meshio
from math import fabs

def run_python(m, days=0, restart_dt=0, log_3d_body_path=0, init_step = False):
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
    if fabs(t) < 1e-15:
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
            if m.discretizer_name == 'pm_discretizer':
                m.reservoir.update_pm_discretizer(time=new_time)
            elif m.discretizer_name == 'mech_discretizer':
                m.reservoir.update_mech_discretizer(time=new_time)
            # update transient boundaries or sources / sinks
            m.reservoir.update_trans(dt, m.engine.X)
            m.timer.node["update"].stop()

        converged = run_timestep_python(m, dt, t)
        if converged:
            t += dt
            ts = ts + 1
            print("# %d \tT = %3g\tDT = %2g\tNI = %d\tLI=%d"
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

def run_single_resolution(timestep, n_steps, mesh_file, discretizer='pm_discretizer'):
    t = timestep * np.ones(n_steps)
    m = Model(discretizer=discretizer, mesh_file=mesh_file)
    m.init()
    redirect_darts_output('log.txt')
    m.output_directory = 'sol_{:s}'.format(m.physics_type)
    m.timer.node["update"] = timer_node()

    ith_step = 0
    m.engine.t = 0.0
    time = 0
    for ith_step, dt in enumerate(t):
        time += dt
        m.params.first_ts = dt
        m.params.max_ts = dt
        run_python(m, dt)

        # m.reservoir.write_to_vtk(m.output_directory, ith_step + 1, m.engine)
        # m.reservoir.write_diff_to_vtk(output_directory, property_array, m.cell_property, ith_step + 1, time)
        #m.reservoir.write_pm_conn_to_file(t_step=ith_step + 1)
    return m.reservoir.calc_deviations(m.engine)

def run_convergence_study(n_res, discretizer, mesh='rect'):
    max_t = 0.1
    timesteps = np.array([0.1, 0.05, 0.025, 0.0125])
    nt = np.array(max_t / timesteps, dtype=np.int32)
    dx = 1.0 / np.array([4.0, 8.0, 16.0, 32.0])

    if mesh == 'rect':
        mesh_file_template = 'meshes/unit_trans_{}.msh'

    devs_u = []
    devs_p = []
    for i in range(n_res):
        print('Run model with resolution #' + str(i))

        mesh_file = mesh_file_template.format(i)
        dev_u, dev_p = run_single_resolution(timestep=timesteps[i], n_steps=nt[i],
                                             mesh_file=mesh_file, discretizer=discretizer)
        devs_u.append(dev_u)
        devs_p.append(dev_p)

    devs_u = np.array(devs_u)
    devs_p = np.array(devs_p)

    x = np.sqrt((timesteps * dx)[:n_res])
    id = np.argsort(x)
    print(id)
    u_order = (np.diff(np.log(devs_u[id])) / np.diff(np.log(x[id])))[0]
    p_order = (np.diff(np.log(devs_p[id])) / np.diff(np.log(x[id])))[0]

    print('dev_u')
    print(devs_u)
    print('u_order = ' + str(u_order))
    print('dev_p')
    print(devs_p)
    print('p_order = ' + str(p_order))

    assert(u_order > 1.0 and p_order > 1.0)

# run_convergence_study(n_res=1, discretizer='pm_discretizer')
run_convergence_study(n_res=3, discretizer='mech_discretizer')