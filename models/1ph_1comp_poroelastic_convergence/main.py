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
                if m.reservoir.thermoporoelasticity:
                    m.reservoir.update_mech_discretizer_thermoporoelasticity(time=new_time)
                else:
                    m.reservoir.update_mech_discretizer_poroelasticity(time=new_time)
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
        dev_e = 0
        if self.reservoir.thermoporoelasticity:
            self.e.dev_e = res[2]
            dev_e = res[2]

        self.e.newton_residual_last_dt = np.sqrt(self.e.dev_u ** 2 + self.e.dev_p ** 2 + dev_e ** 2)
        #self.e.newton_residual_last_dt = self.e.calc_newton_residual()
        self.e.well_residual_last_dt = self.e.calc_well_residual()
        print(str(i) + ': ' + 'rp = ' + str(self.e.dev_p) + '\t' + 'ru = ' + str(self.e.dev_u) + '\t' + \
                    're = ' + str(dev_e) + '\t' + 'rwell = ' + str(self.e.well_residual_last_dt) + '\t' + 'CFL = ' + str(self.e.CFL_max))

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

def run_single_resolution(timestep, n_steps, mesh_file, discretizer='pm_discretizer', mode='poroelastic', is_last_model=False):
    t = timestep * np.ones(n_steps)
    m = Model(discretizer=discretizer, mode=mode, mesh_file=mesh_file)
    m.params.finalize_mpi = is_last_model
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
    return m.reservoir.calc_deviations(m.engine)

def run_convergence_study(n_res, discretizer, mode, mesh='rect'):
    max_t = 0.1
    timesteps = np.array([0.1, 0.05, 0.025, 0.0125])
    nt = np.array(max_t / timesteps, dtype=np.int32)

    if mesh == 'rect':
        dx = 1.0 / np.array([4.0, 8.0, 16.0, 32.0])
        mesh_file_template = 'meshes/unit_trans_{}.msh'
    elif mesh == 'tetra':
        dx = 1.0 / np.array([2.0, 4.0, 8.0, 16.0])
        mesh_file_template = 'meshes/unit_tetra_{}.msh'

    devs_u = []
    devs_p = []
    devs_s = []
    devs_seff = []
    devs_v = []
    devs_t = []
    for i in range(n_res):
        print('Run model with resolution #' + str(i))

        mesh_file = mesh_file_template.format(i)
        if mode == 'thermoporoelastic':
            dev_u, dev_p, dev_s, dev_seff, dev_v, dev_t = run_single_resolution(timestep=timesteps[i], n_steps=nt[i],
                                                mesh_file=mesh_file, discretizer=discretizer,
                                                mode=mode, is_last_model=(i == n_res - 1))
            devs_t.append(dev_t)
        else:
            dev_u, dev_p, dev_s, dev_seff, dev_v = run_single_resolution(timestep=timesteps[i], n_steps=nt[i],
                                                mesh_file=mesh_file, discretizer=discretizer,
                                                mode=mode, is_last_model=(i == n_res - 1))

        devs_u.append(dev_u)
        devs_p.append(dev_p)
        devs_s.append(dev_s)
        devs_seff.append(dev_seff)
        devs_v.append(dev_v)

    devs_u = np.array(devs_u)
    devs_p = np.array(devs_p)
    devs_s = np.array(devs_s)
    devs_seff = np.array(devs_seff)
    devs_v = np.array(devs_v)

    x = np.sqrt((timesteps * dx)[:n_res])
    id = np.argsort(x)
    u_order = (np.diff(np.log(devs_u[id])) / np.diff(np.log(x[id])))[0]
    p_order = (np.diff(np.log(devs_p[id])) / np.diff(np.log(x[id])))[0]
    s_order = (np.diff(np.log(devs_s[id])) / np.diff(np.log(x[id])))[0]
    seff_order = (np.diff(np.log(devs_seff[id])) / np.diff(np.log(x[id])))[0]
    v_order = (np.diff(np.log(devs_v[id])) / np.diff(np.log(x[id])))[0]

    print('dev_u')
    print(devs_u)
    print('u_order = ' + str(u_order))

    print('dev_p')
    print(devs_p)
    print('p_order = ' + str(p_order))

    if mode == 'thermoporoelastic':
        devs_t = np.array(devs_t)
        t_order = (np.diff(np.log(devs_t[id])) / np.diff(np.log(x[id])))[0]
        print('dev_t')
        print(devs_t)
        print('t_order = ' + str(t_order))

    print('dev_s')
    print(devs_s)
    print('s_order = ' + str(s_order))

    print('dev_seff')
    print(devs_seff)
    print('seff_order = ' + str(seff_order))

    print('dev_v')
    print(devs_v)
    print('v_order = ' + str(v_order))

    assert(u_order > 1.0)
    if mode == 'poroelastic':
        assert(p_order > 1.0)
        assert (s_order > 0.5)  # and s_eff_order > 0.5) # TODO: fix stresses in thermoporoelastic mode
        if discretizer == 'mech_discretizer': # TODO: fix Darcy velocity in mech_operators
            assert(v_order > 0.5)
    else:
        assert(t_order > 1.0)

def run_test(args: list = []):
    run_convergence_study(n_res=3, discretizer='pm_discretizer', mode='poroelastic')
    run_convergence_study(n_res=3, discretizer='mech_discretizer', mode='poroelastic', mesh='rect')
    run_convergence_study(n_res=3, discretizer='mech_discretizer', mode='poroelastic', mesh='tetra')
    run_convergence_study(n_res=3, discretizer='mech_discretizer', mode='thermoporoelastic', mesh='rect')
    run_convergence_study(n_res=3, discretizer='mech_discretizer', mode='thermoporoelastic', mesh='tetra')

# run_convergence_study(n_res=3, discretizer='pm_discretizer', mode='poroelastic')
# run_convergence_study(n_res=3, discretizer='mech_discretizer', mode='poroelastic', mesh='rect')
# run_convergence_study(n_res=3, discretizer='mech_discretizer', mode='poroelastic', mesh='tetra')
# run_convergence_study(n_res=3, discretizer='mech_discretizer', mode='thermoporoelastic', mesh='rect')
# run_convergence_study(n_res=3, discretizer='mech_discretizer', mode='thermoporoelastic', mesh='tetra')