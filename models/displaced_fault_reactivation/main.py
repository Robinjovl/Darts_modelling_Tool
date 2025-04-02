from model import Model
from darts.engines import *
import numpy as np
import meshio
import os
from math import fabs

try:
    # if compiled with OpenMP, set to run with 1 thread, as mech tests are not working in the multithread version yet
    from darts.engines import set_num_threads
    set_num_threads(1)
except:
    pass

def run_python(m, days=0, restart_dt=0, log_3d_body_path=0, init_step = False):
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
    if init_step:
        dt = days
    else:
        dt = min(max_dt, days)

    # evaluate end time
    runtime += t
    ts = 0
    m.ith_step_ready_for_reinjection = 0

    while t < runtime:
        if init_step:   new_time = t
        else:           new_time = t + dt

        if not init_step:
            m.timer.node["update"].start()
            # store boundaries taken at previous time step
            m.reservoir.update(dt=dt, time=new_time)
            # evaluate and assign transient boundaries or sources / sinks
            #if res == -1:
            #    return -1
                #new_time -= dt
                #dt = m.reservoir.max_dt_change
                #continue
            # update transient boundaries or sources / sinks
            m.reservoir.update_trans(dt, m.physics.engine.X)
            m.timer.node["update"].stop()

        converged = run_timestep_python(m, dt, t)

        if converged:
            t += dt
            ts = ts + 1
            print("# %d \tT = %3g\tDT = %2g\tNI = %d\tLI=%d"
                   % (ts, t, dt, m.e.n_newton_last_dt, m.e.n_linear_last_dt))
            if not init_step:
                m.reservoir.write_to_vtk(m.output_directory, m.ith_step + 1, m.physics.engine, dt)
                m.ith_step += 1
                if m.ith_step > 1000:
                    os._exit()

            if m.physics.engine.n_newton_last_dt < 4:
                dt *= 1.5
            if dt > max_dt:
               dt = max_dt

            if t + dt > runtime:
               dt = runtime - t
        else:
            new_time -= dt
            if dt / mult_dt > 1.e-8 / 86400:
                dt /= mult_dt

            if dt < 1.e-2 / 86400.0 and m.physics.engine.momentum_inertia == 0.0 and m.turn_on_dynamic_mode: # less than smth -> go to fully dynamic (implicit) stepping
                m.physics.engine.momentum_inertia = 2406.0
                dt = 5.e-4 / 86400 # 500 microseconds
                max_dt = 5.e-4 / 86400 # 500 microseconds
                m.physics.engine.active_linear_solver_id = 1
                print("Fully dynamic mode enabled!!!")

            print("Cut timestep to %.5e" % dt)

        max_slip_area = np.max(np.array(m.slip_area))
        if m.ith_step + 1 > 500 and m.slip_area[-1] < 0.005 * max_slip_area and m.turn_on_dynamic_mode and \
                m.ith_step_ready_for_reinjection == 0:
            m.ith_step_ready_for_reinjection = m.ith_step

        # if m.ith_step + 1 > 500 and m.slip_area[-1] < 0.005 * max_slip_area and m.turn_on_dynamic_mode and \
        #         m.ith_step - m.ith_step_ready_for_reinjection > 500:
        #     m.physics.engine.momentum_inertia = 0.0
        #     dt = 0.001
        #     m.params.max_ts = max_dt = 0.005
        #     m.turn_on_dynamic_mode = False
        #     m.reservoir.wells[0].control = m.physics.new_rate_prod(0.0)
        #     #X = np.array(m.physics.engine.X, copy = False)
        #     m.reservoir.wells[1].control = m.physics.new_bhp_inj(m.reservoir.p_init[m.id_inj])
            # m.physics.engine.active_linear_solver_id = 0
        #     print("Fully dynamic mode disabled!!!")


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
    res_history = []

    for i in range(max_newt + 1):
        self.e.assemble_linear_system(dt)
        res = self.e.calc_newton_dev()
        self.e.dev_p = res[0]
        self.e.dev_u = res[1]
        if len(res) > 2 and res[2] == res[2]:       self.e.dev_g = res[2]
        else:                                       self.e.dev_g = 0.0
        res_history.append((res[0], res[1], res[2]))

        self.e.newton_residual_last_dt = np.sqrt(self.e.dev_u ** 2 + self.e.dev_p ** 2 + self.e.dev_g ** 2)
        #self.e.newton_residual_last_dt = self.e.calc_newton_residual()
        self.e.well_residual_last_dt = self.e.calc_well_residual()
        print(str(i) + ': ' + 'rp = ' + str(self.e.dev_p) + '\t' + 'ru = ' + str(self.e.dev_u) + '\t' + \
                    'rg = ' + str(self.e.dev_g) + '\t' + 'rwell = ' + str(self.e.well_residual_last_dt) + '\t' + 'CFL = ' + str(self.e.CFL_max))
        self.e.n_newton_last_dt = i
        #  check tolerance if it converges
        if ((self.e.dev_p < self.params.tolerance_newton and
             self.e.dev_u < self.params.tolerance_newton and
             self.e.dev_g < self.params.tolerance_newton and
             self.e.well_residual_last_dt < well_tolerance_coefficient * self.params.tolerance_newton )
              or self.e.n_newton_last_dt == self.params.max_i_newton):
            if (i > 0):  # min_i_newton
                if i < max_newt:
                    converged = 1
                else:
                    converged = 0
                break
        if self.e.dev_g > m.cut_off_gap_residual:
            converged = 0
            print('Contact residual cut-off exceeded!!!')
            break

        if False and i > 1 and self.e.dev_g > m.cut_off_gap_residual:
            coef = np.array([0.0, 1.0])
            history = np.array([res_history[-2], res_history[-1]])
            linear_search(self, dt, coef, history)
            converged = 1
        else:
            r_code = self.e.solve_linear_equation()
            self.timer.node["newton update"].start()
            self.e.apply_newton_update(dt)
            self.timer.node["newton update"].stop()
            if i < max_newt:
                converged = 1

    if not hasattr(m, 'slip_area'):
        m.slip_area = [0.0]

    areas = calc_slip_area(m)
    cur_area = sum(areas)
    print('slip area = ' + str(cur_area))
    # print(areas)
    if cur_area - m.slip_area[-1] > 4.2 * m.min_area:
        converged *= 0
    else:
        m.slip_area.append(cur_area)
        converged *= 1

    converged = self.e.post_newtonloop(dt, t, converged)

    self.timer.node['simulation'].stop()
    return converged
def linear_search(self, dt, coef, history):
    print('LS: ' + str(coef[0]) + '\t' + 'res_p = ' + str(history[0][0]) + '\tres_u = ' + str(history[0][1]) + '\tres_g = ' + str(history[0][2]))
    print('LS: ' + str(coef[1]) + '\t' + 'res_p = ' + str(history[1][0]) + '\tres_u = ' + str(history[1][1]) + '\tres_g = ' + str(history[1][2]))
    res_history = np.array([history[0][2], history[1][2]])

    for iter in range(5):
        if coef.size > 2:
            id = res_history.argmin()
            closest_left = np.where(coef < coef[id])[0]
            closest_right = np.where(coef > coef[id])[0]
            if closest_left.size and closest_right.size:
                left = closest_left[coef[closest_left].argmax()]
                right = closest_right[coef[closest_right].argmin()]
                if res_history[left] < res_history[id]:
                    coef = np.append(coef, (coef[id] + coef[left]) / 2)
                elif res_history[right] < res_history[id]:
                    coef = np.append(coef, (coef[id] + coef[right]) / 2)
                else:
                    if res_history[left] < res_history[right]:
                        coef = np.append(coef, coef[id] - (coef[id] - coef[left]) / 4)
                    else:
                        coef = np.append(coef, coef[id] + (coef[right] - coef[id]) / 4)
            elif closest_left.size:
                left = closest_left[coef[closest_left].argmax()]
                if res_history[left] < res_history[id]:
                    coef = np.append(coef, (coef[id] + coef[left]) / 2)
                else:
                    coef = np.append(coef, coef[id] + (coef[id] - coef[left]) / 2)
            elif closest_right.size:
                right = closest_right[coef[closest_right].argmin()]
                if res_history[right] < res_history[id]:
                    coef = np.append(coef, (coef[id] + coef[right]) / 2)
                else:
                    coef = np.append(coef, coef[id] - (coef[right] - coef[id]) / 2)
            if coef[-1] <= 0: coef[-1] = 1.E-2
            if coef[-1] >= 1: coef[-1] = 1.0 - 1.E-2
        else:
            coef = np.append(coef, 0.01)#coef[-1] / 2)

        self.e.newton_update_coefficient = coef[-1] - coef[-2]
        self.e.apply_newton_update(dt)
        self.e.run_single_newton_iteration(dt)
        res = self.e.calc_newton_dev()
        res_history = np.append(res_history, res[2])
        print('LS: ' + str(coef[-1]) + '\t' + 'res_p = ' + str(res[0]) + \
                                            '\tres_u = ' + str(res[1]) + \
                                            '\tres_g = ' + str(res[2]))

    final_id = res_history.argmin()
    self.e.newton_update_coefficient = coef[final_id] - coef[-1]
    self.e.apply_newton_update(dt)
def calc_slip_area(m):
    areas = []
    dz = np.max(m.reservoir.unstr_discr.mesh_data.points[:,2]) - np.min(m.reservoir.unstr_discr.mesh_data.points[:,2])
    for contact in m.physics.engine.contacts:
        cell_ids = np.array(contact.cell_ids, copy=True)
        for i in range(cell_ids.size):
            if contact.states[i] == contact_state.SLIP:
                areas.append(m.reservoir.unstr_discr.faces[cell_ids[i]][4].area / dz)
    return areas
def just_run():
    #t0 = 0.000125
    nt = 100
    max_t = 365
    #t = np.logspace(1, np.log10(max_t), nt)
    t = 5.0 * np.ones(nt)
    #t = np.append(t, 1.E-3 / 86400.0 * np.ones(nt))
    m = Model()
    m.init()
    redirect_darts_output('log.txt')
    m.output_directory = 'sol_{:s}'.format(m.physics_type)
    m.timer.node["update"] = timer_node()
    m.ith_step = 0  # Store initial conditions as ../solution0.vtk
    #m.physics.engine.print_linear_system = True

    # for the estimation of max slip area increase
    m.min_area = 1.e10
    for contact in m.physics.engine.contacts:
        cell_ids = np.array(contact.cell_ids, copy=True)
        for i in range(cell_ids.size):
            m.min_area = min(m.min_area, m.reservoir.unstr_discr.faces[cell_ids[i]][4].area)
    m.min_area /= np.max(m.reservoir.unstr_discr.mesh_data.points[:,2]) - np.min(m.reservoir.unstr_discr.mesh_data.points[:,2])
    print('Min area = ' + str(m.min_area))

    #X = np.array(m.physics.engine.X, copy=False)
    # find equilibrium
    m.reservoir.set_equilibrium()
    m.physics.engine.find_equilibrium = True
    m.physics.engine.print_linear_system = False
    m.physics.engine.scale_rows = True
    m.physics.engine.scale_dimless = False
    # # scaled unknowns
    # m.physics.engine.x_dim = 1.e-6
    # m.physics.engine.p_dim = 1.0
    # m.physics.engine.t_dim = 1.0
    # m.physics.engine.m_dim = 1.0
    m.turn_on_dynamic_mode = True
    if m.turn_on_dynamic_mode:
        m.cut_off_gap_residual = 0.01# if self.e.momentum_inertia else 0.01
    else:
        m.cut_off_gap_residual = 100.0

    m.params.first_ts = 1.0
    run_python(m, 1.0, init_step=True)
    # m.physics.engine.momentum_inertia = 2406.0
    m.reinit(zero_conduction=True)
    m.physics.engine.dt1 = 0.0

    m.physics.engine.find_equilibrium = False
    m.reservoir.apply_geomehcanics_mode(m.physics)
    m.physics.engine.contact_solver = contact_solver.RETURN_MAPPING # local_iterations#flux_from_previous_iteration#return_mapping
    m.setup_contact_friction(m.physics.engine.contact_solver)
    if m.params.linear_type == sim_params.cpu_gmres_fs_cpr:
        m.physics.engine.update_uu_jacobian()

    m.physics.engine.t = 0.0
    time = 0
    #m.params.first_ts = 1.0#0.1
    for ith_step, dt in enumerate(t):
        time += dt
        # if time > 100.0:
        #     m.reservoir.wells[0].control = m.physics.new_bhp_inj(m.reservoir.p_init[m.id_prod])
        m.params.max_ts = dt
        m.params.mult_ts = 10.0
        run_python(m, dt)
        ith_step += 1
    m.print_timers()

just_run()