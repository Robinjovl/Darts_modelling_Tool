# Section of the Python code where we import all dependencies on third party Python modules/libaries or our own
# libraries (exposed C++ code to Python, i.e. darts.engines && darts.physics)
from model import Model
import numpy as np
import meshio
from darts.engines import redirect_darts_output

def run_python(self, days=0, restart_dt=0, log_3d_body_path=0):
    if days:
        runtime = days
    else:
        runtime = self.runtime

    mult_dt = self.params.mult_ts
    max_dt = self.params.max_ts
    self.e = self.physics.engine

    # get current engine time
    t = self.e.t

    # same logic as in engine.run
    if np.fabs(t) < 1e-15:
        dt = self.params.first_ts
    elif restart_dt > 0:
        dt = restart_dt
    else:
        dt = self.params.max_ts

    # evaluate end time
    runtime += t
    ts = 0
    #
    if log_3d_body_path and self.physics.n_vars == 3:
        self.body_path_start()

    good_ts_counter = 0
    while t < runtime:
        # if t == 0:
        #     self.full_arr_operator = np.array(self.e.op_vals_arr)
        # else:
        #     self.full_arr_operator = np.vstack((self.full_arr_operator, np.array(self.e.op_vals_arr)))
        # np.savetxt("Operator_analytical.csv", self.full_arr_operator.T, delimiter=",")

        converged = run_timestep_python(self, dt, t)
        if converged:
            t += dt
            ts = ts + 1
            print("# %d \tT = %f\tDT = %f\tNI = %d\tLI=%d"
                  % (ts, t * 86400, dt * 86400, self.e.n_newton_last_dt, self.e.n_linear_last_dt))

            if self.e.n_newton_last_dt < 3:
                dt *= mult_dt
            if self.e.n_newton_last_dt < 3:
                good_ts_counter += 1
            else:
                good_ts_counter = 0
            if good_ts_counter > 7:
                good_ts_counter = 0
                max_dt *= mult_dt
                self.params.max_ts = max_dt
            if dt > max_dt:
                dt = max_dt

            if t + dt > runtime:
                dt = runtime - t

            if log_3d_body_path and self.physics.n_vars == 3:
                self.body_path_add_bodys(t)
                nb_begin = self.reservoir.nx * self.reservoir.ny * (self.body_path_map_layer - 1) * 3
                nb_end = self.reservoir.nx * self.reservoir.ny * (self.body_path_map_layer) * 3

                self.save_matlab_map(self.body_path_axes[0] + '_ts_' + str(ts), self.e.X[nb_begin:nb_end:3])
                self.save_matlab_map(self.body_path_axes[1] + '_ts_' + str(ts), self.e.X[nb_begin + 1:nb_end:3])
                self.save_matlab_map(self.body_path_axes[2] + '_ts_' + str(ts), self.e.X[nb_begin + 2:nb_end:3])
        else:
            dt /= 2 * mult_dt
            max_dt /= mult_dt
            self.params.max_ts = max_dt
            print("Cut timestep to %f" % (dt * 86400.0))
            # if dt < 1e-8:
            #     break

    # update current engine time
    self.e.t = runtime

    print("TS = %d(%d), NI = %d(%d), LI = %d(%d)" % (self.e.stat.n_timesteps_total, self.e.stat.n_timesteps_wasted,
                                                     self.e.stat.n_newton_total, self.e.stat.n_newton_wasted,
                                                     self.e.stat.n_linear_total, self.e.stat.n_linear_wasted))
def run_timestep_python(self, dt, t):
    # max_newt = self.params.max_i_newton
    max_newt = 100
    self.e.n_linear_last_dt = 0
    well_tolerance_coefficient = 1e2
    self.timer.node['simulation'].start()
    for i in range(max_newt+1):
        self.e.run_single_newton_iteration(dt)
        self.e.newton_residual_last_dt = self.e.calc_newton_residual()
        #if i == 0 and self.e.newton_residual_last_dt > 5.E-5:
        #    self.e.newton_residual_last_dt = 1.0
        #    break
        self.e.well_residual_last_dt = self.e.calc_well_residual()
        self.e.n_newton_last_dt = i
        print('matrix_res = ' + str(self.e.newton_residual_last_dt) + '\t' + 'well_res = ' + str(self.e.well_residual_last_dt))
        #  check tolerance if it converges
        if ((self.e.newton_residual_last_dt < self.params.tolerance_newton and self.e.well_residual_last_dt < well_tolerance_coefficient * self.params.tolerance_newton )
                or self.e.n_newton_last_dt == self.params.max_i_newton):
            if (i > 0):  # min_i_newton
                break
        r_code = self.e.solve_linear_equation()
        self.timer.node["newton update"].start()
        self.e.apply_newton_update(dt)
        self.timer.node["newton update"].stop()
    # End of newton loop
    converged = self.e.post_newtonloop(dt, t)
    #self.reservoir.write_to_vtk_mpfa('sol_cpp_mpfa_compositional', self.cell_property, 0, self.physics)
    self.timer.node['simulation'].stop()
    return converged

def run(discr_type, init_filename=None):
    redirect_darts_output('darts_log' + '.txt')

    # reservoir conditions
    #m = Model(discr_type=discr_type, problem_type='reservoir', mpfa_type='tpfa', init_filename=init_filename, temp_in=320, pres_in=100)
    # benchmark conditions
    m = Model(discr_type=discr_type, problem_type='surface', mpfa_type='mpfa', init_filename=init_filename, temp_in=293., pres_in=1.01325)
    m.init()
    output_directory = 'sol_cpp_' + discr_type + '_{:s}'.format(m.physics_type)

    # Specify some other time-related properties (NOTE: all time parameters are in [days])
    size_report_step = 3600 / 86400.0#1.0 / 86400.0  # Half Size of the reporting step (when output is writen to .vtk format)
    # num_report_steps = int(5.0 / size_report_step)
    max_dt = 0.00001 / 86400.0
    m.max_dt = max_dt
    m.params.max_ts = max_dt
    first_ts = 1 / 86400 / 1.E+6
    m.params.first_ts = first_ts

    # hydrostatic equilibrium is not achieved here because of poorly permeable layers
    if not init_filename:
        run_python(m, 1.e-6)
    if discr_type == 'mpfa':
        m.reservoir.write_to_vtk_mpfa(output_directory, m.cell_property, 0, m.output_props, m.physics)
    else:
        tot_unknws = m.reservoir.unstr_discr.fracture_cell_count + m.reservoir.unstr_discr.matrix_cell_count + len(m.reservoir.wells) * 2
        tot_properties = 2
        pressure_field = m.physics.engine.X[:-1:2]
        saturation_field = m.physics.engine.X[1::2]
        property_array = np.empty((tot_unknws, tot_properties))
        property_array[:, 0] = pressure_field
        property_array[:, 1] = saturation_field
        m.reservoir.write_to_vtk_tpfa(output_directory, property_array, m.cell_property, 0)

    # run injector
    pressure = np.array(m.physics.engine.X, copy=False)[::2]
    if not init_filename:
        m.reservoir.wells[0].control = m.physics.new_rate_inj(m.inj_rate, m.inj_stream, 0)
        #m.reservoir.wells[1].control = m.physics.new_bhp_prod(pressure[m.reservoir.well_cells[1]] - 0.3)

    # Run over all reporting time-steps:
    ith_step = 0
    event1 = True
    event2 = True
    eps = 1e-6
    while m.physics.engine.t < 30:

        # if m.physics.engine.t > 0.09375 - eps and event1:
        #     print("Injector 2 started")
        #     m.reservoir.wells[1].control = m.physics.new_rate_inj(m.inj_rate, m.inj_stream, 0)
        #     event1 = False

        if not init_filename and m.physics.engine.t > 0.20833 - eps and event2:
            print("Both injectors stopped")
            m.reservoir.wells[0].control = m.physics.new_rate_inj(0, m.inj_stream, 0)
            #m.reservoir.wells[1].control = m.physics.new_rate_inj(0, m.inj_stream, 0)
            event2 = False

        run_python(m, size_report_step)

        if discr_type == 'mpfa':
            m.reservoir.write_to_vtk_mpfa(output_directory, m.cell_property, ith_step + 1, m.output_props, m.physics)
        else:
            pressure_field = m.physics.engine.X[:-1:2]
            saturation_field = m.physics.engine.X[1::2]
            property_array = np.empty((tot_unknws, tot_properties))
            property_array[:, 0] = pressure_field
            property_array[:, 1] = saturation_field
            m.reservoir.write_to_vtk_tpfa(output_directory, property_array, m.cell_property, ith_step+1)

        ith_step += 1

    # After the simulation, print some of the simulation timers and statistics,
    # newton iters, etc., how much time spent where:
    m.print_timers()
    m.print_stat()

# discr_type - 'mpfa' or 'tpfa'
run(discr_type='mpfa')#, init_filename='tpfa_14.vtk')