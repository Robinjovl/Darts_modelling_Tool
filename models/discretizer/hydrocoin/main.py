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
                  % (ts, t, dt, self.e.n_newton_last_dt, self.e.n_linear_last_dt))

            if self.e.n_newton_last_dt < 4:
                dt *= mult_dt
            if self.e.n_newton_last_dt < 3:
                good_ts_counter += 1
            else:
                good_ts_counter = 0
            if good_ts_counter > 5:
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
            #max_dt /= mult_dt
            print("Cut timestep to %f" % (dt))
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
    self.timer.node['simulation'].stop()
    return converged

def run(discr_type, mesh_file):
    redirect_darts_output('darts_log' + '.txt')

    m = Model(discr_type=discr_type, pres_in=50., mesh_file=mesh_file)

    # After constructing the model, the simulator needs to be initialized. The init() class method is called, which is
    # inherited (https://www.python-course.eu/python3_inheritance.php) from the parent class DartsModel (found in
    # darts/models/darts_model.py (NOTE: This is not the same as the__init__(self, **) method which each class (should)
    # have).
    m.init()

    # Specify some other time-related properties (NOTE: all time parameters are in [days])
    eps = 1e-6
    size_report_step = 5000.0  # Half Size of the reporting step (when output is writen to .vtk format)
    # num_report_steps = int(5.0 / size_report_step)
    max_dt = 2.0
    m.max_dt = max_dt
    m.params.max_ts = max_dt
    first_ts = 1.0
    m.params.first_ts = first_ts

    # Properties for writing to vtk format:
    # output_directory = 'trial_dir'  # Specify output directory here
    output_directory = 'sol_cpp_' + discr_type + '_{:s}'.format(m.physics_type)
    # Write to vtk using class methods of unstructured discretizer (uses within meshio write to vtk function):
    if discr_type == 'mpfa':
        m.reservoir.write_to_vtk(output_directory, m.cell_property, 0, m.physics)
    else:
        tot_unknws = m.reservoir.unstr_discr.fracture_cell_count + m.reservoir.unstr_discr.matrix_cell_count + len(m.reservoir.wells) * 2
        tot_properties = 2
        pressure_field = m.physics.engine.X[:-1:2]
        saturation_field = m.physics.engine.X[1::2]
        property_array = np.empty((tot_unknws, tot_properties))
        property_array[:, 0] = pressure_field
        property_array[:, 1] = saturation_field
        m.reservoir.unstr_discr.write_to_vtk(output_directory, property_array, m.cell_property, 0)

    # Run over all reporting time-steps:
    ith_step = 0
    #for ith_step in range(num_report_steps):
    while m.physics.engine.t < 30000:

        run_python(m, size_report_step)

        if discr_type == 'mpfa':
            m.reservoir.write_to_vtk(output_directory, m.cell_property, ith_step + 1, m.physics)
        else:
            pressure_field = m.physics.engine.X[:-1:2]
            saturation_field = m.physics.engine.X[1::2]
            property_array = np.empty((tot_unknws, tot_properties))
            property_array[:, 0] = pressure_field
            property_array[:, 1] = saturation_field
            m.reservoir.unstr_discr.write_to_vtk(output_directory, property_array, m.cell_property, ith_step+1)

        ith_step += 1

    # After the simulation, print some of the simulation timers and statistics,
    # newton iters, etc., how much time spent where:
    m.print_timers()
    m.print_stat()

## TPFA super-engine
# structured
# run(discr_type='tpfa', mesh_file='meshes/column_1d_tpfa.msh')
# unstructured
# run(discr_type='tpfa', mesh_file='meshes/column_tetra_tpfa.msh')
## MPFA super-engine
# structured
# run(discr_type='tpfa', mesh_file='meshes/column_1d.msh')
# unstructured
run(discr_type='mpfa', mesh_file='meshes/ccdfm_hydrocoin.msh')