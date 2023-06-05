from math import fabs
import pickle
import os
import numpy as np

from darts.engines import *
from darts.engines import print_build_info as engines_pbi
from darts.discretizer import print_build_info as discretizer_pbi
from darts.print_build_info import print_build_info as package_pbi


class DartsModel:
    """
    Base class with multiple functions

    """

    def __init__(self):
        """"
           Initialize DartsModel class.
        """
        # print out build information
        engines_pbi()
        discretizer_pbi()
        package_pbi()
        self.timer = timer_node()  # Create time_node object for time record
        self.timer.start()  # Start time record
        self.timer.node["simulation"] = timer_node()  # Create timer.node called "simulation" to record simulation time
        self.timer.node["newton update"] = timer_node()
        self.timer.node[
            "initialization"] = timer_node()  # Create timer.node called "initialization" to record initialization time
        self.timer.node["initialization"].start()  # Start recording "initialization" time

        self.params = sim_params()  # Create sim_params object to set simulation parameters

        self.timer.node["initialization"].stop()  # Stop recording "initialization" time


    def init(self):
        """
            Function to initialize the model, which includes:
                - initialize well (perforation) position
                - initialize well rate parameters
                - initialize reservoir condition
                - initialize well control settings
                - build accumulation_flux_operator_interpolator list
                - initialize engine
        """
        self.reservoir.init_wells()
        self.physics.init_wells(self.reservoir.wells)
        self.set_initial_conditions()
        self.set_boundary_conditions()
        self.set_op_list()
        self.reset()

    def reset(self):
        """
        Function to initialize the engine by calling 'init' method.
        """
        self.physics.engine.init(self.reservoir.mesh, ms_well_vector(self.reservoir.wells),
                                 op_vector(self.op_list),
                                 self.params, self.timer.node["simulation"])

    def set_initial_conditions(self):
        pass

    def set_boundary_conditions(self):
        pass

    def set_op_list(self):
        if type(self.physics.acc_flux_itor) == dict:
            self.op_list = [self.physics.acc_flux_itor[0], self.physics.acc_flux_w_itor]
            self.op_num = np.array(self.reservoir.mesh.op_num, copy=False)
            n_res = self.reservoir.mesh.n_res_blocks
            self.op_num[n_res:] = len(self.physics.acc_flux_itor.keys())
        else: # for backward compatibility
            self.op_list = [self.physics.acc_flux_itor]
            
    def run(self, days=0):
        if days:
            runtime = days
        else:
            runtime = self.runtime
        self.physics.engine.run(runtime)

    def run_python(self, days=0, restart_dt=0, timestep_python=False):
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
        if fabs(t) < 1e-15:
            dt = self.params.first_ts
        elif restart_dt > 0:
            dt = restart_dt
        else:
            dt = self.params.max_ts

        # evaluate end time
        runtime += t
        ts = 0

        while t < runtime:
            if timestep_python:
                 converged = self.e.run_timestep(dt, t)
            else:
                 converged = self.run_timestep_python(dt, t)

            if converged:
                t += dt
                ts = ts + 1
                print("# %d \tT = %3g\tDT = %2g\tNI = %d\tLI=%d"
                      % (ts, t, dt, self.e.n_newton_last_dt, self.e.n_linear_last_dt))

                dt *= mult_dt
                if dt > max_dt:
                    dt = max_dt

                if t + dt > runtime:
                    dt = runtime - t

            else:
                dt /= mult_dt
                print("Cut timestep to %2.3f" % dt)
                if dt < 1e-8:
                    break
        # update current engine time
        self.e.t = runtime

        print("TS = %d(%d), NI = %d(%d), LI = %d(%d)" % (self.e.stat.n_timesteps_total, self.e.stat.n_timesteps_wasted,
                                                         self.e.stat.n_newton_total, self.e.stat.n_newton_wasted,
                                                         self.e.stat.n_linear_total, self.e.stat.n_linear_wasted))

    def load_restart_data(self, filename='restart.pkl'):
        """
        Function to load data from previous simulation and uses them for following simulation.
        :param filename: restart_data filename
        """
        if os.path.exists(filename):
            with open(filename, "rb") as fp:
                data = pickle.load(fp)
                days, X, arr_n = data
                self.physics.engine.t = days
                self.physics.engine.X = value_vector(X)
                self.physics.engine.Xn = value_vector(X)
                self.physics.engine.op_vals_arr_n = value_vector(arr_n)

    def save_restart_data(self, filename='restart.pkl'):
        """
        Function to save the simulation data for restart usage.
        :param filename: Name of the file where restart_data stores.
        """
        t = np.copy(self.physics.engine.t)
        X = np.copy(self.physics.engine.X)
        arr_n = np.copy(self.physics.engine.op_vals_arr_n)
        data = [t, X, arr_n]
        with open(filename, "wb") as fp:
            pickle.dump(data, fp, 4)

    # overwrite key to save results over existed
    # diff_norm_normalized_tol defines tolerance for L2 norm of final solution difference , normalized by amount of blocks and variable range
    # diff_abs_max_normalized_tol defines tolerance for maximum of final solution difference, normalized by variable range
    # rel_diff_tol defines tolerance (in %) to a change in integer simulation parameters as linear and newton iterations
    def check_performance(self, overwrite=0, diff_norm_normalized_tol=1e-9, diff_abs_max_normalized_tol=1e-7,
                          rel_diff_tol=1, perf_file='', pkl_suffix=''):
        """
        Function to check the performance data to make sure whether the performance has been changed
        """
        fail = 0
        data_et = self.load_performance_data(perf_file, pkl_suffix=pkl_suffix)
        if data_et and not overwrite:
            data = self.get_performance_data()
            nb = self.reservoir.mesh.n_res_blocks
            nv = self.physics.n_vars

            # Check final solution - data[0]
            # Check every variable separately
            for v in range(nv):
                sol_et = data_et['solution'][v:nb * nv:nv]
                diff = data['solution'][v:nb * nv:nv] - sol_et
                sol_range = np.max(sol_et) - np.min(sol_et)
                diff_abs = np.abs(diff)
                diff_norm = np.linalg.norm(diff)
                diff_norm_normalized = diff_norm / len(sol_et) / sol_range
                diff_abs_max_normalized = np.max(diff_abs) / sol_range
                if diff_norm_normalized > diff_norm_normalized_tol or diff_abs_max_normalized > diff_abs_max_normalized_tol:
                    fail += 1
                    print(
                        '#%d solution check failed for variable %s (range %f): L2(diff)/len(diff)/range = %.2E (tol %.2E), max(abs(diff))/range %.2E (tol %.2E), max(abs(diff)) = %.2E' \
                        % (fail, self.physics.vars[v], sol_range, diff_norm_normalized, diff_norm_normalized_tol,
                           diff_abs_max_normalized, diff_abs_max_normalized_tol, np.max(diff_abs)))
            for key, value in sorted(data.items()):
                if key == 'solution' or type(value) != int:
                    continue
                reference = data_et[key]

                if reference == 0:
                    if value != 0:
                        print('#%d parameter %s is %d (was 0)' % (fail, key, value))
                        fail += 1
                else:
                    rel_diff = (value - data_et[key]) / reference * 100
                    if abs(rel_diff) > rel_diff_tol:
                        print('#%d parameter %s is %d (was %d, %+.2f%%)' % (fail, key, value, reference, rel_diff))
                        fail += 1
            if not fail:
                print('OK, \t%.2f s' % self.timer.node['simulation'].get_timer())
                return 0
            else:
                print('FAIL, \t%.2f s' % self.timer.node['simulation'].get_timer())
                return 1
        else:
            self.save_performance_data(perf_file, pkl_suffix=pkl_suffix)
            print('SAVED')
            return 0

    def get_performance_data(self):
        """
        Function to get the needed performance data
        """
        perf_data = dict()
        perf_data['solution'] = np.copy(self.physics.engine.X)
        perf_data['reservoir blocks'] = self.reservoir.mesh.n_res_blocks
        perf_data['variables'] = self.physics.n_vars
        perf_data['OBL resolution'] = self.physics.n_points
        perf_data['operators'] = self.physics.n_ops
        perf_data['timesteps'] = self.physics.engine.stat.n_timesteps_total
        perf_data['wasted timesteps'] = self.physics.engine.stat.n_timesteps_wasted
        perf_data['newton iterations'] = self.physics.engine.stat.n_newton_total
        perf_data['wasted newton iterations'] = self.physics.engine.stat.n_newton_wasted
        perf_data['linear iterations'] = self.physics.engine.stat.n_linear_total
        perf_data['wasted linear iterations'] = self.physics.engine.stat.n_linear_wasted

        sim = self.timer.node['simulation']
        jac = sim.node['jacobian assembly']
        perf_data['simulation time'] = sim.get_timer()
        perf_data['linearization time'] = jac.get_timer()
        perf_data['linear solver time'] = sim.node['linear solver solve'].get_timer() + sim.node[
            'linear solver setup'].get_timer()
        interp = jac.node['interpolation']
        perf_data['interpolation incl. generation time'] = interp.get_timer()

        return perf_data

    def save_performance_data(self, file_name='', pkl_suffix=''):
        import platform
        """
        Function to save performance data for future comparison.
        :param file_name:
        :return:
        """
        if file_name == '':
            file_name = 'perf_' + platform.system().lower()[:3] + pkl_suffix +'.pkl'
        data = self.get_performance_data()
        with open(file_name, "wb") as fp:
            pickle.dump(data, fp, 4)

    @staticmethod
    def load_performance_data(file_name='', pkl_suffix = ''):
        import platform
        """
        Function to load the performance pkl file at previous simulation.
        :param file_name: performance filename
        """
        if file_name == '':
            file_name = 'perf_' + platform.system().lower()[:3] + pkl_suffix + '.pkl'
        if os.path.exists(file_name):
            with open(file_name, "rb") as fp:
                return pickle.load(fp)
        return 0

    def print_timers(self):
        """
        Function to print the time information, including total time elapsed,
                                        time consumption at different stages of the simulation, etc..
        """
        print(self.timer.print("", ""))

    def print_stat(self):
        """
        Function to print the statistics information, including total timesteps, Newton iteration, linear iteration, etc..
        """
        self.physics.engine.print_stat()

    def export_vtk(self, file_name='data', local_cell_data={}, global_cell_data={}, vars_data_dtype=np.float32,
                   export_grid_data=True):

        # get current engine time
        t = self.physics.engine.t
        nb = self.reservoir.mesh.n_res_blocks
        nv = self.physics.n_vars
        X = np.array(self.physics.engine.X, copy=False)

        for v in range(nv):
            local_cell_data[self.physics.vars[v]] = X[v:nb * nv:nv].astype(vars_data_dtype)

        self.reservoir.export_vtk(file_name, t, local_cell_data, global_cell_data, export_grid_data)

    # destructor to force to destroy all created C objects and free memory
    def __del__(self):
        for name in list(vars(self).keys()):
            delattr(self, name)


    def run_timestep_python(self, dt, t):
        max_newt = self.params.max_i_newton
        max_residual = np.zeros(max_newt + 1)
        self.e.n_linear_last_dt = 0
        well_tolerance_coefficient = 1e2
        self.timer.node['simulation'].start()
        for i in range(max_newt+1):
            self.e.run_single_newton_iteration(dt)
            self.e.newton_residual_last_dt = self.e.calc_newton_residual()

            max_residual[i] = self.e.newton_residual_last_dt
            counter = 0
            for j in range(i):
                if abs(max_residual[i] - max_residual[j])/max_residual[i] < 1e-3:
                    counter += 1
            if counter > 2:
                print("Stationary point detected!")
                break

            self.e.well_residual_last_dt = self.e.calc_well_residual()
            self.e.n_newton_last_dt = i
            #  check tolerance if it converges
            if ((self.e.newton_residual_last_dt < self.params.tolerance_newton and
                 self.e.well_residual_last_dt < well_tolerance_coefficient * self.params.tolerance_newton) or
                    self.e.n_newton_last_dt == self.params.max_i_newton):
                if i > 0:  # min_i_newton
                    break
            r_code = self.e.solve_linear_equation()
            self.timer.node["newton update"].start()
            self.e.apply_newton_update(dt)
            self.timer.node["newton update"].stop()
        # End of newton loop
        converged = self.e.post_newtonloop(dt, t)
        self.timer.node['simulation'].stop()
        return converged

