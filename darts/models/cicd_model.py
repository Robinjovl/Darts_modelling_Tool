from darts.models.darts_model import DartsModel
import numpy as np
import pickle
import os


class CICDModel(DartsModel):
    def __init__(self):
        super().__init__()

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

        :return: Performance data
        :rtype: dict
        """
        perf_data = dict()
        perf_data['solution'] = np.copy(self.engine.X)
        perf_data['reservoir blocks'] = self.reservoir.mesh.n_res_blocks
        perf_data['variables'] = self.physics.n_vars
        perf_data['OBL resolution'] = self.physics.n_points
        perf_data['operators'] = self.physics.n_ops
        perf_data['timesteps'] = self.engine.stat.n_timesteps_total
        perf_data['wasted timesteps'] = self.engine.stat.n_timesteps_wasted
        perf_data['newton iterations'] = self.engine.stat.n_newton_total
        perf_data['wasted newton iterations'] = self.engine.stat.n_newton_wasted
        perf_data['linear iterations'] = self.engine.stat.n_linear_total
        perf_data['wasted linear iterations'] = self.engine.stat.n_linear_wasted

        sim = self.timer.node['simulation']
        jac = sim.node['jacobian assembly']
        perf_data['simulation time'] = sim.get_timer()
        perf_data['linearization time'] = jac.get_timer()
        perf_data['linear solver time'] = sim.node['linear solver solve'].get_timer() + sim.node[
            'linear solver setup'].get_timer()
        interp = jac.node['interpolation']
        perf_data['interpolation incl. generation time'] = interp.get_timer()

        return perf_data

    def save_performance_data(self, file_name: str = '', pkl_suffix: str = ''):
        import platform
        """
        Function to save performance data for future comparison.
        :param file_name:
        :return:
        """
        if file_name == '':
            file_name = 'perf_' + platform.system().lower()[:3] + pkl_suffix + '.pkl'
        data = self.get_performance_data()
        with open(file_name, "wb") as fp:
            pickle.dump(data, fp, 4)

    @staticmethod
    def load_performance_data(file_name: str = '', pkl_suffix: str = ''):
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
