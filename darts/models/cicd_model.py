import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from darts.models.darts_model import DartsModel


class CICDModel(DartsModel):
    def __init__(self):
        super().__init__()

    # overwrite key to save results over existed
    # diff_norm_normalized_tol defines tolerance for L2 norm of final solution difference , normalized by amount of blocks and variable range
    # diff_abs_max_normalized_tol defines tolerance for maximum of final solution difference, normalized by variable range
    # rel_diff_tol defines tolerance (in %) to a change in integer simulation parameters as linear and newton iterations
    def check_performance(
        self,
        overwrite=0,
        diff_norm_normalized_tol_=1e-9,
        diff_abs_max_normalized_tol_=1e-7,
        rel_diff_tol_=1,
        perf_file='',
        pkl_suffix='',
    ):
        """
        Function to check the performance data to make sure whether the performance has been changed
        """
        diff_norm_normalized_tol = diff_norm_normalized_tol_
        diff_abs_max_normalized_tol = diff_abs_max_normalized_tol_
        rel_diff_tol = rel_diff_tol_
        nt = int(os.environ.get('OMP_NUM_THREADS', 1))
        if nt > 1:  # use a looser tolerance for comparison of multithreaded run
            diff_norm_normalized_tol = 1e-2
            diff_abs_max_normalized_tol = 1e-1
            rel_diff_tol = 100

        fail = 0
        data_et = self.load_performance_data(perf_file, pkl_suffix=pkl_suffix)
        if data_et and not overwrite:
            data = self.get_performance_data()
            nb = self.reservoir.mesh.n_res_blocks
            nv = self.physics.n_vars

            # Check final solution - data[0]
            # Check every variable separately
            for v in range(nv):
                sol_et = data_et['solution'][v : nb * nv : nv]
                sol = data['solution'][v : nb * nv : nv]
                diff = sol - sol_et
                sol_range = np.max(sol_et) - np.min(sol_et)
                diff_abs = np.abs(diff)
                diff_norm = np.linalg.norm(diff)
                denom = (
                    sol_range
                    if np.isfinite(sol_range) and sol_range != 0
                    else np.finfo(float).eps
                )
                diff_norm_normalized = diff_norm / (len(sol_et) * denom)
                diff_abs_max_normalized = np.max(diff_abs) / denom
                if (
                    diff_norm_normalized > diff_norm_normalized_tol
                    or diff_abs_max_normalized > diff_abs_max_normalized_tol
                ):
                    fail += 1
                    print(
                        f"#{fail} solution check failed for variable {self.physics.vars[v]} "
                        f"(range {sol_range:f}): L2(diff)/len(diff)/range = {diff_norm_normalized:.2E} "
                        f"(tol {diff_norm_normalized_tol:.2E}), max(abs(diff))/range {diff_abs_max_normalized:.2E} "
                        f"(tol {diff_abs_max_normalized_tol:.2E}), max(abs(diff)) = {np.max(diff_abs):.2E}"
                    )

                    # plot the difference
                    plt.figure()
                    plt.plot(diff)
                    plt.savefig('diff_' + self.physics.vars[v] + '.png')
                    plt.close()

                    # plot the reference and the current solution
                    plt.figure()
                    plt.plot(sol_et, label='ref')
                    plt.plot(sol, label='cur')
                    plt.savefig('sol_' + self.physics.vars[v] + '.png')
                    plt.close()

            for key, value in sorted(data.items()):
                if key == 'solution' or not isinstance(value, int):
                    continue
                reference = data_et[key]

                if reference == 0:
                    if value != 0:
                        print(f"#{fail} parameter {key} is {value:d} (was 0)")
                        fail += 1
                else:
                    rel_diff = (value - data_et[key]) / reference * 100
                    if abs(rel_diff) > rel_diff_tol:
                        print(
                            f"#{fail} parameter {key} is {value:d} (was {reference:d}, {rel_diff:+.2f}%)"
                        )
                        fail += 1
            if not fail:
                print(f"OK, \t{self.timer.node['simulation'].get_timer():.2f} s")
                return 0
            else:
                print(f"FAIL, \t{self.timer.node['simulation'].get_timer():.2f} s")
                return 1
        else:
            self.save_performance_data(perf_file, pkl_suffix=pkl_suffix)
            print('SAVED PKL FILE', perf_file, pkl_suffix)
            return 0

    def get_performance_data(self):
        """
        Function to get the needed performance data

        :return: Performance data
        :rtype: dict
        """
        perf_data = dict()
        perf_data['solution'] = np.copy(self.physics.engine.X)
        perf_data['reservoir blocks'] = self.reservoir.mesh.n_res_blocks
        perf_data['variables'] = self.physics.n_vars
        perf_data['OBL resolution'] = self.physics.n_axes_points
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
        perf_data['linear solver time'] = (
            sim.node['linear solver solve'].get_timer()
            + sim.node['linear solver setup'].get_timer()
        )
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
            file_name = os.path.join(
                'ref', 'perf_' + platform.system().lower()[:3] + pkl_suffix + '.pkl'
            )
        data = self.get_performance_data()
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
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
            file_name = os.path.join(
                'ref', 'perf_' + platform.system().lower()[:3] + pkl_suffix + '.pkl'
            )
        if os.path.exists(file_name):
            with open(file_name, "rb") as fp:
                return pickle.load(fp)
        else:
            print('PKL FILE', file_name, 'does not exist. Skipping.')
        return 0
