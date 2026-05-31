import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

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
        is_gpu = os.environ.get('TEST_GPU', 0)
        if (
            nt > 1 or is_gpu
        ):  # use a looser tolerance for comparison of multithreaded/gpu run
            diff_norm_normalized_tol = 1e-2
            diff_abs_max_normalized_tol = 1e-1
            rel_diff_tol = 100

        fail = 0
        data_et = self.load_performance_data(perf_file, pkl_suffix=pkl_suffix)
        if data_et and not overwrite:
            data = self.get_performance_data()
            n_res_blocks = self.reservoir.mesh.n_res_blocks
            nv = self.physics.n_vars
            ref_solution = np.asarray(data_et['solution'])
            cur_solution = np.asarray(data['solution'])  # current solution
            ref_res_blocks = data_et.get('reservoir blocks', n_res_blocks)
            cur_total_blocks = data.get('total blocks', cur_solution.size // nv)
            ref_total_blocks = data_et.get('total blocks', ref_solution.size // nv)
            cur_well_blocks = cur_total_blocks - n_res_blocks
            ref_well_blocks = ref_total_blocks - ref_res_blocks

            if cur_solution.size != cur_total_blocks * nv:
                fail += 1
                print(
                    f"#{fail} current solution size {cur_solution.size:d} is not equal to "
                    f"total blocks * variables ({cur_total_blocks:d} * {nv:d})"
                )
            if ref_solution.size != ref_total_blocks * nv:
                fail += 1
                print(
                    f"#{fail} reference solution size {ref_solution.size:d} is not equal to "
                    f"total blocks * variables ({ref_total_blocks:d} * {nv:d})"
                )
            if cur_well_blocks < 0:
                fail += 1
                print(
                    f"#{fail} current well block count is negative: "
                    f"total blocks {cur_total_blocks:d}, reservoir blocks {n_res_blocks:d}"
                )
            if ref_well_blocks < 0:
                fail += 1
                print(
                    f"#{fail} reference well block count is negative: "
                    f"total blocks {ref_total_blocks:d}, reservoir blocks {ref_res_blocks:d}"
                )

            def check_solution_block_range(
                block_label: str,
                cur_start_block: int,
                ref_start_block: int,
                cur_block_count: int,
                ref_block_count: int,
            ) -> None:
                """
                Compare one block range in the current and reference solution vectors.

                :param block_label: Label used in failure messages and output plots.
                :param cur_start_block: First block index in the current solution vector.
                :param ref_start_block: First block index in the reference solution vector.
                :param cur_block_count: Number of current blocks to compare.
                :param ref_block_count: Number of reference blocks to compare.
                """
                nonlocal fail
                if cur_block_count != ref_block_count:
                    fail += 1
                    print(
                        f"#{fail} solution check failed for {block_label} blocks: "
                        f"current block count {cur_block_count:d}, reference block count {ref_block_count:d}"
                    )
                    return
                if cur_block_count <= 0:
                    return

                cur_stop = (cur_start_block + cur_block_count) * nv
                ref_stop = (ref_start_block + ref_block_count) * nv
                for v in range(nv):
                    sol_et = ref_solution[ref_start_block * nv + v : ref_stop : nv]
                    sol = cur_solution[cur_start_block * nv + v : cur_stop : nv]
                    diff = sol - sol_et
                    sol_range = np.max(sol_et) - np.min(sol_et)
                    diff_abs = np.abs(diff)
                    diff_norm = np.linalg.norm(diff)
                    # Constant reference values have no meaningful range for normalization.
                    # Use their magnitude scale instead of eps, and skip the L2 check for
                    # this case because it is too sensitive for small well-state vectors.
                    sol_scale = max(np.max(np.abs(sol_et)), 1.0)
                    min_range = np.finfo(float).eps * sol_scale
                    use_range = np.isfinite(sol_range) and sol_range > min_range
                    denom = sol_range if use_range else sol_scale
                    diff_norm_normalized = diff_norm / (len(sol_et) * denom)
                    diff_abs_max_normalized = np.max(diff_abs) / denom
                    if use_range:
                        is_failed = (
                            diff_norm_normalized > diff_norm_normalized_tol
                            or diff_abs_max_normalized > diff_abs_max_normalized_tol
                        )
                        scale_label = "range"
                    else:
                        is_failed = (
                            diff_abs_max_normalized > diff_abs_max_normalized_tol
                        )
                        scale_label = "scale"
                    if is_failed:
                        fail += 1
                        print(
                            f"#{fail} solution check failed for {block_label} variable {self.physics.vars[v]} "
                            f"(range {sol_range:f}): L2(diff)/len(diff)/{scale_label} = {diff_norm_normalized:.2E} "
                            f"(tol {diff_norm_normalized_tol:.2E}), max(abs(diff))/{scale_label} {diff_abs_max_normalized:.2E} "
                            f"(tol {diff_abs_max_normalized_tol:.2E}), max(abs(diff)) = {np.max(diff_abs):.2E}"
                        )

                        # plot the difference
                        plt.figure()
                        plt.plot(diff)
                        plt.xlabel(f'{block_label} block index')
                        plt.ylabel(f'{self.physics.vars[v]} difference')
                        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
                        plt.savefig(f'diff_{block_label}_{self.physics.vars[v]}.png')
                        plt.close()

                        # plot the reference and the current solution
                        plt.figure()
                        plt.plot(sol_et, label='ref')
                        plt.plot(sol, label='cur')
                        plt.legend()
                        plt.xlabel(f'{block_label} block index')
                        plt.ylabel(self.physics.vars[v])
                        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
                        plt.savefig(f'sol_{block_label}_{self.physics.vars[v]}.png')
                        plt.close()

            if not fail:
                # Check reservoir solution
                check_solution_block_range(
                    'reservoir', 0, 0, n_res_blocks, ref_res_blocks
                )
                # Check well solution
                check_solution_block_range(
                    'well',
                    n_res_blocks,
                    ref_res_blocks,
                    cur_well_blocks,
                    ref_well_blocks,
                )

            for key, value in sorted(data.items()):
                if key == 'solution' or not isinstance(value, int):
                    continue
                # Skip keys added after older reference files were generated
                if key not in data_et:
                    continue
                reference = data_et[key]

                if reference == 0:
                    if value != 0:
                        print(f"#{fail} parameter {key} is {value:d} (was 0)")
                        fail += 1
                else:
                    abs_diff = value - reference
                    rel_diff = abs_diff / reference * 100
                    abs_diff_tol = (
                        max(int(rel_diff_tol / 100 * reference), 1)
                        if (nt > 1 or is_gpu)
                        else 0
                    )
                    if not (
                        abs(rel_diff) <= rel_diff_tol or abs(abs_diff) <= abs_diff_tol
                    ):
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
        Get the needed performance data

        :return: Performance data
        :rtype: dict
        """
        perf_data = dict()
        perf_data['solution'] = np.copy(self.physics.engine.X)
        perf_data['reservoir blocks'] = self.reservoir.mesh.n_res_blocks
        perf_data['total blocks'] = self.reservoir.mesh.n_blocks
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
        Save performance data for future comparison.
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
        Load the performance pkl file at previous simulation.
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
