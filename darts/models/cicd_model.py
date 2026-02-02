import os
import pickle
import platform
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from darts.models.darts_model import DartsModel
from darts.tools.flux_tools import (
    get_molar_well_rates,
    get_phase_volumetric_well_rates,
)


# overwrite key to save results over existed
# diff_norm_normalized_tol defines tolerance for L2 norm of final solution difference , normalized by amount of blocks and variable range
# diff_abs_max_normalized_tol defines tolerance for maximum of final solution difference, normalized by variable range
# rel_diff_tol defines tolerance (in %) to a change in integer simulation parameters as linear and newton iterations
def check_performance(
    m: DartsModel,
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
    data_et = load_performance_data(perf_file, pkl_suffix=pkl_suffix)
    if data_et and not overwrite:
        data = get_performance_data(m)
        nb = m.reservoir.mesh.n_res_blocks
        nv = m.physics.n_vars

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
                    f"#{fail} solution check failed for variable {m.physics.vars[v]} "
                    f"(range {sol_range:f}): L2(diff)/len(diff)/range = {diff_norm_normalized:.2E} "
                    f"(tol {diff_norm_normalized_tol:.2E}), max(abs(diff))/range {diff_abs_max_normalized:.2E} "
                    f"(tol {diff_abs_max_normalized_tol:.2E}), max(abs(diff)) = {np.max(diff_abs):.2E}"
                )

                # plot the difference
                plt.figure()
                plt.plot(diff)
                plt.savefig('diff_' + m.physics.vars[v] + '.png')
                plt.close()

                # plot the reference and the current solution
                plt.figure()
                plt.plot(sol_et, label='ref')
                plt.plot(sol, label='cur')
                plt.legend()
                plt.savefig('sol_' + m.physics.vars[v] + '.png')
                plt.close()

        for key, value in sorted(data.items()):
            if key == 'solution' or type(value) is not int:
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
            print(f"OK, \t{m.timer.node['simulation'].get_timer():.2f} s")
            return 0
        else:
            print(f"FAIL, \t{m.timer.node['simulation'].get_timer():.2f} s")
            return 1
    else:
        save_performance_data(m, perf_file, pkl_suffix=pkl_suffix)
        print('SAVED PKL FILE', perf_file, pkl_suffix, file=sys.stderr)
        return 0


def get_performance_data(m: DartsModel):
    """
    Function to get the needed performance data

    :return: Performance data
    :rtype: dict
    """
    perf_data = dict()
    perf_data['solution'] = np.copy(m.physics.engine.X)
    perf_data['reservoir blocks'] = m.reservoir.mesh.n_res_blocks
    perf_data['variables'] = m.physics.n_vars
    perf_data['OBL resolution'] = m.physics.n_axes_points
    perf_data['operators'] = m.physics.n_ops
    perf_data['timesteps'] = m.physics.engine.stat.n_timesteps_total
    perf_data['wasted timesteps'] = m.physics.engine.stat.n_timesteps_wasted
    perf_data['newton iterations'] = m.physics.engine.stat.n_newton_total
    perf_data['wasted newton iterations'] = m.physics.engine.stat.n_newton_wasted
    perf_data['linear iterations'] = m.physics.engine.stat.n_linear_total
    perf_data['wasted linear iterations'] = m.physics.engine.stat.n_linear_wasted

    sim = m.timer.node['simulation']
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


def save_performance_data(m: DartsModel, file_name: str = '', pkl_suffix: str = ''):
    """
    Function to save performance data for future comparison.
    :param file_name:
    :return:
    """
    if file_name == '':
        file_name = os.path.join(
            'ref', 'perf_' + platform.system().lower()[:3] + pkl_suffix + '.pkl'
        )
    data = get_performance_data(m)
    with open(file_name, "wb") as fp:
        pickle.dump(data, fp, 4)


def compare_well_rates(m: DartsModel, time_data_filename: str):
    """
    Compares Python well rates against the rates calculated with legacy c++ function and stored in a given file
    :param time_data_filename: data filename
    :type time_data_filename: str
    """
    # load old well data
    old_data = pd.read_pickle(time_data_filename)
    # old_time = old_data['time'].to_numpy()

    # calculate new rates at all timesteps
    new_molar_rate = get_molar_well_rates(m)
    new_volumetric_rate = get_phase_volumetric_well_rates(m)
    # new_mass_rate = get_mass_well_rate(m, m.reservoir.wells[0])

    rtol = 1.0e-2
    atol = 0.1
    c_pattern = ' : c {} rate (Kmol/day)'
    p_pattern = ' : {} rate (m3/day)'

    # compare
    for well in m.reservoir.wells:
        # molar rates
        old_c = np.array(
            [
                old_data[well.name + c_pattern.format(c)].to_numpy()
                for c in range(m.physics.nc)
            ]
        ).T
        assert np.isclose(
            new_molar_rate[well.name][:, : m.physics.nc], -old_c, rtol=rtol, atol=atol
        ).all()

        # volumetric phase rates
        old_p = np.array(
            [
                old_data[well.name + p_pattern.format(m.physics.phases[p])].to_numpy()
                for p in range(m.physics.nph)
            ]
        ).T
        assert np.isclose(
            new_volumetric_rate[well.name], -old_p, rtol=rtol, atol=atol
        ).all()


def load_performance_data(file_name: str = '', pkl_suffix: str = ''):
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
        print('PKL FILE', file_name, 'does not exist. Skipping.', file=sys.stderr)
    return 0


def compare_solution_with_reference(m: DartsModel, pkl_custom_suffix=''):
    os.makedirs('ref', exist_ok=True)

    pkl_suffix = ''
    if os.getenv('TEST_GPU') is not None and os.getenv('TEST_GPU') == '1':
        pkl_suffix = '_gpu'
    elif os.getenv('ODLS') is not None and os.getenv('ODLS') == '-a':
        pkl_suffix = '_iter'
    else:
        pkl_suffix = '_odls'
    # print('pkl_suffix=', pkl_suffix)

    file_name = os.path.join(
        'ref',
        'perf_'
        + platform.system().lower()[:3]
        + pkl_suffix
        + pkl_custom_suffix
        + '.pkl',
    )
    overwrite = 0
    if os.getenv('UPLOAD_PKL') == '1':
        overwrite = 1

    is_plk_exist = os.path.isfile(file_name)

    failed = check_performance(
        m, perf_file=file_name, overwrite=overwrite, pkl_suffix=pkl_suffix
    )

    if not is_plk_exist or overwrite == '1':
        save_performance_data(m, file_name=file_name, pkl_suffix=pkl_suffix)

    if is_plk_exist:
        return (failed > 0), -1.0  # data[-1]['simulation time']
    else:
        return True, -1.0


def get_platform():
    platform = 'cpu'
    if os.getenv('TEST_GPU') is not None and os.getenv('TEST_GPU') == '1':
        platform = 'gpu'
    return platform


def is_iter_solvers():
    if (
        os.getenv('ODLS') is not None and os.getenv('ODLS') == '-a'
    ):  # run this case only for the build with iterative solvers
        return True
    return False


def pkl_suffix_solvers():
    return '_iter' if is_iter_solvers() else '_odls'


def set_one_thread():
    try:  # if compiled with OpenMP
        # set to run with 1 thread, as mech tests are not working in the multithread version yet
        from darts.engines import set_num_threads

        set_num_threads(1)
    except:
        pass


def reset_num_threads():
    try:  # if compiled with OpenMP
        from darts.engines import set_num_threads

        set_num_threads(int(os.environ['OMP_NUM_THREADS']))
    except:
        pass


def is_overwrite_pkl():
    # overwrite existing pkl files
    overwrite = '0'
    if os.getenv('UPLOAD_PKL') is not None and os.getenv('UPLOAD_PKL') == '1':
        overwrite = '1'
    return overwrite


def is_test_all_models():
    # run larger set of models (takes longer)
    test_all_models = False
    if os.getenv('TEST_ALL_MODELS') is not None and os.getenv('TEST_ALL_MODELS') == '1':
        test_all_models = True
    return test_all_models
