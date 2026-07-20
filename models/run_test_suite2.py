import os
import shutil
import subprocess
import sys
from contextlib import redirect_stdout

from darts.engines import sim_params
from darts.engines import print_build_info as engines_pbi
from compare_well_time_series import (
    compare_generated_well_time_series,
    create_well_time_series_snapshot,
    get_pkl_suffix,
)
from for_each_model import (
    abort_redirection,
    for_each_model,
    for_each_model_adjoint,
    redirect_all_output,
    run_tests,
)


def _ensure_parent_dir(path):
    """Create parent directory for the provided file path if missing."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def _normalize_odls_env():
    """
    Infer iterative-solver runs from the default CPU solver when ODLS is unset.

    Manual `darts run_test_suite2.py ...` runs no longer export ODLS, while the
    default solver still differs between ODLS and iterative builds. The testsuite
    naming and a few solver selections still rely on ODLS, so synthesize it here
    when the build clearly defaults to an iterative CPU solver.
    """
    if os.getenv('TEST_GPU') == '1':
        return False
    if os.getenv('ODLS') is None:
        try:
            if sim_params().linear_type != sim_params.cpu_superlu:
                os.environ['ODLS'] = '-a'
        except Exception:
            pass
    return os.getenv('ODLS') == '-a'


def _pkl_suffix():
    return get_pkl_suffix()

def run_testing(platform, overwrite, iter_solvers, test_all_models):
    base_dir = os.getcwd()  # base directory is models/
    logs_dir = os.path.join(base_dir, "_logs")  # directory in which log files will be saved
    os.makedirs(logs_dir, exist_ok=True)

    model_dir = os.path.abspath(r'.')
    _ensure_parent_dir(os.path.join(model_dir, '_logs', 'placeholder'))

    # set model list to run

    accepted_dirs = [
        '2ph_comp',
        '2ph_comp_solid',
        '2ph_do',
        '2ph_geothermal',
        '2ph_geothermal_mass_flux',
        '2ph_hysteresis',
        '3ph_comp_w',
        '3ph_do',
        '3ph_bo',
        'Uniform_Brugge',
        'Chem_benchmark_new',
        #'CO2_foam_CCS',
        # 'GeoRising' runs below as parametrized PT/PH variants (accepted_dirs_variants)
        'CoaxWell',
        'effect_of_potential_energy',
    ]

    if platform == 'cpu':
        accepted_dirs += [
            # MPFA code is excluded from gpu build due to compilation issues (c++ std 20)
            '2ph_do_thermal_mpfa',
            # 2ph_do_thermal doesn't converge well, so we skip it on GPU
            '2ph_do_thermal',
        ]

        # Tests for drift-flux well model (DFM) (implemented only for CPU)
        accepted_dirs += [
            # Coupled well-reservoir modeling using DFM wells is
            os.path.join('dfm_well', '2ph_1comp_coupled_dfm_well_reservoir'),
            # Single-phase thermal well flow in a DFM well
            os.path.join('dfm_well', '1ph_1comp_thermal_dfm_well_vs_dwell'),
            # Two-phase isothermal well flow in a DFM well
            os.path.join('dfm_well', '2ph_2comp_isothermal_dfm_vertical_well_vs_dwell'),
        ]

    test_dirs_mech = ['1ph_1comp_poroelastic_analytics']
    test_args_mech = []
    for case in ['terzaghi', 'mandel', 'terzaghi_two_layers', 'bai']:
        for discr_name in ['mech_discretizer', 'pm_discretizer']:
            if case == 'bai' and discr_name == 'pm_discretizer':
                continue # is not supported by poroelastic as bai is thermoporoelasticity
            for mesh in ['rect', 'wedge', 'hex']:
                if case == 'terzaghi_two_layers' and mesh == 'hex':
                    continue
                test_args_mech.append([case, discr_name, mesh])

    test_dirs_mech += ['1ph_1comp_poroelastic_convergence']
    test_args_mech = [test_args_mech, [['']]]  # no args for the convergence test

    if iter_solvers:
        test_dirs_mech += ['SPE10_mech']
        physics_list = ['single_phase', 'single_phase_thermal', 'dead_oil', 'dead_oil_thermal']
        meshes_list = ['data_10_10_10']
        test_args_mech_spe10 = []
        for physics in physics_list:
            for mesh in meshes_list:
                test_args_mech_spe10.append([mesh, physics])
        test_args_mech += [test_args_mech_spe10]

        test_dirs_mech += ['displaced_fault_reactivation']
        test_args_fault = []
        config = {'mode': 'quasi_static',
                  'timesteps': [1.0],
                  'depletion': {'mode': 'uniform', 'value': -250.0},
                  'friction_law': 'static',
                  'mesh_file': 'meshes/new_setup_coarse.geo',
                  'cache_discretizer': False}
        config[0] = config['friction_law']  # to make work arg[0] in for_each_model
        test_args_fault += [config]
        config = {'mode': 'quasi_static',
                  'timesteps': [1.0],
                  'depletion': {'mode': 'uniform', 'value': -172.4},  # -172.685 is more precise, requires finer mesh
                  'friction_law': 'slip_weakening',
                  'mesh_file': 'meshes/new_setup_coarse.geo',
                  'cache_discretizer': False}
        config[0] = config['friction_law']  # to make work arg[0] in for_each_model
        test_args_fault += [config]
        test_args_mech += [test_args_fault]

    # CPG (C++ discr)
    test_dirs_cpg = ['cpg_sloping_fault']
    cpg_cases_list = ['generate_5x3x4']
    if iter_solvers:  # run this case only for the build with iterative solvers
        cpg_cases_list += ['generate_51x51x1', '40x40x10', '40x40x10_hcap', '40x40x10_regions']
    test_args_cpg = []
    for case_geom in cpg_cases_list:
        for physics_type in ['geothermal', 'deadoil']:
            # 'wperiodic' variant disabled: the zero-rate "stop" control makes the well
            # block singular for the CPR preconditioner (CPU/GPU) -> "Matrix D can't be
            # inversed"; it only completes on ODLS. Skipped until the well setup or CPR
            # robustness is fixed.
            for wctrl in ['wrate', 'wbhp']:
                if physics_type == 'deadoil' and wctrl == 'wrate':
                    continue  # TODO fix convergence
                case = case_geom + '_' + wctrl
                test_args_cpg.append([case, physics_type])
    test_args_cpg = [test_args_cpg]

    # DFN (python discr)
    test_dirs_dfn = ['fracture_network']
    test_cases_dfn = ['case_1']
    if test_all_models:
        test_cases_dfn += ['case_4', 'case_5']
        #test_cases_dfn += ['whitby', 'case_3', 'case_1_burden_O1', 'case_1_burden_O2']
        #test_cases_dfn += ['case_1_burden_U1', 'case_1_burden_U2', 'case_1_burden_O1_U1', 'case_1_burden_O2_U2']
    test_args_dfn = []
    for case in test_cases_dfn:
        test_args_dfn.append([case])
    test_args_dfn = [test_args_dfn]

    # chemistry tests (multiple cases within a single model folder)
    test_dirs_chem = [os.path.join('chemistry', 'carbonated_water')]
    test_args_chem = [[
        {
            'name': 'cal_phreeqc_phreeqc_1D',
            'domain': '1D',
            'nx': 200,
            'minerals': ['calcite', 'dolomite'],
            'kinetic_mechanisms': ['acidic', 'neutral', 'carbonate'],
            'n_obl_mult': 1,
            'co2_injection': 0.1,
            'max_ts': 1.e-3,
            'flash': 'phreeqc',
            'database': 'phreeqc',
            'output': False,
        },
    ]]

    # for adjoint test
    accepted_dirs_adjoint = ['Adjoint_super_engine', 'Adjoint_PXflash_geothermal']
    if platform == 'cpu':  # MPFA code is excluded from gpu build due to compilation issues (c++ std 20)
        accepted_dirs_adjoint += ['Adjoint_mpfa']

    # Parametrized model.py runs: the same model folder is tested in several
    # formulations, each producing/comparing its own reference pkl. Entries are
    # (directory, proc_kwargs) tuples passed through for_each_model to check_performance.
    accepted_dirs_variants = [
        ('GeoRising', {'formulation': 'PT'}),
        ('GeoRising', {'formulation': 'PH'}),
    ]

    # RUN
    failed_models_m = []
    n_total = 0
    # run tests accepted_dirs/model.py (+ parametrized variants) with comparison of pkl files
    if len(accepted_dirs) or len(accepted_dirs_variants):
        failed_models_m = for_each_model(model_dir, check_performance,
                                         accepted_dirs + accepted_dirs_variants)
    n_total_m = len(accepted_dirs) + len(accepted_dirs_variants)
    n_total += n_total_m

    # check main.py files and compare well time-series pkl files when they are produced
    failed_models_main = []
    accepted_dirs += ['CCS']
    if iter_solvers:  # run this case only for the build with iterative solvers
        accepted_dirs += [ 'SPE11b']
    n_total_mainpy = 0
    models_root = model_dir
    for mdir in accepted_dirs:
        print('running main.py for model', mdir)
        n_total_mainpy += 1
        model_path = os.path.join(models_root, mdir)
        if not os.path.isdir(model_path):
            print(f'SKIP: directory "{model_path}" not found')
            failed_models_main += [mdir + ' (main.py missing dir)']
            continue
        os.chdir(model_path)
        safe_mdir = mdir.replace(os.sep, '__')
        stdout_path = os.path.join(logs_dir, safe_mdir + '_mainpy.log')
        stderr_path = os.path.join(logs_dir, safe_mdir + '_mainpy_err.log')
        _ensure_parent_dir(stdout_path)
        _ensure_parent_dir(stderr_path)
        well_time_series_snapshot = create_well_time_series_snapshot(model_path)
        with open(stdout_path, 'w') as stdout_file, open(stderr_path, 'w') as stderr_file:
            mrun = subprocess.run(["python", "main.py", platform], stdout=stdout_file, stderr=stderr_file)
            rcode = mrun.returncode
        failed_well_time_series = 0
        n_well_time_series = 0
        skipped_well_time_series = False
        if not rcode:
            with open(stdout_path, 'a') as stdout_file:
                print('\nWell time-series comparison:', file=stdout_file)
                with redirect_stdout(stdout_file):
                    failed_well_time_series, n_well_time_series, skipped_well_time_series = compare_generated_well_time_series(
                        model_path,
                        well_time_series_snapshot,
                        overwrite=overwrite,
                        pkl_suffix=_pkl_suffix(),
                    )
        if not rcode and not failed_well_time_series:
            if skipped_well_time_series:
                print('OK (main.py ran without errors; well time-series comparison skipped for multithread run)')
            elif n_well_time_series:
                if str(overwrite) == '1':
                    print('OK (main.py ran without errors; well time-series reference saved)')
                else:
                    print('OK (main.py ran without errors; well time-series comparison passed)')
            else:
                print('OK (main.py ran without errors; no well time-series generated)')
        else:
            if rcode:
                print(f'FAIL (main.py exited with code {rcode}); see {stdout_path} and {stderr_path}')
            if failed_well_time_series:
                print(f'FAIL (well time-series comparison); see {stdout_path}')
            print('FAIL')
            failed_models_main += [mdir + ' (main.py)']
        os.chdir(models_root)
    n_total += n_total_mainpy

    # discretizer tests
    print('\nDiscretizer tests:')
    n_total_discr, failed_models_cpg = run_tests(model_dir, test_dirs=test_dirs_cpg, test_args=test_args_cpg, overwrite=overwrite, platform=platform)
    n_total += n_total_discr

    # fracture network tests
    print('\nFracture network tests:')
    n_total_dfn, failed_models_dfn = run_tests(model_dir, test_dirs=test_dirs_dfn, test_args=test_args_dfn, overwrite=overwrite, platform=platform)
    n_total += n_total_dfn

    # chemistry tests
    print('\nChemistry tests:')
    n_total_chem, failed_models_chem = run_tests(model_dir, test_dirs=test_dirs_chem, test_args=test_args_chem, overwrite=overwrite, platform=platform)
    n_total += n_total_chem

    # poromechanic tests
    print('\nPoromechanics tests:')
    n_total_mech = 0
    failed_models_mech = []
    if platform == 'cpu':  # mech code is excluded from gpu build due to compilation issues (c++ std 20)
        n_total_mech, failed_models_mech = run_tests(model_dir, test_dirs=test_dirs_mech, test_args=test_args_mech, overwrite=overwrite)
    n_total += n_total_mech

    # test for adjoint ------------------start---------------------------------
    print('\nAdjoint tests:')
    failed_models_adj = []
    if len(accepted_dirs_adjoint):
        failed_models_adj = for_each_model_adjoint(model_dir, check_performance_adjoint, accepted_dirs_adjoint)
    n_total_adj = len(accepted_dirs_adjoint)
    n_total += n_total_adj
    # test for adjoint ------------------end---------------------------------

    failed_models = failed_models_m + failed_models_main + failed_models_cpg + failed_models_dfn + \
                    failed_models_mech + failed_models_adj + failed_models_chem
    print('Failed models   :\n\t', '\n\t'.join(failed_models))

    n_failed =  len(failed_models)
    n_passed = n_total - n_failed

    print('Number of failed models by types:')
    print('\tmodel.py', len(failed_models_m))
    print('\tmain.py', len(failed_models_main))
    print('\tcpg', len(failed_models_cpg))
    print('\tdfn', len(failed_models_dfn))
    print('\tmech', len(failed_models_mech))
    print('\tadj', len(failed_models_adj))
    print('\tchem', len(failed_models_chem))

    print("Passed", n_passed, "of", n_total, "tests ")

    if len(sys.argv) == 1 or sys.argv[1] != 'LOG':
        input("Press Enter to continue...") # pause the screen
    else:
        print('exit:', n_failed)
        # exit with code equal to number of failed models
    exit(n_failed)


def check_performance(mod, formulation=None):
    _normalize_odls_env()
    pkl_suffix = _pkl_suffix()
    # A parametrized run (e.g. a formulation) gets its own reference pkl and log so
    # several variants of one model do not overwrite each other.
    tag = '_' + str(formulation) if formulation is not None else ''
    x = os.path.basename(os.getcwd())
    print("Running {:<30}".format(x + tag.replace('_', ' ') + ': '), flush=True)
    # erase previous log file if existed
    models_dir = os.path.dirname(os.path.abspath(__file__))  # /models
    rel_dir = os.path.relpath(os.getcwd(), models_dir)  # e.g., dfm_well/coupled_dfm_well_reservoir
    safe_name = rel_dir.replace(os.sep, '__') + tag
    log_file = os.path.join(models_dir, '_logs', safe_name + '.log')
    _ensure_parent_dir(log_file)
    f = open(log_file, "w")
    f.close()
    log_stream = redirect_all_output(log_file)
    shutil.rmtree("__pycache__", ignore_errors=True)
    # create model instance
    m = mod.Model() if formulation is None else mod.Model(formulation=formulation)
    #m.params.linear_type = sim_params.cpu_superlu

    platform='cpu'
    if os.getenv('TEST_GPU') != None and os.getenv('TEST_GPU') == '1':
        platform='gpu'

        if os.getenv('GPU_DEVICE') != None:
            from darts.engines import set_gpu_device
            set_gpu_device(int(os.getenv('GPU_DEVICE')))

    m.init(platform=platform)

    m.set_output()
    model_path = os.getcwd()
    if formulation is not None:
        # Parametrized runs do not go through the main.py suite path, so produce and
        # check the well time-series here, against a variant-tagged reference
        # (e.g. well_time_data_lin_iter_PT.pkl).
        well_snapshot = create_well_time_series_snapshot(model_path)
    m.run(save_well_data=False, save_reservoir_data=False)
    m.print_stat()
    if formulation is not None:
        m.output.store_well_time_data(save_output_files=True)
    abort_redirection(log_stream)
    overwrite = 0
    if os.getenv('UPLOAD_PKL') != None and os.getenv('UPLOAD_PKL') == '1':
        overwrite = 1
    failed = m.check_performance(overwrite=overwrite, pkl_suffix=pkl_suffix + tag)
    if formulation is not None:
        failed_well_time_series, _, _ = compare_generated_well_time_series(
            model_path,
            well_snapshot,
            overwrite=overwrite,
            pkl_suffix=pkl_suffix + tag,
        )
        failed += failed_well_time_series

    return failed


def check_performance_adjoint(mod):
    x = os.path.basename(os.getcwd())
    print("Running {:<30}".format(x + ': '), flush=True)
    # erase previous log file if existed
    models_dir = os.path.dirname(os.path.abspath(__file__))  # /models
    rel_dir = os.path.relpath(os.getcwd(), models_dir)
    safe_name = rel_dir.replace(os.sep, '__')
    log_file = os.path.join(models_dir, '_logs', safe_name + '.log')
    _ensure_parent_dir(log_file)
    f = open(log_file, "w")
    f.close()
    log_stream = redirect_all_output(log_file)
    mod.prepare_synthetic_observation_data()
    mod.read_observation_data()
    failed = mod.process_adjoint()
    abort_redirection(log_stream)


    return failed

if __name__ == '__main__':

    # print build info
    engines_pbi()

    # multithreaded run can be enabled by setting OMP_NUM_THREADS environment variable
    if os.getenv('OMP_NUM_THREADS') == None:
        os.environ['OMP_NUM_THREADS'] = '1'
    print('OMP_NUM_THREADS=', os.environ['OMP_NUM_THREADS'])

    # cpu/gpu
    platform = 'cpu'
    if os.getenv('TEST_GPU') != None and os.getenv('TEST_GPU') == '1':
        platform = 'gpu'
    print('platform=', platform)

    # overwrite existing pkl files
    overwrite = '0'
    if os.getenv('UPLOAD_PKL') != None and os.getenv('UPLOAD_PKL') == '1':
        overwrite = '1'

    # run larger set of models (takes longer)
    test_all_models = False
    if os.getenv('TEST_ALL_MODELS') != None and os.getenv('TEST_ALL_MODELS') == '1':
        test_all_models = True

    iter_solvers = _normalize_odls_env()

    rcode = run_testing(platform, overwrite, iter_solvers, test_all_models)
    exit(rcode)
