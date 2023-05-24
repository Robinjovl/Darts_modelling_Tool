import darts.engines as darts_engines
from darts.engines import print_build_info as engines_pbi
from darts.print_build_info import print_build_info as package_pbi
from for_each_model import for_each_model, run_tests, abort_redirection, redirect_all_output, for_each_model_adjoint
import sys, os, shutil
from darts.engines import sim_params

model_dir = r'.'

accepted_dirs = ['2ph_comp', '2ph_comp_solid', '2ph_do', '2ph_do_thermal', '2ph_geothermal',
                 '3ph_comp_w', '3ph_do', '3ph_bo',
                 'Uniform_Brugge',
                 'Chem_benchmark_new',
                 #'CO2_foam_CCS',
                 'GeoRising',
                 'CoaxWell'
                 ]

test_dirs = ['1ph_1comp_poroelastic_analytics']
test_args = [
                [['terzaghi', 'non_stabilized', 'rect'],
                ['terzaghi', 'non_stabilized', 'wedge'],
                ['terzaghi', 'non_stabilized', 'hex'],
                ['terzaghi', 'stabilized', 'rect'],
                ['terzaghi', 'stabilized', 'wedge'],
                ['terzaghi', 'stabilized', 'hex'],
                ['mandel', 'non_stabilized', 'rect'],
                ['mandel', 'non_stabilized', 'wedge'],
                ['mandel', 'non_stabilized', 'hex'],
                ['mandel', 'stabilized', 'rect'],
                ['mandel', 'stabilized', 'wedge'],
                ['mandel', 'stabilized', 'hex'],
                ['terzaghi_two_layers', 'non_stabilized', 'rect'],
                ['terzaghi_two_layers', 'non_stabilized', 'wedge']]
            ]

accepted_dirs_adjoint = ['Adjoint_super_engine']  # for adjoint test


def check_performance(mod):
    pkl_suffix = ''
    if os.getenv('ODLS') != None and os.getenv('ODLS') == '0':
        pkl_suffix = '_iter'
    x = os.path.basename(os.getcwd())
    print("Running {:<30}".format(x + ': '), flush=True)
    # erase previous log file if existed
    log_file = os.path.join(os.path.abspath(os.pardir), '_logs/' + str(x) + '.log')
    f = open(log_file, "w")
    f.close()
    log_stream = redirect_all_output(log_file)
    shutil.rmtree("__pycache__", ignore_errors=True)
    # create model instance
    m = mod.Model()
    #m.params.linear_type = sim_params.cpu_superlu
    m.init()
    m.run()
    m.print_stat()
    abort_redirection(log_stream)
    overwrite = 0
    if os.getenv('UPLOAD_PKL') == '1':
        overwrite = 1
    failed = m.check_performance(overwrite=overwrite, pkl_suffix=pkl_suffix)
    log_stream = redirect_all_output(log_file)
    return failed


def check_performance_adjoint(mod):
    x = os.path.basename(os.getcwd())
    print("Running {:<30}".format(x + ': '), flush=True)
    # erase previous log file if existed
    log_file = os.path.join(os.path.abspath(os.pardir), '_logs/' + str(x) + '.log')
    f = open(log_file, "w")
    f.close()
    log_stream = redirect_all_output(log_file)
    mod.prepare_synthetic_observation_data()
    mod.read_observation_data()
    failed = mod.process_adjoint()
    abort_redirection(log_stream)
    log_stream = redirect_all_output(log_file)

    return failed

if __name__ == '__main__':

    # print build info
    engines_pbi()
    package_pbi()

    # set single thread in case of MT version to match the performance characteristics
    os.environ['OMP_NUM_THREADS'] = '1'

    overwrite = '0'
    if os.getenv('UPLOAD_PKL') == '1':
        overwrite = '1'

    failed = for_each_model(model_dir, check_performance, accepted_dirs)

    # poromechanic tests
    n_tot, n_failed = run_tests(model_dir, test_dirs, test_args, overwrite)
    failed += n_failed


    # test for adjoint ------------------start---------------------------------
    import time
    starting_time = time.time()
    failed_ad = for_each_model_adjoint(model_dir, check_performance_adjoint, accepted_dirs_adjoint)
    ending_time = time.time()
    if not failed_ad:
        print('OK, \t%.2f s' % (ending_time - starting_time))
    else:
        print('FAIL, \t%.2f s' % (ending_time - starting_time))

    failed += failed_ad
    # test for adjoint ------------------end---------------------------------

    n_passed = len(accepted_dirs) + n_tot + len(accepted_dirs_adjoint) - failed
    n_total = len(accepted_dirs) + n_tot + len(accepted_dirs_adjoint)

    print("Passed", n_passed, "of", n_total, "models. ")

    if len(sys.argv) == 1:
        input("Press Enter to continue...") # pause the screen
    else:
        if os.getenv('UPLOAD_PKL') == '1':  # do not interrupt ci/cd for uploading generated pkls
            print('exit 0 because of UPLOAD_PKL==1')
            exit(0)
        print('exit:', failed)
        # exit with code equal to number of failed models
        exit(failed)
