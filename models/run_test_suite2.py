from darts.engines import print_build_info as engines_pbi
from for_each_model import run_tests, abort_redirection, redirect_all_output
import sys, os, shutil
import subprocess
from darts.engines import sim_params

def run_testing(platform, overwrite, iter_solvers, test_all_models):
    model_dir = r'.'

    # define a model list to run
    accepted_dirs = ['2ph_comp', '2ph_comp_solid', '2ph_do', '2ph_do_thermal',
                     '2ph_geothermal', '2ph_geothermal_mass_flux',
                     '3ph_comp_w', '3ph_do', '3ph_bo',
                     'Uniform_Brugge',
                     'Chem_benchmark_new',
                     'CCS',
                     #'CO2_foam_CCS',
                     'SPE11b',
                     'GeoRising',
                     'CoaxWell'
                     ]
    accepted_dirs_adjoint = ['Adjoint_super_engine']
    if platform == 'cpu':  # MPFA code is excluded from gpu build due to compilation issues (c++ std 20)
        accepted_dirs += ['2ph_do_thermal_mpfa']
        accepted_dirs_adjoint += ['Adjoint_mpfa']

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

    # CPG (C++ discr)
    test_dirs_cpg = ['cpg_sloping_fault']
    cpg_cases_list = ['generate_5x3x4']
    if iter_solvers:  # run this case only for the build with iterative solvers
        cpg_cases_list += ['generate_51x51x1', 'case_40x40x10']
    test_args_cpg = []
    for case_geom in cpg_cases_list:
        for physics_type in ['geothermal', 'deadoil']:
            for wctrl in ['wrate', 'wbhp']:
                if physics_type == 'deadoil' and wctrl == 'wrate':
                    continue  #TODO fix convergence
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

    # RUN main.py files in the folder listed in the accepted_dirs
    n_failed = n_total = 0
    n_failed_mainpy = n_total_mainpy = 0
    for mdir in accepted_dirs:
        print('running main.py for model', mdir)
        n_total_mainpy += 1
        os.chdir(mdir)
        mrun = subprocess.run(["python", "main.py", platform], stdout=open('../_logs/' + mdir + '.log', 'w'), stderr=open('../_logs/' + mdir + '_err.log', 'w'))
        rcode = mrun.returncode
        n_failed_mainpy += rcode
        if not rcode:
            print('OK')
        else:
            print('FAIL')
            # duplicate 10 last lines to screen from the error log file
            with open('../_logs/' + mdir + '_err.log', 'r') as f:
                s = f.readlines()
                s = s if len(s) <= 10 else s[-10:]
                print('\t' + '\t'.join(s))
        os.chdir('..')
    n_failed += n_failed_mainpy
    n_total += n_total_mainpy

    # discretizer tests
    n_total_discr = n_failed_discr = 0
    n_total_discr, n_failed_discr = run_tests(model_dir, test_dirs=test_dirs_cpg, test_args=test_args_cpg, overwrite=overwrite, platform=platform)
    n_failed += n_failed_discr
    n_total += n_total_discr

    # fracture network tests
    n_total_dfn = n_failed_dfn = 0
    n_total_dfn, n_failed_dfn = run_tests(model_dir, test_dirs=test_dirs_dfn, test_args=test_args_dfn, overwrite=overwrite, platform=platform)
    n_failed += n_failed_dfn
    n_total += n_total_dfn

    # poromechanic tests
    n_total_mech = n_failed_mech = 0
    if platform == 'cpu':  # mech code is excluded from gpu build due to compilation issues (c++ std 20)
        n_total_mech, n_failed_mech = run_tests(model_dir, test_dirs_mech, test_args_mech, overwrite)
    n_failed += n_failed_mech
    n_total += n_total_mech

    n_passed = n_total - n_failed
    print("Passed", n_passed, "of", n_total, "tests ")
    print('n_failed     =', n_failed_mainpy)
    print('n_failed_cpg =', n_failed_discr)
    print('n_failed_dfn =', n_failed_dfn)
    print('n_failed_mech=', n_failed_mech)

    # exit with code equal to the number of failed models
    exit(n_failed)


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

    iter_solvers = False
    if os.getenv('ODLS') != None and os.getenv('ODLS') == '-a':  # run this case only for the build with iterative solvers
        iter_solvers = True
        
    run_testing(platform, overwrite, iter_solvers, test_all_models)
