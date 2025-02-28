from multiprocessing import freeze_support
from datetime import datetime
from main_gen_mesh import generate_mesh
from main_simulation import run_simulation
from set_case import set_input_data
import os, sys
from darts.models.cicd_model import compare_solution_with_reference, get_platform, is_iter_solvers, is_test_all_models

def run_case(case, overwrite='0', platform='cpu'):
    freeze_support()

    input_data = set_input_data(case)

    t1 = datetime.now()
    generate_mesh(input_data)
    t2 = datetime.now()
    mesh_gen_timer = (t2 - t1).total_seconds()

    t1 = datetime.now()
    m = run_simulation(input_data, platform=platform)
    t2 = datetime.now()
    sim_timer = (t2 - t1).total_seconds()

    total_timer = mesh_gen_timer + sim_timer
    print('Mesh generation time:', mesh_gen_timer, 'sec.')
    print('Simulation time:     ', sim_timer, 'sec.')
    print('Total time:          ', total_timer, 'sec.')

    # for CI/CD
    failed, sim_time = False, -1.
    if 'case_1' in case:
        failed, sim_time = compare_solution_with_reference(m=m, pkl_custom_suffix = '_' + case)

    return failed#, total_timer


if __name__ == "__main__":
    platform = get_platform()

    cases_list = ['case_1']

    #cases_list += ['case_1_burden_O1']
    #cases_list += ['case_1_burden_O2']
    #cases_list += ['case_1_burden_U1']
    #cases_list += ['case_1_burden_U2']
    #cases_list += ['case_1_burden_O1_U1']
    #cases_list += ['case_1_burden_O2_U2']

    ##cases_list = ['case_2']
    #cases_list = ['case_3']

    if is_test_all_models():
        cases_list = ['case_4']
        cases_list = ['case_5']

    #cases_list = ['whitby']

    n_failed = 0
    for case in cases_list:
        failed = run_case(case, platform=platform)
        n_failed += failed
        if failed:
            print('FAIL')
        else:
            print('OK')

    exit(n_failed)