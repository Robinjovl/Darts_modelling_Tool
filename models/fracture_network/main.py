from multiprocessing import freeze_support
from datetime import datetime
from main_gen_mesh import generate_mesh
from main_simulation import run_simulation
from set_case import set_input_data
import os, sys
from darts.models.cicd_model import compare_solution_with_reference

def run_test(args: list = [], platform='cpu'):
    if len(args) > 1:
        return test(case=args[0], overwrite=args[1], platform=platform)
    else:
        print('Not enough arguments provided')
        return 1, 0.0

def test(case, overwrite='0', platform='cpu'):
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
    failed, sim_time = compare_solution_with_reference(m=m, pkl_custom_suffix = '_' + case)

    return failed, total_timer


if __name__ == "__main__":
    platform = 'cpu'
    if len(sys.argv) > 1:
        platform = sys.argv[1]
    if platform not in ['cpu', 'gpu']:
        print('unknown platform specified', platform)
        exit(1)

    cases_list = ['case_1']

    cases_list += ['case_1_burden_O1']
    cases_list += ['case_1_burden_O2']
    cases_list += ['case_1_burden_U1']
    cases_list += ['case_1_burden_U2']
    cases_list += ['case_1_burden_O1_U1']
    cases_list += ['case_1_burden_O2_U2']

    ##cases_list = ['case_2']
    cases_list = ['case_3']
    cases_list = ['case_4']
    cases_list = ['case_5']

    cases_list = ['whitby']

    for case in cases_list:
        test(case, platform=platform)