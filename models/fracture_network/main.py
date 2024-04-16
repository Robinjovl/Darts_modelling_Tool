from multiprocessing import freeze_support
from datetime import datetime
from main_gen_mesh import generate_mesh
from main_simulation import run_simulation
from set_case import set_input_data

def run_test(args: list = []):
    if len(args) > 1:
        return test(case=args[0], overwrite=args[1])
    else:
        print('Not enough arguments provided')
        return 1, 0.0

def test(case, overwrite='0'):
    freeze_support()

    input_data = set_input_data(case)

    t1 = datetime.now()
    generate_mesh(input_data)
    t2 = datetime.now()
    mesh_gen_timer = (t2 - t1).total_seconds()

    t1 = datetime.now()
    run_simulation(input_data)
    t2 = datetime.now()
    sim_timer = (t2 - t1).total_seconds()

    print('Mesh generation time:', mesh_gen_timer, 'sec.')
    print('Simulation time:', sim_timer, 'sec.')

    return 0, 0.0

if __name__ == "__main__":
    #cases_list = ['whitby']
    #cases_list = ['case_3']
    #cases_list = ['case_4']
    cases_list = ['case_1']
    cases_list += ['case_1_burden_O1']
    cases_list += ['case_1_burden_O2']
    cases_list += ['case_1_burden_U1']
    cases_list += ['case_1_burden_U2']
    cases_list += ['case_1_burden_O1_U1']
    cases_list += ['case_1_burden_O2_U2']
    for case in cases_list:
        test(case)