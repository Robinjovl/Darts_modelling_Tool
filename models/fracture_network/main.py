from multiprocessing import freeze_support
from datetime import datetime
from main_gen_mesh import generate_mesh
from main_simulation import run_simulation
from set_case import set_input_data
import os, sys

def run_test(args: list = [], platform='cpu'):
    if len(args) > 1:
        return test(case=args[0], overwrite=args[1], platform=platform)
    else:
        print('Not enough arguments provided')
        return 1, 0.0

def test(case, overwrite='0', platform='cpu', compare_with_ref=False):
    freeze_support()

    input_data = set_input_data(case)

    t1 = datetime.now()
    if input_data.geom['mesh_type'] == '2.5D':  # 2.5D mesh generation and cleaning (gmsh based)
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

    if compare_with_ref:
        failed, sim_time = check_performance_local(m, case)
    else:
        failed = False

    return failed, total_timer


def check_performance_local(m, case):
    import platform

    os.makedirs('ref', exist_ok=True)

    pkl_suffix = ''
    if os.getenv('TEST_GPU') != None and os.getenv('TEST_GPU') == '1':
        pkl_suffix = '_gpu'
    elif os.getenv('ODLS') != None and os.getenv('ODLS') == '-a':
        pkl_suffix = '_iter'
    else:
        pkl_suffix = '_odls'
    print('pkl_suffix=', pkl_suffix)

    file_name = os.path.join('ref', 'perf_' + platform.system().lower()[:3] + pkl_suffix + '_' + case + '.pkl')

    overwrite = 0
    if os.getenv('UPLOAD_PKL') == '1':
        overwrite = 1

    is_plk_exist = os.path.isfile(file_name)

    failed = m.check_performance(perf_file=file_name, overwrite=overwrite, pkl_suffix=pkl_suffix)

    if not is_plk_exist or overwrite == '1':
        m.save_performance_data(file_name=file_name, pkl_suffix=pkl_suffix)
        return False, 0.0

    if is_plk_exist:
        return (failed > 0), -1.0 #data[-1]['simulation time']
    else:
        return False, -1.0


if __name__ == "__main__":
    platform = 'cpu'
    if len(sys.argv) > 1:
        platform = sys.argv[1]
    if platform not in ['cpu', 'gpu']:
        print('unknown platform specified', platform)
        exit(1)

    cases_list = []

    cases_list += ['case_1']

    ##cases_list += ['case_1_burden_O1']
    #cases_list += ['case_1_burden_O2']
    #cases_list += ['case_1_burden_U1']
    #cases_list += ['case_1_burden_U2']
    #cases_list += ['case_1_burden_O1_U1']
    #cases_list += ['case_1_burden_O2_U2']

    ##cases_list = ['case_2']
    #cases_list += ['case_3']
    #cases_list += ['case_4']
    #cases_list += ['case_5']

    #cases_list += ['whitby']

    for case in cases_list:
        test(case, platform=platform, compare_with_ref=False)

    #test(case='case_debug')  # 2.5 wedge mesh with one fracture
    #test(case='case_3D_strike0_dip90')  # 3D hexahedral mesh with one fracture
    #test(case='case_3D_strike0_dip0') # 3D tetrahedral mesh with one fracture, fails in vtk output: IndexError: index 12952 is out of bounds for axis 0 with size 506
    #test(case='case_3D_nofrac')  # 3D tetrahedral mesh without fractures
