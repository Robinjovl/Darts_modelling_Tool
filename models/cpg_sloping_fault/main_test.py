import os
from model_geothermal import ModelGeothermal
from model_deadoil import ModelDeadOil
from model_CO2 import ModelCCS

from cases.case_base import set_input_data_base
from cases.case_geothermal import set_input_data_geothermal, set_input_data_well_controls_geothermal
from cases.case_deadoil import set_input_data_deadoil, set_input_data_well_controls_deadoil
from cases.case_co2 import set_input_data_co2, set_input_data_well_controls_co2
from cases.case_geom_generate import set_input_data_geom_generate
from cases.case_geom_grdecl import set_input_data_geom_grdecl
from darts.tools.logging import redirect_all_output, abort_redirection
from output_functions import output_vtk, output_time_data
from itertools import product

def run(physics_type, case_geom, well_controls,redirect_log = False, compare_with_ref = False):
    match physics_type:
        case 'geothermal':
            m = ModelGeothermal()
        case 'DeadOil':
            m = ModelDeadOil()
        case 'CCS':
            m = ModelCCS()
    m.physics_type = physics_type
    case = case_geom + '_' + well_controls
    out_dir = os.path.join('results', m.physics_type + '_' + case)

    # 1. set physics-specific input data
    match m.physics_type:
        case 'geothermal':
            m.idata = set_input_data_geothermal()
        case 'deadoil':
            m.idata = set_input_data_deadoil()
        case 'CCS':
            m.idata = set_input_data_co2()

    # 2. set default input data, generic parameters
    # including time stepping and convergence parameters, boundary conditions and rock properties for all cases and default rock properties
    set_input_data_base(m.idata, case_geom)

    # 3. set grid size and resolution, and add wells
    if 'generate' in case_geom:
        set_input_data_geom_generate(m.idata, case)
    else:
        set_input_data_geom_grdecl(m.idata, case)

    # 4. set well controls
    match m.physics_type:
        case 'geothermal':
            set_input_data_well_controls_geothermal(m.idata, case)
        case 'deadoil':
            set_input_data_well_controls_deadoil(m.idata, case)
        case 'CCS':
            set_input_data_well_controls_co2(m.idata, case)

    if 'geothermal' not in m.physics_type:
        m.idata.geom.burden_layers = 0

    if 'CCS' not in m.physics_type:
        m.idata.sim.DataTS.dt_first = 1e-5

    # now, the data is set to m.idata and will be taken there

    os.makedirs(out_dir, exist_ok=True)
    if 1:
        from darts.engines import redirect_darts_output
        log_filename = os.path.join(out_dir, 'run.log')
        log_stream = redirect_darts_output(log_filename)

    print('----- Test started', 'physics_type:', m.physics_type, 'case:', case, ' ------')

    m.set_physics()

    arrays = m.init_input_arrays()
    # custom arrays can be read here
    # arrays['new_array_name'] = read_float_array(filename, 'new_array_name')
    # arrays['new_array_name'] = read_int_array(filename, 'new_array_name')
    # also rock properties such as permeability can be modified here if needed
    # arrays['PERMX'] *= 0.1
    m.init_reservoir(arrays=arrays)

    # time stepping and convergence parameters
    m.set_sim_params_data_ts(data_ts=m.idata.sim.DataTS)

    m.timer.node["initialization"].stop()

    m.init()
    m.set_output(output_folder=out_dir, all_phase_props=True, verbose=True)
    m.set_well_controls_idata()

    m.reservoir.save_grdecl(m.get_arrays(), os.path.join(out_dir, 'res_init'))

    ret = m.run_simulation()

    m.reservoir.save_grdecl(m.get_arrays(), os.path.join(out_dir, 'res_last'))
    m.print_timers()

    print("Writing vtk files...")
    output_vtk(out_dir, m)
    print("Finished vtk files writing")

    output_time_data(out_dir, m, case)

    print("Computation of the case", case, "completed")
    if compare_with_ref:
        failed, sim_time = check_performance_local(m=m, case=case, physics_type=physics_type)
    else:
        failed, sim_time = 0, 0.0

    if redirect_log:
        abort_redirection(log_stream)
    print('Failed' if failed else 'Passed')
    return ret

# Optional filters (useful if some combos are invalid/too heavy)
#    Return True to keep a combo, False to skip it.
def combo_filter(physics_type: str, case_geom: str, well_controls: str) -> bool:
    # Example rules (edit/remove as needed):
    # if case_geom == 'generate_100x100x100' and well_controls == 'periodic':
    #     return False  # skip extremely heavy combos
    return True
def check_performance_local(m, case, physics_type):
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

    file_name = os.path.join('ref', 'perf_' + platform.system().lower()[:3] + pkl_suffix +
                             '_' + case + '_' + physics_type + '.pkl')
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
def run_all():
    last_ret = None
    for physics_type, case_geom, well_controls in product(PHYSICS, grid_cases, WELL_CONTROLS):

        # extra guard: periodic is only valid for geothermal
        if well_controls == 'periodic' and physics_type != 'geothermal':
            print(f"[SKIP] periodic is only for geothermal -> {physics_type} | {case_geom}")
            continue

        if not combo_filter(physics_type, case_geom, well_controls):
            print(f"[SKIP] {physics_type} | {case_geom} | {well_controls}")
            continue

        tag = f"{physics_type}__{case_geom}__{well_controls}"
        print(f"\n[START] {tag}")
        last_ret = run(physics_type, case_geom, well_controls,redirect_log = False, compare_with_ref=True)

    return last_ret

# --------------------------------------------
# 4) Entry point (keeps your original sentinel)
# --------------------------------------------
if __name__ == '__main__':
    grid_cases = [
        # generated cases
        'generate_5x3x4',
        'generate_51x51x1',
        'generate_51x51x1_faultmult',
        'generate_100x100x100',
        # for grdecl cases, the grid and properties files must be in meshes/<case> folder,
        # the last '_' part is ignored so can use it for naming
        '40x40x10',
        # 'brugge',
        # 'brugge_noburdenlayers',
    ]

    PHYSICS = [
        'geothermal',
        'DeadOil',
        'CCS',
    ]

    WELL_CONTROLS = [
        'rate',
        'bhp',
        'periodic'
    ]

    run_all()


