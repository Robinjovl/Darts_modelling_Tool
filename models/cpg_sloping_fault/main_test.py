import os
import pandas as pd

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

def run(physics_type : str, case: str, out_dir: str, export_vtk=True, redirect_log=False, platform='cpu', compare_with_ref=False):
    case = case
    '''
    :param physics_type: "geothermal" or "dead_oil"
    :param case: input grid name
    :param out_dir: directory name for output files
    :param export_vtk:
    :return:
    '''
    print('Test started', 'physics_type:', physics_type, 'case:', case, 'platform=', platform)

    out_dir = out_dir
    os.makedirs(out_dir, exist_ok=True)
    log_filename = os.path.join(out_dir, 'run.log')
    if redirect_log:
        log_stream = redirect_all_output(log_filename)

    match physics_type:
        case 'geothermal':
            m = ModelGeothermal()
        case 'deadoil':
            m = ModelDeadOil()
        case 'CCS':
            m = ModelCCS()
    m.physics_type = physics_type

    # 1. set physics-specific input data
    match m.physics_type:
        case 'geothermal':
            m.idata = set_input_data_geothermal()
        case 'deadoil':
            m.idata = set_input_data_deadoil()
        case 'CCS':
            m.idata = set_input_data_co2()
    m.timer.node["initialization"].start()
    # 2. set default input data, generic parameters
    # including time stepping and convergence parameters, boundary conditions and rock properties for all cases and default rock properties
    set_input_data_base(m.idata, case)

    # 3. set grid size and resolution, and add wells
    if 'generate' in case:
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
    #
    # if 'CCS' not in m.physics_type:
    #     m.idata.sim.DataTS.dt_first = 1e-5


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

    m.init()
    m.set_output(output_folder=out_dir,
                 all_phase_props = m.idata.compute_all_output_properties, # find this flag in case_base.py
                verbose=True)
    m.set_well_controls_idata()

    m.reservoir.save_grdecl(m.get_arrays(), os.path.join(out_dir, 'res_init'))
    m.timer.node["initialization"].stop()
    ret = m.run_simulation()

    if ret != 0:
        exit(1)

    m.reservoir.save_grdecl(m.get_arrays(), os.path.join(out_dir, 'res_last'))
    m.print_timers()

    # post-processing: read h5 file and write vtk with properties
    if export_vtk:
        print('Post processing properties and vtk output...')

        output_properties_main = m.physics.vars  # only main variables
        output_properties_full = output_properties_main + m.output.properties  # additional properties (might take some time to compute)
        m.reservoir.create_vtk_wells(output_directory=out_dir)
        n_timesteps = len(m.idata.sim.time_steps)
        for ith_step in range(n_timesteps + 1):
            # compute additional properties only for the first and for the last timestep:
            output_properties = output_properties_full if ith_step in [0, n_timesteps] else output_properties_main
            # print('timestep', ith_step, 'output_properties:', output_properties)
            timesteps, property_array = m.output.output_properties(output_properties=output_properties,
                                                                   timestep=ith_step, engine=False)
            if ith_step == 0:
                centers_x, centers_y, centers_z = m.reservoir.get_centers()
                property_array.update({'centers_x': centers_x.reshape(1, -1), 'centers_y': centers_y.reshape(1, -1),
                                       'centers_z': centers_z.reshape(1, -1)})

            if 0:
                # save properties in its own *.h5 file
                os.makedirs(
                    os.path.join(m.output_folder, 'property_arrays'),
                    exist_ok=True
                )
                m.output.save_property_array(timesteps, property_array,
                                             f'property_arrays/property_array_ts{ith_step}.h5')
            else:
                # append properties to reservoir.h5
                m.output.save_property_array(timesteps, property_array)

            m.output.output_to_vtk(output_data=[timesteps, property_array], ith_step=ith_step)

        m.reservoir.centers_to_vtk(os.path.join(out_dir, 'vtk_files'))

    def add_columns_time_data(time_data):
        time_data['Time (years)'] = time_data['time'] / 365.25  # extra column with time in years
        for k in time_data.keys():
            # extra column with temperature in celsius
            if 'BHT' in k:
                time_data[k.replace('K', 'degrees')] = time_data[k] - 273.15
                time_data.drop(columns=k, inplace=True)

    if m.idata.compute_all_output_properties:
        # COMPUTE TIME DATA
        td = m.output.store_well_time_data()
        time_data = pd.DataFrame.from_dict(td)
        # add_columns_time_data(time_data)
        time_data.to_pickle(os.path.join(out_dir, 'time_data.pkl'))
        writer = pd.ExcelWriter(os.path.join(out_dir, 'time_data.xlsx'))
        time_data.to_excel(writer, sheet_name='time_data')
        writer.close()

        # COMPUTE TIME DATA AT FIXED REPORTING STEPS
        time_data_report = pd.DataFrame.from_dict(m.physics.engine.time_data_report)
        add_columns_time_data(time_data_report)
        time_data_report.to_pickle(os.path.join(out_dir, 'time_data_report.pkl'))

        # filter time_data_report and write to xlsx
        # list the column names that should be removed
        press_gridcells = time_data_report.filter(like='reservoir').columns.tolist()
        chem_cols = time_data_report.filter(like='Kmol').columns.tolist()
        # remove columns from data
        time_data_report.drop(columns=press_gridcells + chem_cols, inplace=True)
        # add time in years
        time_data_report['Time (years)'] = time_data_report['time'] / 365.25
        writer = pd.ExcelWriter(os.path.join(out_dir, 'time_data_report.xlsx'))
        time_data_report.to_excel(writer, sheet_name='time_data_report')
        writer.close()

        m.output.store_well_time_data(save_output_files=True)
        m.output.plot_well_time_data()

    m.print_timers()

    if compare_with_ref:
        failed, sim_time = check_performance_local(m=m, case=case, physics_type=physics_type)
    else:
        failed, sim_time = 0, 0.0

    if redirect_log:
        abort_redirection(log_stream)
    print('Failed' if failed else 'Passed')

    return failed, sim_time, time_data, time_data_report, m.idata.well_data.wells.keys(), m.idata.well_is_inj

def run_test(args: list = [], platform='cpu'):
    if len(args) > 1:
        case = args[0]
        physics_type = args[1]

        out_dir = 'results_' + physics_type + '_' + case
        ret = run(case=case, physics_type=physics_type, out_dir=out_dir, platform=platform, compare_with_ref=True)
        return ret[0], ret[1]  # failed_flag, sim_time
    else:
        print('Not enough arguments provided')
        return True, 0.0
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

    if os.getenv('TEST_GPU') != None and os.getenv('TEST_GPU') == '1':
        pkl_suffix = '_gpu'
    elif os.getenv('ODLS') != None and os.getenv('ODLS') == '-a':
        pkl_suffix = '_iter'
    else:
        pkl_suffix = '_odls'
    print('pkl_suffix=', pkl_suffix)

    file_name = os.path.join('ref', 'perf_' + platform.system().lower()[:3] + pkl_suffix +
                             '_' + case.replace('_bhp', '_wbhp').replace('_rate', '_wrate').replace('_periodic', '_wperiodic')   + '_' + physics_type + '.pkl')
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
        last_ret = run(physics_type, case_geom, well_controls,redirect_log = True, compare_with_ref=True)

    return last_ret

# --------------------------------------------
# 4) Entry point (keeps your original sentinel)
# --------------------------------------------
if __name__ == '__main__':
    grid_cases = [
        # generated cases
        # 'generate_5x3x4',
        'generate_51x51x1',
        #'generate_51x51x1_faultmult',
        #'generate_100x100x100',
        # for grdecl cases, the grid and properties files must be in meshes/<case> folder,
        # the last '_' part is ignored so can use it for naming
        #'40x40x10',
        # 'brugge',
        # 'brugge_noburdenlayers',
    ]

    PHYSICS = [
        # 'geothermal',
        'deadoil',
        #'CCS',
    ]

    WELL_CONTROLS = [
        # 'rate',
        'bhp',
        #'periodic'
    ]

    run_all()
