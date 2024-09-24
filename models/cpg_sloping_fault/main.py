from darts.engines import redirect_darts_output
from model_cpg import Model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os

def run(physics_type : str, discr_type : str, case: str, out_dir: str, dt : float, n_time_steps : int, export_vtk=False):
    print('Test started', 'physics_type:', physics_type, 'discr_type:', discr_type, 'case:', case)
    redirect_darts_output(os.path.join(out_dir, 'run.log'))
    m = Model(physics_type=physics_type, discr_type=discr_type, case=case, grid_out_dir=out_dir)

    m.init(output_folder=out_dir)
    m.save_data_to_h5(kind = 'solution')
    m.set_well_controls()
    if export_vtk:
        m.output_to_vtk(ith_step=0, output_directory=out_dir)
    m.save_cubes(os.path.join(out_dir, 'res_init'))

    t = 0
    for ti in range(n_time_steps):
        m.run(dt)
        t += dt
        if export_vtk:
            m.output_to_vtk(ith_step=ti+1, output_directory=out_dir)
        # save to grdecl file
        #m.save_cubes(os.path.join(out_dir, 'res_' + str(ti+1)))
        m.physics.engine.report()
        m.print_well_rate()

    m.save_cubes(os.path.join(out_dir, 'res_last'))
    m.print_timers()
    m.print_stat()

    time_data = pd.DataFrame.from_dict(m.physics.engine.time_data)
    time_data['Time (years)'] = time_data['time'] / 365.25
    time_data.to_pickle(os.path.join(out_dir, 'time_data_' + discr_type + '.pkl'))

    time_data_report = pd.DataFrame.from_dict(m.physics.engine.time_data_report)

    # filter time_data_report and write to xlsx
    # list the column names that should be removed
    press_gridcells = time_data_report.filter(like='reservoir').columns.tolist()
    chem_cols = time_data_report.filter(like='Kmol').columns.tolist()
    # remove columns from data
    time_data_report.drop(columns=press_gridcells + chem_cols, inplace=True)
    # add time in years
    time_data_report['Time (years)'] = time_data_report['time'] / 365.25
    writer = pd.ExcelWriter(os.path.join(out_dir, 'time_data_report_' + discr_type + '_' + physics_type + '.xlsx'))
    time_data_report.to_excel(writer, sheet_name='time_data_report')
    writer.close()

    failed, sim_time = check_performance_local(m, case, discr_type, physics_type)

    return failed, sim_time, time_data_report

#####################################################

def run_test(args: list = []):
    if len(args) > 1:
        return test(case=args[0], physics_type=args[1])
    else:
        print('Not enough arguments provided')
        return 1, 0.0

def test(case: str, physics_type : str):
    dt = 365.25
    n_time_steps = 20

    export_vtk = True

    #discr_types_list = ['cpg']  # cpg reservoir
    #discr_types_list = ['struct']  # struct reservoir
    discr_types_list = ['cpg', 'struct']  # both

    # test grids
    #cases_list = [43] # 10x10x10 sloping fault
    #cases_list = [40] # 10x10x10 no fault
    #cases_list = [40, 43]

    # run all the grids in the folder
    #cases_list = []
    #for c in os.listdir('../../meshes/cpg/'):
    #    if c.startswith("Case_"):
    #        cases_list.append(c)

    print('cases :', case)

    mode = 'run'      # run and plot results
    #mode = 'compare' # read pkl files with results and plot
    print('MODE', mode)

    results = dict()

    if mode == 'run':
        for discr_type in discr_types_list:
            start = time.perf_counter()
            out_dir = 'results_' + discr_type + '_' + physics_type + '_' + case
            failed, sim_time, results[discr_type] = run(physics_type=physics_type, case=case,
                                                        discr_type=discr_type, out_dir=out_dir,
                                                        dt=dt, n_time_steps=n_time_steps, export_vtk=export_vtk)
            end = time.perf_counter()
    elif mode == 'compare':
        for discr_type in discr_types_list:
            out_dir = 'results_' + discr_type + '_' + physics_type + '_' + case
            results[discr_type] = pd.read_pickle(os.path.join(out_dir, 'time_data_' + discr_type + '.pkl'))

    if physics_type == 'geothermal':
        well_name = 'PRD'
        col = well_name + ' : temperature'
        plt.figure()
        for k in results.keys():
            y = np.array(results[k].filter(like=col)) - 273.15
            t = np.array(results[k]['Time (years)'])  # to years
            plt.plot(t, y, label=k)
        plt.ylabel('Temperature prod well, C')
        plt.xlabel('years')
        plt.legend()
        plt.savefig(os.path.join(out_dir, 'well_temperature_' + case + '.png'))
        plt.close()

    return failed, sim_time

def check_performance_local(m, case, discr_type, physics_type):
    import platform

    os.makedirs('ref', exist_ok=True)

    pkl_suffix = ''
    if os.getenv('ODLS') != None and os.getenv('ODLS') == '-a':
        pkl_suffix = '_iter'
    file_name = os.path.join('ref', 'perf_' + platform.system().lower()[:3] + pkl_suffix +
                             '_' + case + '_' + physics_type + '_' + discr_type + '.pkl')
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

if __name__ == '__main__':
    physics_list = ['geothermal', 'dead_oil']
    cases_list = ['generate_5x3x4', 'generate_51x51x1', 'case_40', 'case_43', 'case_40_actnum']
    for physics_type in physics_list:
        for case in cases_list:
            test(case, physics_type)
