from darts.engines import redirect_darts_output

from model_geothermal import ModelGeothermal
from model_deadoil import ModelDeadOil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os

def run(physics_type : str, case: str, out_dir: str, dt : float, n_time_steps : int, export_vtk=False):
    '''
    :param physics_type: "geothermal" or "dead_oil"
    :param case: input grid name
    :param out_dir: directory name for outpult files
    :param dt: timestep length, [days]
    :param n_time_steps: number of timestep
    :param export_vtk:
    :return:
    '''
    print('Test started', 'physics_type:', physics_type, 'case:', case)
    redirect_darts_output(os.path.join(out_dir, 'run.log'))

    if physics_type == 'geothermal':
        m = ModelGeothermal(case=case, grid_out_dir=out_dir)
    elif physics_type == 'dead_oil':
        m = ModelDeadOil(case=case, grid_out_dir=out_dir)
    else:
        print('Error: wrong physics specified:', physics_type)
        exit(1)

    m.init(output_folder=out_dir)
    m.save_data_to_h5(kind = 'solution')
    m.set_well_controls()
    if export_vtk:
        m.output_to_vtk(ith_step=0, output_directory=out_dir)
        m.create_vtk_wells(output_directory=out_dir)
    arrays_save = m.get_arrays()
    m.save_grdecl(arrays_save, os.path.join(out_dir, 'res_init'))

    t = 0
    for ti in range(n_time_steps):
        m.run(dt)
        t += dt
        if export_vtk:
            m.output_to_vtk(ith_step=ti+1, output_directory=out_dir)
        # save to grdecl file after each time step
        #arrays_save = m.get_arrays()
        #m.save_grdecl(arrays_save, os.path.join(out_dir, 'res_' + str(ti+1)))
        m.physics.engine.report()
        m.print_well_rate()

    arrays_save = m.get_arrays()
    m.save_grdecl(arrays_save, os.path.join(out_dir, 'res_last'))
    m.print_timers()
    m.print_stat()

    time_data = pd.DataFrame.from_dict(m.physics.engine.time_data)
    time_data['Time (years)'] = time_data['time'] / 365.25
    time_data.to_pickle(os.path.join(out_dir, 'time_data.pkl'))

    time_data_report = pd.DataFrame.from_dict(m.physics.engine.time_data_report)

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

    return time_data_report

##########################################################################################################

def plot_results(results, out_dir, physics_type):
    if physics_type == 'geothermal':
        well_name = 'PRD'
        col = well_name + ' : temperature'
        plt.figure()
        for k in results.keys():
            y = np.array(results[k].filter(like=col)) - 273.15  # to degrees
            t = np.array(results[k]['Time (years)'])
            plt.plot(t, y, label=k)
        plt.ylabel('Temperature prod well, C')
        plt.xlabel('years')
        plt.legend()
        plt.savefig(os.path.join(out_dir, 'well_temperature_' + case + '.png'))
        plt.close()

##########################################################################################################

if __name__ == '__main__':
    physics_list = ['geothermal', 'dead_oil']

    #physics_list = ['geothermal']
    physics_list = ['dead_oil']

    cases_list = ['case_40x40x10']  # small test case
    #cases_list = ['brugge']

    dt = 365.25  # one report timestep length, [days]
    n_time_steps = 20

    for physics_type in physics_list:
        for case in cases_list:
            out_dir = 'results_' + physics_type + '_' + case

            time_data_report = run(physics_type=physics_type, case=case, out_dir=out_dir,
                                   dt=dt, n_time_steps=n_time_steps)

            # one can read well results from pkl file to add/change well plots without re-running the model
            # time_data_report = pd.read_pickle(os.path.join(out_dir, 'time_data_report.pkl'))

            plot_results(time_data_report, physics_type, out_dir)

