from darts.engines import redirect_darts_output
from darts.tools.plot_darts import *
from model_geothermal import ModelGeothermal
from model_deadoil import ModelDeadOil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os

def run(physics_type : str, case: str, out_dir: str, dt : float, n_time_steps : int, export_vtk=True):
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
    arrays_save = m.get_arrays()
    m.save_grdecl(arrays_save, os.path.join(out_dir, 'res_init'))

    t = 0
    for ti in range(n_time_steps):
        m.run(dt)
        t += dt
        # save to grdecl file after each time step
        #arrays_save = m.get_arrays()
        #m.save_grdecl(arrays_save, os.path.join(out_dir, 'res_' + str(ti+1)))
        m.physics.engine.report()
        m.print_well_rate()

    arrays_save = m.get_arrays()
    m.save_grdecl(arrays_save, os.path.join(out_dir, 'res_last'))
    m.print_timers()
    m.print_stat()

    if export_vtk:
        # read h5 file and write vtk
        m.create_vtk_wells(output_directory=out_dir)
        for ith_step in range(n_time_steps):
            m.output_to_vtk(ith_step=ith_step)
    def add_columns_time_data(time_data):
        time_data['Time (years)'] = time_data['time'] / 365.25
        for k in time_data.keys():
            if 'temperature' in k:
                time_data[k.replace('K', 'degrees')] = time_data[k] - 273.15
                time_data.drop(columns=k, inplace=True)
            if physics_type == 'dead_oil' and 'm3/day' in k:
                time_data[k.replace('m3/day', 'kmol/day')] = time_data[k]
                time_data.drop(columns=k, inplace=True)

    time_data = pd.DataFrame.from_dict(m.physics.engine.time_data)
    add_columns_time_data(time_data)
    time_data.to_pickle(os.path.join(out_dir, 'time_data.pkl'))

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

    return time_data_report

##########################################################################################################
def plot_results(results, physics_type, out_dir):
    well_name = 'PRD'
    plt.rc('font', size=12)

    if physics_type == 'geothermal':
        ax1 = plot_temp_darts(well_name, results)
        ax1.set(xlabel="Days", ylabel="temperature [degrees]")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'well_temperature_' + case + '.png'))
        plt.close()

        ax1 = plot_extracted_energy_darts(results)
        ax1.set(xlabel="Days", ylabel="energy [MJ]")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'energy_extracted_' + case + '.png'))
        plt.close()
    else:
        # rate plotting
        ax1 = plot_total_prod_oil_rate_darts(results)
        ax1.set(xlabel="Days", ylabel="Total produced oil rate, kmol/day")
        plt.savefig(os.path.join(out_dir, 'production_oil_rate_' + case + '.png'), )
        plt.close()

        if False:
            wcut = f'{well_name}' + ' watercut'
            results[wcut] = results[well_name + ' : water rate (m3/day)'] / (results[well_name + ' : water rate (m3/day)'] + results[well_name + ' : oil rate (m3/day)'])
            ax3 = results.plot(x='time', y=wcut, label=wcut)
            ax3.set_ylim(0, 1)
            ax3.set(xlabel="Days", ylabel="Water cut [-]")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'water_cut_' + case + '.png'))
            plt.close()

    # common plots for both physics
    ax = plot_total_inj_water_rate_darts(results)
    ax.set(xlabel="Days", ylabel="Total injected water rate, kmol/day")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'injection_water_rate_' + case + '.png'))
    plt.close()

    ax = plot_total_prod_water_rate_darts(results)
    ax.set(xlabel="Days", ylabel="Total produced water rate, kmol/day")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'production_water_rate_' + case + '.png'))
    plt.close()

    for well_name in ['PRD', 'INJ']:
        ax = plot_bhp_darts(well_name, results)
        ax.set(xlabel="Days", ylabel="BHP [bar]")
        plt.savefig(os.path.join(out_dir, 'well_' + well_name + '_bhp_' + case + '.png'))
        plt.tight_layout()
        plt.close()

##########################################################################################################

if __name__ == '__main__':
    physics_list = ['geothermal', 'dead_oil']

    #physics_list = ['geothermal']
    #physics_list = ['dead_oil']

    cases_list = ['case_40x40x10', 'brugge']  # small test case

    #cases_list = ['case_40x40x10']  # small test case
    #cases_list = ['brugge']

    dt = 365.25  # one report timestep length, [days]
    n_time_steps = 20

    for physics_type in physics_list:
        for case in cases_list:
            out_dir = 'results_' + physics_type + '_' + case

            time_data_report = run(physics_type=physics_type, case=case, out_dir=out_dir, dt=dt, n_time_steps=n_time_steps)

            # one can read well results from pkl file to add/change well plots without re-running the model
            #time_data_report = pd.read_pickle(os.path.join(out_dir, 'time_data.pkl'))

            plot_results(time_data_report, physics_type, out_dir)

