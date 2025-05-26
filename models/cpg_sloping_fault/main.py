import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, sys

from darts.engines import redirect_darts_output
from darts.tools.plot_darts import *
from darts.tools.logging import redirect_all_output, abort_redirection

from model_geothermal import ModelGeothermal
from model_deadoil import ModelDeadOil


def run(physics_type : str, case: str, out_dir: str, export_vtk=True, redirect_log=False, platform='cpu', compare_with_ref=True):
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

    if physics_type == 'geothermal':
        m = ModelGeothermal(iapws_physics=True)
    elif physics_type == 'deadoil':
        m = ModelDeadOil()
    else:
        print('Error: wrong physics specified:', physics_type)
        exit(1)
    m.physics_type = physics_type

    m.set_input_data(case=case)

    m.set_physics()

    arrays = m.init_input_arrays()
    # custom arrays can be read here
    # arrays['new_array_name'] = read_float_array(filename, 'new_array_name')
    # arrays['new_array_name'] = read_int_array(filename, 'new_array_name')
    m.init_reservoir(arrays=arrays)

    # time stepping and convergence parameters
    m.set_sim_params_data_ts(data_ts=m.idata.sim.DataTS)

    m.timer.node["initialization"].stop()

    m.init(platform=platform)
    #m.reservoir.mesh.init_grav_coef(0)
    m.set_output(output_folder=out_dir, all_phase_props=True)
    # m.output.save_data_to_h5(kind='reservoir')
    m.set_well_controls_idata()

    m.reservoir.save_grdecl(m.get_arrays(), os.path.join(out_dir, 'res_init'))

    ret = m.run_simulation()
    if ret != 0:
        exit(1)

    m.reservoir.save_grdecl(m.get_arrays(), os.path.join(out_dir, 'res_last'))
    m.print_timers()

    if export_vtk:
        # read h5 file and write vtk
        m.reservoir.create_vtk_wells(output_directory=out_dir)
        for ith_step in range(len(m.idata.sim.time_steps)+1):
            m.output.output_to_vtk(ith_step=ith_step, output_properties=m.physics.vars + m.output.properties)

        m.reservoir.centers_to_vtk(os.path.join(out_dir, 'vtk_files'))

    def add_columns_time_data(time_data):
        time_data['Time (years)'] = time_data['time'] / 365.25 # extra column with time in years
        for k in time_data.keys():
            # extra column with temperature in celsius
            if 'BHT' in k:
                time_data[k.replace('K', 'degrees')] = time_data[k] - 273.15
                time_data.drop(columns=k, inplace=True)

    # COMPUTE TIME DATA
    td = m.output.store_well_time_data()
    time_data = pd.DataFrame.from_dict(td)
    add_columns_time_data(time_data)
    time_data.to_pickle(os.path.join(out_dir, 'time_data.pkl'))

    # COMPUTE TIME DATA AT FIXED REPORTING STEPS
    time_data_report = pd.DataFrame.from_dict(m.physics.engine.time_data_report)
    add_columns_time_data(time_data_report)
    time_data_report.to_pickle(os.path.join(out_dir, 'time_data_report.pkl'))
    writer = pd.ExcelWriter(os.path.join(out_dir, 'time_data.xlsx'))
    time_data.to_excel(writer, sheet_name='time_data')
    writer.close()

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

    if compare_with_ref:
        failed, sim_time = check_performance_local(m=m, case=case, physics_type=physics_type)
    else:
        failed, sim_time = 0, 0.0

    if redirect_log:
        abort_redirection(log_stream)
    print('Failed' if failed else 'Ok')

    return failed, sim_time, time_data, time_data_report, m.idata.well_data.wells.keys(), m.well_is_inj

##########################################################################################################
def plot_results(wells, well_is_inj, time_data_list, time_data_report_list, label_list, physics_type, out_dir):
    plt.rc('font', size=12)

    for well_name in wells:
        if well_is_inj(well_name):
            continue
        if physics_type == 'geothermal':
            ax = None
            for time_data_report, label in zip(time_data_report_list, label_list):
                ax = plot_temp_darts(well_name, time_data_report, ax=ax)#, label=label)
            ax.set(xlabel="Days", ylabel="temperature [degrees]")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'well_temperature_' + well_name + '_' + case + '.png'))
            plt.close()

            # use time_data here as we are going to compute a cumulative plot
            ax = None
            for time_data, label in zip(time_data_list, label_list):
                ax = plot_extracted_energy_darts(time_data, ax=ax)#, label=label)
            ax.set(xlabel="Days", ylabel="energy [PJ]")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'energy_extracted_' + well_name + '_' + case + '.png'))
            plt.close()
        else:
            # rate plotting
            ax = None
            for time_data_report, label in zip(time_data_report_list, label_list):
                ax = plot_total_prod_oil_rate_darts(time_data_report, ax=ax)#, label=label)
            ax.set(xlabel="Days", ylabel="Total produced oil rate, kmol/day")
            plt.savefig(os.path.join(out_dir, 'production_oil_rate_' + well_name + '_' + case + '.png'), )
            plt.close()

            if False:
                #TODO need to get proper volumetric rates to compute the watercut
                wcut = f'{well_name}' + ' watercut'
                results[wcut] = results[well_name + ' : water rate (m3/day)'] / (results[well_name + ' : water rate (m3/day)'] + results[well_name + ' : oil rate (m3/day)'])
                # results[wcut] = results['well_' + well_name + '_volumetric_rate_water_at_wh']/(results['well_' + well_name + '_volumetric_rate_water_at_wh'] + results['well_' + well_name + '_volumetric_rate_oil_at_wh'] )

                ax3 = results.plot(x='time', y=wcut, label=wcut)
                ax3.set_ylim(0, 1)
                ax3.set(xlabel="Days", ylabel="Water cut [-]")
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, 'water_cut_' + well_name + '_' + case + '.png'))
                plt.close()

    rate_units = 'm3/day' if physics_type == 'geothermal' else 'kmol/day'

    # common plots for both physics
    ax = None
    for time_data_report, label in zip(time_data_report_list, label_list):
        ax = plot_total_inj_water_rate_darts(time_data_report, ax=ax)#, label=label)
    ax.set(xlabel="Days", ylabel="Total injected water rate, " + rate_units)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'injection_water_rate_' + case + '.png'))
    plt.close()

    ax = None
    for time_data_report, label in zip(time_data_report_list, label_list):
        ax = plot_total_prod_water_rate_darts(time_data_report, ax=ax)#, label=label)
    ax.set(xlabel="Days", ylabel="Total produced water rate, " + rate_units)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'production_water_rate_' + case + '.png'))
    plt.close()

    for well_name in wells:
        ax = None
        for time_data_report, label in zip(time_data_report_list, label_list):
            ax = plot_bhp_darts(well_name, time_data_report, ax=ax)#, label=label)
        ax.set(xlabel="Days", ylabel="BHP [bar]")
        plt.savefig(os.path.join(out_dir, 'well_' + well_name + '_bhp_' + case + '.png'))
        plt.tight_layout()
        plt.close()

##########################################################################################################
# for CI/CD
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

def run_test(args: list = [], platform='cpu'):
    if len(args) > 1:
        case = args[0]
        physics_type = args[1]

        out_dir = 'results_' + physics_type + '_' + case
        ret = run(case=case, physics_type=physics_type, out_dir=out_dir, platform=platform, compare_with_ref=True)
        return ret[0], ret[1] #failed_flag, sim_time
    else:
        print('Not enough arguments provided')
        return True, 0.0
##########################################################################################################

if __name__ == '__main__':
    platform = 'cpu'
    if os.getenv('TEST_GPU') != None and os.getenv('TEST_GPU') == '1':
            platform = 'gpu'

    physics_list = []
    physics_list += ['geothermal']
    physics_list += ['deadoil']

    cases_list = []
    cases_list += ['generate_5x3x4']
    #cases_list += ['generate_51x51x1']
    #cases_list += ['generate_51x51x1_faultmult']
    #cases_list += ['generate_100x100x100']
    #cases_list += ['case_40x40x10']

    well_controls = []
    well_controls += ['wrate']
    well_controls += ['wbhp']
    well_controls += ['wperiodic']

    for physics_type in physics_list:
        for case_geom in cases_list:
            for wctrl in well_controls:
                if physics_type == 'deadoil' and wctrl == 'wrate':
                    continue
                case = case_geom + '_' + wctrl
                out_dir = 'results_' + physics_type + '_' + case
                failed, sim_time, time_data, time_data_report, wells, well_is_inj = run(physics_type=physics_type,
                                                                                        case=case, out_dir=out_dir,
                                                                                        redirect_log=False,
                                                                                        platform=platform)

                # one can read well results from pkl file to add/change well plots without re-running the model
                pkl1_dir = '.'
                pkl_fname = 'time_data.pkl'
                pkl_report_fname = 'time_data_report.pkl'
                time_data_list = [time_data]
                time_data_report_list = [time_data_report]
                label_list = [None]

                # compare the current results with another run
                #pkl1_dir = r'../../../open-darts_dev/models/cpg_sloping_fault/results_' + physics_type + '_' + case_geom
                #time_data_1 = pd.read_pickle(os.path.join(pkl1_dir, pkl_fname))
                #time_data_report_1 = pd.read_pickle(os.path.join(pkl1_dir, pkl_report_fname))
                #time_data_list = [time_data_1, time_data]
                #time_data_report_list = [time_data_report_1, time_data_report]
                #label_list = ['1', 'current']

                plot_results(wells=wells, well_is_inj=well_is_inj,
                             time_data_list=time_data_list, time_data_report_list=time_data_report_list, label_list=label_list,
                             physics_type=physics_type, out_dir=out_dir)
