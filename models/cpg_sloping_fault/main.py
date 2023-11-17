from darts.engines import value_vector, redirect_darts_output, sim_params, timer_node

import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
darts_dir = os.path.dirname(os.path.dirname(current_dir))  # 2 levels up
model_dir = os.path.join(darts_dir, '2ph_do')
sys.path.insert(0, model_dir)
from model_cpg import Model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

import os

from darts.reservoirs.cpg_reservoir import save_array, make_full_cube
def run(discr_type : str, gridfile : str, propfile : str, sch_fname : str,
        dt : float, n_time_steps : int,
        export_vtk=False):
    model_suffix = gridfile.split('/')[-2]
    model_dir = os.path.dirname(gridfile)
    vtk_filename = 'Result_' + model_suffix + '_' + discr_type
    print('Test started.', 'discr_type:', discr_type)
    redirect_darts_output(os.path.join(model_dir, 'out_' + model_suffix + '_' + discr_type + '.log'))
    m = Model(discr_type=discr_type, gridfile=gridfile, propfile=propfile, sch_fname=sch_fname)

    # debug
    #if discr_type == 'cpp':
    #    tran_fname = os.path.join(model_dir, 'tran_cpg.grdecl')
    #    nnc_fname = os.path.join(model_dir, 'nnc.txt')
    #    m.reservoir.discr.write_tran_cube(tran_fname, nnc_fname)

    # # add wells
    # if True:
    #     m.add_wells(mode='read', sch_fname=sch_fname, verbose=True)#, well_index=1000) # add only perforations
    #     m.set_well_controls() # set well controls - for dap (case_201) only
    # else:
    #     m.add_wells(mode='generate', sch_fname=sch_fname)

    m.init()
    if export_vtk:
        m.export_vtk(vtk_filename)
    m.params.max_ts = dt
    m.save_cubes(os.path.join(model_dir, 'res_init'))

    t = 0
    #print_range(m, t)

    for ti in range(n_time_steps):
        m.run_python(dt)

        #print_range(m, t)
        t += dt

        if export_vtk:
            m.export_vtk(vtk_filename)

        #m.save_cubes(os.path.join(model_dir, 'res_' + str(ti+1)))
        m.engine.report()

    m.save_cubes(os.path.join(model_dir, 'res_last'))
    m.print_timers()
    m.print_stat()

    time_data = pd.DataFrame.from_dict(m.engine.time_data)
    time_data.to_pickle(os.path.join(model_dir, 'time_data_' + discr_type + '.pkl'))
    time_data_report = pd.DataFrame.from_dict(m.engine.time_data_report)

    if False: # write xls
        # filter data and write to xlsx
        # list the column names that should be removed
        for td_ in [time_data, time_data_report]:
            press_gridcells = td_.filter(like='reservoir').columns.tolist()
            chem_cols = td_.filter(like='Kmol').columns.tolist()
            # remove columns from data
            td_.drop(columns=press_gridcells + chem_cols, inplace=True)
            # add time in years
            td_['Time (years)'] = td_['time'] / 365.25

        writer = pd.ExcelWriter(os.path.join(model_dir, 'time_data_' + discr_type + '.xlsx'))
        time_data.to_excel(writer, 'time_data')
        time_data_report.to_excel(writer, 'time_data_report')
        writer.close()

    return time_data

# calculate total rate (sum over the wells) and make combined plot for 2 results
# darts_df_1, darts_df_2 - two results (dataframes)
# search_str - what to plot (oil rate, water rate,..)
# plot_names - list to use as legend, length should be =2
# prod=True -> production, False -> injection
def plot_total_rate_darts(darts_df_1, darts_df_2, opm_df, opm_smry, plot_names, search_str, mode_suffix):
    acc_df = pd.DataFrame()
    acc_df['time'] = darts_df_1['time']
    name_plot_1, name_plot_2 = plot_names[0], plot_names[1]
    acc_df[name_plot_1] = 0
    acc_df[name_plot_2] = 0
    for col in darts_df_1.columns:
        if search_str in col:
            if ('INJ' not in col) or ('INJ' in col):
                acc_df[name_plot_1] += darts_df_1[col]
                acc_df[name_plot_2] += darts_df_2[col]
    acc_df[name_plot_1] = acc_df[name_plot_1].abs()
    acc_df[name_plot_2] = acc_df[name_plot_2].abs()
    ax = None
    ax = acc_df.plot(x='time', y=name_plot_1, style='.-', color='r', ax=ax, alpha=1)
    ax = acc_df.plot(x='time', y=name_plot_2, style='-.', color='b', ax=ax, alpha=1)
    if opm_df is not None:
        ax = opm_df.plot(x='days', y=opm_smry, style='.-.', color='g', ax=ax, alpha=1)
    ymin, ymax = ax.get_ylim()
    if ymax < 0:
        ax.set_ylim(ymin * 1.1, 0)
    if ymin > 0:
        ax.set_ylim(0, ymax * 1.1)
    #plt.show(block=False)
    return ax

def plot(prefix, time_data_1, time_data_2, plot_names, opm_csv_fname=None):
    plt.rc('font', size=12)
    plt_list = [(': oil rate',   'prod', 'OPM_FOPR'),
                (': water rate', 'prod', 'OPM_FWPR'),
                (': water rate', 'inj',  'OPM_FWIR')]

    opm_df = None
    if opm_csv_fname is not None:
        opm_df = pd.read_csv(opm_csv_fname, delimiter='\t')

    counter = 1
    for (plt_str, mode_suffix, opm_smry) in plt_list:
        #plt.subplot(len(plt_list), 1, counter)
        ax1 = plot_total_rate_darts(time_data_1, time_data_2, opm_df, opm_smry,
                                    plot_names, plt_str, mode_suffix)
        title = mode_suffix + ' ' + plt_str[2:] + ', sm$^3$/day'
        ax1.set(xlabel="Days", ylabel=title, title=title)
        #counter += 1
        #plt.show()
        png_fname = os.path.join(prefix, mode_suffix + '_' + plt_str[2:] + '.png')
        plt.savefig(png_fname)
        print('File', png_fname, 'saved')

#####################################################
def get_case_files(case):
    prefix = r'meshes/case_' + str(case)
    gridfile = prefix + r'/grid.grdecl'
    propfile = prefix + r'/reservoir.in'
    sch_file = prefix + r'/SCH.INC'
    return gridfile, propfile, sch_file


#####################################################

def run_test(args: list = []):
    if len(args) > 1:
        return test(case=args[0], overwrite=args[1])
    else:
        print('Not enough arguments provided')
        return 1, 0.0

def test(case, overwrite='0'):
    # 3 years by 1 month
    dt = 30
    n_time_steps = 12

    export_vtk = False#True

    #discr_types_list = ['cpp'] #cpg
    #discr_types_list = ['python'] #struct
    discr_types_list = ['cpp', 'python']

    # small grids
    #cases_list = [43] # 10x10x10 sloping fault
    #cases_list = [40] # 10x10x10 no fault
    #cases_list = [40, 43]

    # run all the  grids cases
    #cases_list = []
    #for c in os.listdir('../../meshes/cpg/'):
    #    if c.startswith("Case_"):
    #        cases_list.append(c)

    print('cases :', case)

    mode = 'run'      # run and plot results
    #mode = 'compare' # plot results
    print('MODE', mode)

    results = dict()

    gridfile, propfile, sch_fname = get_case_files(case)
    prefix = r'./meshes/case_' + str(case)
    if mode == 'run':
        for discr_type in discr_types_list:
            start = time.perf_counter()
            results[discr_type] = run(discr_type=discr_type, gridfile=gridfile, propfile=propfile,
                                      sch_fname=sch_fname, dt=dt, n_time_steps=n_time_steps,
                                      export_vtk=export_vtk)
            end = time.perf_counter()
    elif mode == 'compare':
        for discr_type in discr_types_list:
            results[discr_type] = pd.read_pickle(prefix + '/darts_time_data_' + discr_type + '.pkl')

    opm_fname = prefix + '/opm_flow_model/opm_rates.csv'
    if not os.path.exists(opm_fname):
        opm_fname = None

    if len(discr_types_list) > 1 and not results[discr_types_list[0]].empty and not results[discr_types_list[1]].empty:
        plot(prefix, results[discr_types_list[0]], results[discr_types_list[1]], discr_types_list, opm_fname)
    else:
        # if only one case did run, then plot it twice, because need to pass 2-nd arg to plot function
        plot(prefix, results[discr_types_list[0]], results[discr_types_list[0]], discr_types_list + ['None'], opm_fname)
        print('case', case, ': Skipped plotting due to some of data absence!')

    return 0, 0.0

if __name__ == '__main__':
    cases_list = [40, 43]
    for case in cases_list:
        test(case)
