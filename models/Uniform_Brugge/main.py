"""
Three-phase black oil model
Unstructured grid
"""

import numpy as np
import pandas as pd
import os

from model import Model
from darts.engines import value_vector, redirect_darts_output

if __name__ == '__main__':
    redirect_darts_output('run.log')
    n = Model()
    # n.params.linear_type = n.params.linear_solver_t.cpu_superlu
    n.init()
    n.set_output()

    prop_list = n.physics.vars + n.output.properties
    # output primary (state) and secondary variables to .vtk files from engine.X at the current engine.time
    n.output.output_to_vtk(ith_step = 0,
                           output_directory = n.output_folder + '/vtk_files_from_engine',
                           output_properties = prop_list,
                           engine = True)

    if True:
        n.run(2000)
        n.print_timers()
        n.print_stat()
        n.output.print_simulation_parameters()

        n.output.output_to_vtk(ith_step = 1,
                               output_directory = n.output_folder + '/vtk_files_from_engine',
                               output_properties = prop_list,
                               engine = True)

    else:
        n.load_restart_data('output/solution.h5')
        time_data = pd.read_pickle("darts_time_data.pkl")

    # compute well rates
    well_rates_dict = n.output.store_well_time_data()

    # save dataframe of well rates
    time_data_df = pd.DataFrame.from_dict(well_rates_dict)
    time_data_df.to_pickle(os.path.join(n.output_folder, "well_time_data.pkl"))  # as a pickle file
    writer = pd.ExcelWriter(os.path.join(n.output_folder, "well_time_data.xlsx"))  # as an excel file
    time_data_df.to_excel(writer, sheet_name='Sheet1')
    writer.close()

    # plot well data
    # td.plot(x='time', y=['well_I1_BHP', 'well_P1_BHP'])\
    #     .get_figure().savefig(n.output_folder + '/bhp_plot.png', dpi=100, bbox_inches='tight')
    # td.plot(x='time', y=['well_P1_volumetric_rate_oil_at_wh', 'well_P5_volumetric_rate_oil_at_wh'])\
    #     .get_figure().savefig(n.output_folder + '/phase_rate_plot.png', dpi=100, bbox_inches='tight')

    # output primary (state) and secondary variables to .vtk files from the solution.h5 file for all available data points
    n.output.output_to_vtk(ith_step = None,
                           output_directory = n.output_folder + '/vtk_files_all_timesteps_from_h5',
                           output_properties = prop_list,
                           engine = False)