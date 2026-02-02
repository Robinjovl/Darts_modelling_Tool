"""
Three-phase black oil model
Unstructured grid
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model import Model
from darts.engines import value_vector, redirect_darts_output
from darts.tools.plot_darts import plot_phase_rate_darts
from darts.models.cicd_model import compare_solution_with_reference, get_platform


def run(platform='cpu'):

    redirect_darts_output('rum.log')
    m = Model()
    # m.params.linear_type = m.params.linear_solver_t.cpu_superlu
    m.init(platform=platform)
    m.set_output(verbose=True)

    # output primary (state) and secondary variables to .vtk files from engine.X at the current engine.time
    prop_list = m.physics.vars + m.output.properties
    m.output.output_to_vtk(ith_step=0,
                           output_directory=m.output_folder + '/vtk_files_from_engine',
                           output_properties=prop_list,
                           engine=True)

    if True:
        m.run(2000)
        m.print_timers()
        m.print_stat()
        m.output.print_simulation_parameters()

        m.output.output_to_vtk(ith_step=1,
                               output_directory=m.output_folder + '/vtk_files_from_engine',
                               output_properties=prop_list,
                               engine=True)
    else:
        # m.load_restart_data()
        m.load_restart_data('output/solution.h5')
        time_data = pd.read_pickle("darts_time_data.pkl")

    # compute and save well time data
    time_data_dict = m.output.store_well_time_data(save_output_files=True)

    # plot well time data
    # time_data_df = pd.DataFrame.from_dict(time_data_dict)
    # time_data_df.plot(x='time', y=['well_I1_BHP', 'well_P1_BHP'])\
    #     .get_figure().savefig(n.output_folder + '/bhp_plot.png', dpi=100, bbox_inches='tight')
    # time_data_df.plot(x='time', y=['well_P1_volumetric_rate_oil_at_wh', 'well_P5_volumetric_rate_oil_at_wh'])\
    #     .get_figure().savefig(n.output_folder + '/phase_rate_plot.png', dpi=100, bbox_inches='tight')

    # output primary (state) and secondary variables to .vtk files from the solution.h5 file for all available data points
    m.output.output_to_vtk(ith_step = None,
                           output_directory = m.output_folder + '/vtk_files_all_timesteps_from_h5',
                           output_properties = prop_list,
                           engine = False)

    m.print_timers()

    # for CI/CD
    failed, sim_time = compare_solution_with_reference(m=m)
    return failed

if __name__ == '__main__':
    exit(run(platform=get_platform()))
