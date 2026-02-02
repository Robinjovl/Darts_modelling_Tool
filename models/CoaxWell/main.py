from darts.engines import value_vector

from model import Model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from darts.models.cicd_model import compare_solution_with_reference, get_platform


def run(platform='cpu'):
    m = Model(resolution=10)
    m.init()
    m.set_output()
    m.output.output_to_vtk(ith_step=0, output_directory='vtk', engine=True)

    m.run(365)
    m.print_timers()
    m.print_stat()
    m.output.output_to_vtk(ith_step=1, output_directory='vtk', engine=True)

    ########################################### SAJJAD, PLS UPDATE THE RATE CALCULATORS
    # # compute and save well time data
    # time_data_dict = m.output.store_well_time_data(save_output_files=True)
    #
    # plot well time data
    # time_data_df = pd.DataFrame.from_dict(time_data_dict)
    # time_data_df.plot(x='time', y=['well_PRD_BHT', 'well_INJ_BHT'])\
    #     .get_figure().savefig(m.output_folder + '/well_temperature.png', dpi=100, bbox_inches='tight')
    # time_data_df.plot(x='time', y=['well_PRD_BHP', 'well_INJ_BHP'])\
    #     .get_figure().savefig(m.output_folder + '/well_BHP.png', dpi=100, bbox_inches='tight')
    # time_data_df.plot(x='time', y=['well_PRD_volumetric_rate_water_at_wh', 'well_PRD_volumetric_rate_water_by_sum_perfs'])\
    #     .get_figure().savefig(m.output_folder + '/well_production_rates.png', dpi=100, bbox_inches='tight')
    # time_data_df.plot(x='time', y=['well_INJ_volumetric_rate_water_at_wh', 'well_INJ_volumetric_rate_steam_at_wh'])\
    #     .get_figure().savefig(m.output_folder + '/well_injection_rates.png', dpi=100, bbox_inches='tight')
    ############################################


    # for CI/CD
    failed, sim_time = compare_solution_with_reference(m=m)
    return failed


if __name__ == '__main__':
    exit(run(platform=get_platform()))
