from darts.engines import value_vector

from model import Model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

m = Model(resolution=10)
m.init()
m.set_output()
m.output.output_to_vtk(ith_step = 0, engine = True)

m.run(365)
m.print_timers()
m.print_stat()
m.output.output_to_vtk(ith_step = 1, engine = False)

########################################### SAJJAD, PLS UPDATE THE RATE CALCULATORS
# # compute well time data
# time_data_dict = m.output.store_well_time_data()
# save well time data
# time_data_df = pd.DataFrame.from_dict(time_data_dict)
# time_data_df.to_pickle(os.path.join(m.output_folder, "well_time_data.pkl"))  # as a pickle file
# writer = pd.ExcelWriter(os.path.join(m.output_folder, "well_time_data.xlsx"))  # as an excel file
# time_data_df.to_excel(writer, sheet_name='Sheet1', index=False)
# writer.close()
#
# time_data_df.plot(x='time', y=['well_PRD_BHT', 'well_INJ_BHT'])\
#     .get_figure().savefig(m.output_folder + '/well_temperature.png', dpi=100, bbox_inches='tight')
# time_data_df.plot(x='time', y=['well_PRD_BHP', 'well_INJ_BHP'])\
#     .get_figure().savefig(m.output_folder + '/well_BHP.png', dpi=100, bbox_inches='tight')
# time_data_df.plot(x='time', y=['well_PRD_volumetric_rate_water_at_wh', 'well_PRD_volumetric_rate_water_by_sum_perfs'])\
#     .get_figure().savefig(m.output_folder + '/well_production_rates.png', dpi=100, bbox_inches='tight')
# time_data_df.plot(x='time', y=['well_INJ_volumetric_rate_water_at_wh', 'well_INJ_volumetric_rate_steam_at_wh'])\
#     .get_figure().savefig(m.output_folder + '/well_injection_rates.png', dpi=100, bbox_inches='tight')
############################################
