from darts.engines import value_vector

from model import Model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


m = Model(resolution=10)
m.init()
m.set_output()
m.output.output_to_vtk(ith_step = 0, engine = True)

m.run(365)
m.print_timers()
m.print_stat()
m.output.output_to_vtk(ith_step = 1, engine = True)

# compute well rates
well_rates_dict = m.output.store_well_time_data()
# print('\n'.join(well_rates_dict.keys()))

# save dataframe of well rates
td = pd.DataFrame.from_dict(well_rates_dict)
td.to_pickle(m.output_folder + "/darts_time_data.pkl")  # as a pickle file
writer = pd.ExcelWriter(m.output_folder + "/darts_time_data.xlsx")  # as an excel file
td.to_excel(writer, sheet_name='Sheet1')
writer.close()

td.plot(x = 'time', y = ['well_PRD_BHT', 'well_INJ_BHT'])\
    .get_figure().savefig(m.output_folder + '/well_temperature.png', dpi=100, bbox_inches='tight')
td.plot(x = 'time', y = ['well_PRD_BHP', 'well_INJ_BHP'])\
    .get_figure().savefig(m.output_folder + '/well_BHP.png', dpi=100, bbox_inches='tight')
td.plot(x = 'time', y = ['well_PRD_volumetric_rate_water_at_wh', 'well_PRD_volumetric_rate_water_by_sum_perfs'])\
    .get_figure().savefig(m.output_folder + '/well_production_rates.png', dpi=100, bbox_inches='tight')
td.plot(x = 'time', y = ['well_INJ_volumetric_rate_water_at_wh', 'well_INJ_volumetric_rate_steam_at_wh'])\
    .get_figure().savefig(m.output_folder + '/well_injection_rates.png', dpi=100, bbox_inches='tight')


# td = pd.DataFrame.from_dict(m.physics.engine.time_data)
# td.to_pickle("darts_time_data.pkl")
# writer = pd.ExcelWriter('time_data.xlsx')
# td.to_excel(writer, sheet_name='Sheet1')
# writer.close()

# string = 'PRD : temperature'
# ax1 = td.plot(x='time', y=[col for col in td.columns if string in col])
# #ax1.plot([0, 3650],[348, 348])
# ax1.tick_params(labelsize=14)
# ax1.set_xlabel('Days', fontsize=14)
# ax1.legend(['temp', 'limit'], fontsize=14)
# plt.grid()
# plt.savefig('prod_temperature.png')
