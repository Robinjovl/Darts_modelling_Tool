# from darts.engines import value_vector
from model import Model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

m = Model(iapws_physics=True)
m.init() #(platform='gpu')
m.set_output()

m.run(36.5)
m.print_timers()
m.print_stat()

output_props = m.physics.vars + m.output.properties
m.output.output_to_vtk(output_properties=output_props) # output all saved time steps to vtk

# compute well rates
well_rates_dict = m.output.store_well_time_data()

# save dictionary of well rates
td = pd.DataFrame.from_dict(well_rates_dict)
td.to_pickle(m.output_folder + "/darts_time_data.pkl") # as a pickle file
writer = pd.ExcelWriter(m.output_folder + "/darts_time_data.xlsx") # as an excel file
td.to_excel(writer, sheet_name='Sheet1')
writer.close()

td.plot(x='time', y=['well_INJ_volumetric_rate_water_at_wh', 'well_PRD_volumetric_rate_water_at_wh'])
plt.show()

ax = td.plot(x='time', y=['well_INJ_BHP', 'well_PRD_BHP'], style=['-b', '-r'], label=['INJ BHP', 'PRD BHP'])
ax.set_ylabel('BHP [bar]')
ax2 = ax.twinx()
td.plot(x='time', y=['well_INJ_BHT', 'well_PRD_BHT'], ax=ax2, style=['--b', '--r'], label=['INJ BHT', 'PRD BHT'])
ax2.set_ylabel('BHT [K]')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='best')
plt.tight_layout()
plt.show()

# td = pd.DataFrame.from_dict(m.physics.engine.time_data)
# td.to_pickle("darts_time_data.pkl")
# string = 'PRD : temperature'
# ax1 = td.plot(x='time', y=[col for col in td.columns if string in col])
# ax1.plot([0, runtime],[348, 348])
# ax1.tick_params(labelsize=14)
# ax1.set_xlabel('Days', fontsize=14)
# ax1.legend(['temp', 'limit'], fontsize=14)
# plt.grid()
# # plt.show()
# plt.savefig('out.png')