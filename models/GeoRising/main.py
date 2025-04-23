from darts.engines import value_vector

from model import Model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


m = Model(iapws_physics=True)

m.init()
# m.init(platform='gpu')

output_props=['pressure', 'temperature', 'enthalpy']
m.output_to_vtk(ith_step=0, output_directory='vtk', output_properties=output_props)

runtime = 365
m.run(runtime)
m.print_timers()
m.print_stat()
m.output_to_vtk(ith_step=1, output_directory='vtk', output_properties=output_props)

td = pd.DataFrame.from_dict(m.physics.engine.time_data)
td.to_pickle("darts_time_data.pkl")
writer = pd.ExcelWriter('time_data.xlsx')
td.to_excel(writer, sheet_name='Sheet1')
writer.close()

string = 'PRD : temperature'
ax1 = td.plot(x='time', y=[col for col in td.columns if string in col])
ax1.plot([0, runtime],[348, 348])
ax1.tick_params(labelsize=14)
ax1.set_xlabel('Days', fontsize=14)
ax1.legend(['temp', 'limit'], fontsize=14)
plt.grid()
# plt.show()
plt.savefig('out.png')