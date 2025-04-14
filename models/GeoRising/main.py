from darts.engines import value_vector

from model import Model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


m = Model(iapws_physics=True)
m.init()#(platform='gpu')
m.set_output()

m.run(365/10)
m.print_timers()
m.print_stat()

output_props = m.physics.vars + m.output.properties
m.output.output_to_vtk(output_properties=output_props) # output all saved time steps to vtk

td = m.output.store_well_time_data()
for i, name in enumerate(td.keys()):
    plt.figure()
    plt.plot(td['time'], td[name])
    plt.savefig('output/figures/' + name + '.png')
    plt.close()


# td = pd.DataFrame.from_dict(m.physics.engine.time_data)
# td.to_pickle("darts_time_data.pkl")

# Save to pickle
import pickle
with open(m.output_folder + '/darts_time_data.pkl', 'wb') as f:
    pickle.dump(td, f)

# writer = pd.ExcelWriter('time_data.xlsx')
# td.to_excel(writer, sheet_name='Sheet1')
# writer.close()

# string = 'PRD : temperature'
# ax1 = td.plot(x='time', y=[col for col in td.columns if string in col])
# ax1.plot([0, runtime],[348, 348])
# ax1.tick_params(labelsize=14)
# ax1.set_xlabel('Days', fontsize=14)
# ax1.legend(['temp', 'limit'], fontsize=14)
# plt.grid()
# # plt.show()
# plt.savefig('out.png')