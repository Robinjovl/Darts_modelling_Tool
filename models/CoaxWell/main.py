from darts.engines import value_vector

from model import Model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


m = Model(resolution=10)

m.init()
m.export_pro_vtk()
m.run(365)
m.print_timers()
m.print_stat()
m.export_pro_vtk()


td = pd.DataFrame.from_dict(m.physics.engine.time_data)
td.to_pickle("darts_time_data.pkl")
writer = pd.ExcelWriter('time_data.xlsx')
td.to_excel(writer, 'Sheet1')
writer.close()

string = 'PRD : temperature'
ax1 = td.plot(x='time', y=[col for col in td.columns if string in col])
#ax1.plot([0, 3650],[348, 348])
ax1.tick_params(labelsize=14)
ax1.set_xlabel('Days', fontsize=14)
ax1.legend(['temp', 'limit'], fontsize=14)
plt.grid()
plt.savefig('prod_temperature.png')
