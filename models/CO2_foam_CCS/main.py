import numpy as np
import pandas as pd
from model import Model
from darts.engines import value_vector, redirect_darts_output

redirect_darts_output('binary.log')

n = Model()
n.init()

time = 10
n.run_python(time)

Xn = np.array(n.physics.engine.X, copy=False)
P = Xn[0::2]
z_co2 = Xn[1::2]

num_wells_tot = 1
tot_properties = 2
tot_unknws = n.reservoir.unstr_discr.matrix_cell_count + num_wells_tot*2
n.cell_property = ['pressure', 'composition']
property_array = np.empty((tot_unknws, tot_properties))
property_array[:, 0] = P
property_array[:, 1] = z_co2

n.reservoir.unstr_discr.write_to_vtk('results', property_array, n.cell_property, time)


n.print_timers()
n.print_stat()
time_data = pd.DataFrame.from_dict(n.physics.engine.time_data)
time_data.to_pickle("darts_time_data.pkl")
n.save_restart_data()


