import numpy as np
import pandas as pd
import os
from model import Model
from darts.engines import value_vector, redirect_darts_output

redirect_darts_output('binary.log')

n = Model()
n.init()

time = 10
n.run(time)

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
n.save_restart_data()

# compute well time data
time_data_dict = n.output.store_well_time_data()

# save well time data
time_data_df = pd.DataFrame.from_dict(time_data_dict)
time_data_df.to_pickle(os.path.join(n.output_folder, "well_time_data.pkl"))  # as a pickle file
writer = pd.ExcelWriter(os.path.join(n.output_folder, "well_time_data.xlsx"))  # as an excel file
time_data_df.to_excel(writer, sheet_name='Sheet1', index=False)
writer.close()