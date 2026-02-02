import numpy as np
import pandas as pd
import os
from darts.engines import redirect_darts_output
from model import Model

import matplotlib.pyplot as plt
redirect_darts_output('binary.log')


filename = 'out'

# define the model
m = Model()

# lower detfurth
(nr, nz) = (1000, 24)
if 1:
    poro = np.ones((nr, nz)) * 0.001
    perm = np.ones((nr, nz)) * 0.001
    poro[:, 4:20] = 0.2
    perm[:, 4:20] = 20
    perm[:, 6:12] = 100
else:
    poro = np.ones((nr, nz)) * 0.075
    perm = np.ones((nr, nz)) * 0.29
    # upper detfurth
    perm[:, :16] = 12.6
    # hardegsen
    poro[:, :10] = 0.09
    perm[:, :10] = 24
    # hardegsen high perm
    poro[:, :8] = 0.2
    perm[:, :8] = 550
    # hardegsen
    poro[:, :6] = 0.09
    perm[:, :6] = 24
    # caprock
    poro[:, :4] = 0.01
    perm[:, :4] = 0.01

m.set_reservoir(nr=nr, dr=1., nz=nz, dz=5, poro=poro.flatten(order='F'), perm=perm.flatten(order='F'))

# init the model
m.ms_well_flag = True
m.init()
# set the output
m.set_output(verbose = True)

""" DEFINE OUTPUT """
props = ['satV', 'rhoV']
output_props = ['pressure'] + props if not m.physics.thermal else ['pressure', 'temperature'] + props
# output_props += ['y' + comp for comp in m.components[2:]]
t_max = m.t_inj if not m.physics.thermal else max(m.reservoir.mesh.initial_state[m.physics.n_vars-1::m.physics.n_vars])
aspect = 'equal'  # 'equal', 'auto' or float
cmap = 'RdBu_r'
logx = True

timestep, property_array = m.output.output_properties(output_properties=output_props, timestep=-1)

m.reservoir.output_to_plt(data=property_array, output_props=output_props, plot_zeros=False,
                          aspect_ratio=aspect, logx=logx, cmap=cmap)
# m.output.output_to_vtk(ith_step=0, output_properties=output_props)  # initial conditions
plt.savefig('step0.png', format='png')

for t in range(2):
    m.run(200)

    timesteps, output = m.output.output_properties(output_properties=output_props, timestep=t+1)

    m.reservoir.output_to_plt(data=property_array, output_props=output_props, plot_zeros=False,
                              aspect_ratio=aspect, logx=logx, cmap=cmap)

    plt.savefig('step' + str(t+1) + '.png', format='png')

    # compute and save well time data in m.output_folder
    time_data_dict = m.output.store_well_time_data()

m.print_timers()
m.print_stat()
