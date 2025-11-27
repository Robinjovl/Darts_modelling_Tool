import numpy as np
import pandas as pd
import os
from darts.engines import value_vector, redirect_darts_output
from model import Model

import matplotlib.pyplot as plt
redirect_darts_output('binary.log')


filename = 'out'

# define the model
m = Model()

# init the model
m.ms_well_flag = True
m.init()
# set the output
m.set_output()

""" DEFINE OUTPUT """
props = ['satV', 'rho_g']
output_props = ['pressure'] + props if not m.physics.thermal else ['pressure', 'temperature'] + props
# output_props += ['y' + comp for comp in m.components[2:]]
lims = {'pressure': [m.p_inj - 10, m.p_inj + 10.], 'temperature': [m.t_inj - 30., m.t_init + 10.], 'satV': [1-m.swc, 1.]}
lims.update({'satLCO2': [0., 1.]})
aspect = 'equal'  # 'equal', 'auto' or float
cmap = 'RdBu_r'
logx = True

timestep, property_array = m.output.output_properties(output_properties=output_props, timestep=-1)

m.reservoir.output_to_plt(data=property_array, output_props=output_props, lims=lims, plot_zeros=False,
                          aspect_ratio=aspect, logx=logx, cmap=cmap)
# m.output.output_to_vtk(ith_step=0, output_properties=output_props)  # initial conditions
plt.savefig('step0.png', format='png')

for t in range(2):
    m.run(200)

    timesteps, output = m.output.output_properties(output_properties=output_props, timestep=t+1)

    m.reservoir.output_to_plt(data=property_array, output_props=output_props, lims=lims, plot_zeros=False,
                              aspect_ratio=aspect, logx=logx, cmap=cmap)

    plt.savefig('step' + str(t+1) + '.png', format='png')

    # compute and save well time data in m.output_folder
    time_data_dict = m.output.store_well_time_data()

m.print_timers()
m.print_stat()
