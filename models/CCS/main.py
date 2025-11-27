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
m.init()
# set the output
m.set_output(verbose = True)

x = np.cumsum(m.x_axes)
y = np.linspace(m.reservoir.nz*2+1, 0, m.reservoir.nz)
X, Y = np.meshgrid(x, y)

properties = m.physics.vars + m.physics.property_operators[0].props_name
print_props = m.physics.vars + ['satV', 'xCO2', 'yH2O']
timesteps, output = m.output.output_properties(output_properties=print_props, timestep=0)
nv = m.physics.n_vars

fig, axs = plt.subplots(len(print_props), 1, figsize=(12, 10), dpi=100, facecolor='w', edgecolor='k')
for i, ith_prop in enumerate(print_props):
    if m.reservoir.nz > 1:
        prop = axs[i].pcolormesh(X, Y, output[ith_prop].reshape(m.reservoir.nz, m.reservoir.nx))
        plt.colorbar(prop, ax=axs[i])
    else:
        axs[i].plot(output[ith_prop])
    axs[i].set_title(ith_prop)

plt.savefig('step0.png', format='png')

for t in range(2):
    m.run(200)

    timesteps, output = m.output.output_properties(output_properties=print_props, timestep=t+1)

    fig, axs = plt.subplots(len(print_props), 1, figsize=(12, 10), dpi=100, facecolor='w', edgecolor='k')
    for i, ith_prop in enumerate(print_props):
        if m.reservoir.nz > 1:
            prop = axs[i].pcolormesh(X, Y, output[ith_prop].reshape(m.reservoir.nz, m.reservoir.nx))
            plt.colorbar(prop, ax=axs[i])
        else:
            axs[i].plot(output[ith_prop])
        axs[i].set_title(ith_prop + str(t+1))

    plt.savefig('step' + str(t+1) + '.png', format='png')

    # compute and save well time data in m.output_folder
    time_data_dict = m.output.store_well_time_data()

m.print_timers()
m.print_stat()
