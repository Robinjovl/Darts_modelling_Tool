import numpy as np
import pandas as pd
from darts.engines import value_vector, redirect_darts_output
from model import Model

import matplotlib.pyplot as plt
redirect_darts_output('binary.log')


filename = 'out'

# define the model
m = Model()

m.set_reservoir()
m.set_wells()

zero = 1e-10
m.set_physics(zero, n_points=1001, temperature=None)

m.initial_values = {m.physics.vars[0]: 100.,
                    m.physics.vars[1]: 0.99995,
                    "temperature": 350.
                    }
m.inj_stream = [0.00005] + [350.] if m.physics.thermal else [0.00005]
m.p_inj = 100.
m.p_prod = 50.

m.set_sim_params(first_ts=1e-3, mult_ts=1.5, max_ts=5, tol_newton=1e-3, tol_linear=1e-5, it_newton=10, it_linear=50)

# init the model
m.init()

m.run_python(250)
m.print_timers()
m.print_stat()

x = np.cumsum(m.x_axes)
y = np.linspace(m.reservoir.nz*2+1, 0, m.reservoir.nz)
X, Y = np.meshgrid(x, y)

properties = m.physics.vars + m.physics.property_operators.props_name
output = m.output_properties()
print_props = [0, 1, 2, 3, 5, 6]

# fig, axs = plt.subplots(len(print_props), 1, figsize=(12, 10), dpi=100, facecolor='w', edgecolor='k')
# for i, ith_prop in enumerate(print_props):
#     #prop = axs[i].pcolormesh(X, Y, output[:, ith_prop].reshape(m.reservoir.nz, m.reservoir.nx))
#     #plt.colorbar(prop, ax=axs[i])
#     axs[i].plot(output[:, ith_prop])
#     axs[i].set_title(properties[ith_prop])
#
# plt.savefig('step0.png', format='png')

for t in range(2):
    m.run_python(60)
    time_data = pd.DataFrame.from_dict(m.engine.time_data)
    m.print_timers()
    m.print_stat()

    #m.params.max_ts = 0.5

    output = m.output_properties()

    fig, axs = plt.subplots(len(print_props), 1, figsize=(12, 10), dpi=100, facecolor='w', edgecolor='k')
    for i, ith_prop in enumerate(print_props):
        # prop = axs[i].pcolormesh(X, Y, output[:, ith_prop].reshape(m.reservoir.nz, m.reservoir.nx))
        # plt.colorbar(prop, ax=axs[i])
        axs[i].plot(output[:, ith_prop])
        axs[i].set_title(properties[ith_prop])

    plt.savefig('step' + str(t+1) + '.png', format='png')

td = pd.DataFrame.from_dict(m.engine.time_data)
td.to_pickle("darts_time_data.pkl")
writer = pd.ExcelWriter('time_data.xlsx')
td.to_excel(writer, 'Sheet1')
writer.save()

plt.show()
