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

zero = 1e-10
m.set_physics(zero, n_points=1001, temperature=None)

if 1:
    def set_initial_conditions():
        dz = m.reservoir.global_data['dz'][0, 0, :]

        # zH2O = 1
        from darts.physics.super.initialize import Initialize
        # depth corresponding to boundary_idx = 10
        b_depth = m.reservoir.global_data['depth'].min() + (m.reservoir.global_data['depth'].max() - m.reservoir.global_data['depth'].min()) / 4.
        boundary_state = {'H2O': 1 - zero, 'pressure': 100., 'temperature': 350.}
        init = Initialize(physics=m.physics)
        X = init.solve(depth_bottom=m.reservoir.global_data['depth'].max(),
                       depth_top=m.reservoir.global_data['depth'].min(),
                       depth_known=b_depth, boundary_state=boundary_state,
                       primary_specs={'H2O': 1-zero}, secondary_specs={})
        m.set_initial_conditions_from_depth_table(depth=init.depths,
                                                  initial_distribution={var: X[i::m.physics.n_vars] for i, var in enumerate(m.physics.vars)})


    m.set_initial_conditions = set_initial_conditions
else:
    m.initial_values = {"pressure": 100.,
                        "H2O": 0.99995,
                        "temperature": 350.
                        }
m.inj_stream = [0.00005]
m.inj_stream += [350.] if m.physics.thermal else []
m.p_inj = 100.
m.p_prod = 50.

m.set_sim_params(first_ts=1e-3, mult_ts=1.5, max_ts=5, tol_newton=1e-3, tol_linear=1e-5, it_newton=10, it_linear=50)

# init the model
m.init()

x = np.cumsum(m.x_axes)
y = np.linspace(m.reservoir.nz*2+1, 0, m.reservoir.nz)
X, Y = np.meshgrid(x, y)

properties = m.physics.vars + m.physics.property_operators[0].props_name
print_props = m.physics.vars + ['satV', 'xCO2', 'yH2O']
timesteps, output = m.output_properties(print_props[m.physics.n_vars:], timestep=0)
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
    time_data = pd.DataFrame.from_dict(m.physics.engine.time_data)
    m.print_timers()
    m.print_stat()

    #m.params.max_ts = 0.5

    timesteps, output = m.output_properties(print_props[m.physics.n_vars:], timestep=t+1)

    fig, axs = plt.subplots(len(print_props), 1, figsize=(12, 10), dpi=100, facecolor='w', edgecolor='k')
    for i, ith_prop in enumerate(print_props):
        if m.reservoir.nz > 1:
            prop = axs[i].pcolormesh(X, Y, output[ith_prop].reshape(m.reservoir.nz, m.reservoir.nx))
            plt.colorbar(prop, ax=axs[i])
        else:
            axs[i].plot(output[ith_prop])
        axs[i].set_title(ith_prop + str(t+1))

    plt.savefig('step' + str(t+1) + '.png', format='png')

td = pd.DataFrame.from_dict(m.physics.engine.time_data)
td.to_pickle("darts_time_data.pkl")
writer = pd.ExcelWriter('time_data.xlsx')
td.to_excel(writer, sheet_name='Sheet1')
writer.close()

