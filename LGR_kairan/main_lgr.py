import numpy as np
import pandas as pd
import sys, os
from LGR_1 import Model
from darts.engines import value_vector, redirect_darts_output
import matplotlib.pyplot as plt
from darts.physics.base.operators_base import PropertyOperators as props

def plot_sol(n):
    Xn = np.array(n.physics.engine.X, copy=False)
    nc = n.property_container.nc + n.thermal
    P = Xn[0:n.reservoir.nb * nc:nc]
    z = np.ones((nc, n.reservoir.nb))
    phi = np.ones(n.reservoir.nb)
    sat_ev = props(n.property_container)
    prop = np.zeros(2*n.property_container.nph)

    plt.figure(num=1, figsize=(12, 8), dpi=100)
    for i in range(nc-1):
        z[i][:] = Xn[i + 1:n.reservoir.nb * nc:nc]
        z[-1][:] -= z[i][:]

    for i in range(n.reservoir.nb):
        state = Xn[i*nc:(i+1)*nc]
        sat_ev.evaluate(state, prop)
        density_tot = np.sum(prop[0:3] * prop[3:6])
        phi[i] -= prop[2]  # (z[-1, i] * density_tot / prop[-1])

    for i in range(3):
        plt.subplot(330 + (i + 1))
        plt.plot(z[i]/(1-z[3]))
        plt.title('Composition' + str(i + 1), y=1)

    i = 3
    plt.subplot(330 + (i + 1))
    plt.plot(phi)
    plt.title('Porosity', y=1)

    i = 4
    plt.subplot(330 + (i + 1))
    plt.plot(P)
    plt.title('Pressure', y=1)

    plt.show()


if __name__ == '__main__':
    darts_model = Model()
    # darts_model.params.linear_type = n.params.linear_solver_t.cpu_superlu
    darts_model.init()
    darts_model.set_output()


    if True:
        darts_model.run(50)
        # darts_model.reservoir.wells[0].control = n.physics.new_bhp_inj(100, 3*[n.zero])
        # darts_model.run_python(300, restart_dt=1e-3)
        darts_model.print_timers()
        darts_model.print_stat()

        # compute well time data
        time_data_dict = darts_model.output.store_well_time_data(save_output_files=True)
        time_data_df = pd.DataFrame.from_dict(time_data_dict)

        darts_model.output.plot_well_time_data(types_of_well_rates=["phases_volumetric_rates"])

    else:
        # darts_model.load_restart_data()
        darts_model.load_restart_data('output/solution.h5')
        time_data = pd.read_pickle("darts_time_data.pkl")

    if True:
        Xn = np.array(darts_model.physics.engine.X, copy=False)
        nc = darts_model.physics.nc + darts_model.physics.thermal
        nb = darts_model.reservoir.mesh.n_res_blocks

        plt.figure(num=1, figsize=(12, 8), dpi=100)
        for i in range(nc if nc < 3 else 3):
            plt.subplot(330 + (i + 1))
            plt.plot(Xn[i:nb*nc:nc])
        plt.savefig('out.png')
    else:
        #plot_sol(n)
        n.print_and_plot('sim_data')

#z_c10 = Xn[nc-1:n.reservoir.nb*nc:nc]

# rho_aq = n.property_container.density_ev['wat'].evaluate(P, z_co2)
# Sg = np.zeros(n.reservoir.nb)
#
# for i in range (n.reservoir.nb):
#     x_list = Xn[i*nc:(i+1)*nc]
#     state = value_vector(x_list)
#     Sg[i] = n.properties(state)

# """ start plots """
# plt.figure(num=1, figsize=(12, 8), dpi=100)
# """ sg and x """
# plt.subplot(211)
# plt.plot(z1)
# #plt.imshow(np.reshape(T, (220, 60)).T)
#
# plt.title('First composition', y=1)
#
# plt.subplot(212)
# plt.plot(P)
# #plt.imshow(np.reshape(P, (220, 60)).T)
# plt.title('Pressure', y=1)

# """ sg and x """
# plt.subplot(223)
# plt.plot(P)
# plt.title('Pressure', y=1)
#
# plt.subplot(224)
# plt.plot(T)
# plt.title('Gas saturation', y=1)