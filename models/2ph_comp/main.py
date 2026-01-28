import numpy as np
import pandas as pd
import sys, os
from model import Model
from darts.engines import value_vector, redirect_darts_output
import matplotlib.pyplot as plt
from darts.physics.base.operators_base import PropertyOperators as props
from darts.models.cicd_model import compare_solution_with_reference, get_platform


def plot_sol(m):
    Xn = np.array(m.physics.engine.X, copy=False)
    nc = m.property_container.nc + m.thermal
    P = Xn[0:m.reservoir.nb * nc:nc]
    z = np.ones((nc, m.reservoir.nb))
    phi = np.ones(m.reservoir.nb)
    sat_ev = props(m.property_container)
    prop = np.zeros(2*m.property_container.nph)

    plt.figure(num=1, figsize=(12, 8), dpi=100)
    for i in range(nc-1):
        z[i][:] = Xn[i + 1:m.reservoir.nb * nc:nc]
        z[-1][:] -= z[i][:]

    for i in range(m.reservoir.nb):
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


def run(platform='cpu'):
    duplicate_output_to_file("run.log")

    print('START')

    m = Model()
    # m.params.linear_type = m.params.linear_solver_t.cpu_superlu
    m.init(platform=platform)
    m.set_output()

    if True:
        m.run(1000)
        # m.reservoir.wells[0].control = m.physics.new_bhp_inj(100, 3*[m.zero])
        # m.run_python(300, restart_dt=1e-3)
        m.print_timers()
        m.print_stat()

        # compute and save well time data
        time_data_dict = m.output.store_well_time_data(save_output_files=True)

        # plot well time data
        m.output.plot_well_time_data(types_of_well_rates=["phases_volumetric_rates"])
    else:
        # m.load_restart_data()
        m.load_restart_data('output/solution.h5')
        time_data = pd.read_pickle("darts_time_data.pkl")

    if True:
        Xn = np.array(m.physics.engine.X, copy=False)
        nc = m.physics.nc + m.physics.thermal
        nb = m.reservoir.mesh.n_res_blocks

        plt.figure(num=1, figsize=(12, 8), dpi=100)
        for i in range(nc if nc < 3 else 3):
            plt.subplot(330 + (i + 1))
            plt.plot(Xn[i:nb*nc:nc])
        plt.savefig('out.png')
    else:
        #plot_sol(n)
        m.print_and_plot('sim_data')

    print('END')

    # for CI/CD
    failed, sim_time = compare_solution_with_reference(m=m)
    return failed

#z_c10 = Xn[nc-1:m.reservoir.nb*nc:nc]

# rho_aq = m.property_container.density_ev['wat'].evaluate(P, z_co2)
# Sg = np.zeros(m.reservoir.nb)
#
# for i in range (m.reservoir.nb):
#     x_list = Xn[i*nc:(i+1)*nc]
#     state = value_vector(x_list)
#     Sg[i] = m.properties(state)

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


# time_data1 = pd.DataFrame.from_dict(m.physics.engine.time_data)
# from darts.tools.plot_darts import *
# writer = pd.ExcelWriter('time_data.xlsx')
# plot_phase_rate_darts('I1', time_data1, 'wat')
#
#
# plt.show()

# from darts.tools.plot_darts import *
# time_data1 = pd.DataFrame.from_dict(m.physics.engine.time_data)
# for i, w in enumerate(m.reservoir.wells):
#     # plot oil rate
#     ax1 = plot_oil_rate_darts(w.name, time_data1, color='b')
#     ax1.tick_params(labelsize=14)
#     ax1.set_xlabel('Days', fontsize=14)
#
#     ax3 = plot_gas_rate_darts(w.name, time_data1, color='b')
#     ax3.tick_params(labelsize=14)
#     ax3.set_xlabel('Days', fontsize=14)
#
# plt.show()

if __name__ == '__main__':
    exit(run(platform=get_platform()))
