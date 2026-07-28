import numpy as np
import pandas as pd
import os

from model import Model
from model_therm import Model_therm
from darts.engines import value_vector, redirect_darts_output
import matplotlib.pyplot as plt
from darts.physics.base.operator_evaluator import PropertyOperators as props

if __name__ == '__main__':

    redirect_darts_output('run.log')
    from darts.engines import well_control_iface

    if 1:
        n = Model()
    else:
        n = Model_therm()

    n.init()
    n.set_output()
    w = n.reservoir.wells[0]
    n.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.MOLAR_RATE, is_inj=True,
                                target=200, phase_name='wat', inj_composition=[n.zero, 0.8, 0.2 - 2 * n.zero],
                                inj_temp=300)
    n.run(1000)
    n.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.MOLAR_RATE, is_inj=True,
                                target=20, phase_name='gas', inj_composition=[1.0 - 3 * n.zero, n.zero, n.zero],
                                inj_temp=350)
    n.run(1000, restart_dt=1e-8)


    Xn = np.array(n.physics.engine.X, copy=False)
    nc = n.physics.nc + n.physics.thermal
    nb = n.reservoir.mesh.n_res_blocks

    plt.figure(num=1, figsize=(12, 8), dpi=100)
    for i in range(n.physics.n_vars):
        num = n.physics.n_vars * 100 + 10 + (i + 1)
        plt.subplot(num)
        plt.plot(Xn[i:nb*nc:nc])
    plt.savefig('out.png')

# nb = n.reservoir.mesh.n_res_blocks
# P = Xn[0:nb*nc:nc]
# z_co2 = Xn[1:nb*nc:nc]
#
# rho_aq = n.physics.property_containers[0].density_ev['wat'].evaluate(P, z_co2)
# Sg = np.zeros(nb)
#
# for i in range (nb):
#     x_list = Xn[i*nc:(i+1)*nc]
#     state = value_vector(x_list)
#     Sg[i] = n.properties(state)
#
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
#
# """ sg and x """
# plt.subplot(223)
# plt.plot(P)
# plt.title('Pressure', y=1)
#
# plt.subplot(224)
# plt.plot(T)
# plt.title('Gas saturation', y=1)
#
#
# time_data1 = pd.DataFrame.from_dict(n.physics.engine.time_data)
# from darts.tools.plot_darts import *
# writer = pd.ExcelWriter('time_data.xlsx')
# plot_phase_rate_darts('I1', time_data1, 'wat')
#
#
# plt.show()
#
# from darts.tools.plot_darts import *
# time_data1 = pd.DataFrame.from_dict(n.physics.engine.time_data)
# for i, w in enumerate(n.reservoir.wells):
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
