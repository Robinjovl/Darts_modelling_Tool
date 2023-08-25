import numpy as np
import pandas as pd

from model import Model
from darts.engines import value_vector, redirect_darts_output
import matplotlib.pyplot as plt
from darts.physics.super.operator_evaluator import PropertyOperators as props


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

def run_darts(mode):
    redirect_darts_output('run_' + mode + '.log')
    n = Model(mode=mode)
    n.init()

    if mode != 'plot':
        n.run()
        n.print_timers()
        n.print_stat()

        time_data = pd.DataFrame.from_dict(n.physics.engine.time_data)
        time_data.to_pickle("darts_time_data.pkl")
        n.save_restart_data()
        writer = pd.ExcelWriter('time_data.xlsx')
        time_data.to_excel(writer, 'Sheet1')
        writer.close()

        Xn = np.array(n.physics.engine.X, copy=False)
        np.save(mode + '.npy', Xn)
    else:
        Xn_rhs = np.load('rhs.npy')
        Xn_wells = np.load('wells.npy')
        nc = n.physics.nc + n.physics.thermal
        plt.autoscale(False)
        plt.ylim(0, 400)
        plt.xlim(0, n.reservoir.nx - 1)
        plt.plot(Xn_rhs[0:n.reservoir.nb*nc:nc], label='rhs')
        plt.plot(Xn_wells[0:n.reservoir.nb * nc:nc], label='wells')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    # run with prod. well, and save solution vector at the last timestep to wells.npy
    # run with rhs_flux, and save solution vector at the last timestep to rhs.npy
    # read .npy file and plot
    mode_list = ['wells', 'rhs', 'plot']
    #mode_list = ['rhs']

    for mode in mode_list:
        run_darts(mode)

