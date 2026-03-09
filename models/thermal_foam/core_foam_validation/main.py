import sys
import numpy as np
import pandas as pd
import os

from model import Model
from darts.engines import value_vector, redirect_darts_output
from darts.physics.base.operators_base import PropertyOperators as props

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
import matplotlib as mpl
import scienceplots
from cycler import cycler
plt.style.use(['science', 'notebook', 'no-latex'])

# Get data from FOSSIL
def read_fossil_pressure_drop(path):
    arq = os.path.join(path,"pressure_drop.csv")
    p = pd.read_csv(arq).to_numpy()
    time, dp = p[:,0], p[:,1]
    return time, dp

def read_fossil_production(path):
    arq = os.path.join(path,'production.csv')
    production = pd.read_csv(arq).to_numpy()[:,6]
    time = pd.read_csv(arq).to_numpy()[:,0]
    return time, production

def plot_sol(n):
    Xn = np.array(n.physics.engine.X, copy=False)
    nc = n.property_container.nc + n.thermal
    nb = n.reservoir.mesh.n_res_blocks
    P = Xn[0:nb * nc:nc]
    z = np.ones((nc, nb))
    phi = np.ones(nb)
    sat_ev = props(n.property_container)
    prop = np.zeros(2*n.property_container.nph)

    plt.figure(num=1, figsize=(12, 8), dpi=100)
    for i in range(nc-1):
        z[i][:] = Xn[i + 1:n.reservoir.nb * nc:nc]
        z[-1][:] -= z[i][:]

    for i in range(nb):
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

    run_simulation = True

    plt.style.use('science')
    plt.rcParams.update({'font.size': 22})
    colors = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51", "#a98467", "#8fbcbb", "#1d70a3"]
    mpl.rcParams.update({
    'lines.linewidth': 2,
    # 'font.family': 'Times New Roman',
    'axes.labelsize': 30,
    'xtick.labelsize': 25,
    'ytick.labelsize': 25,
    'legend.fontsize': 20,
    'font.size': 22,
    'axes.prop_cycle': cycler(color=colors)
    })

    redirect_darts_output('run.log')
    n = Model()
    n.init()
    n.set_output()

    fgs = [0.5, 0.6, 0.7, 0.75, 0.8, 0.9]
    sim_time_sec = np.array([10830, 10521, 7238, 8019, 10512, 5095]) # [s]

    sim_time = sim_time_sec / (60*60*24) # to days
    print(sim_time)
    save_interval=1e-3
    DP = np.zeros_like(fgs)
    dp_transient = []

    if run_simulation:
        for i in range(len(fgs)):
            n.set_fg_Inj(fg=fgs[i])
            for j in range(int(sim_time[i]/save_interval)):
                n.run(save_interval)

                # Get pressure drop
                Xn = np.array(n.physics.engine.X, copy=False)
                nc = n.physics.nc + n.physics.thermal
                nb = n.reservoir.mesh.n_res_blocks
                for k in range(nc if nc < 3 else 3):
                    if k == 0:
                        p = Xn[k:nb*nc:nc]
                        dpn = (p[2]-p[-1]) / 10
                        dp_transient.append(dpn) # 1 bar = 0.1 MPa
                DP[i] = dpn

        n.print_timers()
        n.print_stat()

        # compute well time data
        time_data_dict = n.output.store_well_time_data()

        # POST-PROCESSING
        # Get PV
        volume = np.array(n.reservoir.mesh.volume, copy=False)
        volume = volume[:n.reservoir.nx] # 6 additional volumes at the end are 2 monitoring cells for each reservoir/perforation
        poro = np.array(n.reservoir.mesh.poro, copy=False)
        poro = poro[:n.reservoir.nx]
        volume_poro = sum(volume * poro)
        print("Volume = " + str(sum(volume[:n.reservoir.nx])))
        print("Pore volume = " + str(sum(volume * poro)))

        # Get simulation time vector
        time_data_df = pd.DataFrame.from_dict(time_data_dict)
        time = time_data_df['time'].to_numpy()
        dt = time[1:] - time[:-1]
        time_sec = time[1:]*(24*60*60)

        # Get Cumulative water production
        q_w = np.abs(time_data_df['well_P1_volumetric_rate_wat_at_wh'].to_numpy()) # TODO: The outputs are saved along timeSteps.. how to save p, Sw, etc also every M*ts?
        q_w = q_w[1:]
        Q_w = np.zeros_like(dt)
        for i in range(Q_w.shape[0]):
            Q_w[i] = np.sum(np.multiply(dt[:i],q_w[:i]))
        Q_w /= volume_poro

        # Load FOSSIL data
        path = 'output_core16-STARS-FOSSIL'
        tdp, dp = read_fossil_pressure_drop(path)
        tp, p = read_fossil_production(path)

        # PLOTS
        fig, axs = plt.subplots(3, 1, figsize=(15,15))

        gradient = LinearSegmentedColormap.from_list("grad", ["#264653", "#e9c46a"])
        palet = [gradient(i) for i in np.linspace(0, 1, 6)]
        palet_hex = [f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}" for r, g, b, _ in palet]
        fg_colors = {50: palet_hex[0], 60: palet_hex[1], 70: palet_hex[2], 75: palet_hex[3], 80: palet_hex[4], 90: palet_hex[5]}

        # dp x fg - stationary
        # Load Valdez data
        Valdez_fg, Valdez_dp = np.loadtxt('valdez_2021_stationary.txt', delimiter=',')
        axs[0].plot(Valdez_fg, Valdez_dp, label='Valdez et al., 2021')
        axs[0].scatter(fgs, DP, c='k', label='OpenDARTS')
        axs[0].grid(alpha=0.4)
        axs[0].set_xlabel('$f_g$ [-]')
        axs[0].set_ylabel('$\mu_{app}$ [Pa$\cdot$s]')
        axs[0].legend(frameon=True)

        # dp x time
        # Load experimental pressure drop
        lasurf_data = pd.read_csv('core16_LASURF_filtered.csv')
        for fg_value, group in lasurf_data.groupby('fg'):
            axs[1].plot(group['Time'], group['Pressure Drop'], 
                        color=fg_colors[fg_value],
                        label=f'$f_g$={fg_value/100}',
                        alpha=0.9,
                        linewidth=1.5)

        time_vector_dpt = np.linspace(0, sum(sim_time_sec), len(dp_transient))
        axs[1].plot(tdp, dp/1e6, color='k', label='FOSSIL')
        axs[1].plot(time_vector_dpt, dp_transient, linewidth=3, color=colors[-1], label='OpenDARTS')
        axs[1].set_xlabel('t [s]')
        axs[1].set_ylabel('$\Delta p$ [MPa]')
        axs[1].grid(alpha=0.4)
        # axs[1].legend()

        # CWP x time
        # Load experimental production
        PV = 26.68
        for fg_value, group in lasurf_data.groupby('fg'):
            axs[2].plot(group['Time'], group['Production']/PV, 
                    color=fg_colors[fg_value],
                    label=f'$f_g$={fg_value/100}',
                    alpha=0.4)

        axs[2].plot(tp, p, color='k', label='FOSSIL')
        axs[2].plot(time_sec, Q_w, linewidth=3, color=colors[-1], label='OpenDARTS')
        axs[2].set_xlabel('t [s]')
        axs[2].set_ylabel('Q [PV]')
        axs[2].legend(loc='best')
        axs[2].grid(alpha=0.3)
        axs[2].legend(ncols=3, frameon=True)

        plt.tight_layout()
        plt.savefig('DP_fg_validation.png', dpi=300)
        plt.savefig('DP_fg_validation.pdf', dpi=300)
        plt.show()

        # save QoIs
        np.savetxt(os.path.join(n.output_folder, "DP_fg.txt"), np.column_stack((fgs, DP)), header='fg DP_Pa_s', comments='')
        np.savetxt(os.path.join(n.output_folder, "DP_t.txt"), np.column_stack((time_vector_dpt, dp_transient)), header='t DP_Pa_s', comments='')
        np.savetxt(os.path.join(n.output_folder, "Q_t.txt"), np.column_stack((time_sec, Q_w)), header='t Q_w_PV', comments='')

        # save well time data        
        time_data_df.to_pickle(os.path.join(n.output_folder, "well_time_data.pkl"))  # as a pickle file
        writer = pd.ExcelWriter(os.path.join(n.output_folder, "well_time_data.xlsx"))  # as an excel file
        time_data_df.to_excel(writer, sheet_name='Sheet1', index=False)
        writer.close()

    else:
        # n.load_restart_data()
        n.load_restart_data('output/solution.h5')
        time_data = pd.read_pickle("darts_time_data.pkl")


    if True:
        Xn = np.array(n.physics.engine.X, copy=False)
        nc = n.physics.nc + n.physics.thermal
        nb = n.reservoir.mesh.n_res_blocks

        fig, ax = plt.subplots(1,2, figsize=(7,5))
        for i in range(nc if nc < 3 else 3):
            ax[i].plot(np.linspace(0,0.15,n.reservoir.nx), Xn[i:nb*nc:nc])
            if i == 0:
                # pressure
                p = Xn[i:nb*nc:nc]
                DP = p[0]-p[-1] # [bar]
                print("DP = ", DP*1e5)
        
        ax[0].set_xlabel('$x$ [m]')
        ax[0].set_ylabel('$p$ [bar]')
        # ax[0].set_ylim([40, 41])
        ax[1].set_xlabel('$x$ [m]')
        ax[1].set_ylabel('$S_w$ [-]')
        # ax[1].set_ylim([0,1])
        
        plt.tight_layout()
        plt.savefig('out.png', dpi=300)
        plt.show()

    else:
        #plot_sol(n)
        n.print_and_plot('sim_data')
