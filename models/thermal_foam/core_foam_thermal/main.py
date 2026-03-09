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

def load_baseline(path):
    dp_stationary, fgs = np.loadtxt(os.path.join(path, "DP_fg.txt"), unpack=True, skiprows=1)
    time_dp, dp = np.loadtxt(os.path.join(path, "DP_t.txt"), unpack=True, skiprows=1)
    time_sec, Q_w = np.loadtxt(os.path.join(path, "Q_t.txt"), unpack=True, skiprows=1)
    return time_dp, dp, time_sec, Q_w, dp_stationary, fgs

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


    fgs = [0.5, 0.6, 0.7, 0.75, 0.8, 0.9]
    sim_time_sec = np.array([10830, 10521, 7238, 8019, 10512, 5095]) # [s]

    sim_time = sim_time_sec / (60*60*24) # to days
    print(sim_time)
    save_interval=1e-3

    t_hot = 80+273.15
    t_iso = 60+273.15
    t_cold = 20+273.15

    redirect_darts_output('run.log')

    # RUN HOT INJECTION SIMULATION
    n_hot = Model(t_inj=t_hot)
    n_hot.init()
    n_hot.set_output()

    DP_hot = np.zeros_like(fgs)
    dp_transient_hot = []

    if run_simulation:
        for i in range(len(fgs)):
            n_hot.set_fg_Inj(fg=fgs[i])
            for j in range(int(sim_time[i]/save_interval)):
                n_hot.run(save_interval)

                # Get pressure drop
                Xn = np.array(n_hot.physics.engine.X, copy=False)
                nc = n_hot.physics.nc + n_hot.physics.thermal
                nb = n_hot.reservoir.mesh.n_res_blocks
                for k in range(nc if nc < 3 else 3):
                    if k == 0:
                        p = Xn[k:nb*nc:nc]
                        dpn = (p[2]-p[-1]) / 10
                        dp_transient_hot.append(dpn) # 1 bar = 0.1 MPa
                DP_hot[i] = dpn

        n_hot.print_timers()
        n_hot.print_stat()

        # compute well time data
        time_data_dict_hot = n_hot.output.store_well_time_data()

        # POST-PROCESSING
        # Get PV
        volume = np.array(n_hot.reservoir.mesh.volume, copy=False)
        volume = volume[:n_hot.reservoir.nx] # 6 additional volumes at the end are 2 monitoring cells for each reservoir/perforation
        poro = np.array(n_hot.reservoir.mesh.poro, copy=False)
        poro = poro[:n_hot.reservoir.nx]
        volume_poro = sum(volume * poro)
        print("Volume = " + str(sum(volume[:n_hot.reservoir.nx])))
        print("Pore volume = " + str(sum(volume * poro)))

        # Get simulation time vector
        time_data_df_hot = pd.DataFrame.from_dict(time_data_dict_hot)
        time_hot = time_data_df_hot['time'].to_numpy()
        dt = time_hot[1:] - time_hot[:-1]
        time_sec_hot = time_hot[1:]*(24*60*60)

        # Get Cumulative water production
        q_w_hot = np.abs(time_data_df_hot['well_P1_volumetric_rate_Aq_at_wh'].to_numpy()) # TODO: The outputs are saved along timeSteps.. how to save p, Sw, etc also every M*ts?
        q_w_hot = q_w_hot[1:]
        print(q_w_hot)
        Q_w_hot = np.zeros_like(dt)
        for i in range(Q_w_hot.shape[0]):
            Q_w_hot[i] = np.sum(np.multiply(dt[:i], q_w_hot[:i]))
        Q_w_hot /= volume_poro


    # RUN ISO INJECTION SIMULATION
    n_iso = Model(t_inj=t_iso)
    n_iso.init()
    n_iso.set_output()

    DP_iso = np.zeros_like(fgs)
    dp_transient_iso = []

    if run_simulation:
        for i in range(len(fgs)):
            n_iso.set_fg_Inj(fg=fgs[i])
            for j in range(int(sim_time[i]/save_interval)):
                n_iso.run(save_interval)

                # Get pressure drop
                Xn = np.array(n_iso.physics.engine.X, copy=False)
                nc = n_iso.physics.nc + n_iso.physics.thermal
                nb = n_iso.reservoir.mesh.n_res_blocks
                for k in range(nc if nc < 3 else 3):
                    if k == 0:
                        p = Xn[k:nb*nc:nc]
                        dpn = (p[2]-p[-1]) / 10
                        dp_transient_iso.append(dpn) # 1 bar = 0.1 MPa
                DP_iso[i] = dpn

        n_iso.print_timers()
        n_iso.print_stat()

        # compute well time data
        time_data_dict_iso = n_iso.output.store_well_time_data()

        # POST-PROCESSING
        # Get PV
        volume = np.array(n_iso.reservoir.mesh.volume, copy=False)
        volume = volume[:n_iso.reservoir.nx] # 6 additional volumes at the end are 2 monitoring cells for each reservoir/perforation
        poro = np.array(n_iso.reservoir.mesh.poro, copy=False)
        poro = poro[:n_iso.reservoir.nx]
        volume_poro = sum(volume * poro)
        print("Volume = " + str(sum(volume[:n_iso.reservoir.nx])))
        print("Pore volume = " + str(sum(volume * poro)))

        # Get simulation time vector
        time_data_df_iso = pd.DataFrame.from_dict(time_data_dict_iso)
        time_iso = time_data_df_iso['time'].to_numpy()
        dt = time_iso[1:] - time_iso[:-1]
        time_sec_iso = time_iso[1:]*(24*60*60)

        # Get Cumulative water production
        q_w_iso = np.abs(time_data_df_iso['well_P1_volumetric_rate_Aq_at_wh'].to_numpy()) # TODO: The outputs are saved along timeSteps.. how to save p, Sw, etc also every M*ts?
        q_w_iso = q_w_iso[1:]
        print(q_w_iso)
        Q_w_iso = np.zeros_like(dt)
        for i in range(Q_w_iso.shape[0]):
            Q_w_iso[i] = np.sum(np.multiply(dt[:i], q_w_iso[:i]))
        Q_w_iso /= volume_poro

    # RUN COLD INJECTION SIMULATION
    n_cold = Model(t_inj=t_cold)
    n_cold.init()
    n_cold.set_output()

    DP_cold = np.zeros_like(fgs)
    dp_transient_cold = []

    if run_simulation:
        for i in range(len(fgs)):
            n_cold.set_fg_Inj(fg=fgs[i])
            for j in range(int(sim_time[i]/save_interval)):
                n_cold.run(save_interval)

                # Get pressure drop
                Xn = np.array(n_cold.physics.engine.X, copy=False)
                nc = n_cold.physics.nc + n_cold.physics.thermal
                nb = n_cold.reservoir.mesh.n_res_blocks
                for k in range(nc if nc < 3 else 3):
                    if k == 0:
                        p = Xn[k:nb*nc:nc]
                        dpn = (p[2]-p[-1]) / 10
                        dp_transient_cold.append(dpn) # 1 bar = 0.1 MPa
                DP_cold[i] = dpn

        n_cold.print_timers()
        n_cold.print_stat()

        # compute well time data
        time_data_dict_cold = n_cold.output.store_well_time_data()

        # POST-PROCESSING
        # Get PV
        volume = np.array(n_cold.reservoir.mesh.volume, copy=False)
        volume = volume[:n_cold.reservoir.nx] # 6 additional volumes at the end are 2 monitoring cells for each reservoir/perforation
        poro = np.array(n_cold.reservoir.mesh.poro, copy=False)
        poro = poro[:n_cold.reservoir.nx]
        volume_poro = sum(volume * poro)
        print("Volume = " + str(sum(volume[:n_cold.reservoir.nx])))
        print("Pore volume = " + str(sum(volume * poro)))

        # Get simulation time vector
        time_data_df_cold = pd.DataFrame.from_dict(time_data_dict_cold)
        time_cold = time_data_df_cold['time'].to_numpy()
        dt = time_cold[1:] - time_cold[:-1]
        time_sec_cold = time_cold[1:]*(24*60*60)

        # Get Cumulative water production
        q_w_cold = np.abs(time_data_df_cold['well_P1_volumetric_rate_Aq_at_wh'].to_numpy()) # TODO: The outputs are saved along timeSteps.. how to save p, Sw, etc also every M*ts?
        q_w_cold = q_w_cold[1:]
        print(q_w_cold)
        Q_w_cold = np.zeros_like(dt)
        for i in range(Q_w_cold.shape[0]):
            Q_w_cold[i] = np.sum(np.multiply(dt[:i], q_w_cold[:i]))
        Q_w_cold /= volume_poro


        # Load isothermal baseline data
        time_dp_ref, dp_ref, time_sec_ref, Qw_ref, dp_stationary, fgs = load_baseline(r'/home/lamap/source/DARTS_LAMAP/core_16/output')

        # PLOTS
        fig, axs = plt.subplots(2, 1, figsize=(15,12))

        gradient = LinearSegmentedColormap.from_list("grad", ["#264653", "#e9c46a"])
        palet = [gradient(i) for i in np.linspace(0, 1, 6)]
        palet_hex = [f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}" for r, g, b, _ in palet]
        fg_colors = {50: palet_hex[0], 60: palet_hex[1], 70: palet_hex[2], 75: palet_hex[3], 80: palet_hex[4], 90: palet_hex[5]}

        # dp x time
        time_vector_dpt_hot = np.linspace(0, sum(sim_time_sec), len(dp_transient_hot))
        time_vector_dpt_iso = np.linspace(0, sum(sim_time_sec), len(dp_transient_iso))
        time_vector_dpt_cold = np.linspace(0, sum(sim_time_sec), len(dp_transient_cold))
        # axs[0].plot(time_dp_ref, dp_ref, color='k', label='Isothermal DARTS')
        axs[0].plot(time_vector_dpt_iso, dp_transient_iso, color='k', linewidth=2, label='OpenDARTS iso. inj.')
        axs[0].plot(time_vector_dpt_hot, dp_transient_hot, color=colors[4], linewidth=2, label='OpenDARTS hot inj.')
        axs[0].plot(time_vector_dpt_cold, dp_transient_cold, color=colors[-1], linestyle='--', linewidth=2, label='OpenDARTS cold inj.')
        axs[0].set_xlabel('Time [s]')
        axs[0].set_ylabel('$\Delta p$ [MPa]')
        axs[0].grid()

        # CWP x time
        # axs[1].plot(time_sec_ref, Qw_ref, color='k', label='Isothermal DARTS')
        axs[1].plot(time_sec_iso, Q_w_iso, color='k', linewidth=2, label='OpenDARTS iso. inj.')
        axs[1].plot(time_sec_hot, Q_w_hot, color=colors[4], linewidth=2, label='OpenDARTS hot inj.')
        axs[1].plot(time_sec_cold, Q_w_cold, color=colors[-1], linewidth=2, linestyle='--', label='OpenDARTS cold inj.')
        axs[1].set_xlabel('Time [s]')
        axs[1].set_ylabel('Cumulative wat. prod. [PV]')
        axs[1].legend(loc='best')
        axs[1].grid()
        axs[1].legend(frameon=True)

        # plt.title('QoIs - Only Fdry - no coupled flow-temp effects')
        plt.tight_layout()
        plt.savefig('DP_fg_validation.png', dpi=300)
        plt.show()

        # save well time data        
        time_data_df_hot.to_pickle(os.path.join(n_hot.output_folder, "well_time_data.pkl"))  # as a pickle file
        writer = pd.ExcelWriter(os.path.join(n_hot.output_folder, "well_time_data.xlsx"))  # as an excel file
        time_data_df_hot.to_excel(writer, sheet_name='Sheet1', index=False)
        writer.close()
