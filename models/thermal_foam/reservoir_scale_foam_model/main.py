import numpy as np
import pandas as pd
import os

from model import Model
from darts.engines import value_vector, redirect_darts_output
import matplotlib.pyplot as plt
from darts.physics.base.operators_base import PropertyOperators as props
from matplotlib import cm

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

    redirect_darts_output('run.log')

    t_final = 75000
    n_snapshots = 150

    n = Model()
    # n.params.linear_type = n.params.linear_solver_t.cpu_superlu
    n.init()
    n.set_output()
    time_data_filename = n.output_folder + "/darts_time_data.pkl"

    s_list = []
    p_list = []
    t_list = []

    if True:

        for j in range(n_snapshots):
            n.run(int(t_final/n_snapshots))
            # n.reservoir.wells[0].control = n.physics.new_bhp_inj(100, 3*[n.zero])
            # n.run_python(300, restart_dt=1e-3)
            n.print_timers()
            n.print_stat()


            nx = n.reservoir.nx
            x = np.linspace(0, nx, nx)
            time_vector, property_array = n.output.output_properties(engine=True)
            plt.rcParams['pcolor.shading'] ='nearest'
            # print_props = [0, 1, 3]

            fig, axs = plt.subplots(len(property_array.keys()), 1, figsize=(6, 6), dpi=100, facecolor='w', edgecolor='k')
            # for i, ith_prop in enumerate(property_array.keys()):
            #     # plot array defined in active cells only
            #     arr = property_array[ith_prop]
            #     axs[i].plot(x, arr.reshape(nx))
            #     axs[i].set_title(f'{ith_prop} @ t={time_vector[0]}days')

            arr = property_array['pressure']
            axs[0].plot(x, arr.reshape(nx))
            axs[0].set_title(f'pressure @ t={time_vector[0]}days')
            p_list.append(arr)

            arr = property_array['H2O']
            axs[1].plot(x, arr.reshape(nx))
            axs[1].set_title(f'sw @ t={time_vector[0]}days')
            s_list.append(arr)

            arr = property_array['temperature']
            axs[2].plot(x, arr.reshape(nx))
            axs[2].set_title(f'temperature @ t={time_vector[0]}days')
            t_list.append(arr)


            axs[1].set_ylim([0,1.1])
            axs[0].set_ylabel('P [bar]')
            axs[1].set_ylabel('$S_w$ [-]')
            axs[2].set_ylabel('T [K]')
            axs[1].set_xlabel('x [m]')
            plt.tight_layout()
            plt.savefig(os.path.join(n.output_folder, f"out_{j}.png"))
            # plt.show()

    else:
        # n.load_restart_data()
        n.load_restart_data('output/solution.h5')
        time_data = pd.read_pickle(time_data_filename)


    # POST-PROCESSING

    p_matrix = np.vstack(p_list)
    s_matrix = np.vstack(s_list)
    t_matrix = np.vstack(t_list)   # shape (nt, nx)
    df_p = pd.DataFrame(p_matrix)
    df_s = pd.DataFrame(s_matrix)
    df_t = pd.DataFrame(t_matrix)

    df_p.to_hdf("results_p.h5", key="B", mode="a")
    df_s.to_hdf("results_s.h5", key="B", mode="a")
    df_t.to_hdf("results_t.h5", key="B", mode="a")

    # compute well time data
    time_data_dict = n.output.store_well_time_data()

    # save well time data
    time_data_df = pd.DataFrame.from_dict(time_data_dict)
    time_data_df.to_pickle(time_data_filename)  # as a pickle file
    writer = pd.ExcelWriter(os.path.join(n.output_folder, "well_time_data.xlsx"))  # as an excel file
    time_data_df.to_excel(writer, sheet_name='Sheet1', index=False)
    writer.close()

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
    q_w = np.abs(time_data_df['well_P1_volumetric_rate_Aq_at_wh'].to_numpy()) # TODO: The outputs are saved along timeSteps.. how to save p, Sw, etc also every M*ts?
    q_w = q_w[1:]
    Q_w = np.zeros_like(dt)
    for i in range(Q_w.shape[0]):
        Q_w[i] = np.sum(np.multiply(dt[:i],q_w[:i]))
    Q_w /= volume_poro

    q_matrix = np.vstack((time[1:], Q_w))   # shape (nt, nx)
    dq_p = pd.DataFrame(q_matrix)
    dq_p.to_hdf("results_q.h5", key="B", mode="a")

    plt.figure()
    plt.plot(time[1:], Q_w)
    plt.xlabel('Time [days]')
    plt.ylabel('Cumulative water production [PV]')
    plt.savefig(os.path.join(n.output_folder, "cumulative_water_production.png"))

