import os

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import pandas as pd


def tersurf(
    a,
    b,
    c,
    d,
    components,
    ini_comp=None,
    inj_comp=None,
    title=None,
    label=None,
    line1=None,
    inf_p=None,
):
    """
    :param a: z1
    :param b: z2
    :param c: z3
    :param d: values you want to plot ( e.g. operator values, derivative, hessian,...)
    :param line: in case want to draw trajectory on it
    :param inf_p: inflection point in a given trajectory
    :return:
    """

    # j = 1
    # a = Zr[:,0::3][:,j]
    # b = Zr[:,1::3][:,j]
    # c = Zr[:,2::3][:,j]
    # d = BETA[:, 0::3][:,j]
    # line = np.vstack([Zr[:,0::3][:,j], Zr[:,1::3][:,j]]).T
    # title = 'hello'

    # transfer matrix
    z = np.array([[0, 0], [1, 0], [0, 1], [0, 0]])
    np.transpose([[1 / 2, 1], [np.sqrt(3) / 2, 0]])

    # plot triangle
    # p = np.matmul(z, mt)
    plt.figure(dpi=100)
    # plt.title(title, fontsize = 20, color = 'red')
    # plt.plot(p[:, 0], p[:, 1], '-k', 'linewidth', 1.5)

    x = 0.5 - a * np.cos(np.pi / 3) + b / 2
    y = 0.866 - a * np.sin(np.pi / 3) - b / np.tan(np.pi / 6) / 2
    c = plt.scatter(x, y, c=d, s=20)
    plt.colorbar(c, location='left', label=title, shrink=0.5, aspect=10, pad=-0.05)

    x = 0.5 - z[:, 0] * np.cos(np.pi / 3) + z[:, 1] / 2
    y = 0.866 - z[:, 0] * np.sin(np.pi / 3) - z[:, 1] / np.tan(np.pi / 6) / 2
    plt.plot(x, y, '--k', linewidth=0.5)

    # create the grid
    corners = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) * 0.5]])
    triangle = tri.Triangulation(corners[:, 0], corners[:, 1])

    # creating the grid
    refiner = tri.UniformTriRefiner(triangle)
    trimesh = refiner.refine_triangulation(subdiv=3)

    # plotting the mesh

    plt.ylim([0, 1])
    plt.axis('off')  # translate the data to cords

    if ini_comp is not None:
        a = ini_comp[0]
        b = ini_comp[1]
        x = 0.5 - a * np.cos(np.pi / 3) + b / 2
        y = 0.866 - a * np.sin(np.pi / 3) - b / np.tan(np.pi / 6) / 2
        plt.scatter(x, y, color='k', marker='o', s=25, label='initial')

    if inj_comp is not None:
        a = inj_comp[0]
        b = inj_comp[1]
        x = 0.5 - a * np.cos(np.pi / 3) + b / 2
        y = 0.866 - a * np.sin(np.pi / 3) - b / np.tan(np.pi / 6) / 2
        plt.scatter(x, y, color='k', marker='x', s=25, label='injection')

    plt.triplot(trimesh, color='black', linestyle='--', linewidth=0.8)

    # create a triangulation out of these points
    # T = tri.Triangulation(x, y)    # plot the contour
    # vmin = min(d) # - 0.1   #-10.1
    # vmax = max(d) # + 0.1   #10.1
    # level = np.linspace(vmin, vmax, 101)
    # plt.tricontourf(x, y, T.triangles, d, alpha = 1, cmap='jet', levels = level)
    # plt.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3) / 2, 0], linewidth=1)

    # plot line
    if line1 is not None:
        x = 0.5 - line1[:, 0] * np.cos(np.pi / 3) + line1[:, 1] / 2
        y = (
            0.866
            - line1[:, 0] * np.sin(np.pi / 3)
            - line1[:, 1] / np.tan(np.pi / 6) / 2
        )
        plt.plot(x, y)  # , s = 20)
        # plt.colorbar(c)

    # x = 0.5   - line2[:,0] * np.cos(np.pi/3) + line2[:,1] / 2
    # y = 0.866 - line2[:,0] * np.sin(np.pi/3) - line2[:,1] / np.tan(np.pi / 6) / 2
    # plt.scatter(x, y, s = 5)

    # x = 0.5   - line3[:,0] * np.cos(np.pi/3) + line3[:,1] / 2
    # y = 0.866 - line3[:,0] * np.sin(np.pi/3) - line3[:,1] / np.tan(np.pi / 6) / 2
    # plt.scatter(x, y, s = 5)
    # plt.legend()

    # plt.rc('font', size=10)
    # cax = plt.axes([0.75, 0.55, 0.055, 0.3])
    # plt.colorbar(cax=cax, format='%.3f', label='')

    plt.gcf().text(0.18, 0.1, label[0], fontsize=12, color='black')
    plt.gcf().text(0.5, 0.8, label[1], fontsize=12, color='black')
    plt.gcf().text(0.81, 0.1, label[2], fontsize=12, color='black')

    # if line is not None:
    #     # in case, want to draw random trajectories on the ternary diagrm
    #     # line = line[:,1:]
    #     # traj = np.matmul(line, mt)
    #     # plt.plot(traj[:,0], traj[:,1], '--')
    #     x = 0.5   - line[:,0] * np.cos(np.pi/3) + line[:,1] / 2
    #     y = 0.866 - line[:,0] * np.sin(np.pi/3) - line[:,1] / np.tan(np.pi / 6) / 2
    #     plt.plot(x,y,'-r')
    #     # plt.show()

    # if inf_p is not None:
    #     #inf_p = np.matmul(inf_p, mt)
    #     #plt.scatter(inf_p[:, 0], inf_p[:, 1])
    #     # translate the data to cords
    #     x = 0.5 - inf_p[:,0] * np.cos(np.pi / 3) + inf_p[:,1] / 2
    #     y = 0.866 - inf_p[:,0] * np.sin(np.pi / 3) - inf_p[:,1] / np.tan(np.pi / 6) / 2
    #     plt.scatter(x, y)
    #     plt.show()

    return c


def lump_components(property_array, Mw, components, indices=None):
    if indices is None:
        heavy_Mw_indices = [i for i, mw in enumerate(Mw) if mw > 100.0]
        intermediate_Mw_indices = [
            i for i, mw in enumerate(Mw) if (mw > 18.0 and mw < 100.0)
        ]
        light_Mw_indices = [i for i, mw in enumerate(Mw) if mw <= 18]
    else:
        heavy_Mw_indices = indices[2]
        intermediate_Mw_indices = indices[1]
        light_Mw_indices = indices[0]

    property_array[components[-1]] = np.ones(next(iter(property_array.values())).shape)
    for i in components[:-1]:
        property_array[components[-1]] -= property_array[i]

    z1 = sum(property_array[components[i]] for i in light_Mw_indices)
    z2 = sum(property_array[components[i]] for i in intermediate_Mw_indices)
    z3 = sum(property_array[components[i]] for i in heavy_Mw_indices)

    return z1, z2, z3, (light_Mw_indices, intermediate_Mw_indices, heavy_Mw_indices)


def process_well(m):
    """
    Processes and analyzes well performance data from a reservoir simulation model.

    Parameters:
    m : DartsModel()

    Returns:
    time_data_df : pandas.DataFrame
        A DataFrame containing time-resolved well data including PV, cumulative production,
        recovery factor, GOR, and void replacement ratio (VVR).
    """

    # compute well rates
    time_data_dict = m.output.store_well_time_data(
        ["phases_volumetric_rates", "phases_mass_rates"], save_output_files=False
    )
    time_data_df = pd.DataFrame(time_data_dict)  # cast as a dataframe
    # m.output.plot_well_time_data(types_of_well_rates=["phases_volumetric_rates", "phases_mass_rates"])

    GRV = np.array(m.reservoir.mesh.volume)[: m.reservoir.nx]  # gross rock volume
    poro = np.array(m.reservoir.mesh.poro)[: m.reservoir.nx]
    Bo = 1.6  # oil formation volume factor, typical range 1.1 - 2.0
    E = 370  # gas expansion factor
    Bg = 1 / E  # gas formation volume factor, typical range 0.002 - 0.005

    time = time_data_df['time']
    gas_rate = time_data_df['well_P1_volumetric_rate_gas_at_wh'] * (1 / Bg)
    oil_rate = time_data_df['well_P1_volumetric_rate_oil_at_wh'] * (1 / Bo)
    try:
        wat_rate = time_data_df['well_P1_volumetric_rate_wat_at_wh']
    except:
        wat_rate = oil_rate - oil_rate

    # cumulative oil injection at rc
    Ni = np.cumsum(
        time_data_df['well_I1_volumetric_rate_oil_at_wh'] * np.diff(time, prepend=0.0)
    )

    # cumulative gas injection at rc
    Gi = np.cumsum(
        time_data_df['well_I1_volumetric_rate_gas_at_wh'] * np.diff(time, prepend=0.0)
    )

    try:
        # cumulative water injection at rc
        Wi = np.cumsum(
            time_data_df['well_I1_volumetric_rate_wat_at_wh']
            * np.diff(time, prepend=0.0)
        )
    except:
        Wi = Ni - Ni

    # express volumes of oil and gas injected in PVs
    time_data_df['Ni'], time_data_df['Gi'], time_data_df['Wi'] = Ni, Gi, Wi
    PV = (Ni + Gi + Wi) / sum(GRV * poro)
    time_data_df['PV'] = PV

    # ---- HC production
    plt.figure()
    Np = -np.cumsum(
        oil_rate * np.diff(time, prepend=0.0)
    )  # cumulative oil production at sc
    Gp = -np.cumsum(
        gas_rate * np.diff(time, prepend=0.0)
    )  # cumulative gas production at sc
    try:
        Wp = -np.cumsum(wat_rate * np.diff(time, prepend=0.0))
    except:
        Wp = Np - Np

    time_data_df['Np'], time_data_df['Gp'], time_data_df['Wp'] = Np, Gp, Wp

    fig, ax1 = plt.subplots()
    ax1.plot(PV, Np, '-x', label='Oil', color='tab:red')
    if 'wat' in m.phases:
        ax1.plot(PV, Wp, '-x', label='Water', color='tab:blue')
        ax1.set_ylabel("Np (sm³) [red], Wp (sm³) [blue]")
    else:
        ax1.set_ylabel('Np (sm³)', color='tab:red')
    ax1.set_xlabel('PV')
    ax2 = ax1.twinx()
    ax2.plot(PV, Gp, '-x', label='Gas', color='tab:green')
    ax2.set_ylabel('Gp (sm³)', color='tab:green')
    ax2.tick_params(axis='y', labelcolor='tab:green')
    # fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
    plt.title("Oil and Gas Production")
    plt.grid(True)
    plt.tight_layout(pad=2)
    plt.savefig(os.path.join(m.output_folder, 'productions.png'))
    plt.show()

    # ---- WATER CUT
    plt.figure()
    plt.grid()
    WC = wat_rate / (gas_rate + oil_rate + wat_rate)
    time_data_df['WC'] = WC
    plt.plot(PV, WC, '-o')
    plt.xlabel('PV')
    plt.ylabel('WC')
    plt.title('Water cut')
    plt.tight_layout(pad=2)
    plt.savefig(os.path.join(m.output_folder, 'water_cut.png'))
    plt.close()

    # ---- GAS OIL RATIO
    plt.figure()
    plt.grid()
    m3_to_scf, m3_to_bbl = 35.31467, 6.289811
    Rp = gas_rate / oil_rate  # producing GOR at sc, m3/m3
    # Rp = Gp/Np
    time_data_df['GOR [m3/m3]'] = Rp
    time_data_df['GOR [scf/bbl]'] = Rp * (m3_to_scf / m3_to_bbl)
    plt.plot(PV, Rp * (m3_to_scf / m3_to_bbl), '-o')
    plt.xlabel('PV')
    plt.ylabel('GOR (scf/bbl)')
    plt.title('Producing gas/oil ratio at sc')
    plt.tight_layout(pad=2)
    plt.savefig(os.path.join(m.output_folder, 'GOR.png'))
    plt.close()

    # ---- VOIDAGE REPLACEMENT RATIO
    # solution gas-oil ratio (the amount of gas dissolved in the oil)
    # Rs = 0

    # Rs = (Vgas / Voil) # solution gas-oil ratio (scf/stb)
    # Vgas = gas volume at STP (scf)
    # Voil = oil volume at STP (stb)

    # VRR = (Wi + Ni + Gi) / (Bo * Np + Bg * (Rp-Rs) * Np + Wp) # material balance
    VRR = (Wi + Ni + Gi) / (
        Bo * Np + Bg * Gp + Wp
    )  # material balance, calculated at reservoir conditions

    # VRR = Gi / (Bo +  Gp) # material balance
    time_data_df['VRR'] = VRR

    # target_rate = time_data_df['well_I1_volumetric_rate_gas_at_wh'].iloc[-1] / 1 * (Bo + Bg * Rp.iloc[-1])
    time_data_df.to_pickle(os.path.join(m.output_folder, "well_time_data.pkl"))

    return time_data_df


def compute_component_rf(m, comp_list, time_data_df):
    """
    compute recovery factor per components based on the mass initially in place per componets
    """
    prop_list = m.output.properties

    timestamps, property_array = m.output.output_properties(
        output_properties=prop_list, timestep=0, engine=False
    )
    mass_components, mass_vapor, mass_oil, mass_water = m.get_mass_components(
        property_array
    )

    if 0:
        RF = {comp: [] for comp in m.components}
        FIIP = {
            comp: np.sum(values) for comp, values in mass_components.items()
        }  # mass initially in place per component

        for ts in range(m.Nt + 1):
            try:  # in case one calls this function before m.Nt has been reached in the simulation
                timestamps, property_array = m.output.output_properties(
                    output_properties=prop_list, timestep=ts, engine=False
                )

                mass_components, mass_vapor, mass_oil, mass_water = (
                    m.get_mass_components(property_array)
                )
                FIP = {comp: np.sum(values) for comp, values in mass_components.items()}

                for comp in m.components:
                    rf_value = (FIP[comp] - FIIP[comp]) / FIIP[comp]
                    RF[comp].append(-rf_value)

            except:
                pass

        for comp in m.components:
            RF[comp] = np.array(RF[comp])

        timesteps = np.arange(len(RF[comp])) * 365

    else:
        RF = {}
        FIIP = {
            comp: np.sum(values) for comp, values in mass_components.items()
        }  # moles initially in place per component
        time_data_dict = m.output.store_well_time_data(
            ["components_mass_rates"], save_output_files=False
        )
        timesteps = time_data_dict['time']

        for comp in m.components:
            Cp = np.cumsum(
                -time_data_dict[f"well_P1_mass_rate_{comp}_by_sum_perfs"]
                * np.diff(timesteps, prepend=0.0)
            )
            rf_value = 1 - (FIIP[comp] - Cp) / FIIP[comp]
            RF[comp] = rf_value

    try:
        # ---- plot recovery factors per component
        if m.specs['rate'] is not None:
            PV = (m.rate * timesteps) / sum(
                np.array(m.reservoir.mesh.volume) * np.array(m.reservoir.mesh.poro)
            )
            RF['PV'] = PV
        else:
            PV = time_data_df['PV']
            RF['PV'] = PV
        # plot results

        plt.figure()
        # for comp, rf_values in RF.items():
        for comp in comp_list:
            plt.plot(PV, RF[comp], '-x', label=comp)
        plt.xlabel("PV")
        plt.ylabel("Recovery Factor")
        plt.title("Component-wise Recovery Factor")
        plt.grid(True)
        plt.legend(title="Component", loc='best')
        # plt.xlim(0, 1)
        plt.savefig(
            os.path.join(m.output_folder, "RF_components.png"),
            dpi=300,
            bbox_inches='tight',
        )
        plt.show()

        # save recovery factor data
        RF_df = pd.DataFrame(RF)
        RF_df.to_pickle(os.path.join(m.output_folder, "RF.pkl"))

    except:
        pass

    return 0


def plot_phase_rate_darts(
    well_name, darts_df, ph, style='-', color='#00A6D6', ax=None, alpha=1
):
    search_str = well_name + ' : ' + ph + ' rate'
    ax = darts_df.plot(
        x='time',
        y=[col for col in darts_df.columns if search_str in col],
        style=style,
        color=color,
        ax=ax,
        alpha=alpha,
    )
    return ax


def plot_bhp_darts(well_name, darts_df, style='-', color='#00A6D6', ax=None):
    search_str = well_name + ' : BHP'
    ax = darts_df.plot(
        x='time',
        y=[col for col in darts_df.columns if search_str in col],
        style=style,
        color=color,
        ax=ax,
    )
    return ax


def plot_oil_rate_darts(
    well_name, darts_df, style='-', color='#00A6D6', ax=None, alpha=1
):
    search_str = well_name + ' : oil rate'
    ax = darts_df.plot(
        x='time',
        y=[col for col in darts_df.columns if search_str in col],
        style=style,
        color=color,
        ax=ax,
        alpha=alpha,
    )
    return ax


def plot_oil_rate_darts_2(
    well_name, darts_df, style='-', color='#00A6D6', ax=None, alpha=1
):
    search_str = well_name + ' : oil rate'
    darts_df['time'] = -darts_df['time']
    ax = darts_df.plot(
        x='time',
        y=[col for col in darts_df.columns if search_str in col],
        style=style,
        color=color,
        ax=ax,
        alpha=alpha,
    )
    return ax


def plot_gas_rate_darts(well_name, darts_df, style='-', color='#00A6D6', ax=None):
    search_str = well_name + ' : gas rate'
    ax = darts_df.plot(
        x='time',
        y=[col for col in darts_df.columns if search_str in col],
        style=style,
        color=color,
        ax=ax,
    )
    return ax


def plot_water_rate_darts(
    well_name, darts_df, style='-', color='#00A6D6', ax=None, alpha=1
):
    search_str = well_name + ' : water rate'
    ax = darts_df.plot(
        x='time',
        y=[col for col in darts_df.columns if search_str in col],
        style=style,
        color=color,
        ax=ax,
        alpha=alpha,
    )

    ymin, ymax = ax.get_ylim()
    if ymax < 0:
        ax.set_ylim(ymin * 1.1, 0)

    if ymin > 0:
        ax.set_ylim(0, ymax * 1.1)

    return ax


def plot_watercut_darts(
    well_name, darts_df, style='-', color='#00A6D6', ax=None, alpha=1, label=''
):
    wat = well_name + ' : water rate (m3/day)'
    oil = well_name + ' : oil rate (m3/day)'
    wcut = well_name + ' watercut'
    darts_df[wcut] = darts_df[wat] / (darts_df[wat] + darts_df[oil])

    if label == '':
        label = wcut
    ax = darts_df.plot(
        x='time', y=wcut, style=style, color=color, ax=ax, alpha=alpha, label=label
    )

    ax.set_ylim(0, 1)

    return ax


def plot_water_rate_darts_2(
    well_name, darts_df, style='-', color='#00A6D6', ax=None, alpha=1
):
    search_str = well_name + ' : water rate'
    darts_df['time'] = -darts_df['time']
    ax = darts_df.plot(
        x='time',
        y=[col for col in darts_df.columns if search_str in col],
        style=style,
        color=color,
        ax=ax,
        alpha=alpha,
    )

    ymin, ymax = ax.get_ylim()
    if ymax < 0:
        ax.set_ylim(ymin * 1.1, 0)

    if ymin > 0:
        ax.set_ylim(0, ymax * 1.1)

    return ax


def plot_water_rate_vs_obsrate(
    well_name, darts_df, truth_df, style='-', color='#00A6D6', ax=None, marker="o"
):
    search_str = well_name + ' : water rate'
    darts_df = darts_df.set_index('time', drop=False)
    ax.scatter(
        x=abs(
            darts_df[[col for col in truth_df.columns if search_str in col]].loc[
                truth_df.time, :
            ]
        ),
        y=abs(truth_df[[col for col in truth_df.columns if search_str in col]]),
        marker=marker,
    )
    max_rate = max(
        max(abs(darts_df[search_str + ' (m3/day)'].values)),
        max(abs(truth_df[search_str + ' (m3/day)'].values)),
    )
    min_rate = min(
        min(abs(darts_df[search_str + ' (m3/day)'].values)),
        min(abs(truth_df[search_str + ' (m3/day)'].values)),
    )
    plt.plot([max_rate, min_rate], [max_rate, min_rate], color=color, marker='x')
    plt.ylabel('Truth Data')
    plt.xlabel('Simulation Data')
    ymin, ymax = ax.get_ylim()
    if ymax < 0:
        ax.set_ylim(ymin * 1.1, 0)

    if ymin > 0:
        ax.set_ylim(0, ymax * 1.1)
    plt.grid(True)

    return ax


def plot_water_rate_vs_obsrate_time(
    well_name,
    darts_df,
    truth_df,
    style='-',
    color='#00A6D6',
    ax=None,
    marker="o",
    time=0,
):
    search_str = well_name + ' : water rate'
    darts_df = darts_df.set_index('time', drop=False)
    truth_df = truth_df.set_index('time', drop=False)
    ax.scatter(
        x=abs(
            darts_df[[col for col in truth_df.columns if search_str in col]].loc[
                truth_df.time[time], :
            ]
        ),
        y=abs(
            truth_df[[col for col in truth_df.columns if search_str in col]].loc[
                truth_df.time[time], :
            ]
        ),
        marker=marker,
        label=(well_name + ' time: ' + time.__str__()),
    )
    ax.legend(loc=5)
    plt.ylabel('Truth Data')
    plt.xlabel('Simulation Data')
    ymin, ymax = ax.get_ylim()
    if ymax < 0:
        ax.set_ylim(ymin * 1.1, 0)

    if ymin > 0:
        ax.set_ylim(0, ymax * 1.1)
    plt.grid(True)

    return ax


def plot_oil_rate_rate_vs_obsrate(
    well_name, darts_df, truth_df, style='-', color='#00A6D6', ax=None, marker="o"
):
    search_str = well_name + ' : oil rate'
    darts_df = darts_df.set_index('time', drop=False)
    ax.scatter(
        x=abs(
            darts_df[[col for col in truth_df.columns if search_str in col]].loc[
                truth_df.time, :
            ]
        ),
        y=abs(truth_df[[col for col in truth_df.columns if search_str in col]]),
        marker=marker,
    )
    min(abs(darts_df[search_str + ' (m3/day)'].values))
    min_oil_rate_truth = min(abs(truth_df[search_str + ' (m3/day)'].values))
    max(abs(darts_df[search_str + ' (m3/day)'].values))
    max_oil_rate_truth = max(abs(truth_df[search_str + ' (m3/day)'].values))
    plt.plot(
        [max_oil_rate_truth, min_oil_rate_truth],
        [max_oil_rate_truth, min_oil_rate_truth],
        color=color,
        marker='x',
    )
    plt.ylabel('Truth Data')
    plt.xlabel('Simulation Data')

    ymin, ymax = ax.get_ylim()
    if ymax < 0:
        ax.set_ylim(ymin * 1.1, 0)

    if ymin > 0:
        ax.set_ylim(0, ymax * 1.1)
    plt.grid(True)

    return ax


def plot_total_inj_water_rate_darts(
    darts_df, style='-', color='#00A6D6', ax=None, alpha=1
):
    acc_df = pd.DataFrame()
    acc_df['time'] = darts_df['time']
    acc_df['total'] = 0
    search_str = ' : water rate'
    for col in darts_df.columns:
        if search_str in col:
            # if sum(darts_df[col]) > 0:
            if 'I' in col:
                #     acc_df['total'] += darts_df[col]
                for i in range(0, len(darts_df[col])):
                    if darts_df[col][i] >= 0:
                        # acc_df['total'][i] += darts_df[col][i]
                        acc_df.loc[i, 'total'] += darts_df[col][i]

    ax = acc_df.plot(x='time', y='total', style=style, color=color, ax=ax, alpha=alpha)

    ymin, ymax = ax.get_ylim()
    if ymax < 0:
        ax.set_ylim(ymin * 1.1, 0)

    if ymin > 0:
        ax.set_ylim(0, ymax * 1.1)

    return ax


def plot_total_inj_gas_rate_darts(
    darts_df, style='-', color='#00A6D6', ax=None, alpha=1
):
    acc_df = pd.DataFrame()
    acc_df['time'] = darts_df['time']
    acc_df['total'] = 0
    search_str = ' : gas rate'
    for col in darts_df.columns:
        if search_str in col:
            # if sum(darts_df[col]) > 0:
            if 'I' in col:
                #     acc_df['total'] += darts_df[col]
                for i in range(0, len(darts_df[col])):
                    if darts_df[col][i] >= 0:
                        # acc_df['total'][i] += darts_df[col][i]
                        acc_df.loc[i, 'total'] += darts_df[col][i]

    ax = acc_df.plot(x='time', y='total', style=style, color=color, ax=ax, alpha=alpha)

    ymin, ymax = ax.get_ylim()
    if ymax < 0:
        ax.set_ylim(ymin * 1.1, 0)

    if ymin > 0:
        ax.set_ylim(0, ymax * 1.1)

    return ax


def plot_water_rate_prediction(darts_df, style='-', color='#00A6D6', ax=None, alpha=1):
    acc_df = pd.DataFrame()
    acc_df['time'] = darts_df['time']
    acc_df['total'] = 0
    search_str = ' : water rate'
    for col in darts_df.columns:
        if search_str in col:
            if 'I' not in col:
                acc_df['total'] += darts_df[col]

    acc_df['total'] = acc_df['total'].abs()
    ax = acc_df.plot(x='time', y='total', style=style, color=color, ax=ax, alpha=alpha)

    ymin, ymax = ax.get_ylim()
    if ymax < 0:
        ax.set_ylim(ymin * 1.1, 0)

    if ymin > 0:
        ax.set_ylim(0, ymax * 1.1)

    return ax


def plot_total_prod_water_rate_darts(
    darts_df, style='-', color='#00A6D6', ax=None, alpha=1
):
    acc_df = pd.DataFrame()
    acc_df['time'] = darts_df['time']
    acc_df['total'] = 0
    search_str = ' : water rate'
    for col in darts_df.columns:
        if search_str in col:
            # if sum(darts_df[col]) < 0:
            if 'I' not in col:
                acc_df['total'] += darts_df[col]

    acc_df['total'] = acc_df['total'].abs()
    ax = acc_df.plot(x='time', y='total', style=style, color=color, ax=ax, alpha=alpha)

    ymin, ymax = ax.get_ylim()
    if ymax < 0:
        ax.set_ylim(ymin * 1.1, 0)

    if ymin > 0:
        ax.set_ylim(0, ymax * 1.1)

    return ax


def plot_acc_prod_water_rate_darts(
    darts_df, style='-', color='#00A6D6', ax=None, alpha=1
):
    acc_df = pd.DataFrame()
    acc_df['time'] = darts_df['time']
    acc_df['total'] = 0
    search_str = 'water  acc'
    for col in darts_df.columns:
        if search_str in col:
            # if sum(darts_df[col]) < 0:
            if 'I' not in col:
                acc_df['total'] += darts_df[col]

    acc_df['total'] = acc_df['total'].abs()
    ax = acc_df.plot(x='time', y='total', style=style, color=color, ax=ax, alpha=alpha)

    ymin, ymax = ax.get_ylim()
    if ymax < 0:
        ax.set_ylim(ymin * 1.1, 0)

    if ymin > 0:
        ax.set_ylim(0, ymax * 1.1)

    return ax


def plot_acc_prod_oil_rate_darts(
    darts_df, style='-', color='#00A6D6', ax=None, alpha=1
):
    acc_df = pd.DataFrame()
    acc_df['time'] = darts_df['time']
    acc_df['total'] = 0
    search_str = 'oil  acc'
    for col in darts_df.columns:
        if search_str in col:
            # if sum(darts_df[col]) < 0:
            if 'I' not in col:
                acc_df['total'] += darts_df[col]

    acc_df['total'] = acc_df['total'].abs()
    ax = acc_df.plot(x='time', y='total', style=style, color=color, ax=ax, alpha=alpha)

    ymin, ymax = ax.get_ylim()
    if ymax < 0:
        ax.set_ylim(ymin * 1.1, 0)

    if ymin > 0:
        ax.set_ylim(0, ymax * 1.1)

    return ax


def plot_total_prod_oil_rate_darts(
    darts_df, style='-', color='#00A6D6', ax=None, alpha=1
):
    acc_df = pd.DataFrame()
    acc_df['time'] = darts_df['time']
    acc_df['total'] = 0
    search_str = ' : oil rate'
    for col in darts_df.columns:
        if search_str in col:
            # if sum(darts_df[col]) < 0:
            if 'I' not in col:
                acc_df['total'] += darts_df[col]

    acc_df['total'] = acc_df['total'].abs()
    ax = acc_df.plot(x='time', y='total', style=style, color=color, ax=ax, alpha=alpha)

    ymin, ymax = ax.get_ylim()
    if ymax < 0:
        ax.set_ylim(ymin * 1.1, 0)

    if ymin > 0:
        ax.set_ylim(0, ymax * 1.1)

    return ax


def plot_total_prod_gas_rate_darts(
    darts_df, style='-', color='#00A6D6', ax=None, alpha=1
):
    acc_df = pd.DataFrame()
    acc_df['time'] = darts_df['time']
    acc_df['total'] = 0
    search_str = ' : gas rate'
    for col in darts_df.columns:
        if search_str in col:
            # if sum(darts_df[col]) < 0:
            if 'I' not in col:
                acc_df['total'] += darts_df[col]

    acc_df['total'] = acc_df['total'].abs()
    ax = acc_df.plot(x='time', y='total', style=style, color=color, ax=ax, alpha=alpha)

    ymin, ymax = ax.get_ylim()
    if ymax < 0:
        ax.set_ylim(ymin * 1.1, 0)

    if ymin > 0:
        ax.set_ylim(0, ymax * 1.1)

    return ax


def plot_temp_darts(well_name, darts_df, style='-', color='#00A6D6', ax=None):
    search_str = well_name + ' : temperature'
    ax = darts_df.plot(
        x='time',
        y=[col for col in darts_df.columns if search_str in col],
        style=style,
        color=color,
        ax=ax,
    )

    return ax


def plot_extracted_energy_darts(darts_df, style='-', color='#00A6D6', ax=None):
    search_str = ' : energy'
    y = [
        col for col in darts_df.columns if search_str in col
    ]  # get columns with 'energy' data
    y = darts_df[y].sum(axis=1)  # sum over the wells
    t = darts_df['time']
    dt = np.append(0, np.ediff1d(t))  # add the first time step
    col_name = 'energy extracted, PJ'
    darts_df[col_name] = -(y * dt).cumsum() * 1e-12  # kJ/day -> PJ
    ax = darts_df.plot(x='time', y=col_name, style=style, color=color, ax=ax)
    return ax


# def tersurf(a, b, c, d, line=None, inf_p=None):
# import matplotlib.tri as tri

# """
# :param a: z1
# :param b: z2
# :param c: z3
# :param d: values you want to plot ( e.g. operator values, derivative, hessian,...)
# :param line: in case want to draw trajectory on it
# :param inf_p: inflection point in a given trajectory
# :return:
# """
# z = np.array([[0, 0], [1, 0], [0, 1], [0, 0]])
# # transfer matrix
# # mt = np.transpose([[1 / 2, 1], [np.sqrt(3) / 2, 0]])
# # plot triangle
# # p = np.matmul(z, mt)
# # plt.figure(figsize=(10, 8), dpi=100)
# # plt.plot(p[:, 0], p[:, 1], 'k', 'linewidth', 1.5)
# x = 0.5 - z[:, 0] * np.cos(np.pi / 3) + z[:, 1] / 2
# y = 0.866 - z[:, 0] * np.sin(np.pi / 3) - z[:, 1] / np.tan(np.pi / 6) / 2
# plt.plot(x, y, 'k', 'linewidth', 1.5)
# # create the grid
# corners = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) * 0.5]])
# triangle = tri.Triangulation(corners[:, 0], corners[:, 1])
# # creating the grid
# refiner = tri.UniformTriRefiner(triangle)
# trimesh = refiner.refine_triangulation(subdiv=3)

# # plotting the mesh
# plt.triplot(trimesh, color='navajowhite', linestyle='--', linewidth=0.8)
# plt.ylim([0, 1])
# plt.axis('off')

# # translate the data to cords
# x = 0.5 - a * np.cos(np.pi / 3) + b / 2
# y = 0.866 - a * np.sin(np.pi / 3) - b / np.tan(np.pi / 6) / 2

# # create a triangulation out of these points
# T = tri.Triangulation(x, y)
# # plot the contour
# vmin = min(d) - 0.1  # -10.1
# vmax = max(d) + 0.1  # 10.1
# level = np.linspace(vmin, vmax, 101)
# plt.tricontourf(x, y, T.triangles, d, cmap='jet', levels=level)
# plt.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3) / 2, 0], linewidth=1)
# plt.rc('font', size=12)
# cax = plt.axes([0.75, 0.55, 0.055, 0.3])
# plt.colorbar(cax=cax, format='%.3f', label='')
# # plt.gcf().text(0.08, 0.1, '$C_1$', fontsize=20, color='black')
# # plt.gcf().text(0.91, 0.1, '$CO_2$', fontsize=20, color='black')
# # plt.gcf().text(0.5, 0.8, '$H_2O$', fontsize=20, color='black')
# if line is not None:
# # in case, want to draw random trajectories on the ternary diagrm
# line = line[:, 1:]
# # traj = np.matmul(line, mt)
# # plt.plot(traj[:,0], traj[:,1], '--')
# x = 0.5 - line[:, 0] * np.cos(np.pi / 3) + line[:, 1] / 2
# y = 0.866 - line[:, 0] * np.sin(np.pi / 3) - line[:, 1] / np.tan(np.pi / 6) / 2
# plt.plot(x, y, '--')
# if inf_p is not None:
# # inf_p = np.matmul(inf_p, mt)
# # plt.scatter(inf_p[:, 0], inf_p[:, 1])
# # translate the data to cords
# x = 0.5 - inf_p[:, 0] * np.cos(np.pi / 3) + inf_p[:, 1] / 2
# y = (
# 0.866
# - inf_p[:, 0] * np.sin(np.pi / 3)
# - inf_p[:, 1] / np.tan(np.pi / 6) / 2
# )
# plt.scatter(x, y)
