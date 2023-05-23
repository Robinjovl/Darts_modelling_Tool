import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_phase_rate_darts(well_name, darts_df, ph, style='-', color='#00A6D6', ax=None, alpha=1):
    search_str = well_name + ' : ' + ph + ' rate'
    ax = darts_df.plot(x='time', y=[col for col in darts_df.columns if search_str in col], style=style, color=color,
                       ax=ax, alpha=alpha)

    plt.show(block=False)
    return ax

def plot_bhp_darts(well_name, darts_df, style='-', color='#00A6D6', ax=None):
    search_str = well_name + ' : BHP'
    ax = darts_df.plot(x='time', y=[col for col in darts_df.columns if search_str in col], style=style, color=color,
                       ax=ax)

    plt.show(block=False)
    return ax


def plot_oil_rate_darts(well_name, darts_df, style='-', color='#00A6D6', ax=None, alpha=1):
    search_str = well_name + ' : oil rate'
    ax = darts_df.plot(x='time', y=[col for col in darts_df.columns if search_str in col], style=style, color=color,
                       ax=ax, alpha=alpha)

    plt.show(block=False)
    return ax

def plot_oil_rate_darts_2(well_name, darts_df, style='-', color='#00A6D6', ax=None, alpha=1):
    search_str = well_name + ' : oil rate'
    darts_df['time']  = -darts_df['time']
    ax = darts_df.plot(x='time', y=[col for col in darts_df.columns if search_str in col], style=style, color=color,
                       ax=ax, alpha=alpha)

    plt.show(block=False)
    return ax

def plot_gas_rate_darts(well_name, darts_df, style='-', color='#00A6D6', ax=None):
    search_str = well_name + ' : gas rate'
    ax = darts_df.plot(x='time', y=[col for col in darts_df.columns if search_str in col], style=style, color=color,
                       ax=ax)

    plt.show(block=False)
    return ax


def plot_water_rate_darts(well_name, darts_df, style='-', color='#00A6D6', ax=None, alpha=1):
    search_str = well_name + ' : water rate'
    ax = darts_df.plot(x='time', y=[col for col in darts_df.columns if search_str in col], style=style, color=color,
                       ax=ax, alpha=alpha)

    ymin, ymax = ax.get_ylim()
    if ymax < 0:
        ax.set_ylim(ymin * 1.1, 0)

    if ymin > 0:
        ax.set_ylim(0, ymax * 1.1)

    plt.show(block=False)
    return ax


def plot_watercut_darts(well_name, darts_df, style='-', color='#00A6D6', ax=None, alpha=1, label=''):
    wat = well_name + ' : water rate (m3/day)'
    oil = well_name + ' : oil rate (m3/day)'
    wcut = well_name + ' watercut'
    darts_df[wcut] = darts_df[wat] / (darts_df[wat] + darts_df[oil])

    if label == '':
        label = wcut
    ax = darts_df.plot(x='time', y=wcut, style=style, color=color,
                       ax=ax, alpha=alpha, label=label)

    ax.set_ylim(0, 1)


    plt.show(block=False)
    return ax

def plot_water_rate_darts_2(well_name, darts_df, style='-', color='#00A6D6', ax=None, alpha=1):
    search_str = well_name + ' : water rate'
    darts_df['time']  = -darts_df['time']
    ax = darts_df.plot(x='time', y=[col for col in darts_df.columns if search_str in col], style=style, color=color,
                       ax=ax, alpha=alpha)

    ymin, ymax = ax.get_ylim()
    if ymax < 0:
        ax.set_ylim(ymin * 1.1, 0)

    if ymin > 0:
        ax.set_ylim(0, ymax * 1.1)

    plt.show(block=False)
    return ax

def plot_water_rate_vs_obsrate(well_name, darts_df, truth_df,  style='-', color='#00A6D6', ax=None, marker="o"):
    search_str = well_name + ' : water rate'
    darts_df = darts_df.set_index('time', drop=False)
    ax.scatter(x=abs(darts_df[[col for col in truth_df.columns if search_str in col]].loc[truth_df.time, :]),
               y=abs(truth_df[[col for col in truth_df.columns if search_str in col]]), marker=marker)
    max_rate = max(max(abs(darts_df[search_str + ' (m3/day)'].values)), max(abs(truth_df[search_str + ' (m3/day)'].values)))
    min_rate = min(min(abs(darts_df[search_str + ' (m3/day)'].values)), min(abs(truth_df[search_str + ' (m3/day)'].values)))
    plt.plot( [max_rate, min_rate], [max_rate, min_rate], color=color, marker='x')
    plt.ylabel('Truth Data')
    plt.xlabel('Simulation Data')
    ymin, ymax = ax.get_ylim()
    if ymax < 0:
        ax.set_ylim(ymin * 1.1, 0)

    if ymin > 0:
        ax.set_ylim(0, ymax * 1.1)
    plt.grid(True)
    plt.show(block=False)
    return ax

def plot_water_rate_vs_obsrate_time(well_name, darts_df, truth_df,  style='-', color='#00A6D6', ax=None, marker="o", time=0):
    search_str = well_name + ' : water rate'
    darts_df = darts_df.set_index('time', drop=False)
    truth_df = truth_df.set_index('time', drop=False)
    ax.scatter(x=abs(darts_df[[col for col in truth_df.columns if search_str in col]].loc[truth_df.time[time], :]),
               y=abs(truth_df[[col for col in truth_df.columns if search_str in col]].loc[truth_df.time[time], :]), marker=marker, label=(well_name+' time: '+ time.__str__()))
    ax.legend(loc=5)
    plt.ylabel('Truth Data')
    plt.xlabel('Simulation Data')
    ymin, ymax = ax.get_ylim()
    if ymax < 0:
        ax.set_ylim(ymin * 1.1, 0)

    if ymin > 0:
        ax.set_ylim(0, ymax * 1.1)
    plt.grid(True)
    plt.show(block=False)
    return ax

def plot_oil_rate_rate_vs_obsrate(well_name, darts_df, truth_df,  style='-', color='#00A6D6', ax=None, marker="o"):
    search_str = well_name + ' : oil rate'
    darts_df = darts_df.set_index('time', drop=False)
    ax.scatter(x=abs(darts_df[[col for col in truth_df.columns if search_str in col]].loc[truth_df.time, :]),
               y=abs(truth_df[[col for col in truth_df.columns if search_str in col]]), marker=marker)
    min_oil_rate_sim = min(abs(darts_df[search_str + ' (m3/day)'].values))
    min_oil_rate_truth = min(abs(truth_df[search_str + ' (m3/day)'].values))
    max_oil_rate_sim = max(abs(darts_df[search_str + ' (m3/day)'].values))
    max_oil_rate_truth = max(abs(truth_df[search_str + ' (m3/day)'].values))
    max_rate = max(max_oil_rate_sim, max_oil_rate_truth)
    min_rate = min(min_oil_rate_sim, min_oil_rate_truth)
    plt.plot( [max_oil_rate_truth, min_oil_rate_truth], [max_oil_rate_truth, min_oil_rate_truth], color=color, marker='x')
    plt.ylabel('Truth Data')
    plt.xlabel('Simulation Data')

    ymin, ymax = ax.get_ylim()
    if ymax < 0:
        ax.set_ylim(ymin * 1.1, 0)

    if ymin > 0:
        ax.set_ylim(0, ymax * 1.1)
    plt.grid(True)
    plt.show(block=False)
    return ax

def plot_total_inj_water_rate_darts(darts_df, style='-', color='#00A6D6', ax=None, alpha=1):
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

    ax = acc_df.plot(x='time', y='total', style=style, color=color,
                       ax=ax, alpha=alpha)

    ymin, ymax = ax.get_ylim()
    if ymax < 0:
        ax.set_ylim(ymin * 1.1, 0)

    if ymin > 0:
        ax.set_ylim(0, ymax * 1.1)

    plt.show(block=False)
    return ax

def plot_total_inj_gas_rate_darts(darts_df, style='-', color='#00A6D6', ax=None, alpha=1):
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

    ax = acc_df.plot(x='time', y='total', style=style, color=color,
                       ax=ax, alpha=alpha)

    ymin, ymax = ax.get_ylim()
    if ymax < 0:
        ax.set_ylim(ymin * 1.1, 0)

    if ymin > 0:
        ax.set_ylim(0, ymax * 1.1)

    plt.show(block=False)
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
    ax = acc_df.plot(x='time', y='total', style=style, color=color,
                       ax=ax, alpha=alpha)

    ymin, ymax = ax.get_ylim()
    if ymax < 0:
        ax.set_ylim(ymin * 1.1, 0)

    if ymin > 0:
        ax.set_ylim(0, ymax * 1.1)

    plt.show(block=False)
    return ax


def plot_total_prod_water_rate_darts(darts_df, style='-', color='#00A6D6', ax=None, alpha=1):
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
    ax = acc_df.plot(x='time', y='total', style=style, color=color,
                       ax=ax, alpha=alpha)

    ymin, ymax = ax.get_ylim()
    if ymax < 0:
        ax.set_ylim(ymin * 1.1, 0)

    if ymin > 0:
        ax.set_ylim(0, ymax * 1.1)

    plt.show(block=False)
    return ax

def plot_acc_prod_water_rate_darts(darts_df, style='-', color='#00A6D6', ax=None, alpha=1):
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
    ax = acc_df.plot(x='time', y='total', style=style, color=color,
                       ax=ax, alpha=alpha)

    ymin, ymax = ax.get_ylim()
    if ymax < 0:
        ax.set_ylim(ymin * 1.1, 0)

    if ymin > 0:
        ax.set_ylim(0, ymax * 1.1)

    plt.show(block=False)
    return ax

def plot_acc_prod_oil_rate_darts(darts_df, style='-', color='#00A6D6', ax=None, alpha=1):
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
    ax = acc_df.plot(x='time', y='total', style=style, color=color,
                       ax=ax, alpha=alpha)

    ymin, ymax = ax.get_ylim()
    if ymax < 0:
        ax.set_ylim(ymin * 1.1, 0)

    if ymin > 0:
        ax.set_ylim(0, ymax * 1.1)

    plt.show(block=False)
    return ax

def plot_total_prod_oil_rate_darts(darts_df, style='-', color='#00A6D6', ax=None, alpha=1):
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
    ax = acc_df.plot(x='time', y='total', style=style, color=color,
                       ax=ax, alpha=alpha)

    ymin, ymax = ax.get_ylim()
    if ymax < 0:
        ax.set_ylim(ymin * 1.1, 0)

    if ymin > 0:
        ax.set_ylim(0, ymax * 1.1)

    plt.show(block=False)
    return ax

def plot_total_prod_gas_rate_darts(darts_df, style='-', color='#00A6D6', ax=None, alpha=1):
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
    ax = acc_df.plot(x='time', y='total', style=style, color=color,
                       ax=ax, alpha=alpha)

    ymin, ymax = ax.get_ylim()
    if ymax < 0:
        ax.set_ylim(ymin * 1.1, 0)

    if ymin > 0:
        ax.set_ylim(0, ymax * 1.1)

    plt.show(block=False)
    return ax

def plot_temp_darts(well_name, darts_df, style='-', color='#00A6D6', ax=None):
    search_str = well_name + ' : temperature'
    ax = darts_df.plot(x='time', y=[col for col in darts_df.columns if search_str in col], style=style, color=color,
                       ax=ax)

    plt.show(block=False)
    return ax

def tersurf(a, b, c, d, line = None, inf_p = None):
    import matplotlib.tri as tri
    """
    :param a: z1
    :param b: z2
    :param c: z3
    :param d: values you want to plot ( e.g. operator values, derivative, hessian,...)
    :param line: in case want to draw trajectory on it
    :param inf_p: inflection point in a given trajectory
    :return: 
    """
    z = np.array([[0, 0], [1, 0], [0, 1], [0, 0]])
    # transfer matrix
    mt = np.transpose([[1 / 2, 1], [np.sqrt(3) / 2, 0]])
    # plot triangle
    #p = np.matmul(z, mt)
    # plt.figure(figsize=(10, 8), dpi=100)
    #plt.plot(p[:, 0], p[:, 1], 'k', 'linewidth', 1.5)
    x = 0.5 - z[:,0] * np.cos(np.pi / 3) + z[:,1] / 2
    y = 0.866 - z[:,0] * np.sin(np.pi / 3) - z[:,1] / np.tan(np.pi / 6) / 2
    plt.plot(x, y, 'k', 'linewidth', 1.5)
    # create the grid
    corners = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) * 0.5]])
    triangle = tri.Triangulation(corners[:, 0], corners[:, 1])
    # creating the grid
    refiner = tri.UniformTriRefiner(triangle)
    trimesh = refiner.refine_triangulation(subdiv=3)

    # plotting the mesh
    plt.triplot(trimesh, color='navajowhite', linestyle='--', linewidth=0.8)
    plt.ylim([0, 1])
    plt.axis('off')

    # translate the data to cords
    x = 0.5 - a * np.cos(np.pi / 3) + b / 2
    y = 0.866 - a * np.sin(np.pi / 3) - b / np.tan(np.pi / 6) / 2

    # create a triangulation out of these points
    T = tri.Triangulation(x, y)
    # plot the contour
    vmin = min(d) - 0.1   #-10.1
    vmax = max(d) + 0.1   #10.1
    level = np.linspace(vmin, vmax, 101)
    plt.tricontourf(x, y, T.triangles, d, cmap='jet', levels = level)
    plt.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3) / 2, 0], linewidth=1)
    plt.rc('font', size=12)
    cax = plt.axes([0.75, 0.55, 0.055, 0.3])
    plt.colorbar(cax=cax, format='%.3f', label='')
    # plt.gcf().text(0.08, 0.1, '$C_1$', fontsize=20, color='black')
    # plt.gcf().text(0.91, 0.1, '$CO_2$', fontsize=20, color='black')
    # plt.gcf().text(0.5, 0.8, '$H_2O$', fontsize=20, color='black')
    if line is not None:
        # in case, want to draw random trajectories on the ternary diagrm
        line = line[:,1:]
        #traj = np.matmul(line, mt)
        #plt.plot(traj[:,0], traj[:,1], '--')
        x = 0.5 - line[:, 0] * np.cos(np.pi / 3) + line[:, 1] / 2
        y = 0.866 - line[:,0] * np.sin(np.pi / 3) - line[:,1] / np.tan(np.pi / 6) / 2
        plt.plot(x,y, '--')
        if inf_p is not None:
            #inf_p = np.matmul(inf_p, mt)
            #plt.scatter(inf_p[:, 0], inf_p[:, 1])
            # translate the data to cords
            x = 0.5 - inf_p[:,0] * np.cos(np.pi / 3) + inf_p[:,1] / 2
            y = 0.866 - inf_p[:,0] * np.sin(np.pi / 3) - inf_p[:,1] / np.tan(np.pi / 6) / 2
            plt.scatter(x, y)
    plt.show()