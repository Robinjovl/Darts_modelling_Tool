import numpy as np
import pandas as pd
import sys
import os
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from matplotlib import rcParams
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from model import Model
from darts.engines import redirect_darts_output
from darts.physics.base.operators_base import PropertyOperators as props
from darts.tools.hdf5_tools import load_hdf5_to_dict

rcParams["text.usetex"]=False
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.rc('legend',fontsize=14)

def find_ffmpeg():
    common_paths = [
        '/usr/bin/ffmpeg', '/usr/local/bin/ffmpeg',  # Unix-like systems
        'C:\\ffmpeg\\bin\\ffmpeg.exe',               # Windows
        '/opt/local/bin/ffmpeg',                     # Other potential locations
        'C:\\work\\packages\\ffmpeg-6.0\\bin\\ffmpeg.exe'
    ]
    for path in common_paths:
        if os.path.exists(path):
            return path
    return os.getenv('FFMPEG_PATH', 'ffmpeg')  # Fallback to hoping it's in the PATH

rcParams['animation.ffmpeg_path'] = find_ffmpeg()

def animate_solution_1d_separate_plots(paths, n_cells, labels, lower_lims, upper_lims, video_fname='plot.mp4'):
    data0 = load_hdf5_to_dict(filename=paths[0] + 'solution.h5')['dynamic']
    n_cells_max = max(n_cells)
    c = [np.arange(n_cells_max, step=int(n_cells_max / n_cells[i])) for i in range(len(n_cells))]
    n_plots = len(data0['variable_names'])
    n_steps = data0['time'].size # number of saved snapshots
    colors = ['b', 'r', 'g', 'm', 'c', 'y', 'k']
    fig, ax = plt.subplots(nrows=n_plots, sharex=True, figsize=(7, 20))
    for i in range(n_plots):
        if data0['variable_names'][i] == 'pressure':
            ax[i].set_ylabel('pressure, bar', fontsize=14)
        else:
            ax[i].set_ylabel(data0['variable_names'][i], fontsize=14)
    ax[n_plots - 1].set_xlabel('x, m', fontsize=16)

    lines = []
    i = 0
    for k, path in enumerate(paths):
        filename = path + 'solution.h5'
        data = load_hdf5_to_dict(filename=filename)['dynamic']

        for i in range(n_plots):
            li, = ax[i].plot(c[k], data['X'][0,:n_cells[k],i], linewidth=1, color=colors[k], linestyle='-', label=labels[k])
            lines.append(li)

    time_text = ax[0].text(0.4, 0.93, 'time = ' + str(round(data0['time'][0], 4)) + ' days',
                           fontsize=16, rotation='horizontal', transform=fig.transFigure)

    for i in range(n_plots):
        ax[i].set_ylim(lower_lims[i], upper_lims[i])

    ax[0].legend(loc='upper right', prop={'size': 14})
    nt = n_steps # num of updates

    def animate(i):
        time_to_update = False
        for k, path in enumerate(paths):
            each_ith = int(n_steps / nt)
            ind = int(n_steps * i / n_steps)
            if ind % each_ith == 0:
                data = load_hdf5_to_dict(filename=path + 'solution.h5')['dynamic']
                for j in range(n_plots):
                    lines[n_plots * k + j].set_data(c[k], data['X'][i,:n_cells[k],j])
                time_text.set_text('time = ' + str(round(data0['time'][i], 4)) + ' days')

        return lines


    anim = FuncAnimation(fig, animate, interval=100, repeat=True, frames=np.arange(1, n_steps))

    # writer = animation.PillowWriter(fps=20,
    #                                 metadata=dict(artist='Me'),
    #                                 bitrate=1800)
    # anim.save(paths[0] + 'comparison.gif', writer=writer)

    fig.tight_layout()
    writervideo = animation.FFMpegWriter(fps=1)
    anim.save(paths[0] + video_fname, writer=writervideo)
    plt.close(fig)

def animate_solution_1d_single_plot(paths, n_cells, labels, lower_lim, upper_lim, video_fname='plot.mp4'):
    data0 = load_hdf5_to_dict(filename=paths[0] + 'solution.h5')['dynamic']
    n_cells_max = max(n_cells)
    c = [np.arange(n_cells_max, step=int(n_cells_max / n_cells[i])) for i in range(len(n_cells))]
    n_plots = len(data0['variable_names'])
    n_steps = data0['time'].size  # number of saved snapshots
    colors = ['b', 'r', 'g', 'm', 'c', 'y', 'k']
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.set_xlabel('x, m', fontsize=14)
    ax.set_ylabel('Composition', fontsize=14)
    ax.set_ylim(lower_lim, upper_lim)

    lines = []
    for k, path in enumerate(paths):
        for i in range(1, n_plots):
            li, = ax.semilogy(c[k], data0['X'][0, :n_cells[k], i],
                          linewidth=1, color=colors[i % len(colors)],
                          linestyle='-', label=f"{data0['variable_names'][i]}")
            lines.append(li)

    ax.legend(loc='upper right', fontsize=10)
    time_text = ax.text(0.02, 0.95, f"time = {round(data0['time'][0], 4)} days",
                        fontsize=12, transform=ax.transAxes)

    nt = n_steps  # number of updates

    def animate(i):
        for k, path in enumerate(paths):
            each_ith = int(n_steps / nt)
            ind = int(n_steps * i / n_steps)
            if ind % each_ith == 0:
                data = load_hdf5_to_dict(filename=path + 'solution.h5')['dynamic']
                for j in range(n_plots - 1):
                    lines[n_plots * k + j].set_data(c[k], data['X'][i, :n_cells[k], j + 1])
                time_text.set_text(f"time = {round(data['time'][i], 4)} days")

        return lines + [time_text]

    anim = FuncAnimation(fig, animate, interval=100, repeat=True, frames=np.arange(1, n_steps))

    writervideo = animation.FFMpegWriter(fps=1)
    anim.save(paths[0] + video_fname, writer=writervideo)
    plt.close(fig)

def plot_comparison(params, path_prefix, pic_fname='comparison.png', L=1000, add_inset_figs=True):
    lw = 1.
    fs_legend = 12
    colors = ['b', 'r', 'g', 'm', 'c', 'k']
    fig, ax = plt.subplots(figsize=(10, 7))

    n_res = len(params['itor_type'])
    for i in range(n_res):
        model_path = get_output_folder( itor_type = params['itor_type'][i], itor_mode = params['itor_mode'][i],
                                        obl_points = params['obl_points'][i], n_comps = params['n_comps'][i],
                                        reservoir_type = params['reservoir_type'][i], nx = params['nx'][i],
                                        is_barycentric = params['barycentric'][i] )
        path = os.path.join(path_prefix, model_path, 'solution.h5')
        data = load_hdf5_to_dict(filename=path)['dynamic']
        components = get_components(n_comps=params['n_comps'][i])

        if params['reservoir_type'][i] == '1D':
            n_cells = params['nx'][i]
            ids = np.arange(n_cells)
            dx = L / params['nx'][i]
            c = np.linspace(dx / 2, L - dx / 2, params['nx'][i])
        elif params['reservoir_type'][i] == '2D':
            n_cells = params['nx'][i]
            ids = np.array([i + n_cells * i for i in range(n_cells)])
            dx = L / params['nx'][i]
            c = np.sqrt(2) * np.linspace(dx / 2, L - dx / 2, params['nx'][i])

        nc = params['n_comps'][i]
        for j in range(nc - 1):
            label = components[j] if i == 0 else None
            ax.plot(c, data['X'][-1, ids, j + 1], linewidth=lw, color=colors[j], linestyle=params['linestyles'][i], label=label)

        # last component
        last_component = 1.0 - np.sum(data['X'][-1, ids, 1:], axis=1)
        j = nc - 1
        label = components[j] if i == 0 else None
        ax.plot(c, last_component, linewidth=lw, color=colors[j], linestyle=params['linestyles'][i], label=label)

        # saturation
        n = Model(obl_points=params['obl_points'][i], components=get_components(n_comps),
                  reservoir_type=params['reservoir_type'][i], nx=params['nx'][i], itor_mode=params['itor_mode'][i],
                  itor_type=params['itor_type'][i], is_barycentric=params['barycentric'][i])
        sat = np.zeros(ids.size)
        for k, id in enumerate(ids):
            n.physics.property_containers[0].evaluate(data['X'][-1, id, :])
            sat[k] = n.physics.property_containers[0].output_props['sat0']()

        label = 'SatV' if i == 0 else None
        ax.plot(c, sat, linewidth=lw, color='orange', linestyle=params['linestyles'][i], label=label)

    ax.set_xlabel('x, m', fontsize=14)
    ax.set_ylabel('Composition / Vapour Saturation', fontsize=14)
    x_max = L if params['reservoir_type'][0] == '1D' else np.sqrt(2) * L
    ax.set_xlim(0.0, x_max)

    # Primary legend for descriptive variables (colors)
    handles, labels = ax.get_legend_handles_labels()
    primary_legend = ax.legend(handles=handles, labels=labels, loc='upper right', fontsize=fs_legend)

    # Custom legend for linestyles
    linestyle_labels = {
        '-': 'MLin, N=1024',
        '--': 'MLin, N=64',
        '-.': 'LinS, N=64',
        ':': 'LinD, N=64'
        # Add more linestyles and their descriptions if needed
    }
    custom_lines = [
        Line2D([0], [0], color='k', linestyle=ls, linewidth=lw, label=label)
        for ls, label in linestyle_labels.items()
    ]
    linestyle_legend = ax.legend(
        handles=custom_lines,
        title='Interpolations',
        loc='upper center',
        fontsize=fs_legend,
        title_fontsize=fs_legend
    )
    ax.add_artist(primary_legend)
    ax.add_artist(linestyle_legend)

    if add_inset_figs:
        if params['reservoir_type'][0] == '1D':
            # ----- Adding Inset Plot 1 -----
            # Define the region to zoom in (adjust these limits based on your data)
            x1, x2 = 880, 980   # x-axis limits for the inset
            y1, y2 = 0.34, 0.45   # y-axis limits for the inset

            # Create inset axes
            axins = inset_axes(ax, width="60%", height="60%", loc='lower left',
                               bbox_to_anchor=(0.55, 0.47, 0.4, 0.4),
                               bbox_transform=ax.transAxes)

            # Plot the same data on the inset axes
            for i in range(n_res):
                model_path = get_output_folder(
                    itor_type=params['itor_type'][i],
                    itor_mode=params['itor_mode'][i],
                    obl_points=params['obl_points'][i],
                    n_comps=params['n_comps'][i],
                    reservoir_type=params['reservoir_type'][i],
                    nx=params['nx'][i],
                    is_barycentric=params['barycentric'][i]
                )
                path = os.path.join(path_prefix, model_path, 'solution.h5')
                data = load_hdf5_to_dict(filename=path)['dynamic']
                components = get_components(n_comps=params['n_comps'][i])

                if params['reservoir_type'][i] == '1D':
                    n_cells = params['nx'][i]
                    dx = L / params['nx'][i]
                    c = np.linspace(dx / 2, L - dx / 2, params['nx'][i])
                nc = params['n_comps'][i]
                for j in range(nc - 1):
                    axins.plot(
                        c,
                        data['X'][-1, :n_cells, j + 1],
                        linewidth=lw,
                        color=colors[j],
                        linestyle=params['linestyles'][i],
                        label=None  # No labels in inset
                    )
                last_component = 1.0 - np.sum(data['X'][-1, :n_cells, 1:], axis=1)
                j = nc - 1
                axins.plot(
                    c,
                    last_component,
                    linewidth=lw,
                    color=colors[j],
                    linestyle=params['linestyles'][i],
                    label=None
                )

            # Set the limits for the inset
            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)

            # Remove ticks for clarity
            axins.set_xticks([])
            axins.set_yticks([])

            # Optional: Add a box around the zoomed area in the main plot
            ax.indicate_inset_zoom(axins, edgecolor="black")

            # Alternatively, use lines to connect the inset to the main plot
            mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.8")

            # ----- End of Inset Plot 1 -----

            # ----- Adding Inset Plot 2 -----
            # Define the region to zoom in (adjust these limits based on your data)
            x1, x2 = 5, 30   # x-axis limits for the inset
            y1, y2 = 0.90, 1.01   # y-axis limits for the inset

            # Create inset axes
            axins = inset_axes(ax, width="60%", height="60%", loc='lower left',
                               bbox_to_anchor=(0.02, 0.15, 0.25, 0.6),
                               bbox_transform=ax.transAxes)

            # Plot the same data on the inset axes
            for i in range(n_res):
                model_path = get_output_folder(
                    itor_type=params['itor_type'][i],
                    itor_mode=params['itor_mode'][i],
                    obl_points=params['obl_points'][i],
                    n_comps=params['n_comps'][i],
                    reservoir_type=params['reservoir_type'][i],
                    nx=params['nx'][i],
                    is_barycentric=params['barycentric'][i]
                )
                path = os.path.join(path_prefix, model_path, 'solution.h5')
                data = load_hdf5_to_dict(filename=path)['dynamic']
                components = get_components(n_comps=params['n_comps'][i])

                if params['reservoir_type'][i] == '1D':
                    n_cells = params['nx'][i]
                    dx = L / params['nx'][i]
                    c = np.linspace(dx / 2, L - dx / 2, params['nx'][i])
                nc = params['n_comps'][i]
                for j in range(nc - 1):
                    axins.plot(
                        c,
                        data['X'][-1, :n_cells, j + 1],
                        linewidth=lw,
                        color=colors[j],
                        linestyle=params['linestyles'][i],
                        label=None  # No labels in inset
                    )
                # last component
                last_component = 1.0 - np.sum(data['X'][-1, :n_cells, 1:], axis=1)
                j = nc - 1
                axins.plot(
                    c,
                    last_component,
                    linewidth=lw,
                    color=colors[j],
                    linestyle=params['linestyles'][i],
                    label=None
                )

                # saturation
                n = Model(obl_points=params['obl_points'][i], components=get_components(n_comps),
                          reservoir_type=params['reservoir_type'][i], nx=params['nx'][i], itor_mode=params['itor_mode'][i],
                          itor_type=params['itor_type'][i], is_barycentric=params['barycentric'][i])
                sat = np.zeros(n_cells)
                for k in range(n_cells):
                    n.physics.property_containers[0].evaluate(data['X'][-1, k, :])
                    sat[k] = n.physics.property_containers[0].output_props['sat0']()
                axins.plot(c, sat, linewidth=lw, color='orange', linestyle=params['linestyles'][i], label=None)

            # Set the limits for the inset
            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)

            # Remove ticks for clarity
            axins.set_xticks([])
            axins.set_yticks([])

            # Optional: Add a box around the zoomed area in the main plot
            ax.indicate_inset_zoom(axins, edgecolor="black")

            # Alternatively, use lines to connect the inset to the main plot
            mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.8")

            # ----- End of Inset Plot 2 -----
        elif params['reservoir_type'][0] == '2D':
            # ----- Adding Inset Plot 3 -----
            # Define the region to zoom in (adjust these limits based on your data)
            x1, x2 = 60, 120  # x-axis limits for the inset
            y1, y2 = 0.8, 1.01  # y-axis limits for the inset

            # Create inset axes
            axins = inset_axes(ax, width="60%", height="60%", loc='lower left',
                               bbox_to_anchor=(0.04, 0.15, 0.25, 0.65),
                               bbox_transform=ax.transAxes)

            # Plot the same data on the inset axes
            for i in range(n_res):
                model_path = get_output_folder(
                    itor_type=params['itor_type'][i],
                    itor_mode=params['itor_mode'][i],
                    obl_points=params['obl_points'][i],
                    n_comps=params['n_comps'][i],
                    reservoir_type=params['reservoir_type'][i],
                    nx=params['nx'][i],
                    is_barycentric=params['barycentric'][i]
                )
                path = os.path.join(path_prefix, model_path, 'solution.h5')
                data = load_hdf5_to_dict(filename=path)['dynamic']
                components = get_components(n_comps=params['n_comps'][i])

                if params['reservoir_type'][i] == '1D':
                    n_cells = params['nx'][i]
                    ids = np.arange(n_cells)
                    dx = L / params['nx'][i]
                    c = np.linspace(dx / 2, L - dx / 2, params['nx'][i])
                elif params['reservoir_type'][i] == '2D':
                    n_cells = params['nx'][i]
                    ids = np.array([i + n_cells * i for i in range(n_cells)])
                    dx = L / params['nx'][i]
                    c = np.sqrt(2) * np.linspace(dx / 2, L - dx / 2, params['nx'][i])
                nc = params['n_comps'][i]
                for j in range(nc - 1):
                    axins.plot(
                        c,
                        data['X'][-1, ids, j + 1],
                        linewidth=lw,
                        color=colors[j],
                        linestyle=params['linestyles'][i],
                        label=None  # No labels in inset
                    )
                # last component
                last_component = 1.0 - np.sum(data['X'][-1, ids, 1:], axis=1)
                j = nc - 1
                axins.plot(
                    c,
                    last_component,
                    linewidth=lw,
                    color=colors[j],
                    linestyle=params['linestyles'][i],
                    label=None
                )

                # saturation
                n = Model(obl_points=params['obl_points'][i], components=get_components(n_comps),
                          reservoir_type=params['reservoir_type'][i], nx=params['nx'][i],
                          itor_mode=params['itor_mode'][i], itor_type=params['itor_type'][i],
                          is_barycentric=params['barycentric'][i])
                sat = np.zeros(n_cells)
                for k, id in enumerate(ids):
                    n.physics.property_containers[0].evaluate(data['X'][-1, id, :])
                    sat[k] = n.physics.property_containers[0].output_props['sat0']()
                axins.plot(c, sat, linewidth=lw, color='orange', linestyle=params['linestyles'][i], label=None)

            # Set the limits for the inset
            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)

            # Remove ticks for clarity
            axins.set_xticks([])
            axins.set_yticks([])

            # Optional: Add a box around the zoomed area in the main plot
            ax.indicate_inset_zoom(axins, edgecolor="black")

            # Alternatively, use lines to connect the inset to the main plot
            mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec="0.8")

            # ----- End of Inset Plot 3 -----

    fig.tight_layout()
    fig.savefig(os.path.join(path_prefix, pic_fname))

def get_output_folder(itor_mode, itor_type, obl_points, n_comps, reservoir_type, nx: int = None, is_barycentric: bool = False):
    if nx is None:
        output_folder = 'output_' + itor_type + '_' + itor_mode + '_' + str(obl_points) + '_{}comp'.format(n_comps) + '_' + reservoir_type
    else:
        if itor_type == 'linear' and is_barycentric:
            output_folder = 'output_' + itor_type + '_' + itor_mode + '_' + str(obl_points) + '_{}comp'.format(n_comps) + '_barycentric_' + reservoir_type + '_' + str(nx)
        else:
            output_folder = 'output_' + itor_type + '_' + itor_mode + '_' + str(obl_points) + '_{}comp'.format(n_comps) + '_' + reservoir_type + '_' + str(nx)
    return output_folder

def get_components(n_comps: int):
    if n_comps == 3:
        components = ['CO2', 'C1', 'C4']
    elif n_comps == 4:
        components = ['CO2', 'C1', 'C4', 'C10']
    elif n_comps == 6:
        components = ['CO2', 'C1', 'C4', 'C10', 'C16', 'C20']
    elif n_comps == 8:
        components = ['CO2', 'C1', 'C2', 'C4', 'C6', 'C10', 'C16', 'C20']
    elif n_comps == 10:
        components = ['CO2', 'C1', 'C2', 'C4', 'C5', 'nC5', 'C6', 'C10', 'C16', 'C20']
    elif n_comps == 12:
        components = ['CO2', 'C1', 'C2', 'C3', 'C4', 'nC4', 'C5', 'nC5', 'C6', 'C10', 'C16', 'C20']
    elif n_comps == 14:
        components = ['CO2', 'C1', 'C2', 'C3', 'C4', 'nC4', 'C5', 'nC5', 'C6', 'C8', 'C10', 'C12', 'C16', 'C20']
    elif n_comps == 16:
        components = ['CO2', 'C1', 'C2', 'C3', 'C4', 'nC4', 'C5', 'nC5',
                      'C6', 'C8', 'C10', 'C12', 'C14', 'C16', 'C18', 'C20']
    elif n_comps == 20:
        components = ['CO2', 'C1', 'C2', 'C3', 'C4', 'nC4', 'C5', 'nC5', 'C6', 'C7',
                      'C8', 'C9', 'C10', 'C11', 'C12', 'C14', 'C16', 'C18', 'C19', 'C20']
    else:
        print('{} is not a valid number of components'.format(n_comps))
        components = []
    return components

def run(itor_mode, itor_type, obl_points, n_comps, reservoir_type, nx: int = None, is_barycentric: bool = False, vtk_output: bool = False):
    output_folder = get_output_folder(itor_mode=itor_mode, itor_type=itor_type, obl_points=obl_points, n_comps=n_comps,
                                      reservoir_type=reservoir_type, nx=nx, is_barycentric=is_barycentric)

    if itor_type == 'linear':
        log_3d_body_path = False
    else:
        log_3d_body_path = False#True

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    redirect_darts_output(os.path.join(output_folder, 'log.out'))

    n = Model(obl_points=obl_points, components=get_components(n_comps), reservoir_type=reservoir_type, nx=nx,
              itor_mode=itor_mode, itor_type=itor_type, is_barycentric=is_barycentric)
    n.init(itor_mode=itor_mode, itor_type=itor_type, output_folder=output_folder, is_barycentric=is_barycentric)

    n_months = 2 * 12
    if reservoir_type != '1D':
        if vtk_output:
            n.output_to_vtk(ith_step=0)
        if reservoir_type != '2D':
            n.set_wells_spe10()
            n_months = 10 * 12

    for i in range(n_months):
        if reservoir_type.split('_')[0] == 'spe10':
            ts_mult == 4.0 if  reservoir_type == 'spe10_20_40_40' else 1.0
            t = n.physics.engine.t
            if t < 70:
                n.params.max_ts = ts_mult * 0.25
            elif t < 100:
                n.params.max_ts = ts_mult * 0.35
            elif t < 400:
                n.params.max_ts = ts_mult * 0.5
            elif t < 2000:
                n.params.max_ts = ts_mult * 1.0
            else:
                n.params.max_ts = ts_mult * 1.5


        n.run(30.5, log_3d_body_path=log_3d_body_path)
        if reservoir_type != '1D' and vtk_output:
            n.output_to_vtk(ith_step=i + 1)

    n.timer.stop()
    n.print_timers()
    n.print_stat()

    if reservoir_type == '1D' and vtk_output:
        # populate input lists for comparing multiple solutions
        animate_solution_1d_single_plot(paths=[output_folder + '/'],
                            labels=[itor_type + ', ' + itor_mode + ', N=' + str(nx)],
                            n_cells=[nx],
                            lower_lim=8.e-4,
                            upper_lim=1.5 * n.ini_comp[1])

    return n.timer, n.physics.engine.stat

def test_performance(params, n_repeat: int = 1):
    n_models = len(params['itor_type'])
    n_repeat = 1
    output = {'total_time': [],
              'interpolation_time': [],
              'reservoir_interpolation_time': [],
              'point_generation_time': [],
              'timesteps': [],
              'nonlinear_iterations': [],
              'linear_iterations': [],
              'wasted_timesteps': [],
              'wasted_linear_iterations': [],
              'wasted_nonlinear_iterations': []}

    for i in range(n_models):
        for key, val in output.items():
            val.append([])
        for j in range(n_repeat):
            timer, stat = run(itor_type=params['itor_type'][i], itor_mode=params['itor_mode'][i],
                              obl_points=params['obl_points'][i], n_comps=params['n_comps'][i],
                              reservoir_type=params['reservoir_type'][i], nx=params['nx'][i],
                              is_barycentric=params['barycentric'][i])

            output['timesteps'][i].append(stat.n_timesteps_total)
            output['nonlinear_iterations'][i].append(stat.n_newton_total)
            output['linear_iterations'][i].append(stat.n_linear_total)
            output['wasted_timesteps'][i].append(stat.n_timesteps_wasted)
            output['wasted_nonlinear_iterations'][i].append(stat.n_newton_wasted)
            output['wasted_linear_iterations'][i].append(stat.n_linear_wasted)

            output['total_time'][i].append(timer.get_timer())
            output['interpolation_time'][i].append(timer.node['simulation'].node['jacobian assembly']\
                                                   .node['interpolation'].get_timer())
            output['reservoir_interpolation_time'][i].append(timer.node['simulation'].node['jacobian assembly']\
                                                   .node['interpolation'].node['reservoir 0 interpolation'].get_timer())
            nodes1 = [nd for nd in timer.node['simulation'].node['jacobian assembly'].node['interpolation'].\
                node['reservoir 0 interpolation'].node]
            if 'point generation' in nodes1:
                output['point_generation_time'][i].append(timer.node['simulation'].node['jacobian assembly'] \
                                                    .node['interpolation'].node['reservoir 0 interpolation'].\
                                                          node['point generation'].get_timer())
            elif 'body generation' in nodes1:
                output['point_generation_time'][i].append(timer.node['simulation'].node['jacobian assembly'] \
                                                    .node['interpolation'].node['reservoir 0 interpolation'].\
                                                          node['body generation'].node['point generation'].get_timer())
            else:
                output['point_generation_time'][i].append(0.0)

        for key, val in output.items():
            output[key][i] = np.array(val[i])

    for key, val in output.items():
        output[key] = np.array(val)

    # for key, val in output.items():
    #     print(f'{key}: {val.flatten()}')

    return output

def write_performance_output(filename, param_arrays, res_arrays):
    dict_params = { key: np.concatenate([d[key] for d in param_arrays], axis=0) for key in param_arrays[0].keys() }
    dict_res = { key: np.concatenate([d[key] for d in res_arrays], axis=0) for key in res_arrays[0].keys() }

    out_dict = {}
    for key, value in dict_res.items():
        out_dict[key] = np.mean(value, axis=-1)

    df = pd.DataFrame({**dict_params, **out_dict})
    df.to_excel(filename, index=False)

def test_linear_multilinear_obl_points():
    n_repeat = 1
    n_runs = 9
    nx = 300
    # 1D
    params1 = {'itor_type': 6 * ['linear'] + 3 * ['multilinear'],
              'itor_mode': n_runs * ['adaptive'],
              'obl_points': 3 * [64, 256, 1024],
              'n_comps': n_runs * [6],
              'barycentric': 3 * [False] + 3 * [True] + 3 * [False],
              'reservoir_type': n_runs * ['1D'],
              'nx': n_runs * [nx]}
    out_type_1d = test_performance(params=params1, n_repeat=n_repeat)
    # 2D
    params2 = {'itor_type': 6 * ['linear'] + 3 * ['multilinear'],
              'itor_mode': n_runs * ['adaptive'],
              'obl_points': 3 * [64, 256, 1024],
              'n_comps': n_runs * [6],
              'barycentric': 3 * [False] + 3 * [True] + 3 * [False],
              'reservoir_type': n_runs * ['2D'],
              'nx': n_runs * [nx]}
    out_type_2d = test_performance(params=params2, n_repeat=n_repeat)

    print(f'Linear vs Multilinear in 1D setup with nx={nx}')
    for key, val in out_type_1d.items():
        print(f'{key}: {val.flatten()}')
    print('\n')
    print(f'Linear vs Multilinear in 2D setup with nx=ny={nx}')
    for key, val in out_type_2d.items():
        print(f'{key}: {val.flatten()}')

    write_performance_output(filename='test_linear_multilinear_obl_points.xlsx',
                             param_arrays=[params1, params2],
                             res_arrays=[out_type_1d, out_type_2d])

def test_linear_multilinear_components():
    n_repeat = 1
    n_runs = 9
    # 1D
    params1 = {'itor_type': 6 * ['linear'] + 3 * ['multilinear'],
              'itor_mode': n_runs * ['adaptive'],
              'obl_points': n_runs * [100],
              'n_comps': 3 * [4, 6, 8],
              'barycentric': 3 * [False] + 3 * [True] + 3 * [False],
              'reservoir_type': n_runs * ['1D'],
              'nx': n_runs * [1000]}
    out_type_1d = test_performance(params=params1, n_repeat=n_repeat)
    # 2D
    params2 = {'itor_type': 6 * ['linear'] + 3 * ['multilinear'],
              'itor_mode': n_runs * ['adaptive'],
              'obl_points': n_runs * [100],
              'n_comps': 3 * [4, 6, 8],
              'barycentric': 3 * [False] + 3 * [True] + 3 * [False],
              'reservoir_type': n_runs * ['2D'],
              'nx': n_runs * [100]}
    out_type_2d = test_performance(params=params2, n_repeat=n_repeat)

    print('Linear vs Multilinear in 1D setup with nx=1000')
    for key, val in out_type_1d.items():
        print(f'{key}: {val.flatten()}')
    print('\n')
    print('Linear vs Multilinear in 2D setup with nx=100')
    for key, val in out_type_2d.items():
        print(f'{key}: {val.flatten()}')
    print('\n')

    write_performance_output(filename='test_linear_multilinear_components.xlsx',
                             param_arrays=[params1, params2],
                             res_arrays=[out_type_1d, out_type_2d])

def test_linear_multilinear_nx():
    n_repeat = 1
    n_runs = 9
    # 1D
    params1 = {'itor_type': 6 * ['linear'] + 3 * ['multilinear'],
              'itor_mode': n_runs * ['adaptive'],
              'obl_points': n_runs * [256],
              'n_comps': n_runs * [6],
              'barycentric': 3 * [False] + 3 * [True] + 3 * [False],
              'reservoir_type': n_runs * ['1D'],
              'nx': 3 * [100, 1000, 10000]}
    out_type_1d = test_performance(params=params1, n_repeat=n_repeat)
    # 2D
    params2 = {'itor_type': 6 * ['linear'] + 3 * ['multilinear'],
              'itor_mode': n_runs * ['adaptive'],
              'obl_points': n_runs * [256],
              'n_comps': n_runs * [6],
              'barycentric': 3 * [False] + 3 * [True] + 3 * [False],
              'reservoir_type': n_runs * ['2D'],
              'nx': 3 * [10, 100, 1000]}
    out_type_2d = test_performance(params=params2, n_repeat=n_repeat)

    write_performance_output(filename='test_linear_multilinear_nx.xlsx',
                             param_arrays=[params1, params2],
                             res_arrays=[out_type_1d, out_type_2d])

# test_linear_multilinear_obl_points()
# test_linear_multilinear_components()
# test_linear_multilinear_nx()

n_comps = 3
obl_points = 1024 # 1024 # 128
nx = 300
# 1D
# run(itor_type='multilinear', itor_mode='adaptive', obl_points=obl_points, n_comps=n_comps, reservoir_type='1D', nx=nx, is_barycentric=False, vtk_output=True)
# 2D
# run(itor_type='linear', itor_mode='adaptive', obl_points=obl_points, n_comps=n_comps, reservoir_type='2D', nx=nx, is_barycentric=True, vtk_output=False)
# SPE10
run(itor_type='multilinear', itor_mode='adaptive', obl_points=obl_points, n_comps=n_comps, reservoir_type='SPE10_60_220_85', is_barycentric=False, vtk_output=True)

# params = {'itor_type': ['multilinear', 'multilinear', 'linear', 'linear'],
#            'itor_mode': 4 * ['adaptive'],
#            'obl_points': [1024] + 3 * [64],
#            'n_comps': 4 * [6],
#            'barycentric': 3 * [False] + [True],
#            'reservoir_type': 4 * ['1D'],
#            'nx': 4 * [300],
#            'linestyles': ['-', '--', '-.', ':']}
# plot_comparison(params=params, path_prefix='for_paper', pic_fname='obl_points_1d.png')
#
# params = {'itor_type': ['multilinear', 'multilinear', 'linear', 'linear'],
#            'itor_mode': 4 * ['adaptive'],
#            'obl_points': [1024] + 3 * [64],
#            'n_comps': 4 * [6],
#            'barycentric': 3 * [False] + [True],
#            'reservoir_type': 4 * ['2D'],
#            'nx': 4 * [300],
#            'linestyles': ['-', '--', '-.', ':']}
# plot_comparison(params=params, path_prefix='for_paper', pic_fname='obl_points_2d.png')

# n_comps = 14
# paths = [get_output_folder(itor_type='linear', itor_mode='adaptive', obl_points=128, n_comps=n_comps, reservoir_type='1D', nx=100, is_barycentric=False) + '/',
#          get_output_folder(itor_type='linear', itor_mode='adaptive', obl_points=128, n_comps=n_comps, reservoir_type='1D', nx=1000, is_barycentric=False) + '/',
#          get_output_folder(itor_type='linear', itor_mode='adaptive', obl_points=128, n_comps=n_comps, reservoir_type='1D', nx=4000, is_barycentric=False) + '/']
# labels = ['nx=100', 'nx=1000', 'nx=4000']
# upper_lims = np.array([140, 1.01] + [0.275, 0.125, 0.100, 0.075, 0.075, 0.065, 0.065, 0.060, 0.050, 0.040, 0.030, 0.020, 0.015])
# upper_lims[2:] *= 1.2
# animate_solution_1d_single_plot(paths=paths,
#                     labels=labels,
#                     n_cells=[100, 1000, 4000],
#                     lower_lims=[48.9] + (n_comps-1) * [-1.e-2],
#                     upper_lims=upper_lims,
#                     video_fname='comparison.mp4')


# paths = [get_output_folder(itor_type='linear', itor_mode='adaptive', obl_points=obl_points, n_comps=n_comps, reservoir_type='1D', nx=100, is_barycentric=False) + '/']
# labels = ['nx = 100']
# upper_lims = np.array([140, 1.01] + [0.240, 0.120, 0.090, 0.070, 0.070, 0.060, 0.060, 0.050, 0.045,
#                              0.040, 0.035, 0.030, 0.025, 0.020, 0.015, 0.010, 0.007, 0.005, 0.003])
# upper_lims[2:] *= 1.2
# animate_solution_1d_single_plot(paths=paths,
#                     labels=labels,
#                     n_cells=[100],
#                     lower_lim=8.e-4,
#                     upper_lim=0.3,
#                     video_fname='comparison.mp4')