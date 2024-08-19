import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from matplotlib import rcParams

from model import Model
from darts.engines import value_vector, redirect_darts_output
from darts.physics.operators_base import PropertyOperators as props
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

def animate_solution_1d(paths, n_cells, labels, lower_lims, upper_lims, video_fname='plot.mp4'):
    data0 = load_hdf5_to_dict(filename=paths[0] + 'solution.h5')['dynamic']
    n_cells_max = max(n_cells)
    c = [np.arange(n_cells_max, step=int(n_cells_max / n_cells[i])) for i in range(len(n_cells))]
    n_plots = len(data0['variable_names'])
    n_steps = data0['time'].size # number of saved snapshots
    colors = ['b', 'r', 'g', 'm', 'c', 'y', 'k']
    fig, ax = plt.subplots(nrows=n_plots, sharex=True, figsize=(7, 12))
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

def run(python_assembly: bool = False, output: bool = True):
    redirect_darts_output('run.log')
    n = Model(python_assembly=python_assembly)
    # n.params.linear_type = n.params.linear_solver_t.cpu_superlu
    n.init()
    if python_assembly:
        n.init_assembly()

    for i in range(12):
        n.run(30.5)

    n.print_timers()
    n.print_stat()

    if output:
        # populate input lists for comparing multiple solutions
        upper_lims = np.array([1.05 * n.initial_values[n.physics.vars[0]]] + [1.01])
        animate_solution_1d(paths=[n.output_folder + '/'],
                            labels=['with JAX' if python_assembly else 'w/o JAX'],
                            n_cells=[n.nx],
                            lower_lims=[0.95 * n.initial_values[n.physics.vars[0]]] + (2-1) * [-1.e-2],
                            upper_lims=upper_lims)


if __name__ == '__main__':
    run(python_assembly=True)