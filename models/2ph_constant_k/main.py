import numpy as np
import pandas as pd
import sys
import os
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from matplotlib import rcParams

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

def animate_solution_1d(paths, n_cells, labels, lower_lims, upper_lims, video_fname='plot.mp4'):
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

def get_output_folder(itor_mode, itor_type, obl_points, n_comps, reservoir_type, nx: int = None, is_barycentric: bool = False):
    if nx is None:
        output_folder = 'output_' + itor_type + '_' + itor_mode + '_' + str(obl_points) + '_{}comp'.format(n_comps) + '_' + reservoir_type
    else:
        if itor_type == 'linear' and is_barycentric:
            output_folder = 'output_' + itor_type + '_' + itor_mode + '_' + str(obl_points) + '_{}comp'.format(n_comps) + '_barycentric_' + reservoir_type + '_' + str(nx)
        else:
            output_folder = 'output_' + itor_type + '_' + itor_mode + '_' + str(obl_points) + '_{}comp'.format(n_comps) + '_' + reservoir_type + '_' + str(nx)
    return output_folder

def run(itor_mode, itor_type, obl_points, n_comps, reservoir_type, nx: int = None, is_barycentric: bool = False, vtk_output: bool = False):
    output_folder = get_output_folder(itor_mode=itor_mode, itor_type=itor_type, obl_points=obl_points, n_comps=n_comps,
                                      reservoir_type=reservoir_type, nx=nx, is_barycentric=is_barycentric)

    if itor_type == 'linear':
        log_3d_body_path = False
    else:
        log_3d_body_path = False#True

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    redirect_darts_output('log.out')

    if n_comps == 3:
        components = ['CO2', 'C1', 'C6']
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
        return

    n = Model(obl_points=obl_points, components=components, reservoir_type=reservoir_type, nx=nx)
    n.init(itor_mode=itor_mode, itor_type=itor_type, output_folder=output_folder, is_barycentric=is_barycentric)

    if reservoir_type != '1D':
        if vtk_output:
            n.output_to_vtk(ith_step=0)
        if reservoir_type != '2D':
            n.params.first_ts = n.params.max_ts = 0.001
            n.run(1.0, log_3d_body_path=log_3d_body_path)
            n.physics.engine.t = 0.0
            n.set_spe10_well_controls_initialized()

    for i in range(12):
        n.run(30.5, log_3d_body_path=log_3d_body_path)
        if reservoir_type != '1D' and vtk_output:
            n.output_to_vtk(ith_step=i + 1)

    n.timer.stop()
    n.print_timers()
    n.print_stat()

    if reservoir_type == '1D' and vtk_output:
        # populate input lists for comparing multiple solutions
        upper_lims = np.array([160, 1.01] + n.ini_comp[1:])
        upper_lims[2:] *= 1.2
        animate_solution_1d(paths=[output_folder + '/'],
                            labels=[itor_type + ', ' + itor_mode + ', N=' + str(nx)],
                            n_cells=[nx],
                            lower_lims = [48.9] + (n_comps-1) * [-1.e-2],
                            upper_lims = upper_lims)

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
    # 1D
    params1 = {'itor_type': 6 * ['linear'] + 3 * ['multilinear'],
              'itor_mode': n_runs * ['adaptive'],
              'obl_points': 3 * [64, 256, 1024],
              'n_comps': n_runs * [6],
              'barycentric': 3 * [False] + 3 * [True] + 3 * [False],
              'reservoir_type': n_runs * ['1D'],
              'nx': n_runs * [1000]}
    out_type_1d = test_performance(params=params1, n_repeat=n_repeat)
    # 2D
    params2 = {'itor_type': 6 * ['linear'] + 3 * ['multilinear'],
              'itor_mode': n_runs * ['adaptive'],
              'obl_points': 3 * [64, 256, 1024],
              'n_comps': n_runs * [6],
              'barycentric': 3 * [False] + 3 * [True] + 3 * [False],
              'reservoir_type': n_runs * ['2D'],
              'nx': n_runs * [100]}
    out_type_2d = test_performance(params=params2, n_repeat=n_repeat)

    print('Linear vs Multilinear in 1D setup with nx=1000')
    for key, val in out_type_1d.items():
        print(f'{key}: {val.flatten()}')
    print('\n')
    print('Linear vs Multilinear in 2D setup with nx=ny=100')
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
              'obl_points': n_runs * [100],
              'n_comps': n_runs * [6],
              'barycentric': 3 * [False] + 3 * [True] + 3 * [False],
              'reservoir_type': n_runs * ['1D'],
              'nx': 3 * [100, 1000, 10000]}
    out_type_1d = test_performance(params=params1, n_repeat=n_repeat)
    # 2D
    params2 = {'itor_type': 6 * ['linear'] + 3 * ['multilinear'],
              'itor_mode': n_runs * ['adaptive'],
              'obl_points': n_runs * [100],
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

run(itor_type='linear', itor_mode='adaptive', obl_points=128, n_comps=3, reservoir_type='1D', nx=100, is_barycentric=False, vtk_output=False)
# run(itor_type='linear', itor_mode='adaptive', obl_points=128, n_comps=16, reservoir_type='1D', nx=100, is_barycentric=False, vtk_output=True)
# run(itor_type='linear', itor_mode='adaptive', obl_points=128, n_comps=14, reservoir_type='1D', nx=4000, is_barycentric=False, vtk_output=True)

# n_comps = 14
# paths = [get_output_folder(itor_type='linear', itor_mode='adaptive', obl_points=128, n_comps=n_comps, reservoir_type='1D', nx=100, is_barycentric=False) + '/',
#          get_output_folder(itor_type='linear', itor_mode='adaptive', obl_points=128, n_comps=n_comps, reservoir_type='1D', nx=1000, is_barycentric=False) + '/',
#          get_output_folder(itor_type='linear', itor_mode='adaptive', obl_points=128, n_comps=n_comps, reservoir_type='1D', nx=4000, is_barycentric=False) + '/']
# labels = ['nx=100', 'nx=1000', 'nx=4000']
# upper_lims = np.array([140, 1.01] + [0.275, 0.125, 0.100, 0.075, 0.075, 0.065, 0.065, 0.060, 0.050, 0.040, 0.030, 0.020, 0.015])
# upper_lims[2:] *= 1.2
# animate_solution_1d(paths=paths,
#                     labels=labels,
#                     n_cells=[100, 1000, 4000],
#                     lower_lims=[48.9] + (n_comps-1) * [-1.e-2],
#                     upper_lims=upper_lims,
#                     video_fname='comparison.mp4')

# run(itor_type='linear', itor_mode='adaptive', obl_points=1024, reservoir_type='2D', nx=10)
# run(itor_type='linear', itor_mode='adaptive', obl_points=1024, reservoir_type='spe10_20_40_40')
# run(itor_type='multilinear', itor_mode='adaptive', obl_points=1024)
# run(itor_mode='static_nested', obl_points=4)