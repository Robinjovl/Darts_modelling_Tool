import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib import rcParams
from scipy import interpolate
rcParams['animation.ffmpeg_path'] = r'c:\ffmpeg\bin\ffmpeg.exe'
rcParams["text.usetex"]=False
#rcParams["font.sans-serif"] = ["Liberation Sans"]
#rcParams["font.serif"] = ["Liberation Serif"]
plt.rc('xtick',labelsize=11)
plt.rc('ytick',labelsize=12)

def plot_conv(data, n_cells, dx, dt, filename, params):
    x = np.sqrt(dx * dt)
    print('np.sqrt(dx * dt)=', x)
    id = np.argsort(x)

    mini = 1e+10
    maxi = 1e-10
    for k, v in data.items():
        data[k] = np.array(v)
        if k == 's':
            data[k] = params['s_mult'] * data[k]
        maxi = max(maxi, np.max(data[k]))
        mini = min(mini, np.min(data[k]))

    ylims = (mini / 1.5, maxi * 1.5)
    xlims = (np.min(x), np.max(x))

    o1 = interpolate.interp1d(x, ylims[0] * x / xlims[0])
    o2 = interpolate.interp1d(x, 30 * ylims[0] * (x / xlims[0]) ** 2)
    o_point_x = ( xlims[0] + xlims[1] ) / 2
    angle_o1 = 180 + np.degrees(np.arctan2(o1(x[-1]) - o1(x[0]), x[-1] - x[0])) + params['angle_o1']
    angle_o2 = 180 + np.degrees(np.arctan2(o2(x[-1]) - o2(x[0]), x[-1] - x[0])) + params['angle_o2']

    labels = {'p': r'$\|p - p_h\|_{L2}$',
              'u': r'$\|{\bf u} - {\bf u}_h\|_{L2}$',
              'uy': r'$\|{\bf u} - {\bf u}_h\|_{L2}$',
              't': r'$\|T - T_h\|_{L2}$',
              'v': r'$\|{\bf q}^{(f)} - {\bf q}^{(f)}_h\|_{L2}$',
              's': r'$\|{\bf \sigma} - {\bf \sigma}_h\|_{L2}$',}
    colors = {'p': 'b', 'u': 'r', 'uy': 'r', 't': 'g', 'v': 'g', 's': 'm'}
    fig, err = plt.subplots(nrows=1, sharex=True, figsize=(6, 6))

    # err.loglog(x, o2(x), color='k', linewidth=.8, linestyle='-.')#, label='2nd order')
    # err.text(o_point_x, o2(o_point_x), '2nd order', fontsize=12, rotation=angle_o2)
    for k, val in data.items():
        order = (np.diff(np.log(val[id])) / np.diff(np.log(x[id])))[0]
        if k == 's':
            label = str(params['s_mult']) + labels[k] + ': ' + str(round(order, 2))
        else:
            label = labels[k] + ': ' + str(round(order, 2))
        err.loglog(x, val, 'r-', color=colors[k], label=label, marker='o', markersize=7, markerfacecolor='none')

    err.loglog(x, 2 * o1(x), color='k', linewidth=.8, linestyle='--')#, label='1st order')
    err.text(o_point_x, 1.6 * o1(o_point_x), '1st order', fontsize=12, rotation=angle_o1)

    err.set_xlabel(r'$\sqrt{\Delta x\Delta t}, \sqrt{m\cdot day}$', fontsize=16)
    err.set_ylabel('error norm', fontsize=16)

    err.legend(prop={'size': 14})
    #err.grid(True)
    fig.tight_layout()
    fig.savefig(filename)
    #plt.show()
    #plt.close()

def plot_conv_main(nx_list, T=10., case='terzaghi', mesh='rect'):
    prefix = case + '_mech_discretizer_' + mesh
    data = dict()
    n_cells = []
    if case == 'bai':
        mesh_size = np.array([1., 7., 1.])
        vars = ['uy', 'p', 't']
        an_mult = {'uy': -1, 'p': 1.e-5, 't': 1}
    else:
        mesh_size = np.array([100., 100., 10.])  # mesh size by x, y, z
        vars = ['u', 'p']
        an_mult = {'u': 1, 'p': 1}

    for mesh_resolution in nx_list:
        folder = 'sol_' + prefix +'_' + str(mesh_resolution)
        last_n_elements = 3 if case == 'bai' else mesh_resolution
        for var in vars:
            arrays = np.loadtxt(os.path.join(folder, var + '_data.txt'))  # the columns are t, x, sol_analytic, sol
            sol = arrays[-last_n_elements:, 2].flatten()
            sol_analytic = an_mult[var] * arrays[-last_n_elements:, 3].flatten()
            size = last_n_elements  # the number of cells in the mesh
            volume_total = mesh_size[0] * mesh_size[1] * mesh_size[2]
            volume_cell = np.zeros(size) + volume_total / size
            if var not in data.keys():
                data[var] = []
            deviation_norm = np.sqrt( np.dot((sol - sol_analytic)**2, volume_cell ) / volume_total)
            deviation_max = (sol - sol_analytic).max()
            print('nx=', mesh_resolution, 'var=', var, 'deviation_norm=', deviation_norm, 'deviation_max=', deviation_max)
            data[var].append(deviation_norm) # a norm weighted by volume
        n_cells.append(size)
    dx = mesh_size[0] / np.array(nx_list)
    dt = T / np.array(nx_list)
    params = {'angle_o1': 26, 'angle_o2': 26, 's_mult': 0.1}
    plot_conv(data=data, n_cells=n_cells, dx=dx, dt=dt, filename=prefix + '.png', params=params)

# read from vtk
# import meshio
# filename_0 = os.path.join(folder, 'solution0.vtk')
# data_0 = meshio.read(filename_0)
# connectivity = data_0.cells_dict['hexahedron']
# n_cells = len(connectivity)
# volume = #TODO

# filename = os.path.join(folder, 'solution60.vtk')  # the last timestep
# data = meshio.read(filename)

# for var in ['ux', 'uy', 'uz', 'p']:
#    a = np.array(data.cell_data[var])

if __name__ == '__main__':
    nx_list = [5, 15, 30, 40, 50, 80]
    plot_conv_main(nx_list=nx_list)