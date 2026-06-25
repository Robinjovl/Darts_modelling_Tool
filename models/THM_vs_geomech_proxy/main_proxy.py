import numpy as np
import os
import meshio
from datetime import datetime
from matplotlib import pyplot as plt
from scipy.interpolate import griddata as gd
from functools import reduce

from main import run

# unit conversion factors
m2mm = 1e3
bars2mpa = 0.1
Pa2bars = 1e-5
g_grav = 9.81  # m/s^2

def fmt(x):
    return '{:.3}'.format(x)

def read_vtk_darts_solution(folder, timestep : int):
    filename = os.path.join(folder, 'solution'+str(timestep)+'.vtu')
    if not os.path.isfile(filename):
        print('THM vtk file not found:', filename)
        print('Run THM first to generate vtk files with p,T changes (by setting run_thm = True)')
        raise FileNotFoundError(filename)
    msh = meshio.read(filename)
    print('Reading', filename)
    #print("\tCells:", msh.cells_dict.keys())
    #print("\tCell Data:", msh.cell_data.keys())
    return msh

def geomech_init_geometry(mesh_data):
    #if hasattr(self, 'prisms'):  # do only once
    #    return

    # coordinates 3 lines, Nnodes columns
    points = mesh_data.points.T

    nodes = np.zeros((3, len(points[0][:])))
    for k in range(len(points[0][:])):
        nodes[0][k] = points[1][k]
        nodes[1][k] = points[0][k]
        nodes[2][k] = points[2][k]

    connectivity = mesh_data.cells_dict['hexahedron']

    prisms = np.zeros((len(connectivity), 6))
    for k in range(len(connectivity)):
        prism = connectivity[k]
        xloc, yloc, zloc = [], [], []
        for i in prism:
            yloc.append(nodes[1][i])
            xloc.append(nodes[0][i])
            zloc.append(nodes[2][i])
        prisms[k][0] = np.amin(yloc)
        prisms[k][1] = np.amax(yloc)

        prisms[k][2] = np.amin(xloc)
        prisms[k][3] = np.amax(xloc)

        prisms[k][4] = np.amax(zloc)
        prisms[k][5] = np.amin(zloc)

    return prisms

class thm_solution:
    pass

def read_thm_solution_from_vtk(m, folder : str, timestep: int):
    thm_sol = thm_solution()

    msh_initial = read_vtk_darts_solution(folder=folder, timestep=0)
    poro = np.array(msh_initial.cell_data['poro']).flatten()
    thm_sol.p_init = np.array(msh_initial.cell_data['pressure']).flatten()
    thm_sol.Szz_init = np.array(msh_initial.cell_data['tot_stress'])[0, :, 2] * bars2mpa # ZZ

    msh_last = read_vtk_darts_solution(folder=folder, timestep=timestep)
    p_last = np.array(msh_last.cell_data['pressure']).flatten()
    thm_sol.ux_last = np.array(msh_last.cell_data['ux']).flatten()
    thm_sol.uy_last = np.array(msh_last.cell_data['uy']).flatten()
    thm_sol.uz_last = np.array(msh_last.cell_data['uz']).flatten()
    thm_sol.delta_Sxx_last = np.array(msh_last.cell_data['delta_eff_stress'])[0, :, 0] * bars2mpa #  XX
    thm_sol.delta_Syy_last = np.array(msh_last.cell_data['delta_eff_stress'])[0, :, 1] * bars2mpa #  YY
    thm_sol.delta_Szz_last = np.array(msh_last.cell_data['delta_eff_stress'])[0, :, 2] * bars2mpa # ZZ
    thm_sol.delta_total_Sxx_last = np.array(msh_last.cell_data['delta_tot_stress'])[0, :, 0] * bars2mpa #  XX
    thm_sol.delta_total_Syy_last = np.array(msh_last.cell_data['delta_tot_stress'])[0, :, 1] * bars2mpa #  YY
    thm_sol.delta_total_Szz_last = np.array(msh_last.cell_data['delta_tot_stress'])[0, :, 2] * bars2mpa # ZZ
    thm_sol.qx_last = np.array(msh_last.cell_data['strain'])[0, :, 0]  #  XX
    thm_sol.qy_last = np.array(msh_last.cell_data['strain'])[0, :, 1]  #  YY
    thm_sol.qz_last = np.array(msh_last.cell_data['strain'])[0, :, 2]  # ZZ

    thm_sol.folder = folder

    if 'delta_pressure' in msh_last.cell_data.keys():
        thm_sol.delta_pressure = np.array(msh_last.cell_data['delta_pressure']).flatten()
    else:
        p_initial = np.array(msh_initial.cell_data['pressure']).flatten()
        thm_sol.delta_pressure = p_last - p_initial
    thm_sol.delta_pressure *= 0.1 # bars to MPa

    # delta_temperature is zero in isothermal case
    thm_sol.delta_temperature = np.zeros_like(thm_sol.delta_pressure)
    if 'delta_temperature' in msh_last.cell_data.keys():
        thm_sol.delta_temperature = np.array(msh_last.cell_data['delta_temperature']).flatten()

    prisms = geomech_init_geometry(msh_initial)
    print('\tprisms all', prisms.shape[0])

    #perm_threshold_for_prisms = 1e-6
    perm_threshold_for_prisms = 0 # do not use [rsv] filtering

    if m.idata.rock.perm_non_rsv <= perm_threshold_for_prisms: #use only the permeable part, assuming there is no p,T change in the impermeable part
        rsv = poro > m.idata.rock.poro_non_rsv  # reservoir (permeable) cells only will be used as input for proxy
    else: # there will be pressure diffusion, so need to use all cells
        rsv = poro > 0 # use all cells in the proxy

    thm_sol.delta_pressure_rsv = thm_sol.delta_pressure[rsv]
    thm_sol.delta_temperature_rsv = thm_sol.delta_temperature[rsv]
    thm_sol.prisms_rsv = prisms[rsv, :]
    print('\tprisms rsv', thm_sol.prisms_rsv.shape[0])

    # centroids are only used for THM data plotting, they are not used in proxy
    centroids = np.zeros((prisms.shape[0], 3))
    centroids[:, 0] = (prisms[:, 0] +  prisms[:, 1]) * 0.5 # Y
    centroids[:, 1] = (prisms[:, 2] +  prisms[:, 3]) * 0.5 # X
    centroids[:, 2] = (prisms[:, 4] +  prisms[:, 5]) * 0.5 # z
    thm_sol.centroids = centroids

    n_dim = 3  # X,Y,Z
    thm_sol.bounds = [0]*n_dim
    for k in range(n_dim):
        thm_sol.bounds[k] = centroids[:, k].min(), centroids[:, k].max()

    thm_sol.rsv_centroids = centroids[rsv, :]
    return thm_sol

def plot_delta_pressure_along_x(case, physics_type, wells_type, timesteps,
                                y=0.0, z=2200.0, x_range=None, n_points=200,
                                report_step=None, output_folder=None, generate_mesh=False):
    """Plot THM delta_pressure [MPa] along the X axis at fixed Y, Z, overlaying several timesteps.

    Reads the THM solution vtk for each timestep, interpolates delta_pressure from the cell
    centers onto a line of points running along X (at the given Y and Z), and draws all
    timesteps on a single combined figure.

    :param case: mesh case name, e.g. '41_41_66'
    :param physics_type: 'single_phase' or 'single_phase_thermal'
    :param wells_type: 'inj', 'prod', 'doublet', ...
    :param timesteps: list of vtk timestep indices to overlay
    :param y, z: fixed coordinates [m] of the line; it runs along X (defaults Y=0, Z=2200)
    :param x_range: optional (x_min, x_max) [m]; defaults to the THM centroid X-extent
    :param n_points: number of sample points along X
    :param report_step: optional [days] per vtk step; if given, legend labels show time in years
    :param output_folder: where to save the png; defaults to <results folder>/plots_delta_pressure_x
    :param generate_mesh: passed to Model (kept False to reuse an existing mesh)
    """
    folder = os.path.join('results', 'sol_cpp_' + physics_type + '_' + wells_type + '_' + case)

    # lightweight model just to provide idata (rsv filtering etc.) to the vtk reader
    from model import Model
    m = Model(model_folder=case, physics_type=physics_type, uniform_props=False,
              wells_type=wells_type, decouple_geomech=True, generate_mesh=generate_mesh,
              dummy='yes')
    m.set_input_data()

    if output_folder is None:
        output_folder = os.path.join(folder, 'plots_delta_pressure_x')
    os.makedirs(output_folder, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    for ts in timesteps:
        thm_sol = read_thm_solution_from_vtk(m, folder=folder, timestep=ts)
        # sample points along X at fixed Y, Z
        if x_range is None:
            x_min, x_max = thm_sol.centroids[:, 1].min(), thm_sol.centroids[:, 1].max()
        else:
            x_min, x_max = x_range
        points_x = np.linspace(x_min, x_max, n_points)
        points_y = np.full_like(points_x, y)
        points_z = np.full_like(points_x, z)
        # interpolate delta_pressure [MPa] from cell centers (X,Y,Z) onto the line points
        dp_line = gd((thm_sol.centroids[:, 1], thm_sol.centroids[:, 0], thm_sol.centroids[:, 2]),
                     thm_sol.delta_pressure, (points_x, points_y, points_z), method='linear')
        label = f't = {ts * report_step / 365.25:.3g} years' if report_step is not None else f'timestep {ts}'
        ax.plot(points_x, dp_line, label=label)

    ax.set_xlabel('X, m.')
    ax.set_ylabel('Pressure change, MPa.')
    ax.set_title(f'Pressure change along X (Y={y:g}, Z={z:g})')
    ax.legend(fontsize=8)
    ax.minorticks_on()
    ax.grid(which='major', linestyle='-', linewidth=0.8)
    ax.grid(which='minor', linestyle=':', linewidth=0.5)
    fig.tight_layout()
    out = os.path.join(output_folder, 'delta_pressure_along_x.png')
    fig.savefig(out)
    plt.close(fig)
    print('Saved', out)

def run_geomech_proxy(case, physics_type='single_phase',
                      wells_type=None, timestep=1, modes=[],
                      generate_mesh=True, n_threads=1, use_gpu=False, read_from_cache=False):
    folder = os.path.join('results', 'sol_cpp_' + physics_type + '_'  + wells_type + '_' + case)  # where vtk files are located

    # init geomech proxy
    from geomechanics import geomech
    g = geomech()
    # just to set input data
    from model import Model
    m = Model(model_folder=case, physics_type=physics_type, uniform_props=False,
              wells_type=wells_type, decouple_geomech=True, generate_mesh=generate_mesh,
              dummy='yes')
    m.set_input_data()
    # elastic constants
    g.poisson = m.idata.rock.nu
    g.young = m.idata.rock.E if np.isscalar(m.idata.rock.E) else m.idata.rock.E.mean()
    g.young *= bars2mpa
    g.thermal_expansion = m.idata.rock.th_expn_orig
    g.biot = m.idata.rock.biot

    g.set_num_threads(n_threads)
    print('N_THREADS =', n_threads)
    if use_gpu:
        from geomechanics import HAS_GPU
        if not HAS_GPU:
            raise RuntimeError('use_gpu=True but _proxygeomech_cuda is not available')
        g.set_platform('gpu')
        print('PLATFORM = gpu')
    else:
        print('PLATFORM = cpu')

    thm_sol = read_thm_solution_from_vtk(m, folder=folder, timestep=timestep)
    g.centroids = thm_sol.rsv_centroids

    output_folder = os.path.join(thm_sol.folder, 'plots_timestep_' + str(timestep))
    os.makedirs(output_folder, exist_ok=True)

    # plot THM solution
    if False:
        from plot_vtk_pyvista import plot_vtk_pyvista
        model_folder=case
        m.output_directory = os.path.join('results', 'sol_cpp_' + physics_type + '_' + wells_type + '_' + model_folder)
        plot_vtk_pyvista(m.output_directory, tstep_to_plot=0, idata=m.idata)  # initial
        plot_vtk_pyvista(m.output_directory, tstep_to_plot=-1, idata=m.idata) # last


    def find_cell_by_point(point):
        # find an index of the cell, closest to the desired point
        cell = ((thm_sol.centroids[:, 0] - point[0]) ** 2 + (thm_sol.centroids[:, 1] - point[1]) ** 2 + (
                    thm_sol.centroids[:, 2] - point[2]) ** 2).argmin()
        #print('get_thm_solution', 'closest cell is', thm_sol.centroids[cell, :], 'point', point)
        return cell

    def get_cell_center(point):
        cell = find_cell_by_point(point)
        return thm_sol.centroids[cell, 0], thm_sol.centroids[cell, 1], thm_sol.centroids[cell, 2]

    def get_thm_by_interp(array_dict, points_x, points_y, points_z, method='linear'):
        array_dict_interp = dict()
        for arr_name, arr in array_dict.items():
            # interpolate the solution (arr) from cell centers to given set of points
            array_dict_interp[arr_name] = gd((thm_sol.centroids[:, 1], thm_sol.centroids[:, 0], thm_sol.centroids[:, 2]), \
                arr, (points_x, points_y, points_z), method=method)
        return array_dict_interp

    def get_thm_dp_dt(point, verbose=False): # pressure (in MPa) and temperature (in K) changes
        cell = find_cell_by_point(point)
        dp = thm_sol.delta_pressure[cell]
        dt = thm_sol.delta_temperature[cell]
        return dp, dt

    def get_thm_displs(point, verbose=False):
        cell = find_cell_by_point(point)
        ux_thm = thm_sol.ux_last[cell]
        uy_thm = thm_sol.uy_last[cell]
        uz_thm = thm_sol.uz_last[cell]
        return ux_thm, uy_thm, uz_thm

    def get_thm_strain(point, verbose=False):
        cell = find_cell_by_point(point)
        qx_thm = thm_sol.qx_last[cell]
        qy_thm = thm_sol.qy_last[cell]
        qz_thm = thm_sol.qz_last[cell]
        return qx_thm, qy_thm, qz_thm

    def get_thm_stress(point, verbose=False): # effective stress in MPa
        cell = find_cell_by_point(point)
        return thm_sol.delta_Sxx_last[cell],  thm_sol.delta_Syy_last[cell],  thm_sol.delta_Szz_last[cell]

    def get_thm_total_stress(point, verbose=False): # total stress in MPa
        cell = find_cell_by_point(point)
        return thm_sol.delta_total_Sxx_last[cell],  thm_sol.delta_total_Syy_last[cell],  thm_sol.delta_total_Szz_last[cell]

    def get_thm_stress_by_deriv(point):
        # compute strain and stress in python from THM displacements (ux_last, etc)
        from geomechanics import deriv

        # find the closest neighbout cell in each direction. Since we don't know the cells size, we will move step by step
        n_neig = 6 # X- X+, Y-, Y+, Z-, Z+
        cell_neig = np.zeros(n_neig, dtype=int)
        point_neig = np.zeros((n_neig, thm_sol.n_dim))
        step = [10., 10., 5.]  # [m] to find neighboring cells, should be less than cell size

        mults = [0.5] * thm_sol.n_dim

        cell = find_cell_by_point(point)
        i = 0
        for side in [-1, 1]:
            for k in range(thm_sol.n_dim):
                cell_neig[i] = cell
                point_neig[i] = point.copy()  # start from the point itself
                if point_neig[i][k] < thm_sol.bounds[k][0] or point_neig[i][k] > thm_sol.bounds[k][1]: # if the starting point is out of bounds, start from the closest centroid
                    cell_neig[i] = find_cell_by_point(point_neig[i])
                    mults[k] = 1.0  # one-sided derivative
                j = 0
                while cell_neig[i] == cell:  # move, until reached the next cell
                    point_neig[i][k] += side * step[k]
                    if (point_neig[i][k] < thm_sol.bounds[k][0] and side == -1) or (point_neig[i][k] > thm_sol.bounds[k][1] and side == 1):  # reached the boundary
                        cell_neig[i] = cell
                        point_neig[i][k] = point[k]  # return to the original point
                        mults[k] = 1.0  # one-sided derivative
                        break
                    cell_neig[i] = find_cell_by_point(point_neig[i])
                    if j > 1000:  # just in case, to avoid infinite loop
                        print('Error: can not find a neighboring cell for point', point, 'side', side, 'k', k)
                        exit(1)
                point_neig[i] = thm_sol.centroids[cell_neig[i], :]
                i += 1

        #print('point', point)
        #print('z- neig center', centroids[find_cell_by_point(point_neig[2]), 2])
        #print('z+ neig center', centroids[find_cell_by_point(point_neig[5]), 2])

        ux_x_minus_, ux_x_plus_ = thm_sol.ux_last[cell_neig[0]], thm_sol.ux_last[cell_neig[3]]
        ux_y_minus_, ux_y_plus_ = thm_sol.ux_last[cell_neig[1]], thm_sol.ux_last[cell_neig[4]]
        ux_z_minus_, ux_z_plus_ = thm_sol.ux_last[cell_neig[2]], thm_sol.ux_last[cell_neig[5]]

        uy_x_minus_, uy_x_plus_ = thm_sol.uy_last[cell_neig[0]], thm_sol.uy_last[cell_neig[3]]
        uy_y_minus_, uy_y_plus_ = thm_sol.uy_last[cell_neig[1]], thm_sol.uy_last[cell_neig[4]]
        uy_z_minus_, uy_z_plus_ = thm_sol.uy_last[cell_neig[2]], thm_sol.uy_last[cell_neig[5]]

        uz_x_minus_, uz_x_plus_ = thm_sol.uz_last[cell_neig[0]], thm_sol.uz_last[cell_neig[3]]
        uz_y_minus_, uz_y_plus_ = thm_sol.uz_last[cell_neig[1]], thm_sol.uz_last[cell_neig[4]]
        uz_z_minus_, uz_z_plus_ = thm_sol.uz_last[cell_neig[2]], thm_sol.uz_last[cell_neig[5]]

        # distances between two neighbor cells in each direction
        dist_x = np.fabs(point_neig[0][0] - point_neig[3][0]) # X+ and X-
        dist_y = np.fabs(point_neig[1][1] - point_neig[4][1]) #
        dist_z = np.fabs(point_neig[2][2] - point_neig[5][2])

        dux_dx = deriv(ux_x_minus_, ux_x_plus_, dist_x, mult=mults[0])
        dux_dy = deriv(ux_y_minus_, ux_y_plus_, dist_y, mult=mults[1])
        dux_dz = deriv(ux_z_minus_, ux_z_plus_, dist_z, mult=mults[2])

        duy_dx = deriv(uy_x_minus_, uy_x_plus_, dist_x, mult=mults[0])
        duy_dy = deriv(uy_y_minus_, uy_y_plus_, dist_y, mult=mults[1])
        duy_dz = deriv(uy_z_minus_, uy_z_plus_, dist_z, mult=mults[2])

        duz_dx = deriv(uz_x_minus_, uz_x_plus_, dist_x, mult=mults[0])
        duz_dy = deriv(uz_y_minus_, uz_y_plus_, dist_y, mult=mults[1])
        duz_dz = deriv(uz_z_minus_, uz_z_plus_, dist_z, mult=mults[2])

        # Voight notation: 6 values for each fault point: 3 diagonal (xx, yy, zz) + 3 off-diagonal values (yz, xz, xy)
        strain = np.vstack(
            [dux_dx, duy_dy, duz_dz,
             0.5 * (duy_dz + duz_dy),
             0.5 * (dux_dz + duz_dx),
             0.5 * (dux_dy + duy_dx)])

        assert np.any(~np.isnan(strain))


        # volumetric_strain=div(displ)
        volumetric_strain = -(dux_dx + duy_dy + duz_dz)

        n_points = 1
        kronecker = np.array(n_points * [1, 1, 1, 0, 0, 0]).reshape(n_points, 6).transpose()
        stress = g.young * (-strain + g.poisson / (1 - 2 * g.poisson) *
                               volumetric_strain * kronecker) / (1 + g.poisson)

        return strain[0], strain[1], strain[2], stress[0], stress[1], stress[2]  # Sxx, Syy, Szz


    def get_proxy_displs(eval_points):  # Y,X,Z
        eps = 0  # [m], to avoid r=0 for the integral in the geomech proxy 1/r
        eval_points_eps = eval_points + eps
        #eval_points_eps = eval_points_eps.transpose()
        upx1, upy1, upz1, utx1, uty1, utz1 = g.calc_displacements_cpp(eval_points_eps, thm_sol.prisms_rsv, thm_sol.delta_pressure_rsv, thm_sol.delta_temperature_rsv)
        ux = upx1 + utx1
        uy = upy1 + uty1
        uz = upz1 + utz1
        return uy, ux, uz # thermoporoelastic displacements [m]

    def get_eval_points(mode, shift_x=0, shift_y=0):
        if mode == 'centers':
            # X<->Y
            eval_points = np.zeros_like(thm_sol.centroids)
            eval_points[:, 1] = thm_sol.centroids[:, 0] # X
            eval_points[:, 0] = thm_sol.centroids[:, 1]
            eval_points[:, 2] = thm_sol.centroids[:, 2]
        elif mode == 'vertical':
            # test with a vertical line
            z = np.unique(thm_sol.centroids[:, 2])
            eval_points = np.zeros((z.size, 3))
            eval_points[:, 1] = thm_sol.centroids[:, 0].mean() + shift_x
            eval_points[:, 0] = thm_sol.centroids[:, 1].mean() + shift_y
            eval_points[:, 2] = z
        return eval_points.transpose()

    def get_proxy_strain_stress(eval_points_):
        if len(eval_points_.shape) == 1:  # just one point (1d array)
            eval_points = np.zeros((3, 1))
            eval_points[0, :] = eval_points_[0]  # X<->Y
            eval_points[1, :] = eval_points_[1]
            eval_points[2, :] = eval_points_[2]
        else: # multiple points, 2d array
            eval_points = eval_points_
        # returns thermoporoelastic strain and  stress in MPa, (6, n_points), 6 - Voight notation
        eps = 0  # [m], to avoid r=0 for the integral in the geomech proxy 1/r
        eval_points_eps = eval_points + eps
        res = g.calc_strain_stress_cpp(eval_points_eps, thm_sol.prisms_rsv, thm_sol.delta_pressure_rsv, thm_sol.delta_temperature_rsv)
        stress_p, strain_p, stress_total_p, stress_t, strain_t, stress_total_t, stress, strain, stress_total = res
        #[Sp_xx, Sp_yy, Sp_zz, Sp_yz, Sp_xz, Sp_xy] = stress_p
        #[St_xx, St_yy, St_zz, St_yz, St_xz, St_xy] = stress_t
        return strain[1,:], strain[0,:], strain[2,:], \
            stress[1,:], stress[0,:], stress[2,:], stress_total[1,:], stress_total[0,:], stress_total[2,:]  # xx yy zz


    def compare_vert_line(points, suffix='', loc='', output_folder='.', array_names={}):
        z_range = points[2,:]

        array_dict = {'dp': thm_sol.delta_pressure, 'dt': thm_sol.delta_temperature}
        array_dict_interp = get_thm_by_interp(array_dict, points[1,:], points[0,:], points[2,:], method='linear') # obtain thm solutiona at points using interpolation
        dp = array_dict_interp['dp']
        dt = array_dict_interp['dt']

        ux_prx, uy_prx, uz_prx = get_proxy_displs(points)
        qx_prx, qy_prx, qz_prx, sx_prx, sy_prx, sz_prx, sx_total_prx, sy_total_prx, sz_total_prx =  get_proxy_strain_stress(points)

        # get total from effective stress
        if False: # to avoid smoothing at the rsv boundaries
            dp_z = np.zeros_like(z_range)
            z_range_rsv = reduce(np.logical_and, [z_range > 2100, z_range < 2200])
            dp_z[z_range_rsv] = dp.max() #m.idata.other.delta_p #MPa
        sx_total_prx = sx_prx + m.idata.rock.biot * dp
        sy_total_prx = sy_prx + m.idata.rock.biot * dp
        sz_total_prx = sz_prx + m.idata.rock.biot * dp

        plot_thm2 = False

        n_points = points.shape[1]
        ux_thm = np.zeros(n_points); uy_thm = np.zeros(n_points); uz_thm = np.zeros(n_points);
        qx_thm = np.zeros(n_points); qy_thm = np.zeros(n_points); qz_thm = np.zeros(n_points);
        sx_thm = np.zeros(n_points); sy_thm = np.zeros(n_points); sz_thm = np.zeros(n_points);
        sx_total_thm = np.zeros(n_points); sy_total_thm = np.zeros(n_points); sz_total_thm = np.zeros(n_points);
        qx_thm2 = np.zeros(n_points); qy_thm2 = np.zeros(n_points); qz_thm2 = np.zeros(n_points);
        sx_thm2 = np.zeros(n_points); sy_thm2 = np.zeros(n_points); sz_thm2 = np.zeros(n_points);

        for i in range(n_points):  # use XY from point and different Z
            point = np.array([points[0, i], points[1, i], points[2, i]])  # YXZ - > XYZ
            ux_thm[i], uy_thm[i], uz_thm[i] = get_thm_displs(point)
            if 'strain' in array_names:
                qx_thm[i], qy_thm[i], qz_thm[i] = get_thm_strain(point)
            if 'delta_eff_stress_z' in array_names:
                sx_thm[i], sy_thm[i], sz_thm[i] = get_thm_stress(point)
            if 'delta_total_stress_z' in array_names:
                sx_total_thm[i], sy_total_thm[i], sz_total_thm[i] = get_thm_total_stress(point)

            if ('strain' in array_names or 'delta_eff_stress_z' in array_names) and plot_thm2:
                qx_thm2[i], qy_thm2[i], qz_thm2[i], \
                sx_thm2[i], sy_thm2[i], sz_thm2[i] = get_thm_stress_by_deriv(point)

        for array_name in array_names:
            match array_name:
                case 'displ_z': prx = uz_prx * m2mm; thm = uz_thm * m2mm; s = 'Vertical displacement, mm.' + ' at ' + loc
                case 'displ_y': prx = uy_prx * m2mm; thm = uy_thm * m2mm; s = 'Horizontal displacement (Y), mm.' + ' at ' + loc
                case 'displ_x': prx = ux_prx * m2mm; thm = ux_thm * m2mm; s = 'Horizontal displacement (X), mm.' + ' at ' + loc

                case 'strain_z': prx = qz_prx; thm = qz_thm; thm2 = qz_thm2; s = 'Vertical strain' + ' at ' + loc
                case 'strain_y': prx = qy_prx; thm = qy_thm; thm2 = qy_thm2; s = 'Horizontal strain (YY)' + ' at ' + loc
                case 'strain_x': prx = qx_prx; thm = qx_thm; thm2 = qx_thm2; s = 'Horizontal strain (XX)' + ' at ' + loc

                case 'delta_eff_stress_z': prx = sz_prx; thm = sz_thm; thm2 = sz_thm2; s = 'Vertical effective stress change, MPa.' + ' at ' + loc
                case 'delta_eff_stress_y': prx = sy_prx; thm = sy_thm; thm2 = sy_thm2; s = 'Horizontal effective stress change (YY), MPa.' + ' at ' + loc
                case 'delta_eff_stress_x': prx = sx_prx; thm = sx_thm; thm2 = sx_thm2; s = 'Horizontal effective stress change (XX), MPa.' + ' at ' + loc

                case 'delta_total_stress_z': prx = sz_total_prx; thm = sz_total_thm; thm2 = sz_thm2; s = 'Vertical total stress change, MPa.' + ' at ' + loc
                case 'delta_total_stress_y': prx = sy_total_prx; thm = sy_total_thm; thm2 = sy_thm2; s = 'Horizontal total stress change (YY), MPa.' + ' at ' + loc
                case 'delta_total_stress_x': prx = sx_total_prx; thm = sx_total_thm; thm2 = sx_thm2; s = 'Horizontal total stress change (XX), MPa.' + ' at ' + loc

                case 'delta_pressure': prx = None; thm = dp; s = 'Pressure change, MPa.' + ' at ' + loc
                case 'delta_temperature': prx = None; thm = dt; s = 'Temperature change, K.' + ' at ' + loc
                #delta_temperature

            plot_mesh_layers = False
            if plot_mesh_layers:
                for zi in m.reservoir.Zc:
                    if z_range.min() <= zi <= z_range.max():
                        plt.axhline(y=zi, color='gray', linestyle='dotted')

            plt.axhline(y=m.idata.other.rsv_top, color='black', linestyle='dotted')#, xmin=0.95, xmax=1.0)
            plt.axhline(y=m.idata.other.rsv_bottom, color='black', linestyle='dotted', label=r'reservoir top/bottom')#, xmin=0.95, xmax=1.0)
            plt.plot(thm, z_range, label='THM', c='blue')#, marker='.')
            if prx is not None:
                plt.plot(prx, z_range, label='Proxy', linestyle='--', c='red')#, marker='.')

            if ('stress' in array_names or 'strain' in array_names) and plot_thm2:
                plt.plot(thm2, z_range, label=array_name + '_THM2', color='black')#marker='.',

            # for comparizon with analytical solution laterally infinite rsv
            #if mode == 'delta_total_stress_z' and np.fabs(prx.max()) < 0.05 and np.fabs(thm_sol.max()) < 0.05: # vertical stress is almost zero
            #    plt.xlim(-0.25, 0.25)

            plt.gca().invert_yaxis()
            plt.xlabel(s)
            plt.title(s)
            plt.ylabel('Depth, m.')
            plt.legend()
            #plt.grid()
            plt.minorticks_on()
            plt.grid(which='major', linestyle='-', linewidth=0.8)
            plt.grid(which='minor', linestyle=':', linewidth=0.5)
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, array_name + '_' + loc + '_' + suffix + '.png'))
            plt.close()

            # plot the difference THM solution vs Proxy
            if prx is not None:
                diff = thm - prx
                with np.errstate(divide='ignore', invalid='ignore'):
                    rel_diff = np.where(np.abs(thm) > 0, (diff / np.abs(thm)) * 100.0, np.nan)
                fig, ax1 = plt.subplots()
                ax1.axhline(y=m.idata.other.rsv_top, color='black', linestyle='dotted')
                ax1.axhline(y=m.idata.other.rsv_bottom, color='black', linestyle='dotted', label=r'reservoir top/bottom')
                ax1.plot(diff, z_range, c='green', label='abs. diff')
                ax1.invert_yaxis()
                ax1.set_xlabel('Absolute diff. ' + s)
                ax1.set_title(s)
                ax1.set_ylabel('Depth, m.')
                ax1.minorticks_on()
                ax1.grid(which='major', linestyle='-', linewidth=0.8)
                ax1.grid(which='minor', linestyle=':', linewidth=0.5)
                if True:  # plot only abs. diff
                    ax1.legend()
                else:  # plot also rel.diff
                    ax2 = ax1.twiny()
                    ax2.plot(rel_diff, z_range, c='orange', linestyle='--', label='rel. diff')
                    ax2.set_xlabel('Relative diff., %')
                    lines1, labels1 = ax1.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax1.legend(lines1 + lines2, labels1 + labels2)
                fig.tight_layout()
                fig.savefig(os.path.join(output_folder, array_name + '_' + loc + '_' + suffix + '_diff'+ '.png'))
                plt.close(fig)


    def plot_contour(array_dict, points_x, points_y, output_folder, layer=0, slice='XY', vlims=None, n_levels=12, idata=None):
        # plot contours XY plane, 1 layer by z
        # vlims: optional dict {arr_name: (vmin, vmax)} to fix colorbar range
        _default_w, _default_h = plt.rcParams['figure.figsize']
        for arr_name, arr in array_dict.items():
            if slice == 'XZ':
                plt.figure(figsize=(_default_w, _default_h * 2))
            if len(arr.shape) == 3:
                arr_layer = arr[:, :, layer]
            else:
                arr_layer = arr
            if vlims is not None and arr_name in vlims:
                vmin, vmax = vlims[arr_name]
            else:
                vmin, vmax = arr_layer.min(), arr_layer.max()
            if np.isclose(vmin, vmax):
                vmin, vmax = vmin - 0.01, vmax + 0.01
            levels = np.linspace(vmin, vmax, n_levels)
            if vmin < 0 < vmax:
                levels = np.sort(np.unique(np.append(levels, 0.)))
            cs = plt.contourf(points_x, points_y, np.ma.masked_invalid(arr_layer), levels=levels, vmin=vmin, vmax=vmax)
            plt.colorbar(cs, orientation='horizontal', pad=0.12)
            #plt.colorbar(cs, extend='neither').set_ticks([vmin, vmax])
            plt.gca().set_aspect(2) # 2X vertical scale
            #plt.gca().set_aspect('equal')
            plt.minorticks_on()
            plt.xlabel('X, m.')
            if slice == 'XY':
                plt.ylabel('Y')
            elif slice == 'XZ':
                plt.ylabel('Depth, m.')
                plt.gca().invert_yaxis()
                if idata is not None:
                    plt.axhline(y=idata.other.rsv_top,    color='black', linewidth=0.8, linestyle='--')
                    plt.axhline(y=idata.other.rsv_bottom, color='black', linewidth=0.8, linestyle='--', label='reservoir top/bottom')
                    plt.axvline(x=idata.other.rsv_x1, color='black', linewidth=0.5, linestyle=':')
                    plt.axvline(x=idata.other.rsv_x2, color='black', linewidth=0.5, linestyle=':', label='reservoir sides')
                    z_well_top = points_y.min()
                    z_well_bot = idata.other.rsv_bottom
                    if wells_type in ('prod', 'doublet'):
                        plt.plot([idata.other.prod_well_coords[0]] * 2, [z_well_top, z_well_bot],
                                 color='red',  linewidth=1.5, label='production well')
                    if wells_type in ('inj', 'doublet'):
                        plt.plot([idata.other.inj_well_coords[0]] * 2,  [z_well_top, z_well_bot],
                                 color='cyan', linewidth=1.5, label='injection well')
                    plt.legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, 0.15), bbox_transform=plt.gcf().transFigure, ncol=2)
            parts = arr_name.split(' - ', 1)
            plt.title('\n'.join(parts) if len(parts) > 1 else arr_name)
            fig_fname = arr_name + '_contour.png'
            fig_fname = fig_fname.replace(' ', '_')
            fig_path = os.path.join(output_folder, fig_fname)
            print(f'Saving {fig_fname}, min={arr_layer.min():.4g}, max={arr_layer.max():.4g}')
            plt.savefig(fig_path, bbox_inches='tight')
            plt.close()


    def plot_imshow(array_dict, points_x, points_y, output_folder, layer=0, slice='XY'):
        # plot contours XY plane, 1 layer by z
        for arr_name, arr in array_dict.items():
            if len(arr.shape) == 3:
                arr_layer = arr[:, :, layer]
            else:
                arr_layer = arr
            cs = plt.imshow(arr_layer)
            plt.colorbar(cs)
            plt.gca().set_aspect('equal')
            plt.xlabel('I')
            if slice == 'XY':
                plt.ylabel('J')
            elif slice == 'XZ':
                plt.ylabel('K')
                plt.gca().invert_yaxis()
            plt.title(arr_name)
            plt.savefig(os.path.join(output_folder, arr_name + '_imshow.png'))
            plt.close()

    def plot_mesh_skeleton(output_folder):
        Xc = m.idata.other.Xc
        Zc = m.idata.other.Zc
        rsv_top = m.idata.other.rsv_top
        rsv_bottom = m.idata.other.rsv_bottom

        Xc_plot = Xc[(Xc >= -2100) & (Xc <= 2100)]
        Zc_plot = Zc[(Zc >= rsv_top - 500) & (Zc <= rsv_bottom + 500)]

        rsv_xy = m.idata.other.rsv_xy

        fig, ax = plt.subplots(figsize=(8, 6))
        for xi in Xc_plot:
            ax.axvline(x=xi, color='steelblue', linewidth=0.7, zorder=1)
        for zi in Zc_plot:
            ax.axhline(y=zi, color='coral', linewidth=0.7, zorder=1)
        # reservoir boundary rectangle (dashed red, lines clipped to their crossing points)
        from matplotlib.patches import Rectangle
        rect = Rectangle((-rsv_xy, rsv_top), 2 * rsv_xy, rsv_bottom - rsv_top,
                         linewidth=2., edgecolor='red', linestyle='--', facecolor='none', zorder=3,
                         label='reservoir boundary')
        ax.add_patch(rect)
        # wells (vertical lines), same as in plot_contour XZ slice
        z_well_top = Zc_plot.min()
        z_well_bot = rsv_bottom
        if wells_type in ('prod', 'doublet'):
            ax.plot([m.idata.other.prod_well_coords[0]] * 2, [z_well_top, z_well_bot],
                    color='red', linewidth=1.5, label='production well', zorder=4)
        if wells_type in ('inj', 'doublet'):
            ax.plot([m.idata.other.inj_well_coords[0]] * 2, [z_well_top, z_well_bot],
                    color='darkblue', linewidth=1.5, label='injection well', zorder=4)
        # legend (reservoir boundary is always drawn; wells added when present)
        ax.legend(fontsize=8, loc='upper right')
        ax.set_xlim(Xc_plot.min(), Xc_plot.max())
        ax.set_ylim(Zc_plot.min(), Zc_plot.max())
        ax.invert_yaxis()
        ax.set_xlabel('X, m.')
        ax.set_ylabel('Depth, m.')
        ax.set_title('Mesh skeleton')
        ax.set_aspect('auto')
        fig.tight_layout()
        fig.savefig(os.path.join(output_folder, 'mesh_skeleton.png'))
        plt.close(fig)
        print('Saved mesh_skeleton.png')

    def save_html_prx_vs_thm(base_names, output_folder, filename='proxy_vs_thm_2d.html',
                              locs=None, array_names=None, suffix_1d='all'):
        # dp + dt contours at the top
        contour_cells = []
        for fname, title in [('dp_contour.png', '&#916;P contour (XZ)'), ('dt_contour.png', '&#916;T contour (XZ)')]:
            if os.path.exists(os.path.join(output_folder, fname)):
                contour_cells.append(f'<td style="text-align:center"><b>{title}</b><br>'
                                     f'<img src="{fname}" style="max-width:100%"></td>')
        contour_html = ''
        if contour_cells:
            contour_html = (f'<h2>&#916;P / &#916;T contours (XZ slice)</h2>'
                            f'<table><tr>{"".join(contour_cells)}</tr></table>')

        # 2D proxy vs THM table
        html_rows = []
        for b in base_names:
            prx_file  = f'{b} - Proxy_contour.png'.replace(' ', '_')
            thm_file  = f'{b} - THM_contour.png'.replace(' ', '_')
            diff_file = f'{b} - Difference_contour.png'.replace(' ', '_')
            html_rows.append(f'''
              <tr>
                <td style="text-align:center"><img src="{prx_file}" style="max-width:100%"></td>
                <td style="text-align:center"><img src="{thm_file}" style="max-width:100%"></td>
                <td style="text-align:center"><img src="{diff_file}" style="max-width:100%"></td>
              </tr>''')

        # 1D plots: rows=array_names, columns=locations in fixed order
        col_order = ['(250,250)', 'inj_well', 'prod_well']
        active_locs = [l for l in col_order if locs and l in locs]
        plots_1d_html = ''
        if active_locs and array_names:
            th_locs = ''.join(f'<th>{l}</th>' for l in active_locs)
            rows_1d = []
            for array_name in array_names:
                cells = []
                for loc in active_locs:
                    plot_file = f'{array_name}_{loc}_{suffix_1d}.png'
                    if os.path.exists(os.path.join(output_folder, plot_file)):
                        cells.append(f'<td style="text-align:center">'
                                     f'<img src="{plot_file}" style="max-width:100%"></td>')
                    else:
                        cells.append('<td style="text-align:center;color:gray">N/A</td>')
                rows_1d.append(f'<tr><td style="white-space:nowrap"><b>{array_name}</b></td>{"".join(cells)}</tr>')
            plots_1d_html = f'''
        <h2>1D vertical profiles</h2>
        <table border="1" cellspacing="4" cellpadding="4" style="width:100%">
          <tr><th>Mode</th>{th_locs}</tr>
          {''.join(rows_1d)}
        </table>'''

        html = f'''<!DOCTYPE html>
        <html>
        <head><meta charset="utf-8"><title>Proxy vs THM Report</title></head>
        <body>
        {contour_html}
        <h2>Proxy vs THM 2D slices (XZ)</h2>
        <table border="1" cellspacing="4" cellpadding="4">
          <tr><th>Proxy</th><th>THM</th><th>Difference (THM - Proxy)</th></tr>
          {''.join(html_rows)}
        </table>
        {plots_1d_html}
        </body>
        </html>'''
        html_path = os.path.join(output_folder, filename)
        with open(html_path, 'w') as f:
            f.write(html)
        print(f'Saved HTML report: {html_path}')

    def save_pdf_report(base_names, output_folder, filename='proxy_vs_thm_report.pdf',
                        locs=None, array_names=None, suffix_1d='all', read_from_cache=False):
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.image as mpimg

        def load_img(img_name):
            if not img_name:
                return None
            img_path = os.path.join(output_folder, img_name)
            return mpimg.imread(img_path) if os.path.exists(img_path) else None

        def add_ncol_page(pdf, names_labels, row_title='', col_fontsize=9, title_fontsize=10):
            # names_labels: list of (img_name, col_title); missing files are skipped
            loaded = [(load_img(n), lbl) for n, lbl in names_labels]
            loaded = [(im, lbl) for im, lbl in loaded if im is not None]
            if not loaded:
                return
            n = len(loaded)
            fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
            if n == 1:
                axes = [axes]
            for ax, (im, lbl) in zip(axes, loaded):
                ax.imshow(im)
                ax.axis('off')
                ax.set_title(lbl, fontsize=col_fontsize)
            if row_title:
                fig.suptitle(row_title, fontsize=title_fontsize, fontweight='bold')
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        col_order = ['(250,250)', 'inj_well', 'prod_well']
        active_locs = [l for l in col_order if locs and l in locs]

        pdf_path = os.path.join(output_folder, filename)
        with PdfPages(pdf_path) as pdf:
            # dp + dt contours
            add_ncol_page(pdf,
                [('dp_contour.png', 'ΔP contour (XZ)'), ('dt_contour.png', 'ΔT contour (XZ)')],
                row_title='ΔP / ΔT contours (XZ slice)', col_fontsize=7, title_fontsize=8)
            # 2D proxy vs THM: 3 columns per page
            for b in base_names:
                add_ncol_page(pdf,
                    [(f'{b} - Proxy_contour.png'.replace(' ', '_'), 'Proxy'),
                     (f'{b} - THM_contour.png'.replace(' ', '_'), 'THM'),
                     (f'{b} - Difference_contour.png'.replace(' ', '_'), 'Difference (THM-Proxy)')],
                    row_title=b, col_fontsize=7, title_fontsize=8)
            # 1D profiles: one page per array_name, columns = locations
            if active_locs and array_names:
                for array_name in array_names:
                    add_ncol_page(pdf,
                        [(f'{array_name}_{loc}_{suffix_1d}.png', loc) for loc in active_locs],
                        row_title=array_name)
        print(f'Saved PDF report: {pdf_path}')


    # mesh skeleton with wells (independent of modes / proxy computation)
    plot_mesh_skeleton(output_folder)

    # reference points for the 1D vertical profiles; taken from idata.other.points_xy
    # (set in examples/base.py) as a list of [x, y, label]
    points_xy = dict()
    for x_pt, y_pt, label in m.idata.other.points_xy:  # the order is actually Y,X
        points_xy[label] = [y_pt, x_pt]

    if True: # evaluate along the wells
        if wells_type in ['prod', 'doublet']:
            points_xy['prod_well'] = m.idata.other.prod_well_coords[:2]  # skip z coord
        if wells_type in ['inj', 'doublet']:
            points_xy['inj_well'] = m.idata.other.inj_well_coords[:2]

    base_names = [] # for html/pdf reports

    # compute with proxy for 2D slice
    if 'plot_2d_slices' in modes:
        points_x = np.unique(g.centroids[:, 0])
        points_y = np.array([0.])
        points_z = np.unique(g.centroids[:, 2])

        if '2d_slices_41_71' in modes: # while reading a coarser grid as sources, evaluate at finer grid centers
            # replace points_x by ones from the finer grid
            # Xc from nx == 71 case in model.py: # rsv corners and near-well (middle) are refined
            Xc_left = np.array([-8000,-6000,-5000,-4000,-3000,-2500,-2000,-1600,-1400,-1200] +
                          [-1100, -1050, -1030, -1010, -1000,  -990,  -980, -950, -900] +
                          np.arange(-800, -100, 100).tolist() +
                          np.arange(-100, 0, 10).tolist())
            Xc = np.hstack([Xc_left, -Xc_left[::-1]]) # add the right part symmetrically
            points_x = (Xc[1:] + Xc[:-1]) * 0.5 # centers

        # cut points far from the reservoir for better zoom in plots
        points_x = points_x[reduce(np.logical_and, [points_x > -5000., points_x < 5000.])]
        points_z = points_z[reduce(np.logical_and, [points_z > 1400., points_z < 3400.])]

        points_x_3d, points_y_3d, points_z_3d = np.meshgrid(points_x, points_y, points_z)

        print('plotting 2D slices, n_eval_points =', points_x.size * points_y.size * points_z.size, 'n_src_points = ', g.centroids[:, 0].size)
        p_nx, p_ny, p_nz = points_x.size, points_y.size, points_z.size
        points = np.zeros((3, points_x_3d.size))
        points[0, :], points[1, :], points[2, :] = points_x_3d.flatten(), points_y_3d.flatten(), points_z_3d.flatten()

        dp = gd((g.centroids[:, 1], g.centroids[:, 0], g.centroids[:, 2]), \
            thm_sol.delta_pressure, (points[1, :], points[0, :], points[2, :]), method='nearest', fill_value=0.).reshape((p_nx, p_ny, p_nz))
        array_dict = {'Pressure change, MPa':dp[:,0,:].transpose()}
        if thermal:
            dt = gd((g.centroids[:, 1], g.centroids[:, 0], g.centroids[:, 2]), \
                thm_sol.delta_temperature, (points[1, :], points[0, :], points[2, :]), method='nearest', fill_value=0.).reshape((p_nx, p_ny, p_nz))
            array_dict.update({'Temperature change, K':dt[:,0,:].transpose()})
        plot_contour(array_dict, points_x, points_z, output_folder=output_folder, slice='XZ', idata=m.idata)
        #plot_imshow(array_dict, points_x, points_z, output_folder=output_folder, slice = 'XZ')

        # if caching is requested but the pkl is missing, fall back to recomputing the proxy
        pkl_path = os.path.join(output_folder, "displs_stresses_prx.pkl")
        if read_from_cache and not os.path.exists(pkl_path):
            print(f'read_from_cache=True but "{pkl_path}" not found; '
                  f'forcing read_from_cache=False (recomputing proxy)')
            read_from_cache = False

        if not read_from_cache: # run proxy and save PKLs
            print('computing proxy...')
            ux_prx, uy_prx, uz_prx = get_proxy_displs(points)
            ux_prx = ux_prx.reshape((p_nx, p_ny, p_nz))
            uy_prx = uy_prx.reshape((p_nx, p_ny, p_nz))
            uz_prx = uz_prx.reshape((p_nx, p_ny, p_nz))

            _, _, _, sxx_prx, syy_prx, szz_prx, sxx_total_prx, syy_total_prx, szz_total_prx = get_proxy_strain_stress(points)  # need to X<->Y if non-symmetric
            sxx_prx = sxx_prx.reshape((p_nx, p_ny, p_nz))
            syy_prx = syy_prx.reshape((p_nx, p_ny, p_nz))
            szz_prx = szz_prx.reshape((p_nx, p_ny, p_nz))
            sxx_total_prx = sxx_total_prx.reshape((p_nx, p_ny, p_nz))
            syy_total_prx = syy_total_prx.reshape((p_nx, p_ny, p_nz))
            szz_total_prx = szz_total_prx.reshape((p_nx, p_ny, p_nz))

            # save to pkl
            import pickle
            data = {'ux_prx': ux_prx, 'uy_prx': uy_prx, 'uz_prx': uz_prx,
                    'sxx_prx': sxx_prx, 'syy_prx': syy_prx, 'szz_prx': szz_prx,
                    'sxx_total_prx': sxx_total_prx, 'syy_total_prx': syy_total_prx, 'szz_total_prx': szz_total_prx}
            with open(pkl_path, "wb") as f:
                pickle.dump(data, f)
        else: # do not rerun proxy, read from PKl files (if only plotting is changed)
            import pickle
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            ux_prx = data['ux_prx']
            uy_prx = data['uy_prx']
            uz_prx = data['uz_prx']
            sxx_prx = data['sxx_prx']
            syy_prx = data['syy_prx']
            szz_prx = data['szz_prx']
            sxx_total_prx = data['sxx_total_prx']
            syy_total_prx = data['syy_total_prx']
            szz_total_prx = data['szz_total_prx']

        array_dict = {'Horizontal displacements (X), mm. - Proxy' : ux_prx[:,0,:].transpose() * m2mm,
                      'Vertical displacements, mm. - Proxy' : uz_prx[:,0,:].transpose() * m2mm,
                      'Horizontal effective stress change (XX), MPa - Proxy' : sxx_prx[:,0,:].transpose(),
                      'Vertical effective stress change, MPa - Proxy' : szz_prx[:,0,:].transpose(),
                      'Horizontal total stress change (XX), MPa - Proxy' : sxx_total_prx[:,0,:].transpose(),
                      'Vertical total stress change, MPa - Proxy' : szz_total_prx[:,0,:].transpose()
                      }
        shared_cbar = True  # use same colorbar min/max for prx and thm plots of the same variable

        # compare proxy vs thm on finer mesh although coarser mesh data used as the input for proxy
        if '2d_slices_41_71' in modes:
            folder_71 = folder.replace('41_41_66', '71_71_66')
            thm_sol = read_thm_solution_from_vtk(m, folder=folder_71, timestep=timestep)

        # THM 2D plots on the same grid
        thm_raw = {'Horizontal displacements (X), mm. - THM': thm_sol.ux_last * m2mm,
                   'Vertical displacements, mm. - THM': thm_sol.uz_last * m2mm,
                   'Horizontal effective stress change (XX), MPa - THM': thm_sol.delta_Sxx_last,
                   'Vertical effective stress change, MPa - THM': thm_sol.delta_Szz_last,
                   'Horizontal total stress change (XX), MPa - THM': thm_sol.delta_total_Sxx_last,
                   'Vertical total stress change, MPa - THM': thm_sol.delta_total_Szz_last}
        thm_interp_method = 'nearest' # nearest is better here as eval points are centroids
        #thm_interp_method = 'linear' #
        thm_interp = get_thm_by_interp(thm_raw, points[1, :], points[0, :], points[2, :], method=thm_interp_method)
        array_dict_thm = {k: v.reshape((p_nx, p_ny, p_nz))[:, 0, :].transpose()
                          for k, v in thm_interp.items()}

        base_names = ['Horizontal displacements (X), mm.']
        base_names += ['Vertical displacements, mm.']
        base_names += ['Horizontal effective stress change (XX), MPa']
        base_names += ['Vertical effective stress change, MPa']
        base_names += ['Horizontal total stress change (XX), MPa']
        base_names += ['Vertical total stress change, MPa']
        if shared_cbar:
            vlims_prx = {f'{b} - Proxy': (min(array_dict[f'{b} - Proxy'].min(),  array_dict_thm[f'{b} - THM'].min()),
                                      max(array_dict[f'{b} - Proxy'].max(), array_dict_thm[f'{b} - THM'].max()))
                         for b in base_names}
            vlims_thm = {f'{b} - THM': vlims_prx[f'{b} - Proxy'] for b in base_names}
        else:
            vlims_prx = vlims_thm = None

        plot_contour(array_dict, points_x, points_z, output_folder=output_folder, slice='XZ', vlims=vlims_prx, idata=m.idata)
        #plot_imshow(array_dict, points_x, points_z, output_folder=output_folder, slice = 'XZ')

        plot_contour(array_dict_thm, points_x, points_z, output_folder=output_folder, slice='XZ', vlims=vlims_thm, idata=m.idata)

        print('Array ranges:')
        for b in base_names:
            prx = array_dict[f'{b} - Proxy']
            thm = array_dict_thm[f'{b} - THM']
            print(f'  {b}  Proxy [{prx.min():.4g}, {prx.max():.4g}]  THM [{thm.min():.4g}, {thm.max():.4g}]')

        # THM - Proxy difference 2D plots
        diff_clip = None
        #diff_clip = 0.1  # nullify stress differences larger than this value as they affect the axis range but located in very small vicinity
        array_dict_diff = {}
        for b in base_names:
            d = array_dict_thm[f'{b} - THM'] - array_dict[f'{b} - Proxy']
            if diff_clip is not None and 'stress' in b:
                d = np.where(np.abs(d) <= diff_clip, d, 0.)
            #rd = np.where(np.abs(array_dict_thm[f'{b}_thm']) > 0, (d / np.abs(array_dict_thm[f'{b}_thm'])) * 100., 0.)
            array_dict_diff[f'{b} - Difference'] = d
            #array_dict_diff[f'{b} - Relative Difference'] = rd
        plot_contour(array_dict_diff, points_x, points_z, output_folder=output_folder, slice='XZ', idata=m.idata)

        if False:
            print('Relative difference THM vs Proxy (% of |THM|):')
            for b in base_names:
                thm_arr = array_dict_thm[f'{b} - THM'].ravel()
                diff_arr = array_dict_diff[f'{b} - Difference'].ravel()
                with np.errstate(divide='ignore', invalid='ignore'):
                    rel = np.where(np.abs(thm_arr) > 0, np.abs(diff_arr / thm_arr) * 100., np.nan)
                n_total = np.sum(~np.isnan(rel))
                for thr in [1., 5., 10.]:
                    n_over = np.sum(rel > thr)
                    print(f'  {b}: >{thr:.0f}%: {n_over}/{n_total} ({100.*n_over/n_total if n_total>0 else 0:.1f}%)')

        pass  # HTML/PDF reports are saved after 1D plots are generated (see below)

    if 'plot_2d_thm_41_vs_71' in modes:
        from model import Model
        m_41 = Model(model_folder='41_41_66', physics_type=physics_type, uniform_props=False,
                     wells_type=wells_type, decouple_geomech=True, generate_mesh=False, dummy='yes')
        m_41.set_input_data()

        folder_41 = os.path.join('results', 'sol_cpp_' + physics_type + '_' + wells_type + '_41_41_66')
        folder_71 = os.path.join('results', 'sol_cpp_' + physics_type + '_' + wells_type + '_71_71_66')

        thm_sol_41 = read_thm_solution_from_vtk(m_41, folder=folder_41, timestep=timestep)
        thm_sol_71 = read_thm_solution_from_vtk(m, folder=folder_71, timestep=timestep)

        # eval grid from 71 centroids (finer), same x/z filter as plot_2d_slices
        pts_x = np.unique(thm_sol_71.centroids[:, 0])  # follows existing naming convention
        pts_y = np.array([0.])
        pts_z = np.unique(thm_sol_71.centroids[:, 2])
        pts_x = pts_x[reduce(np.logical_and, [pts_x > -5000., pts_x < 5000.])]
        pts_z = pts_z[reduce(np.logical_and, [pts_z > 1400., pts_z < 3400.])]

        p_nx, p_ny, p_nz = pts_x.size, pts_y.size, pts_z.size
        px3, py3, pz3 = np.meshgrid(pts_x, pts_y, pts_z)
        pts = np.zeros((3, px3.size))
        pts[0, :], pts[1, :], pts[2, :] = px3.flatten(), py3.flatten(), pz3.flatten()

        def interp_thm_dp(thm_s):
            return gd(
                (thm_s.centroids[:, 1], thm_s.centroids[:, 0], thm_s.centroids[:, 2]),
                thm_s.delta_pressure,
                (pts[1, :], pts[0, :], pts[2, :]),
                method='linear', fill_value=0.
            ).reshape((p_nx, p_ny, p_nz))

        dp_41 = interp_thm_dp(thm_sol_41)
        dp_71 = interp_thm_dp(thm_sol_71)

        vmin = min(dp_41.min(), dp_71.min())
        vmax = max(dp_41.max(), dp_71.max())
        vlims_cmp = {'delta_pressure, MPa - THM 41': (vmin, vmax),
                     'delta_pressure, MPa - THM 71': (vmin, vmax)}
        plot_contour({'delta_pressure, MPa - THM 41': dp_41[:, 0, :].transpose()},
                     pts_x, pts_z, output_folder=output_folder, slice='XZ', vlims=vlims_cmp, idata=m.idata)
        plot_contour({'delta_pressure, MPa - THM 71': dp_71[:, 0, :].transpose()},
                     pts_x, pts_z, output_folder=output_folder, slice='XZ', vlims=vlims_cmp, idata=m.idata)
        plot_contour({'delta_pressure, MPa - Difference (71-41)': (dp_71 - dp_41)[:, 0, :].transpose()},
                     pts_x, pts_z, output_folder=output_folder, slice='XZ', idata=m.idata)

    array_names = []
    if 'plot_vertic_line' in modes: # 1D plots (along vertical lines at points_xy)
        for k in points_xy.keys():
            point_xy = points_xy[k]
            print('1D plots for point', k, 'YX=', point_xy)
            array_names += ['delta_pressure']
            array_names += ['delta_temperature']
            array_names += ['displ_z', 'displ_y', 'displ_x']
            array_names += ['delta_eff_stress_z', 'delta_eff_stress_y', 'delta_eff_stress_x']
            array_names += ['delta_total_stress_z', 'delta_total_stress_y', 'delta_total_stress_x']
            #array_names += ['strain_z', 'strain_y', 'strain_x']
            #array_names = ['strain_x']  # debug
            #z_min = 0.
            #z_max = thm_sol.centroids[:, 2].max() #+ 1000.
            z_step = 5.  # m.
            #z_range_all = np.arange(z_min, z_max+1., z_step)# more points
            z_range_all = m.idata.other.Zc

            z_interp_eps = 5. # m # remove points close to rsv boundary, as they create descrepancies even with 'nearest' interpolation
            z_range_all_filter = reduce(np.logical_and, [np.fabs(z_range_all - m.idata.other.rsv_top) > z_interp_eps, np.fabs(z_range_all - m.idata.other.rsv_bottom) > z_interp_eps])
            z_range_all = z_range_all[z_range_all_filter]
            n_points = z_range_all.size
            points_all = np.zeros((3, n_points))
            points_all[0, :] = point_xy[0]  # X<->Y
            points_all[1, :] = point_xy[1]
            points_all[2, :] = z_range_all

            compare_vert_line(points_all, suffix='all', loc=k, output_folder=output_folder, array_names=array_names)

            if False: # zoomed in plot for the rsv part, if its thickness is quite small
                z_range_rsv = np.arange(m.idata.other.rsv_top-100., m.idata.other.rsv_bottom+100., z_step)
                # remove points close to rsv boundary, as they create descrepancies even with 'nearest' interpolation
                z_range_rsv_filter = reduce(np.logical_and, [np.fabs(z_range_rsv - m.idata.other.rsv_top) > z_interp_eps, np.fabs(z_range_rsv - m.idata.other.rsv_bottom) > z_interp_eps])
                z_range_rsv = z_range_rsv[z_range_rsv_filter]
                n_points = z_range_rsv.size
                points_rsv = np.zeros((3, n_points))
                points_rsv[0, :] = point_xy[1] # X<->Y
                points_rsv[1, :] = point_xy[0]
                points_rsv[2, :] = z_range_rsv
                compare_vert_line(points_rsv, suffix='rsv', loc=k, output_folder=output_folder, array_names=array_names)

    save_html_prx_vs_thm(base_names, output_folder=output_folder,
                         locs=list(points_xy.keys()), array_names=array_names, suffix_1d='all')
    save_pdf_report(base_names, output_folder=output_folder,
                    locs=list(points_xy.keys()), array_names=array_names, suffix_1d='all', read_from_cache=read_from_cache)

    # print vert displs and stresses change at a point
    if 'print_at_point' in modes:
        point = np.array([0., 0., 0.5*(m.idata.other.rsv_bottom + m.idata.other.rsv_top)]) # middle rsv depth

        ux_thm, uy_thm, uz_thm = get_thm_displs(point)
        ux_prx, uy_prx, uz_prx = get_proxy_displs(point)# need to X<->Y
        print('Compare at the point=', point)
        print('\tTHM   ', 'ux=', fmt(ux_thm*m2mm), 'uy=', fmt(uy_thm*m2mm), 'uz=', fmt(uz_thm*m2mm), 'mm.')
        print('\tProxy ', 'ux=', fmt(ux_prx[0]*m2mm), 'uy=', fmt(uy_prx[0]*m2mm), 'uz=', fmt(uz_prx[0]*m2mm), 'mm.')

        dp = get_thm_dp_dt(point)[0]
        #dsxx_thm = get_thm_stress(point)[0]
        dsxx_total_thm = get_thm_total_stress(point)[0]
        #dsxx_thm2 = get_thm_stress_by_deriv(point) * bars2mpa
        dsxx_total_prx = get_proxy_strain_stress(point)[3][0]  # need to X<->Y if non-symmetric
        dsxx_total_prx += m.idata.rock.biot * dp # get total from effective stress

        print('Compare at the point=', point)
        print('\tTHM   ', 'delta_P=', fmt(dp), 'MPa')
        print('\tTHM   ', 'delta_total_Sxx=', fmt(dsxx_total_thm), 'MPa')
        #print('\tTHM_by_deriv', 'delta_Sxx=', dsxx_thm2, 'MPa')
        print('\tProxy ', 'delta_total_Sxx=', fmt(dsxx_total_prx), 'MPa')

        # for uniform pressure change in the reservoir with Biot=1 and poisson ratio=0.25 should be 2/3
        print('THM delta_total_Sxx_thm_max =', fmt(np.fabs(thm_sol.delta_total_Sxx_last).max()))
        print('THM delta_total_Szz_thm_max =', fmt(np.fabs(thm_sol.delta_total_Szz_last).max()))
        print('THM delta_pressure_max=', np.fabs(thm_sol.delta_pressure).max())
        print('THM delta_total_Sxx_thm_max / delta_pressure_max=', fmt(np.fabs(thm_sol.delta_total_Sxx_last).max() / np.fabs(thm_sol.delta_pressure).max()))  # MAX
        print('THM delta_total_Sxx_thm_point / delta_pressure_point =', fmt(dsxx_total_thm / dp)) # at point
        print('Analytical delta_total_Sxx/dp =', m.idata.rock.biot * (1 - 2 * m.idata.rock.nu)/(1 - m.idata.rock.nu))

    if 'check_initial' in modes: # check initial pressure and stress for THM
        max_depth = thm_sol.bounds[2][1]  # max z m
        rsv_thickness = m.idata.other.rsv_bottom - m.idata.other.rsv_top
        poro_rsv = m.idata.rock.porosity
        poro_non_rsv = m.idata.rock.poro_non_rsv
        rock_dens = m.idata.rock.density
        fluid_dens = m.idata.fluid.density

        p_init_by_density = g_grav * fluid_dens * max_depth * Pa2bars
        szz_init_by_density = g_grav * (rock_dens*(1-poro_non_rsv) + fluid_dens*poro_non_rsv) * (max_depth - rsv_thickness) * Pa2bars
        szz_init_by_density += g_grav * (rock_dens*(1-poro_rsv) + fluid_dens*poro_rsv) * rsv_thickness * Pa2bars
        # Note, that THM depth is at cell center
        print('Pressure at depth ', max_depth, 'by gradient=', fmt(p_init_by_density),
              'THM=', fmt(thm_sol.p_init.max()), 'bars')
        print('StressZZ at depth ', max_depth, 'by gradient=', fmt(szz_init_by_density),
              'THM=', fmt(thm_sol.Szz_init.max()), 'bars')

if __name__ == '__main__':

    # nx ny nz
    cases = []
    #cases += ['7_7_5']  # for debugging
    #cases += ['17_17_15'] # for testing

    cases += ['41_41_66'] # without refinement
    #cases += ['71_71_66'] #refined middle and tips
    #cases += ['71_71_90']  #
    #cases += ['83_83_90'] # mesh is horizontally refined at inj well location
    #cases += ['97_97_90']   # mesh is horizontally refined at doublet locations

    #uniform_props = True
    uniform_props = False  # reservoir and non-reservoir in surrounding

    #thermal = False
    thermal = True

    physics_types_list = []
    if not thermal:
        physics_types_list += ['single_phase']
    else:
        physics_types_list += ['single_phase_thermal']

    wells_types_list = []
    #wells_types_list += ['none']
    #wells_types_list += ['prod']
    if not thermal:
        wells_types_list += ['inj']
    else:
        wells_types_list += ['doublet']

    # for THM solver run
    if not thermal:
        n_years = 1
    else:
        n_years = 30
    sim_time = 365.25 * n_years
    report_step = 365.25 / 4

    def get_timestep_index(t_years):
        return int((t_years * 365.25) / report_step)

    # which timestep to read from vtk (delta p,T for proxy and u,stress for comparison)
    timestep_list = [get_timestep_index(n_years)]  # last or pre-last timestep

    # process a few timesteps: 1 year, 10 years, 20 years, +last from above
    if thermal:
        for y in [1, 10, 20]:
            timestep_list += [get_timestep_index(y)]

    # short run (should be then also enabled in main.py for proper comparison)
    #sim_time = 30 # days
    #report_step = sim_time  # days
    #timestep_list = [1]

    print('timestep_list for proxy :', timestep_list)

    run_thm = True  # runs THM first, then Proxy
    #run_thm = False # don't recompute THM (use vtk files from its previous run)

    generate_mesh=False # skips mesh generation (uses a mesh from previous run), use if nothing mesh related was changed
    #generate_mesh=True

    modes = []
    #modes += ['check_initial'] # check initial pressure and stress for THM
    #modes += ['print_at_point'] # compare both THM and proxy versus analytical solution
    modes += ['plot_horiz_line']
    modes += ['plot_vertic_line']
    modes += ['plot_2d_slices']
    #modes += ['2d_slices_41_71'] # coarse mesh THM (nx=41) => finer eval points in proxy (nx=71) and compare it against finer THM (nx=71); only if case ='41_41_66'
    #modes += ['plot_2d_thm_41_vs_71']  # compare delta_pressure: THM 41_41_66 vs THM 71_71_66. This doesn't run proxy

    # for proxy:
    n_threads = 24  # CPU cores
    use_gpu = False  # CUDA
    read_from_cache = True # read stresses computed from the previous proxy run from .pkl file (useful when only ploting was changed)
    #read_from_cache = False

    for case in cases:
        for physics_type, wells_type in zip(physics_types_list, wells_types_list):
            print('\n\n' + '='*30)
            print(physics_type, wells_type)

            # run THM with no mechanics->flow impact
            t1 = datetime.now()
            if run_thm:
                run(model_folder=case, physics_type=physics_type,
                    uniform_props=uniform_props, wells_type=wells_type,
                    decouple_geomech=True, generate_mesh=generate_mesh,
                    report_step=report_step, sim_time=sim_time,
                    plot_vtk_timesteps=[0, -1]) # plot initial and last timesteps
            t2 = datetime.now()
            thm_time = t2 - t1

            # run geomech proxy
            if 'plot_horiz_line' in modes:
                # combined delta_pressure along X (Y=0, Z=2200) over all timesteps
                plot_delta_pressure_along_x(case=case, physics_type=physics_type,
                                            wells_type=wells_type, timesteps=[1,2,3,4],
                                            y=0.0, z=2200.0, report_step=report_step)
            for timestep in timestep_list:
                print('The timestep for plots and proxy-apply:', timestep)
                t1 = datetime.now()
                run_geomech_proxy(case=case, physics_type=physics_type,
                                  wells_type=wells_type, modes=modes,
                                  timestep=timestep, n_threads=n_threads, use_gpu=use_gpu,
                                  read_from_cache=read_from_cache)
                t2 = datetime.now()
                proxy_time = t2 - t1

            print('case', case, physics_type, wells_type, 'done')
            print('THM   time', thm_time)
            print('proxy time', proxy_time) # counts only the last timestep
