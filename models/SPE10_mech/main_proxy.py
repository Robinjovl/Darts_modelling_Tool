import numpy as np
import os
import meshio
from datetime import datetime
from matplotlib import pyplot as plt
from scipy.interpolate import griddata as gd
from functools import reduce

from main import run
from darts.reservoirs.unstruct_reservoir_mech import get_bulk_modulus

# unit conversion factors
m2mm = 1e3
bars2mpa = 0.1
Pa2bars = 1e-5

def fmt(x):
    return '{:.3}'.format(x)

def read_vtk_darts_solution(folder, timestep : int):
    filename = os.path.join(folder, 'solution'+str(timestep)+'.vtu')
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

def run_geomech_proxy(case, physics_type='single_phase', wells_type=None, timestep=1, generate_mesh=True):
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
    g.thermal_exp_coeff = m.idata.rock.th_expn / get_bulk_modulus(E=m.idata.rock.E, nu=m.idata.rock.nu)# 1/°C

    # read THM solution from vtk
    msh_initial = read_vtk_darts_solution(folder=folder, timestep=0)
    poro = np.array(msh_initial.cell_data['poro']).flatten()
    p_init = np.array(msh_initial.cell_data['pressure']).flatten()
    Szz_init = np.array(msh_initial.cell_data['tot_stress'])[0, :, 2] * bars2mpa # ZZ

    msh_last = read_vtk_darts_solution(folder=folder, timestep=timestep)
    p_last = np.array(msh_last.cell_data['pressure']).flatten()
    ux_last = np.array(msh_last.cell_data['ux']).flatten()
    uy_last = np.array(msh_last.cell_data['uy']).flatten()
    uz_last = np.array(msh_last.cell_data['uz']).flatten()
    delta_Sxx_last = np.array(msh_last.cell_data['delta_eff_stress'])[0, :, 0] * bars2mpa #  XX
    delta_Syy_last = np.array(msh_last.cell_data['delta_eff_stress'])[0, :, 1] * bars2mpa #  YY
    delta_Szz_last = np.array(msh_last.cell_data['delta_eff_stress'])[0, :, 2] * bars2mpa # ZZ
    delta_total_Sxx_last = np.array(msh_last.cell_data['delta_tot_stress'])[0, :, 0] * bars2mpa #  XX
    delta_total_Syy_last = np.array(msh_last.cell_data['delta_tot_stress'])[0, :, 1] * bars2mpa #  YY
    delta_total_Szz_last = np.array(msh_last.cell_data['delta_tot_stress'])[0, :, 2] * bars2mpa # ZZ
    qx_last = np.array(msh_last.cell_data['strain'])[0, :, 0]  #  XX
    qy_last = np.array(msh_last.cell_data['strain'])[0, :, 1]  #  YY
    qz_last = np.array(msh_last.cell_data['strain'])[0, :, 2]  # ZZ
    
    folder = os.path.join(folder, 'timestep_' + str(timestep))
    os.makedirs(folder, exist_ok=True)
    
    if 'delta_pressure' in msh_last.cell_data.keys():
        delta_pressure = np.array(msh_last.cell_data['delta_pressure']).flatten()
    else:
        p_initial = np.array(msh_initial.cell_data['pressure']).flatten()
        delta_pressure = p_last - p_initial
    delta_pressure *= 0.1 # bars to MPa
    
    # delta_temperature is zero in isothermal case
    delta_temperature = np.zeros_like(delta_pressure) 
    if 'delta_temperature' in msh_last.cell_data.keys():
        delta_temperature = np.array(msh_last.cell_data['delta_temperature']).flatten()

    prisms = geomech_init_geometry(msh_initial)
    print('\tprisms all', prisms.shape[0])

    #TODO do not use the whole mesh - use only the permeable part, assuming there is no p,T change in the impermeable part
    rsv = poro > m.idata.rock.poro_non_rsv  # reservoir (permeable) cells only for use in the proxy
    #rsv = poro > 0 # use all cells in the proxy
    delta_pressure_rsv = delta_pressure[rsv]
    delta_temperature_rsv = delta_temperature[rsv]
    prisms_rsv = prisms[rsv, :]
    print('\tprisms rsv', prisms.shape[0])

    # centroids are only used for THM data plotting, they are not used in proxy
    centroids = np.zeros((prisms.shape[0], 3))
    centroids[:, 0] = (prisms[:, 2] +  prisms[:, 3]) * 0.5 # Y
    centroids[:, 1] = (prisms[:, 0] +  prisms[:, 1]) * 0.5 # X
    centroids[:, 2] = (prisms[:, 4] +  prisms[:, 5]) * 0.5 # z

    n_dim = 3  # X,Y,Z
    bounds = [0]*n_dim
    for k in range(n_dim):
        bounds[k] = centroids[:, k].min(), centroids[:, k].max()

    def find_cell_by_point(point):
        # find an index of the cell, closest to the desired point
        cell = ((centroids[:, 0] - point[0]) ** 2 + (centroids[:, 1] - point[1]) ** 2 + (
                    centroids[:, 2] - point[2]) ** 2).argmin()
        #print('get_thm_solution', 'closest cell is', centroids[cell, :], 'point', point)
        return cell

    def get_cell_center(point):
        cell = find_cell_by_point(point)
        return centroids[cell, 0], centroids[cell, 1], centroids[cell, 2]

    def get_thm_by_interp(array_dict, points_x, points_y, points_z, method='linear'):
        array_dict_interp = dict()
        for arr_name, arr in array_dict.items():
            # interpolate the solution (arr) from cell centers to given set of points
            array_dict_interp[arr_name] = gd((centroids[:, 1], centroids[:, 0], centroids[:, 2]), \
                arr, (points_x, points_y, points_z), method=method)
        return array_dict_interp

    def get_thm_dp_dt(point, verbose=False): # pressure (in MPa) and temperature (in K) changes
        cell = find_cell_by_point(point)
        dp = delta_pressure[cell] 
        dt = delta_temperature[cell]
        return dp, dt

    def get_thm_displs(point, verbose=False):
        cell = find_cell_by_point(point)
        ux_thm = ux_last[cell]
        uy_thm = uy_last[cell]
        uz_thm = uz_last[cell]
        return ux_thm, uy_thm, uz_thm

    def get_thm_strain(point, verbose=False):
        cell = find_cell_by_point(point)
        qx_thm = qx_last[cell]
        qy_thm = qy_last[cell]
        qz_thm = qz_last[cell]
        return qx_thm, qy_thm, qz_thm

    def get_thm_stress(point, verbose=False): # effective stress in MPa
        cell = find_cell_by_point(point)
        return delta_Sxx_last[cell],  delta_Syy_last[cell],  delta_Szz_last[cell]

    def get_thm_total_stress(point, verbose=False): # total stress in MPa
        cell = find_cell_by_point(point)
        return delta_total_Sxx_last[cell],  delta_total_Syy_last[cell],  delta_total_Szz_last[cell]

    def get_thm_stress_by_deriv(point): 
        # compute strain and stress in python from THM displacements (ux_last, etc)
        from geomechanics import deriv

        # find the closest neighbout cell in each direction. Since we don't know the cells size, we will move step by step
        n_neig = 6 # X- X+, Y-, Y+, Z-, Z+
        cell_neig = np.zeros(n_neig, dtype=int)
        point_neig = np.zeros((n_neig, n_dim))
        step = [10., 10., 5.]  # [m] to find neighboring cells, should be less than cell size

        mults = [0.5] * n_dim

        cell = find_cell_by_point(point)
        i = 0
        for side in [-1, 1]:
            for k in range(n_dim):
                cell_neig[i] = cell
                point_neig[i] = point.copy()  # start from the point itself
                if point_neig[i][k] < bounds[k][0] or point_neig[i][k] > bounds[k][1]: # if the starting point is out of bounds, start from the closest centroid
                    cell_neig[i] = find_cell_by_point(point_neig[i])
                    mults[k] = 1.0  # one-sided derivative
                j = 0
                while cell_neig[i] == cell:  # move, until reached the next cell
                    point_neig[i][k] += side * step[k]
                    if (point_neig[i][k] < bounds[k][0] and side == -1) or (point_neig[i][k] > bounds[k][1] and side == 1):  # reached the boundary
                        cell_neig[i] = cell
                        point_neig[i][k] = point[k]  # return to the original point
                        mults[k] = 1.0  # one-sided derivative
                        break
                    cell_neig[i] = find_cell_by_point(point_neig[i])
                    if j > 1000:  # just in case, to avoid infinite loop
                        print('Error: can not find a neighboring cell for point', point, 'side', side, 'k', k)
                        exit(1)
                point_neig[i] = centroids[cell_neig[i], :]
                i += 1
                
        #print('point', point)
        #print('z- neig center', centroids[find_cell_by_point(point_neig[2]), 2])
        #print('z+ neig center', centroids[find_cell_by_point(point_neig[5]), 2])

        ux_x_minus_, ux_x_plus_ = ux_last[cell_neig[0]], ux_last[cell_neig[3]]
        ux_y_minus_, ux_y_plus_ = ux_last[cell_neig[1]], ux_last[cell_neig[4]]
        ux_z_minus_, ux_z_plus_ = ux_last[cell_neig[2]], ux_last[cell_neig[5]]

        uy_x_minus_, uy_x_plus_ = uy_last[cell_neig[0]], uy_last[cell_neig[3]]
        uy_y_minus_, uy_y_plus_ = uy_last[cell_neig[1]], uy_last[cell_neig[4]]
        uy_z_minus_, uy_z_plus_ = uy_last[cell_neig[2]], uy_last[cell_neig[5]]

        uz_x_minus_, uz_x_plus_ = uz_last[cell_neig[0]], uz_last[cell_neig[3]]
        uz_y_minus_, uz_y_plus_ = uz_last[cell_neig[1]], uz_last[cell_neig[4]]
        uz_z_minus_, uz_z_plus_ = uz_last[cell_neig[2]], uz_last[cell_neig[5]]

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
        eps = 1  # [m], to avoid r=0 for the integral in the geomech proxy 1/r
        eval_points_eps = eval_points + eps
        #eval_points_eps = eval_points_eps.transpose()
        upy1, upx1, upz1, uty1, utx1, utz1 = g.calc_displacements_cpp(eval_points_eps, prisms_rsv, delta_pressure_rsv, delta_temperature_rsv)
        ux = upx1 + utx1
        uy = upy1 + uty1
        uz = upz1 + utz1
        return ux, uy, uz # thermoporoelastic displacements [m]

    def get_eval_points(mode, shift_x=0, shift_y=0):
        if mode == 'centers':
            # X<->Y
            eval_points = np.zeros_like(centroids)
            eval_points[:, 1] = centroids[:, 0] # X
            eval_points[:, 0] = centroids[:, 1]
            eval_points[:, 2] = centroids[:, 2]
        elif mode == 'vertical':
            # test with a vertical line
            z = np.unique(centroids[:, 2])
            eval_points = np.zeros((z.size, 3))
            eval_points[:, 1] = centroids[:, 0].mean() + shift_x
            eval_points[:, 0] = centroids[:, 1].mean() + shift_y
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
        eps = 1  # [m], to avoid r=0 for the integral in the geomech proxy 1/r
        eval_points_eps = eval_points + eps
        res = g.calc_strain_stress_cpp(eval_points_eps, prisms_rsv, delta_pressure_rsv, delta_temperature_rsv)
        stress_p, strain_p, stress_t, strain_t, stress, strain = res
        #[Sp_xx, Sp_yy, Sp_zz, Sp_yz, Sp_xz, Sp_xy] = stress_p
        #[St_xx, St_yy, St_zz, St_yz, St_xz, St_xy] = stress_t
        return strain[1,:], strain[0,:], strain[2,:], \
            stress[1,:], stress[0,:], stress[2,:] # xx yy zz


    def compare_vert_line(points, suffix='', loc='', output_folder='.', modes={}):
        z_range = points[2,:]

        array_dict = {'dp': delta_pressure, 'dt': delta_temperature}
        array_dict_interp = get_thm_by_interp(array_dict, points[1,:], points[0,:], points[2,:], method='nearest') # obtain thm solutiona at points using interpolation
        dp = array_dict_interp['dp']
        dt = array_dict_interp['dt']
        
        ux_prx, uy_prx, uz_prx = get_proxy_displs(points)
        qx_prx, qy_prx, qz_prx, sx_prx, sy_prx, sz_prx =  get_proxy_strain_stress(points)
        
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
            point = np.array([points[1, i], points[0, i], points[2, i]])  # YXZ - > XYZ
            ux_thm[i], uy_thm[i], uz_thm[i] = get_thm_displs(point)
            if 'strain' in modes:
                qx_thm[i], qy_thm[i], qz_thm[i] = get_thm_strain(point)
            if 'delta_eff_stress_z' in modes:
                sx_thm[i], sy_thm[i], sz_thm[i] = get_thm_stress(point)
            if 'delta_total_stress_z' in modes:
                sx_total_thm[i], sy_total_thm[i], sz_total_thm[i] = get_thm_total_stress(point)
                
            if ('strain' in modes or 'delta_eff_stress_z' in modes) and plot_thm2:
                qx_thm2[i], qy_thm2[i], qz_thm2[i], \
                sx_thm2[i], sy_thm2[i], sz_thm2[i] = get_thm_stress_by_deriv(point)
            
        for mode in modes:
            match mode:
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
                        
            plt.axhline(y=m.idata.other.rsv_top, color='red', linestyle='dotted', label='rsv top')#, xmin=0.95, xmax=1.0)
            plt.axhline(y=m.idata.other.rsv_bottom, color='red', linestyle='dotted', label='rsv bottom')#, xmin=0.95, xmax=1.0)
            plt.plot(thm, z_range, label=mode + '_THM')#, marker='.')
            if prx is not None:
                plt.plot(prx, z_range, label=mode + '_proxy', linestyle='--')#, marker='.')
            if ('stress' in mode or 'strain' in mode) and plot_thm2:
                plt.plot(thm2, z_range, label=mode + '_THM2', color='black')#marker='.', 
                
            if mode == 'delta_total_stress_z' and prx.max() < 0.05 and thm.max() < 0.05: # vertical stress is almost zero
                plt.xlim(-0.25, 0.25)    
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
            plt.savefig(os.path.join(output_folder, mode + '_' + loc + '_' + suffix + '.png'))
            plt.close()

    def plot_contour(array_dict, points_x, points_y, output_folder, layer=0):
        # plot contours XY plane, 1 layer by z
        for arr_name, arr in array_dict.items():
            if len(arr.shape) == 3:
                arr_layer = arr[:, :, layer]
            else:
                arr_layer = arr
            cs = plt.contourf(points_x, points_y, arr_layer, levels=10)  
            plt.colorbar(cs)
            plt.gca().set_aspect('equal')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title(arr_name)
            plt.savefig(os.path.join(output_folder, arr_name + '.png'))
            plt.close()
    
    points_xy = dict()
    #points_xy['center'] = centroids[:, 0].mean(), centroids[:, 1].mean()]  # middle point of the mesh
    points_xy['(50,50)'] = [50., 50.]  # middle point of the mesh but shift abit to make it at the cell centers by XY
    #points_xy['(450,0)'] = [0., 450.]  # the order is actually Y,X
    #points_xy['(450,450)'] = [450., 450.]  # the order is actually Y,X
    #points_xy['(6000,6000)'] = [6000., 6000.]  # the order is actually Y,X
    
    if False:
        if wells_type in ['prod', 'doublet']:
            points_xy['prod well'] = m.prod_well_coords[:-1] # -1 to skip z coord
        if wells_type in ['inj', 'doublet']:
            points_xy['inj well'] = m.inj_well_coords[:-1]

    # plot 2D THM displs (XY plane)
    if False:
        #points_x = np.arange(bounds[0][0], bounds[0][1], 250) # the whole mesh by XY
        points_x = np.arange(-1000, 1000, 100)  # only internal XY part
        points_y = points_x
        points_z = np.array([2150])
        #points_z = np.hstack([np.arange(0, 2000, 200), np.arange(2100, 2200, 10), np.arange(2300, 4000, 200)])
        #points_z = np.arange(1800, 2400, 25)
        points_x_3d, points_y_3d, points_z_3d = np.meshgrid(points_x, points_y, points_z)
        array_dict = {'ux_thm_2d': ux_last, 'uy_thm_2d': uy_last, 'uz_thm_2d': uz_last}
        array_dict.update({'delta_stress_XX_MPa': delta_Sxx_last, 'delta_stress_ZZ_MPa': delta_Szz_last})
        #for arr_name in array_dict.keys():
        #    array_dict[arr_name] = array_dict[arr_name][rsv]
        array_dict.update({'delta_pressure_MPa': delta_pressure, 'delta_temperature': delta_temperature})
        array_dict_interp = get_thm_by_interp(array_dict, points_x_3d, points_y_3d, points_z_3d)
        plot_contour(array_dict_interp, points_x, points_y, output_folder=folder)

    # compute with proxy in 3D volume
    if False:
        print('plotting 2D slices')
        p_nx, p_ny, p_nz = points_x.size, points_y.size, points_z.size
        points = np.zeros((3, points_x_3d.size))
        points[1, :], points[0, :], points[2, :] = points_x_3d.flatten(), points_y_3d.flatten(), points_z_3d.flatten()
        ux_prx, uy_prx, uz_prx = get_proxy_displs(points)
        ux_prx_3d = ux_prx.reshape((p_nx, p_ny, p_nz))
        uy_prx_3d = uy_prx.reshape((p_nx, p_ny, p_nz))
        uz_prx_3d = uz_prx.reshape((p_nx, p_ny, p_nz))
        
        # save to pkl
        displs = {'ux_prx_2d': ux_prx_3d, 'uy_prx_2d': uy_prx_3d, 'uz_prx_2d': uz_prx_3d}
        import pickle
        with open(os.path.join(folder, "displs_prx.pkl"), "wb") as f:   # note 'wb' = write binary
            pickle.dump(displs, f)
        
        #rsv_idx = np.where((self.rsv_top < m.reservoir.Zc) & (m.reservoir.Zc < self.rsv_top)).min()
        
        #plt.contourf(uz_prx_3d[:,10,:].transpose())  # XZ plane
        #plt.contourf(uz_prx_3d[:,:,20])  # XY plane 
        #plt.plot(ux_prx_3d[10,10,:])  # along Z-axis
        
        array_dict = {'ux_prx_3d':ux_prx_3d[:,:,0].transpose(), 'uy_prx_3d':uy_prx_3d[:,:,0].transpose(), 
                      'uz_prx_3d':uz_prx_3d[:,:,0].transpose()}
        plot_contour(array_dict, points_x, points_y, output_folder=folder)
        
        # plot 1D plots at different X-layers to check the strain computation
        if False:
            j = 10
            for i in range(9,12):
                plt.plot(points_z, ux_prx_3d[i,j,:], label='prx_'+str(i), marker='.')
                ux_thm = []
                for p_z in points_z:
                    p = points_x[i], points_y[j], p_z
                    ux_thm.append(get_thm_displs(p)[0])
                plt.plot(points_z, ux_thm, label='thm_'+str(i), linestyle='--', marker='.')
                
            plt.gca().invert_yaxis()
            plt.legend()
            plt.grid()
            plt.savefig(os.path.join(folder, 'Ux_by_depth.png'))
            plt.close()

    for k in points_xy.keys():
        point_xy = points_xy[k]
        print('1D plots for point', k, 'YX=', point_xy)
        modes = ['delta_pressure']
        modes += ['displ_z', 'displ_y', 'displ_x']
        modes += ['delta_eff_stress_z', 'delta_eff_stress_y', 'delta_eff_stress_x']
        modes += ['delta_total_stress_z', 'delta_total_stress_y', 'delta_total_stress_x']
        #modes += ['strain_z', 'strain_y', 'strain_x']
        #modes = ['strain_x']  # debug

        # compare U-Z at a line along z-axis
        z_min = 0.
        z_max = centroids[:, 2].max() #+ 1000.
        z_step = 5.4345653  # m.  # add a random number to avoid cell centers/ not needed maybe
        
        z_range_all = np.arange(z_min, z_max+1., z_step)
        z_interp_eps = 5. # m # remove points close to rsv boundary, as they create descrepancies even with 'nearest' interpolation
        z_range_all_filter = reduce(np.logical_and, [np.fabs(z_range_all - m.idata.other.rsv_top) > z_interp_eps, np.fabs(z_range_all - m.idata.other.rsv_bottom) > z_interp_eps])
        z_range_all = z_range_all[z_range_all_filter]
        n_points = z_range_all.size
        points_all = np.zeros((3, n_points))
        points_all[0, :] = point_xy[1]  # X<->Y
        points_all[1, :] = point_xy[0]
        points_all[2, :] = z_range_all
        
        z_range_rsv = np.arange(m.idata.other.rsv_top-100., m.idata.other.rsv_bottom+100., z_step)
        # remove points close to rsv boundary, as they create descrepancies even with 'nearest' interpolation
        z_range_rsv_filter = reduce(np.logical_and, [np.fabs(z_range_rsv - m.idata.other.rsv_top) > z_interp_eps, np.fabs(z_range_rsv - m.idata.other.rsv_bottom) > z_interp_eps])
        z_range_rsv = z_range_rsv[z_range_rsv_filter]
        n_points = z_range_rsv.size
        points_rsv = np.zeros((3, n_points))
        points_rsv[0, :] = point_xy[1] # X<->Y
        points_rsv[1, :] = point_xy[0]            
        points_rsv[2, :] = z_range_rsv
        
        compare_vert_line(points_all, suffix='all', loc=k, output_folder=folder, modes=modes)
        compare_vert_line(points_rsv, suffix='rsv', loc=k, output_folder=folder, modes=modes)

    # print vert displs and stresses change at a point
    if True:
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
        print('THM delta_total_Sxx_thm_max =', fmt(np.fabs(delta_total_Sxx_last).max()))
        print('THM delta_total_Szz_thm_max =', fmt(np.fabs(delta_total_Szz_last).max()))
        print('THM delta_pressure_max=', np.fabs(delta_pressure).max())
        print('THM delta_total_Sxx_thm_max / delta_pressure_max=', fmt(np.fabs(delta_total_Sxx_last).max() / np.fabs(delta_pressure).max()))  # MAX
        print('THM delta_total_Sxx_thm_point / delta_pressure_point =', fmt(dsxx_total_thm / dp)) # at point
        
    if False: # check initial pressure and stress for THM
        max_depth = bounds[2][1]  # max z m
        rsv_thickness = m.idata.other.rsv_bottom - m.idata.other.rsv_top
        poro_rsv = m.idata.rock.porosity
        poro_non_rsv = m.idata.rock.poro_non_rsv
        rock_dens = m.idata.rock.density
        fluid_dens = m.idata.fluid.density
        
        p_init_by_density = 9.81 * fluid_dens * max_depth * Pa2bars
        szz_init_by_density = 9.81 * (rock_dens*(1-poro_non_rsv) + fluid_dens*poro_non_rsv) * (max_depth - rsv_thickness) * Pa2bars
        szz_init_by_density += 9.81 * (rock_dens*(1-poro_rsv) + fluid_dens*poro_rsv) * rsv_thickness * Pa2bars
        # Note, that THM depth is at cell center 
        print('Pressure at depth ', max_depth, 'by gradient=', fmt(p_init_by_density), 
              'THM=', fmt(p_init.max()), 'bars')
        print('StressZZ at depth ', max_depth, 'by gradient=', fmt(szz_init_by_density),
              'THM=', fmt(Szz_init.max()), 'bars')

if __name__ == '__main__':

    #case = '6_6_5'  # for debugging
    #case = '16_16_15'
    case = '34_34_57'  # z 0 - 5 km 
    #case = '34_34_65'  # z 0 - 10 km
    #case='34_35_57' # perm_frac
    
    #case = '34_34_15'
    #case = '16_16_65'

    #uniform_props = True
    uniform_props = False  # reservoir and non-reservoir in surrounding

    physics_types_list = []
    physics_types_list += ['single_phase']
    #physics_types_list += ['single_phase_thermal']

    wells_types_list = []
    #wells_types_list += ['none']
    #wells_types_list += ['prod']
    wells_types_list += ['inj']
    #wells_types_list += ['doublet']
    
    # for THM solver run
    #n_years = 1
    #n_years = 2
    n_years = 5
    #n_years = 10
    #n_years = 30
    #n_years = 50
    sim_time = 365.25 * n_years
    report_step = 365.25 / 4

    # short run
    #sim_time = 90 # days
    #report_step = sim_time  # days
    
    # which timestep to read from vtk (delta p,T for proxy and u,stress for comparison)
    timestep = int((n_years * 365.25) / report_step)  # last or pre-last timestep
    #timestep = 1
    #timestep = 4
    
    #run_thm = True
    run_thm = False
    
    #generate_mesh=False
    generate_mesh=True

    for physics_type in physics_types_list:
        for wells_type in wells_types_list:
            print('\n\n' + '='*30)
            print(physics_type, wells_type)

            # run THM with no mechanics->flow impact
            t1 = datetime.now()
            if run_thm:
                run(model_folder=case, physics_type=physics_type, 
                    uniform_props=uniform_props, wells_type=wells_type, 
                    decouple_geomech=True, generate_mesh=generate_mesh,
                    report_step=report_step, sim_time=sim_time)
            t2 = datetime.now()
            thm_time = t2 - t1

            # run geomech proxy
            print('The timestep for plots and proxy-apply:', timestep)
            t1 = datetime.now()
            run_geomech_proxy(case=case, physics_type=physics_type, wells_type=wells_type, timestep=timestep)
            t2 = datetime.now()
            proxy_time = t2 - t1

            print('case', case, 'done')
            print('THM   time', thm_time)
            print('proxy time', proxy_time)