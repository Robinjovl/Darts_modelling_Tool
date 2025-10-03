import numpy as np
import os
import meshio
from datetime import datetime

from main import run
from darts.reservoirs.unstruct_reservoir_mech import get_bulk_modulus

# unit conversion factors
m2mm = 1e3
bars2mpa = 0.1

def read_vtk_darts_solution(folder, timestep : int):
    filename = os.path.join(folder, 'solution'+str(timestep)+'.vtu')
    msh = meshio.read(filename)
    print('Reading', filename)
    print("\tCells:", msh.cells_dict.keys())
    print("\tCell Data:", msh.cell_data.keys())
    return msh


def get_pressure(self, physics):
    '''
    :param physics:
    :return: 1d current timestep pressure array
    '''
    # Temporarily store mesh_data in copy:
    Mesh = meshio.read(self.mesh_filename)

    # Allocate empty new cell_data dictionary:
    cell_property = ['u_x', 'u_y', 'u_z', 'p']
    props_num = len(cell_property)
    property_array = np.array(physics.engine.X, copy=False)
    available_matrix_geometries = ['hexahedron', 'wedge', 'tetra']
    available_fracture_geometries = ['quad', 'triangle']
    # Matrix
    geom_id = 0
    Mesh.cells = []
    cell_data = {}
    for ith_geometry in self.unstr_discr.mesh_data.cells_dict.keys():
        if ith_geometry in available_matrix_geometries:
            Mesh.cells.append(self.unstr_discr.mesh_data.cells[geom_id])
            # Add matrix data to dictionary:
            for i in range(props_num):
                if cell_property[i] not in cell_data: cell_data[cell_property[i]] = []
                cell_data[cell_property[i]].append(property_array[i:props_num * self.unstr_discr.mat_cells_tot:props_num])
    return cell_data['p'][-1]


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

def run_geomech_proxy(case, physics_type='single_phase', wells_type=None):
    folder = 'sol_cpp_' + physics_type + '_'  + wells_type + '_' + case

    # init geomech proxy
    from geomechanics import geomech
    g = geomech()
    # just to set input data
    from model import Model
    m = Model(model_folder=case, physics_type=physics_type, uniform_props=False, wells_type=wells_type, decouple_geomech=True, generate_mesh=True)
    # elastic constants
    g.poisson = m.idata.rock.nu
    g.young = m.idata.rock.E.mean() * bars2mpa # bars to MPa
    g.thermal_exp_coeff = m.idata.rock.th_expn / get_bulk_modulus(E=m.idata.rock.E, nu=m.idata.rock.nu)# 1/°C

    msh_initial = read_vtk_darts_solution(folder=folder, timestep=0)
    poro = np.array(msh_initial.cell_data['poro']).flatten()
    p_initial = np.array(msh_initial.cell_data['pressure']).flatten()

    msh_last    = read_vtk_darts_solution(folder=folder, timestep=1)
    p_last = np.array(msh_last.cell_data['pressure']).flatten()
    uz_last = np.array(msh_last.cell_data['uz']).flatten()
    delta_Sxx_last = np.array(msh_last.cell_data['delta_tot_stress'])[0, :, 0] # third dimension: 0 is XX
    #delta_Szz_last = np.array(msh_last.cell_data['delta_tot_stress'])[0, :, 2] # third dimension: 2 is ZZ

    delta_pressure = (p_last - p_initial) * 0.1 # bars to MPa
    delta_temperature = np.zeros_like(delta_pressure) #TODO

    prisms = geomech_init_geometry(msh_initial)
    print('\tprisms all', prisms.shape[0])

    # do not use the whole mesh - use only the permeable part, assuming there is no p,T change in the impermeable part
    rsv = poro > m.idata.rock.poro_non_rsv  # reservoir cells
    #delta_pressure = delta_pressure[rsv]
    #delta_temperature = delta_temperature[rsv]
    #prisms = prisms[rsv, :]
    print('\tprisms rsv', prisms.shape[0])

    # where to compare the results - middle XYZ
    centroids = np.zeros((prisms.shape[0], 3))
    centroids[:, 0] = (prisms[:, 2] +  prisms[:, 3]) * 0.5 # x
    centroids[:, 1] = (prisms[:, 0] +  prisms[:, 1]) * 0.5 # y
    centroids[:, 2] = (prisms[:, 4] +  prisms[:, 5]) * 0.5 # z

    def get_thm_displs(point, verbose=False):
        # find an index of the cell, closest to the desired point
        cell = ((centroids[:, 0] - point[0]) ** 2 + (centroids[:, 1] - point[1]) ** 2 + (centroids[:, 2] - point[2]) ** 2).argmin()
        uz_thm = uz_last[cell]
        if verbose:
            print('get_thm_solution', 'closest cell is', centroids[cell, :], 'point', point)
        return uz_thm

    def get_thm_stress(point, verbose=False):
        # find an index of the cell, closest to the desired point
        cell = ((centroids[:, 0] - point[0]) ** 2 + (centroids[:, 1] - point[1]) ** 2 + (centroids[:, 2] - point[2]) ** 2).argmin()
        if verbose:
            print('get_thm_solution', 'closest cell is', centroids[cell, :], 'point', point)
        return delta_Sxx_last[cell]

    def get_proxy_displs(point):
        eval_points = np.zeros((1,3))  # just one point
        eps = 1  # [m], to avoid r=0 for the integral in the geomech proxy 1/r
        eval_points[0] = np.array([point[1]+eps, point[0]+eps, point[2]+eps]) # Y,X,Z
        upx1, upy1, upz1, utx1, uty1, utz1 = g.calc_displacements_cpp(eval_points, prisms, delta_pressure, delta_temperature)
        return upz1[0] + utz1[0] # thermoporoelastic vertical displacement uz, in m

    def get_proxy_stress(point):
        eval_points = np.zeros((1,3))  # just one point
        eps = 1  # [m], to avoid r=0 for the integral in the geomech proxy 1/r
        eval_points[0] = np.array([point[1]+eps, point[0]+eps, point[2]+eps]) # Y,X,Z
        eval_points = eval_points.transpose()
        res = g.calc_strain_stress_cpp(eval_points, prisms, delta_pressure, delta_temperature)
        stress_p, strain_p, stress_t, strain_t, stress, strain = res
        [Sp_xx, Sp_yy, Sp_zz, Sp_yz, Sp_xz, Sp_xy] = stress_p
        [St_xx, St_yy, St_zz, St_yz, St_xz, St_xy] = stress_t
        [S_xx, S_yy, S_zz, S_yz, S_xz, S_xy] = stress

        return Sp_xx + St_xx # thermoporoelastic stress XX in MPa


    def compare_vert_line(point_xy, z_min, z_max, suffix, z_step=100, output_folder='.', mode='displ_z'):
        point = np.array([point_xy[1], point_xy[2], 0.])  # XY from point and Z will be changed
        z_range = np.arange(z_min, z_max+1., z_step)
        thm = []
        prx = []
        for z in z_range:  # use XY from point and different Z
            point[2] = z
            if mode == 'displ_z':
                thm.append(get_thm_displs(point) * m2mm)
                prx.append(get_proxy_displs(point) * m2mm)
                label = 'uz'
            elif mode == 'stress':
                thm.append(get_thm_stress(point) * bars2mpa)
                prx.append(get_proxy_stress(point))
                label = 'delta_stress'

        from matplotlib import pyplot as plt
        plt.plot(thm, z_range, label=label + '_thm')
        plt.plot(prx, z_range, label=label + '_prx')
        plt.axhline(y=m.reservoir.rsv_top, color='red', linestyle='--', label='rsv top')
        plt.axhline(y=m.reservoir.rsv_bottom, color='red', linestyle='--', label='rsv bottom')
        plt.gca().invert_yaxis()
        if mode == 'displ_z':
            s = 'Vertical displacement, mm.'
        elif mode == 'stress':
            s = 'Horizontal stress delta, MPa.'
        s += ' at ' + point_xy[0]
        plt.xlabel(s)
        plt.title(s)
        plt.ylabel('Depth, m.')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(output_folder, mode + '_' + point_xy[0] + '_' + suffix + '.png'))
        plt.close()

    ######################################

    points_xy = []
    points_xy += [['center', centroids[:, 0].mean(), centroids[:, 1].mean()]]  # middle point of the mesh

    if wells_type in ['prod', 'doublet']:
        points_xy += [['prod well'] + m.prod_well_coords[:-1]] # -1 to skip z coord
    if wells_type in ['inj', 'doublet']:
        points_xy += [['inj well'] + m.inj_well_coords[:-1]]

    for point_xy in points_xy:
        print('plotting for point', point_xy[0], 'XY=', point_xy[1:3])
        for mode in ['displ_z', 'stress']:
            # compare U-Z at a line along z-axis
            z_min = 0.
            z_max = centroids[:, 2].max() #+ 1000.
            compare_vert_line(point_xy, z_min, z_max, 'all', z_step=20, output_folder=folder, mode=mode)
            compare_vert_line(point_xy, m.reservoir.rsv_top-100., m.reservoir.rsv_bottom+100.,'rsv',  z_step=10, output_folder=folder, mode=mode)

    # compare vert displs at the middle point at the surface and print
    point = [points_xy[0][1], points_xy[0][2], 0.] # at the surface (depth=0)
    uz_thm = get_thm_displs(point)*m2mm
    uz_prx = get_proxy_displs(point)*m2mm
    print('Compare at single point ', point)
    print('\tTHM   ', 'uz=', uz_thm, 'mm.')
    print('\tProxy ', 'uz=', uz_prx, 'mm.')

    # compare delta Sxx at the middle point in the reservoir and print
    point[2] = (m.reservoir.rsv_top + m.reservoir.rsv_bottom) * 0.5  # at the middle depth of the reservoir
    dsxx_thm = get_thm_stress(point)*bars2mpa
    dsxx_prx = get_proxy_stress(point)
    print('Compare at single point ', point)
    print('\tTHM   ', 'delta_Sxx=', dsxx_thm, 'MPa')
    print('\tProxy ', 'delta_Sxx=', dsxx_prx, 'MPa')

if __name__ == '__main__':

    #case = '6_6_5'  # for debugging
    #case = '34_34_15'
    #case = '16_16_53'
    #case = '28_28_53' # bad allocation only for thermal
    #case = '28_28_29'  # crashes after initialization
    #case = '28_28_37'  #
    #case = '28_28_53'  #
    case = '34_34_54'  #
    #case = '34_34_53' # bad allocation for both isothermal and thermal

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

    for physics_type in physics_types_list:
        for wells_type in wells_types_list:
            print('\n\n' + '='*30)
            print(physics_type, wells_type)

            # run THM with no mechanics->flow impact
            t1 = datetime.now()
            run(model_folder=case, physics_type=physics_type, uniform_props=uniform_props, wells_type=wells_type, decouple_geomech=True, generate_mesh=True)
            t2 = datetime.now()
            thm_time = t2 - t1

            # run geomech proxy
            t1 = datetime.now()
            run_geomech_proxy(case=case, physics_type=physics_type, wells_type=wells_type)
            t2 = datetime.now()
            proxy_time = t2 - t1

            print('case', case, 'done')
            print('THM   time', thm_time)
            print('proxy time', proxy_time)