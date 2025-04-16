import numpy as np
import os
import meshio
from darts.discretizer import elem_type, elem_loc
from darts.discretizer import matrix33 as disc_matrix33
from darts.discretizer import Stiffness as disc_stiffness
from darts.reservoirs.unstruct_reservoir_mech import set_domain_tags, get_lambda_mu, get_biot_modulus
from darts.reservoirs.unstruct_reservoir_mech import UnstructReservoirMech
from darts.input.input_data import InputData
from darts.engines import timer_node, ms_well, ms_well_vector
import copy
import vtk

def read_vtk_darts_solution(folder, timestep : int):
    filename = os.path.join(folder, 'solution'+str(timestep)+'.vtk')
    msh = meshio.read(filename)
    print("Cells:", msh.cells_dict.keys())
    print("Cell Data:", msh.cell_data.keys())
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

    print('prisms', prisms.shape)
    return prisms

def run_geomech_proxy(case):
    folder = 'sol_cpp_single_phase_' + case

    msh_initial = read_vtk_darts_solution(folder=folder, timestep=0)
    p_initial = np.array(msh_initial.cell_data['pressure']).flatten()

    msh_last    = read_vtk_darts_solution(folder=folder, timestep=20)
    p_last = np.array(msh_last.cell_data['pressure']).flatten()
    uz_last = np.array(msh_last.cell_data['uz']).flatten()

    delta_pressure = (p_last - p_initial) * 0.1 # bars to MPa
    delta_temperature = np.zeros_like(delta_pressure) #TODO

    prisms = geomech_init_geometry(msh_initial)

    # where to compare the results - middle XYZ
    centroids = np.zeros((prisms.shape[0], 3))
    centroids[:, 0] = (prisms[:, 2] +  prisms[:, 3]) * 0.5 # x
    centroids[:, 1] = (prisms[:, 0] +  prisms[:, 1]) * 0.5 # y
    centroids[:, 2] = (prisms[:, 4] +  prisms[:, 5]) * 0.5 # z

    # init geomech proxy
    from geomechanics import geomech
    g = geomech()
    # just to set input data
    from model import Model
    m = Model(model_folder='data_' + case, physics_type='single_phase', uniform_props=False)
    # elastic constants
    g.poisson = m.idata.rock.nu
    g.young = m.idata.rock.E.mean() * 0.1 # bars to MPa
    g.thermal_exp_coeff = m.idata.rock.th_expn # 1/°C

    def get_thm_solution(point):
        # find an index of the cell, closest to the desired point
        cell = ((centroids[:, 0] - point[0]) ** 2 + (centroids[:, 1] - point[1]) ** 2 + (centroids[:, 2] - point[2]) ** 2).argmin()
        uz_thm = uz_last[cell]
        return uz_thm

    def get_proxy_solution(point):
        eval_points = np.zeros((1,3))  # just one point
        eps = 1  # [m], to avoid r=0 for the integral in the geomech proxy 1/r
        eval_points[0] = np.array([point[1]+eps, point[0]+eps, point[2]+eps]) # Y,X,Z
        upx1, upy1, upz1, utx1, uty1, utz1 = g.calc_displacements_cpp(eval_points, prisms, delta_pressure, delta_temperature)
        uz_proxy = upz1[0]
        return uz_proxy

    point = np.array([centroids[:, 0].mean(), centroids[:, 1].mean(), centroids[:, 2].mean()])  # middle point

    # compare 1 line along z-axis
    z_min = 0.
    z_max = centroids[:, 2].max() + 3000.
    z_range = np.arange(z_min, z_max, 100)

    uz_thm = []
    uz_prx = []
    for z in z_range:
        point[2] = z
        uz_thm.append(get_thm_solution(point))
        uz_prx.append(get_proxy_solution(point))

    from matplotlib import pyplot as plt
    plt.plot(uz_thm, z_range, label='uz_thm')
    plt.plot(uz_prx, z_range, label='uz_prx')
    plt.gca().invert_yaxis()
    plt.xlabel('Vertical displacement, m.')
    plt.ylabel('Depth, m.')
    plt.title('Vertical displacement, m.')
    plt.legend()
    plt.grid()
    plt.show()

    # compare 1 point and print
    point[2] = 0. # surface
    uz_thm = get_thm_solution(point)
    uz_prx = get_proxy_solution(point)
    print('point ', point)
    print('THM   ', 'uz=', uz_thm, 'm.')
    print('Proxy ', 'uz=', uz_prx, 'm.')

run_geomech_proxy(case='6_6_5')

#run_geomech_proxy(case='24_24_12')