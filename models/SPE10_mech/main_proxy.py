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

def read_vtk_darts_solution(timestep : int):
    #folder = 'sol_cpp_single_phase_10_10_10'
    folder = 'sol_cpp_single_phase_16_16_12'
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

def run_geomech_proxy():
    msh_initial = read_vtk_darts_solution(timestep=0)
    p_initial = np.array(msh_initial.cell_data['pressure']).flatten()

    msh_last    = read_vtk_darts_solution(timestep=20)
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
    coord = centroids[:, 0].mean(), centroids[:, 1].mean(), 0 #centroids[:, 2].mean()  # middle point
    cell = ((centroids[:, 0] - coord[0]) ** 2 + (centroids[:, 1] - coord[1]) ** 2 + (centroids[:, 2] - coord[2]) ** 2).argmin()

    # Compute center of the cell
    y1, y2, x1, x2, z1, z2 = prisms[cell]
    center = np.array([(y2 + y1)/2, (x2 + x1)/2, (z2 + z1)/2])
    # define evaluation point for proxy
    center_surface = center.copy()
    center_surface[2] = 0.
    eval_points = np.zeros((1,3))
    eval_points[0] = np.array([center_surface])

    from geomechanics import geomech
    g = geomech()

    # just to set input data
    from model import Model
    m = Model(model_folder='data_16_16_12', physics_type='single_phase', uniform_props=False)

    # elastic constants
    g.poisson = m.idata.rock.nu
    g.young = m.idata.rock.E.mean() * 0.1 # bars to MPa
    g.thermal_exp_coeff = m.idata.rock.th_expn # 1/°C
    upx1, upy1, upz1, utx1, uty1, utz1 = g.calc_displacements_cpp(eval_points, prisms, delta_pressure, delta_temperature)

    print('THM   uz=', uz_last[cell], 'm.')
    print('Proxy uz=', -upz1[0], 'm.')

run_geomech_proxy()