import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
import meshio
import os
from scipy.interpolate import griddata
rcParams["text.usetex"]=False
plt.rc('xtick',labelsize=16)
plt.rc('ytick',labelsize=16)
font = {'family' : 'normal',
        'size'   : 18}
plt.rc('legend',fontsize=14)

def read_vtk(filename, props):
    mesh = meshio.read(filename=filename)

    # cell data
    centers = np.empty([0, 3])
    cell_data = {}
    for geom_name, geom in mesh.cells_dict.items():
        centers = np.append(centers, np.average(mesh.points[geom], axis=1), axis=0)
        for prop in props:
            if prop in mesh.cell_data_dict:
                if prop not in cell_data: cell_data[prop] = []
                cell_data[prop].append(mesh.cell_data_dict[prop][geom_name])

    # point data
    points = mesh.points
    point_data = {}
    for prop_name, prop in mesh.point_data.items():
        if prop_name in props:
            point_data[prop_name] = prop

    return centers, cell_data, points, point_data

def write_to_vtk(orig_filename, filename, data):
    # Temporarily store mesh_data in copy:
    Mesh = meshio.read(orig_filename)
    # mesh = copy.copy(Mesh)

    available_geometries = ['hexahedron', 'wedge', 'tetra', 'quad', 'triangle']

    props_num = len(data)

    # Matrix
    geom_id = 0
    # Mesh.cells = {}
    cell_data = {}
    for ith_geometry in Mesh.cells_dict.keys():
        if ith_geometry in available_geometries:
            # Mesh.cells[ith_geometry] = Mesh.cells_dict[ith_geometry]
            # Add matrix data to dictionary:
            for prop, cur_data in data.items():
                if prop not in cell_data: cell_data[prop] = []
                cell_data[prop].append(cur_data)
        geom_id += 1

    #calc_tetra_volumes(mesh.cells[1].data[mesh.cell_data_dict['gmsh:physical']['tetra'] == 97], mesh.points)

    # Store solution for each time-step:
    mesh = meshio.Mesh(
        Mesh.points,
        Mesh.cells,
        cell_data=cell_data)
    meshio.write(filename, mesh)

def compare_initial_state(darts_folder, geos_folder):
    # Darts
    darts_props = ['E', 'perm', 'poro', 'pressure', 'tot_stress', 'ux', 'uy', 'uz']
    darts_centers, darts_data, __, __ = read_vtk(darts_folder + '/solution0.vtk', darts_props)
    # Geos
    geos_props = ['totalDisplacement', 'pressure',
                  'rockPerm_permeability', 'rockPorosity_referencePorosity',
                  'skeleton_stress', 'skeleton_bulkModulus']
    geos_centers, geos_cell_data, geos_points, geos_point_data = read_vtk(geos_folder + '/rank_0.vtu', geos_props)
    biot = 1.0
    nu = 0.2
    total_stress_geos = geos_cell_data['skeleton_stress'][0] - biot * geos_cell_data['pressure'][0][:,np.newaxis]
    darts_perm = np.column_stack((darts_data['perm'][0][:, 0], darts_data['perm'][0][:, 4], darts_data['perm'][0][:, 8]))
    young_modulus_geos = 3 * geos_cell_data['skeleton_bulkModulus'][0] * (1 - 2 * nu)
    darts_cell_displacements = np.vstack((darts_data['ux'][0], darts_data['uy'][0], darts_data['uz'][0])).T
    geosx_cell_displacements = griddata(geos_points, geos_point_data['totalDisplacement'], darts_centers, method='linear')

    # write comparison
    md = 0.9869 * 1.e-15
    output_data = {'youngModulus_geos': young_modulus_geos / 1.e+6,
                   'youngModulus_darts': darts_data['E'][0] / 10,
                   'youngModulus_difference': young_modulus_geos / 1.e+6 - darts_data['E'][0] / 10,
                   'permeability_geos': geos_cell_data['rockPerm_permeability'][0] / md,
                   'permeability_darts': darts_perm,
                   'permeability_difference': geos_cell_data['rockPerm_permeability'][0] / md - darts_perm,
                   'porosity_geos': geos_cell_data['rockPorosity_referencePorosity'][0],
                   'porosity_darts': darts_data['poro'][0],
                   'porosity_difference': geos_cell_data['rockPorosity_referencePorosity'][0] - darts_data['poro'][0],
                   'displacements_geos': geosx_cell_displacements,
                   'displacements_darts': darts_cell_displacements,
                   'displacements_difference': geosx_cell_displacements - darts_cell_displacements,
                   'total_stress_geosx': total_stress_geos / 1.e+6,
                   'total_stress_darts': darts_data['tot_stress'][0] / 10,
                   'total_stress_difference': total_stress_geos / 1.e+6 - darts_data['tot_stress'][0] / 10,
                   'pressure_geosx': geos_cell_data['pressure'][0] / 1.e+6,
                   'pressure_darts': darts_data['pressure'][0] / 10,
                   'pressure_difference': geos_cell_data['pressure'][0] / 1.e+6 - darts_data['pressure'][0] / 10 }
    write_to_vtk(orig_filename=darts_folder + '/solution0.vtk',
                 filename=darts_folder + '/comparison.vtk',
                 data=output_data)



darts_folder = './sol_cpp_10_10_10'
geos_folder = './geos/vtkOutput/000001/mesh/Level0/Domain'
compare_initial_state(darts_folder=darts_folder, geos_folder=geos_folder)