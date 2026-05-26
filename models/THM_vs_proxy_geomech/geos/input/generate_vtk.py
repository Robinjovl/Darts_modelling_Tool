import numpy as np
from math import inf, pi
import meshio
import os
import copy
import math
from darts.tools.keyword_file_tools import load_single_keyword

def write_to_vtk_with_faces(inputfolder, outname):
    # Temporarily store mesh_data in copy:
    mshfile = inputfolder + '/spe10.msh'
    Mesh = meshio.read(mshfile)
    mesh = copy.copy(Mesh)

    available_geometries = ['hexahedron', 'wedge', 'tetra', 'quad', 'triangle']
    available_3d_geometries = ['hexahedron', 'wedge', 'tetra']

    nx, ny, nz = int(inputfolder.split('_')[-3]), \
                int(inputfolder.split('_')[-2]), \
                int(inputfolder.split('_')[-1])

    porosity = np.flip(np.swapaxes(load_single_keyword(inputfolder + '/poro.txt', 'PORO', cache=0).
                                   reshape(nz, ny, nx), 0, 2), axis=2).flatten()
    permeability = np.flip(np.swapaxes(load_single_keyword(inputfolder + '/perm.txt', 'PERM', cache=0).
                                       reshape(nz, ny, nx, 3), 0, 2), axis=2).flatten()
    E = np.flip(np.swapaxes(load_single_keyword(inputfolder + '/young.txt', 'YOUNG', cache=0).
                            reshape(nz, ny, nx), 0, 2), axis=2).flatten()
    nu = 0.2
    biot = 1.0
    p_init = np.flip(np.swapaxes(load_single_keyword(inputfolder + '/ref_pres.txt', 'REF_PRESSURE', cache=0).
                                 reshape(nz, ny, nx), 0, 2), axis=2).flatten()

    K = E / 3 / (1 - 2 * nu) * 1.e+10
    K_grain = 1.e+27 # K / (1 - biot)
    # G = E / 2 / (1 + nu)
    md = 0.9869 * 1.e-15
    cell_property = {'CellEntityIds': [], 'bulkModulus': K, 'biotCoefficient': biot * np.ones(K.size),
                        'porosity': porosity, 'referencePressure': p_init * 1.e+5,
                        'permeability': permeability.reshape((porosity.size, 3)) * md }
    # Matrix
    geom_id = 0
    Mesh.cells = {}
    cell_data = {}
    for ith_geometry in mesh.cells_dict.keys():
        if ith_geometry in available_3d_geometries:
            Mesh.cells[ith_geometry] = mesh.cells_dict[ith_geometry]
            # Add matrix data to dictionary:
            for key, prop in cell_property.items():
                if key not in cell_data: cell_data[key] = []
                if key == 'CellEntityIds':
                    cell_data[key].append(
                        np.abs(np.array(mesh.cell_data_dict['gmsh:physical'][ith_geometry], dtype=np.int64),
                               dtype=np.int64))
                    continue
                if ith_geometry in available_3d_geometries:
                    cell_data[key].append(prop)
                else:
                    cell_data[key].append([])

        geom_id += 1

    #calc_tetra_volumes(mesh.cells[1].data[mesh.cell_data_dict['gmsh:physical']['tetra'] == 97], mesh.points)

    # Store solution for each time-step:
    mesh = meshio.Mesh(
        Mesh.points,
        Mesh.cells,
        cell_data=cell_data)
    meshio.write(outname, mesh)

    return 0

inputfolder = '../../meshes/data_10_10_10'
outname = 'data_10_10_10/geos_input.vtu'
write_to_vtk_with_faces(inputfolder=inputfolder, outname=outname)
