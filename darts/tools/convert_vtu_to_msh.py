import copy
import os

import meshio
import numpy as np


def convert_vtu_to_msh(vtkfile, msh_filename=None):
    '''
    reads a mesh in vtu format (paraview) and outputs a .msh file in gmsh format
    Note that gmsh doesn't properly recognise physical tags for surfaces for some reason, but once loaded with meshio, it works fine
    :param vtkfile: input mesh to convert
    :param msh_filename: output mesh filename, if None, the name will be the same as input with .msh extension
    '''
    # Check if msh file exists and is up to date
    if msh_filename is None:
        msh_filename = vtkfile.split('.')[0] + '.msh'
    if os.path.exists(msh_filename):
        vtu_mtime = os.path.getmtime(vtkfile)
        msh_mtime = os.path.getmtime(msh_filename)
        if msh_mtime >= vtu_mtime:
            print(f"{msh_filename} is up to date with {vtkfile}. Skipping conversion.")
            return 0

    # Temporarily store mesh_data in copy:
    Mesh = meshio.read(vtkfile)
    msh = copy.copy(Mesh)

    types = msh.cell_data['matType']  # meshIt
    # types = msh.cell_data['CellEntityIds']  # geos

    tet_types, tri_types = types

    n_tri_types = tri_types.max()

    n_tetrahedrons = tet_types.size
    n_triangles = tri_types.size

    tags_m = {}
    tags_s = {}
    # 0: matrix
    tags_m['MATRIX'] = 0
    # 1-6 boundaries: 1: z-, 2: z+, 3: x-, 4: y-, 5: y+, 6: x+
    tags_s['BND_X-'] = 3
    tags_s['BND_X+'] = 6
    tags_s['BND_Y-'] = 4
    tags_s['BND_Y+'] = 5
    tags_s['BND_Z-'] = 1
    tags_s['BND_Z+'] = 2
    # 7-... : faults
    for t in range(7, n_tri_types + 1):
        tags_s['FAULT_' + str(t)] = t

    tags = {}
    tags.update(tags_m)
    tags.update(tags_s)

    face_tags = np.zeros(n_triangles, dtype=int) - 999
    for t in tags_s.keys():
        ti = tags_s[t]
        face_tags[tri_types == ti] = ti

    cell_tags = np.zeros(n_tetrahedrons, dtype=int) - 999
    for t in tags_m.keys():
        ti = tags_m[t]
        cell_tags[tet_types == ti] = ti

    cell_data = {}
    cell_data["gmsh:physical"] = [cell_tags, face_tags]
    cell_data["gmsh:geometrical"] = [
        np.zeros_like(cell_tags) + 4,
        np.zeros_like(face_tags) + 2,
    ]  # 4 - tetra, 2 - triangle

    print('Writing ', msh_filename)

    mesh = meshio.Mesh(Mesh.points, Mesh.cells, cell_data=cell_data)

    meshio.write(msh_filename, mesh, file_format="gmsh22")
    # meshio.write(msh_filename, mesh, file_format="xdmf")

    # import pickle
    # with open("all_faults_1/cell_data.pkl","wb") as f:
    #    pickle.dump(cell_data, f)

    return 0


# convert_vtu_to_msh('all_faults_1/all_faults_1.vtu')
