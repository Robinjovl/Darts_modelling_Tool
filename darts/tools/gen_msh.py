import copy
import os
import sys

import gmsh
import meshio
import numpy as np


def generate_box_3d(
    X: float,
    Y: float,
    Z: float,
    NX: int,
    NY: int,
    NZ: int,
    tags: dict,
    filename: str = None,
    is_transfinite: bool = True,
    is_recombine: bool = True,
    refinement_mult: bool = 1.0,
    popup=False,
):
    '''
    generates a rectangular-box structured-like mesh with hexahedron (right prism) cells in the unstructured mesh format (gmsh 2).
    :param X: a box size ialong X-axis
    :param Y: a box size ialong Y-axis
    :param Z: a box size ialong Z-axis
    :param NX: a number of cells along X-axis
    :param NY: a number of cells along Y-axis
    :param NZ: a number of cells along Z-axis
    :param tags: a dictionary of physical tags, should contain keys: 'matrix', 'bnd_xm', 'bnd_xp', 'bnd_ym', 'bnd_yp', 'bnd_zm', 'bnd_zp'
    :param filename: the file name for the output .msh file (if None, will be generated)
    :param is_transfinite: mesh generation option, should be True for the structured-like case
    :param is_recombine: mesh generation option, should be True for the structured-like case
    :param refinement_mult:
    :param popup: shoe gmsh GUI after mesh generation
    :return filename
    '''
    gmsh.initialize()

    gmsh.model.add("box_3d")

    lc = X / NX / refinement_mult

    n_max = 2  # 2 points
    y = [0, X]
    x = [0, Y]
    z = [0, Z]
    for k, z_cur in enumerate(z):
        for j, y_cur in enumerate(y):
            for i, x_cur in enumerate(x):
                id = i + n_max * j + n_max**2 * k
                x0 = x_cur
                y0 = y_cur
                z0 = z_cur
                gmsh.model.geo.addPoint(x0, y0, z0, lc, id)

    nx_mult = refinement_mult
    nx = np.array(nx_mult * np.array([NX], dtype=np.int32), dtype=np.int32)

    ny_mult = refinement_mult
    ny = np.array(ny_mult * np.array([NY], dtype=np.int32), dtype=np.int32)

    nz_mult = 1  # refinement_mult
    nz = np.array(nz_mult * np.array([NZ], dtype=np.int32), dtype=np.int32)

    # add x-lines
    for k in range(0, len(z)):
        for j in range(0, len(y)):
            for i in range(0, len(x) - 1):
                p_id1 = i + n_max * j + n_max**2 * k
                p_id2 = i + 1 + n_max * j + n_max**2 * k
                id = i + 1 + n_max * j + n_max**2 * k
                gmsh.model.geo.addLine(p_id1, p_id2, id)
                if is_transfinite:
                    gmsh.model.geo.mesh.setTransfiniteCurve(id, nx[i] + 1)
    # add y-lines
    for k in range(0, len(z)):
        for j in range(0, len(y) - 1):
            for i in range(0, len(x)):
                p_id1 = i + n_max * j + n_max**2 * k
                p_id2 = i + n_max * (j + 1) + n_max**2 * k
                id = i + 1 + n_max * j + n_max**2 * k + n_max**3
                gmsh.model.geo.addLine(p_id1, p_id2, id)
                if is_transfinite:
                    gmsh.model.geo.mesh.setTransfiniteCurve(id, ny[j] + 1)
    # add z-lines
    for k in range(0, len(z) - 1):
        for j in range(0, len(y)):
            for i in range(0, len(x)):
                p_id1 = i + n_max * j + n_max**2 * k
                p_id2 = i + n_max * j + n_max**2 * (k + 1)
                id = i + 1 + n_max * j + n_max**2 * k + 2 * n_max**3
                gmsh.model.geo.addLine(p_id1, p_id2, id)
                if is_transfinite:
                    gmsh.model.geo.mesh.setTransfiniteCurve(id, nz[k] + 1)

    # # add curve loops & surfaces
    surfaces = []
    # x-y
    z_plus = []
    z_minus = []
    for k in range(0, len(z)):
        for j in range(0, len(y) - 1):
            for i in range(0, len(x) - 1):
                id = i + 1 + n_max * j + n_max**2 * k
                # [bottom, right, top, left]
                l_id1 = i + 1 + n_max * j + n_max**2 * k
                l_id2 = i + 2 + n_max * j + n_max**2 * k + n_max**3
                l_id3 = i + 1 + n_max * (j + 1) + n_max**2 * k
                l_id4 = i + 1 + n_max * j + n_max**2 * k + n_max**3
                gmsh.model.geo.addCurveLoop([l_id1, l_id2, -l_id3, -l_id4], id)
                gmsh.model.geo.addPlaneSurface([id], id)
                if is_transfinite:
                    gmsh.model.geo.mesh.setTransfiniteSurface(
                        id,
                        "Left",
                        [
                            i + n_max * j + n_max**2 * k,
                            i + 1 + n_max * j + n_max**2 * k,
                            i + 1 + n_max * (j + 1) + n_max**2 * k,
                            i + n_max * (j + 1) + n_max**2 * k,
                        ],
                    )
                if is_recombine:
                    gmsh.model.geo.mesh.setRecombine(2, id)
                surfaces.append((2, id))
                if k == 0:
                    z_minus.append(id)
                elif k == len(z) - 1:
                    z_plus.append(id)
    # y-z
    x_plus = []
    x_minus = []
    fault = []
    for k in range(0, len(z) - 1):
        for j in range(0, len(y) - 1):
            for i in range(0, len(x)):
                id = i + 1 + n_max * j + n_max**2 * k + n_max**3
                # [bottom, right, top, left]
                l_id1 = i + 1 + n_max * j + n_max**2 * k + n_max**3
                l_id2 = i + 1 + n_max * (j + 1) + n_max**2 * k + 2 * n_max**3
                l_id3 = i + 1 + n_max * j + n_max**2 * (k + 1) + n_max**3
                l_id4 = i + 1 + n_max * j + n_max**2 * k + 2 * n_max**3
                gmsh.model.geo.addCurveLoop([l_id1, l_id2, -l_id3, -l_id4], id)
                gmsh.model.geo.addPlaneSurface([id], id)
                if is_transfinite:
                    gmsh.model.geo.mesh.setTransfiniteSurface(
                        id,
                        "Left",
                        [
                            i + n_max * j + n_max**2 * k,
                            i + n_max * (j + 1) + n_max**2 * k,
                            i + n_max * (j + 1) + n_max**2 * (k + 1),
                            i + n_max * j + n_max**2 * (k + 1),
                        ],
                    )
                if is_recombine:
                    gmsh.model.geo.mesh.setRecombine(2, id)
                surfaces.append((2, id))
                if i == 0:
                    x_minus.append(id)
                elif i == len(x) - 1:
                    x_plus.append(id)
                elif i == 1:
                    fault.append(id)
    # x-z
    y_plus = []
    y_minus = []
    for k in range(0, len(z) - 1):
        for j in range(0, len(y)):
            for i in range(0, len(x) - 1):
                id = i + 1 + n_max * j + n_max**2 * k + 2 * n_max**3
                # [bottom, right, top, left]
                l_id1 = i + 1 + n_max * j + n_max**2 * k
                l_id2 = i + 2 + n_max * j + n_max**2 * k + 2 * n_max**3
                l_id3 = i + 1 + n_max * j + n_max**2 * (k + 1)
                l_id4 = i + 1 + n_max * j + n_max**2 * k + 2 * n_max**3
                gmsh.model.geo.addCurveLoop([l_id1, l_id2, -l_id3, -l_id4], id)
                gmsh.model.geo.addPlaneSurface([id], id)
                if is_transfinite:
                    gmsh.model.geo.mesh.setTransfiniteSurface(
                        id,
                        "Left",
                        [
                            i + n_max * j + n_max**2 * k,
                            i + 1 + n_max * j + n_max**2 * k,
                            i + 1 + n_max * j + n_max**2 * (k + 1),
                            i + n_max * j + n_max**2 * (k + 1),
                        ],
                    )
                if is_recombine:
                    gmsh.model.geo.mesh.setRecombine(2, id)
                surfaces.append((2, id))
                if j == 0:
                    y_minus.append(id)
                elif j == len(y) - 1:
                    y_plus.append(id)
    ## surface loops & volumes
    reservoir = []
    for k in range(0, len(z) - 1):
        for j in range(0, len(y) - 1):
            for i in range(0, len(x) - 1):
                id = i + n_max * j + n_max**2 * k
                # z-axis
                s_id1 = i + 1 + n_max * j + n_max**2 * k
                s_id2 = i + 1 + n_max * j + n_max**2 * (k + 1)
                # x-axis
                s_id3 = i + 1 + n_max * j + n_max**2 * k + n_max**3
                s_id4 = i + 2 + n_max * j + n_max**2 * k + n_max**3
                # y-axis
                s_id5 = i + 1 + n_max * j + n_max**2 * k + 2 * n_max**3
                s_id6 = i + 1 + n_max * (j + 1) + n_max**2 * k + 2 * n_max**3
                gmsh.model.geo.addSurfaceLoop(
                    [s_id1, s_id2, s_id3, s_id4, s_id5, s_id6], id
                )
                gmsh.model.geo.addVolume([id], id)

                if is_transfinite:
                    gmsh.model.geo.mesh.setTransfiniteVolume(
                        id,
                        [
                            i + n_max * j + n_max**2 * k,
                            i + 1 + n_max * j + n_max**2 * k,
                            i + 1 + n_max * (j + 1) + n_max**2 * k,
                            i + n_max * (j + 1) + n_max**2 * k,
                            i + n_max * j + n_max**2 * (k + 1),
                            i + 1 + n_max * j + n_max**2 * (k + 1),
                            i + 1 + n_max * (j + 1) + n_max**2 * (k + 1),
                            i + n_max * (j + 1) + n_max**2 * (k + 1),
                        ],
                    )
                reservoir.append(id)

    gmsh.model.geo.synchronize()

    # boundary tags
    gmsh.model.addPhysicalGroup(2, x_minus, tags['BND_X-'])
    gmsh.model.addPhysicalGroup(2, x_plus, tags['BND_X+'])
    gmsh.model.addPhysicalGroup(2, y_minus, tags['BND_Y-'])
    gmsh.model.addPhysicalGroup(2, y_plus, tags['BND_Y+'])
    gmsh.model.addPhysicalGroup(2, z_minus, tags['BND_Z-'])
    gmsh.model.addPhysicalGroup(2, z_plus, tags['BND_Z+'])

    # volumes
    gmsh.model.addPhysicalGroup(3, reservoir, tags['MATRIX'])

    # since we read the mesh with MshIO in c++ discretizer and it doesn't work with the msh format 4
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.1)
    gmsh.model.mesh.generate(3)

    if filename is None:  # default file names
        if is_transfinite and is_recombine:
            filename = "transfinite_3d.msh"
        elif is_transfinite and not is_recombine:
            filename = "transfinite_wedges_3d.msh"
        elif not is_transfinite and is_recombine:
            filename = "unstructured_hexahedrons_3d.msh"
        elif not is_transfinite and not is_recombine:
            filename = "tetra_3d.msh"

    print('Writing ', filename)
    gmsh.write(filename)

    # Launch the GUI to see the results:
    if popup:
        gmsh.fltk.run()

    gmsh.finalize()

    return filename


def write_to_vtk_with_faces(mshfile):
    '''
    reads a mesh in gmsh format and outputs a .vtu file in VTK format
    :param mshfile: input mesh to convert
    '''
    # Temporarily store mesh_data in copy:
    Mesh = meshio.read(mshfile)
    mesh = copy.copy(Mesh)

    available_geometries = ['hexahedron', 'wedge', 'tetra', 'quad', 'triangle']

    cell_property = ['CellEntityIds']
    props_num = len(cell_property)

    # Matrix
    geom_id = 0
    Mesh.cells = {}
    cell_data = {}
    for ith_geometry in mesh.cells_dict.keys():
        if ith_geometry in available_geometries:
            Mesh.cells[ith_geometry] = mesh.cells_dict[ith_geometry]
            # Add matrix data to dictionary:
            for i in range(props_num):
                if cell_property[i] not in cell_data:
                    cell_data[cell_property[i]] = []
                cell_data[cell_property[i]].append(
                    np.abs(
                        np.array(
                            mesh.cell_data_dict['gmsh:physical'][ith_geometry],
                            dtype=np.int64,
                        ),
                        dtype=np.int64,
                    )
                )
        geom_id += 1

    vtk_filename = mshfile.split('.')[0] + '.vtu'
    print('Writing ', vtk_filename)
    mesh = meshio.Mesh(Mesh.points, Mesh.cells, cell_data=cell_data)
    meshio.write(vtk_filename, mesh)

    return 0


if __name__ == '__main__':
    tags = dict()
    tags['BND_X-'] = 991
    tags['BND_X+'] = 992
    tags['BND_Y-'] = 993
    tags['BND_Y+'] = 994
    tags['BND_Z-'] = 995
    tags['BND_Z+'] = 996
    tags['MATRIX'] = 99991

    filename = generate_box_3d(
        X=300.0,
        Y=200.0,
        Z=100.0,
        NX=30,
        NY=20,
        NZ=10,
        filename=None,
        tags=tags,
        is_transfinite=True,
        is_recombine=True,
        refinement_mult=1,
    )
    write_to_vtk_with_faces(mshfile=filename)
