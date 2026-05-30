from os import makedirs

import numpy as np
import meshio
import os
import copy
import gmsh
import sys



def generate_box_3d(X : float, Y : float, Z : float, NX : int, NY : int, NZ : int, tags : dict, filename : str = None,
                    is_transfinite : bool = True, is_recombine : bool  = True, refinement_mult : bool = 1.0,
                    fault_refinement_mult = 1.0, fault_angle : float = None, z_minus_hybrid = False, two_rocks = False,
                    msh_ver=2.1, popup=False, Xc=None, Yc=None, Zc=None, rsv_top=None, rsv_bottom=None, 
                    rsv_x1=None, rsv_x2=None, rsv_y1=None, rsv_y2=None):
    '''
    generates a rectangular-box structured-like mesh with hexahedron (right prism) cells in the unstructured mesh format (gmsh 2).
    :param X: a box size along X-axis
    :param Y: a box size along Y-axis
    :param Z: a box size along Z-axis
    :param NX: a number of cells along X-axis
    :param NY: a number of cells along Y-axis
    :param NZ: a number of cells along Z-axis
    :param tags: a dictionary of physical tags, should contain keys: 'matrix', 'bnd_xm', 'bnd_xp', 'bnd_ym', 'bnd_yp', 'bnd_zm', 'bnd_zp'
    :param filename: the file name for the output .msh file (if None, will be generated)
    :param is_transfinite: mesh generation option, should be True for the structured-like case
    :param is_recombine: mesh generation option, should be True for the structured-like case
    :param refinement_mult:
    :param fault_angle: fault angle, degrees
    :param z_minus_hybrid: bool - create a separate surface for the middle cell at Z- boundary for the fixed+roller BC
    :param popup: shoe gmsh GUI after mesh generation
    :return filename
    '''
    gmsh.initialize()

    # Suppress all Gmsh output to the terminal
    gmsh.option.setNumber("General.Terminal", 0)

    # You can also control the verbosity level, where 0 is quiet, 1 is warnings, etc.
    # gmsh.option.setNumber("General.Verbosity", 0) # This might also be useful

    gmsh.model.add("box_3d")

    lc = X / NX / refinement_mult

    x = [0, X]
    y = [0, Y]
    if fault_angle is None:  # no fault
        z = [0, Z]
    else: # with a fault
        a = (Z - X / np.tan(np.radians(fault_angle))) /2. # the distance from a fault to z bottom (z=0) surface
        z = [0, lambda x_: a + x_ / np.tan(np.radians(fault_angle)), Z]
        lc_fault = lc * fault_refinement_mult

     # field scale model, refined in the middle, coarse in surrounding, z up to the surface
    if Xc is not None:
        x = Xc
        y = Yc
        z = Zc

        nx = len(x) - 1
        ny = len(y) - 1
        nz = len(z) - 1
        suffix = str(nx) + '_' + str(ny) + '_' + str(nz)
        folder = os.path.join('meshes', suffix)
        os.makedirs(folder, exist_ok=True)
        filename = os.path.join(folder, 'mesh.msh')

    # add 2 physical surfaces for Z- boundary to set hybrid mechanical boundary conditions there:
    # one cell in the middle with FIXED BC, and the rest is ROLLER
    # this is needed to avoid stress accumulation when setting FIXED BC to the whole Z- boundary surface
    if z_minus_hybrid:
        assert NX % 2 != 0 and NY % 2 != 0, 'NX and NY should be odd if z_minus_hybrid=True'

        eps_x = eps_y = lc * 0.1

        mid_x = X * 0.5
        if is_transfinite:
            eps_x = X / NY * 0.5  # 0.5 dx
        x_additional = [mid_x - eps_x, mid_x + eps_x]

        mid_y = Y * 0.5
        if is_transfinite:
            eps_y = Y / NY * 0.5  # 0.5 dy
        y_additional = [mid_y - eps_y, mid_y + eps_y]

        x = [x[0]] + x_additional + [x[-1]]
        y = [y[0]] + y_additional + [y[-1]]

    n_max_x = len(x)
    n_max_xy = n_max_x * len(y)
    n_max_xyz = n_max_xy * len(z)
    
    for k, z_cur in enumerate(z):
        for j, y_cur in enumerate(y):
            for i, x_cur in enumerate(x):
                id = i + n_max_x * j + n_max_xy * k
                x0 = x_cur
                y0 = y_cur
                z0, lc0 = [z_cur, lc] if np.isscalar(z_cur) else [z_cur(x_cur), lc_fault]
                gmsh.model.geo.addPoint(x0, y0, z0, lc0, id)

    nx_mult = refinement_mult
    NX_A = [NX] if len(x) == 1 else [NX//2, 1, NX//2] # one special cell in the middle

    ny_mult = refinement_mult
    NY_A = [NY] if len(y) == 1 else [NY // 2, 1, NY // 2] # one special cell in the middle

    nz_mult = 1#refinement_mult
    NZ_F = [NZ] if len(z) == 1 else [NZ // 2, NZ // 2] # fault

    if Xc is not None:
        NX_A = [1]*len(x)
        NY_A = [1]*len(y)
        NZ_F = [1]*len(z)

    nx = np.array(nx_mult * np.array(NX_A, dtype=np.int32), dtype=np.int32)
    ny = np.array(ny_mult * np.array(NY_A, dtype=np.int32), dtype=np.int32)
    nz = np.array(nz_mult * np.array(NZ_F, dtype=np.int32), dtype=np.int32)

    # add x-lines
    for k in range(0, len(z)):
        for j in range(0, len(y)):
            for i in range(0, len(x) - 1):
                p_id1 = i + n_max_x * j + n_max_xy * k
                p_id2 = i + 1 + n_max_x * j + n_max_xy * k
                id = i + 1 + n_max_x * j + n_max_xy * k
                gmsh.model.geo.addLine(p_id1, p_id2, id)
                if is_transfinite:
                    gmsh.model.geo.mesh.setTransfiniteCurve(id, nx[i] + 1)
    # add y-lines
    for k in range(0, len(z)):
        for j in range(0, len(y) - 1):
            for i in range(0, len(x)):
                p_id1 = i + n_max_x * j + n_max_xy * k
                p_id2 = i + n_max_x * (j + 1) + n_max_xy * k
                id = i + 1 + n_max_x * j + n_max_xy * k + n_max_xyz
                gmsh.model.geo.addLine(p_id1, p_id2, id)
                if is_transfinite:
                    gmsh.model.geo.mesh.setTransfiniteCurve(id, ny[j] + 1)
    # add z-lines
    for k in range(0, len(z) - 1):
        for j in range(0, len(y)):
            for i in range(0, len(x)):
                p_id1 = i + n_max_x * j + n_max_xy * k
                p_id2 = i + n_max_x * j + n_max_xy * (k + 1)
                id = i + 1 + n_max_x * j + n_max_xy * k + 2 * n_max_xyz
                gmsh.model.geo.addLine(p_id1, p_id2, id)
                if is_transfinite:
                    gmsh.model.geo.mesh.setTransfiniteCurve(id, nz[k] + 1)

    # for debugging
    #gmsh.model.geo.synchronize()
    #gmsh.fltk.run()

    # # add curve loops & surfaces
    surfaces = []
    # x-y
    z_plus = []
    z_minus_fixed = []
    z_minus = []
    fault = []
    for k in range(0, len(z)):
        for j in range(0, len(y) - 1):
            for i in range(0, len(x) - 1):
                id = i + 1 + n_max_x * j + n_max_xy * k
                # [bottom, right, top, left]
                l_id1 = i + 1 + n_max_x * j + n_max_xy * k
                l_id2 = i + 2 + n_max_x * j + n_max_xy * k + n_max_xyz
                l_id3 = i + 1 + n_max_x * (j + 1) + n_max_xy * k
                l_id4 = i + 1 + n_max_x * j + n_max_xy * k + n_max_xyz
                gmsh.model.geo.addCurveLoop([l_id1, l_id2, -l_id3, -l_id4], id)
                gmsh.model.geo.addPlaneSurface([id], id)
                if is_transfinite:
                    gmsh.model.geo.mesh.setTransfiniteSurface(id, "Left",
                                                [i + n_max_x * j + n_max_xy * k,
                                                 i + 1 + n_max_x * j + n_max_xy * k,
                                                 i + 1 + n_max_x * (j + 1) + n_max_xy * k,
                                                 i + n_max_x * (j + 1) + n_max_xy * k])
                if is_recombine:
                    gmsh.model.geo.mesh.setRecombine(2, id)
                surfaces.append((2, id))
                if k == 0:
                    if 0 < i < len(x) - 2 and 0 < j < len(y) - 2:
                        z_minus_fixed.append(id)
                    else:
                        z_minus.append(id)
                elif k == len(z) - 1:
                    z_plus.append(id)
                elif k == 1:
                    fault.append(id)
    # y-z
    x_plus = []
    x_minus = []
    for k in range(0, len(z) - 1):
        for j in range(0, len(y) - 1):
            for i in range(0, len(x)):
                id = i + 1 + n_max_x * j + n_max_xy * k + n_max_xyz
                # [bottom, right, top, left]
                l_id1 = i + 1 + n_max_x * j + n_max_xy * k + n_max_xyz
                l_id2 = i + 1 + n_max_x * (j + 1) + n_max_xy * k + 2 * n_max_xyz
                l_id3 = i + 1 + n_max_x * j + n_max_xy * (k + 1) + n_max_xyz
                l_id4 = i + 1 + n_max_x * j + n_max_xy * k + 2 * n_max_xyz
                gmsh.model.geo.addCurveLoop([l_id1, l_id2, -l_id3, -l_id4], id)
                gmsh.model.geo.addPlaneSurface([id], id)
                if is_transfinite:
                    gmsh.model.geo.mesh.setTransfiniteSurface(id, "Left",
                                                [i + n_max_x * j + n_max_xy * k,
                                                 i + n_max_x * (j + 1) + n_max_xy * k,
                                                 i + n_max_x * (j + 1) + n_max_xy * (k + 1),
                                                 i + n_max_x * j + n_max_xy * (k + 1)])
                if is_recombine:
                    gmsh.model.geo.mesh.setRecombine(2, id)
                surfaces.append((2, id))
                if i == 0:
                    x_minus.append(id)
                elif i == len(x) - 1:
                    x_plus.append(id)

    # x-z
    y_plus = []
    y_minus = []
    for k in range(0, len(z) - 1):
        for j in range(0, len(y)):
            for i in range(0, len(x) - 1):
                id = i + 1 + n_max_x * j + n_max_xy * k + 2 * n_max_xyz
                # [bottom, right, top, left]
                l_id1 = i + 1 + n_max_x * j + n_max_xy * k
                l_id2 = i + 2 + n_max_x * j + n_max_xy * k + 2 * n_max_xyz
                l_id3 = i + 1 + n_max_x * j + n_max_xy * (k + 1)
                l_id4 = i + 1 + n_max_x * j + n_max_xy * k + 2 * n_max_xyz
                gmsh.model.geo.addCurveLoop([l_id1, l_id2, -l_id3, -l_id4], id)
                gmsh.model.geo.addPlaneSurface([id], id)
                if is_transfinite:
                    gmsh.model.geo.mesh.setTransfiniteSurface(id, "Left",
                                                [i + n_max_x * j + n_max_xy * k,
                                                 i + 1 + n_max_x * j + n_max_xy * k,
                                                 i + 1 + n_max_x * j + n_max_xy * (k + 1),
                                                 i + n_max_x * j + n_max_xy * (k + 1)])
                if is_recombine:
                    gmsh.model.geo.mesh.setRecombine(2, id)
                surfaces.append((2, id))
                if j == 0:
                    y_minus.append(id)
                elif j == len(y) - 1:
                    y_plus.append(id)

    ## surface loops & volumes
    reservoir = []
    reservoir_1 = []
    reservoir_2 = []
    for k in range(0, len(z) - 1):
        for j in range(0, len(y) - 1):
            for i in range(0, len(x) - 1):
                id = i + n_max_x * j + n_max_xy * k
                # z-axis
                s_id1 = i + 1 + n_max_x * j + n_max_xy * k
                s_id2 = i + 1 + n_max_x * j + n_max_xy * (k + 1)
                # x-axis
                s_id3 = i + 1 + n_max_x * j + n_max_xy * k + n_max_xyz
                s_id4 = i + 2 + n_max_x * j + n_max_xy * k + n_max_xyz
                # y-axis
                s_id5 = i + 1 + n_max_x * j + n_max_xy * k + 2 * n_max_xyz
                s_id6 = i + 1 + n_max_x * (j + 1) + n_max_xy * k + 2 * n_max_xyz

                gmsh.model.geo.addSurfaceLoop([s_id1, s_id2, s_id3, s_id4, s_id5, s_id6], id)
                gmsh.model.geo.addVolume([id], id)

                if is_transfinite:
                    gmsh.model.geo.mesh.setTransfiniteVolume(id, [i + n_max_x * j + n_max_xy * k,
                                                                 i + 1 + n_max_x * j + n_max_xy * k,
                                                                 i + 1 + n_max_x * (j + 1) + n_max_xy * k,
                                                                 i + n_max_x * (j + 1) + n_max_xy * k,
                                                                 i + n_max_x * j + n_max_xy * (k + 1),
                                                                 i + 1 + n_max_x * j + n_max_xy * (k + 1),
                                                                 i + 1 + n_max_x * (j + 1) + n_max_xy * (k + 1),
                                                                 i + n_max_x * (j + 1) + n_max_xy * (k + 1) ])
                if Zc is None:
                    if two_rocks:
                        if k == 0:
                            reservoir_1.append(id)
                        else:
                            reservoir_2.append(id)
                    else:
                        reservoir.append(id)
                else:  # depths list is specified
                    if two_rocks:
                        from functools import reduce
                        rsv = reduce(np.logical_and, [ rsv_top >= z[k], z[k] > rsv_bottom,
                                                      rsv_y1 <= y[j],  y[j] <= rsv_y2,
                                                      rsv_x1 <= x[i],  x[i] <= rsv_x2])
                        if rsv:#rsv_top >= z[k] > rsv_bottom:
                            reservoir_1.append(id)
                        else:
                            reservoir_2.append(id)
                    else:
                        reservoir.append(id)

    gmsh.model.geo.synchronize()

    # boundary tags
    gmsh.model.addPhysicalGroup(2, x_minus, tags['BND_X-'])
    gmsh.model.addPhysicalGroup(2, x_plus, tags['BND_X+'])
    gmsh.model.addPhysicalGroup(2, y_minus, tags['BND_Y-'])
    gmsh.model.addPhysicalGroup(2, y_plus, tags['BND_Y+'])
    gmsh.model.addPhysicalGroup(2, z_minus, tags['BND_Z-'])
    if z_minus_hybrid:
        gmsh.model.addPhysicalGroup(2, z_minus_fixed, tags['BND_Z-F'])

    gmsh.model.addPhysicalGroup(2, z_plus, tags['BND_Z+'])

    if 'FRAC' in tags:
        gmsh.model.addPhysicalGroup(2, fault, tags['FRAC'])

    # volumes
    if two_rocks:
        gmsh.model.addPhysicalGroup(3, reservoir_1, tags['MATRIX_1'])
        gmsh.model.addPhysicalGroup(3, reservoir_2, tags['MATRIX_2'])
    else:
        gmsh.model.addPhysicalGroup(3, reservoir, tags['MATRIX_1'])

    # since we read the mesh with MshIO in c++ discretizer and it doesn't work with the msh format 4
    gmsh.option.setNumber("Mesh.MshFileVersion", msh_ver)
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

    # convert msh to vtk
    write_to_vtk_with_faces(mshfile=filename)

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
                if cell_property[i] not in cell_data: cell_data[cell_property[i]] = []
                cell_data[cell_property[i]].append(np.abs(np.array(mesh.cell_data_dict['gmsh:physical'][ith_geometry], dtype=np.int64), dtype=np.int64))
        geom_id += 1

    vtk_filename = mshfile.split('.')[0] + '.vtu'
    print('Writing ', vtk_filename)
    mesh = meshio.Mesh(
        Mesh.points,
        Mesh.cells,
        cell_data=cell_data)
    meshio.write(vtk_filename, mesh)

    return 0


if __name__ == '__main__':
    tags = dict()
    tags['BND_X-'] = 991
    tags['BND_X+'] = 992
    tags['BND_Y-'] = 993
    tags['BND_Y+'] = 994
    tags['BND_Z-'] = 995
    tags['BND_Z-F'] = 997
    tags['BND_Z+'] = 996

    tags['MATRIX_1'] = 99991
    tags['MATRIX_2'] = 99992

    tags['FRAC'] = 9991

    fault_angle_list = []
    #fault_angle_list += [None]  # no fault
    fault_angle_list += [35.]

    z_minus_hybrid_list = []
    #z_minus_hybrid_list += [False]
    z_minus_hybrid_list += [True]

    two_rocks_list = []
    #two_rocks_list += [False]
    two_rocks_list += [True]

    test_all = False
    #test_all = True
    if test_all:
        for fault_angle in fault_angle_list:
            for z_minus_hybrid in z_minus_hybrid_list:
                for two_rocks in two_rocks_list:
                    # hexahedron cells
                    filename = generate_box_3d(X=0.03, Y=0.03, Z=0.06, NX=21, NY=31, NZ=21,
                                               filename=os.path.join('meshes', 'box_uniform_rect.msh'), tags=tags,
                                               is_transfinite=True, is_recombine=True, popup=True,
                                               fault_angle=fault_angle, z_minus_hybrid=z_minus_hybrid, two_rocks=two_rocks)

                    # tetrahedron cells
                    filename = generate_box_3d(X=0.03, Y=0.03, Z=0.06, NX=11, NY=11, NZ=11,
                                               filename=os.path.join('meshes', 'box_uniform_tetra.msh'), tags=tags,
                                               is_transfinite=False, is_recombine=False, popup=True,
                                               fault_angle=fault_angle, z_minus_hybrid=z_minus_hybrid, two_rocks=two_rocks)

    # for field scale model
    tags_no_fault = tags.copy()
    tags_no_fault.pop('FRAC')
    if False:
        filename = generate_box_3d(X=2000, Y=2000, Z=4000, NX=21, NY=21, NZ=21, tags=tags_no_fault, is_transfinite=True, is_recombine=True, popup=True)
        write_to_vtk_with_faces(filename)
    
    if True:  # rsv_top < rsv < rsv_bottom (tag MATRIX_1) and non-rsv (tag MATRIX_2)
        rsv_top = 2100.
        rsv_bottom = 2200.
        rsv_xy = 1000.
        
        # small
        #x_list = np.array([-2000, -1000, 0, 1000, 2000])
        #z_list = -1 * np.array([0,1000,1500,rsv_top,rsv_bottom,3000,4000])
        
        #case = '34_34_57'  # z 0 - 5 km 
        x_list = np.array([-15000,-8000,-4000,-2400,-1600,-1200,-1100,-1000] + np.arange(-900, 1000, 100).tolist() + [1000, 1100,1200, 1600, 2400, 4000,8000,15000])
        z_list = -np.hstack([np.arange(0, rsv_top - 100 + 1, 100),
                                 np.arange(rsv_top - 50, rsv_bottom + 50 + 1, 25),
                                 rsv_bottom + 100,
                                 np.arange(rsv_bottom + 200, 5000 + 1, 100)])
        filename = generate_box_3d(X=2000, Y=2000, Z=4000, NX=21, NY=21, NZ=21, tags=tags_no_fault, 
                                   Xc=x_list, Yc=x_list, Zc=z_list, rsv_top=-rsv_top, rsv_bottom=-rsv_bottom, 
                                   rsv_x1=-rsv_xy, rsv_x2=rsv_xy, rsv_y1=-rsv_xy, rsv_y2=rsv_xy,
                                   msh_ver=4.2, # geos fails with a negative volume issue for gmsh 2.1 format https://github.com/GEOS-DEV/GEOS/issues/2154
                                   two_rocks=True, is_transfinite=True, is_recombine=True, popup=True)
        
    print('Finished')
    
