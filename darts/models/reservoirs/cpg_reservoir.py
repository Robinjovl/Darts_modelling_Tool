from darts.engines import conn_mesh, ms_well, ms_well_vector
from darts.discretizer import load_single_float_keyword, load_single_int_keyword
from darts.discretizer import value_vector as value_vector_discr
from darts.discretizer import index_vector as index_vector_discr
import numpy as np

from opmcpg._cpggrid import UnstructuredGrid, process_cpg_grid
from opmcpg._cpggrid import value_vector as value_vector_cpggrid
from opmcpg._cpggrid import index_vector as index_vector_cpggrid

from darts.discretizer import Mesh, Elem, Discretizer, BoundaryCondition, elem_loc, elem_type
from darts.discretizer import index_vector, value_vector, matrix33, vector_matrix33, vector_vector3
from darts.discretizer import load_single_float_keyword

import datetime
import darts
from pyevtk import hl, vtk
from darts.mesh.struct_discretizer import StructDiscretizer

import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir2 = os.path.dirname(parentdir)
sys.path.insert(0, os.path.join(parentdir2, 'python'))

# Definitions for the unstructured reservoir class:
class CPG_Reservoir:
    def __init__(self, gridfile, propfile, faultfile=None):
        """
        Class constructor for UnstructReservoir class
        :param gridfile: file that contains CPG grid
        :param propfile: file that contains properties defined on grid
        """
        # Create mesh object (C++ object used by DARTS for all mesh related quantities):
        self.mesh = conn_mesh()

        self.discretize_cpg(gridfile, propfile)
        #self.discr.write_mpfa_results('conn.dat')

        mpfa_tran = np.array(self.discr.flux_vals, copy=False)
        mpfa_tranD = np.array(self.discr.flux_vals_thermal, copy=False)
        ids = np.array(self.discr.get_one_way_tpfa_transmissibilities())
        cell_m = np.array(self.discr.cell_m)[ids]
        cell_p = np.array(self.discr.cell_p)[ids]
        tran = mpfa_tran[::2][ids]
        tranD = mpfa_tranD[1::2][ids]

        #self.discr.write_tran_cube('tran_cpg.grdecl', 'nnc_cpg.txt')
        if faultfile is not None:
            self.apply_fault_mult(faultfile, cell_m, cell_p, mpfa_tran, ids)
            #self.discr.write_tran_cube('tran_faultmult.grdecl', 'nnc_faultmult.txt')

        tran = np.fabs(tran)
        self.mesh.init(darts.engines.index_vector(cell_m), darts.engines.index_vector(cell_p),
                       darts.engines.value_vector(tran), darts.engines.value_vector(tranD))

        # debug
        #d = {'cell_m': cell_m, 'cell_p': cell_p, 'tran': tran, 'tranD': tranD}
        #import pandas as pd
        #df_cpg = pd.DataFrame(data=d)
        #df_struct = pd.read_excel('tran_struct.xlsx')
        #with pd.ExcelWriter('tran.xlsx') as writer:
        #    df_struct.to_excel(writer, sheet_name='struct')
        #    df_cpg.to_excel(writer, sheet_name='cpg')

        # Create numpy arrays wrapped around mesh data (no copying, this will severely slow down the process!)
        self.mesh.depth = darts.engines.value_vector(self.discr_mesh.depths)
        self.mesh.volume = darts.engines.value_vector(self.discr_mesh.volumes)
        self.bc = np.array(self.mesh.bc, copy=False)

        # rock thermal properties
        self.hcap = np.array(self.mesh.heat_capacity, copy=False)
        self.conduction = np.array(self.mesh.rock_cond, copy=False)

        # create list of wells
        self.wells = []

        self.snap_counter = 0

        self.vtk_z = 0
        self.vtk_y = 0
        self.vtk_x = 0
        self.vtk_filenames_and_times = {}
        self.vtkobj = 0
        self.vtk_grid_type = 1

    def read_arrays(self, gridfile: str, propfile: str):
        self.dims_cpp = index_vector_discr()
        load_single_int_keyword(self.dims_cpp, gridfile, "SPECGRID", 3)
        self.dims = np.array(self.dims_cpp, copy=False)

        self.permx_cpp, self.permy_cpp, self.permz_cpp = value_vector_discr(), value_vector_discr(), value_vector_discr()
        load_single_float_keyword(self.permx_cpp, propfile, 'PERMX', -1)
        load_single_float_keyword(self.permy_cpp, propfile, 'PERMY', -1)
        self.permx = np.array(self.permx_cpp, copy=False)
        self.permy = np.array(self.permy_cpp, copy=False)
        for perm_str in ['PERMEABILITYXY', 'PERMEABILITY']:
            if self.permx.size == 0 or self.permy.size == 0:
                load_single_float_keyword(self.permx_cpp, propfile, perm_str, -1)
                self.permy_cpp = self.permx_cpp
                self.permx = np.array(self.permx_cpp, copy=False)
                self.permy = np.array(self.permy_cpp, copy=False)
        load_single_float_keyword(self.permz_cpp, propfile, 'PERMZ', -1)
        self.permz = np.array(self.permz_cpp, copy=False)

        self.poro_cpp = self.discr_mesh.poro
        load_single_float_keyword(self.poro_cpp, propfile, 'PORO', -1)
        self.poro = np.array(self.poro_cpp, copy=False)

        self.coord_cpp = self.discr_mesh.coord
        load_single_float_keyword(self.coord_cpp, gridfile, 'COORD', -1)
        self.coord = np.array(self.coord_cpp, copy=False)

        self.zcorn_cpp = self.discr_mesh.zcorn
        load_single_float_keyword(self.zcorn_cpp, gridfile, 'ZCORN', -1)
        self.zcorn = np.array(self.zcorn_cpp, copy=False)

        self.actnum_cpp = self.discr_mesh.actnum
        self.actnum = np.array([])
        for fname in [gridfile, propfile]:
            if self.actnum.size == 0:
                load_single_int_keyword(self.actnum_cpp, fname, 'ACTNUM', -1)
                self.actnum = np.array(self.actnum_cpp, copy=False)
        if self.actnum.size == 0:
            self.actnum = np.ones(self.dims[0] * self.dims[1] * self.dims[2])
            print('No ACTNUM found in input files. ACTNUM=1 will be used')


    def discretize_cpg(self, gridfile: str, propfile: str):
        '''
        reads grid and reservoir properties, initialize mesh, creates discretizer object and computes
        transmissibilities using two point flux approximation
        :param gridfile: text file with DIMENS, COORD, ZCORN data (grdecl)
        :param propfile: text file with PORO, PERM data (grdecl)
        :return: None
        '''

        # empty dict just to pass to func
        displaced_tags = dict()
        displaced_tags[elem_loc.MATRIX] = set()
        displaced_tags[elem_loc.FRACTURE] = set()
        displaced_tags[elem_loc.BOUNDARY] = set()
        displaced_tags[elem_loc.FRACTURE_BOUNDARY] = set()

        self.discr_mesh = Mesh()
        result_fname = 'results.grdecl'
        minpv = 0

        self.read_arrays(gridfile, propfile)

        dims_cpp = index_vector_cpggrid(self.dims)
        coord_cpp = value_vector_cpggrid(self.coord)
        zcorn_cpp = value_vector_cpggrid(self.zcorn)
        actnum_cpp = index_vector_cpggrid(self.actnum)
        ugrid = process_cpg_grid(dims_cpp, coord_cpp, zcorn_cpp, actnum_cpp, minpv, result_fname)

        self.nx = self.discr_mesh.nx = self.dims[0]
        self.ny = self.discr_mesh.ny = self.dims[1]
        self.nz = self.discr_mesh.nz = self.dims[2]
        self.nb = self.mesh.n_res_blocks
        self.discr_mesh.n_cells = ugrid.number_of_cells
        # cells + boundary_faces, approximate
        self.discr_mesh.num_of_elements = self.discr_mesh.n_cells + \
                                          2 * (self.nx * self.ny +
                                               self.ny * self.nz +
                                               self.nx * self.nz)

        number_of_nodes = ugrid.number_of_nodes
        number_of_cells = ugrid.number_of_cells
        number_of_faces = ugrid.number_of_faces
        node_coordinates = value_vector(np.array(ugrid.node_coordinates, copy=False))
        face_nodes = index_vector(np.array(ugrid.face_nodes, copy=False))
        face_nodepos = index_vector(np.array(ugrid.face_nodepos, copy=False))
        face_cells = index_vector(np.array(ugrid.face_cells, copy=False))
        cell_faces = index_vector(np.array(ugrid.cell_faces, copy=False))
        cell_facetag = index_vector(np.array(ugrid.cell_facetag, copy=False))
        global_cell = index_vector(np.array(ugrid.global_cell, copy=False))
        cell_facepos = index_vector(np.array(ugrid.cell_facepos, copy=False))
        cell_volumes = value_vector(np.array(ugrid.cell_volumes, copy=False))
        cell_centroids = value_vector(np.array(ugrid.cell_centroids, copy=False))
        face_normals = value_vector(np.array(ugrid.face_normals, copy=False))
        face_areas = value_vector(np.array(ugrid.face_areas, copy=False))
        face_centroids = value_vector(np.array(ugrid.face_centroids, copy=False))

        face_order = index_vector()

        res = self.discr_mesh.cpg_elems_nodes(
            number_of_nodes, number_of_cells, number_of_faces,
            node_coordinates, face_nodes, face_nodepos,
            face_cells, cell_faces, cell_facepos,
            cell_volumes, face_order)

        bnd_faces_num = res[0]
        #self.discr_mesh.print_elems_nodes()

        self.discr_mesh.construct_local_global()

        self.discr_mesh.cpg_cell_props(number_of_nodes, number_of_cells, number_of_faces,
                                           cell_volumes, cell_centroids, global_cell,
                                           face_areas, face_centroids,
                                           bnd_faces_num, face_order)

        self.discr_mesh.cpg_connections(number_of_cells, number_of_faces,
                                                       node_coordinates, face_nodes, face_nodepos,
                                                       face_cells, cell_faces, cell_facepos,
                                                       face_centroids, face_areas, face_normals,
                                                       cell_facetag, displaced_tags)

        self.discr_mesh.generate_adjacency_matrix()

        self.discr = Discretizer()
        self.cpp_bc = self.set_boundary_conditions(displaced_tags)
        self.discr.set_mesh(self.discr_mesh)
        self.discr.init()

        self.volume_all_cells = np.array(self.discr_mesh.volumes, copy=False)
        self.depth_all_cells = np.array(self.discr_mesh.depths, copy=False)
        self.actnum = np.array(self.discr_mesh.actnum, copy=False)
        # self.centroids = np.array(self.discr_mesh.centroids, copy=False)

        self.discr.set_permeability(self.permx_cpp, self.permy_cpp, self.permz_cpp)

        n_all = self.nx * self.ny * self.nz
        print("Number of all cells    = ", n_all)
        print("Number of active cells = ", self.discr_mesh.n_cells)

        #poro could be modified here
        #self.poro[poro < 1e-2] = 1e-2
        self.discr.set_porosity(self.discr_mesh.poro)
        self.mesh.poro = darts.engines.value_vector(self.discr.poro)
        self.poro = np.array(self.discr_mesh.poro, copy=False)

        # calculate transmissibilities
        self.discr.calc_tpfa_transmissibilities(displaced_tags)


    def calc_well_index(self, i, j, k, well_radius=0.1524, segment_direction='z_axis', skin=0):
        """
        Class method which construct the well index for each well segment/perforation
        :param i: "human" counting of x-location coordinate of perforation
        :param j: "human" counting of y-location coordinate of perforation
        :param k: "human" counting of z-location coordinate of perforation
        :param well_radius: radius of the well-bore
        :param segment_direction: direction in which the segment perforates the reservoir block
        :param skin: skin factor for pressure loss around well-bore due to formation damage
        :return well_index: well-index of particular perforation
        """
        assert (i > 0), "Perforation block coordinate should be positive"
        assert (j > 0), "Perforation block coordinate should be positive"
        assert (k > 0), "Perforation block coordinate should be positive"
        assert (i <= self.nx), "Perforation block coordinate should not exceed corresponding reservoir dimension"
        assert (j <= self.ny), "Perforation block coordinate should not exceed corresponding reservoir dimension"
        assert (k <= self.nz), "Perforation block coordinate should not exceed corresponding reservoir dimension"
        i -= 1
        j -= 1
        k -= 1

        # compute reservoir block index
        res_block = self.discr_mesh.get_global_index(i, j, k)

        # local index
        local_block = self.discr_mesh.global_to_local[res_block]

        # check if target grid block is active
        if local_block > -1:
            dx, dy, dz = self.discr_mesh.calc_cell_sizes(i, j, k)

            eps = 1e-6  # to avoid divizion by zero
            kx = self.permx[res_block] + eps
            ky = self.permy[res_block] + eps
            kz = self.permz[res_block] + eps

            well_index = 0
            if segment_direction == 'z_axis':
                peaceman_rad = 0.28 * np.sqrt(np.sqrt(ky / kx) * dx ** 2 + np.sqrt(kx / ky) * dy ** 2) / \
                               ((ky / kx) ** (1 / 4) + (kx / ky) ** (1 / 4))
                well_index = 2 * np.pi * dz * np.sqrt(kx * ky) / (np.log(peaceman_rad / well_radius) + skin)
                conduction_rad = 0.28 * np.sqrt(dx ** 2 + dy ** 2) / 2.
                well_indexD = 2 * np.pi * dz / (np.log(conduction_rad / well_radius) + skin)
                if kx == 0 or ky == 0: well_index = 0.0
            elif segment_direction == 'x_axis':
                peaceman_rad = 0.28 * np.sqrt(np.sqrt(ky / kz) * dz ** 2 + np.sqrt(kz / ky) * dy ** 2) / \
                               ((ky / kz) ** (1 / 4) + (kz / ky) ** (1 / 4))
                well_index = 2 * np.pi * dz * np.sqrt(kz * ky) / (np.log(peaceman_rad / well_radius) + skin)
                conduction_rad = 0.28 * np.sqrt(dz ** 2 + dy ** 2) / 2.
                well_indexD = 2 * np.pi * dx / (np.log(conduction_rad / well_radius) + skin)
                if kz == 0 or ky == 0: well_index = 0.0
            elif segment_direction == 'y_axis':
                peaceman_rad = 0.28 * np.sqrt(np.sqrt(kz / kx) * dx ** 2 + np.sqrt(kx / kz) * dz ** 2) / \
                               ((kz / kx) ** (1 / 4) + (kx / kz) ** (1 / 4))
                well_index = 2 * np.pi * dz * np.sqrt(kx * kz) / (np.log(peaceman_rad / well_radius) + skin)
                conduction_rad = 0.28 * np.sqrt(dz ** 2 + dx ** 2) / 2.
                well_indexD = 2 * np.pi * dy / (np.log(conduction_rad / well_radius) + skin)
                if kx == 0 or kz == 0: well_index = 0.0

            well_index = well_index * StructDiscretizer.darcy_constant
        else:
            well_index = 0

        return local_block, well_index, well_indexD

    def set_boundary_volume(self, xy_minus=-1, xy_plus=-1, yz_minus=-1, yz_plus=-1, xz_minus=-1, xz_plus=-1):
        mesh_volume = np.array(self.volume_all_cells, copy=False)

        # get 3d shape
        volume = make_full_cube(mesh_volume[:self.discr_mesh.n_cells], self.actnum)
        volume = volume.reshape(self.nx, self.ny, self.nz, order='F')

        actnum3d = self.actnum.reshape(self.nx, self.ny, self.nz, order='F')

        # apply changes
        if xy_minus > -1:
            for i in range(self.nx):
                for j in range(self.ny):
                    k = 0
                    while k < self.nz and actnum3d[i, j, k] == 0:  # search first active cell
                        k += 1
                    if k < self.nz:
                        volume[i, j, k] = xy_minus
        if xy_plus > -1:
            for i in range(self.nx):
                for j in range(self.ny):
                    k = self.nz - 1
                    while k >= 0 and actnum3d[i, j, k] == 0:
                        k -= 1
                    if k >= 0:
                        volume[i, j, k] = xy_plus
        if yz_minus > -1:
            for k in range(self.nz):
                for j in range(self.ny):
                    i = 0
                    while i < self.nx and actnum3d[i, j, k] == 0:
                        i += 1
                    if i < self.nx:
                        volume[i, j, k] = yz_minus
        if yz_plus > -1:
            for k in range(self.nz):
                for j in range(self.ny):
                    i = self.nx - 1
                    while i >= 0 and actnum3d[i, j, k] == 0:
                        i -= 1
                    if i >= 0:
                        volume[i, j, k] = yz_plus
        if xz_minus > -1:
            for k in range(self.nz):
                for i in range(self.nx):
                    j = 0
                    while j < self.ny and actnum3d[i, j, k] == 0:
                        j += 1
                    if j < self.ny:
                        volume[i, j, k] = xz_minus
        if xz_plus > -1:
            for k in range(self.nz):
                for i in range(self.nx):
                    j = self.ny - 1
                    while j >= 0 and actnum3d[i, j, k] == 0:
                        j -= 1
                    if j >= 0:
                        volume[i, j, k] = xz_plus
        volume_1d = np.reshape(volume, self.discr_mesh.nx*self.discr_mesh.ny*self.discr_mesh.nz, order='F') # back to 1D
        # apply actnum and assign to mesh.volume
        mesh_volume[:self.discr_mesh.n_cells] = volume_1d[self.discr_mesh.local_to_global]

    def set_boundary_conditions(self, physical_tags):
        return
        bc = BoundaryCondition()

        boundary_range = self.discr_mesh.region_ranges[elem_loc.BOUNDARY]
        a = np.zeros(boundary_range[1] - boundary_range[0])
        b = np.zeros(boundary_range[1] - boundary_range[0])
        r = np.zeros(boundary_range[1] - boundary_range[0])

        # no-flow (impermeable) bc
        a[:] = 0.
        b[:] = 1.
        r[:] = 0.

        bc.a = value_vector(a)
        bc.b = value_vector(b)
        bc.r = value_vector(r)

        return bc

    def add_well(self, name, depth=0):
        """
        Class method which adds wells heads to the reservoir (Note: well head is not equal to a perforation!)
        :param name:
        :param depth:
        :return:
        """
        well = ms_well()
        well.name = name
        well.segment_volume = 0.0785 * 40  # 2.5 * pi * 0.15**2 / 4
        well.well_head_depth = depth
        well.well_body_depth = depth
        well.segment_transmissibility = 1e5
        well.segment_depth_increment = 1
        self.wells.append(well)
        return 0

    # ijk indices are is 1-based (starts from 1)
    def add_perforation(self, well, i, j, k, well_radius=0.1524, well_index=-1, well_indexD=-1, segment_direction='z_axis', skin=0,
                        multi_segment=True,
                        verbose=False):
        # calculate well index and get local index of reservoir block
        res_block_local, wi, wiD = self.calc_well_index(i=i, j=j, k=k, well_radius=well_radius,
                                                               segment_direction=segment_direction,
                                                               skin=skin)

        if well_index == -1:
            well_index = wi

        if well_indexD == -1:
            well_indexD = wiD

        # set well segment index (well block) equal to index of perforation layer
        if multi_segment:
            well_block = len(well.perforations)
        else:
            well_block = 0

        # add completion only if target block is active
        if res_block_local > -1:
            if len(well.perforations) == 0:
                well.well_head_depth = self.depth_all_cells[res_block_local]
                well.well_body_depth = well.well_head_depth
                dx, dy, dz = self.discr_mesh.calc_cell_sizes(i - 1, j - 1, k - 1)
                well.segment_depth_increment = dz
                well.segment_volume *= well.segment_depth_increment
            for p in well.perforations:
                if p[0] == well_block and p[1] == res_block_local:
                    print('Neglected duplicate perforation for well %s to block [%d, %d, %d]' % (well.name, i, j, k))
                    return
            well.perforations = well.perforations + [(well_block, res_block_local, well_index,well_indexD)]
            if verbose:
                print('Added perforation for well %s to block %d [%d, %d, %d] with WI=%f WID=%f' % (
                    well.name, res_block_local, i, j, k, well_index, well_indexD))
        else:
            if verbose:
                print('Neglected perforation for well %s to block [%d, %d, %d] (inactive block)' % (well.name, i, j, k))

    def init_wells(self):
        """
        Class method which initializes the wells (adding wells and their perforations to the reservoir)
        :return:
        """

        # Add wells to the DARTS mesh object and sort connection (DARTS related):
        self.mesh.add_wells(ms_well_vector(self.wells))
        self.mesh.reverse_and_sort()
        self.mesh.init_grav_coef()
        return 0

    def write_mpfa_conn_to_file(self, path = 'mpfa_conn.dat'):
        stencil = np.array(self.discr.flux_stencil, copy=False)
        trans = np.array(self.discr.flux_vals, copy=False)


        f = open(path, 'w')
        f.write(str(len(self.discr.cell_m)) + '\n')

        for conn_id in range(len(self.discr.cell_m)):
            cells = stencil[self.discr.flux_offset[conn_id]:self.discr.flux_offset[conn_id + 1]]
            coefs= trans[self.discr.flux_offset[conn_id]:self.discr.flux_offset[conn_id + 1]]
            #row = str(self.discr.cell_m[conn_id]) + '\t' + str(self.discr.cell_p[conn_id])
            row = str(self.discr.cell_m[conn_id]) + '\t' + str(self.discr.cell_p[conn_id]) + '\t\t'
            #row_cells = ''#str(cells)
            #row_vals = ''#str(coefs)
            for i in range(cells.size):
                if np.abs(coefs[i]) > 1.E-10:
                    row += str(cells[i]) + '\t' + str('{:.2e}'.format(coefs[i])) + '\t'
                    #row_cells += str(cells[i]) + '\t'
                    #row_vals += str('{:.2e}'.format(coefs[i])) + '\t'
            f.write(row + '\n')# + row_cells + '\n' + row_vals + '\n')
        f.close()

    def export_vtk(self, file_name, t, local_cell_data, global_cell_data, export_constant_data=True):

        nb = self.nx * self.ny * self.nz
        cell_data = global_cell_data.copy()

        # only for the first export call
        if len(self.vtk_filenames_and_times) == 0:
            self.generate_cpg_vtk_grid()
            self.vtk_path = './vtk_data/'
            if len(self.vtk_filenames_and_times) == 0:
                os.makedirs(self.vtk_path, exist_ok=True)

            if export_constant_data:
                global_data = {'permx': self.permx, 'permy': self.permy, 'permz': self.permz, 'poro': self.poro}
                #mesh_geom_dtype = np.float32
                for key, data in global_data.items():
                    cell_data[key] = data

        vtk_file_name = self.vtk_path + file_name + '_ts%d' % len(self.vtk_filenames_and_times)

        for key, value in local_cell_data.items():
            global_array = np.ones(nb, dtype=value.dtype) * np.nan
            dummy_zeros = np.zeros(self.discr_mesh.n_cells - self.mesh.n_res_blocks) # workaround for the issue in case of cells without active neighbours
            v = np.append(value[:self.mesh.n_res_blocks], dummy_zeros)
            global_array[self.discr_mesh.local_to_global] = v[:]
            cell_data[key] = global_array

        if self.vtk_grid_type == 0:
            vtk_file_name = hl.gridToVTK(vtk_file_name, self.vtk_x, self.vtk_y, self.vtk_z, cellData=cell_data)
        else:
            for key, value in cell_data.items():
                self.vtkobj.AppendScalarData(key, cell_data[key][self.actnum == 1])

            vtk_file_name = self.vtkobj.Write2VTU(vtk_file_name)
            if len(self.vtk_filenames_and_times) == 0 and export_constant_data:
                for key, data in global_data.items():
                    self.vtkobj.VTK_Grids.GetCellData().RemoveArray(key)
                self.vtkobj.VTK_Grids.GetCellData().RemoveArray('cellNormals')

        # in order to have correct timesteps in Paraview, write down group file
        # since the library in use (pyevtk) requires the group file to call .save() method in the end,
        # and does not support reading, track all written files and times and re-write the complete
        # group file every time

        self.vtk_filenames_and_times[vtk_file_name] = t

        self.group = vtk.VtkGroup(file_name)
        for fname, t in self.vtk_filenames_and_times.items():
            self.group.addFile(fname, t)
        self.group.save()

    def generate_vtk_grid(self, strict_vertical_layers=True, compute_depth_by_dz_sum=True):
        # interpolate 2d array using grid (xx, yy) and specified method
        def interpolate_slice(xx, yy, array, method):
            array = np.ma.masked_invalid(array)
            # get only the valid values
            x1 = xx[~array.mask]
            y1 = yy[~array.mask]
            newarr = array[~array.mask]
            array = griddata((x1, y1), newarr.ravel(),
                             (xx, yy),
                             method=method)
            return array

        def interpolate_zeroes_2d(array):
            array[array == 0] = np.nan
            x = np.arange(0, array.shape[1])
            y = np.arange(0, array.shape[0])
            xx, yy = np.meshgrid(x, y)

            # stage 1 - fill in interior data using cubic interpolation
            array = interpolate_slice(xx, yy, array, 'cubic')
            # stage 2 - fill exterior data using nearest
            array = interpolate_slice(xx, yy, array, 'nearest')
            return array

        def interpolate_zeroes_3d(array_3d):
            if array_3d[array_3d == 0].size > 0:
                array_3d[array_3d == 0] = np.nan
                x = np.arange(0, array_3d.shape[1])
                y = np.arange(0, array_3d.shape[0])
                xx, yy = np.meshgrid(x, y)
                # slice array over third dimension
                for k in range(array_3d.shape[2]):
                    array = array_3d[:, :, k]
                    if array[np.isnan(array) == False].size > 3:
                        # stage 1 - fill in interior data using cubic interpolation
                        array = interpolate_slice(xx, yy, array, 'cubic')

                    if array[np.isnan(array) == False].size > 0:
                        # stage 2 - fill exterior data using nearest
                        array_3d[:, :, k] = interpolate_slice(xx, yy, array, 'nearest')
                    else:
                        if k > 0:
                            array_3d[:, :, k] = np.mean(array_3d[:, :, k - 1])
                        else:
                            array_3d[:, :, k] = np.mean(array_3d)

            return array_3d

        # consider 16-bit float is enough for mesh geometry
        mesh_geom_dtype = np.float32

        # get tops from depths
        if np.isscalar(self.global_data['depth']):
            tops = self.global_data['depth'] * np.ones((self.nx, self.ny))
            compute_depth_by_dz_sum = True
        elif compute_depth_by_dz_sum:
            tops = self.global_data['depth'][:self.nx * self.ny]
            tops = np.reshape(tops, (self.nx, self.ny), order='F').astype(mesh_geom_dtype)
        else:
            depths = np.reshape(self.global_data['depth'], (self.nx, self.ny, self.nz), order='F').astype(mesh_geom_dtype)

        # tops_avg = np.mean(tops[tops > 0])
        # tops[tops <= 0] = 2000

        # average x-s of the left planes for the left cross-section (i=1)
        lefts = 0 * np.ones((self.ny, self.nz))
        # average y-s of the front planes for the front cross_section (j=1)
        fronts = 0 * np.ones((self.nx, self.nz))

        self.vtk_x = np.zeros((self.nx + 1, self.ny + 1, self.nz + 1), dtype=mesh_geom_dtype)
        self.vtk_y = np.zeros((self.nx + 1, self.ny + 1, self.nz + 1), dtype=mesh_geom_dtype)
        self.vtk_z = np.zeros((self.nx + 1, self.ny + 1, self.nz + 1), dtype=mesh_geom_dtype)

        if compute_depth_by_dz_sum:
            tops = interpolate_zeroes_2d(tops)
            tops_padded = np.pad(tops, 1, 'edge')
        else:
            depths_padded = np.pad(depths, 1, 'edge').astype(mesh_geom_dtype)
        lefts_padded = np.pad(lefts, 1, 'edge')
        fronts_padded = np.pad(fronts, 1, 'edge')

        dx_padded = np.pad(self.discretizer.len_cell_xdir, 1, 'edge').astype(mesh_geom_dtype)
        dy_padded = np.pad(self.discretizer.len_cell_ydir, 1, 'edge').astype(mesh_geom_dtype)
        dz_padded = np.pad(self.discretizer.len_cell_zdir, 1, 'edge').astype(mesh_geom_dtype)

        if strict_vertical_layers:
            print("Interpolating missing data in DX...")
            dx_padded_top = interpolate_zeroes_2d(dx_padded[:, :, 0])
            dx_padded = np.repeat(dx_padded_top[:, :, np.newaxis], dx_padded.shape[2], axis=2)

            print("Interpolating missing data in DY...")
            dy_padded_top = interpolate_zeroes_2d(dy_padded[:, :, 0])
            dy_padded = np.repeat(dy_padded_top[:, :, np.newaxis], dy_padded.shape[2], axis=2)
        else:
            print("Interpolating missing data in DX...")
            interpolate_zeroes_3d(dx_padded)
            print("Interpolating missing data in DY...")
            interpolate_zeroes_3d(dy_padded)

        # DZ=0 can actually be correct values in case of zero-thickness inactive blocks
        # So we don`t need to interpolate them

        # print("Interpolating missing data in DZ...")
        # interpolate_zeroes_3d(dz_padded)

        if not compute_depth_by_dz_sum:
            print("Interpolating missing data in DEPTH...")
            interpolate_zeroes_3d(depths_padded)

        # initialize k=0 as sum of 4 neighbours
        if compute_depth_by_dz_sum:
            self.vtk_z[:, :, 0] = (tops_padded[:-1, :-1] +
                                   tops_padded[:-1, 1:] +
                                   tops_padded[1:, :-1] +
                                   tops_padded[1:, 1:]) / 4
        else:
            self.vtk_z[:, :, 0] = (depths_padded[:-1, :-1, 0] - dz_padded[:-1, :-1, 0] / 2 +
                                   depths_padded[:-1, 1:, 0] - dz_padded[:-1, 1:, 0] / 2 +
                                   depths_padded[1:, :-1, 0] - dz_padded[1:, :-1, 0] / 2 +
                                   depths_padded[1:, 1:, 0] - dz_padded[1:, 1:, 0] / 2) / 4
        # initialize i=0
        self.vtk_x[0, :, :] = (lefts_padded[:-1, :-1] +
                               lefts_padded[:-1, 1:] +
                               lefts_padded[1:, :-1] +
                               lefts_padded[1:, 1:]) / 4
        # initialize j=0
        self.vtk_y[:, 0, :] = (fronts_padded[:-1, :-1] +
                               fronts_padded[:-1, 1:] +
                               fronts_padded[1:, :-1] +
                               fronts_padded[1:, 1:]) / 4

        # assign the rest coordinates by averaged size of neigbouring cells
        if compute_depth_by_dz_sum:
            self.vtk_z[:, :, 1:] = (dz_padded[:-1, :-1, 1:-1] +
                                    dz_padded[:-1, 1:, 1:-1] +
                                    dz_padded[1:, :-1, 1:-1] +
                                    dz_padded[1:, 1:, 1:-1]) / 4
        else:
            self.vtk_z[:, :, 1:] = (depths_padded[:-1, :-1, 1:-1] + dz_padded[:-1, :-1, 1:-1] / 2 +
                                    depths_padded[:-1, 1:, 1:-1] + dz_padded[:-1, 1:, 1:-1] / 2 +
                                    depths_padded[1:, :-1, 1:-1] + dz_padded[1:, :-1, 1:-1] / 2 +
                                    depths_padded[1:, 1:, 1:-1] + dz_padded[1:, 1:, 1:-1] / 2) / 4

        self.vtk_x[1:, :, :] = (dx_padded[1:-1, :-1, :-1] +
                                dx_padded[1:-1, :-1, 1:] +
                                dx_padded[1:-1, 1:, :-1] +
                                dx_padded[1:-1, 1:, 1:]) / 4

        self.vtk_y[:, 1:, :] = (dy_padded[:-1, 1:-1, :-1] +
                                dy_padded[:-1, 1:-1, 1:] +
                                dy_padded[1:, 1:-1, :-1] +
                                dy_padded[1:, 1:-1, 1:]) / 4

        self.vtk_x = np.cumsum(self.vtk_x, axis=0)
        self.vtk_y = np.cumsum(self.vtk_y, axis=1)
        if compute_depth_by_dz_sum:
            self.vtk_z = np.cumsum(self.vtk_z, axis=2)

        # convert to negative coordinate
        z_scale = -1
        self.vtk_z *= z_scale

    def generate_cpg_vtk_grid(self):

        from darts.tools import GRDECL2VTK

        self.vtkobj = GRDECL2VTK.GeologyModel()
        self.vtkobj.GRDECL_Data.COORD = self.discr_mesh.coord
        self.vtkobj.GRDECL_Data.ZCORN = self.discr_mesh.zcorn
        self.vtkobj.GRDECL_Data.NX = self.nx
        self.vtkobj.GRDECL_Data.NY = self.ny
        self.vtkobj.GRDECL_Data.NZ = self.nz
        self.vtkobj.GRDECL_Data.N = self.nx * self.ny * self.nz
        self.vtkobj.GRDECL_Data.GRID_type = 'CornerPoint'
        self.vtkobj.GRDECL2VTK(self.discr_mesh.actnum)
        # self.vtkobj.decomposeModel()


    def apply_fault_mult(self, faultfile, cell_m, cell_p, mpfa_tran, ids):
        #Faults

        keep_reading = True
        prev_fault_name = ''

        with open(faultfile) as f:
            while True:
                buff = f.readline()
                strline = buff.split()
                if len(strline) == 0 or '/' == strline[0]:
                    break
                fault_name = strline[0]
                # multiply tran
                i1 = int(strline[1])
                j1 = int(strline[2])
                k1 = int(strline[3])
                i2 = int(strline[4])
                j2 = int(strline[5])
                k2 = int(strline[6])
                fault_tran_mult = float(strline[7])
                if i1 > self.discr_mesh.nx or j1 > self.discr_mesh.ny or k1 > self.discr_mesh.nz:
                    print ('Error:', i1,j1,k1, 'out of grid', buff)
                    continue # skip
                if i2 > self.discr_mesh.nx or j2 > self.discr_mesh.ny or k2 > self.discr_mesh.nz:
                    print ('Error:', i2,j2,k2, 'out of grid', buff)
                    continue # skip

                m_idx = self.discr_mesh.global_to_local[self.discr_mesh.get_global_index(i1-1, j1-1, k1-1)]
                p_idx = self.discr_mesh.global_to_local[self.discr_mesh.get_global_index(i2-1, j2-1, k2-1)]

                p = set(np.where(cell_p == p_idx)[0]) # find cell idx in cell_p
                m = set(np.where(cell_m == m_idx)[0])
                res = m & p # find connection (cell should be in both
                if len(res) > 0:
                    idx = res.pop()
                    mpfa_tran[2*ids[idx]] *= fault_tran_mult

                #print('fault tran mult', fault_tran_mult)

    def apply_volume_depth(self):
        self.depth = np.array(self.mesh.depth, copy=False)
        self.volume = np.array(self.mesh.volume, copy=False)

        #self.depth_all_cells[self.depth_all_cells < 1e-6] = 1e-6
        #self.volume_all_cells[self.volume_all_cells < 1e-6] = 1e-6

        self.depth[:] = self.depth_all_cells
        self.volume[:] = self.volume_all_cells


#####################################################################

def save_array(arr: np.array, fname: str, keyword: str, actnum: np.array, mode='w'):
    '''
    writes numpy array of n_active_cell size to text file in GRDECL format with n_cells_total
    :param arr: numpy array to write
    :param fname: filename
    :param keyword: keyword for array
    :param actnum: actnum array
    :param mode: 'w' to rewrite the file or 'a' to append
    :return: None
    '''
    arr_full = make_full_cube(arr, actnum)
    with open(fname, mode) as f:
        f.write(keyword + '\n')
        s = ''
        for i in range(arr_full.size):
            s += str(arr_full[i]) + ' '
            if (i+1) % 6 == 0:  # write only 6 values per row
                f.write(s + '\n')
                s = ''
        f.write(s + '\n')
        f.write('/\n')
        print('Array saved to file', fname, ' (keyword ' + keyword + ')')


def make_full_cube(cube: np.array, actnum: np.array):
    '''
    returns 1d-array of size nx*ny*nz, filled with zeros where actnum is zero
    :param cube: 1d-array of size n_active_cells
    :param actnum: 1d-array of size nx*ny*nz
    :return:
    '''
    if actnum.size == cube.size:
        return cube
    cube_full = np.zeros(actnum.size)
    cube_full[actnum > 0] = cube
    return cube_full
    
   