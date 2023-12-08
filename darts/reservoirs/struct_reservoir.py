import os
from math import pi
from typing import Union

import numpy as np
from darts.reservoirs.reservoir_base import ReservoirBase
from darts.engines import conn_mesh, ms_well, ms_well_vector, timer_node, value_vector, index_vector
from darts.reservoirs.mesh.struct_discretizer import StructDiscretizer
from pyevtk import hl, vtk
from scipy.interpolate import griddata


class StructReservoir(ReservoirBase):
    def __init__(self, timer: timer_node, nx: int, ny: int, nz: int, dx, dy, dz, permx, permy, permz, poro, depth,
                 rcond=0, hcap=0, actnum=1, global_to_local=0, op_num=0, coord=0, zcorn=0, is_cpg=False, cache=False):
        """
        Class constructor method

        :param timer: timer object to measure discretization time
        :param nx: number of reservoir blocks in the x-direction
        :param ny: number of reservoir blocks in the y-direction
        :param nz: number of reservoir blocks in the z-direction
        :param dx: size of the reservoir blocks in the x-direction (scalar or vector form) [m]
        :param dy: size of the reservoir blocks in the y-direction (scalar or vector form) [m]
        :param dz: size of the reservoir blocks in the z-direction (scalar or vector form) [m]
        :param permx: permeability of the reservoir blocks in the x-direction (scalar or vector form) [mD]
        :param permy: permeability of the reservoir blocks in the y-direction (scalar or vector form) [mD]
        :param permz: permeability of the reservoir blocks in the z-direction (scalar or vector form) [mD]
        :param poro: porosity of the reservoir blocks
        :param actnum: attribute of activity of the reservoir blocks (all are active by default)
        :param global_to_local: one can define arbitrary indexing (mapping from global to local) for local
          arrays. Default indexing is by X (fastest),then Y, and finally Z (slowest)
        :param op_num: index of required operator set of the reservoir blocks (the first by default).
          Use to introduce PVTNUM, SCALNUM, etc.
        :param coord: COORD keyword values for more accurate geometry during VTK export (no values by default)
        :param zcron: ZCORN keyword values for more accurate geometry during VTK export (no values by default)

        """
        super().__init__(timer, cache)

        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.n = nx * ny * nz

        self.permx = permx
        self.permy = permy
        self.permz = permz
        self.global_data = {'dx': dx, 'dy': dy, 'dz': dz,
                            'poro': poro, 'permx': permx, 'permy': permy, 'permz': permz, 'rcond': rcond, 'hcap': hcap,
                            'depth': depth, 'actnum': actnum, 'op_num': op_num,
                            }

        self.actnum = actnum
        self.coord = coord
        self.zcorn = zcorn
        self.is_cpg = is_cpg
        self.global_to_local = global_to_local

        self.vtk_z = 0
        self.vtk_y = 0
        self.vtk_x = 0
        self.vtk_filenames_and_times = {}
        self.vtkobj = 0

        if np.isscalar(self.coord):
            # Usual structured grid generated from DX, DY, DZ, DEPTH
            self.vtk_grid_type = 0
        else:
            # CPG grid from COORD ZCORN
            self.vtk_grid_type = 1

        self.boundary_volumes = {'xy_minus': None, 'xy_plus': None,
                                 'yz_minus': None, 'yz_plus': None,
                                 'xz_minus': None, 'xz_plus': None}
        self.connected_well_segments = {}
        self.wells = []

    def discretize(self, verbose: bool = False):
        self.discretizer = StructDiscretizer(nx=self.nx, ny=self.ny, nz=self.nz, global_data=self.global_data,
                                             global_to_local=self.global_to_local, coord=self.coord, zcorn=self.zcorn,
                                             is_cpg=self.is_cpg)

        self.timer.node['connection list generation'] = timer_node()
        self.timer.node['connection list generation'].start()
        if self.discretizer.is_cpg:
            cell_m, cell_p, tran, tran_thermal = self.discretizer.calc_cpg_discr()
        else:
            cell_m, cell_p, tran, tran_thermal = self.discretizer.calc_structured_discr()
        self.timer.node['connection list generation'].stop()

        volume = self.discretizer.calc_volumes()
        self.global_data['volume'] = volume

        # apply actnum filter if needed - all arrays providing a value for a single grid block should be passed
        arrs = [self.global_data['poro'], self.global_data['rcond'], self.global_data['hcap'],
                self.global_data['depth'], volume, self.global_data['op_num']]
        cell_m, cell_p, tran, tran_thermal, arrs_local = self.discretizer.apply_actnum_filter(self.actnum, cell_m, cell_p,
                                                                                              tran, tran_thermal, arrs)
        poro, rcond, hcap, depth, volume, op_num = arrs_local
        self.global_data['global_to_local'] = self.discretizer.global_to_local

        # Assign layer properties
        self.set_layer_properties()

        # Initialize mesh using built connection list
        self.mesh = conn_mesh()
        self.mesh.init(index_vector(cell_m), index_vector(cell_p), value_vector(tran), value_vector(tran_thermal))

        # Create numpy arrays wrapped around mesh data (no copying)
        np.array(self.mesh.poro, copy=False)[:] = poro
        np.array(self.mesh.rock_cond, copy=False)[:] = rcond
        np.array(self.mesh.heat_capacity, copy=False)[:] = hcap
        np.array(self.mesh.depth, copy=False)[:] = depth
        self.volume = np.array(self.mesh.volume, copy=False)
        self.volume[:] = volume
        np.array(self.mesh.op_num, copy=False)[:] = op_num

        self.set_boundary_volume(self.boundary_volumes)

        return

    def set_boundary_volume(self, boundary_volumes: dict):
        # apply changes
        volume = self.discretizer.volume
        if boundary_volumes['xy_minus'] is not None:
            volume[:, :, 0] = boundary_volumes['xy_minus']
        if boundary_volumes['xy_plus'] is not None:
            volume[:, :, -1] = boundary_volumes['xy_plus']
        if boundary_volumes['yz_minus'] is not None:
            volume[0, :, :] = boundary_volumes['yz_minus']
        if boundary_volumes['yz_plus'] is not None:
            volume[-1, :, :] = boundary_volumes['yz_plus']
        if boundary_volumes['xz_minus'] is not None:
            volume[:, 0, :] = boundary_volumes['xz_minus']
        if boundary_volumes['xz_plus'] is not None:
            volume[:, -1, :] = boundary_volumes['xz_plus']
        # reshape to 1d
        volume = np.reshape(volume, self.discretizer.nodes_tot, order='F')
        # apply actnum and assign to mesh.volume
        self.volume[:] = volume[self.discretizer.local_to_global]

    def add_perforation(self, well_name: str, cell_index: Union[int, tuple], well_radius: float = 0.1524,
                        well_index: float = None, well_indexD: float = None, segment_direction: str = 'z_axis',
                        skin: float = 0, multi_segment: bool = False, verbose: bool = False):
        """
        Function to add perforations to wells.
        """
        well = self.get_well(well_name)

        # calculate well index and get local index of reservoir block
        i, j, k = cell_index
        res_block_local, wi, wid = self.discretizer.calc_well_index(i, j, k, well_radius=well_radius,
                                                                    segment_direction=segment_direction, skin=skin)

        if well_index is None:
            well_index = wi

        if well_indexD is None:
            well_indexD = wid

        assert well_index >= 0
        assert well_indexD >= 0

        # set well segment index (well block) equal to index of perforation layer
        if multi_segment:
            well_block = len(well.perforations)
        else:
            well_block = 0

        # add completion only if target block is active
        if res_block_local > -1:
            if len(well.perforations) == 0:
                well.well_head_depth = np.array(self.mesh.depth, copy=False)[res_block_local]
                well.well_body_depth = well.well_head_depth
                if self.discretizer.is_cpg:
                    dx, dy, dz = self.discretizer.calc_cell_dimensions(i - 1, j - 1, k - 1)
                    # TODO: need segment_depth_increment and segment_length logic
                    if segment_direction == 'z_axis':
                        well.segment_depth_increment = dz
                    elif segment_direction == 'x_axis':
                        well.segment_depth_increment = dx
                    else:
                        well.segment_depth_increment = dy
                else:
                    well.segment_depth_increment = self.discretizer.len_cell_zdir[i - 1, j - 1, k - 1]

                well.segment_volume *= well.segment_depth_increment
            for p in well.perforations:
                if p[0] == well_block and p[1] == res_block_local:
                    print('Neglected duplicate perforation for well %s to block [%d, %d, %d]' % (well.name, i, j, k))
                    return
            well.perforations = well.perforations + [(well_block, res_block_local, well_index, well_indexD)]
            if verbose:
                print('Added perforation for well %s to block %d [%d, %d, %d] with WI=%f and WID=%f' % (
                    well.name, res_block_local, i, j, k, well_index, well_indexD))
        else:
            if verbose:
                print('Neglected perforation for well %s to block [%d, %d, %d] (inactive block)' % (well.name, i, j, k))
        return

    def find_cell_index(self, coord: Union[list, np.ndarray]) -> int:
        """
        Function to find nearest cell to specified coordinate

        :param coord: XYZ-coordinates
        :type coord: list or np.ndarray
        :returns: Index of cell
        :rtype: int
        """
        min_dis = None
        idx = None
        for j, centroid in enumerate(self.discretizer.centroids_all_cells):
            dis = np.linalg.norm(np.array(coord) - centroid)
            if (min_dis is not None and dis < min_dis) or min_dis is None:
                min_dis = dis
                idx = j
        return idx

    def init_wells(self, verbose: bool = False) -> ms_well_vector:
        for w in self.wells:
            assert (len(w.perforations) > 0), "Well %s does not perforate any active reservoir blocks" % w.name
        self.mesh.add_wells(ms_well_vector(self.wells))

        # connect perforations of wells (for example, for closed loop geothermal)
        # dictionary: key is a pair of 2 well names; value is a list of well perforation indices to connect
        # example {(well_1.name, well_2.name): [(w1_perf_1, w2_perf_1),(w1_perf_2, w2_perf_2)]}
        for well_pair in self.connected_well_segments.keys():
            well_1 = self.get_well(well_pair[0])
            well_2 = self.get_well(well_pair[1])
            for perf_pair in self.connected_well_segments[well_pair]:
                self.mesh.connect_segments(well_1, well_2, perf_pair[0], perf_pair[1], 1)

        self.mesh.reverse_and_sort()
        self.mesh.init_grav_coef()

        return self.wells

    def get_cell_cpg_widths(self):
        assert self.discretizer.is_cpg

        dx = np.zeros(self.nx * self.ny * self.nz)
        dy = np.zeros(self.nx * self.ny * self.nz)
        dz = np.zeros(self.nx * self.ny * self.nz)
        for k in range(self.nz):
            for j in range(self.ny):
                for i in range(self.nx):
                    id = i + self.nx * (j + k * self.ny)
                    dx[id], dy[id], dz[id] = self.discretizer.calc_cell_dimensions(i, j, k)
        dx *= self.global_data['actnum']
        dy *= self.global_data['actnum']
        dz *= self.global_data['actnum']
        return dx, dy, dz

    def get_cell_cpg_widths_new(self):
        assert (self.discretizer.is_cpg == True)
        dx = self.discretizer.convert_to_flat_array(np.fabs(self.discretizer.cell_data['faces'][:, :, :, 1, 1] - self.discretizer.cell_data['faces'][:, :, :, 0, 1])[:,:,:,0], 'dx')
        dy = self.discretizer.convert_to_flat_array(np.fabs(self.discretizer.cell_data['faces'][:, :, :, 3, 1] - self.discretizer.cell_data['faces'][:, :, :, 2, 1])[:,:,:,1], 'dy')
        dz = self.discretizer.convert_to_flat_array(np.fabs(self.discretizer.cell_data['faces'][:, :, :, 5, 1] - self.discretizer.cell_data['faces'][:, :, :, 4, 1])[:,:,:,2], 'dz')
        dx *= self.global_data['actnum']
        dy *= self.global_data['actnum']
        dz *= self.global_data['actnum']
        return dx, dy, dz

    def output_to_vtk(self, file_name, t, local_cell_data, global_cell_data, export_constant_data=True):

        nb = self.discretizer.nodes_tot
        cell_data = global_cell_data.copy()

        # only for the first export call
        if len(self.vtk_filenames_and_times) == 0:
            if self.vtk_grid_type == 0:
                if (self.n == self.nx) or (self.n == self.ny) or (self.n == self.nz) or (self.ny == 1):
                    self.generate_vtk_grid(
                        compute_depth_by_dz_sum=False)  # Add this (if condition) for special 1D or 2D crossection
                else:
                    self.generate_vtk_grid()
            else:
                self.generate_cpg_vtk_grid()
            self.vtk_path = './vtk_data/'
            if len(self.vtk_filenames_and_times) == 0:
                os.makedirs(self.vtk_path, exist_ok=True)

            if export_constant_data:
                mesh_geom_dtype = np.float32
                for key, data in self.global_data.items():
                    if np.isscalar(data):
                        if type(data) == int:
                            data = data * np.ones(nb, dtype=int)
                        else:
                            data = data * np.ones(nb, dtype=mesh_geom_dtype)
                    cell_data[key] = data

        vtk_file_name = self.vtk_path + file_name + '_ts%d' % len(self.vtk_filenames_and_times)

        for key, value in local_cell_data.items():
            global_array = np.ones(nb, dtype=value.dtype) * np.nan
            global_array[self.discretizer.local_to_global] = value
            cell_data[key] = global_array

        if self.vtk_grid_type == 0:
            vtk_file_name = hl.gridToVTK(vtk_file_name, self.vtk_x, self.vtk_y, self.vtk_z, cellData=cell_data)
        else:
            for key, value in cell_data.items():
                self.vtkobj.AppendScalarData(key, cell_data[key][self.global_data['actnum'] == 1])

            vtk_file_name = self.vtkobj.Write2VTU(vtk_file_name)
            if len(self.vtk_filenames_and_times) == 0:
                for key, data in self.global_data.items():
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

        nx = self.discretizer.nx
        ny = self.discretizer.ny
        nz = self.discretizer.nz

        # consider 16-bit float is enough for mesh geometry
        mesh_geom_dtype = np.float32

        # get tops from depths
        if np.isscalar(self.global_data['depth']):
            tops = self.global_data['depth'] * np.ones((nx, ny))
            compute_depth_by_dz_sum = True
        elif compute_depth_by_dz_sum:
            tops = self.global_data['depth'][:nx * ny]
            tops = np.reshape(tops, (nx, ny), order='F').astype(mesh_geom_dtype)
        else:
            depths = np.reshape(self.global_data['depth'], (nx, ny, nz), order='F').astype(mesh_geom_dtype)

        # tops_avg = np.mean(tops[tops > 0])
        # tops[tops <= 0] = 2000

        # average x-s of the left planes for the left cross-section (i=1)
        lefts = 0 * np.ones((ny, nz))
        # average y-s of the front planes for the front cross_section (j=1)
        fronts = 0 * np.ones((nx, nz))

        self.vtk_x = np.zeros((nx + 1, ny + 1, nz + 1), dtype=mesh_geom_dtype)
        self.vtk_y = np.zeros((nx + 1, ny + 1, nz + 1), dtype=mesh_geom_dtype)
        self.vtk_z = np.zeros((nx + 1, ny + 1, nz + 1), dtype=mesh_geom_dtype)

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
        self.vtkobj.GRDECL_Data.COORD = self.coord
        self.vtkobj.GRDECL_Data.ZCORN = self.zcorn
        self.vtkobj.GRDECL_Data.NX = self.nx
        self.vtkobj.GRDECL_Data.NY = self.ny
        self.vtkobj.GRDECL_Data.NZ = self.nz
        self.vtkobj.GRDECL_Data.N = self.n
        self.vtkobj.GRDECL_Data.GRID_type = 'CornerPoint'
        self.vtkobj.GRDECL2VTK(self.global_data['actnum'])
        # self.vtkobj.decomposeModel()
