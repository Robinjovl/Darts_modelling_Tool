import numpy as np
import pickle
import meshio
import os

from math import inf, pi, asin
from darts.engines import conn_mesh, ms_well, ms_well_vector, index_vector, value_vector, timer_node
from darts.reservoirs.unstruct_reservoir import UnstructReservoir
from darts.reservoirs.mesh.unstruct_discretizer import UnstructDiscretizer
from darts.reservoirs.mesh.geometry.unstructured import Unstructured
# from darts.reservoirs.mesh.geometry.fluidflower import FluidFlower
from fluidflower import FluidFlower
from darts.reservoirs.mesh.geometry.wells import CircularWell, WellCell


class FluidFlowerNewStruct(UnstructReservoir):
    def __init__(self, specs, timer, layer_props, well_geometry, mesh_file, reservoir_cache = True, velocity_cache = False, wells_cache = False):
        self.specs = specs
        self.timer = timer.node['initialization']
        self.reservoir_cache = reservoir_cache
        self.velocity_cache = velocity_cache
        self.wells_cache = wells_cache

        # self.timer.node["set_wells"] = timer_node()
        self.well_geometry = well_geometry
        # self.layer_props = layer_props


        self.physical_tags = {
            "matrix": [],
            "fracture": [],
            "boundary": [],
            "fracture_boundary": [],
            "output": [],
        }
        self.physical_tags['matrix'] = [1, 2, 3, 4, 5, 6, 7]
        self.mesh_file = mesh_file

        self.discretizer = UnstructDiscretizer(mesh_file=self.mesh_file, physical_tags=self.physical_tags, verbose=False)

        self.init_props(layer_props)
        self.centroids = self.c
        super().__init__(timer=timer, mesh_file=self.mesh_file,
                         permx=self.permx, permy=self.permy, permz=self.permz, poro=self.poro,
                         hcap=self.hcap, rcond=self.rcond, cache = False)
        self.op_num = np.array(self.mesh_data.cell_data['gmsh:physical'][0] - 1, dtype=np.float32)

    def init_props(self, layer_props):
        self.mesh_data = meshio.read(self.mesh_file)
        # c = np.array([c.centroid for key, c in self.discretizer.mat_cell_info_dict.items()])
        self.c = np.mean(self.mesh_data.points[self.mesh_data.cells[0].data], axis=1)
        self.op_num = self.mesh_data.cell_data['gmsh:physical'][0]
        n_res_blocks = self.c.shape[0]
        self.n_res_block = n_res_blocks
        self.permx = np.zeros(n_res_blocks)
        self.permy = np.zeros(n_res_blocks)
        self.permz = np.zeros(n_res_blocks)
        self.poro = np.zeros(n_res_blocks)
        self.rcond = np.zeros(n_res_blocks)
        self.hcap = 2125.0

        for tag, pp in layer_props.items():
            id = np.where(self.op_num == tag)[0]
            centroids = self.c[id]
            perm_reference = pp.perm * np.eye(3)
            perm_reference[2,2] *= pp.anisotropy[2]
            # print(tag, pp, centroids.shape)
            reference = self.map_cartesian_to_reference(centroids.T)
            F = self.jacobian_reference_to_cartesian(reference.T)
            perm_cartesian = np.einsum('ijk,jl,lmk->imk', F, perm_reference, F, optimize=True)
            self.permx[id] = perm_cartesian[0, 0]
            self.permy[id] = perm_cartesian[1, 1]
            self.permz[id] = perm_cartesian[2, 2]
            self.poro[id] = pp.poro
            self.rcond[id] = pp.rcond

    def discretize_velocities(self, cell_m, cell_p, geom_coef, n_res_blocks):
        print("Reconstructing velocities.....")
        self.timer.node["velocity reconstruction"].start()

        cache = self.velocity_cache
        if not cache:
            # filter well connections
            inds = np.where(np.logical_and(cell_m < n_res_blocks, cell_p < n_res_blocks))[0]
            cell_m = cell_m[inds]
            cell_p = cell_p[inds]

            # find indices/directions of boundary cells - TODO

            # approximate normals
            dr = self.discretizer.centroid_all_cells[cell_p] - self.discretizer.centroid_all_cells[cell_m]
            n = dr * geom_coef[:, np.newaxis]

            # unique elements & and starting positions of each element
            _, idx_start = np.unique(cell_m, return_index=True)

            # group indices of elements with the same value (cell_m is already sorted)
            res = np.split(np.arange(cell_m.size), idx_start[1:])

            # form matrices for each cell
            all_elements = []
            offsets = [0]
            current_offset = 0
            for i in range(n_res_blocks):
                A = n[res[i]]
                A_T = A.T
                A_TA = A_T @ A

                # Use pseudoinverse if the matrix rank is less than the number of columns
                if np.linalg.matrix_rank(A_TA) < A_TA.shape[0]:
                    least_squares = np.linalg.pinv(A_TA) @ A_T
                else:
                    least_squares = np.linalg.inv(A_TA) @ A_T

                # Flatten the matrix and add it to the all_elements array
                flattened_matrix = least_squares.flatten()
                all_elements.extend(flattened_matrix)

                # Update the current offset and add it to the offsets array
                current_offset += A.shape[0]
                offsets.append(current_offset)

            all_elements_ = np.array(all_elements)
            offsets_ = np.array(offsets, dtype=np.int32)

            # if not os.path.isfile(self.mesh_file + '.veclocity.cache'):
            #     print('Create velocity cache...')
            #     meshObject = {}
            #     meshObject['all_elements_'] = all_elements_
            #     meshObject['offsets_'] = offsets_

            #     with open(self.mesh_file + '.veclocity.cache', 'wb') as handle:
            #         pickle.dump(meshObject, handle, protocol=4)

        elif os.path.isfile(self.mesh_file + '.veclocity.cache') and cache:
            print('Read from velocity cache...')
            with open(self.mesh_file + '.veclocity.cache', 'rb') as handle:
                meshObject = pickle.load(handle)

            all_elements_ = meshObject['all_elements_']
            offsets_ = meshObject['offsets_']
        self.timer.node["velocity reconstruction"].stop()

        return all_elements_, offsets_


    def discretize(self, verbose: bool = True):
        cache = self.reservoir_cache
        self.discretizer.verbose = verbose
        self.discretizer.load_mesh(
            permx=self.permx,
            permy=self.permy,
            permz=self.permz,
            frac_aper=0,
            cache=cache,
        )

        self.discretizer.store_volume_all_cells()
        self.discretizer.store_depth_all_cells()
        self.discretizer.store_centroids_all_cells()
        # Keep the legacy centroid attribute alive for SPE11c well placement code.
        self.discretizer.centroid_all_cells = self.discretizer.centroids_all_cells

        self.cell_m, self.cell_p, self.tran, self.tran_thermal = (
            self.discretizer.calc_connections_all_cells(cache=cache)
        )

        mesh = conn_mesh()
        mesh.init(
            index_vector(self.cell_m),
            index_vector(self.cell_p),
            value_vector(self.tran),
            value_vector(self.tran_thermal),
        )

        np.array(mesh.poro, copy=False)[:] = self.poro
        np.array(mesh.rock_cond, copy=False)[:] = self.rcond
        np.array(mesh.heat_capacity, copy=False)[:] = self.hcap
        np.array(mesh.op_num, copy=False)[:] = self.op_num

        n_elements = self.discretizer.mat_cells_tot + self.discretizer.frac_cells_tot
        np.array(mesh.depth, copy=False)[:] = (
            2000 + self.discretizer.depth_all_cells[:n_elements][::-1]
        )
        np.array(mesh.volume, copy=False)[:] = self.discretizer.volume_all_cells[
            :n_elements
        ]

        self.centroids = self.discretizer.centroids_all_cells
        self.mesh = mesh
        return mesh


    def set_wells(self, verbose: bool = False):

        cache = self.wells_cache
        # self.timer.node["set_wells"].start()

        if not cache:
            t = np.linspace(0, 1, 1000)
            self.well_cells = []
            self.well_size = []
            for name, geometry in self.well_geometry.items():
                print(name, geometry)

                start = geometry['head']
                end = geometry['tail']
                pts = start + np.outer(t, end - start)
                if geometry['coordinates'] == 'reference':
                    pts = self.map_reference_to_cartesian(pts.T)

                # filter centroids to avoid crazy big distance matrix
                min_pts, max_pts = 0.75 * np.min(pts, axis=0), 1.25 * np.max(pts, axis=0)
                indices_bounding_box = np.where(
                    np.logical_and(
                        np.all(self.discretizer.centroid_all_cells >= min_pts, axis=1),
                        np.all(self.discretizer.centroid_all_cells <= max_pts, axis=1)
                    )
                )[0]
                filtered_centroids = self.discretizer.centroid_all_cells[indices_bounding_box]
                pts_expanded = pts[:, np.newaxis, :]

                distances = np.sum((pts_expanded - filtered_centroids) ** 2, axis=2)
                ids = np.argmin(distances, axis=1)
                ids_unique = np.unique(ids)
                self.well_cells.append(indices_bounding_box[ids_unique])

            # if not os.path.isfile(self.mesh_file + '.well_cells.cache'):
            #     print('Writing well_cells cache...')
            #     meshObject = {}
            #     meshObject['well_cells0'] = self.well_cells[0]
            #     meshObject['well_cells1'] = self.well_cells[1]
            #     with open(self.mesh_file + '.well_cells.cache', 'wb') as handle:
            #         pickle.dump(meshObject, handle, protocol=4)

        elif os.path.isfile(self.mesh_file + '.well_cells.cache') and cache:
            print('Read from well_cells cache...')
            with open(self.mesh_file + '.well_cells.cache', 'rb') as handle:
                meshObject = pickle.load(handle)

            self.well_cells = []
            self.well_cells.append(meshObject['well_cells0'])
            self.well_cells.append(meshObject['well_cells1'])

        if self.specs['RHS'] is False:
            for well_no, well in enumerate(self.well_cells):
                self.add_well("I%d" % well_no)
                for well_cell in well:
                    self.add_perforation("I%d" % well_no,
                                         res_cell_idx = well_cell,
                                         well_index = 10000,
                                         well_indexD = 100,
                                         ms_epm = True,
                                         verbose = True)
        else:
            pass

        # self.well_size = []
        # for i, well_cell in enumerate(self.well_cells):
        #     self.well_size.append(len(well_cell.well))


        # self.timer.node["set_wells"].stop()
        return

    def map_cartesian_to_reference(self, c):
        # implementation of equation 4.2 in SPE11 description
        u = c[0]
        v = c[1]
        w = c[2] - 150 * (1 - ((c[1] - 2500) / 2500) ** 2) - c[1] / 500
        return np.array([u, v, w]).T

    def map_reference_to_cartesian(self, r):
        # implementation of equation 4.1 in SPE11 description
        x = r[0]
        y = r[1]
        z = r[2] + 150 * (1 - ((r[1] - 2500) / 2500) ** 2) + r[1] / 500
        return np.array([x, y, z]).T

    def jacobian_reference_to_cartesian(self, r):
        if len(r.shape) > 1:
            F = np.eye(3)[:, :, np.newaxis] * np.ones((1, 1, r.shape[1]))
        else:
            F = np.eye(3)
        F[2, 1] = -3 / 25 * (r[1] - 2500) / 2500 + 1 / 500
        return F
