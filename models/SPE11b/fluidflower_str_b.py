import numpy as np
from darts.reservoirs.struct_reservoir import StructReservoir

from darts.reservoirs.mesh.geometry.structured import Structured
# from darts.reservoirs.mesh.geometry.fluidflower import FluidFlower
from fluidflower import FluidFlower


class FluidFlowerStruct(StructReservoir):
    def __init__(self, timer, layer_properties, layers_to_regions, model_specs, well_centers):
        str_mesh = Structured(dim=2, axs=[0, 2])

        spe11b = FluidFlower(lc=[0.1])
        spe11b.convert_to_spe11b()
        str_mesh.add_shape(spe11b)
        # str_mesh.add_shape(FluidFlower(lc=[0.1]))

        self.specs= model_specs
        nx, ny, nz = self.specs['nx'], 1, self.specs['nz']
        x, y, z, cell_to_layer, actnum = str_mesh.generate_mesh2(nx=nx, ny=ny, nz=nz)

        dx, dy, dz = x[1] - x[0], 1.0, z[1] - z[0]

        # Define depth, porosity and permeability for each cell
        grid_size = nx * ny * nz
        ACTNUM, depth, poro, permx, permy, permz, hcap, rcond, op_num = (
            np.zeros(grid_size), np.zeros(grid_size), np.zeros(grid_size), np.zeros(grid_size), np.zeros(grid_size),
            np.zeros(grid_size), np.zeros(grid_size), np.zeros(grid_size), np.zeros(grid_size, dtype=int)
        )

        n_local = np.sum(actnum)
        self.centroids = np.zeros((n_local, 3))
        self.seal = []
        self.bot_cells = []
        self.top_cells = []
        self.boundary_cells = []
        self.well_cells = []
        local_idx = 0
        # water_column_depth = 1200.0
        self.min_depth = 1200. - nz * dz
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    global_idx = k * nx * ny + j * nx + i
                    depth[global_idx] = 1200. - k * dz
                    if actnum[i, j, k]:
                        tag = cell_to_layer[i, j, k] - 1 + 900001
                        ACTNUM[global_idx] = 1
                        poro[global_idx] = layer_properties[tag].poro
                        perm = layer_properties[tag].perm
                        anisotropy = layer_properties[tag].anisotropy
                        permx[global_idx] = perm * anisotropy[0]
                        permy[global_idx] = perm * anisotropy[1]
                        permz[global_idx] = perm * anisotropy[2]
                        hcap[global_idx] = layer_properties[tag].hcap
                        rcond[global_idx] = layer_properties[tag].rcond
                        op_num[global_idx] = layers_to_regions[layer_properties[tag].type]

                        self.centroids[local_idx] = np.array([x[i], y[j] + 0.5 * dy, z[k]])
                        # self.centroids[local_idx] = np.array([x[i] + 0.5 * dx, y[j] + 0.5 * dy, z[k] + 0.5 * dz])
                        # poro[local_idx] = poro[global_idx]
                        # op_num[local_idx] = op_num[global_idx]
                        if layer_properties[tag].type == "1":
                            self.seal.append(local_idx)
                        if i == 0 or i == (nx - 1):
                            self.boundary_cells.append(local_idx)
                        local_idx += 1
        for i in range(len(depth)):
            if depth[i] > 1199.9:
                self.bot_cells.append(i)
            elif depth[i] < dz * 1.01:
                self.top_cells.append(i)
        super().__init__(timer, nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz, permx=permx, permy=permy, permz=permz,
                         poro=poro, rcond=rcond, hcap=hcap, op_num=op_num, depth=depth, actnum=actnum)

        self.well_centers = well_centers

    def set_layer_properties(self):
        pass

    def set_wells(self, verbose: bool = False):
        from darts.engines import value_vector
        
        if self.specs['RHS']:
            for name, center in self.well_centers.items():
                cell_index = self.find_cell_index(center)
                self.well_cells.append(cell_index)

        else:
            self.well_cells = []
            for name, center in self.well_centers.items():
                cell_index = self.find_cell_index(center)
                self.well_cells.append(cell_index)

            for well_nr in range(2):
                k = int(self.well_cells[well_nr] / (self.nx * self.ny) - 1)
                i = int(np.abs(self.nx - (self.well_cells[well_nr] - k * (self.nx * self.ny))))
                j = 1

                try:
                    assert k * self.nx * self.ny + j * self.nx + i == self.well_cells[well_nr]
                except:
                    print(f"Assertion Failed: (i={i}, j={j}, k={k})")
                    print(f"Computed Index: {k * self.nx * self.ny + j * self.nx + i}")
                    print(f"Expected Index: {self.well_cells[well_nr]}")
                    raise

                self.add_well("I%d" % well_nr)
                self.add_perforation("I%d" % well_nr, cell_index=(i, j, k), well_indexD=0, verbose=True)
                
                # well_idx = self.wells[well_nr].well_body_idx 
                # self.mesh.op_num[well_idx] = self.mesh.op_num[self.well_cells[well_nr]]
                
        return