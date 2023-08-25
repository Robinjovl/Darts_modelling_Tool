from darts.models.reservoirs.struct_reservoir import StructReservoir
from darts.models.cicd_model import CICDModel
from darts.physics.properties.iapws.iapws_property_vec import _Backward1_T_Ph_vec
from darts.tools.keyword_file_tools import load_single_keyword
import numpy as np
from darts.engines import value_vector, sim_params

from darts.physics.geothermal.physics import Geothermal
from darts.physics.geothermal.property_container import PropertyContainer


class Model(CICDModel):
    def __init__(self, resolution=10, n_points=128):
        # call base class constructor
        super().__init__()

        self.timer.node["initialization"].start()

        self.set_reservoir(resolution)
        self.set_wells(resolution)
        self.set_physics(n_points)

        self.set_sim_params(first_ts=1e-5, mult_ts=8, max_ts=31, runtime=365, tol_newton=1e-4, tol_linear=1e-6,
                            it_newton=20, it_linear=40, newton_type=sim_params.newton_global_chop,
                            newton_params=value_vector([1]))

        self.timer.node["initialization"].stop()

    def set_reservoir(self, resolution):
        y_scale = 3
        (nx, ny, nz) = (resolution, y_scale * resolution, resolution)
        nb = nx * ny * nz
        perm = np.ones(nb) * 2000

        poro = np.ones(nb) * 0.2
        dx = 20. / resolution
        dy = 20. / resolution
        dz = 20. / resolution

        # discretize structured reservoir
        self.reservoir = StructReservoir(self.timer, nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz,
                                         permx=perm, permy=perm, permz=perm * 0.1, poro=poro, depth=2000)

        self.reservoir.set_boundary_volume(xz_minus=1e8, xz_plus=1e8, yz_minus=1e8, yz_plus=1e8, xy_minus=1e8,
                                           xy_plus=1e8)

        # # rock heat capacity and rock thermal conduction
        # hcap = np.array(self.reservoir.mesh.heat_capacity, copy=False)
        # rcond = np.array(self.reservoir.mesh.rock_cond, copy=False)
        # hcap.fill(2200)
        # rcond.fill(500)

        return

    def set_wells(self, resolution):
        # add well's start locations
        self.iw = [resolution//2, resolution//2]
        self.jw = [4, self.reservoir.ny - 4]

        n = self.reservoir.nz//2
        j_mid = self.reservoir.ny//2

        well_radius = 0.3

        # add well
        self.reservoir.add_well("INJ")
        # add perforations with well_index=0 (closed pipe, only thermal losses)
        for j in range(self.jw[0], j_mid + 1):
            self.reservoir.add_perforation(well=self.reservoir.wells[-1], i=self.iw[0], j=j, k=n + 1,
                                           well_radius=well_radius, segment_direction='y_axis',
                                           well_index=0)#, verbose=True)
        self.reservoir.add_well("PRD")
        # add perforations with well_index=0 (closed pipe, only thermal losses)
        for j in range(self.jw[1], j_mid, -1):
            self.reservoir.add_perforation(well=self.reservoir.wells[-1], i=self.iw[1], j=j, k=n + 1,
                                           well_radius=well_radius, segment_direction='y_axis',
                                           well_index=0)#, verbose=True)

        # connect the last two perforations of two wells
        well_1 = self.reservoir.wells[0]
        well_2 = self.reservoir.wells[1]
        perf_1 = len(well_1.perforations)  # last segment is n_perf+1
        perf_2 = len(well_2.perforations)
        # dictionary: key is a pair of 2 well names; value is a list of well perforation indices to connect
        self.reservoir.connected_well_segments = {(well_1.name, well_2.name): [(perf_1, perf_2)]}

        # rock heat capacity and rock thermal conduction
        hcap = np.array(self.reservoir.mesh.heat_capacity, copy=False)
        rcond = np.array(self.reservoir.mesh.rock_cond, copy=False)
        hcap.fill(2200)
        rcond.fill(500)
        return

    def set_physics(self, n_points):
        # create pre-defined physics for geothermal
        property_container = PropertyContainer()
        self.physics = Geothermal(self.timer, n_points=n_points, min_p=1, max_p=351, min_e=1000, max_e=50000, cache=False)
        self.physics.add_property_region(property_container)
        self.physics.init_physics()
        return

    def set_initial_conditions(self):
        self.physics.set_uniform_initial_conditions(self.reservoir.mesh, uniform_pressure=200,
                                                    uniform_temperature=450)

    def set_boundary_conditions(self):
        for i, w in enumerate(self.reservoir.wells):
            if i == 0:
                w.control = self.physics.new_bhp_water_inj(205, 300)
            else:
                w.control = self.physics.new_bhp_prod(195)

    def compute_temperature(self, X):
        nb = self.reservoir.mesh.n_res_blocks
        temp = _Backward1_T_Ph_vec(X[0:2 * nb:2] / 10, X[1:2 * nb:2] / 18.015)
        return temp

    def set_op_list(self):
        self.op_list = [self.physics.acc_flux_itor[0], self.physics.acc_flux_w_itor]
        op_num = np.array(self.reservoir.mesh.op_num, copy=False)
        op_num[self.reservoir.mesh.n_res_blocks:] = 1

    def export_pro_vtk(self, file_name='Results'):
        X = np.array(self.physics.engine.X, copy=False)
        nb = self.reservoir.mesh.n_res_blocks
        temp = self.compute_temperature(X)
        local_cell_data = {'Temperature': temp,
                           'Perm': self.reservoir.global_data['permx'][self.reservoir.discretizer.local_to_global]}

        self.export_vtk(file_name, local_cell_data=local_cell_data)
