from darts.models.reservoirs.struct_reservoir import StructReservoir
from darts.models.physics.geothermal import Geothermal
from darts.models.darts_model import DartsModel, sim_params
from darts.models.physics.iapws.iapws_property_vec import _Backward1_T_Ph_vec
from darts.tools.keyword_file_tools import load_single_keyword
import numpy as np
from darts.engines import value_vector


class Model(DartsModel):
    def __init__(self, n_points=128):
        # call base class constructor
        super().__init__()

        self.timer.node["initialization"].start()

        res = 2
        (nx, ny, nz) = (20 * res, 60 * res, 20 * res)
        nb = nx * ny * nz
        perm = np.ones(nb) * 2000
        #perm = load_single_keyword('permXVanEssen.in', 'PERMX')
        perm = perm[:nb]

        poro = np.ones(nb) * 0.2
        self.dx = 20 / res
        self.dy = 20 / res
        dz = np.ones(nb) * 20 / res

        # discretize structured reservoir
        self.reservoir = StructReservoir(self.timer, nx=nx, ny=ny, nz=nz, dx=self.dx, dy=self.dy, dz=dz, permx=perm,
                                         permy=perm, permz=perm * 0.1, poro=poro, depth=2000)

        self.reservoir.set_boundary_volume(xz_minus=1e8, xz_plus=1e8, yz_minus=1e8, yz_plus=1e8, xy_minus=1e8,
                                           xy_plus=1e8)
        # add well's locations
        self.iw = [10*res, 10*res]
        self.jw = [4*res, 56*res]

        n = int(nz/2)
        j_mid = int(ny/2)

        well_radius = 0.3

        # add well
        self.reservoir.add_well("INJ")
        # add perforations to te payzone
        for j in range(self.jw[0], j_mid + 1):
            self.reservoir.add_perforation(well=self.reservoir.wells[-1], i=self.iw[0], j=j, k=n + 1,
                                           well_radius=well_radius, segment_direction='y_axis', well_index=-2)

        # self.reservoir.add_perforation(well=self.reservoir.wells[-1], i=self.iw[0], j=j_mid, k=n + 1,
        #                                well_radius=0.16, segment_direction='y_axis')

        # add well
        self.reservoir.add_well("PRD")
        # add perforations to te payzone
        for j in range(self.jw[1], j_mid, -1):
            self.reservoir.add_perforation(self.reservoir.wells[-1], self.iw[1], j, n + 1, well_radius,
                                           segment_direction='y_axis', well_index=-2)

        # self.reservoir.add_perforation(self.reservoir.wells[-1], self.iw[1], j_mid, n + 1, 0.16,
        #                                segment_direction='y_axis')

        # rock heat capacity and rock thermal conduction
        hcap = np.array(self.reservoir.mesh.heat_capacity, copy=False)
        rcond = np.array(self.reservoir.mesh.rock_cond, copy=False)
        hcap.fill(2200)
        rcond.fill(500)

        # create pre-defined physics for geothermal
        self.physics = Geothermal(self.timer, n_points, 1, 351, 1000, 50000, cache=False)

        self.params.first_ts = 1e-5
        self.params.mult_ts = 8
        self.params.max_ts = 31

        # Newton tolerance is relatively high because of L2-norm for residual and well segments
        self.params.tolerance_newton = 1e-4
        self.params.tolerance_linear = 1e-6
        self.params.max_i_newton = 20
        self.params.max_i_linear = 40

        self.params.newton_type = sim_params.newton_global_chop
        self.params.newton_params = value_vector([1])

        self.runtime = 365
        # self.physics.engine.silent_mode = 0
        self.timer.node["initialization"].stop()

    def set_initial_conditions(self):
        self.physics.set_uniform_initial_conditions(self.reservoir.mesh, uniform_pressure=200,
                                                    uniform_temperature=450)

    def set_boundary_conditions(self):
        for i, w in enumerate(self.reservoir.wells):
            if i == 0:
                #w.control = self.physics.new_rate_water_inj(8000, 300)
                w.control = self.physics.new_bhp_water_inj(205, 300)
            else:
                #w.control = self.physics.new_rate_water_prod(8000)
                w.control = self.physics.new_bhp_prod(195)

    def compute_temperature(self, X):
        from darts.models.physics.iapws.iapws_property_vec import _Backward1_T_Ph_vec
        nb = self.reservoir.mesh.n_res_blocks
        temp = _Backward1_T_Ph_vec(X[0:2 * nb:2] / 10, X[1:2 * nb:2] / 18.015)
        return temp

    def set_op_list(self):
        self.op_list = [self.physics.acc_flux_itor, self.physics.acc_flux_itor_well]
        op_num = np.array(self.reservoir.mesh.op_num, copy=False)
        op_num[self.reservoir.mesh.n_res_blocks:] = 1


    def export_pro_vtk(self, file_name='Results'):
        X = np.array(self.physics.engine.X, copy=False)
        nb = self.reservoir.mesh.n_res_blocks
        temp = _Backward1_T_Ph_vec(X[0:2 * nb:2] / 10, X[1:2 * nb:2] / 18.015)
        local_cell_data = {'Temperature': temp,
                           'Perm': self.reservoir.global_data['permx'][self.reservoir.discretizer.local_to_global]}

        self.export_vtk(file_name, local_cell_data=local_cell_data)