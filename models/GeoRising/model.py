from darts.models.reservoirs.struct_reservoir import StructReservoir
from darts.models.cicd_model import CICDModel
from darts.physics.properties.iapws.iapws_property_vec import _Backward1_T_Ph_vec
from darts.tools.keyword_file_tools import load_single_keyword
import numpy as np
from darts.engines import value_vector, sim_params

from darts.physics.geothermal.physics import Geothermal
from darts.physics.geothermal.property_container import PropertyContainer


class Model(CICDModel):
    def __init__(self, n_points=128):
        # call base class constructor
        super().__init__()

        self.timer.node["initialization"].start()

        self.set_reservoir()
        self.set_wells()
        self.set_physics(n_points)

        self.set_sim_params(first_ts=1e-3, mult_ts=8, max_ts=365, runtime=3650, tol_newton=1e-2, tol_linear=1e-6,
                            it_newton=20, it_linear=40, newton_type=sim_params.newton_global_chop,
                            newton_params=value_vector([1]))

        self.timer.node["initialization"].stop()

    def set_reservoir(self):
        (nx, ny, nz) = (60, 60, 3)
        nb = nx * ny * nz
        perm = np.ones(nb) * 2000
        perm = load_single_keyword('permXVanEssen.in', 'PERMX')
        perm = perm[:nb]

        poro = np.ones(nb) * 0.2
        dx = 30
        dy = 30
        dz = np.ones(nb) * 30

        # discretize structured reservoir
        self.reservoir = StructReservoir(self.timer, nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz, permx=perm,
                                         permy=perm, permz=perm * 0.1, poro=poro, depth=2000)

        self.reservoir.set_boundary_volume(xz_minus=1e8, xz_plus=1e8, yz_minus=1e8, yz_plus=1e8)

        # # rock heat capacity and rock thermal conduction
        # hcap = np.array(self.reservoir.mesh.heat_capacity, copy=False)
        # rcond = np.array(self.reservoir.mesh.rock_cond, copy=False)
        # hcap.fill(2200)
        # rcond.fill(500)

        return

    def set_wells(self):
        # add well's locations
        self.iw = [30, 30]
        self.jw = [14, 46]

        # add well
        self.reservoir.add_well("INJ")
        n_perf = self.reservoir.nz
        # add perforations to te payzone
        for n in range(1, n_perf):
            self.reservoir.add_perforation(well=self.reservoir.wells[-1], i=self.iw[0], j=self.jw[0], k=n + 1,
                                           well_radius=0.16)

        # add well
        self.reservoir.add_well("PRD")
        # add perforations to te payzone
        for n in range(1, n_perf):
            self.reservoir.add_perforation(self.reservoir.wells[-1], self.iw[1], self.jw[1], n + 1, 0.16)

        # rock heat capacity and rock thermal conduction
        hcap = np.array(self.reservoir.mesh.heat_capacity, copy=False)
        rcond = np.array(self.reservoir.mesh.rock_cond, copy=False)
        hcap.fill(2200)
        rcond.fill(500)

        return

    def set_physics(self, n_points):
        # create pre-defined physics for geothermal
        property_container = PropertyContainer()
        self.physics = Geothermal(self.timer, n_points=n_points, min_p=1, max_p=351, min_e=1000, max_e=10000, cache=False)
        self.physics.add_property_region(property_container)
        self.physics.init_physics()
        return

    def set_initial_conditions(self):
        self.physics.set_uniform_initial_conditions(self.reservoir.mesh, uniform_pressure=200,
                                                    uniform_temperature=350)

    def set_boundary_conditions(self):
        for i, w in enumerate(self.reservoir.wells):
            if i == 0:
                w.control = self.physics.new_rate_water_inj(8000, 300)
                # w.control = self.physics.new_bhp_water_inj(230, 308.15)
            else:
                w.control = self.physics.new_rate_water_prod(8000)
                # w.control = self.physics.new_bhp_prod(180)

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
        temp = _Backward1_T_Ph_vec(X[0:2 * nb:2] / 10, X[1:2 * nb:2] / 18.015)
        local_cell_data = {'Temperature': temp,
                           'Perm': self.reservoir.global_data['permx'][self.reservoir.discretizer.local_to_global]}

        self.export_vtk(file_name, local_cell_data=local_cell_data)
