from darts.reservoirs.struct_reservoir import StructReservoir
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

        T_init = 350.
        state_init = value_vector([200., 0.])
        enth_init = self.physics.property_containers[0].total_enthalpy(T_init).evaluate(state_init)
        self.initial_values = {self.physics.vars[0]: state_init[0],
                               self.physics.vars[1]: enth_init
                               }

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
        reservoir = StructReservoir(self.timer, nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz,
                                    permx=perm, permy=perm, permz=perm * 0.1, poro=poro, depth=2000,
                                    hcap=2200, rcond=500)
        reservoir.boundary_volumes['yz_minus'] = 1e8
        reservoir.boundary_volumes['yz_plus'] = 1e8
        reservoir.boundary_volumes['xz_minus'] = 1e8
        reservoir.boundary_volumes['xz_plus'] = 1e8

        # add well's locations
        iw = [30, 30]
        jw = [14, 46]

        # add well
        perf_list = [(iw[0], jw[0], k+1) for k in range(1, nz)]
        reservoir.add_well("INJ", perf_list=perf_list, well_radius=0.16, multi_segment=True)

        # add well
        perf_list = [(iw[1], jw[1], k+1) for k in range(1, nz)]
        reservoir.add_well("PRD", perf_list=perf_list, well_radius=0.16, multi_segment=True)

        return super().set_reservoir(reservoir)

    def set_physics(self, n_points):
        # create pre-defined physics for geothermal
        property_container = PropertyContainer()
        physics = Geothermal(self.timer, n_points=n_points, min_p=1, max_p=351, min_e=1000, max_e=10000, cache=False)
        physics.add_property_region(property_container)

        return super().set_physics(physics)

    def set_well_controls(self):
        for i, w in enumerate(self.reservoir.wells):
            if i == 0:
                w.control = self.physics.new_rate_water_inj(8000, 300)
                # w.control = self.physics.new_bhp_water_inj(230, 308.15)
            else:
                w.control = self.physics.new_rate_water_prod(8000)
                # w.control = self.physics.new_bhp_prod(180)

    def compute_temperature(self, X):
        nb = self.mesh.n_res_blocks
        temp = _Backward1_T_Ph_vec(X[0:2 * nb:2] / 10, X[1:2 * nb:2] / 18.015)
        return temp

    def set_op_list(self):
        self.op_list = [self.physics.acc_flux_itor[0], self.physics.acc_flux_w_itor]
        op_num = np.array(self.mesh.op_num, copy=False)
        op_num[self.mesh.n_res_blocks:] = 1

    def export_pro_vtk(self, file_name='Results'):
        X = np.array(self.engine.X, copy=False)
        nb = self.mesh.n_res_blocks
        temp = _Backward1_T_Ph_vec(X[0:2 * nb:2] / 10, X[1:2 * nb:2] / 18.015)
        local_cell_data = {'Temperature': temp,
                           'Perm': self.reservoir.global_data['permx'][self.reservoir.discretizer.local_to_global]
                           }

        self.export_vtk(file_name, local_cell_data=local_cell_data)
