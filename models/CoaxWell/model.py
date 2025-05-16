from darts.reservoirs.struct_reservoir import StructReservoir
from darts.models.cicd_model import CICDModel
from darts.physics.properties.iapws.iapws_property_vec import _Backward1_T_Ph_vec
from darts.tools.keyword_file_tools import load_single_keyword
import numpy as np
from darts.engines import value_vector, sim_params
from darts.engines import well_control_iface

from darts.physics.geothermal.physics import Geothermal
from darts.physics.geothermal.property_container import PropertyContainer


class Model(CICDModel):
    def __init__(self, resolution=10, n_points=128):
        # call base class constructor
        super().__init__()

        self.timer.node["initialization"].start()

        self.resolution = resolution
        self.set_reservoir(resolution)
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
                                         permx=perm, permy=perm, permz=perm * 0.1, poro=poro, depth=2000, hcap=2200, rcond=500
                                         )
        self.reservoir.boundary_volumes['xy_minus'] = 1e8
        self.reservoir.boundary_volumes['xy_plus'] = 1e8
        self.reservoir.boundary_volumes['yz_minus'] = 1e8
        self.reservoir.boundary_volumes['yz_plus'] = 1e8
        self.reservoir.boundary_volumes['xz_minus'] = 1e8
        self.reservoir.boundary_volumes['xz_plus'] = 1e8

        return

    def set_wells(self):
        # add well's start locations
        iw = [self.resolution // 2, self.resolution // 2]
        jw = [4, self.reservoir.ny - 4]

        n = self.reservoir.nz // 2
        j_mid = self.reservoir.ny // 2

        well_radius = 0.3

        # add well
        self.reservoir.add_well("INJ")
        for j in range(jw[0], j_mid + 1):
            self.reservoir.add_perforation("INJ", cell_index=(iw[0], j, n + 1), well_radius=well_radius,
                                           segment_direction='y_axis', well_index=0, multi_segment=True)
        perf_1 = len(self.reservoir.wells[-1].perforations)  # last segment is n_perf+1

        self.reservoir.add_well("PRD")
        for j in range(jw[1], j_mid, -1):
            self.reservoir.add_perforation("PRD", cell_index=(iw[1], j, n + 1), well_radius=well_radius,
                                           segment_direction='y_axis', well_index=0, multi_segment=True)
        perf_2 = len(self.reservoir.wells[-1].perforations)

        # connect the last two perforations of two wells
        # dictionary: key is a pair of 2 well names; value is a list of well perforation indices to connect
        self.reservoir.connected_well_segments = {
            (self.reservoir.wells[0].name, self.reservoir.wells[1].name): [(perf_1, perf_2)]
        }

    def set_physics(self, n_points):
        # create pre-defined physics for geothermal
        property_container = PropertyContainer()
        self.physics = Geothermal(self.timer, n_points=n_points, min_p=1, max_p=351, min_e=1000, max_e=50000, cache=False)
        self.physics.add_property_region(property_container)

        return

    def set_initial_conditions(self):
        input_distribution = {'pressure': 200.,
                              'temperature': 450.
                              }
        return self.physics.set_initial_conditions_from_array(mesh=self.reservoir.mesh,
                                                              input_distribution=input_distribution)

    def set_well_controls(self):
        for i, w in enumerate(self.reservoir.wells):
            if i == 0:
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.BHP,
                                               is_inj=True, target=205, phase_name='water', inj_temp=300.)
            else:
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.BHP,
                                               is_inj=False, target=195., phase_name='water')

    def compute_temperature(self, X):
        nb = self.reservoir.mesh.n_blocks
        temp = _Backward1_T_Ph_vec(X[0:2 * nb:2] / 10, X[1:2 * nb:2] / 18.015)
        return temp
