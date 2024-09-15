import numpy as np

from model_cpg import Model_CPG

from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer
from darts.physics.properties.basic import ConstFunc, PhaseRelPerm
from darts.physics.properties.density import DensityBasic

from darts.engines import value_vector
class ModelPropertiesDeadOil(PropertyContainer):
    def __init__(self, phases_name, components_name, min_z=1e-11):
        # Call base class constructor
        self.nph = len(phases_name)
        Mw = np.ones(self.nph)
        super().__init__(phases_name=phases_name, components_name=components_name, Mw=Mw, min_z=min_z,
                         temperature=1.)

    def evaluate(self, state):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        # Composition vector and pressure from state:
        vec_state_as_np = np.asarray(state)
        pressure = vec_state_as_np[0]

        zc = np.append(vec_state_as_np[1:], 1 - np.sum(vec_state_as_np[1:]))

        self.clean_arrays()
        # two-phase flash - assume water phase is always present and water component last
        for i in range(self.nph):
            self.x[i, i] = 1

        self.ph = [0, 1]

        for j in self.ph:
            # molar weight of mixture
            M = np.sum(self.x[j, :] * self.Mw)
            self.dens[j] = self.density_ev[self.phases_name[j]].evaluate(pressure)  # output in [kg/m3]
            self.dens_m[j] = self.dens[j] / M
            self.mu[j] = self.viscosity_ev[self.phases_name[j]].evaluate()  # output in [cp]

        self.nu = zc
        self.compute_saturation(self.ph)

        for j in self.ph:
            self.kr[j] = self.rel_perm_ev[self.phases_name[j]].evaluate(self.sat[j])
            self.pc[j] = 0

        return

    def evaluate_at_cond(self, pressure, zc):
        self.sat[:] = 0

        ph = [0, 1]
        for j in ph:
            self.dens_m[j] = self.density_ev[self.phases_name[j]].evaluate(1, 0)

        self.dens_m = [1025, 0.77]  # to match DO based on PVT

        self.nu = zc
        self.compute_saturation(ph)

        return self.sat, self.dens_m
class ModelDeadOil(Model_CPG):
    def __init__(self, discr_type='cpp', case='generate', grid_out_dir=None, n_points=400):
        super().__init__(discr_type=discr_type, case=case, grid_out_dir=grid_out_dir, n_points=n_points)
    def set_physics(self):
        zero = 1e-13
        components = ["w", "o"]
        phases = ["wat", "oil"]

        self.inj = value_vector([zero])
        self.ini = value_vector([1 - zero])

        property_container = ModelPropertiesDeadOil(phases_name=phases, components_name=components, min_z=zero/10)

        property_container.density_ev = dict([('wat', DensityBasic(compr=1e-5, dens0=1014)),
                                              ('oil', DensityBasic(compr=5e-3, dens0=700))])
        property_container.viscosity_ev = dict([('wat', ConstFunc(0.89)),
                                                ('oil', ConstFunc(50))])
        property_container.rel_perm_ev = dict([('wat', PhaseRelPerm("wat", 0.1, 0.1)),
                                               ('oil', PhaseRelPerm("oil", 0.1, 0.1))])

        # create physics
        self.physics = Compositional(components, phases, self.timer,
                                     n_points=self.n_points, min_p=0, max_p=1000, min_z=zero, max_z=1 - zero)
        self.physics.add_property_region(property_container)

        self.initial_values = {self.physics.vars[0]: 400,
                               self.physics.vars[1]: self.ini,
                               }

    def set_well_controls(self):
        for i, w in enumerate(self.reservoir.wells):
            if self.well_is_inj(w.name):  # INJ well
                # BHP control
                w.control = self.physics.new_bhp_inj(450, self.inj)  # bars
                # rate control
                #w.control = self.physics.new_rate_inj(200, self.inj, 0)  # Kmol/day, composition, composition-index
                #w.constraint = self.physics.new_bhp_inj(450, self.inj)   # bars, composition
            else:  # PROD well
                # BHP control
                w.control = self.physics.new_bhp_prod(350)  # bars
                # rate control
                #w.control = self.physics.new_rate_prod(200)   # Kmol/day
                #w.constraint = self.physics.new_bhp_prod(350) # bars