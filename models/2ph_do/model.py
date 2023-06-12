from darts.models.reservoirs.struct_reservoir import StructReservoir
from darts.models.darts_model import DartsModel
from darts.engines import value_vector
import numpy as np

from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer

from darts.physics.properties.basic import ConstFunc, PhaseRelPerm
from darts.physics.properties.density import DensityBasic


class Model(DartsModel):
    def __init__(self):
        # call base class constructor
        super().__init__()

        # measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        """Reservoir construction"""
        # reservoir geometry： for realistic case, one just needs to load the data and input it
        self.reservoir = StructReservoir(self.timer, nx=100, ny=1, nz=1, dx=10.0, dy=10.0, dz=1, permx=300, permy=300,
                                         permz=300, poro=0.2, depth=100)

        hcap = np.array(self.reservoir.mesh.heat_capacity, copy=False)
        rcond = np.array(self.reservoir.mesh.rock_cond, copy=False)

        # well model or boundary conditions
        self.reservoir.add_well("I1")
        self.reservoir.add_perforation(well=self.reservoir.wells[-1], i=1, j=1, k=1, multi_segment=False)

        self.reservoir.add_well("P1")
        self.reservoir.add_perforation(self.reservoir.wells[-1], 100, 1, 1, multi_segment=False)

        """Physical properties"""
        self.zero = 1e-13
        components = ['w', 'o']
        phases = ['wat', 'oil']
        property_container = ModelProperties(phases_name=phases, components_name=components, min_z=self.zero/10)

        property_container.density_ev = dict([('wat', DensityBasic(compr=1e-5, dens0=1014)),
                                              ('oil', DensityBasic(compr=5e-3, dens0=500))])
        property_container.viscosity_ev = dict([('wat', ConstFunc(0.3)),
                                                ('oil', ConstFunc(0.03))])
        property_container.rel_perm_ev = dict([('wat', PhaseRelPerm("wat", 0.1, 0.1)),
                                               ('oil', PhaseRelPerm("oil", 0.1, 0.1))])

        # create physics
        self.physics = Compositional(components, phases, self.timer,
                                     n_points=400, min_p=0, max_p=1000, min_z=self.zero, max_z=1 - self.zero)
        self.physics.add_property_region(property_container)
        self.physics.init_physics()

        self.params.first_ts = 0.01
        self.params.mult_ts = 2
        self.params.max_ts = 5
        self.params.tolerance_newton = 1e-3
        self.params.tolerance_linear = 1e-6
        # self.params.newton_type = 2
        # self.params.newton_params = value_vector([0.2])

        self.runtime = 300
        self.inj = value_vector([self.zero])
        self.ini = value_vector([1 - self.zero])

        self.timer.node["initialization"].stop()

    def set_initial_conditions(self):
        self.physics.set_uniform_initial_conditions(self.reservoir.mesh, uniform_pressure=400,
                                                      uniform_composition=self.ini)

    def set_boundary_conditions(self):
        for i, w in enumerate(self.reservoir.wells):
            if i == 0:
                w.control = self.physics.new_rate_inj(200, self.inj, 1)
                w.constraint = self.physics.new_bhp_inj(450, self.inj)
                #w.control = self.physics.new_bhp_inj(450, self.inj)
            else:
                w.control = self.physics.new_bhp_prod(350)


class ModelProperties(PropertyContainer):
    def __init__(self, phases_name, components_name, min_z=1e-11):
        # Call base class constructor
        self.nph = len(phases_name)
        Mw = np.ones(self.nph)
        super().__init__(phases_name, components_name, Mw, min_z, temperature=1.)

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

        ph = [0, 1]

        for j in ph:
            # molar weight of mixture
            M = np.sum(self.x[j, :] * self.Mw)
            self.dens[j] = self.density_ev[self.phases_name[j]].evaluate(pressure)  # output in [kg/m3]
            self.dens_m[j] = self.dens[j] / M
            self.mu[j] = self.viscosity_ev[self.phases_name[j]].evaluate()  # output in [cp]

        self.nu = zc
        self.compute_saturation(ph)

        for j in ph:
            self.kr[j] = self.rel_perm_ev[self.phases_name[j]].evaluate(self.sat[j])
            self.pc[j] = 0

        mass_source = np.zeros(self.nc)

        return ph, self.sat, self.x, self.dens, self.dens_m, self.mu, self.kr, self.pc, mass_source

    def evaluate_at_cond(self, pressure, zc):
        self.sat[:] = 0

        ph = [0, 1]
        for j in ph:
            self.dens_m[j] = self.density_ev[self.phases_name[j]].evaluate(1, 0)

        self.dens_m = [1025, 0.77]  # to match DO based on PVT

        self.nu = zc
        self.compute_saturation(ph)

        return self.sat, self.dens_m
