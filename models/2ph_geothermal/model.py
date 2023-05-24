from darts.models.reservoirs.struct_reservoir import StructReservoir
from darts.models.darts_model import DartsModel
from darts.models.physics_sup.physics_comp_sup import Compositional
from darts.engines import value_vector
import numpy as np
from darts.models.physics_sup.property_container import PropertyContainer
from darts.models.physics_sup.properties_basic import ConstFunc, Density, Enthalpy, PhaseRelPerm

class Model(DartsModel):

    def __init__(self):
        # call base class constructor
        super().__init__()

        # measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        """Reservoir construction"""
        # reservoir geometry： for realistic case, one just needs to load the data and input it
        self.reservoir = StructReservoir(self.timer, nx=500, ny=1, nz=1, dx=10.0, dy=10.0, dz=1, permx=300, permy=300,
                                         permz=300, poro=0.2, depth=100)

        hcap = np.array(self.reservoir.mesh.heat_capacity, copy=False)
        rcond = np.array(self.reservoir.mesh.rock_cond, copy=False)

        hcap.fill(2200)
        rcond.fill(181.44)

        # well model or boundary conditions
        self.reservoir.add_well("I1")
        self.reservoir.add_perforation(well=self.reservoir.wells[-1], i=1, j=1, k=1, multi_segment=False)

        self.reservoir.add_well("P1")
        self.reservoir.add_perforation(self.reservoir.wells[-1], 500, 1, 1, multi_segment=False)

        self.zero = 1e-13

        """Physical properties"""
        self.property_container = model_properties(phases_name=['wat', 'gas'], components_name=['w'],
                                                   min_z=self.zero/10)

        # Define property evaluators based on custom properties

        self.flash_ev = []
        self.property_container.density_ev = dict([('wat', Density(compr=1e-5, dens0=1014)),
                                                   ('gas', Density(compr=5e-3, dens0=50))])
        self.property_container.viscosity_ev = dict([('wat', ConstFunc(0.3)),
                                                     ('gas', ConstFunc(0.03))])
        self.property_container.rel_perm_ev = dict([('wat', PhaseRelPerm("oil", 0.1, 0.1)),
                                                    ('gas', PhaseRelPerm("gas", 0.1, 0.1))])
        self.property_container.enthalpy_ev = dict([('wat', Enthalpy(hcap=4.18)),
                                                    ('gas', Enthalpy(hcap=0.035))])
        self.property_container.conductivity_ev = dict([('wat', ConstFunc(1.)),
                                                        ('gas', ConstFunc(1.))])

        self.property_container.rock_energy_ev = Enthalpy(hcap=1.0)


        # create physics
        self.thermal = 1
        self.physics = Compositional(self.property_container, self.property_container.components_name, self.property_container.phases_name,
                                     self.timer, n_points=400, min_p=0, max_p=1000, min_z=self.zero, max_z=1-self.zero,
                                     min_t=273.15 + 20, max_t=273.15 + 200, thermal=self.thermal)
        self.params.first_ts = 0.0001
        self.params.mult_ts = 2
        self.params.max_ts = 5
        self.params.tolerance_newton = 1e-3
        self.params.tolerance_linear = 1e-6
        # self.params.newton_type = 2
        # self.params.newton_params = value_vector([0.2])

        self.runtime = 1000
        self.inj = value_vector([300])

        self.timer.node["initialization"].stop()

    def set_initial_conditions(self):
        self.physics.set_uniform_T_initial_conditions(self.reservoir.mesh, uniform_pressure=200,
                                                      uniform_composition=[1], uniform_temp=350)

    def set_boundary_conditions(self):
        for i, w in enumerate(self.reservoir.wells):
            if i == 0:
                #w.control = self.physics.new_rate_inj(200, self.inj, 1)
                #w.control = self.physics.new_bhp_inj(210, self.inj)
                w.control = self.physics.new_rate_inj(5, self.inj, 0)
                #w.control = self.physics.new_bhp_inj(450, self.inj)
            else:
                w.control = self.physics.new_bhp_prod(180)


class model_properties(PropertyContainer):
    def __init__(self, phases_name, components_name, min_z=1e-11):
        # Call base class constructor
        self.nph = len(phases_name)
        Mw = np.ones(self.nph)
        super().__init__(phases_name, components_name, Mw, min_z)
        self.x = np.ones((self.nph, self.nc))

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

        self.ph = [0]

        for j in self.ph:
            M = 0
            # molar weight of mixture
            for i in range(self.nc):
                M += self.Mw[i] * self.x[j][i]
            self.dens[j] = self.density_ev[self.phases_name[j]].evaluate(pressure, 0)  # output in [kg/m3]
            self.dens_m[j] = self.dens[j] / M
            self.mu[j] = self.viscosity_ev[self.phases_name[j]].evaluate()  # output in [cp]

        self.nu[0] = 1
        self.compute_saturation(self.ph)

        for j in self.ph:
            self.kr[j] = self.rel_perm_ev[self.phases_name[j]].evaluate(self.sat[j])
            self.pc[j] = 0

        kin_rates = np.zeros(1)

        return self.sat, self.x, self.dens, self.dens_m, self.mu, kin_rates, self.kr, self.pc, self.ph

    def evaluate_at_cond(self, pressure, zc):

        self.sat[:] = 0

        ph = [0]
        for j in ph:
            self.dens_m[j] = self.density_ev[self.phases_name[j]].evaluate(1, 0)

        self.dens_m = [1025, 0.77]  # to match DO based on PVT

        self.nu[0] = 1
        self.compute_saturation(ph)


        return self.sat, self.dens_m
