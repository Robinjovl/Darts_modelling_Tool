from darts.models.reservoirs.struct_reservoir import StructReservoir
from darts.models.cicd_model import CICDModel
from darts.engines import value_vector
import numpy as np

from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer

from darts.physics.properties.basic import ConstFunc, PhaseRelPerm
from darts.physics.properties.density import DensityBasic
from darts.physics.properties.enthalpy import EnthalpyBasic


class Model(CICDModel):
    def __init__(self):
        # call base class constructor
        super().__init__()

        # measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        self.set_reservoir()
        self.set_physics()
        self.set_wells()

        self.set_sim_params(first_ts=0.0001, mult_ts=2, max_ts=5, runtime=1000, tol_newton=1e-3, tol_linear=1e-6)

        self.timer.node["initialization"].stop()

    def set_reservoir(self):
        """Reservoir construction"""
        # reservoir geometry： for realistic case, one just needs to load the data and input it
        self.reservoir = StructReservoir(self.timer, nx=500, ny=1, nz=1, dx=10.0, dy=10.0, dz=1, permx=300, permy=300,
                                         permz=300, poro=0.2, depth=100)

        hcap = np.array(self.reservoir.mesh.heat_capacity, copy=False)
        rcond = np.array(self.reservoir.mesh.rock_cond, copy=False)

        hcap.fill(2200)
        rcond.fill(181.44)
        return

    def set_wells(self):
        # well model or boundary conditions
        self.reservoir.add_well("I1")
        self.reservoir.add_perforation(well=self.reservoir.wells[-1], i=1, j=1, k=1, multi_segment=False)

        self.reservoir.add_well("P1")
        self.reservoir.add_perforation(self.reservoir.wells[-1], 500, 1, 1, multi_segment=False)
        return

    def set_physics(self):
        """Physical properties"""
        zero = 1e-13
        components = ['w']
        phases = ['wat', 'gas']

        self.inj = value_vector([300])

        property_container = ModelProperties(phases_name=phases, components_name=components, min_z=zero/10)

        # Define property evaluators based on custom properties
        property_container.density_ev = dict([('wat', DensityBasic(compr=1e-5, dens0=1014)),
                                              ('gas', DensityBasic(compr=5e-3, dens0=50))])
        property_container.viscosity_ev = dict([('wat', ConstFunc(0.3)),
                                                ('gas', ConstFunc(0.03))])
        property_container.rel_perm_ev = dict([('wat', PhaseRelPerm("oil", 0.1, 0.1)),
                                               ('gas', PhaseRelPerm("gas", 0.1, 0.1))])
        property_container.enthalpy_ev = dict([('wat', EnthalpyBasic(hcap=4.18)),
                                               ('gas', EnthalpyBasic(hcap=0.035))])
        property_container.conductivity_ev = dict([('wat', ConstFunc(1.)),
                                                   ('gas', ConstFunc(1.))])
        property_container.rock_energy_ev = EnthalpyBasic(hcap=1.0)

        # create physics
        thermal = True
        self.physics = Compositional(components, phases, self.timer,
                                     n_points=400, min_p=0, max_p=1000, min_z=zero, max_z=1-zero,
                                     min_t=273.15 + 20, max_t=273.15 + 200, thermal=thermal)
        self.physics.add_property_region(property_container)
        self.physics.init_physics()

        return

    def set_initial_conditions(self):
        self.physics.set_uniform_initial_conditions(self.reservoir.mesh, uniform_pressure=200,
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


class ModelProperties(PropertyContainer):
    def __init__(self, phases_name, components_name, min_z=1e-11):
        # Call base class constructor
        self.nph = len(phases_name)
        Mw = np.ones(self.nph)
        super().__init__(phases_name, components_name, Mw, min_z, temperature=None)
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

        mass_source = np.zeros(1)

        return self.ph, self.sat, self.x, self.dens, self.dens_m, self.mu, self.kr, self.pc, mass_source

    def evaluate_at_cond(self, pressure, zc):

        self.sat[:] = 0

        ph = [0]
        for j in ph:
            self.dens_m[j] = self.density_ev[self.phases_name[j]].evaluate(1, 0)

        self.dens_m = [1025, 0.77]  # to match DO based on PVT

        self.nu[0] = 1
        self.compute_saturation(ph)

        return self.sat, self.dens_m
