from darts.reservoirs.struct_reservoir import StructReservoir
from darts.models.cicd_model import DartsModel
import numpy as np

from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer

from darts.physics.properties.basic import ConstFunc
from darts.physics.properties.density import DensityBasic
from darts.physics.properties.enthalpy import EnthalpyBasic

# Simplified water viscosity correlation as function of T
class WaterVisc():  # not checked, taking from ChatGPT
    def __init__(self):
        self.A = 2.414e-2  # kPa·s
        self.B = 247.8
        self.C = 140.0

    def evaluate(self, temperature):
        visc = self.A * 10 ** (self.B / (temperature - self.C))
        return visc

# describe model class
class Model(DartsModel):
    def __init__(self):
        # call base class constructor
        super().__init__()

        # measure time spend on model initialization
        self.timer.node["initialization"].start()

        """Reservoir definition """
        nx = 1000
        perm = 100
        self.reservoir = StructReservoir(self.timer, nx=nx, ny=1, nz=1, dx=600 / nx, dy=1.0, dz=1, poro=0.2,
                                         permx=perm, permy=perm, permz=perm, hcap=2000, rcond=100, depth=1000)

        """Physical properties"""
        self.zero = 1e-12
        epsilon = 1e-9
        components = ['brine', 'gas']
        phases = ['wat']

        property_container = ModelProperties(phases_name=phases, components_name=components, eps_z=epsilon)

        # Define property evaluators based on custom properties
        property_container.density_ev = dict([('wat', DensityBasic(compr=1e-5, dens0=1014))])
        property_container.viscosity_ev = dict([('wat', WaterVisc())])
        property_container.enthalpy_ev = dict([('wat', EnthalpyBasic(hcap=4.18))])
        property_container.conductivity_ev = dict([('wat', ConstFunc(75))])

        property_container.rock_energy_ev = EnthalpyBasic(hcap=1.0)

        """Create physics"""
        thermal = True
        state_spec = Compositional.StateSpecification.PT if thermal else Compositional.StateSpecification.P
        self.physics = Compositional(components, phases, self.timer, state_spec=state_spec,
                                     n_points=400, min_p=0, max_p=1000, min_z=0.0, max_z=1.0, epsilon_z=epsilon,
                                     min_t=273.15 + 20, max_t=273.15 + 200)
        self.physics.add_property_region(property_container)
        self.set_sim_params(first_ts=1e-4, mult_ts=2, max_ts=1)

        # end of initialization
        self.timer.node["initialization"].stop()


    def set_wells(self):
        self.reservoir.add_well("I1")
        self.reservoir.add_perforation("I1", res_cell_idx=(1, 1, 1))
        self.reservoir.add_well("P1")
        self.reservoir.add_perforation("P1", res_cell_idx=(self.reservoir.nx, 1, 1))

    def set_initial_conditions(self):
        input_distribution = {self.physics.vars[0]: 200.,
                              self.physics.vars[1]: 0.99,
                              self.physics.vars[2]: 350.
                              }
        return self.physics.set_initial_conditions_from_array(mesh=self.reservoir.mesh,
                                                              input_distribution=input_distribution)

    def set_well_controls(self):
        from darts.engines import well_control_iface
        for i, w in enumerate(self.reservoir.wells):
            if i == 0:
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.MASS_RATE,
                                               is_inj=True, target=100., phase_name='wat',
                                               inj_composition=[1 - self.zero], inj_temp=300)
            else:
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.BHP,
                                               is_inj=False, target=180.)

# Simplified property evaluation for single-phase model`
class ModelProperties(PropertyContainer):
    def __init__(self, phases_name, components_name, eps_z):
        # Call base class constructor
        nc = len(components_name)
        Mw = np.ones(nc)
        super().__init__(phases_name, components_name, Mw, eps_z=eps_z, temperature=None)

    def evaluate(self, state):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        # Composition vector and pressure from state:
        vec_state_as_np = np.asarray(state)
        self.pressure = vec_state_as_np[0]
        self.temperature = vec_state_as_np[-1] if self.thermal else self.temperature

        zc = np.append(vec_state_as_np[1:self.nc], 1 - np.sum(vec_state_as_np[1:self.nc]))

        self.clean_arrays()
        j = 0
        # two-phase flash - assume water phase is always present and water component last
        self.x[j, :] = zc

        self.ph = np.array([j], dtype=np.intp)

        # molar weight of mixture
        M = np.sum(self.x[j, :] * self.Mw)
        self.dens[j] = self.density_ev[self.phases_name[j]].evaluate(self.pressure)  # output in [kg/m3]
        self.dens_m[j] = self.dens[j] / M
        self.mu[j] = self.viscosity_ev[self.phases_name[j]].evaluate(self.temperature)  # output in [cp]

        self.sat[j] = 1
        self.kr[j] = 1
        self.pc[j] = 0

        return
