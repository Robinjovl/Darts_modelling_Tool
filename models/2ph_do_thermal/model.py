from darts.reservoirs.struct_reservoir import StructReservoir
from darts.models.darts_model import DartsModel
from darts.engines import value_vector, ms_well
from darts.nonlinear_solvers import NewtonSolver
import numpy as np

from darts.physics.base.physics import PhysicsBase
from darts.physics.base.property_container import PropertyContainer

from darts.physics.properties.basic import ConstFunc, PhaseRelPerm
from darts.physics.properties.density import DensityBasic
from darts.physics.properties.enthalpy import EnthalpyBasic


class Model(DartsModel):
    def __init__(self):
        # call base class constructor
        super().__init__()

        # measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        self.set_reservoir()
        self.set_physics()

        # solver configuration moved to set_solver() (called from base reset())

        self.timer.node["initialization"].stop()

    def set_solver(self):
        if self.linear_solver is None:
            from darts.linear_solvers import LinearSolver
            self.linear_solver = LinearSolver(model=self)
        self.linear_solver.set_sim_params(first_ts=0.0001, mult_ts=2, max_ts=2, runtime=1000 )
        super().set_solver()  # platform default nonlinear + linear solvers
        self.nonlinear_solver = NewtonSolver(tolerance=1e-3)
        self.linear_solver.spec.tolerance = 1e-6

    def set_reservoir(self):
        nx = 1000
        self.reservoir = StructReservoir(self.timer, nx=nx, ny=1, nz=1, dx=10.0, dy=10.0, dz=1,
                                         permx=300, permy=300, permz=300, hcap=2200, rcond=181.44, poro=0.2, depth=100)
        return

    def set_wells(self):
        self.reservoir.add_well("I1")
        self.reservoir.add_perforation("I1", res_cell_idx=(1, 1, 1))
        self.reservoir.add_well("P1")
        self.reservoir.add_perforation("P1", res_cell_idx=(self.reservoir.nx, 1, 1))

    def set_physics(self):
        """Physical properties"""
        zero = 1e-13
        epsilon = 1e-14
        components = ['w', 'o']
        phases = ['wat', 'oil']

        self.inj = value_vector([1 - zero, 300])
        self.ini = value_vector([zero])

        property_container = ModelProperties(phases_name=phases, components_name=components, eps_z=epsilon)

        # Define property evaluators based on custom properties
        property_container.density_ev = dict([('wat', DensityBasic(compr=1e-5, dens0=1014)),
                                              ('oil', DensityBasic(compr=5e-3, dens0=50))])
        property_container.viscosity_ev = dict([('wat', ConstFunc(0.3)),
                                                ('oil', ConstFunc(0.03))])
        property_container.rel_perm_ev = dict([('wat', PhaseRelPerm("gas", 0.1, 0.1)),
                                               ('oil', PhaseRelPerm("oil", 0.1, 0.1))])
        property_container.enthalpy_ev = dict([('wat', EnthalpyBasic(hcap=4.18)),
                                               ('oil', EnthalpyBasic(hcap=0.035))])
        property_container.conductivity_ev = dict([('wat', ConstFunc(1.)),
                                                   ('oil', ConstFunc(1.))])

        property_container.rock_energy_ev = EnthalpyBasic(hcap=1.0)

        # create physics
        thermal = True
        state_spec = PhysicsBase.StateSpecification.PT if thermal else PhysicsBase.StateSpecification.P
        self.physics = PhysicsBase(components, phases, self.timer, state_spec=state_spec,
                                     axes_step=[2.5, 2.5e-3, 0.5],  # p [bar], z, T [K]
                                     axes_origin=[0.0, epsilon, 273.15],
                                     epsilon_z=epsilon, extrapolation_flag=True)
        self.physics.add_property_region(property_container)

        return

    def set_initial_conditions(self):
        input_distribution = {self.physics.vars[0]: 200.,
                              self.physics.vars[1]: self.ini[0],
                              self.physics.vars[2]: 350.
                              }
        return self.physics.set_initial_conditions_from_array(mesh=self.reservoir.mesh,
                                                              input_distribution=input_distribution)

    def set_well_controls(self):
        from darts.engines import well_control_iface
        for i, w in enumerate(self.reservoir.wells):
            if i == 0:
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.MOLAR_RATE,
                                               is_inj=True, target=5000., phase_name='wat', inj_composition=self.inj[:-1],
                                               inj_temp=self.inj[-1])
            else:
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.BHP,
                                               is_inj=False, target=180.)


class ModelProperties(PropertyContainer):
    def __init__(self, phases_name, components_name, eps_z=1e-11):
        # Call base class constructor
        self.nph = len(phases_name)
        Mw = np.ones(self.nph)
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
        # two-phase flash - assume water phase is always present and water component last
        for i in range(self.nph):
            self.x[i, i] = 1

        self.ph = np.array([0, 1], dtype=np.intp)

        for j in self.ph:
            # molar weight of mixture
            M = np.sum(self.x[j, :] * self.Mw)
            self.dens[j] = self.density_ev[self.phases_name[j]].evaluate(self.pressure)  # output in [kg/m3]
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
