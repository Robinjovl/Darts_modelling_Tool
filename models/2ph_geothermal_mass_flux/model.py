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
    def __init__(self, mode='rhs', well_rate=1, outflow=1000):
        # call base class constructor
        super().__init__()

        # measure time spend on reading/initialization
        self.timer.node["initialization"].start()
        self.mode = mode
        self.well_rate = well_rate
        self.set_reservoir()
        self.wells_mode = mode
        self.set_physics()

        # solver/time-stepping configuration moved to set_solver() (called by base reset())

        # add outflux to the middle cell
        self.inflow_cells = np.array([self.reservoir.nx // 2])
        self.inflow_var_idx = 0
        self.outflow = outflow if mode == 'rhs' else 0

        self.timer.node["initialization"].stop()

    def set_solver(self):
        self.ts_control.dt_first = 0.0001
        self.ts_control.dt_min = 1e-15
        self.ts_control.dt_mult = 2
        self.ts_control.dt_max = 5
        self.ts_control.runtime = 1
        super().set_solver()  # platform default nonlinear + linear solvers
        self.nonlinear_solver = NewtonSolver(tolerance=1e-3)
        self.linear_solver.spec.tolerance = 1e-6

    def set_reservoir(self):
        """Reservoir construction"""
        # reservoir geometry： for realistic case, one just needs to load the data and input it
        nx = 100
        self.reservoir = StructReservoir(self.timer, nx=nx, ny=1, nz=1, dx=10.0, dy=10.0, dz=1,
                                         permx=5, permy=5, permz=5, poro=0.2, hcap=2200, rcond=181.44, depth=100)

        return

    def set_wells(self):
        if self.wells_mode == 'wells':
            self.reservoir.add_well("P1")
            self.reservoir.add_perforation("P1", res_cell_idx=(self.reservoir.nx // 2, 1, 1), ms_epm=False)

    def set_physics(self):
        """Physical properties"""
        zero = 1e-13
        epsilon = 1e-14
        components = ['w']
        phases = ['wat', 'gas']

        self.inj = value_vector([300])

        property_container = ModelProperties(phases_name=phases, components_name=components, eps_z=epsilon)

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

        # create physics
        thermal = True
        state_spec = PhysicsBase.StateSpecification.PT if thermal else PhysicsBase.StateSpecification.P
        self.physics = PhysicsBase(components, phases, self.timer, state_spec=state_spec,
                                     axes_step=[2.5, 0.4511],  # p [bar], T [K]
                                     axes_origin=[0.0, 273.15 + 20],
                                     epsilon_z=epsilon, extrapolation_flag=True)
        self.physics.add_property_region(property_container)

        return

    def set_initial_conditions(self):
        input_distribution = {self.physics.vars[0]: 200.,
                              self.physics.vars[1]: 350.,
                              }
        return self.physics.set_initial_conditions_from_array(mesh=self.reservoir.mesh,
                                                              input_distribution=input_distribution)

    def set_well_controls(self):
        from darts.engines import well_control_iface
        for i, w in enumerate(self.reservoir.wells):
            if 'I' in w.name:
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.MOLAR_RATE,
                                               is_inj=True, target=self.well_rate, phase_name='wat',
                                               inj_composition=self.inj[:-1], inj_temp=self.inj[-1])
            else:
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.MOLAR_RATE,
                                               is_inj=False, target=-self.well_rate, phase_name='wat')


    def set_rhs_flux(self, t: float = None):
        '''
        function to specify the inflow or outflow to the cells
        it sets up self.rhs_flux vector on nvar * ncells size
        which will be added to rhs in darts_model.run_python function
        :param inflow_cells: cell indices where to apply inflow or outflow
        :param inflow_var_idx: variable index [0..nvars-1]
        :param outflow: inflow_var_idx<nc => kMol/day, else kJ/day (thermal var)
        if outflow < 0 then it is actually inflow
        '''
        nv = self.physics.n_vars
        nb = self.reservoir.mesh.n_blocks
        n_res_blocks = self.reservoir.mesh.n_res_blocks
        rhs_flux = np.zeros(nb * nv)

        # extract pointer to values corresponding to var_idx
        rhs_flux_var = rhs_flux[self.inflow_var_idx:n_res_blocks*nv:nv]
        # set values for the cells defined in inflow_cells
        rhs_flux_var[self.inflow_cells] = self.outflow

        return rhs_flux


class ModelProperties(PropertyContainer):
    def __init__(self, phases_name, components_name, eps_z=1e-11):
        # Call base class constructor
        self.nph = len(phases_name)
        Mw = np.ones(self.nph)
        super().__init__(phases_name, components_name, Mw, eps_z=eps_z, temperature=None)
        self.x = np.ones((self.nph, self.nc))

    def evaluate_flash(self, state):
        """
        Compute the (trivial, single-phase) flash: pressure, temperature, phase state.

        :param state: state variables [pres, comp_0, ..., comp_N-1]
        """
        # Composition vector and pressure from state:
        vec_state_as_np = np.asarray(state)
        self.pressure = vec_state_as_np[0]
        self.temperature = vec_state_as_np[-1] if self.thermal else self.temperature

        self.ph = np.array([0], dtype=np.intp)
        self.nu[0] = 1
        self.x[0, 0] = 1.

    def evaluate_properties(self, state):
        """
        Compute derived phase properties (density, viscosity, saturation, relperm)
        from the flash results currently held by this container.

        :param state: state variables [pres, comp_0, ..., comp_N-1]
        """
        for j in self.ph:
            M = 0
            # molar weight of mixture
            for i in range(self.nc):
                M += self.Mw[i] * self.x[j][i]
            self.dens[j] = self.density_ev[self.phases_name[j]].evaluate(self.pressure, 0)  # output in [kg/m3]
            self.dens_m[j] = self.dens[j] / M
            self.mu[j] = self.viscosity_ev[self.phases_name[j]].evaluate()  # output in [cp]

        self.compute_saturation(self.ph)

        for j in self.ph:
            self.kr[j] = self.rel_perm_ev[self.phases_name[j]].evaluate(self.sat[j])
            self.pc[j] = 0

        return
