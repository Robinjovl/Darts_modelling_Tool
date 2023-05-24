import numpy as np
from darts.models.reservoirs.struct_reservoir import StructReservoir
from darts.models.darts_model import DartsModel
from darts.engines import value_vector, sim_params

from darts.models.physics_sup.properties_basic import PhaseRelPerm, CapillaryPressure, RockEnergyEvaluator, ConstFunc
from darts.models.physics_sup.property_container import PropertyContainer

from darts.models.physics_sup.physics_comp_sup import Compositional
from darts.models.physics_sup.operator_evaluator_sup import DefaultPropertyEvaluator

from darts.models.physics_sup.flash.flash import NF2
from dartsflash import PR, Ziabakhsh2012, FlashParams
from darts.models.physics_sup.flash.components import ComponentProperties
from darts.models.physics_sup.flash.properties import DensityBrineCO2, ViscosityCO2, ViscosityAq, EnthalpyIdeal
from darts.models.physics_sup.flash.eos_properties import EoSDensity, EoSEnthalpy


class Model(DartsModel):
    def __init__(self, n_points=1000, temp_init=350, temp_inj=350):
        # call base class constructor
        super().__init__()
        self.n_points = n_points
        # measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        self.set_reservoir()
        self.set_physics(n_points, temp_ini=temp_init)
        self.set_wells(p_inj=60, t_inj=temp_inj, p_prod=30)

        self.params.first_ts = 1e-3
        self.params.mult_ts = 1.5
        self.params.max_ts = 5

        # Newton tolerance is relatively high because of L2-norm for residual and well segments
        self.params.tolerance_newton = 1e-3
        self.params.tolerance_linear = 1e-5
        self.params.max_i_newton = 10
        self.params.max_i_linear = 50

        self.params.newton_type = sim_params.newton_global_chop
        self.params.newton_params = value_vector([1])

        self.runtime = 1000
        self.timer.node["initialization"].stop()

    def set_reservoir(self):
        self.nx = 100
        self.ny = 1
        self.nz = 40
        nb = self.nx * self.ny * self.nz
        dz = 2
        self.depth = np.zeros(nb)
        n_layer = self.nx*self.ny
        for k in range(self.nz):
            self.depth[k*n_layer:(k+1)*n_layer] = 1000 + k * dz

        self.x_axes = np.logspace(-0.3, 2, self.nx)
        dx = np.tile(self.x_axes, self.nz)

        self.reservoir = StructReservoir(self.timer, nx=self.nx, ny=self.ny, nz=self.nz, dx=dx, dy=10, dz=dz,
                                         permx=100, permy=100, permz=10, poro=0.2, depth=self.depth)

        # get wrapper around local array (length equal to active blocks number)
        hcap = np.array(self.reservoir.mesh.heat_capacity, copy=False)
        rcond = np.array(self.reservoir.mesh.rock_cond, copy=False)

        hcap.fill(2200)  # 2200 kJ/m3.K for sandstone
        rcond.fill(100)

    def set_wells(self, p_inj=60, t_inj=None, p_prod=30):
        """Create well objects"""
        self.reservoir.add_well('I1')
        for n in range(self.nz-1, self.nz):
            self.reservoir.add_perforation(well=self.reservoir.wells[-1], i=1, j=1, k=n+1,
                                           well_index=100, multi_segment=False, verbose=False)
        self.reservoir.add_well('P1')
        for n in range(self.nz):
            self.reservoir.add_perforation(well=self.reservoir.wells[-1], i=self.nx, j=self.ny, k=n+1,
                                           well_index=100, multi_segment=False, verbose=False)

        """Set operational conditions"""
        self.p_inj = p_inj
        self.p_prod = p_prod

        self.inj_stream = [0.99995]
        if self.thermal:
            self.inj_stream.append(t_inj)

    def set_physics(self, n_points, temperature=None, temp_ini=350.):
        self.zero = 1e-7

        """Physical properties"""
        # Fluid components, ions and solid
        self.components = ["CO2", "H2O"]
        self.phases = ["Aq", "V"]
        nc = len(self.components)
        comp_data = ComponentProperties.comp_data(ComponentProperties(), self.components)
        pr = PR(self.components, comp_data)
        # aq = Jager2003(self.components, [])
        aq = Ziabakhsh2012(self.components, [])

        flash_params = FlashParams(nc)

        # EoS-related parameters
        flash_params.add_eos("PR", pr)
        flash_params.add_eos("AQ", aq)
        flash_params.eos_used = ["AQ", "PR"]
        flash_params.add_initial_guess(self.components, comp_data)
        flash_params.flash_initial_guess = [3]

        # Flash-related parameters
        # flash_params.split_switch_tol = 1e-3

        self.ini_stream = [0.00005]
        self.p_init = 60.

        self.temperature = temperature  # if None, then thermal=True
        if self.temperature is None:
            self.thermal = True
            self.init_temp = temp_ini
        else:
            self.thermal = False
            self.init_temp = temperature

        """ properties correlations """
        self.property_container = PropertyContainer(phases_name=self.phases, components_name=self.components,
                                                    Mw=comp_data["Mw"], temperature=self.temperature, min_z=self.zero/10)

        self.property_container.flash_ev = NF2(nc, flash_params)
        self.property_container.density_ev = dict([('V', EoSDensity(pr, comp_data["Mw"])),
                                                   ('Aq', DensityBrineCO2(self.components))])
        self.property_container.viscosity_ev = dict([('V', ViscosityCO2()),
                                                     ('Aq', ViscosityAq(self.components))])
        self.property_container.rel_perm_ev = dict([('V', PhaseRelPerm("gas")),
                                                    ('Aq', PhaseRelPerm("oil"))])

        h_ideal = EnthalpyIdeal(self.components)
        self.property_container.enthalpy_ev = dict([('V', EoSEnthalpy(pr, h_ideal)),
                                                    ('Aq', EoSEnthalpy(aq, h_ideal))])
        self.property_container.conductivity_ev = dict([('V', ConstFunc(0.)),
                                                        ('Aq', ConstFunc(0.)), ])

        self.property_container.capillary_pressure_ev = CapillaryPressure()
        self.property_container.rock_energy_ev = RockEnergyEvaluator()

        self.nc = self.ne = len(self.components)
        self.vars = ['P'] + self.components[:-1]
        if self.thermal:
            self.ne += 1
            self.vars += ['T']

        """ Activate physics """
        # self.output_props = PropertyEvaluator(self.vars, self.property_container)
        self.physics = Compositional(self.property_container, self.components, self.phases, self.timer, n_points=n_points, min_p=1, max_p=400,
                                     min_z=self.zero/10, max_z=1-self.zero/10, min_t=273.15, max_t=373.15,
                                     thermal=self.thermal, cache=0)

    def set_initial_conditions(self):
        if self.thermal:
            self.physics.set_uniform_T_initial_conditions(self.reservoir.mesh, self.p_init, self.ini_stream, self.init_temp)
        else:
            self.physics.set_uniform_initial_conditions(self.reservoir.mesh, self.p_init, self.ini_stream)

        self.timer.node["initialization"].stop()

    def set_boundary_conditions(self):
        # define all wells as closed
        for i, w in enumerate(self.reservoir.wells):
            if 'I' in w.name:
                w.control = self.physics.new_bhp_inj(self.p_inj, self.inj_stream)
            else:
                w.control = self.physics.new_bhp_prod(self.p_prod)

    def output_properties(self):
        n_vars = self.physics.property_operators.n_vars
        n_props = self.physics.property_operators.n_props
        tot_props = n_vars + n_props

        property_array = np.zeros((self.reservoir.nb, tot_props))
        for j in range(n_vars):
            property_array[:, j] = self.physics.engine.X[j:self.reservoir.nb * n_vars:n_vars]

        values = value_vector(np.zeros(n_props))

        for i in range(self.reservoir.nb):
            state = []
            for j in range(n_vars):
                state.append(property_array[i, j])
            state = value_vector(np.asarray(state))
            self.physics.property_itor.evaluate(state, values)

            for j in range(n_props):
                property_array[i, j + n_vars] = values[j]

        return property_array


class PropertyEvaluator(DefaultPropertyEvaluator):
    def __init__(self, variables, property_container):
        super().__init__(variables, property_container)  # Initialize base-class

        self.props = ['sat0', 'xCO2']
        self.n_props = len(self.props)

    def evaluate(self, state, values):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        (self.sat, self.x, rho, self.rho_m, self.mu, kin_rates, self.kr, self.pc, self.ph) = self.property.evaluate(state)

        values[0] = self.sat[0]
        values[1] = self.x[0, 0]

        return 0
