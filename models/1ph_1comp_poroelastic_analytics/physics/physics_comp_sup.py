import numpy as np
from darts.engines import *
from darts.models.physics.physics_base import PhysicsBase
from physics.operator_evaluator_sup import *

# Define our own operator evaluator class
class Poroelasticity(PhysicsBase):
    class operators_storage:
        def __init__(self):
            self.reservoir_operators = {}
            self.wellbore_operators = {}
            self.rate_operators = {}
            self.property_operators = {}

    def __init__(self, property_container, timer, n_points, min_p, max_p,
                 platform='cpu', itor_type='multilinear', itor_mode='adaptive', itor_precision='d', cache=False):
        super().__init__(cache)
        # Obtain properties from user input during initialization:
        self.n_dim = 3
        self.timer = timer.node["simulation"]
        self.components = property_container.components_name
        self.nc = property_container.nc
        self.phases = property_container.phases_name
        self.nph = property_container.nph
        self.n_vars = self.nc + self.n_dim
        NE = self.n_vars - self.n_dim
        comps = ['z' + str(i) for i in range(self.nc - 1)]

        self.n_ops = 2 * NE
        self.n_axes_points = index_vector([n_points] * (self.n_vars - self.n_dim))

        operators = self.set_operators(property_container)

        """ Name of interpolation method and engine used for this physics: """
        assert(self.nc == self.nph == 1)
        self.engine = eval("engine_pm_%s" % (platform))()
        self.vars = ['p'] + comps + ['ux', 'uy', 'uz']
        self.n_axes_min = value_vector([min_p])
        self.n_axes_max = value_vector([max_p])

        self.acc_flux_itor = {}
        for i in operators.reservoir_operators.keys():
            self.acc_flux_itor[i] = self.create_interpolator(operators.reservoir_operators[i], self.n_vars - self.n_dim,
                                                             self.n_ops, self.n_axes_points,
                                                             self.n_axes_min, self.n_axes_max, platform=platform)
            self.create_itor_timers(self.acc_flux_itor[i], 'reservoir %d interpolation' % i)

        self.acc_flux_w_itor = self.create_interpolator(operators.wellbore_operators, self.n_vars - self.n_dim, self.n_ops,
                                                        self.n_axes_points,
                                                        self.n_axes_min, self.n_axes_max, platform=platform)

        self.property_itor = self.create_interpolator(operators.property_operators, self.n_vars - self.n_dim, self.n_ops,
                                                      self.n_axes_points, self.n_axes_min, self.n_axes_max,
                                                      platform=platform)


        self.rate_itor = self.create_interpolator(operators.rate_operators, self.n_vars - self.n_dim, self.nph, self.n_axes_points,
                                              self.n_axes_min, self.n_axes_max, platform=platform)

        self.create_itor_timers(self.acc_flux_w_itor, 'wellbore interpolation')
        self.create_itor_timers(self.rate_itor, 'well controls interpolation')

        # create well controls
        self.new_bhp_prod = lambda bhp: bhp_prod_well_control(bhp)
        self.new_bhp_inj = lambda bhp: bhp_inj_well_control(bhp, value_vector([]))
        # water stream
        # min_z is the minimum composition for interpolation
        # 2*min_z is the minimum composition for simulation
        # let`s take 3*min_z as the minimum composition for injection to be safely within both limits

        self.water_inj_stream = value_vector([1])
        # self.new_bhp_water_inj = lambda bhp: bhp_inj_well_control(bhp, self.water_inj_stream)
        self.new_rate_inj = lambda rate: rate_inj_well_control(self.rate_phases, 0, self.n_components,
                                                               self.n_components,
                                                               rate,
                                                               self.water_inj_stream, self.rate_itor)
        # self.new_bhp_prod = lambda bhp: bhp_prod_well_control(bhp)
        # self.new_rate_water_prod = lambda rate: rate_prod_well_control(self.rate_phases, 0, self.n_components,
        #                                                                self.n_components,
        #                                                                rate, self.rate_itor)
        #
        self.new_rate_prod = lambda rate: rate_prod_well_control(self.rate_phases, 0, self.n_components,
                                                                 self.n_components,
                                                                 rate, self.rate_itor)
        # self.new_rate_liq_prod = lambda rate: rate_prod_well_control(self.rate_phases, 2, self.n_components,
        #                                                                self.n_components,
        #                                                                rate, self.rate_itor)
        # self.new_acc_flux_itor = lambda new_acc_flux_etor: acc_flux_itor_name(new_acc_flux_etor,
        #                                                                       index_vector([n_points, n_points]),
        #                                                                       value_vector([min_p, min_z]),
        #                                                                       value_vector([max_p, 1 - min_z]))

    def init_wells(self, wells):
        """""
        Function to initialize the well rates for each well
        Arguments:
            -wells: well_object array
        """
        for w in wells:
            assert isinstance(w, ms_well)
            # w.init_rate_parameters(self.n_components, self.rate_phases, self.rate_itor)
            w.init_mech_rate_parameters(self.engine.N_VARS, self.engine.P_VAR, self.n_components, self.phases,
                                        self.rate_itor)

    def set_uniform_initial_conditions(self, mesh, uniform_pressure, uniform_displacement: list):
        assert isinstance(mesh, conn_mesh)
        nb = mesh.n_blocks

        # set initial pressure
        pressure = np.array(mesh.pressure, copy=False)
        pressure.fill(uniform_pressure)
        # set initial displacements
        displacement = np.array(mesh.displacement, copy=False)
        for i in range(self.n_dim):
            displacement[i::self.n_dim] = uniform_displacement[i]

    def set_nonuniform_initial_conditions(self, mesh, initial_pressure, initial_displacement: list):
        assert isinstance(mesh, conn_mesh)
        nb = mesh.n_blocks
        n_res_blocks = mesh.n_res_blocks

        # set initial pressure
        pressure = np.array(mesh.pressure, copy=False)
        pressure[:n_res_blocks] = initial_pressure
        # set initial displacements
        displacement = np.array(mesh.displacement, copy=False)
        for i in range(self.n_dim):
            displacement[i:self.n_dim * n_res_blocks:self.n_dim] = initial_displacement[i]

    def set_operators(self, property_container):  # default definition of operators

        thermal = False
        operators = self.operators_storage()

        if thermal:
            operators.reservoir_operators[0] = ReservoirThermalOperators(property_container)
            operators.wellbore_operators = ReservoirThermalOperators(property_container)
        else:
            operators.reservoir_operators[0] = ReservoirOperators(property_container)
            operators.wellbore_operators = WellOperators(property_container)

        operators.rate_operators = RateOperators(property_container)

        operators.property_operators = DefaultPropertyEvaluator(property_container)

        return operators
