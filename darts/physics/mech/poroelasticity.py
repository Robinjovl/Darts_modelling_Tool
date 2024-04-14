from darts.engines import *
from darts.physics.physics_base import PhysicsBase
from darts.physics.super.operator_evaluator import *
import numpy as np

class Poroelasticity(PhysicsBase):
    """
    This is the Physics class for compositional poroelastic simulation.

    It includes:
    - Creating Reservoir, Well, Rate and Property operators and interpolators for P-z or P-T-z compositional simulation
    - Initializing the :class:`super_engine`
    - Setting well controls (rate, bhp)
    - Defining initial and boundary conditions
    """
    def __init__(self, components: list, phases: list, timer: timer_node, n_points: int,
                 min_p: float, max_p: float, min_z: float, max_z: float, min_t: float = None, max_t: float = None,
                 thermal: bool = False, cache: bool = False, discretizer: str = 'mech_discretizer'):
        """
        This is the constructor of the Compositional Physics class.

        It defines the OBL grid for P-z or P-T-z compositional simulation.

        :param components: List of components
        :type components: list
        :param phases: List of phases
        :type phases: list
        :param timer: Timer object
        :type timer: :class:`darts.engines.timer_node`
        :param n_points: Number of OBL points along axes
        :type n_points: int
        :param min_p, max_p: Minimum, maximum pressure
        :type min_p, max_p: float
        :param min_z, max_z: Minimum, maximum composition
        :type min_z, max_z: float
        :param min_t, max_t: Minimum, maximum temperature, default is None
        :type min_t, max_t: float
        :param thermal: Switch for (iso)thermal simulation
        :type thermal: bool
        :param cache: Switch to cache operator values
        :type cache: bool
        :param discretizer: Name of discretizer
        :type discretizer: str
        """
        # Define nc, nph and (iso)thermal
        nc = len(components)
        nph = len(phases)
        self.thermal = thermal
        self.n_dim = 3
        self.discretizer_name = discretizer

        # Define STATE(!) variables and OBL axes: pressure, nc-1 components and possibly temperature
        variables = ['pressure'] + components[:-1]
        if self.thermal:
            variables += ['temperature']
            axes_min = value_vector([min_p] + [min_z] * (nc - 1) + [min_t])
            axes_max = value_vector([max_p] + [max_z] * (nc - 1) + [max_t])
        else:
            axes_min = value_vector([min_p] + [min_z] * (nc - 1))
            axes_max = value_vector([max_p] + [max_z] * (nc - 1))

        n_vars = len(variables)

        if self.discretizer_name == 'mech_discretizer':
            n_ops = n_vars + nph * n_vars + nph + nph * n_vars + n_vars + 3 + 2 * nph + 1 + 1
        elif self.discretizer_name == 'pm_discretizer':
            n_ops = 2 * n_vars
            assert(self.thermal == False)

        # Call PhysicsBase constructor
        super().__init__(variables=variables, nc=nc, phases=phases, n_ops=n_ops,
                         axes_min=axes_min, axes_max=axes_max, n_points=n_points, timer=timer, cache=cache)

    def init_physics(self, regions: list = None, discr_type: str = 'tpfa', platform: str = 'cpu',
                     itor_type: str = 'multilinear', itor_mode: str = 'adaptive', itor_precision: str = 'd',
                     verbose: bool = False, discretizer: str = 'mech_discretizer'):
        """
        Function to initialize all contained objects within the Physics object.

        :param regions: List of regions. It contains the keys of the `property_containers` and `reservoir_operators` dict
        :type regions: list
        :param discr_type: Discretization type, 'tpfa' (default) or 'mpfa'
        :type discr_type: str
        :param platform: Switch for CPU/GPU engine, 'cpu' (default) or 'gpu'
        :type platform: str
        :param itor_type: Type of interpolation method, 'multilinear' (default) or 'linear'
        :type itor_type: str
        :param itor_mode: Mode of interpolation, 'adaptive' (default) or 'static'
        :type itor_mode: str
        :param itor_precision: Precision of interpolation, 'd' (default) - double precision or 's' - single precision
        :type itor_precision: str
        :param verbose: Set verbose level
        :type verbose: bool
        :param discretizer: name of the discretizer
        :type discretizer: str
        """
        # If no list of regions has been provided, generate it from the keys of self.property_containers dict
        if regions is None:
            regions = [key for key in self.property_containers.keys()]

        # Define operators, set engine, set interpolators and define well controls
        self.set_operators(regions)
        engine = self.set_engine(discretizer, platform)
        self.set_interpolators(platform, itor_type, itor_mode, itor_precision)
        self.define_well_controls()
        return engine

    def set_engine(self, discretizer: str = 'mech_discretizer', platform: str = 'cpu'):
        """
        Function to set :class:`engine_super` object.

        :param discretizer: Which discretizer in use (affect the choice of engine):
        'mech_discretizer' (default) or 'pm_discretizer'
        :type discretizer: str
        :param platform: Switch for CPU/GPU engine, 'cpu' (default) or 'gpu'
        :type platform: str
        """
        if discretizer == 'mech_discretizer':
            if self.thermal:
                engine = eval("engine_super_elastic_%s%d_%d_t" % (platform, self.nc, self.nph))()
            else:
                engine = eval("engine_super_elastic_%s%d_%d" % (platform, self.nc, self.nph))()
        elif discretizer == 'pm_discretizer':
            engine = eval("engine_pm_%s" % (platform))()

        return engine

    def define_well_controls(self):
        # define well control factories
        # Injection wells (upwind method requires both bhp and inj_stream for bhp controlled injection wells):
        self.new_bhp_inj = lambda bhp, inj_stream: bhp_inj_well_control(bhp, value_vector(inj_stream))
        self.new_rate_inj = lambda rate, inj_stream, iph: rate_inj_well_control(self.phases, iph, self.n_vars,
                                                                                self.n_vars, rate, value_vector(inj_stream),
                                                                                self.rate_itor)
        # Production wells:
        self.new_bhp_prod = lambda bhp: bhp_prod_well_control(bhp)
        self.new_rate_prod = lambda rate, iph: rate_prod_well_control(self.phases, iph, self.n_vars,
                                                                      self.n_vars, rate, self.rate_itor)
        return

    def init_wells(self, wells, engine):
        """""
        Function to initialize the well rates for each well
        Arguments:
            -wells: well_object array
        """
        for w in wells:
            assert isinstance(w, ms_well)
            w.init_mech_rate_parameters(engine.N_VARS, engine.P_VAR, self.n_vars, self.phases,
                                        self.rate_itor)#, self.thermal)

    def set_operators(self, regions):
        """
        Function to set operator objects: :class:`ReservoirOperators` for each of the reservoir regions,
        :class:`WellOperators` for the well cells, :class:`RateOperators` for evaluation of rates
        and a :class:`PropertyOperator` for the evaluation of properties.

        :param regions: List of regions. It contains the keys of the `property_containers` and `reservoir_operators` dict
        :type regions: list
        :param output_properties: Output property operators object, default is None
        """
        if self.thermal:
            for region, prop_container in self.property_containers.items():
                self.reservoir_operators[region] = GeomechanicsReservoirThermalOperators(prop_container)
            self.wellbore_operators = GeomechanicsReservoirThermalOperators(self.property_containers[regions[0]])
        else:
            if self.discretizer_name == 'pm_discretizer':
                for region, prop_container in self.property_containers.items():
                    self.reservoir_operators[region] = SinglePhaseGeomechanicsReservoirOperators(prop_container)
                self.wellbore_operators = SinglePhaseGeomechanicsWellOperators(self.property_containers[regions[0]])
            elif self.discretizer_name == 'mech_discretizer':
                for region, prop_container in self.property_containers.items():
                    self.reservoir_operators[region] = GeomechanicsReservoirOperators(prop_container)
                self.wellbore_operators = GeomechanicsReservoirOperators(self.property_containers[regions[0]])

        self.rate_operators = RateOperators(self.property_containers[regions[0]])

        return

    def set_uniform_initial_conditions(self, mesh, uniform_pressure,
                                                    uniform_displacement: list,
                                                    uniform_composition: list = None,
                                                    uniform_temperature = None):
        assert isinstance(mesh, conn_mesh)
        nb = mesh.n_blocks

        # set initial pressure
        pressure = np.array(mesh.pressure, copy=False)
        pressure.fill(uniform_pressure)

        # set initial composition
        if self.nc > 1:
            mesh.composition.resize(nb * (self.nc - 1))
            composition = np.array(mesh.composition, copy=False)
            for c in range(self.nc - 1):
                composition[c::(self.nc - 1)] = uniform_composition[c]

        # set initial temperature
        if self.thermal:
            temperature = np.array(mesh.temperature, copy=False)
            temperature.fill(uniform_temperature)

        # set initial displacements
        displacement = np.array(mesh.displacement, copy=False)
        for i in range(self.n_dim):
            displacement[i::self.n_dim] = uniform_displacement[i]

    def set_nonuniform_initial_conditions(self, mesh, initial_pressure: np.ndarray,
                                                      initial_displacement: np.ndarray,
                                                      initial_composition: np.ndarray = None,
                                                      initial_temperature: np.ndarray = None):
        assert isinstance(mesh, conn_mesh)
        nb = mesh.n_blocks
        n_res_blocks = mesh.n_res_blocks

        # set initial pressure
        pressure = np.array(mesh.pressure, copy=False)
        pressure[:n_res_blocks] = initial_pressure

        # set initial composition
        if self.nc > 1:
            mesh.composition.resize(nb * (self.nc - 1))
            composition = np.array(mesh.composition, copy=False)
            for c in range(self.nc - 1):
                composition[c::(self.nc - 1)] = initial_composition[c]

        # set initial temperature
        if self.thermal:
            temperature = np.array(mesh.temperature, copy=False)
            temperature[:n_res_blocks] = initial_temperature

        # set initial displacements
        displacement = np.array(mesh.displacement, copy=False)
        for i in range(self.n_dim):
            displacement[i:self.n_dim * n_res_blocks:self.n_dim] = initial_displacement[i]
