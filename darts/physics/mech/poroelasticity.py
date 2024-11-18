from darts.engines import *
from darts.physics.super.physics import Compositional
from darts.physics.super.operator_evaluator import *
from darts.physics.base.operators_base import PropertyOperators
import numpy as np


class Poroelasticity(Compositional):
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
                 thermal: bool = False, cache: bool = False, discretizer: str = 'mech_discretizer',
                 axes_min = None, axes_max = None, n_axes_points = None):
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
        :param axes_min: Minimum bounds of OBL axes
        :type axes_min: list or np.ndarray
        :param axes_max: Maximum bounds of OBL axes
        :type axes_max: list or np.ndarray
        :param n_axes_points: Number of points over OBL axes
        :type n_axes_points: list or np.ndarray
        """
        # Define nc, nph and (iso)thermal
        super().__init__(components, phases, timer, n_points, min_p, max_p, min_z, max_z, min_t, max_t, thermal, cache,
                         axes_min, axes_max, n_axes_points)

        self.n_dim = 3
        self.discretizer_name = discretizer

        if self.discretizer_name == 'mech_discretizer':
            self.n_ops = (self.n_vars + self.nph * self.n_vars + self.nph + self.nph * self.n_vars + self.n_vars
                          + 3 + 2 * self.nph + 1 + 1)
        else:  # if self.discretizer_name == 'pm_discretizer':
            self.n_ops = 2 * self.n_vars
            assert not self.thermal

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
                return eval("engine_super_elastic_%s%d_%d_t" % (platform, self.nc, self.nph))()
            else:
                return eval("engine_super_elastic_%s%d_%d" % (platform, self.nc, self.nph))()
        else:  # discretizer == 'pm_discretizer':
            return eval("engine_pm_%s" % (platform))()

    def set_operators(self):
        """
        Function to set operator objects: :class:`ReservoirOperators` for each of the reservoir regions,
        :class:`WellOperators` for the well cells, :class:`RateOperators` for evaluation of rates
        and a :class:`PropertyOperator` for the evaluation of properties.
        """
        if self.discretizer_name == "pm_discretizer":
            for region, prop_container in self.property_containers.items():
                self.reservoir_operators[region] = SinglePhaseGeomechanicsOperators(prop_container, self.thermal)
                self.property_operators[region] = PropertyOperators(prop_container, self.thermal)
                self.mass_flux_operators[region] = MassFluxOperators(self.property_containers[region], self.thermal)
            self.wellbore_operators = SinglePhaseGeomechanicsOperators(self.property_containers[self.regions[0]], self.thermal)
        else:
            for region, prop_container in self.property_containers.items():
                self.reservoir_operators[region] = GeomechanicsReservoirOperators(prop_container, self.thermal)
                self.property_operators[region] = PropertyOperators(prop_container, self.thermal)
                self.mass_flux_operators[region] = MassFluxOperators(self.property_containers[region], self.thermal)
            self.wellbore_operators = GeomechanicsReservoirOperators(self.property_containers[self.regions[0]], False)

        self.rate_operators = RateOperators(self.property_containers[self.regions[0]])

        return

    def init_wells(self, wells):
        """""
        Function to initialize the well rates for each well
        Arguments:
            -wells: well_object array
        """
        for w in wells:
            assert isinstance(w, ms_well)
            w.init_mech_rate_parameters(self.engine.N_VARS, self.engine.P_VAR, self.n_vars,
                                        self.n_ops, self.phases, self.rate_itor, self.thermal)

    def set_uniform_initial_conditions(self, mesh, uniform_pressure, uniform_displacement: list,
                                       uniform_composition: list = None, uniform_temperature: float = None):
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

    def set_nonuniform_initial_conditions(self, mesh, initial_pressure: np.ndarray, initial_displacement: np.ndarray,
                                          initial_composition: np.ndarray = None, initial_temperature: np.ndarray = None):
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
