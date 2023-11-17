import numpy as np
from darts.engines import *
from darts.physics.physics_base import PhysicsBase

from darts.physics.super.operator_evaluator import *


class Compositional(PhysicsBase):
    """
    This is the Physics class for Compositional simulation.

    It includes:
    - Creating Reservoir, Well, Rate and Property operators and interpolators for P-z or P-T-z compositional simulation
    - Initializing the :class:`super_engine`
    - Setting well controls (rate, bhp)
    - Defining initial and boundary conditions
    """
    def __init__(self, components: list, phases: list, timer: timer_node, n_points: int,
                 min_p: float, max_p: float, min_z: float, max_z: float, min_t: float = None, max_t: float = None,
                 thermal: bool = False, cache: bool = False):
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
        """
        # Define nc, nph and (iso)thermal
        nc = len(components)
        nph = len(phases)
        self.thermal = thermal

        # Define variables and OBL axes: pressure, nc-1 components and possibly temperature
        variables = ['pressure'] + components[:-1]
        if self.thermal:
            variables += ['temperature']
            axes_min = value_vector([min_p] + [min_z] * (nc - 1) + [min_t])
            axes_max = value_vector([max_p] + [max_z] * (nc - 1) + [max_t])
        else:
            axes_min = value_vector([min_p] + [min_z] * (nc - 1))
            axes_max = value_vector([max_p] + [max_z] * (nc - 1))

        n_vars = len(variables)
        n_ops = n_vars + nph * n_vars + nph + nph * n_vars + n_vars + 3 + 2 * nph + 1

        # Call PhysicsBase constructor
        super().__init__(variables=variables, nc=nc, phases=phases, n_ops=n_ops,
                         axes_min=axes_min, axes_max=axes_max, n_points=n_points, timer=timer, cache=cache)

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
                self.reservoir_operators[region] = ReservoirThermalOperators(prop_container)
            self.wellbore_operators = ReservoirThermalOperators(self.property_containers[regions[0]])
        else:
            for region, prop_container in self.property_containers.items():
                self.reservoir_operators[region] = ReservoirOperators(prop_container)
            self.wellbore_operators = WellOperators(self.property_containers[regions[0]])

        self.rate_operators = RateOperators(self.property_containers[regions[0]])

        return

    def set_engine(self, discr_type: str = 'tpfa', platform: str = 'cpu'):
        """
        Function to set :class:`engine_super` object.

        :param discr_type: Type of discretization, 'tpfa' (default) or 'mpfa'
        :type discr_type: str
        :param platform: Switch for CPU/GPU engine, 'cpu' (default) or 'gpu'
        :type platform: str
        """
        if discr_type == 'mpfa':
            if self.thermal:
                return eval("engine_super_mp_%s%d_%d_t" % (platform, self.nc, self.nph))()
            else:
                return eval("engine_super_mp_%s%d_%d" % (platform, self.nc, self.nph))()
        else:
            if self.thermal:
                return eval("engine_super_%s%d_%d_t" % (platform, self.nc, self.nph))()
            else:
                return eval("engine_super_%s%d_%d" % (platform, self.nc, self.nph))()

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

    def set_uniform_initial_conditions(self, mesh: conn_mesh,
                                       uniform_pressure: float, uniform_composition: list, uniform_temp: float = None):
        """
        Function to set uniform initial conditions.

        :param mesh: Mesh object
        :type mesh:
        :param uniform_pressure: Uniform pressure setting
        :type uniform_pressure: float
        :param uniform_composition: Uniform composition setting
        :type uniform_composition: list
        :param uniform_temp: Uniform temperature setting, default is None for isothermal
        :type uniform_temp: float
        """
        assert isinstance(mesh, conn_mesh)

        nb = mesh.n_blocks
        """ Uniform Initial conditions """
        # set initial pressure
        pressure = np.array(mesh.pressure, copy=False)
        pressure.fill(uniform_pressure)

        # if thermal, set initial temperature
        if uniform_temp is not None:
            temperature = np.array(mesh.temperature, copy=False)
            temperature.fill(uniform_temp)

         # set initial composition
        mesh.composition.resize(nb * (self.nc - 1))
        composition = np.array(mesh.composition, copy=False)
        # composition[:] = np.array(uniform_composition)
        if self.nc == 2:
            for c in range(self.nc - 1):
                composition[c::(self.nc - 1)] = uniform_composition[:]
        else:
            for c in range(self.nc - 1):  # Denis
                composition[c::(self.nc - 1)] = uniform_composition[c]
