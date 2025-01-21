import numpy as np
from typing import Union
from darts.engines import *
from darts.physics.base.physics_base import PhysicsBase

from darts.physics.base.operators_base import PropertyOperators
from darts.physics.super.operator_evaluator import ReservoirOperators, WellOperators, RateOperators, MassFluxOperators
from darts.physics.super.initialize import Initialize


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
                 state_spec: PhysicsBase.StateSpecification = PhysicsBase.StateSpecification.ISOTHERMAL,
                 cache: bool = False, axes_min = None, axes_max = None, n_axes_points = None):
        """
        This is the constructor of the Compositional Physics class.

        It defines the OBL grid for P-z or P-T-z compositional simulation.
        Use axes_min, axes_max, n_axes_points to define non-uniform OBL properties for different compositions.

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
        :param state_spec: State specification - 0) ISOTHERMAL (default), 1) PT, 2) PH
        :type state_spec: StateSpecification
        :param cache: Switch to cache operator values
        :type cache: bool
        :param axes_min: (optional) Minimum bounds of OBL axes
        :type axes_min: (optional) list or np.ndarray
        :param axes_max: (optional) Maximum bounds of OBL axes
        :type axes_max: (optional) list or np.ndarray
        :param n_axes_points: (optional) Number of points over OBL axes
        :type n_axes_points: (optional) list or np.ndarray
        """
        # Define nc, nph and (iso)thermal
        nc = len(components)
        nph = len(phases)
        self.thermal = (state_spec > PhysicsBase.StateSpecification.ISOTHERMAL)

        # Define state variables and OBL axes: pressure, nc-1 components and possibly temperature/enthalpy
        variables = ['pressure'] + components[:-1]
        if self.thermal:
            variables += ['temperature'] if state_spec == PhysicsBase.StateSpecification.PT else ['enthalpy']

        n_vars = len(variables)
        # Number of operators = NE /*acc*/ + NE * NP /*flux*/ + NP /*UPSAT*/ + NE * NP /*gradient*/ + NE /*kinetic*/
        # + 2 * NP /*gravpc*/ + 1 /*poro*/ + NP /*enthalpy*/ + 2 /*temperature and pressure*/
        # = NE * (2 * nph + 2) + 4 * nph + 3
        n_ops = n_vars * (2 * nph + 2) + 4 * nph + 3

        # axes_min
        if axes_min is None:
            if self.thermal:
                axes_min = value_vector([min_p] + [min_z] * (nc - 1) + [min_t])
            else:
                axes_min = value_vector([min_p] + [min_z] * (nc - 1))
        else:
            axes_min = value_vector(axes_min)

        # axes_max
        if axes_max is None:
            if self.thermal:
                axes_max = value_vector([max_p] + [max_z] * (nc - 1) + [max_t])
            else:
                axes_max = value_vector([max_p] + [max_z] * (nc - 1))
        else:
            axes_max = value_vector(axes_max)

        # n_axes_points
        if n_axes_points is None:
            n_axes_points = index_vector([n_points] * n_vars)
        else:
            n_axes_points = index_vector(n_axes_points)

        # Call PhysicsBase constructor
        super().__init__(state_spec=state_spec, variables=variables, nc=nc, phases=phases, n_ops=n_ops,
                         axes_min=axes_min, axes_max=axes_max, n_axes_points=n_axes_points, timer=timer, cache=cache)

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

    def set_operators(self):
        """
        Function to set operator objects: :class:`ReservoirOperators` for each of the reservoir regions,
        :class:`WellOperators` for the well cells, :class:`RateOperators` for evaluation of rates
        and a :class:`PropertyOperator` for the evaluation of properties.
        """
        for region in self.regions:
            self.reservoir_operators[region] = ReservoirOperators(self.property_containers[region], self.thermal)
            self.property_operators[region] = PropertyOperators(self.property_containers[region], self.thermal)
            self.mass_flux_operators[region] = MassFluxOperators(self.property_containers[region], self.thermal)

        if self.thermal:
            self.wellbore_operators = ReservoirOperators(self.property_containers[self.regions[0]], self.thermal)
        else:
            self.wellbore_operators = WellOperators(self.property_containers[self.regions[0]], self.thermal)

        self.rate_operators = RateOperators(self.property_containers[self.regions[0]])

        return

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
                                       pressure_input: Union[float, list, np.ndarray],
                                       composition_input: Union[list, np.ndarray] = None,
                                       temperature_input: Union[float, list, np.ndarray] = None):
        """
        Method to set initial conditions by arrays or uniformly for all cells

        :param mesh: conn_mesh object
        :param pressure_input: Pressure [bar], uniform or array
        :param composition_input: List of compositions [z_0, ..., z_{nc-1}], set of scalars or arrays
        :param temperature_input: Temperature [K], only required for thermal models, uniform or array
        """
        assert isinstance(mesh, conn_mesh)

        nb = mesh.n_blocks
        """ Uniform Initial conditions """
        # set initial pressure
        pressure = np.array(mesh.pressure, copy=False)
        pressure[:] = pressure_input

        # if thermal, set initial temperature or enthalpy
        if self.thermal:
            if self.state_spec == PhysicsBase.StateSpecification.PT:
                temperature = np.array(mesh.temperature, copy=False)
                temperature[:] = temperature_input
            else:
                enthalpy = np.array(mesh.temperature, copy=False)  # TODO: access first and second state variable, not T or H by name
                if hasattr(pressure_input, '__len__'):
                    # Pressure specified as an array
                    for j in range(mesh.n_blocks):
                        state = value_vector([pressure_input[j], 0])
                        temp = temperature_input[j] if hasattr(temperature_input, "__len__") else temperature_input
                        enthalpy[j] = self.property_containers[0].compute_total_enthalpy(state, temp)
                else:
                    state = value_vector([pressure_input, 0])
                    enth = self.property_containers[0].compute_total_enthalpy(state, temperature_input)
                    enthalpy[:] = enth

        # set initial composition
        mesh.composition.resize(nb * (self.nc - 1))
        composition = np.array(mesh.composition, copy=False)
        # composition[:] = np.array(uniform_composition)
        if self.nc == 2:
            for c in range(self.nc - 1):
                composition[c::(self.nc - 1)] = composition_input[:] if not hasattr(composition_input[0], "__len__") \
                    else composition_input[0, :]
        else:
            for c in range(self.nc - 1):  # Denis
                composition[c::(self.nc - 1)] = composition_input[c] if not hasattr(composition_input[0], "__len__") \
                    else composition_input[c, :]

    def set_nonuniform_initial_conditions(self, mesh: conn_mesh, pressure_grad: float = 0., temperature_grad: float = 0.,
                                          ref_depth_p: float = 0., p_at_ref_depth: float = 1.,
                                          ref_depth_T: float = 0., T_at_ref_depth: float = 293.15,
                                          composition_input: Union[list, np.ndarray] = None):
        """
        Method to set initial conditions with gradients

        :param mesh: conn_mesh object
        :param pressure_grad: Pressure gradient [bar/km], calculates pressure based on depth [1/km], default is 0
        :param temperature_grad: Temperature gradient [K/km], calculates temperature based on depth [1/km], default is 0
        :param ref_depth_p: Reference depth for pressure [km], default is 0
        :param p_at_ref_depth: Pressure at reference depth [bar], default is 1
        :param ref_depth_T: Reference depth for temperature [K], default is 0
        :param T_at_ref_depth: Temperature at reference depth [K], default is 293.15
        :param composition_input: List of compositions [z_0, ..., z_{nc-1}], set of scalars or arrays
        """
        assert isinstance(mesh, conn_mesh)
        nb = mesh.n_blocks

        """ Non-Uniform Initial conditions """
        depth = np.array(mesh.depth, copy=True)
        # set initial pressure
        pressure = np.array(mesh.pressure, copy=False)
        pressure[:] = (depth[:pressure.size] / 1000 - ref_depth_p) * pressure_grad + p_at_ref_depth

        # if thermal, set initial temperature/enthalpy
        if self.thermal:
            if self.state_spec == PhysicsBase.StateSpecification.PT:
                temperature = np.array(mesh.temperature, copy=False)
                temperature[:] = (depth[:pressure.size] / 1000 - ref_depth_T) * temperature_grad + T_at_ref_depth
            else:
                depth = np.array(mesh.depth, copy=True)

                # set initial enthalpy through given temperature and pressure
                enthalpy = np.array(mesh.temperature, copy=False)  # TODO: access first and second state variable, not T or H by name
                temperature = (depth[:pressure.size] / 1000 - ref_depth_T) * temperature_grad + T_at_ref_depth

                for j in range(mesh.n_blocks):
                    comp = composition_input[:, j] if hasattr(composition_input[0], "__len__") else composition_input
                    state = value_vector([pressure[j], 0] + comp)
                    enthalpy[j] = self.property_containers[0].compute_total_enthalpy(state, temperature[j])

        # set initial composition
        mesh.composition.resize(nb * (self.nc - 1))
        composition = np.array(mesh.composition, copy=False)
        # composition[:] = np.array(uniform_composition)
        if self.nc == 2:
            for c in range(self.nc - 1):
                composition[c::(self.nc - 1)] = composition_input[:] if not hasattr(composition_input[0], "__len__") \
                    else composition_input[0, :]
        else:
            for c in range(self.nc - 1):  # Denis
                composition[c::(self.nc - 1)] = composition_input[c] if not hasattr(composition_input[0], "__len__") \
                    else composition_input[c, :]

    def init_wells(self, wells):
        """
        Function to initialize the well rates for each well.

        :param wells: List of :class:`ms_well` objects
        """
        for w in wells:
            assert isinstance(w, ms_well)
            w.init_rate_parameters(self.n_vars, self.n_ops, self.phases, self.rate_itor, self.thermal)
