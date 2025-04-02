import numpy as np
from typing import Union
import warnings
from scipy.interpolate import interp1d
from darts.engines import *
from darts.physics.base.physics_base import PhysicsBase

from darts.physics.base.operators_base import PropertyOperators
from darts.physics.super.operator_evaluator import ReservoirOperators, WellOperators, RateOperators, MassFluxOperators


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
                 state_spec: PhysicsBase.StateSpecification = PhysicsBase.StateSpecification.P,
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
        :param state_spec: State specification - 0) P (default), 1) PT, 2) PH
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
        self.thermal = (state_spec > PhysicsBase.StateSpecification.P)

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
        super().__init__(state_spec=state_spec, variables=variables, components=components, phases=phases, n_ops=n_ops,
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

    def set_initial_conditions_from_depth_table(self, mesh: conn_mesh, input_distribution: dict,
                                                input_depth: Union[list, np.ndarray]):
        """
        Function to set initial conditions from given distribution of properties over depth.

        :param mesh: conn_mesh object
        :param input_distribution: Initial distributions of unknowns over depth, must have keys equal to self.vars
                                   and each entry is scalar or array of length equal to depths
        :param input_depth: Array of depths over which depth table has been specified
        """
        # Assertions of consistent depth table specification
        assert np.all([variable in input_distribution.keys() for variable in self.vars[1:self.nc]]), \
            "Initial state for must be specified for all primary variables"
        assert not self.thermal or ('temperature' in input_distribution.keys() or
                                    'enthalpy' in input_distribution.keys()), \
            "Temperature or enthalpy must be specified for thermal models"
        input_depth = input_depth if not (np.isscalar(input_depth) or len(input_depth) == 1) else np.array([input_depth, input_depth + 1.]).flatten()
        for key, input_values in input_distribution.items():
            input_distribution[key] = input_values if not (np.isscalar(input_values) or len(input_values) == 1) else \
                (np.ones(len(input_depth)) * input_values)
            assert len(input_distribution[key]) == len(input_depth)

        # Get depths and primary variable arrays from mesh object
        depths = np.asarray(mesh.depth)

        # adjust the size of initial_state array in c++
        mesh.initial_state.resize(mesh.n_blocks * self.n_vars)

        # Loop over variables to fill initial_state vector in c++
        for ith_var, variable in enumerate(self.vars):
            if variable == "enthalpy" and "enthalpy" not in input_distribution.keys():
                # If temperature has been provided, interpolate pressure and temperature to compute enthalpies
                p_itor = interp1d(input_depth, input_distribution['pressure'], kind='linear', fill_value='extrapolate')
                pressure = p_itor(depths)

                t_itor = interp1d(input_depth, input_distribution['temperature'], kind='linear', fill_value='extrapolate')
                temperature = t_itor(depths)

                z_itors = [interp1d(input_depth, input_distribution[comp], kind='linear', fill_value='extrapolate') for
                           comp in self.components[:-1]]
                zi = np.array([z_itor(depths) for z_itor in z_itors])

                values = np.empty(mesh.n_res_blocks)
                for j in range(mesh.n_res_blocks):
                    zc = np.append(np.asarray(zi[:, j]), 1. - np.sum(zi[:, j])) if self.nc > 1 else np.array([1.])

                    values[j] = self.property_containers[0].compute_total_enthalpy(pressure[j], temperature[j], zc)
            else:
                # Else, interpolate primary variable
                itor = interp1d(input_depth, input_distribution[variable], kind='linear', fill_value='extrapolate')
                values = itor(depths)

            values = np.resize(np.asarray(values), mesh.n_blocks)
            np.asarray(mesh.initial_state)[ith_var::self.n_vars] = values

    def set_initial_conditions_from_array(self, mesh: conn_mesh, input_distribution: dict):
        """
        Function to set uniform initial reservoir condition

        :param mesh: conn_mesh object
        :param input_distribution: Initial distributions of unknowns over grid, must have keys equal to self.vars
                                   and each entry is scalar or array of length equal to number of cells
        """
        for variable, values in input_distribution.items():
            if not np.isscalar(values) and not len(values) == mesh.n_blocks:
                warnings.warn('Initial condition for variable {} has different length, resizing {} to {}'.
                              format(variable, len(values), mesh.n_blocks))
                input_distribution[variable] = np.resize(np.asarray(values), mesh.n_blocks)

        # adjust the size of initial_state array in c++
        mesh.initial_state.resize(mesh.n_blocks * self.n_vars)

        # set initial pressure
        np.asarray(mesh.initial_state)[0::self.n_vars] = input_distribution['pressure']

        # if thermal, set initial temperature or enthalpy
        if self.thermal:
            if self.state_spec == PhysicsBase.StateSpecification.PT:
                np.asarray(mesh.initial_state)[(self.n_vars - 1)::self.n_vars] = input_distribution['temperature']
            else:
                # interpolate pressure and temperature to compute enthalpies
                enthalpy = np.empty(mesh.n_blocks)
                if not np.isscalar(input_distribution['pressure']):
                    # Pressure specified as an array
                    for j in range(mesh.n_blocks):
                        state = value_vector([input_distribution['pressure'][j], 0])
                        temp = input_distribution['temperature'][j] if not np.isscalar(input_distribution['temperature']) else input_distribution['temperature']
                        enthalpy[j] = self.property_containers[0].compute_total_enthalpy(state, temp)
                else:
                    state = value_vector([input_distribution['pressure'], 0])  # enthalpy is dummy variable
                    enth = self.property_containers[0].compute_total_enthalpy(state, input_distribution['temperature'])
                    enthalpy[:] = enth

                np.asarray(mesh.initial_state)[(self.n_vars-1)::self.n_vars] = enthalpy

        # set initial composition
        for c in range(self.nc-1):
            np.asarray(mesh.initial_state)[(c+1)::self.n_vars] = input_distribution[self.vars[c+1]] \
                if np.isscalar(input_distribution[self.vars[c+1]]) else input_distribution[self.vars[c+1]][:]

    def init_wells(self, wells):
        """
        Function to initialize the well rates for each well.

        :param wells: List of :class:`ms_well` objects
        """
        for w in wells:
            assert isinstance(w, ms_well)
            w.init_rate_parameters(self.n_vars, self.n_ops, self.phases, self.rate_itor, self.thermal)
