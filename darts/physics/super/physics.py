import warnings
from typing import Union

import numpy as np
from scipy.interpolate import interp1d

from darts.engines import *
from darts.physics.base.operators_base import (
    PropertyOperators,
    WellControlOperators,
    WellInitOperators,
)
from darts.physics.base.physics_base import PhysicsBase
from darts.physics.super.operator_evaluator import ReservoirOperators, WellOperators


class Compositional(PhysicsBase):
    """
    This is the Physics class for Compositional simulation.

    It includes:
    - Creating Reservoir, Well, Rate and Property operators and interpolators for P-z or P-T-z compositional simulation
    - Initializing the :class:`super_engine`
    - Setting well controls (rate, bhp)
    - Defining initial and boundary conditions
    """

    def __init__(
        self,
        components: list,
        phases: list,
        timer: timer_node,
        n_points: int,
        min_p: float,
        max_p: float,
        min_z: float,
        max_z: float,
        min_t: float = None,
        max_t: float = None,
        state_spec: PhysicsBase.StateSpecification = PhysicsBase.StateSpecification.P,
        cache: bool = False,
        axes_min=None,
        axes_max=None,
        n_axes_points=None,
    ):
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
        self.thermal = state_spec > PhysicsBase.StateSpecification.P

        # Define state variables and OBL axes: pressure, nc-1 components and possibly temperature/enthalpy
        variables = ['pressure'] + components[:-1]
        if self.thermal:
            variables += (
                ['temperature']
                if state_spec == PhysicsBase.StateSpecification.PT
                else ['enthalpy']
            )

        n_vars = len(variables)
        # Number of operators = NE /*acc*/ + NE * NP /*flux*/ + NP /*UPSAT*/ + NE * NP /*gradient*/ + NE /*kinetic*/
        # + 2 * NP /*gravpc*/ + 1 /*poro*/ + NP /*enthalpy*/ + 2 /*temperature and pressure*/
        # = NE * (2 * nph + 2) + 4 * nph + 3
        n_ops = n_vars * (2 * nph + 2) + 4 * nph + 3

        # axes_min
        if axes_min is None:
            if self.thermal:
                axes_min = [min_p] + [min_z] * (nc - 1) + [min_t]
            else:
                axes_min = [min_p] + [min_z] * (nc - 1)

        # axes_max
        if axes_max is None:
            if self.thermal:
                axes_max = [max_p] + [max_z] * (nc - 1) + [max_t]
            else:
                axes_max = [max_p] + [max_z] * (nc - 1)

        # n_axes_points
        if n_axes_points is None:
            n_axes_points = index_vector([n_points] * n_vars)
        else:
            n_axes_points = index_vector(n_axes_points)

        # Call PhysicsBase constructor
        super().__init__(
            state_spec=state_spec,
            variables=variables,
            components=components,
            phases=phases,
            n_ops=n_ops,
            axes_min=axes_min,
            axes_max=axes_max,
            n_axes_points=n_axes_points,
            timer=timer,
            cache=cache,
        )

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
                return eval(
                    "engine_super_mp_%s%d_%d_t" % (platform, self.nc, self.nph)
                )()
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
        :class:`WellOperators` for the well segments, :class:`WellControlOperators` for well control
        and a :class:`PropertyOperator` for the evaluation of properties.
        """
        for region in self.regions:
            self.reservoir_operators[region] = ReservoirOperators(
                self.property_containers[region], self.thermal
            )
            self.property_operators[region] = PropertyOperators(
                self.property_containers[region], self.thermal
            )

        if self.thermal:
            self.well_operators = ReservoirOperators(
                self.property_containers[self.regions[0]], self.thermal
            )
        else:
            self.well_operators = WellOperators(
                self.property_containers[self.regions[0]], self.thermal
            )

        self.well_ctrl_operators = WellControlOperators(
            self.property_containers[self.regions[0]], self.thermal
        )
        self.well_init_operators = WellInitOperators(
            self.property_containers[self.regions[0]],
            self.thermal,
            is_pt=(self.state_spec <= PhysicsBase.StateSpecification.PT),
        )

        return

    def set_initial_conditions_from_depth_table(
        self,
        mesh: conn_mesh,
        input_distribution: dict,
        input_depth: Union[list, np.ndarray],
        global_to_local=None,
    ):
        """
        Function to set initial conditions from given distribution of properties over depth.

        :param mesh: conn_mesh object
        :param input_distribution: Initial distributions of unknowns over depth, must have keys equal to self.vars
                                   and each entry is scalar or array of length equal to depths
        :param input_depth: Array of depths over which depth table has been specified
        :param global_to_local: array of indices mapping to active elements
        """
        # Assertions of consistent depth table specification
        assert np.all(
            [
                variable in input_distribution.keys()
                for variable in self.vars[1 : self.nc]
            ]
        ), "Initial state for must be specified for all primary variables"
        assert not self.thermal or (
            'temperature' in input_distribution.keys()
            or 'enthalpy' in input_distribution.keys()
        ), "Temperature or enthalpy must be specified for thermal models"
        input_depth = (
            input_depth
            if not (np.isscalar(input_depth) or len(input_depth) == 1)
            else np.array([input_depth, input_depth + 1.0]).flatten()
        )
        for key, input_values in input_distribution.items():
            input_distribution[key] = (
                input_values
                if not (np.isscalar(input_values) or len(input_values) == 1)
                else (np.ones(len(input_depth)) * input_values)
            )
            assert len(input_distribution[key]) == len(input_depth)

        # Get depths and primary variable arrays from mesh object
        depths = np.asarray(mesh.depth)[: mesh.n_res_blocks]
        if global_to_local is not None:
            depths = depths[global_to_local]

        # adjust the size of initial_state array in c++
        mesh.initial_state.resize(mesh.n_res_blocks * self.n_vars)

        # Loop over variables to fill initial_state vector in c++
        for ith_var, variable in enumerate(self.vars):
            if variable == "enthalpy" and "enthalpy" not in input_distribution.keys():
                # If temperature has been provided, interpolate pressure and temperature to compute enthalpies
                p_itor = interp1d(
                    input_depth,
                    input_distribution['pressure'],
                    kind='linear',
                    fill_value='extrapolate',
                )
                pressure = p_itor(depths)

                t_itor = interp1d(
                    input_depth,
                    input_distribution['temperature'],
                    kind='linear',
                    fill_value='extrapolate',
                )
                temperature = t_itor(depths)

                z_itors = [
                    interp1d(
                        input_depth,
                        input_distribution[comp],
                        kind='linear',
                        fill_value='extrapolate',
                    )
                    for comp in self.components[:-1]
                ]
                zi = np.array([z_itor(depths) for z_itor in z_itors])

                values = np.empty(mesh.n_res_blocks)
                for j in range(mesh.n_res_blocks):
                    zc = (
                        np.append(np.asarray(zi[:, j]), 1.0 - np.sum(zi[:, j]))
                        if self.nc > 1
                        else np.array([1.0])
                    )
                    state_pt = np.array([pressure[j]] + list(zc) + [temperature[j]])

                    values[j] = self.property_containers[0].compute_total_enthalpy(
                        state_pt
                    )
            else:
                # Else, interpolate primary variable
                itor = interp1d(
                    input_depth,
                    input_distribution[variable],
                    kind='linear',
                    fill_value='extrapolate',
                )
                values = itor(depths)

            values = np.resize(np.asarray(values), mesh.n_res_blocks)
            np.asarray(mesh.initial_state)[ith_var :: self.n_vars] = values

    def set_initial_conditions_from_array(
        self, mesh: conn_mesh, input_distribution: dict
    ):
        """
        Function to set uniform initial reservoir condition

        :param mesh: conn_mesh object
        :param input_distribution: Initial distributions of unknowns over grid, must have keys equal to self.vars
                                   and each entry is scalar or array of length equal to number of cells
        """
        for variable, values in input_distribution.items():
            if not np.isscalar(values) and not len(values) == mesh.n_res_blocks:
                warnings.warn(
                    'Initial condition for variable {} has different length, resizing {} to {}'.format(
                        variable, len(values), mesh.n_res_blocks
                    ),
                    stacklevel=2,
                )
                input_distribution[variable] = np.resize(
                    np.asarray(values), mesh.n_res_blocks
                )

        # adjust the size of initial_state array in c++
        mesh.initial_state.resize(mesh.n_res_blocks * self.n_vars)

        # set initial pressure
        np.asarray(mesh.initial_state)[0 :: self.n_vars] = input_distribution[
            'pressure'
        ]

        # if thermal, set initial temperature or enthalpy
        if self.thermal:
            if self.state_spec == PhysicsBase.StateSpecification.PT:
                np.asarray(mesh.initial_state)[(self.n_vars - 1) :: self.n_vars] = (
                    input_distribution['temperature']
                )
            else:
                # interpolate pressure and temperature to compute enthalpies
                enthalpy = np.empty(mesh.n_res_blocks)
                if not np.isscalar(input_distribution['pressure']):
                    # Pressure specified as an array
                    for j in range(mesh.n_res_blocks):
                        components = [
                            (
                                input_distribution[name][j]
                                if not np.isscalar(input_distribution[name])
                                else input_distribution[name]
                            )
                            for name in self.property_containers[0].components_name[:-1]
                        ]
                        temp = (
                            input_distribution['temperature'][j]
                            if not np.isscalar(input_distribution['temperature'])
                            else input_distribution['temperature']
                        )

                        state = np.array(
                            [input_distribution['pressure'][j], *components, temp]
                        )
                        enthalpy[j] = self.property_containers[
                            0
                        ].compute_total_enthalpy(state)
                else:
                    components = [
                        input_distribution[name]
                        for name in self.property_containers[0].components_name[:-1]
                    ]
                    state = value_vector(
                        [
                            input_distribution['pressure'],
                            *components,
                            input_distribution['temperature'],
                        ]
                    )  # enthalpy is dummy variable
                    enth = self.property_containers[0].compute_total_enthalpy(state)
                    enthalpy[:] = enth

                np.asarray(mesh.initial_state)[
                    (self.n_vars - 1) :: self.n_vars
                ] = enthalpy

        # set initial composition
        for c in range(self.nc - 1):
            np.asarray(mesh.initial_state)[(c + 1) :: self.n_vars] = (
                input_distribution[self.vars[c + 1]]
                if np.isscalar(input_distribution[self.vars[c + 1]])
                else input_distribution[self.vars[c + 1]][:]
            )
