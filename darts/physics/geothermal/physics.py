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
from darts.physics.geothermal.operator_evaluator import *


class Geothermal(PhysicsBase):
    """
    This is the Physics class for Geothermal simulation.

    It includes:
    - Creating Reservoir, Well, Rate and Property operators and interpolators for P-H simulation
    - Initializing the :class:`nce_g_engine`
    - Setting well controls (rate, bhp)
    - Defining initial and boundary conditions
    """

    def __init__(
        self,
        timer: timer_node,
        n_points: int,
        min_p: float,
        max_p: float,
        min_e: float,
        max_e: float,
        cache: bool = False,
    ):
        """
        This is the constructor of the Geothermal Physics class.

        It defines the OBL grid for P-H simulation.

        :param timer: Timer object
        :type timer: :class:`darts.engines.timer_node`
        :param n_points: Number of OBL points along axes
        :type n_points: int
        :param min_p, max_p: Minimum, maximum pressure
        :type min_p, max_p: float
        :param min_e, max_e: Minimum, maximum enthalpy
        :type min_e, max_e: float
        :param cache: Switch to cache operator values
        :type cache: bool
        """
        # Set nc=1, thermal=True
        components = ["H2O"]

        # Define phases and variables
        phases = ['water', 'steam']
        variables = ['pressure', 'enthalpy']
        state_spec = PhysicsBase.StateSpecification.PH

        # Define OBL axes
        self.axes_min = value_vector([min_p, min_e])
        self.axes_max = value_vector([max_p, max_e])
        n_axes_points = index_vector([n_points] * len(variables))

        # Define number of operators:
        # N_OPS = NC /*acc*/ + NC * NP /*flux*/ + 2 + NP /*energy acc, flux, cond*/ + NP /*density*/ + 1 /*temperature*/
        # = nc + nc*NP + 2 + NP + NP + 1 = 10
        n_ops = 10

        # Call PhysicsBase constructor
        super().__init__(
            state_spec=state_spec,
            variables=variables,
            components=components,
            phases=phases,
            n_ops=n_ops,
            axes_min=self.axes_min,
            axes_max=self.axes_max,
            n_axes_points=n_axes_points,
            timer=timer,
            cache=cache,
        )
        self.PT_axes_min = value_vector([min_p, 273.15])
        self.PT_axes_max = value_vector([max_p, 273.15 + 300.0])

        self.thermal = True

    def determine_obl_bounds(
        self,
        min_p: float,
        max_p: float,
        min_z: float = None,
        max_z: float = None,
        min_t: float = None,
        max_t: float = None,
        state_spec: PhysicsBase.StateSpecification = PhysicsBase.StateSpecification.PH,
    ):
        """
        Overload determine_obl_bounds() method to hardcode OBL axes of pressure-enthalpy and PT-axes for WellInitOperators
        """
        return self.axes_min, self.axes_max

    def set_operators(self):
        """
        Function to set operator objects: :class:`acc_flux_gravity_evaluator` for each of the reservoir regions,
        :class:`acc_flux_gravity_evaluator_python_well` for the well segments
        and :class:`geothermal_rate_custom_evaluator_python` for evaluation of rates.
        """
        for region in self.regions:
            self.reservoir_operators[region] = acc_flux_gravity_evaluator_python(
                self.property_containers[region]
            )
            self.property_operators[region] = PropertyOperators(
                self.property_containers[region], thermal=True
            )
        self.well_operators = acc_flux_gravity_evaluator_python_well(
            self.property_containers[self.regions[0]]
        )

        # create well control operators evaluator
        self.well_ctrl_operators = WellControlOperators(
            self.property_containers[self.regions[0]], self.thermal
        )
        self.well_init_operators = WellInitOperators(
            self.property_containers[self.regions[0]],
            self.thermal,
            is_pt=(self.state_spec <= PhysicsBase.StateSpecification.PT),
        )

        return

    def set_engine(self, discr_type: str = 'tpfa', platform: str = 'cpu'):
        """
        Function to set :class:`engine_nce_g` object.

        :param discr_type: Type of discretization, 'tpfa' (default) or 'mpfa'
        :type discr_type: str
        :param platform: Switch for CPU/GPU engine, 'cpu' (default) or 'gpu'
        :type platform: str
        """
        return eval("engine_nce_g_%s%d_%d" % (platform, self.nc, self.nph))()

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
        assert 'pressure' in input_distribution.keys() and (
            'temperature' in input_distribution.keys()
            or 'enthalpy' in input_distribution.keys()
        )
        input_depth = (
            input_depth if not np.isscalar(input_depth) else np.array([input_depth])
        )
        for key, input_values in input_distribution.values():
            input_values = (
                input_values
                if not np.isscalar(input_values)
                else np.ones(len(input_depth)) * input_values
            )
            assert len(input_values) == len(input_depth)

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

                values = np.empty(mesh.n_res_blocks)
                for j in range(mesh.n_res_blocks):
                    state_pt = np.array([pressure[j], temperature[j]])
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

            np.asarray(mesh.initial_state)[ith_var :: self.n_vars] = values

    def set_initial_conditions_from_array(
        self, mesh: conn_mesh, input_distribution: dict
    ):
        """ ""
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

        # interpolate pressure and temperature to compute enthalpies
        enthalpy = np.empty(mesh.n_res_blocks)
        if 'enthalpy' in input_distribution.keys():
            enth = (
                np.ones(mesh.n_res_blocks) * input_distribution['enthalpy']
                if not np.isscalar(input_distribution['enthalpy'])
                else input_distribution['enthalpy']
            )
            enthalpy[:] = enth
        elif not np.isscalar(input_distribution['pressure']):
            # Pressure specified as an array
            for j in range(mesh.n_res_blocks):
                temp = (
                    input_distribution['temperature'][j]
                    if not np.isscalar(input_distribution['temperature'])
                    else input_distribution['temperature']
                )
                state_pt = np.array([input_distribution['pressure'][j], temp])
                enthalpy[j] = self.property_containers[0].compute_total_enthalpy(
                    state_pt
                )
        else:
            state_pt = np.array(
                [input_distribution['pressure'], input_distribution['temperature']]
            )
            enth = self.property_containers[0].compute_total_enthalpy(state_pt)
            enthalpy[:] = enth

        np.asarray(mesh.initial_state)[(self.n_vars - 1) :: self.n_vars] = enthalpy
