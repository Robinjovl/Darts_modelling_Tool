import warnings

import numpy as np
from scipy.interpolate import interp1d

from darts.engines import *
from darts.physics.base.operators_base import (
    PropertyOperators,
    ThermalVarOperator,
    WellCtrlOperators,
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
        axes_step: list[float],
        axes_origin: list[float] = None,
        thermal_var_axes_step: list[float] = None,
        thermal_var_axes_origin: list[float] = None,
        cache: bool = False,
    ):
        """
        Constructor of the Geothermal Physics class. Defines the OBL grid for P-H simulation.

        :param timer: Timer object.
        :param axes_step: [p_step, e_step] — per-axis cell size for the pressure-enthalpy grid.
        :param axes_origin: [p_origin, e_origin] grid origin. Default ``[1.0, 1000.0]``
            (1 bar pressure, 1000 kJ/kmol enthalpy reference matching the in-tree models).
        :param thermal_var_axes_step: [p_step, T_step] for the ThermalVarOperator grid
            (P–T parametrization). Default ``[axes_step[0], 1.0]`` (1 K per cell).
        :param thermal_var_axes_origin: [p_origin, T_origin] origin for the ThermalVarOperator
            grid. Default ``[axes_origin[0], 273.15]``.
        :param cache: Cache supporting points to disk between runs.
        """
        components = ["H2O"]
        phases = ['water', 'steam']
        variables = ['pressure', 'enthalpy']
        state_spec = PhysicsBase.StateSpecification.PH

        # N_OPS = NC + NC*NP + 2 + NP + NP + 1 = 10
        n_ops = 10

        assert len(axes_step) == 2, "axes_step must have 2 entries: [p_step, e_step]"
        if axes_origin is None:
            # Sensible defaults matching open-DARTS unit conventions and the
            # enthalpy reference used by every in-tree Geothermal model:
            #   pressure → 1 bar
            #   enthalpy → 1000 kJ/kmol
            # Override via axes_origin when a different EOS reference is needed.
            axes_origin = [1.0, 1000.0]
        assert len(axes_origin) == 2

        # ThermalVarOperator (P-T) grid setup — store on self so PhysicsBase.set_interpolators
        # picks it up via getattr.
        if thermal_var_axes_step is None:
            thermal_var_axes_step = [axes_step[0], 1.0]
        if thermal_var_axes_origin is None:
            # Pressure origin inherits the user's choice; temperature uses 273.15 K (0 °C).
            thermal_var_axes_origin = [axes_origin[0], 273.15]
        self.thermal_var_axes_step = list(thermal_var_axes_step)
        self.thermal_var_axes_origin = list(thermal_var_axes_origin)

        super().__init__(
            state_spec=state_spec,
            variables=variables,
            components=components,
            phases=phases,
            n_ops=n_ops,
            timer=timer,
            axes_step=list(axes_step),
            axes_origin=list(axes_origin),
            cache=cache,
        )

        self.thermal = True

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
                self.property_containers[region],
                thermal=True,
                extrapolation_flag=False,
            )
        self.well_operators = acc_flux_gravity_evaluator_python_well(
            self.property_containers[self.regions[0]]
        )

        self.well_ctrl_operators = WellCtrlOperators(
            self.property_containers[self.regions[0]],
            self.thermal,
            extrapolation_flag=False,
        )

        self.thermal_var_operator = ThermalVarOperator(
            self.property_containers[self.regions[0]],
            self.thermal,
            is_pt=(self.state_spec <= PhysicsBase.StateSpecification.PT),
            extrapolation_flag=False,
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
        return eval(f"engine_nce_g_{platform}{self.nc:d}_{self.nph:d}")()

    def set_initial_conditions_from_depth_table(
        self, mesh: conn_mesh, input_distribution: dict, input_depth: list | np.ndarray
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
        for _key, input_values in input_distribution.items():
            input_values = (
                input_values
                if not np.isscalar(input_values)
                else np.ones(len(input_depth)) * input_values
            )
            assert len(input_values) == len(input_depth)

        # Get depths and primary variable arrays from mesh object
        depths = np.asarray(mesh.depth)[: mesh.n_res_blocks]

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
                    f'Initial condition for variable {variable} has different length, resizing {len(values)} to {mesh.n_res_blocks}',
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
