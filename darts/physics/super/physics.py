import warnings
from collections.abc import Iterable

import numpy as np
from scipy.interpolate import interp1d

from darts.engines import *
from darts.physics.base.operators_base import (
    PropertyOperators,
    ThermalVarOperator,
    WellControlOperators,
)
from darts.physics.base.physics_base import HistoryField, PhysicsBase
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
        epsilon_z: float,
        sim_eps_multiplier: float = 10,
        extrapolation_flag: bool = True,
        min_t: float = None,
        max_t: float = None,
        state_spec: PhysicsBase.StateSpecification = PhysicsBase.StateSpecification.P,
        cache: bool = False,
        axes_min=None,
        axes_max=None,
        n_axes_points=None,
        history_fields: Iterable[HistoryField] | None = None,
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
        :param epsilon_z: Epsilon value for composition OBL axes (min_axis_z, max_axis_z)
        :type epsilon_z: float
        :param sim_eps_multiplier: Multiplier to epsilon_z to obtain sim_eps (minimum offset of solution state from
                                    OBL bounds, calculated as min_sim_z/max_sim_z in engine), default is 10
        :type sim_eps_multiplier: float
        :param extrapolation_flag: Switch to turn on extrapolation logic (z[last component] < 0 in case nc >= 3)
        :type extrapolation_flag: bool
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
        :param history_fields: (optional) List of :class:`HistoryField` descriptors declaring
                               auxiliary OBL axes (e.g. ``sg_max`` for Killough hysteresis).
                               Fields with ``n_axis_points=None`` inherit ``n_points``. Pass
                               ``None`` or an empty list for standard drainage-only behaviour.
        :type history_fields: Iterable[HistoryField] or None
        """
        # Define nc, nph and (iso)thermal
        nc = len(components)
        nph = len(phases)
        self.thermal = state_spec > PhysicsBase.StateSpecification.P

        # Define state variables and OBL axes: pressure, nc-1 components and possibly temperature/enthalpy
        variables = ["pressure"] + components[:-1]
        if self.thermal:
            variables += (
                ["temperature"]
                if state_spec == PhysicsBase.StateSpecification.PT
                else ["enthalpy"]
            )

        n_vars = len(variables)
        # Number of operators = NE /*acc*/ + NE * NP /*flux*/ + NP * /*density*/ + NP /*UPSAT*/ + NE * NP /*gradient*/ + NE /*kinetic*/
        # + 2 * NP /*gravpc*/ + 1 /*poro*/ + NP /*LAMBDA*/ + NP /*SAT*/ + NP /*enthalpy*/
        # + 2 /*temperature and pressure*/
        # = NE * (2 * nph + 2) + 6 * nph + 3

        n_ops = n_vars * (2 * nph + 2) + 7 * nph + 3

        # axes_min
        if axes_min is None:
            axz_min = (
                [min_z + epsilon_z for i in range(nc - 1)]
                if np.isscalar(min_z)
                else [min_z[i] + epsilon_z for i in range(nc - 1)]
            )
            if self.thermal:
                axes_min = [min_p] + axz_min + [min_t]
            else:
                axes_min = [min_p] + axz_min

        # axes_max
        if axes_max is None:
            axz_max = (
                [max_z - (nc - 1) * epsilon_z for i in range(nc - 1)]
                if np.isscalar(min_z)
                else [max_z[i] - (nc - 1) * epsilon_z for i in range(nc - 1)]
            )
            if self.thermal:
                axes_max = [max_p] + axz_max + [max_t]
            else:
                axes_max = [max_p] + axz_max

        # n_axes_points
        if n_axes_points is None:
            n_axes_points = index_vector([n_points] * n_vars)
        else:
            n_axes_points = index_vector(n_axes_points)

        self.extrapolation_flag = extrapolation_flag
        self.dz = (
            (axes_max[1] - axes_min[1]) / (n_axes_points[1] - 1) if nc > 1 else None
        )
        if self.extrapolation_flag:
            # ASSERT EQUAL DZ FOR EACH COMPOSITION AXIS
            for i in range(nc - 1):
                assert (
                    np.abs(
                        (axes_max[i + 1] - axes_min[i + 1]) / (n_axes_points[i + 1] - 1)
                        - self.dz
                    )
                    < 1e-15
                ), (
                    "To use extrapolation logic, dz should be equal along all compositional axes"
                )

        assert sim_eps_multiplier > 1, (
            "Multiplier for epsilon must be greater than 1 to have consistent "
            "OBL axes/solution vector in engine"
        )

        # Fill in per-field defaults (mostly n_axis_points falling back to n_points) so that
        # callers can pass HistoryField(label="sg_max") without repeating axis resolution.
        resolved_history_fields = [
            HistoryField(
                label=h.label,
                axis_min=h.axis_min,
                axis_max=h.axis_max,
                n_axis_points=(
                    h.n_axis_points if h.n_axis_points is not None else n_points
                ),
                default=h.default,
            )
            for h in (history_fields or [])
        ]

        # Call PhysicsBase constructor
        super().__init__(
            state_spec=state_spec,
            variables=variables,
            components=components,
            phases=phases,
            n_ops=n_ops,
            axes_min=axes_min,
            axes_max=axes_max,
            sim_eps=epsilon_z * sim_eps_multiplier,
            n_axes_points=n_axes_points,
            timer=timer,
            cache=cache,
            history_fields=resolved_history_fields,
        )

    def set_engine(self, discr_type: str = "tpfa", platform: str = "cpu"):
        """
        Function to set :class:`engine_super` object.

        :param discr_type: Type of discretization, 'tpfa' (default) or 'mpfa'
        :type discr_type: str
        :param platform: Switch for CPU/GPU engine, 'cpu' (default) or 'gpu'
        :type platform: str
        """
        if discr_type == "mpfa":
            if self.thermal:
                return eval(f"engine_super_mp_{platform}{self.nc:d}_{self.nph:d}_t")()
            else:
                return eval(f"engine_super_mp_{platform}{self.nc:d}_{self.nph:d}")()
        else:
            if self.thermal:
                return eval(f"engine_super_{platform}{self.nc:d}_{self.nph:d}_t")()
            else:
                return eval(f"engine_super_{platform}{self.nc:d}_{self.nph:d}")()

    def set_operators(self):
        """
        Function to set operator objects: :class:`ReservoirOperators` for each of the reservoir regions,
        :class:`WellOperators` for the well segments, :class:`WellControlOperators` for well control
        and a :class:`PropertyOperator` for the evaluation of properties.
        """
        for region in self.regions:
            self.reservoir_operators[region] = ReservoirOperators(
                self.property_containers[region],
                self.thermal,
                extrapolation_flag=self.extrapolation_flag,
                dz=self.dz,
            )
            self.property_operators[region] = PropertyOperators(
                self.property_containers[region],
                self.thermal,
                extrapolation_flag=self.extrapolation_flag,
                dz=self.dz,
            )

        self.well_operators = WellOperators(
            self.property_containers[self.regions[0]],
            self.thermal,
            extrapolation_flag=self.extrapolation_flag,
            dz=self.dz,
        )

        self.well_ctrl_operators = WellControlOperators(
            self.property_containers[self.regions[0]],
            self.thermal,
            extrapolation_flag=self.extrapolation_flag,
            dz=self.dz,
        )
        self.thermal_var_operator = ThermalVarOperator(
            self.property_containers[self.regions[0]],
            self.thermal,
            is_pt=(self.state_spec <= PhysicsBase.StateSpecification.PT),
            extrapolation_flag=self.extrapolation_flag,
            dz=self.dz,
        )

        return

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
        assert np.all(
            [
                variable in input_distribution.keys()
                for variable in self.vars[1 : self.nc]
            ]
        ), "Initial state for must be specified for all primary variables"
        assert not self.thermal or (
            "temperature" in input_distribution.keys()
            or "enthalpy" in input_distribution.keys()
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

        # adjust the size of initial_state array in c++
        mesh.initial_state.resize(mesh.n_res_blocks * self.n_vars)

        # Loop over variables to fill initial_state vector in c++
        for ith_var, variable in enumerate(self.vars):
            if variable == "enthalpy" and "enthalpy" not in input_distribution.keys():
                # If temperature has been provided, interpolate pressure and temperature to compute enthalpies
                p_itor = interp1d(
                    input_depth,
                    input_distribution["pressure"],
                    kind="linear",
                    fill_value="extrapolate",
                )
                pressure = p_itor(depths)

                t_itor = interp1d(
                    input_depth,
                    input_distribution["temperature"],
                    kind="linear",
                    fill_value="extrapolate",
                )
                temperature = t_itor(depths)

                z_itors = [
                    interp1d(
                        input_depth,
                        input_distribution[comp],
                        kind="linear",
                        fill_value="extrapolate",
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
                    kind="linear",
                    fill_value="extrapolate",
                )
                values = itor(depths)

            values = np.resize(np.asarray(values), mesh.n_res_blocks)
            np.asarray(mesh.initial_state)[ith_var :: self.n_vars] = values

        self.populate_mesh_history_defaults(mesh)

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
            "pressure"
        ]

        # if thermal, set initial temperature or enthalpy
        if self.thermal:
            if self.state_spec == PhysicsBase.StateSpecification.PT:
                np.asarray(mesh.initial_state)[(self.n_vars - 1) :: self.n_vars] = (
                    input_distribution["temperature"]
                )
            elif self.state_spec == PhysicsBase.StateSpecification.PH:
                if 'enthalpy' in input_distribution.keys():
                    enthalpy = input_distribution["enthalpy"]
                elif 'temperature' in input_distribution.keys():
                    # interpolate pressure and temperature to compute enthalpies
                    enthalpy = np.empty(mesh.n_res_blocks)
                    if not np.isscalar(input_distribution['pressure']):
                        # Pressure specified as an array
                        for j in range(mesh.n_res_blocks):
                            composition = [
                                (
                                    input_distribution[component][j]
                                    if not np.isscalar(input_distribution[component])
                                    else input_distribution[component]
                                )
                                for component in self.property_containers[
                                    0
                                ].components_name[:-1]
                            ]
                            temp = (
                                input_distribution['temperature'][j]
                                if not np.isscalar(input_distribution['temperature'])
                                else input_distribution['temperature']
                            )

                            state = np.array(
                                [input_distribution['pressure'][j]]
                                + composition
                                + [temp]
                            )
                            enthalpy[j] = self.property_containers[
                                0
                            ].compute_total_enthalpy(state)
                    else:
                        composition = [
                            input_distribution[component]
                            for component in self.property_containers[
                                0
                            ].components_name[:-1]
                        ]
                        state = value_vector(
                            [input_distribution['pressure']]
                            + composition
                            + [input_distribution['temperature']]
                        )  # enthalpy is dummy variable
                        enth = self.property_containers[0].compute_total_enthalpy(state)
                        enthalpy[:] = enth
                else:
                    raise KeyError(
                        'Initial state must specify either temperature or enthalpy, but neither was provided!'
                    )

                np.asarray(mesh.initial_state)[(self.n_vars - 1) :: self.n_vars] = (
                    enthalpy
                )

        # set initial composition
        for c in range(self.nc - 1):
            np.asarray(mesh.initial_state)[(c + 1) :: self.n_vars] = (
                input_distribution[self.vars[c + 1]]
                if np.isscalar(input_distribution[self.vars[c + 1]])
                else input_distribution[self.vars[c + 1]][:]
            )

        # Broadcast HistoryField.default values into mesh.Xhis_bounds so boundary cells
        # (MPFA / mech engines with n_bounds > 0) start from the configured default instead
        # of the engine's zero fallback in build_Xop.
        self.populate_mesh_history_defaults(mesh)

    def evaluate_flash(
        self,
        state_spec: dict = None,
        compositions: dict = None,
        obl_interval_multiplier: float = 1.0,
        plot_flash_results: bool = False,
        region: int = 0,
    ):
        """
        Method to evaluate and plot flash for specified states and compositions.

        :param state_spec: Dictionary containing values for state specification variables, default is None which evaluates full OBL space
        :type state_spec: dict
        :param compositions: Dictionary containing composition ranges, default is None which evaluates full OBL space
        :type compositions: dict
        :param obl_interval_multiplier: Multiplier to OBL axis intervals if ranges are obtained from OBL axes, default 1
        :type obl_interval_multiplier: float
        :param plot_flash_results: Whether to plot flash results in phase diagram
        :type plot_flash_results: bool
        :param region: Property region
        :type region: int
        """
        from dartsflash.dartsflash import DARTSFlash

        flash_ev = self.property_containers[region].flash_ev
        assert isinstance(flash_ev, DARTSFlash), (
            "Flash evaluator should be DARTSFlash object to utilize this feature"
        )

        # Set ranges of state specification
        state_vars = [self.vars[0], self.vars[-1]] if self.thermal else [self.vars[0]]
        state_spec = state_spec if state_spec is not None else {}
        for i, spec in enumerate(state_vars):
            # create entry if it doesn't exist
            state_spec[spec] = state_spec[spec] if spec in state_spec.keys() else None
            # define array if it hasn't been defined
            spec_idx = 0 if i == 0 else -1
            state_spec[spec] = (
                state_spec[spec]
                if state_spec[spec] is not None
                else (
                    np.linspace(
                        self.axes_min[spec_idx],
                        self.axes_max[spec_idx],
                        int(self.n_axes_points[spec_idx] / obl_interval_multiplier),
                    )
                )
            )

        # Show warning if other variable has been specified
        if not np.all([spec in state_vars for spec in state_spec.keys()]):
            import warnings

            warnings.warn(
                "Not all specified variables in state_spec are primary variables",
                stacklevel=2,
            )
            print("Variables", state_vars, "State specifications", state_spec.keys())

        # Set ranges of compositions
        compositions = compositions if compositions is not None else {}
        compositions[self.components[-1]] = 1.0
        for i, comp in enumerate(self.components[:-1]):
            # create entry if it doesn't exist
            compositions[comp] = (
                compositions[comp] if comp in compositions.keys() else None
            )
            # define array if it hasn't been defined
            compositions[comp] = (
                compositions[comp]
                if compositions[comp] is not None
                else (
                    np.linspace(
                        self.axes_min[i + 1],
                        self.axes_max[i + 1],
                        int(self.n_axes_points[i + 1] / obl_interval_multiplier),
                    )
                )
            )

        # Evaluate
        _flash_results = flash_ev.evaluate_flash(
            state_spec=state_spec, compositions=compositions, mole_fractions=True
        )

        # Plot
        if plot_flash_results:
            pass
