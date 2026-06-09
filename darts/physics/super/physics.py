import warnings
from collections.abc import Iterable

import numpy as np
from scipy.interpolate import interp1d

from darts.engines import *
from darts.physics.base.operators_base import (
    PropertyOperators,
    ThermalVarOperator,
    WellCtrlOperators,
)
from darts.physics.base.physics_base import HistoryField, PhysicsBase
from darts.physics.super.operator_evaluator import ReservoirOperators, WellOperators


class Compositional(PhysicsBase):
    """
    Physics class for compositional simulation.

    Creates reservoir, well, rate and property operators and interpolators for
    P-z or P-T-z compositional simulation; initializes the :class:`super_engine`;
    sets well controls; and defines initial / boundary conditions.

    The OBL grid is defined by ``axes_step`` (per-axis cell size) and an optional
    ``axes_origin`` (default zeros, with ``epsilon_z`` added on composition axes).
    The adaptive multi-index-keyed interpolator caches cells on demand wherever
    the solver lands; there is no fixed grid window.
    """

    def __init__(
        self,
        components: list,
        phases: list,
        timer: timer_node,
        axes_step: list[float],
        axes_origin: list[float] = None,
        epsilon_z: float = 1e-9,
        sim_eps_multiplier: float = 10,
        extrapolation_flag: bool = True,
        state_spec: PhysicsBase.StateSpecification = PhysicsBase.StateSpecification.P,
        cache: bool = False,
        history_fields: Iterable[HistoryField] | None = None,
    ):
        """
        :param components: List of components.
        :param phases: List of phases.
        :param timer: Timer object.
        :param axes_step: Per-axis cell size [p_step, z_step_1, ..., z_step_{nc-1}, t_step?].
            For ``extrapolation_flag=True`` the composition steps must all be equal.
        :param axes_origin: Per-axis grid origin. Defaults match the open-DARTS unit
            conventions: pressure = 1 bar, compositions = ``epsilon_z`` (composition
            floor, avoids the simplex boundary), thermal axis = 273.15 K for
            ``state_spec=PT`` or 0 for ``state_spec=PH`` (enthalpy reference depends on
            the EOS — pass an explicit ``axes_origin`` to override).
        :param epsilon_z: Composition-axis offset (default 1e-9).
        :param sim_eps_multiplier: Multiplier on ``epsilon_z`` to obtain ``sim_eps``.
        :param extrapolation_flag: Enable extrapolation logic (z[last] < 0 if nc >= 3).
        :param state_spec: P (default), PT, or PH.
        :param cache: Cache supporting points to disk between runs.
        :param history_fields: (optional) List of :class:`HistoryField` descriptors declaring
                               auxiliary OBL axes (e.g. ``sg_max`` for Killough hysteresis).
                               Pass ``None`` or an empty list for standard drainage-only
                               behaviour.
        """
        nc = len(components)
        nph = len(phases)
        self.thermal = state_spec > PhysicsBase.StateSpecification.P

        # State variables: pressure, nc-1 components, optional thermal var.
        variables = ["pressure"] + components[:-1]
        if self.thermal:
            variables += (
                ["temperature"]
                if state_spec == PhysicsBase.StateSpecification.PT
                else ["enthalpy"]
            )

        n_vars = len(variables)
        # NE * (2 * nph + 2) + 7 * nph + 3
        n_ops = n_vars * (2 * nph + 2) + 7 * nph + 3

        assert len(axes_step) == n_vars, (
            f"axes_step must have {n_vars} entries, got {len(axes_step)}"
        )
        assert sim_eps_multiplier > 1, (
            "sim_eps_multiplier must be > 1 for consistent OBL axes / solution clipping"
        )

        if axes_origin is None:
            # Sensible defaults matching open-DARTS unit conventions:
            #   pressure        → 1 bar
            #   compositions    → epsilon_z (composition-axis floor)
            #   thermal axis (when state_spec > P):
            #     PT → 273.15 K (0 °C, conventional standard temperature)
            #     PH → 0        (enthalpy reference is EOS-specific; override per case)
            axes_origin = [1.0] + [epsilon_z] * (nc - 1)
            if self.thermal:
                if state_spec == PhysicsBase.StateSpecification.PT:
                    axes_origin.append(273.15)
                else:
                    axes_origin.append(0.0)
        assert len(axes_origin) == n_vars

        self.extrapolation_flag = extrapolation_flag
        self.dz = axes_step[1] if nc > 1 else None
        if extrapolation_flag and nc > 1:
            for i in range(nc - 1):
                assert abs(axes_step[1 + i] - self.dz) < 1e-15, (
                    "extrapolation requires equal dz across all composition axes"
                )

        # Fill in per-field defaults (mostly n_axis_points falling back to n_points) so that
        # callers can pass HistoryField(label="sg_max") without repeating axis resolution.
        # n_points isn't available with the multi-index adaptive grid; HistoryField specs
        # must set n_axis_points explicitly, so we just pass entries through.
        resolved_history_fields = list(history_fields or [])
        super().__init__(
            state_spec=state_spec,
            variables=variables,
            components=components,
            phases=phases,
            n_ops=n_ops,
            timer=timer,
            axes_step=list(axes_step),
            axes_origin=list(axes_origin),
            sim_eps=epsilon_z * sim_eps_multiplier,
            cache=cache,
            history_fields=resolved_history_fields,
        )

    def set_engine(self, discr_type: str = "tpfa", platform: str = "cpu"):
        """
        :class:`engine_super` factory.

        :param discr_type: 'tpfa' (default) or 'mpfa'.
        :param platform: 'cpu' (default) or 'gpu'.
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
        Set operator objects: :class:`ReservoirOperators` per region, :class:`WellOperators`
        for well segments, :class:`WellCtrlOperators` for well controls, and a
        :class:`PropertyOperator` for property evaluation.
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

        self.well_ctrl_operators = WellCtrlOperators(
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

        # Broadcast HistoryField.default values into mesh.Xhistory_bounds so boundary cells
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

        # Number of base sampling points per axis for the flash pre-evaluation sweep.
        # This only controls how finely the flash is pre-tabulated over the OBL axes; it
        # is independent of the (now unbounded) OBL grid. obl_interval_multiplier coarsens
        # the sweep to match the user's OBL step multiplier.
        flash_sweep_n = 1024

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
                    np.arange(flash_sweep_n // obl_interval_multiplier)
                    * (self.axes_step[spec_idx] * obl_interval_multiplier)
                    + self.axes_origin[spec_idx]
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
                    np.arange(flash_sweep_n // obl_interval_multiplier)
                    * (self.axes_step[i + 1] * obl_interval_multiplier)
                    + self.axes_origin[i + 1]
                )
            )

        # Evaluate
        _flash_results = flash_ev.evaluate_flash(
            state_spec=state_spec, compositions=compositions, mole_fractions=True
        )

        # Plot
        if plot_flash_results:
            pass
