import abc
import atexit
import hashlib
import os
import pickle
import signal
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from functools import total_ordering

import numpy as np

from darts.engines import *
from darts.interpolators import *
from darts.physics.base.operators_base import ThermalVarOperator, WellControlOperators


@dataclass
class HistoryField:
    """Declarative description of one OBL history variable.

    History variables enter the OBL interpolator state but are NOT Newton unknowns — the physics
    advances them outside of Newton (e.g. max gas saturation updated after each converged timestep
    for Killough relative-permeability hysteresis).

    :param label: Axis label used for interpolator state ordering (e.g. ``"sg_max"``)
    :param axis_min: Lower bound of the OBL axis for this history variable
    :param axis_max: Upper bound of the OBL axis for this history variable
    :param n_axis_points: Number of OBL supporting points on this axis; ``None`` falls back to the
                          physics default ``n_points``
    :param default: Reservoir initial value and fallback value at wells / boundaries
    """

    label: str
    axis_min: float = 0.0
    axis_max: float = 1.0
    n_axis_points: int | None = None
    default: float = 0.0


class PhysicsBase:
    """
    This is a base class for Physics definition.

    Physics contains all necessary objects to initialize and run the DARTS :class:`engine`.

    The Physics object is composed of :class:`PropertyContainer` objects for each of the regions and a set of operators.
    The operators consist of :class:`ReservoirOperators` objects for each of the regions, a :class:`WellOperators`,
    a :class:`WellControlOperators`, a :class:`ThermalVarOperator` and a :class:`PropertyOperators` object.
    For each set of operators (evaluators, etor), an interpolator (itor) object is created for use in the :class:`engine`.

    :ivar engine: Engine object
    :type engine: :class:`engine_base`
    :ivar property_containers: Set of :class:`PropertyContainer` objects for each of the regions for evaluation of reservoir cell properties
    :type property_containers: dict
    :ivar reservoir_operators: Set of :class:`ReservoirOperators` objects for each of the regions for evaluation of reservoir cell states
    :type reservoir_operators: dict
    :ivar property_operators: :class:`PropertyOperators` object for evaluation and interpolation of properties
    :type property_operators: dict
    :ivar well_operators: :class:`WellOperators` object for evaluation of well cell states
    :type well_operators: dict
    :ivar well_ctrl_operators: :class:`WellControlOperators` object for well control
    :type well_ctrl_operators: WellControlOperators
    :ivar thermal_var_operator: :class:`ThermalVarOperator` object for generic state specification
    :type thermal_var_operator: ThermalVarOperator
    :ivar regions: List of property regions
    :type regions: list
    """

    engine: engine_base
    well_operators: operator_set_evaluator_iface
    well_ctrl_operators: WellControlOperators
    thermal_var_operator: ThermalVarOperator

    @total_ordering
    class StateSpecification(Enum):
        P = 0
        PT = 1
        PH = 2
        PS = 3

        def __lt__(self, other):
            if self.__class__ is other.__class__:
                return self.value < other.value
            return NotImplemented

    def __init__(
        self,
        state_spec: StateSpecification,
        variables: list,
        components: list,
        phases: list,
        n_ops: int,
        axes_min: value_vector,
        axes_max: value_vector,
        n_axes_points: index_vector,
        timer: timer_node,
        sim_eps: float = None,
        cache: bool = False,
        history_fields: Iterable['HistoryField'] | None = None,
    ):
        """
        This is the constructor of the PhysicsBase class. It creates a `simulation` timer node and initializes caching.

        :param state_spec: State specification - 0) P, 1) PT, 2) PH
        :type state_spec: StateSpecification
        :param variables: List of independent variables
        :type variables: list
        :param components: Components
        :type components: list
        :param phases: List of phases
        :type phases: list
        :param n_ops: Number of operators
        :type n_ops: int
        :param axes_min, axes_max: Minimum, maximum of each OBL axis
        :type axes_min, axes_max: :class:`darts.interpolators.value_vector`
        :param n_axes_points: Number of OBL points along axes
        :type n_axes_points: index_vector
        :param timer: Timer object
        :param sim_eps: Epsilon composition for simulation that solution should remain away from OBL bounds
                        (in engine, min_sim_z = min_axis_z + sim_eps, max_sim_z = max_axis_z - sim_eps)
        :type sim_eps: float
        :type cache: :class:`darts.interpolators.timer_node`
        :param cache: Switch to cache operator values
        :type cache: bool
        :param history_fields: Optional list of :class:`HistoryField` descriptors. Each entry
                               declares one auxiliary OBL axis (e.g. ``sg_max`` for Killough
                               hysteresis) that is fed into operator interpolation but is NOT
                               a Newton unknown. Pass ``None`` or an empty list to disable
                               history-aware behaviour entirely (matches legacy flow physics).
        :type history_fields: Iterable[HistoryField] or None
        """
        # Define variables and number of operators
        self.state_spec = state_spec
        self.is_ph = state_spec > PhysicsBase.StateSpecification.PT
        self.vars = variables
        self.n_vars = len(variables)

        self.components = components
        self.nc = len(components)
        self.thermal = self.n_vars - self.nc
        self.phases = phases
        self.nph = len(phases)
        self.n_ops = n_ops

        # Define PTz bounds for OBL grid
        self.PT_axes_min = axes_min
        self.PT_axes_max = axes_max
        self.n_axes_points = n_axes_points
        self.sim_eps = sim_eps if sim_eps is not None else 1e-12

        # Initialize timer for simulation and caching
        self.timer = timer.node["simulation"]
        self.cache = cache
        # list of created interpolators
        # is used on destruction to save cache data
        if self.cache:
            self.created_itors = []
            atexit.register(self.write_cache)

        self.regions = []
        self.property_containers = {}
        self.reservoir_operators = {}
        self.property_operators = {}

        # Optional OBL history variables (e.g. max gas saturation for Killough hysteresis).
        # An empty list disables history-aware behaviour; a non-empty list extends the OBL
        # interpolation state without touching the Newton system.
        self.history_fields: list[HistoryField] = list(history_fields or [])
        # Populated by set_interpolators when history_fields is non-empty, so create_interpolator
        # can decide between primary-axes and extended-axes interpolator construction.
        self._extended_axes_min: value_vector | None = None
        self._extended_axes_max: value_vector | None = None
        self._extended_n_axes_points: index_vector | None = None

    @property
    def n_his(self) -> int:
        """Number of OBL history variables configured (``len(history_fields)``).

        :returns: Count of auxiliary OBL axes that enter interpolation but not the Newton system
        :rtype: int
        """
        return len(self.history_fields)

    @property
    def n_state(self) -> int:
        """Total OBL interpolation-state size per cell = ``n_vars + n_his``.

        :returns: Number of axes the reservoir / well interpolators consume per cell
        :rtype: int
        """
        return self.n_vars + self.n_his

    def get_interpolator_axes(
        self,
    ) -> tuple[value_vector, value_vector, index_vector]:
        """Return the ``(axes_min, axes_max, n_axes_points)`` triple that OBL interpolators
        running on the full OBL state ``[X | Xhis]`` should be built on.

        When ``history_fields`` is empty this is identical to ``(self.axes_min, self.axes_max,
        self.n_axes_points)``. When non-empty the primary axes are extended with one axis per
        history field (using ``n_axes_points[0]`` as the default axis resolution). The result
        is cached on ``self._extended_axes_*`` so callers (``set_interpolators`` and
        :mod:`darts.output`) see identical bounds.

        :returns: ``(axes_min, axes_max, n_axes_points)`` suitable for :meth:`create_interpolator`
        :rtype: tuple[value_vector, value_vector, index_vector]
        """
        if not self.history_fields:
            return self.axes_min, self.axes_max, self.n_axes_points
        if self._extended_axes_min is None:
            h_min = [h.axis_min for h in self.history_fields]
            h_max = [h.axis_max for h in self.history_fields]
            h_npts = [
                (
                    h.n_axis_points
                    if h.n_axis_points is not None
                    else self.n_axes_points[0]
                )
                for h in self.history_fields
            ]
            self._extended_axes_min = value_vector(list(self.axes_min) + h_min)
            self._extended_axes_max = value_vector(list(self.axes_max) + h_max)
            self._extended_n_axes_points = index_vector(
                list(self.n_axes_points) + h_npts
            )
        return (
            self._extended_axes_min,
            self._extended_axes_max,
            self._extended_n_axes_points,
        )

    def get_interpolator_state_labels(self) -> list:
        """Return axis labels used by the OBL interpolators, in storage order.

        The first ``n_vars`` entries are the primary Newton unknown labels (``self.vars``),
        followed by one label per history field (e.g. ``"sg_max"``). Used by
        :mod:`darts.output` to label columns when dumping operator state.

        :returns: Ordered list of axis labels of length ``n_state``
        :rtype: list[str]
        """
        return list(self.vars) + [h.label for h in self.history_fields]

    def get_history_default(self, label: str) -> float:
        """Return the configured initial / fallback value for a history field.

        :param label: Label of the history field, must match one declared in ``history_fields``
        :type label: str
        :returns: The ``default`` attribute of the matching :class:`HistoryField`
        :rtype: float
        :raises KeyError: If no history field has the requested label
        """
        for h in self.history_fields:
            if h.label == label:
                return h.default
        raise KeyError(label)

    def get_engine_history_array(self, label: str, n_blocks: int = None) -> np.ndarray:
        """Return a per-cell copy of ``engine.Xhis`` restricted to one history axis.

        The engine stores ``Xhis`` as a flat ``[(n_blocks + n_bounds) * n_his]`` buffer in
        cell-major order (all ``n_his`` values for cell 0, then cell 1, ...). This helper
        pulls out just the reservoir blocks for one label and returns a copy (safe to mutate).

        :param label: Label of the history field to extract, must match one in ``history_fields``
        :type label: str
        :param n_blocks: Number of reservoir blocks to read. When ``None``, inferred as
                         ``Xhis.size // n_his`` (i.e. all cells including boundaries)
        :type n_blocks: int, optional
        :returns: One-dimensional array of shape ``(n_blocks,)`` with the requested axis values
        :rtype: numpy.ndarray
        :raises RuntimeError: If no history fields are configured on this physics
        :raises KeyError: If no history field has the requested label
        """
        if not self.history_fields:
            raise RuntimeError("Physics has no history fields configured")
        idx = next(
            (i for i, h in enumerate(self.history_fields) if h.label == label), -1
        )
        if idx < 0:
            raise KeyError(label)
        Xhis = np.asarray(self.engine.Xhis, copy=False)
        n_his = self.n_his
        if n_blocks is None:
            n_blocks = Xhis.size // n_his
        return Xhis.reshape(-1, n_his)[:n_blocks, idx].copy()

    def get_engine_interpolator_state(self, n_blocks: int = None) -> np.ndarray:
        """Return the full OBL state ``[X | Xhis]`` flattened in cell-major order.

        Used by :mod:`darts.output` to dump operator inputs for post-processing. The layout is
        interleaved so callers that stride by ``n_state`` pick out one state variable per cell:
        ``result[j::n_state]`` is the ``j``-th state axis for every reservoir cell.

        :param n_blocks: Number of reservoir blocks. When ``None``, inferred from
                         ``engine.X.size // n_vars``
        :type n_blocks: int, optional
        :returns: One-dimensional array of length ``n_blocks * n_state`` with primary vars and
                  history values interleaved per cell
        :rtype: numpy.ndarray
        """
        if n_blocks is None:
            n_blocks = self.engine.X.size // self.n_vars
        X = np.asarray(self.engine.X, copy=False).reshape(-1, self.n_vars)[:n_blocks]
        if not self.history_fields:
            return X.flatten()
        Xhis = np.asarray(self.engine.Xhis, copy=False).reshape(-1, self.n_his)[
            :n_blocks
        ]
        return np.concatenate([X, Xhis], axis=1).flatten()

    def set_engine_history_array(
        self, label: str, values, n_blocks: int = None
    ) -> None:
        """Overwrite one axis of ``engine.Xhis`` with a per-cell scalar or array.

        Writes go through a full round-trip copy because ``engine.Xhis`` is exposed to Python
        as a ``value_vector`` that does not support strided in-place assignment.

        :param label: Label of the history field to write, must match one in ``history_fields``
        :type label: str
        :param values: Scalar applied to every cell, or array of length ``n_blocks``
        :type values: float or numpy.ndarray
        :param n_blocks: Number of reservoir blocks to write. When ``None``, inferred as
                         ``Xhis.size // n_his``
        :type n_blocks: int, optional
        :returns: None
        :raises RuntimeError: If no history fields are configured on this physics
        :raises KeyError: If no history field has the requested label
        """
        if not self.history_fields:
            raise RuntimeError("Physics has no history fields configured")
        idx = next(
            (i for i, h in enumerate(self.history_fields) if h.label == label), -1
        )
        if idx < 0:
            raise KeyError(label)
        n_his = self.n_his
        Xhis_flat = np.asarray(self.engine.Xhis, copy=True)
        if n_blocks is None:
            n_blocks = Xhis_flat.size // n_his
        if np.isscalar(values):
            values = np.full(n_blocks, float(values))
        values = np.asarray(values, dtype=float)
        Xhis_view = Xhis_flat.reshape(-1, n_his)
        Xhis_view[:n_blocks, idx] = values
        self.engine.Xhis = value_vector(Xhis_flat.tolist())

    def init_physics(
        self,
        discr_type: str = 'tpfa',
        platform: str = 'cpu',
        itor_type: str = 'multilinear',
        itor_mode: str = 'adaptive',
        itor_precision: str = 'd',
        verbose: bool = False,
        is_barycentric: bool = False,
        n_solid: int = None,
    ):
        """
        Function to initialize all contained objects within the Physics object.

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
        :param is_barycentric: Flag which turn on barycentric interpolation on Delaunay simplices
        :type is_barycentric: bool
        :param n_solid: Number of solid minerals for element-based reactive flow
        :type n_solid: int
        """
        # Define OBL axes
        self.axes_min, self.axes_max = self.determine_obl_bounds(
            min_p=self.PT_axes_min[0],
            max_p=self.PT_axes_max[0],
            min_t=self.PT_axes_min[-1],
            max_t=self.PT_axes_max[-1],
            min_z=self.PT_axes_min[1 : self.nc],
            max_z=self.PT_axes_max[1 : self.nc],
            state_spec=self.state_spec,
        )

        # set engine, operators and create interpolators
        self.engine = self.set_engine(discr_type, platform)

        # Tell the engine how many per-cell history variables to reserve in its Xop / Xhis
        # buffers. Must be set before engine.init() allocates them.
        if hasattr(self.engine, "n_his_runtime"):
            self.engine.n_his_runtime = self.n_his

        # for separate mineral fraction in reactive flow formulations
        if n_solid is not None:
            self.engine.n_solid = n_solid

        # Set state specification in the engine
        self.set_state_spec(state_spec=self.state_spec)

        self.set_operators()
        self.set_interpolators(
            platform, itor_type, itor_mode, itor_precision, is_barycentric
        )
        return

    def set_state_spec(self, state_spec: StateSpecification):
        """
        Set the state specification in the engine

        :param state_spec: State specification
        :type state_spec: StateSpecification
        """
        if state_spec == self.StateSpecification.P:
            self.engine.state_spec = self.engine.StateSpecification.P
        elif state_spec == self.StateSpecification.PT:
            self.engine.state_spec = self.engine.StateSpecification.PT
        elif state_spec == self.StateSpecification.PH:
            self.engine.state_spec = self.engine.StateSpecification.PH
        elif state_spec == self.StateSpecification.PS:
            self.engine.state_spec = self.engine.StateSpecification.PS
        else:
            raise NotImplementedError()

    def add_property_region(self, property_container, region: int = 0):
        """
        Function to add :class:`PropertyContainer` object for specified region to `property_containers` dict.

        :param property_container: Object for evaluation of properties
        :type property_container: :class:`PropertyContainer`
        :param region: Tag of the region, to be used as a key in `property_containers` dict
        """
        # Tell the property container how many OBL history variables the physics appends
        # to the state vector so that it can locate primary vars correctly (e.g. temperature
        # at position [nc] rather than [-1] when sg_max is appended), and pass the ordered
        # labels so it can expose {label: value} to history-aware evaluators.
        if hasattr(property_container, "n_his"):
            property_container.n_his = self.n_his
        if hasattr(property_container, "history_labels"):
            property_container.history_labels = [h.label for h in self.history_fields]
        self.property_containers[region] = property_container
        self.regions.append(region)
        return

    def set_operators(self):
        """
        Function to set operator objects: :class:`ReservoirOperators` for each of the reservoir regions,
        :class:`WellOperators` for the well cells, :class:`RateOperators` for evaluation of rates
        and a :class:`PropertyOperator` for the evaluation of properties.

        In PhysicsBase, this is an empty function, needs to be overloaded in child classes.
        """
        pass

    @abc.abstractmethod
    def set_engine(
        self, discr_type: str = 'tpfa', platform: str = 'cpu'
    ) -> engine_base:
        """
        Function to set :class:`engine` object.

        In PhysicsBase, this is an empty function, needs to be overloaded in child classes.

        :param discr_type: Type of discretization, 'tpfa' (default) or 'mpfa'
        :type discr_type: str
        :param platform: Switch for CPU/GPU engine, 'cpu' (default) or 'gpu'
        :type platform: str
        :returns: :class:`Engine` object
        """
        pass

    def set_interpolators(
        self,
        platform='cpu',
        itor_type='multilinear',
        itor_mode='adaptive',
        itor_precision='d',
        is_barycentric: bool = False,
    ):
        """
        Function to initialize set interpolator objects based on the set of operators.
        It creates timers for each of the interpolators.

        If history_fields is non-empty, the reservoir / well / well-control interpolators are built
        on the extended axes set (primary OBL axes + one axis per history field); the thermal_var
        interpolator always stays on the primary PT axes.

        :param platform: Switch for CPU/GPU engine, 'cpu' (default) or 'gpu'
        :type platform: str
        :param itor_type: Type of interpolation method, 'multilinear' (default) or 'linear'
        :type itor_type: str
        :param itor_mode: Mode of interpolation, 'adaptive' (default) or 'static'
        :type itor_mode: str
        :param itor_precision: Precision of interpolation, 'd' (default) - double precision or 's' - single precision
        :type itor_precision: str
        :param is_barycentric: Flag which turn on barycentric interpolation on Delaunay simplices
        :type is_barycentric: bool
        """
        # When history fields are configured, the reservoir / well / well-control interpolators
        # run on the extended axes set (primary OBL axes + one axis per history field).
        # get_interpolator_axes caches the extension on self._extended_axes_* so the output path
        # (Output.set_phase_properties) can reuse the same bounds.
        acc_axes_min, acc_axes_max, acc_n_pts = self.get_interpolator_axes()

        self.acc_flux_itor = {}
        self.property_itor = {}
        for region in self.regions:
            self.acc_flux_itor[region], _ = self.create_interpolator(
                self.reservoir_operators[region],
                n_ops=self.n_ops,
                axes_min=acc_axes_min,
                axes_max=acc_axes_max,
                n_axes_points=acc_n_pts,
                platform=platform,
                algorithm=itor_type,
                mode=itor_mode,
                precision=itor_precision,
                timer_name=f'reservoir {region:d} interpolation',
                region=str(region),
                is_barycentric=is_barycentric,
            )

            self.property_itor[region], _ = self.create_interpolator(
                self.property_operators[region],
                n_ops=self.n_ops,
                axes_min=acc_axes_min,
                axes_max=acc_axes_max,
                n_axes_points=acc_n_pts,
                platform=platform,
                algorithm=itor_type,
                mode=itor_mode,
                precision=itor_precision,
                timer_name=f'property {region:d} interpolation',
                region=str(region),
                is_barycentric=is_barycentric,
            )

        self.acc_flux_w_itor, _ = self.create_interpolator(
            self.well_operators,
            n_ops=self.n_ops,
            axes_min=acc_axes_min,
            axes_max=acc_axes_max,
            n_axes_points=acc_n_pts,
            timer_name='well interpolation',
            platform=platform,
            algorithm=itor_type,
            mode=itor_mode,
            precision=itor_precision,
            region='-1',
            is_barycentric=is_barycentric,
        )

        self.well_ctrl_itor, _ = self.create_interpolator(
            self.well_ctrl_operators,
            n_ops=self.well_ctrl_operators.n_ops,
            axes_min=acc_axes_min,
            axes_max=acc_axes_max,
            n_axes_points=acc_n_pts,
            timer_name='well controls interpolation',
            platform=platform,
            algorithm=itor_type,
            mode=itor_mode,
            precision=itor_precision,
            is_barycentric=is_barycentric,
        )
        self.thermal_var_itor, _ = self.create_interpolator(
            self.thermal_var_operator,
            n_ops=self.thermal_var_operator.n_ops,
            axes_min=value_vector(self.PT_axes_min),
            axes_max=value_vector(self.PT_axes_max),
            n_axes_points=self.n_axes_points,
            timer_name='well initialization',
            platform=platform,
            algorithm=itor_type,
            mode=itor_mode,
            precision=itor_precision,
            is_barycentric=is_barycentric,
        )
        return

    def evaluate_interpolators(
        self,
        itor: operator_set_evaluator_iface,
        etor: operator_set_evaluator_iface,
        states,
    ):
        # Create values, dvalues and idxs arrays
        n_states = len(states)
        physical_points = np.where(np.sum(states[:, 1:-1], axis=1) <= 1.0, True, False)
        states = value_vector(
            np.stack([states[:, j] for j in range(self.n_vars)]).T.flatten()
        )
        values = value_vector(np.zeros(self.n_ops * n_states))
        values_numpy = np.array(values, copy=False)
        dvalues = value_vector(np.zeros(self.n_ops * n_states * self.n_vars))

        idxs = index_vector([i for i in range(n_states)])

        # Interpolate operators
        itor.evaluate_with_derivatives(states, idxs, values, dvalues)

        # Fill operator array
        operator_array = {}
        for i in range(self.n_ops):
            op_type_idx = np.flatnonzero(
                [i >= np.array([tup[0] for tup in etor.op_names])]
            )[-1]
            op_name = (
                etor.op_names[op_type_idx][1]
                + "_"
                + str(i - etor.op_names[op_type_idx][0])
            )
            operator_array[op_name] = values_numpy[i :: self.n_ops]
            operator_array[op_name][~physical_points] = np.nan

        return operator_array

    def set_well_controls(
        self,
        wctrl: well_control_iface,
        control_type: well_control_iface.WellControlType,
        is_inj: bool,
        target: float,
        phase_name: str = None,
        inj_composition: list = None,
        inj_temp: float = None,
    ):
        """
        Set well control/constraint. It will call set_bhp_control() or set_rate_control() on the control or constraint
        well_control_iface object that lives in ms_well. In order to deactivate a control or constraint, pass WellControlType.NONE.

        :param wctrl: well_control_iface object responsible for control/constraint. It must be set to:
                      - well_obj.control for well control
                      - well_obj.constraint for well constraint
        :param control_type: Well control type -2) NONE (if constraint needs to be deactivated), -1) BHP,
                             0) MOLAR_RATE, 1) MASS_RATE, 2) VOLUMETRIC_RATE, 3) ADVECTIVE_HEAT_RATE; default is BHP
        :param is_inj: Is injection well (true) or production well (false)
        :param target: Target BHP or rate, consistent with well control type
        :param phase_name: Name of the phase rate of which is controlled. This input is required if well control is of the rate type.
        :param inj_composition: Composition of the injected phase. This input is required if it is an injection well.
        :param inj_temp: Temperature of the injected phase. This input is required if it is an injection well.
        """
        # Define well controls specification: BHP/rate, injected fluid composition, and injected fluid temperature
        inj_composition = (
            value_vector(inj_composition)
            if inj_composition is not None
            else value_vector(np.zeros(self.nc - 1))
        )  # for BHP controlled production well, pass dummy variables
        inj_temp = (
            inj_temp if inj_temp is not None else 0.0
        )  # for isothermal case or production well, pass dummy variables
        phase_idx = (
            self.phases.index(phase_name) if phase_name is not None else 0
        )  # for BHP controlled production well, pass dummy variables

        # Pass controls specification to ms_well object
        if control_type == well_control_iface.BHP:
            wctrl.set_bhp_control(is_inj, target, inj_composition, inj_temp)
        else:
            # Injection/production rate
            target = (
                np.abs(target) if is_inj else -np.abs(target)
            )  # + for inj, - for prod
            wctrl.set_rate_control(
                is_inj, control_type, phase_idx, target, inj_composition, inj_temp
            )

        return

    def determine_obl_bounds(
        self,
        min_p: float,
        max_p: float,
        min_z: float = None,
        max_z: float = None,
        min_t: float = None,
        max_t: float = None,
        state_spec: StateSpecification = StateSpecification.PH,
    ):
        """
        Function to compute bounds of OBL grid for different state specifications

        :param min_p: Minimum pressure [bar]
        :param max_p: Maximum pressure [bar]
        :param min_z: Minimum composition, can be scalar or list
        :param max_z: Maximum composition, can be scalar or list
        :param min_t: Minimum temperature [K]
        :param max_t: Maximum temperature [K]
        :param state_spec: StateSpecification, P, PT or PH
        """
        assert np.isscalar(min_z) or len(min_z) == self.nc - 1, (
            "min_z must be a scalar or a vector of length nc-1."
        )
        assert np.isscalar(max_z) or len(max_z) == self.nc - 1, (
            "max_z must be a scalar or a vector of length nc-1."
        )

        if state_spec <= PhysicsBase.StateSpecification.PT:
            axes_min, axes_max = (
                value_vector(self.PT_axes_min),
                value_vector(self.PT_axes_max),
            )

        elif state_spec == PhysicsBase.StateSpecification.PH:
            pz_axes_min = [min_p] + (
                [min_z for i in range(self.nc - 1)]
                if np.isscalar(min_z)
                else list(min_z)
            )
            pz_axes_max = [max_p] + (
                [max_z for i in range(self.nc - 1)]
                if np.isscalar(max_z)
                else list(max_z)
            )

            min_h, max_h = np.nan, np.nan
            for i in range(self.nc):
                for pres in [min_p, max_p]:
                    for temp in [min_t, max_t]:
                        zi = np.array(
                            [1.0 if i == ii else 0.0 for ii in range(self.nc - 1)]
                        )
                        hi = self.property_containers[0].compute_total_enthalpy(
                            state_pt=np.array([pres] + list(zi) + [temp])
                        )
                        min_h = hi if hi < min_h or np.isnan(min_h) else min_h
                        max_h = hi if hi > max_h or np.isnan(max_h) else max_h

            axes_min = value_vector(pz_axes_min + [min_h])
            axes_max = value_vector(pz_axes_max + [max_h])

        else:
            raise RuntimeError(f"Unknown state specification: {state_spec}")

        return axes_min, axes_max

    @abc.abstractmethod
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
        pass

    @abc.abstractmethod
    def set_initial_conditions_from_array(
        self, mesh: conn_mesh, input_distribution: dict
    ):
        """
        Method to set initial conditions by arrays or uniformly for all cells

        :param mesh: conn_mesh object
        :param input_distribution: Initial distributions of unknowns over grid, must have keys equal to self.vars
                                   and each entry is scalar or array of length equal to number of cells
        """
        pass

    def populate_mesh_history_defaults(self, mesh) -> None:
        """Allocate and fill ``mesh.Xhis_bounds`` from ``history_fields`` defaults.

        The engine's ``build_Xop`` reads boundary-cell history values from ``mesh.Xhis_bounds``
        and falls back to zero when the buffer is empty. This helper writes the configured
        ``HistoryField.default`` for each field into every boundary cell so engines that use
        non-zero defaults (MPFA / mechanical paths with boundary cells) behave correctly.
        No-op when ``history_fields`` is empty or ``mesh.n_bounds == 0``.

        :param mesh: Connection mesh the engine will run on
        :type mesh: darts.engines.conn_mesh
        :returns: None
        """
        if not self.history_fields:
            return
        n_bounds = int(getattr(mesh, "n_bounds", 0))
        if n_bounds <= 0:
            return
        defaults = np.array([h.default for h in self.history_fields], dtype=float)
        # cell-major layout: [(h_0, h_1, ..., h_{n_his-1}) for cell 0, cell 1, ...]
        flat = np.tile(defaults, n_bounds)
        mesh.Xhis_bounds = value_vector(flat.tolist())

    def init_wells(self, wells):
        """
        Function to initialize the well rates for each well.

        When ``history_fields`` is non-empty, the per-field default values are also broadcast
        to each well's ``Xhis_well_default`` and to its ``control`` / ``constraint`` objects.
        This is the well-side analog of ``mesh.Xhis_bounds``: when the engine evaluates an
        operator on a well cell it pads the extended OBL state with these fallback values.

        :param wells: List of multi-segment wells to initialise
        :type wells: list[darts.engines.ms_well]
        :returns: None
        """
        for w in wells:
            assert isinstance(w, ms_well)
            w.init_rate_parameters(
                self.n_vars,
                self.n_ops,
                self.phases,
                self.well_ctrl_itor,
                self.thermal_var_itor,
                self.thermal,
            )

        if self.history_fields:
            defaults = value_vector([h.default for h in self.history_fields])
            for w in wells:
                w.Xhis_well_default = defaults
                if hasattr(w, "control"):
                    w.control.Xhis_well_default = defaults
                if hasattr(w, "constraint"):
                    w.constraint.Xhis_well_default = defaults

    def create_interpolator(
        self,
        evaluator: operator_set_evaluator_iface,
        axes_min: value_vector,
        axes_max: value_vector,
        timer_name: str,
        n_ops: int,
        n_axes_points: index_vector = None,
        algorithm: str = 'multilinear',
        mode: str = 'adaptive',
        platform: str = 'cpu',
        precision: str = 'd',
        region: str = '',
        is_barycentric: bool = False,
    ):
        """
        Create interpolator object according to specified parameters

        :param evaluator: State operators to be interpolated. Evaluator object is used to generate supporting points
        :type evaluator: darts.interpolators.operator_set_evaluator_iface
        :param timer_name: Name of timer object
        :type timer_name: str
        :param n_ops: Number of operators
        :type n_ops: int
        :param axes_min: Minimal bounds of OBL axes
        :type axes_min: value_vector
        :param axes_max: Maximal bounds of OBL axes
        :type axes_max: value_vector
        :param algorithm: interpolator type:
            'multilinear' (default) - piecewise multilinear generalization of piecewise bilinear interpolation on rectangles;
            'linear' - a piecewise linear generalization of piecewise linear interpolation on triangles
        :type algorithm: str
        :param mode: interpolator mode:
            'adaptive' (default) - only supporting points required to perform interpolation are evaluated on-the-fly;
            'static' - all supporting points are evaluated during itor object construction
        :type mode: str
        :param platform: platform used for interpolation calculations :
            'cpu' (default) - interpolation happens on CPU;
            'gpu' - interpolation happens on GPU
        :type platform: str
        :param precision: precision used in interpolation calculations:
            'd' (default) - supporting points are stored and interpolation is performed using double precision;
            's' - supporting points are stored and interpolation is performed using single precision
        :type precision: str
        :type region: str
        :param region: str(region index) for reservoir operator, str(-1) for well operator, '' for others
        needed to make different filenames for cache as self.well_operators has the same type ReservoirOperators
        :param is_barycentric: Flag which turn on barycentric interpolation on Delaunay simplices
        :type is_barycentric: bool

        :returns: tuple (interpolator, effective_n_ops)
        :rtype: tuple[operator_set_gradient_evaluator_iface, int]
        """
        # check input OBL props
        if axes_min is None:
            axes_min = self.axes_min
        if axes_max is None:
            axes_max = self.axes_max
        if n_axes_points is None:
            n_axes_points = self.n_axes_points

        # verify then inputs are valid
        n_dims = len(n_axes_points)
        assert len(axes_min) == n_dims
        assert len(axes_max) == n_dims
        for n_p in n_axes_points:
            assert n_p > 1

        itor_name = f"{algorithm}_{mode}_{platform}_interpolator_i_{precision}_{n_dims:d}_{n_ops:d}"
        itor = None
        general = False
        cache_loaded = 0
        signature_n_ops = n_ops
        # try to create itor with 32-bit index type first (kinda a bit faster)
        try:
            if algorithm == 'linear':
                itor = eval(itor_name)(
                    evaluator, n_axes_points, axes_min, axes_max, is_barycentric
                )
            else:
                itor = eval(itor_name)(evaluator, n_axes_points, axes_min, axes_max)
        except (ValueError, NameError):
            # 32-bit index type did not succeed: either total amount of points is out of range or has not been compiled
            # try 64 bit now raising exception this time if goes wrong:
            if (
                np.prod(np.array(n_axes_points), dtype=np.float64)
                < np.iinfo(np.int64).max
            ):
                itor_name = itor_name.replace('interpolator_i', 'interpolator_l')
            else:
                itor_name = itor_name.replace('interpolator_i', 'interpolator_ll')
            try:
                if algorithm == 'linear':
                    itor = eval(itor_name)(
                        evaluator,
                        n_axes_points,
                        axes_min,
                        axes_max,
                        is_barycentric,
                    )
                else:
                    itor = eval(itor_name)(evaluator, n_axes_points, axes_min, axes_max)
            except (ValueError, NameError) as err:
                # Try to find a templatized interpolator with the same name pattern
                # but with the closest possible higher n_ops available in darts.interpolators.
                try:
                    import importlib
                    import re

                    engines_module = importlib.import_module("darts.interpolators")
                    base_prefix = itor_name.rsplit('_', 1)[0]
                    pattern = rf"^{re.escape(base_prefix)}_(\d+)$"
                    # Find candidates with higher n_ops
                    candidates = []
                    for attr_name in dir(engines_module):
                        match = re.match(pattern, attr_name)
                        if match:
                            available_n_ops = int(match.group(1))
                            if available_n_ops > n_ops:
                                candidates.append((available_n_ops, attr_name))

                    if candidates:
                        # Sort candidates by n_ops in ascending order
                        candidates.sort(key=lambda x: x[0])
                        selected_n_ops, selected_name = candidates[0]
                        selected_cls = getattr(engines_module, selected_name)
                        if algorithm == 'multilinear':
                            itor = selected_cls(
                                evaluator, n_axes_points, axes_min, axes_max
                            )
                        elif algorithm == 'linear':
                            itor = selected_cls(
                                evaluator,
                                n_axes_points,
                                axes_min,
                                axes_max,
                                is_barycentric,
                            )
                        else:
                            raise ValueError("Invalid algorithm: " + algorithm)
                        signature_n_ops = selected_n_ops
                        print(
                            "Falling back to interpolator with higher n_ops:",
                            selected_name,
                            f"(n_ops={selected_n_ops})",
                        )
                    else:
                        raise RuntimeError(
                            "No higher n_ops templatized interpolator found"
                        )
                except Exception:
                    # As a last resort, try the general implementation if available
                    try:
                        itor = eval("multilinear_adaptive_cpu_interpolator_general")(
                            evaluator,
                            n_axes_points,
                            axes_min,
                            axes_max,
                            n_dims,
                            n_ops,
                        )
                        general = True
                    except Exception:
                        raise ValueError(
                            "Number of operators is incorrect, no templatized interpolator exists"
                        ) from err

        if self.cache:
            # create unique signature for interpolator
            itor_cache_signature = f"{type(evaluator).__name__}_{mode}_{precision}_{n_dims:d}_{signature_n_ops:d}_{region}"
            # geenral itor has a different point_data format
            if general:
                itor_cache_signature += "_general_"
            for dim in range(n_dims):
                itor_cache_signature += (
                    f"_{n_axes_points[dim]:d}_{axes_min[dim]:e}_{axes_max[dim]:e}"
                )
            # compute signature hash to uniquely identify itor parameters and load correct cache
            itor_cache_signature_hash = str(
                hashlib.md5(itor_cache_signature.encode()).hexdigest()
            )
            itor_cache_filename = 'obl_point_data_' + itor_cache_signature_hash + '.pkl'

            if hasattr(self, 'cache_dir'):
                itor_cache_filename = os.path.join(self.cache_dir, itor_cache_filename)
            # if cache file exists, read it safely
            if os.path.exists(itor_cache_filename):
                print(
                    "Reading cached point data for ",
                    type(itor).__name__,
                    'from',
                    itor_cache_filename,
                )
                loaded_point_data = self._safe_pickle_load(itor_cache_filename)
                if loaded_point_data is not None:
                    itor.point_data = loaded_point_data
                    print(len(itor.point_data.keys()), "points loaded")
                    cache_loaded = 1
                else:
                    print("Cached point data is invalid, ignoring.")
            if mode == 'adaptive':
                # for adaptive itors, delay obl data save moment, because
                # during simulations new points will be evaluated.
                # on model destruction (or interpreter exit), itor point data will be written to disk
                self.created_itors.append((itor, itor_cache_filename))

        itor.init()
        # for static itors, save the cache immediately after init, if it has not been already loaded
        # otherwise, there is no point to save the same data over and over
        if self.cache and mode == 'static' and not cache_loaded:
            print("Writing point data for ", type(itor).__name__)
            self._atomic_pickle_dump(itor.point_data, itor_cache_filename)

        self.create_itor_timers(itor, timer_name)
        return itor, signature_n_ops

    def create_itor_timers(
        self, itor: operator_set_gradient_evaluator_iface, timer_name: str
    ):
        """
        Create timers for interpolators.

        :param itor: The object which performs evaluation of operator gradient (interpolators currently, AD-based in future)
        :type itor: operator_set_gradient_evaluator_iface object
        :param timer_name: Timer name to be used for the given interpolator
        :type timer_name: str
        """
        try:
            # in case this is a subsequent call, create only timer node for the given timer
            self.timer.node["jacobian assembly"].node["interpolation"].node[
                timer_name
            ] = timer_node()
        except:
            # in case this is first call, create first only timer nodes for jacobian assembly and interpolation
            self.timer.node["jacobian assembly"] = timer_node()
            self.timer.node["jacobian assembly"].node["interpolation"] = timer_node()
            self.timer.node["jacobian assembly"].node["interpolation"].node[
                timer_name
            ] = timer_node()

        # assign created timer to interpolator
        itor.init_timer_node(
            self.timer.node["jacobian assembly"].node["interpolation"].node[timer_name]
        )

    def write_cache(self):
        # this function can be called two ways
        #   1. Destructor (__del__) method
        #   2. Via atexit function, before interpreter exits
        # In either case it should only be invoked by the earliest call (which can be 1 or 2 depending on situation)
        # Switch cache off to prevent the second call
        self.cache = False
        for itor, fname in self.created_itors:
            filename = fname
            if hasattr(self, 'cache_dir'):
                if (
                    os.path.basename(fname) == fname
                ):  # could already have a folder in fname
                    filename = os.path.join(self.cache_dir, fname)
            print("Writing point data for ", type(itor).__name__, 'to', filename)
            # Temporarily ignore SIGINT/SIGTERM to avoid partial writes during sudden termination
            prev_int = None
            prev_term = None
            try:
                try:
                    prev_int = signal.signal(signal.SIGINT, signal.SIG_IGN)
                except Exception:
                    prev_int = None
                try:
                    prev_term = signal.signal(signal.SIGTERM, signal.SIG_IGN)
                except Exception:
                    prev_term = None
                self._atomic_pickle_dump(itor.point_data, filename)
            finally:
                if prev_int is not None:
                    try:
                        signal.signal(signal.SIGINT, prev_int)
                    except Exception:
                        pass
                if prev_term is not None:
                    try:
                        signal.signal(signal.SIGTERM, prev_term)
                    except Exception:
                        pass

    def _atomic_pickle_dump(self, obj, final_path: str):
        """
        Atomically write pickle to final_path using a temporary file followed by os.replace.
        Ensures data is flushed (fsync) to disk before the rename to avoid corruption.
        """
        directory = os.path.dirname(final_path) or "."
        try:
            os.makedirs(directory, exist_ok=True)
        except Exception:
            pass

        fd, tmp_path = tempfile.mkstemp(
            prefix=os.path.basename(final_path) + ".tmp.", suffix=".pkl", dir=directory
        )
        try:
            with os.fdopen(fd, "wb") as fp:
                pickle.dump(obj, fp, protocol=4)
                fp.flush()
                try:
                    os.fsync(fp.fileno())
                except Exception:
                    pass
            os.replace(tmp_path, final_path)
            # Best-effort directory fsync to persist the rename
            try:
                dir_fd = os.open(directory, os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            except Exception:
                pass
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    def _safe_pickle_load(self, path: str):
        """
        Safely load a pickle file. If the file is corrupted or truncated, delete it and return None.
        """
        try:
            with open(path, "rb") as fp:
                return pickle.load(fp)
        except Exception as err:
            print(
                "Failed to read cached point data from",
                path,
                "-",
                type(err).__name__,
                str(err),
            )
            try:
                os.remove(path)
                print("Removed corrupted cache file", path)
            except Exception:
                pass
            return None

    def body_path_start(self, output_folder):
        """
        Function that prepare hypercube output demonstrating occupancy of state space (for adaptive interpolators)

        :param output_folder: folder to write output to
        """
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        with open(os.path.join(output_folder, 'body_path.txt'), "w") as fp:
            self.processed_body_idxs = set()
            for id in range(self.n_vars):
                fp.write(
                    f"{self.n_axes_points[id]:d} {self.axes_min[id]:f} {self.axes_max[id]:f} {self.vars[id]}\n"
                )
            fp.write('Body Index Data\n')

    def body_path_add_bodys(self, output_folder, time):
        """
        Function performs hypercube output demonstrating occupancy of state space (for adaptive interpolators)

        :param output_folder: folder to write output to
        :param time: current time
        """
        with open(os.path.join(output_folder, 'body_path.txt'), "a") as fp:
            fp.write(f'T={time:f}\n')
            itor = self.acc_flux_itor[0]
            all_idxs = set(itor.get_hypercube_indexes())
            new_idxs = all_idxs - self.processed_body_idxs
            for i in new_idxs:
                fp.write(f'{i:d}\n')
            self.processed_body_idxs = all_idxs

    def __del__(self):
        # first write cache
        if self.cache:
            self.write_cache()
        # Now destroy all objects in physics
        for name in list(vars(self).keys()):
            delattr(self, name)
