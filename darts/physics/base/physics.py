import atexit
import hashlib
import os
import signal
import threading
import warnings
from collections.abc import Iterable
from enum import Enum
from functools import total_ordering
from typing import Any

import numpy as np
from scipy.interpolate import interp1d

from darts.engines import *
from darts.interpolators import *
from darts.physics.base.operator_evaluator import (
    PropertyOperators,
    ReservoirOperators,
    ThermalVarOperator,
    WellCtrlOperators,
    WellOperators,
)
from darts.physics.base.history_extension import HistoryField, HistoryStateSupport
from darts.tools.obl_cache import OblCacheCodec


class PhysicsBase:
    """
    Physics class for compositional (and compositional-thermal) simulation with the OBL super engine.

    Contains everything required to initialize and drive the DARTS :class:`engine`:
    OBL grid setup (P-z or P-T-z / P-H-z), property containers, operator evaluators,
    interpolators, well controls, initial conditions, and history-aware OBL axes.

    The OBL grid is defined by ``axes_step`` (per-axis cell size) and an optional
    ``axes_origin``. Defaults are pressure = 1 bar, composition axes = ``epsilon_z``,
    and thermal axis = 273.15 K for P-T or 0 for P-H. The adaptive multi-index-keyed
    interpolator caches cells on demand wherever the solver lands; there is no fixed
    grid window.

    The Physics object is composed of :class:`PropertyContainer` objects for each of the regions and a set of operators.
    The operators consist of :class:`ReservoirOperators` objects for each of the regions, a :class:`WellOperators`,
    a :class:`WellCtrlOperators`, a :class:`ThermalVarOperator` and a :class:`PropertyOperators` object.
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
    :ivar well_ctrl_operators: :class:`WellCtrlOperators` object for well controls
    :type well_ctrl_operators: WellCtrlOperators
    :ivar thermal_var_operator: :class:`ThermalVarOperator` object for generic state specification
    :type thermal_var_operator: ThermalVarOperator
    :ivar regions: List of property regions
    :type regions: list
    """

    engine: engine_base
    well_operators: operator_set_evaluator_iface
    well_ctrl_operators: WellCtrlOperators
    thermal_var_operator: ThermalVarOperator

    # The OBL supporting-point cache *file format* (the mmap-arena codec) lives in
    # darts.tools.obl_cache.OblCacheCodec — a stateless codec this class composes once and
    # delegates every file read/write to. PhysicsBase keeps only the orchestration: the
    # write_cache loop over created_itors, the dirty-point trackers, SIGINT/SIGTERM + __del__
    # flushing, cache-path resolution, and the interpolator pickle cache.
    _cache_codec = OblCacheCodec()

    @total_ordering
    class StateSpecification(Enum):
        P = 0
        PT = 1
        PH = 2
        PS = 3

        def __lt__(self, other: object) -> bool:
            if self.__class__ is other.__class__:
                return self.value < other.value
            return NotImplemented

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
        state_spec: 'PhysicsBase.StateSpecification' = None,
        cache: bool = False,
        history_fields: Iterable[HistoryField] | None = None,
    ) -> None:
        """
        Configure the OBL grid and physics state for a compositional simulation.

        Adaptive interpolators key on a signed multi-index, so the grid is fully
        defined by ``axes_step`` (per-axis cell size) and ``axes_origin`` (per-axis
        offset). The cache grows past any prescribed window on demand.

        :param components: List of components.
        :param phases: List of phases.
        :param timer: Timer object.
        :type timer: :class:`darts.engines.timer_node`
        :param axes_step: Per-axis cell size [p_step, z_step_1, ..., z_step_{nc-1}, t_step?].
            For ``extrapolation_flag=True`` the composition steps must all be equal.
        :param axes_origin: Per-axis grid origin. Defaults match the open-DARTS unit
            conventions: pressure = 1 bar, compositions = ``epsilon_z`` (composition
            floor, avoids the simplex boundary), thermal axis = 273.15 K for
            ``state_spec=PT`` or 0 for ``state_spec=PH`` (enthalpy reference depends on
            the EOS — pass an explicit ``axes_origin`` to override).
        :param epsilon_z: Composition-axis offset (default 1e-9).
        :param sim_eps_multiplier: Multiplier on ``epsilon_z`` to obtain ``sim_eps``, the
            epsilon below which the Newton update is clipped to the physical [0,1] simplex.
        :param extrapolation_flag: Enable extrapolation logic (z[last] < 0 if nc >= 3).
        :param state_spec: State specification (P, PT, PH, PS). Default is P (isothermal).
        :type state_spec: StateSpecification
        :param cache: Cache supporting points to disk between runs.
        :param history_fields: Optional :class:`HistoryField` descriptors declaring auxiliary
            OBL axes (e.g. ``sg_max`` for Killough hysteresis) that are fed into operator
            interpolation but are NOT Newton unknowns. Pass ``None`` or an empty list for
            standard drainage-only behaviour.
        """
        # Default state_spec must be supplied here (rather than in the signature) because the
        # class reference PhysicsBase is not yet resolvable at default-evaluation time.
        if state_spec is None:
            state_spec = PhysicsBase.StateSpecification.P

        # Compute compositional layout: nc, nph and (iso)thermal
        nc = len(components)
        nph = len(phases)
        self.thermal = state_spec > PhysicsBase.StateSpecification.P

        # Build the variable list: pressure, nc-1 components, optional temperature/enthalpy
        variables = ["pressure"] + components[:-1]
        if self.thermal:
            variables += (
                ["temperature"]
                if state_spec == PhysicsBase.StateSpecification.PT
                else ["enthalpy"]
            )

        n_vars = len(variables)
        # Number of operators = NE /*acc*/ + NE * NP /*flux*/ + NP * /*density*/ + NP /*UPSAT*/
        # + NE * NP /*gradient*/ + NE /*kinetic*/ + 2 * NP /*gravpc*/ + 1 /*poro*/
        # + NP /*LAMBDA*/ + NP /*SAT*/ + NP /*enthalpy*/ + 2 /*temperature and pressure*/
        # = NE * (2 * nph + 2) + 7 * nph + 3
        n_ops = n_vars * (2 * nph + 2) + 7 * nph + 3

        assert sim_eps_multiplier > 1, (
            "sim_eps_multiplier must be > 1 for consistent OBL axes / solution clipping"
        )

        # Define variables and number of operators
        self.state_spec = state_spec
        self.is_ph = state_spec > PhysicsBase.StateSpecification.PT
        self.vars = variables
        self.n_vars = n_vars

        self.components = components
        self.nc = nc
        self.phases = phases
        self.nph = nph
        self.n_ops = n_ops

        # OBL grid: cell size + origin per axis. With multi-index-keyed adaptive
        # interpolators these two are the only state the cache needs; no max/n_points.
        # Validate with exceptions (not assert, which `python -O` strips): each step
        # is divided into on the C++ hot path, so a missing/zero/negative/NaN value
        # would silently produce inf/NaN indices instead of a clear error.
        if axes_step is None:
            raise ValueError("axes_step is required (per-axis OBL cell size)")
        if len(axes_step) != self.n_vars:
            raise ValueError(
                f"axes_step must have {self.n_vars} entries, got {len(axes_step)}"
            )
        self.axes_step = [float(s) for s in axes_step]
        for i, s in enumerate(self.axes_step):
            # rejects NaN, +/-inf and <= 0 in one expression
            if not (0.0 < s < float("inf")):
                raise ValueError(
                    f"axes_step[{i}]={s!r} must be finite and strictly positive"
                )

        self.sim_eps = epsilon_z * sim_eps_multiplier

        if axes_origin is None:
            # Sensible defaults matching open-DARTS unit conventions:
            #   pressure (axis 0)           → 1 bar
            #   compositions (axes 1..nc-1) → epsilon_z (composition-axis floor)
            #   thermal axis (last, when thermal):
            #     PT → 273.15 K (0 °C, conventional standard temperature)
            #     PH → 0       (enthalpy reference is EOS-specific; override per case)
            #     PS → 0       (entropy reference is EOS-specific; override per case)
            axes_origin = [1.0] + [epsilon_z] * (self.nc - 1)
            if self.thermal:
                if state_spec == PhysicsBase.StateSpecification.PT:
                    axes_origin.append(273.15)
                else:
                    axes_origin.append(0.0)
        assert len(axes_origin) == self.n_vars, (
            f"axes_origin must have {self.n_vars} entries, got {len(axes_origin)}"
        )
        self.axes_origin = [float(o) for o in axes_origin]

        # Extrapolation logic (z[last component] < 0 when nc >= 3) requires a single dz
        # shared by every composition axis.
        self.extrapolation_flag = extrapolation_flag
        self.dz = self.axes_step[1] if self.nc > 1 else None
        if extrapolation_flag and self.nc > 1:
            for i in range(self.nc - 1):
                assert abs(self.axes_step[1 + i] - self.dz) < 1e-15, (
                    "extrapolation requires equal dz across all composition axes"
                )

        # Initialize timer for simulation and caching
        self.timer = timer.node["simulation"]
        self.cache = cache
        # list of created interpolators
        # is used on destruction to save cache data
        if self.cache:
            self.created_itors = []
            self._cache_finalized = False
            self._last_flushed_sizes = {}
            # Fallback key set for interpolators without native dirty-point tracking.
            self._flushed_point_keys = {}
            # PID of the process that owns this cache. Cache file writes must only happen
            # here: a forked child (e.g. a ParallelEvaluator worker, which inherits this
            # object, its atexit handler and SIGTERM/SIGINT flush handler) must never
            # write the cache from its stale copy — that would race/corrupt the owner's
            # files. The guard in write_cache / _finalize_cache enforces this.
            self._cache_owner_pid = os.getpid()
            atexit.register(self._finalize_cache)
            self._install_signal_handlers()

        self.regions = []
        self.property_containers = {}
        self.reservoir_operators = {}
        self.property_operators = {}
        # Output-side property operators/interpolators are populated lazily by
        # OutputBase.set_phase_properties / filter_phase_props. Kept separate so
        # the physics-managed property_operators / property_itor (set by
        # set_interpolators, possibly wrapped by ParallelEvaluator) are never
        # overwritten by output reconfiguration.
        self.output_property_operators = {}
        self.output_property_itor = {}

        self.history = HistoryStateSupport(history_fields)

    def check_properties(self):
        """
        Assert all properties have been specified for each phase in each region

        """
        for region in self.regions:
            assert self.property_containers[region].check_properties()
        return

    @property
    def n_history(self) -> int:
        """
        Return the number of configured OBL history variables (``len(history_fields)``).

        :returns: Count of auxiliary OBL axes that enter interpolation but not the Newton system
        :rtype: int
        """
        return self.history.n_fields

    @property
    def history_fields(self) -> list[HistoryField]:
        """
        Return configured history descriptors in OBL storage order.

        :returns: Ordered history descriptors.
        """
        return self.history.fields

    @property
    def n_state(self) -> int:
        """
        Return the total OBL interpolation-state size per cell = ``n_vars + n_history``.

        :returns: Number of axes the reservoir / well interpolators consume per cell
        :rtype: int
        """
        return self.n_vars + self.n_history

    def get_interpolator_state_labels(self) -> list:
        """
        Return axis labels used by the OBL interpolators, in storage order.

        The first ``n_vars`` entries are the primary Newton unknown labels (``self.vars``),
        followed by one label per history field (e.g. ``"sg_max"``). Used by
        :mod:`darts.output` to label columns when dumping operator state.

        :returns: Ordered list of axis labels of length ``n_state``
        :rtype: list[str]
        """
        return list(self.vars) + self.history.labels

    def get_history_default(self, label: str) -> float:
        """
        Return the configured initial / fallback value for a history field.

        :param label: Label of the history field, must match one declared in ``history_fields``
        :type label: str
        :returns: The ``default`` attribute of the matching :class:`HistoryField`
        :rtype: float
        :raises KeyError: If no history field has the requested label
        """
        return self.history.default(label)

    def get_engine_history_array(self, label: str, n_blocks: int = None) -> np.ndarray:
        """
        Return a per-cell copy of ``engine.Xhistory`` restricted to one history axis.

        The engine stores ``Xhistory`` as a flat ``[(n_blocks + n_bounds) * n_history]`` buffer in
        cell-major order (all ``n_history`` values for cell 0, then cell 1, ...). This helper
        pulls out just the reservoir blocks for one label and returns a copy (safe to mutate).

        :param label: Label of the history field to extract, must match one in ``history_fields``
        :type label: str
        :param n_blocks: Number of reservoir blocks to read. When ``None``, inferred as
                         ``Xhistory.size // n_history`` (i.e. all cells including boundaries)
        :type n_blocks: int, optional
        :returns: One-dimensional array of shape ``(n_blocks,)`` with the requested axis values
        :rtype: numpy.ndarray
        :raises RuntimeError: If no history fields are configured on this physics
        :raises KeyError: If no history field has the requested label
        """
        return self.history.get_engine_array(self.engine, label, n_blocks)

    def get_engine_interpolator_state(self, n_blocks: int = None) -> np.ndarray:
        """
        Return the full OBL state ``[X | Xhistory]`` flattened in cell-major order.

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
        return self.history.get_interpolator_state(self.engine, self.n_vars, n_blocks)

    def set_engine_history_array(
        self, label: str, values, n_blocks: int = None
    ) -> None:
        """
        Overwrite one axis of ``engine.Xhistory`` with a per-cell scalar or array.

        The selected history column is updated through a reshaped writable NumPy view of
        ``engine.Xhistory``.

        :param label: Label of the history field to write, must match one in ``history_fields``
        :type label: str
        :param values: Value(s) for the single history axis identified by ``label``: either a
                       scalar applied to every cell, or a one-dimensional array-like of length
                       ``n_blocks``
        :type values: float or array-like
        :param n_blocks: Number of reservoir blocks to write. When ``None``, inferred as
                         ``Xhistory_flat.size // n_history``
        :type n_blocks: int, optional
        :returns: None
        :raises RuntimeError: If no history fields are configured on this physics
        :raises KeyError: If no history field has the requested label
        """
        self.history.set_engine_array(self.engine, label, values, n_blocks)

    def init_physics(
        self,
        discr_type: str = 'tpfa',
        platform: str = 'cpu',
        itor_type: str = 'multilinear',
        itor_precision: str = 'd',
        verbose: bool = False,
        is_barycentric: bool = False,
        n_solid: int | None = None,
        parallel_evaluation: bool = False,
        n_workers: int | None = None,
        evaluator_factory_hook=None,
        verbose_evaluators: bool = False,
    ) -> None:
        """
        Initialise engines, operators, and interpolators for this physics object.

        :param discr_type: Discretization type, 'tpfa' (default) or 'mpfa'
        :type discr_type: str
        :param platform: Switch for CPU/GPU engine, 'cpu' (default) or 'gpu'
        :type platform: str
        :param itor_type: Type of interpolation method, 'multilinear' (default) or 'linear'
        :type itor_type: str
        :param itor_precision: Precision of interpolation, 'd' (default) - double precision or 's' - single precision
        :type itor_precision: str
        :param verbose: Set verbose level
        :type verbose: bool
        :param is_barycentric: Flag which turn on barycentric interpolation on Delaunay simplices
        :type is_barycentric: bool
        :param n_solid: Number of solid minerals for element-based reactive flow
        :type n_solid: int
        :param parallel_evaluation: Enable parallel batch evaluation of supporting points via multiprocessing
        :type parallel_evaluation: bool
        :param n_workers: Number of worker processes for parallel evaluation (default: ``os.cpu_count()``)
        :type n_workers: int, optional
        :param evaluator_factory_hook: Callable ``(region: int) -> callable`` that returns a factory
                                       for constructing a fresh evaluator per worker process.
                                       Required when ``parallel_evaluation=True``.
        :type evaluator_factory_hook: callable
        """
        # OBL grid is fully defined by (axes_origin, axes_step) — see __init__.
        # No more determine_obl_bounds() call: the adaptive interpolator caches cells
        # on demand wherever the solver lands.

        # Whether parallel-evaluation workers (and the main process's duplicate serial
        # evaluator builds) are allowed to print. Default False = silenced; the model sets
        # it True at its highest verbosity. Read by _wrap_evaluators_parallel below.
        self._verbose_evaluators = verbose_evaluators

        # set engine, operators and create interpolators
        self.engine = self.set_engine(discr_type, platform)

        # Tell the engine how many per-cell history variables to reserve in its Xop / Xhistory
        # buffers. Must be set before engine.init() allocates them.
        self.history.configure_engine(self.engine)

        # for separate mineral fraction in reactive flow formulations
        if n_solid is not None:
            self.engine.n_solid = n_solid

        # Set state specification in the engine
        self.set_state_spec(state_spec=self.state_spec)

        # Run checks on Physics and PropertyContainer consistency:
        # Assert all properties have been specified for each phase in each region
        for region in self.regions:
            self.property_containers[region].check_properties()

        # Set operators and interpolators
        self.set_operators()
        self.set_interpolators(
            platform,
            itor_type,
            itor_precision,
            is_barycentric,
            parallel_evaluation=parallel_evaluation,
            n_workers=n_workers,
            evaluator_factory_hook=evaluator_factory_hook,
        )

        # When history fields are active, verify that all hysteresis-bearing evaluators in
        # every region share consistent trapping parameters. Catches silent drift between
        # rel_perm_ev and capillary_pressure_ev built from independent Corey sources.
        if self.history_fields:
            for pc in self.property_containers.values():
                if hasattr(pc, "validate_history_consistency"):
                    pc.validate_history_consistency()
        return

    def set_state_spec(self, state_spec: StateSpecification) -> None:
        """
        Set the state specification on the underlying engine.

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

    def add_property_region(self, property_container: Any, region: int = 0) -> None:
        """
        Register a property container for one region and propagate history metadata.

        :param property_container: Object for evaluation of properties
        :type property_container: :class:`PropertyContainer`
        :param region: Tag of the region, to be used as a key in `property_containers` dict
        """
        # Tell the property container how many OBL history variables the physics appends
        # to the state vector so that it can locate primary vars correctly (e.g. temperature
        # at position [nc] rather than [-1] when sg_max is appended), and pass the ordered
        # labels so it can expose {label: value} to history-aware evaluators.
        self.history.configure_property_container(property_container)
        self.property_containers[region] = property_container
        self.regions.append(region)
        return

    def set_operators(self) -> None:
        """
        Build :class:`ReservoirOperators` for each region, :class:`WellOperators` for the well cells,
        :class:`WellCtrlOperators` for well controls, :class:`ThermalVarOperator` for the thermal
        state variable, and :class:`PropertyOperators` for property evaluation.
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

    def set_engine(
        self, discr_type: str = 'tpfa', platform: str = 'cpu'
    ) -> engine_base:
        """
        Create and return the ``engine_super`` instance for this physics.

        :param discr_type: Type of discretization, 'tpfa' (default) or 'mpfa'
        :type discr_type: str
        :param platform: Switch for CPU/GPU engine, 'cpu' (default) or 'gpu'
        :type platform: str
        :returns: :class:`engine_super` object
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

    def set_interpolators(
        self,
        platform='cpu',
        itor_type='multilinear',
        itor_precision='d',
        is_barycentric: bool = False,
        parallel_evaluation: bool = False,
        n_workers: int | None = None,
        evaluator_factory_hook=None,
    ) -> None:
        """
        Initialise the interpolator set and timer nodes for the configured operators.

        If history_fields is non-empty, the reservoir / well / well-control interpolators are built
        on the extended axes set (primary OBL axes + one axis per history field); the thermal_var
        interpolator always stays on the primary PT axes.

        :param platform: Switch for CPU/GPU engine, 'cpu' (default) or 'gpu'
        :type platform: str
        :param itor_type: Type of interpolation method, 'multilinear' (default) or 'linear'
        :type itor_type: str
        :param itor_precision: Precision of interpolation, 'd' (default) - double precision or 's' - single precision
        :type itor_precision: str
        :param is_barycentric: Flag which turn on barycentric interpolation on Delaunay simplices
        :type is_barycentric: bool
        :param parallel_evaluation: Enable parallel batch evaluation of supporting points via multiprocessing
        :type parallel_evaluation: bool
        :param n_workers: Number of worker processes for parallel evaluation (default: os.cpu_count())
        :type n_workers: int
        :param evaluator_factory_hook: Callable ``(attribute: str, region: int | None) -> callable``
            that returns a factory function for constructing a fresh evaluator per worker
            process. Required when ``parallel_evaluation=True``. Each factory must return an
            ``operator_set_evaluator_iface``. The hook is queried once per wrap target
            (``reservoir_operators[r]``, ``property_operators[r]``, ``well_operators``,
            ``well_ctrl_operators``, ``thermal_var_operator``).
        :type evaluator_factory_hook: callable
        """
        # Optionally wrap every evaluator with ParallelEvaluator for batch parallelism.
        # All five wrap targets share a single multiprocessing pool so the total worker
        # process count stays at n_workers regardless of how many evaluators are wrapped.
        if parallel_evaluation:
            self._wrap_evaluators_parallel(
                self._parallel_wrap_targets(),
                evaluator_factory_hook,
                n_workers,
            )

        # Reservoir/property/well interpolators consume the full OBL state
        # [primary vars | history vars]. The thermal-var interpolator uses a separate
        # primary PT grid for well initialization.
        operator_axes_origin, operator_axes_step = self.history.extend_axes(
            self.axes_origin, self.axes_step
        )
        self.acc_flux_itor = {}
        self.property_itor = {}
        for region in self.regions:
            self.acc_flux_itor[region], _ = self.create_interpolator(
                self.reservoir_operators[region],
                n_ops=self.n_ops,
                axes_step=operator_axes_step,
                axes_origin=operator_axes_origin,
                platform=platform,
                algorithm=itor_type,
                precision=itor_precision,
                timer_name=f'reservoir {region:d} interpolation',
                region=str(region),
                is_barycentric=is_barycentric,
            )

            self.property_itor[region], _ = self.create_interpolator(
                self.property_operators[region],
                n_ops=self.n_ops,
                axes_step=operator_axes_step,
                axes_origin=operator_axes_origin,
                platform=platform,
                algorithm=itor_type,
                precision=itor_precision,
                timer_name=f'property {region:d} interpolation',
                region=str(region),
                is_barycentric=is_barycentric,
            )

        self.acc_flux_w_itor, _ = self.create_interpolator(
            self.well_operators,
            n_ops=self.n_ops,
            axes_step=operator_axes_step,
            axes_origin=operator_axes_origin,
            timer_name='well interpolation',
            platform=platform,
            algorithm=itor_type,
            precision=itor_precision,
            region='-1',
            is_barycentric=is_barycentric,
        )

        self.well_ctrl_itor, self.n_well_ctrl_itor_ops = self.create_interpolator(
            self.well_ctrl_operators,
            n_ops=self.well_ctrl_operators.n_ops,
            axes_step=operator_axes_step,
            axes_origin=operator_axes_origin,
            timer_name='well controls interpolation',
            platform=platform,
            algorithm=itor_type,
            precision=itor_precision,
            is_barycentric=is_barycentric,
        )
        # Thermal-var interpolator uses a PT-based grid; the derived physics class may
        # set self.thermal_var_axes_step / self.thermal_var_axes_origin to override the
        # default (which mirrors the main grid).
        thermal_step = getattr(self, 'thermal_var_axes_step', None)
        thermal_origin = getattr(self, 'thermal_var_axes_origin', None)
        self.thermal_var_itor, _ = self.create_interpolator(
            self.thermal_var_operator,
            n_ops=self.thermal_var_operator.n_ops,
            axes_step=thermal_step,
            axes_origin=thermal_origin,
            include_history=False,
            timer_name='well initialization',
            platform=platform,
            algorithm=itor_type,
            precision=itor_precision,
            is_barycentric=is_barycentric,
        )
        return

    def _parallel_wrap_targets(self):
        """
        Return the list of (attribute, region_or_None) tuples whose evaluators
        should be wrapped with ParallelEvaluator when ``parallel_evaluation=True``.

        Subclasses with a different operator layout (e.g. chemistry, which has
        ``initial_operators`` instead of a separate ``well_operators``) override
        this method to return their own target list.
        """
        targets = []
        for region in self.regions:
            targets.append(('reservoir_operators', region))
            targets.append(('property_operators', region))
        targets.append(('well_operators', None))
        targets.append(('well_ctrl_operators', None))
        targets.append(('thermal_var_operator', None))
        return targets

    def _wrap_evaluators_parallel(
        self, targets, evaluator_factory_hook, n_workers, start_method=None
    ):
        """
        Replace each evaluator listed in ``targets`` with a :class:`ParallelEvaluator`
        backed by a single shared :class:`SharedEvaluatorPool`. The pool is stored on
        ``self._shared_evaluator_pool`` so it lives as long as the physics object.

        The evaluator object currently held by each target attribute is handed to
        its wrapper as the serial evaluator (single-point ``evaluate()`` path), so
        wrapping never triggers factory-side model reconstruction in the main
        process; the factories are only exercised inside pool workers.

        :param targets: list of ``(attribute_name, region_or_None)`` tuples
        :param evaluator_factory_hook: callable ``(attribute, region) -> factory``
            producing a picklable factory that returns a fresh evaluator.
        :param n_workers: pool size; defaults to ``os.cpu_count()``.
        :param start_method: multiprocessing start method (``None`` = platform default).
        """
        if evaluator_factory_hook is None:
            raise ValueError(
                "parallel_evaluation=True requires evaluator_factory_hook: "
                "a callable(attribute, region) -> callable that returns a factory "
                "for constructing a fresh evaluator per worker process."
            )
        from darts.physics.base.parallel_evaluator import (
            ParallelEvaluator,
            SharedEvaluatorPool,
        )

        # Build one factory per wrap target. Keys are (attr, region) tuples.
        factories = {
            (attr, region): evaluator_factory_hook(attr, region)
            for attr, region in targets
        }
        # Silence workers + duplicate serial-evaluator builds unless the model asked for
        # evaluator output (highest verbosity), so only one evaluator's output is shown.
        silence = not getattr(self, '_verbose_evaluators', False)
        # One pool shared by every wrap; the pool worker pre-builds one evaluator per key.
        self._shared_evaluator_pool = SharedEvaluatorPool(
            factories,
            n_workers=n_workers,
            start_method=start_method,
            silence_workers=silence,
        )

        for attr, region in targets:
            key = (attr, region)
            # The wrap target already exists on the physics -- hand it to the
            # wrapper as its serial evaluator instead of letting the factory
            # reconstruct the whole model once per target.
            existing = getattr(self, attr, None)
            if region is not None and existing is not None:
                existing = (
                    existing.get(region)
                    if hasattr(existing, "get")
                    else existing[region]
                )
            wrapped = ParallelEvaluator(
                evaluator_factory=factories[key],
                shared_pool=self._shared_evaluator_pool,
                key=key,
                silence=silence,
                serial_evaluator=existing,
            )
            if region is None:
                setattr(self, attr, wrapped)
            else:
                getattr(self, attr)[region] = wrapped

    def _extend_parallel_wrap(
        self, new_targets, evaluator_factory_hook, start_method=None
    ):
        """
        Extend an existing shared evaluator pool with additional wrap targets.

        Used by :meth:`OutputBase.set_phase_properties` to add
        ``('output_property_operators', region)`` keys to a pool that was created
        at ``set_interpolators`` time with only the physics-side keys. Because
        ``multiprocessing.Pool`` does not support adding worker initializers
        post-creation, this method shuts down the existing pool and rebuilds it
        with the merged factories. Existing :class:`ParallelEvaluator` wrappers
        have their ``_shared_pool`` reference repointed at the new pool so they
        keep working transparently.

        :param new_targets: list of ``(attribute_name, region_or_None)`` tuples
            to add. Existing targets in the pool are preserved.
        :param evaluator_factory_hook: callable
            ``(attribute, region) -> picklable factory`` for the new targets.
        :param start_method: optional multiprocessing start method override.
        """
        old_pool = getattr(self, '_shared_evaluator_pool', None)
        if old_pool is None:
            raise RuntimeError(
                "_extend_parallel_wrap requires an existing shared evaluator pool; "
                "call _wrap_evaluators_parallel first (or enable parallel_evaluation)."
            )
        from darts.physics.base.parallel_evaluator import (
            ParallelEvaluator,
            SharedEvaluatorPool,
        )

        old_factories = dict(old_pool._factories)
        new_factories = {
            (attr, region): evaluator_factory_hook(attr, region)
            for attr, region in new_targets
        }
        merged = {**old_factories, **new_factories}

        silence = not getattr(self, '_verbose_evaluators', False)
        n_workers = old_pool.n_workers
        old_pool.shutdown()
        self._shared_evaluator_pool = SharedEvaluatorPool(
            merged,
            n_workers=n_workers,
            start_method=start_method,
            silence_workers=silence,
        )

        # Repoint all existing ParallelEvaluator wrappers at the new pool so
        # they continue to dispatch through a live pool.
        for attr, region in old_factories.keys():
            existing = (
                getattr(self, attr)
                if region is None
                else getattr(self, attr).get(region)
            )
            if isinstance(existing, ParallelEvaluator):
                existing._shared_pool = self._shared_evaluator_pool

        # Wrap each newly added target.
        for attr, region in new_targets:
            key = (attr, region)
            wrapped = ParallelEvaluator(
                evaluator_factory=new_factories[key],
                shared_pool=self._shared_evaluator_pool,
                key=key,
                silence=silence,
            )
            if region is None:
                setattr(self, attr, wrapped)
            else:
                getattr(self, attr)[region] = wrapped

    def evaluate_interpolators(
        self,
        itor: operator_set_evaluator_iface,
        etor: operator_set_evaluator_iface,
        states: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """
        Evaluate interpolated operators on a batch of states.

        :param itor: Interpolator object used for operator evaluation
        :type itor: operator_set_evaluator_iface
        :param etor: Evaluator that provides operator-name metadata
        :type etor: operator_set_evaluator_iface
        :param states: Two-dimensional array of state points to evaluate. History-aware
            interpolators expect ``n_state`` columns; primary-only interpolators expect
            ``n_vars`` columns.
        :type states: numpy.ndarray
        :returns: Mapping from operator name to evaluated values per state
        :rtype: dict[str, numpy.ndarray]
        """
        # Create values, dvalues and idxs arrays
        states = np.asarray(states, dtype=float)
        if states.ndim != 2:
            raise ValueError("states must be a two-dimensional array")
        n_states = len(states)
        n_state_axes = states.shape[1]
        if n_state_axes not in {self.n_vars, self.n_state}:
            raise ValueError(
                f"states must have {self.n_vars} or {self.n_state} columns, "
                f"got {n_state_axes}"
            )
        primary_states = states[:, : self.n_vars]
        if self.nc > 1:
            physical_points = np.sum(primary_states[:, 1 : self.nc], axis=1) <= 1.0
        else:
            physical_points = np.ones(n_states, dtype=bool)
        states = value_vector(states.flatten())
        values = value_vector(np.zeros(self.n_ops * n_states))
        values_numpy = np.array(values, copy=False)
        dvalues = value_vector(np.zeros(self.n_ops * n_states * n_state_axes))

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
        phase_name: str | None = None,
        inj_composition: list | None = None,
        inj_temp: float | None = None,
    ) -> None:
        """
        Set a well control or constraint on the target :class:`well_control_iface`.

        Call ``set_bhp_control()`` or ``set_rate_control()`` on the control or constraint
        object that lives on ``ms_well``. To deactivate a control or constraint, pass
        ``WellControlType.NONE``.

        :param wctrl: well_control_iface object responsible for control/constraint. It must be set to:
                      - well_obj.control for well control
                      - well_obj.constraint for well constraint
        :param control_type: Well control type -2) NONE (if constraint needs to be deactivated), -1) BHP,
                             0) MOLAR_RATE, 1) MASS_RATE, 2) VOLUMETRIC_RATE, 3) ADVECTIVE_HEAT_RATE; default is BHP
        :param is_inj: Is injection well (true) or production well (false)
        :param target: Target BHP or rate, consistent with well control type
        :param phase_name: Name of the phase rate of which is controlled. This input can be used if well control is of
                           the rate type. If not specified and well control is of the rate type, total rate will be
                           controlled.
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

        phase_idx = None
        if phase_name is not None:
            phase_idx = self.phases.index(phase_name)

        # Pass controls specification to ms_well object
        if control_type == well_control_iface.BHP:
            wctrl.set_bhp_control(is_inj, target, inj_composition, inj_temp)
        elif (
            well_control_iface.BHP.value
            < control_type.value
            < well_control_iface.NUMBER_OF_RATE_TYPES.value
        ):
            # Injection/production rate
            target = (
                np.abs(target) if is_inj else -np.abs(target)
            )  # + for inj, - for prod
            # If phase_idx is None, total rate is controlled
            wctrl.set_rate_control(
                is_inj, control_type, phase_idx, target, inj_composition, inj_temp
            )

        return

    # determine_obl_bounds() was removed. With the multi-index-keyed adaptive
    # interpolator the OBL grid is fully defined by (axes_origin, axes_step); cells
    # outside any prescribed window are materialized on demand. For state_spec=PH the
    # enthalpy origin/step are supplied directly by the user (or derived in the
    # derived physics class).

    def set_initial_conditions_from_depth_table(
        self, mesh: conn_mesh, input_distribution: dict, input_depth: list | np.ndarray
    ) -> None:
        """
        Set initial conditions from a depth-tabulated distribution of properties.

        :param mesh: conn_mesh object
        :param input_distribution: Distributions of primary unknowns over depth; keys must
                                   match ``self.vars`` (each entry is a scalar or an array
                                   matching the length of ``input_depth``)
        :param input_depth: Depths at which the distribution is tabulated
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
                # Temperature was provided; interpolate P and T to compute enthalpies
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
                # Interpolate primary variable directly
                itor = interp1d(
                    input_depth,
                    input_distribution[variable],
                    kind="linear",
                    fill_value="extrapolate",
                )
                values = itor(depths)

            values = np.resize(np.asarray(values), mesh.n_res_blocks)
            np.asarray(mesh.initial_state)[ith_var :: self.n_vars] = values

        self.history.populate_boundary_defaults(mesh)

    def set_initial_conditions_from_array(
        self, mesh: conn_mesh, input_distribution: dict
    ) -> None:
        """
        Set initial conditions from arrays or uniform cell-wise values.

        :param mesh: conn_mesh object
        :param input_distribution: Distributions of primary unknowns; keys must match ``self.vars``
                                   (each entry is a scalar or an array of length ``n_res_blocks``)
        """
        for variable, values in input_distribution.items():
            if not np.isscalar(values) and not len(values) == mesh.n_res_blocks:
                warnings.warn(
                    f'Initial condition for variable {variable} has different length, '
                    f'resizing {len(values)} to {mesh.n_res_blocks}',
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
                        'Initial state must specify either temperature or enthalpy, '
                        'but neither was provided!'
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
        self.history.populate_boundary_defaults(mesh)

    def init_wells(self, wells: list[ms_well]) -> None:
        """
        Initialise well-rate parameters and propagate history defaults to wells.

        When ``history_fields`` is non-empty, the per-field default values are also broadcast
        to each well's ``Xhistory_well_default`` and to its ``control`` / ``constraint`` objects.
        This is the well-side analogue of ``mesh.Xhistory_bounds``: when the engine evaluates an
        operator on a well cell it pads the extended OBL state with these fallback values.

        :param wells: List of multi-segment wells to initialise
        :type wells: list[darts.engines.ms_well]
        :returns: None
        """
        for w in wells:
            assert isinstance(w, ms_well)
            w.init_physics(
                self.n_vars,
                self.n_ops,
                self.phases,
                self.well_ctrl_itor,
                self.thermal_var_itor,
                self.thermal,
            )

        self.history.populate_well_defaults(wells)

    def create_interpolator(
        self,
        evaluator: operator_set_evaluator_iface,
        timer_name: str,
        n_ops: int,
        axes_step: list[float] = None,
        axes_origin: list[float] = None,
        algorithm: str = 'multilinear',
        platform: str = 'cpu',
        precision: str = 'd',
        region: str = '',
        is_barycentric: bool = False,
        include_history: bool = True,
    ) -> tuple[operator_set_gradient_evaluator_iface, int]:
        """
        Create an interpolator object using (axes_origin, axes_step) to define the grid.

        Defaults to ``self.axes_step`` / ``self.axes_origin`` from PhysicsBase and,
        when history fields are configured, appends one ``HistoryField`` axis per
        history variable. These axes are passed straight to the C++ interpolator
        constructor, which takes ``(axes_origin, axes_step)`` natively. The grid is
        unbounded: cells are enumerated on demand via signed multi-index keys.

        When point-data caching is enabled (``self.cache``), the cache file name is
        derived from the evaluator's class name among other shape parameters; a
        :class:`ParallelEvaluator` wrapper is unwrapped to its serial evaluator
        first, so serial and parallel runs of identical physics share one cache. A
        cache file left by a version whose signature still carried the ``adaptive``
        token hashes to a different name; it is detected and kept in use, so no
        existing cache is orphaned.

        :param evaluator: Operator-set evaluator used to materialize supporting points.
        :param timer_name: Name of the timer subnode for this interpolator.
        :param n_ops: Number of operators.
        :param axes_step: Per-axis cell size (defaults to self.axes_step).
        :param axes_origin: Per-axis grid origin (defaults to self.axes_origin).
        :param algorithm: 'multilinear' (default) or 'linear'.
        :param platform: 'cpu' (default) or 'gpu'.
        :param precision: 'd' (default) or 's'.
        :param region: Per-region tag used to disambiguate cache file names.
        :param is_barycentric: Enable Delaunay-based barycentric interpolation.
        :param include_history: Append history-field axes when using the default axes.
            Disable for primary-only helper interpolators such as thermal well initialization.
        :returns: (interpolator, effective_n_ops)
        """
        use_default_axes = axes_step is None and axes_origin is None
        if axes_step is None:
            axes_step = self.axes_step
        if axes_origin is None:
            axes_origin = self.axes_origin
        if include_history and self.history_fields and use_default_axes:
            axes_origin, axes_step = self.history.extend_axes(axes_origin, axes_step)
        axes_step = [float(s) for s in axes_step]
        axes_origin = [float(o) for o in axes_origin]
        if len(axes_step) != len(axes_origin):
            raise ValueError(
                f"axes_step length {len(axes_step)} != axes_origin length "
                f"{len(axes_origin)}"
            )
        if not axes_step:
            raise ValueError("at least one interpolator axis is required")
        for i, s in enumerate(axes_step):
            if not (0.0 < s < float("inf")):
                raise ValueError(
                    f"axes_step[{i}]={s!r} must be finite and strictly positive"
                )
        for i, o in enumerate(axes_origin):
            if not (-float("inf") < o < float("inf")):
                raise ValueError(f"axes_origin[{i}]={o!r} must be finite")

        # The C++ interpolator ctors take (axes_origin, axes_step) natively; the grid
        # is unbounded (origin + step only), cells are enumerated on demand.
        n_dims = len(axes_step)
        axes_origin_vec = value_vector(axes_origin)
        axes_step_vec = value_vector(axes_step)

        # Build the constructor argument tuple (everything after `evaluator`) once, then
        # reuse it across the 32-bit / 64-bit / higher-n_ops / general fallbacks.
        if algorithm == 'linear':
            ctor_args = (axes_origin_vec, axes_step_vec, is_barycentric)
        else:
            ctor_args = (axes_origin_vec, axes_step_vec)

        # Exposed interpolator name pattern. It carries neither an index-type letter (the
        # index-type template parameter was dropped -- storage is keyed on a multi-index,
        # so the index type is not part of the class identity) nor an "adaptive" token
        # (the static interpolators are gone, so it distinguishes nothing):
        #   {algorithm}_{platform}_interpolator_{precision}_{n_dims}_{n_ops}
        itor_base = f"{algorithm}_{platform}_interpolator"
        itor_name = f"{itor_base}_{precision}_{n_dims:d}_{n_ops:d}"
        itor = None
        general = False
        cache_loaded = 0
        signature_n_ops = n_ops
        try:
            itor = eval(itor_name)(evaluator, *ctor_args)
            err = None
        except (ValueError, NameError) as e:
            err = e
        if itor is None:
            # Try to find a templatized interpolator with the same name pattern
            # but with the closest possible higher n_ops available in darts.interpolators.
            try:
                import importlib
                import re

                engines_module = importlib.import_module("darts.interpolators")
                # Find candidates with higher n_ops
                candidates = []
                pattern = rf"^{re.escape(itor_base)}_{precision}_{n_dims:d}_(\d+)$"
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
                    if algorithm not in ('multilinear', 'linear'):
                        raise ValueError("Invalid algorithm: " + algorithm)
                    itor = selected_cls(evaluator, *ctor_args)
                    signature_n_ops = selected_n_ops
                    print(
                        "Falling back to interpolator with higher n_ops:",
                        selected_name,
                        f"(n_ops={selected_n_ops})",
                    )
                else:
                    raise RuntimeError("No higher n_ops templatized interpolator found")
            except Exception:
                # No compiled template for this (n_dims, n_ops). Name the two real
                # causes instead of the old misleading "operators" message (there is
                # no '*_general' interpolator — that fallback was always dead).
                raise ValueError(
                    f"No compiled OBL interpolator template for "
                    f"(n_dims={n_dims}, n_ops={n_ops}). "
                    f"If n_dims exceeds the compiled maximum, rebuild with a larger "
                    f"-DOPENDARTS_MAX_DIMS (must be >= n_dims). The interpolator "
                    f"family is selected by -DOPENDARTS_INTERPOLATOR_PROFILE "
                    f"(MINIMAL omits the 'linear' templates). Tried: {itor_name}."
                ) from err

        # In-RAM cap on the derived hypercube cache (LRU on CPU; clear-on-overflow on
        # GPU). Purely an in-memory bound: it does NOT change the on-disk OBL cache
        # format or the persisted supporting-point cache (only point_data is saved).
        # 0/absent = unbounded (legacy behaviour). Read from the physics attribute so it
        # applies uniformly to every physics that goes through create_interpolator, and
        # hasattr-guarded so interpolators without the method (e.g. linear) are unaffected.
        hypercube_cap = getattr(self, 'hypercube_cap', 0)
        if hypercube_cap and hasattr(itor, 'set_hypercube_cap'):
            itor.set_hypercube_cap(int(hypercube_cap))

        if self.cache:
            # create unique signature for interpolator
            # Unwrap a ParallelEvaluator so the cache identity follows the REAL
            # evaluator class: otherwise every wrapped target hashes as
            # 'ParallelEvaluator', which (a) splits the cache between serial and
            # parallel runs of identical physics and (b) collides distinct
            # same-shape targets (e.g. ConversionOperators vs ThermalVarOperator)
            # onto one cache file.
            signature_evaluator = getattr(evaluator, "_serial_evaluator", evaluator)

            def cache_filename(itor_token: str) -> str:
                """Cache file name for one spelling of the interpolator identity token."""
                itor_cache_signature = f"{type(signature_evaluator).__name__}{itor_token}{precision}_{n_dims:d}_{signature_n_ops:d}_{region}"
                # geenral itor has a different point_data format
                if general:
                    itor_cache_signature += "_general_"
                # Cache identity is (axes_origin, axes_step) per axis — these define WHICH
                # physical points the cache contains, which is what matters for cache reuse.
                # Legacy bounded-window values are no longer part of adaptive-grid identity,
                # so runs with identical (origin, step) tuples share a cache.
                for dim in range(n_dims):
                    itor_cache_signature += (
                        f"_origin={axes_origin[dim]:e}_step={axes_step[dim]:e}"
                    )
                itor_cache_signature += "_fmtv2"
                # compute signature hash to uniquely identify itor parameters and load correct cache
                itor_cache_signature_hash = str(
                    hashlib.md5(itor_cache_signature.encode()).hexdigest()
                )
                name = 'obl_point_data_' + itor_cache_signature_hash + '.pkl'
                if hasattr(self, 'cache_dir'):
                    name = os.path.join(self.cache_dir, name)
                return name

            # Caches created from here on use the simplified identity token, mirroring the
            # interpolator names (no "adaptive": static interpolation is gone). A cache
            # written by an earlier version hashes its "_adaptive_" spelling to a different
            # file name; when such a file is already there it simply stays the cache file
            # for this run (read from and appended to in place), so no existing cache is
            # orphaned and none is duplicated on disk.
            itor_cache_filename = cache_filename('_')
            if not os.path.exists(itor_cache_filename):
                legacy_cache_filename = cache_filename('_adaptive_')
                if os.path.exists(legacy_cache_filename):
                    print(
                        "Using OBL cache written under the legacy signature:",
                        legacy_cache_filename,
                    )
                    itor_cache_filename = legacy_cache_filename
            # Fast path: a numpy-array snapshot (.keys.npy / .vals.npy) next to the
            # pickle restores the whole cache with one bulk read + a single C++ copy,
            # skipping pickle's per-point object graph and the dict round-trip. Used
            # when it exists and is at least as new as the pickle; otherwise fall back
            # to the pickle below and (re)generate the snapshot for next time.
            fast_loaded_size = self._load_cache(itor, itor_cache_filename)
            if fast_loaded_size is not None:
                print(
                    fast_loaded_size,
                    "points loaded from fast array cache for",
                    type(itor).__name__,
                )
                loaded_size = fast_loaded_size
                cache_loaded = 1
                self._last_flushed_sizes[id(itor)] = loaded_size
                if hasattr(itor, 'clear_point_data_delta'):
                    itor.clear_point_data_delta()
            # if cache file exists, read it safely
            elif os.path.exists(itor_cache_filename):
                print(
                    "Reading cached point data for ",
                    type(itor).__name__,
                    'from',
                    itor_cache_filename,
                )
                loaded_point_data = self._safe_pickle_load(itor_cache_filename)
                if loaded_point_data is not None:
                    # The canonical cache format is the tuple-keyed multi-index export
                    # (point_data_full). Adaptive interpolators are unbounded and expose
                    # only this view; older builds expose the legacy integer-keyed
                    # point_data. Pick whichever the loaded file + interpolator support.
                    if not loaded_point_data:
                        # Empty cache (e.g. a previous run flushed before any points were
                        # evaluated). Treat as a no-op so adaptive interpolators do not
                        # report a misleading "legacy integer-key format" warning below.
                        print("Cached point data is empty, ignoring.")
                        loaded_size = 0
                    elif hasattr(itor, "point_data_full") and isinstance(
                        next(iter(loaded_point_data)), tuple
                    ):
                        itor.point_data_full = loaded_point_data
                        loaded_size = len(itor.point_data_full.keys())
                        print(loaded_size, "points loaded (full multi-index format)")
                        cache_loaded = 1
                    elif hasattr(itor, "point_data") and not hasattr(
                        itor, "point_data_full"
                    ):
                        # Static interpolator with a legacy integer-keyed cache.
                        itor.point_data = loaded_point_data
                        loaded_size = len(loaded_point_data)
                        print(loaded_size, "points loaded (legacy integer-key format)")
                        cache_loaded = 1
                    else:
                        # Old integer-keyed cache for an adaptive interpolator: no longer
                        # loadable (the unbounded grid has no integer-key packing). It will
                        # be regenerated and re-saved in the multi-index format.
                        print(
                            "Cached point data is in the legacy integer-key format, which "
                            "adaptive interpolators no longer support; ignoring (it will be "
                            "regenerated in multi-index format)."
                        )
                        loaded_size = 0
                    if cache_loaded:
                        # Loaded points are already on disk, so reset the append-only
                        # dirty bookkeeping (development branch incremental-cache hooks).
                        self._last_flushed_sizes[id(itor)] = loaded_size
                        if hasattr(itor, 'clear_point_data_delta'):
                            itor.clear_point_data_delta()
                        elif hasattr(itor, 'point_data') and not hasattr(
                            itor, 'point_data_full'
                        ):
                            # Legacy integer-key tracker (older interpolator builds).
                            self._flushed_point_keys[id(itor)] = set(
                                loaded_point_data.keys()
                            )
                else:
                    print("Cached point data is invalid, ignoring.")
            # delay the obl data save moment, because during simulations new points
            # will be evaluated; on model destruction (or interpreter exit) the
            # itor point data will be written to disk
            self.created_itors.append((itor, itor_cache_filename))

        itor.init()

        self.create_itor_timers(itor, timer_name)
        return itor, signature_n_ops

    def create_itor_timers(
        self, itor: operator_set_gradient_evaluator_iface, timer_name: str
    ) -> None:
        """
        Create timer nodes for one interpolator.

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

    def write_cache(self) -> None:
        """
        Flush cached interpolator point data to disk.

        Safe to call repeatedly: per-itor skip when ``point_data`` has not grown
        since the previous flush. Existing cache files are updated by appending
        framed delta records containing only supporting points materialized since
        the last successful flush.
        """
        # Only the process that created this cache may write it. A forked child (e.g. a
        # parallel-evaluator worker) inherits this object but holds a stale interpolator
        # copy; letting it write would race/corrupt the owner's cache files.
        if os.getpid() != getattr(self, '_cache_owner_pid', os.getpid()):
            return
        if not getattr(self, 'created_itors', None):
            return
        if not hasattr(self, '_last_flushed_sizes'):
            self._last_flushed_sizes = {}
        if not hasattr(self, '_flushed_point_keys'):
            self._flushed_point_keys = {}
        codec = self._cache_codec
        for itor, fname in self.created_itors:
            filename = self._cache_filename(fname)
            itor_id = id(itor)
            cur_size = codec._point_data_size(itor)
            if self._last_flushed_sizes.get(itor_id, -1) == cur_size:
                continue
            # The cache is a C++-built mmap arena; an interpolator without that API
            # (an older .so) is not cacheable here -> skip, no fallback.
            if not hasattr(itor, 'build_arena_file'):
                self._last_flushed_sizes[itor_id] = cur_size
                continue

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
                # Per-point evaluation epochs for the points about to be written, as raw
                # arrays. Captured before _mark_point_data_*_flushed() clears the dirty
                # trackers. (None, None) for interpolators without native epoch tracking
                # (no epoch frame is written).
                epoch_keys, epoch_vals = codec._point_data_epoch_delta_arrays(itor)

                if codec._is_cache(filename):
                    # Existing arena cache: append only the points materialized since the
                    # last flush as a trailing DELTA(+EPOCH) frame, then recompact if the
                    # un-compacted tail grew large.
                    if itor_id in getattr(self, '_force_recompact', ()):
                        # ABI/placement-mismatch recovery loaded everything into the
                        # overlay; rebuild the arena with THIS binary's placement so future
                        # loads mmap cleanly (a pure-replay run would otherwise never do it).
                        # Carry historical epochs (old file) + this run's new epochs.
                        print(
                            "Recompacting cache",
                            filename,
                            "(placement/ABI mismatch recovery)",
                        )
                        codec._write_base(
                            itor,
                            filename,
                            itor.obl_arena_hash_id(),
                            epoch_keys,
                            epoch_vals,
                            preserve_epochs_from=filename,
                        )
                        self._force_recompact.discard(itor_id)
                        self._mark_point_data_flushed(itor, None, cur_size)
                        continue
                    dkeys, dvals = codec._point_data_delta_arrays(itor)
                    if dkeys is not None and len(dkeys):
                        print(
                            "Appending point data for ",
                            type(itor).__name__,
                            f'({len(dkeys)} new / {cur_size} total points) to',
                            filename,
                        )
                        codec._append_frame(filename, codec._KIND_DELTA, dkeys, dvals)
                        if epoch_keys is not None and len(epoch_keys):
                            codec._append_frame(
                                filename, codec._KIND_EPOCH, epoch_keys, epoch_vals
                            )
                    self._mark_point_data_delta_flushed(itor, None, cur_size)
                    codec._maybe_compact(itor, filename)
                    continue

                # No arena cache yet (absent, or a foreign/older file at this path): write a
                # fresh consolidated arena base atomically (any existing file is replaced by
                # the os.replace). No re-flash -- the points are already in memory.
                print(
                    "Writing point data for ",
                    type(itor).__name__,
                    f'({cur_size} points) to',
                    filename,
                )
                codec._write_base(
                    itor,
                    filename,
                    itor.obl_arena_hash_id(),
                    epoch_keys,
                    epoch_vals,
                )
                if not codec._verify_arena(filename, cur_size):
                    print("WARNING: cache write verification failed for", filename)
                self._mark_point_data_flushed(itor, None, cur_size)
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

    def _cache_filename(self, fname: str) -> str:
        filename = fname
        if hasattr(self, 'cache_dir'):
            if os.path.basename(fname) == fname:  # could already have a folder in fname
                filename = os.path.join(self.cache_dir, fname)
        return filename

    def _mark_point_data_flushed(self, itor, point_data: dict, size: int) -> None:
        itor_id = id(itor)
        self._last_flushed_sizes[itor_id] = size
        if hasattr(itor, 'clear_point_data_delta'):
            itor.clear_point_data_delta()
            self._flushed_point_keys.pop(itor_id, None)
        else:
            # Non-native itor (no dirty tracker): record ALL currently-cached keys as
            # flushed so the fallback delta (_point_data_delta) stays bounded. FC0x writers
            # pass point_data=None and only ever handle native itors, so this is a defensive
            # path -- derive the key set from the itor when no dict was supplied.
            keys = (
                point_data.keys()
                if point_data is not None
                else getattr(itor, 'point_data', {}).keys()
            )
            self._flushed_point_keys[itor_id] = set(keys)

    def _mark_point_data_delta_flushed(self, itor, delta: dict, size: int) -> None:
        itor_id = id(itor)
        self._last_flushed_sizes[itor_id] = size
        if hasattr(itor, 'clear_point_data_delta'):
            itor.clear_point_data_delta()
            self._flushed_point_keys.pop(itor_id, None)
        else:
            # Non-native fallback: after a flush every currently-cached key is persisted, so
            # record the full key set (bounding the next fallback delta). See note above.
            if delta is not None:
                self._flushed_point_keys.setdefault(itor_id, set()).update(delta.keys())
            else:
                self._flushed_point_keys[itor_id] = set(
                    getattr(itor, 'point_data', {}).keys()
                )

    def _finalize_cache(self):
        # Re-entrancy guard against atexit + __del__ + signal all firing within one
        # shutdown. The latch is released at the end so that work done *after* a caught
        # SIGINT (handler flushes, then user code continues and evaluates more points) is
        # still flushed by a later finalize; the per-itor size checks in write_cache
        # make a repeated call a cheap no-op.
        # Only the owning process finalizes the cache (forked children inherit this
        # object + its atexit/signal handlers but must never write the cache).
        if os.getpid() != getattr(self, '_cache_owner_pid', os.getpid()):
            return
        if getattr(self, '_cache_finalized', False):
            return
        self._cache_finalized = True
        try:
            # write_cache() now writes the self-contained arena cache file directly; there is no
            # separate derived snapshot to refresh.
            self.write_cache()
        finally:
            self._cache_finalized = False

    def _install_signal_handlers(self):
        # Flush OBL cache on SIGTERM/SIGINT before re-raising, so adaptive point data
        # survives external termination (job timeout, manual kill). SIGKILL cannot be intercepted.
        # Signal handlers can only be installed from the main thread.
        if threading.current_thread() is not threading.main_thread():
            return
        if not hasattr(self, '_prev_signal_handlers'):
            self._prev_signal_handlers = {}
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                prev = signal.getsignal(sig)
                if prev is self._signal_flush_handler:
                    continue
                self._prev_signal_handlers[sig] = prev
                signal.signal(sig, self._signal_flush_handler)
            except (ValueError, OSError):
                pass

    def _signal_flush_handler(self, signum, frame):
        try:
            self._finalize_cache()
        except Exception as exc:
            try:
                print(f"OBL cache flush on signal {signum} failed: {exc}")
            except Exception:
                pass
        # Restore previous handler and re-raise so original termination semantics take effect
        # (default action on SIGTERM, KeyboardInterrupt on SIGINT).
        prev = self._prev_signal_handlers.get(signum, signal.SIG_DFL)
        try:
            signal.signal(signum, prev if prev is not None else signal.SIG_DFL)
        except Exception:
            pass
        os.kill(os.getpid(), signum)

    # ------------------------------------------------------------------
    # Cache I/O facade. PhysicsBase delegates file reads/writes to the stateless
    # darts.tools.obl_cache.OblCacheCodec (composed as self._cache_codec); these few
    # methods are the entry points the rest of PhysicsBase + offline tooling call.
    # ------------------------------------------------------------------
    def _atomic_pickle_dump(self, *args, **kwargs):
        """Atomically write an interpolator point_data pickle (see OblCacheCodec)."""
        return self._cache_codec._atomic_pickle_dump(*args, **kwargs)

    def _load_cache(self, itor, pkl_path):
        # Delegate the load to the codec; if a cache file loaded points but left NO mmap'd
        # arena attached, the codec went the ABI/placement-mismatch occupied-slot recovery
        # path -> flag the itor so write_cache rebuilds the arena with this binary's
        # placement (else a pure-replay run never recompacts).
        n = self._cache_codec._load_cache(itor, pkl_path)
        if (
            n
            and self._cache_codec._is_cache(pkl_path)
            and hasattr(itor, 'has_arena')
            and not itor.has_arena()
        ):
            if not hasattr(self, '_force_recompact'):
                self._force_recompact = set()
            self._force_recompact.add(id(itor))
        return n

    @classmethod
    def load_point_epochs(cls, path):
        """Read per-point evaluation epochs from an OBL cache file (offline analysis)."""
        return cls._cache_codec.load_point_epochs(path)

    def _safe_pickle_load(self, *args, **kwargs):
        return self._cache_codec._safe_pickle_load(*args, **kwargs)

    def body_path_start(self, output_folder: str) -> None:
        """
        Start hypercube-occupancy output for adaptive interpolators.

        :param output_folder: folder to write output to
        """
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        with open(os.path.join(output_folder, 'body_path.txt'), "w") as fp:
            self.processed_body_idxs = set()
            axes_origin, axes_step = self.history.extend_axes(
                self.axes_origin, self.axes_step
            )
            labels = self.get_interpolator_state_labels()
            for idx, label in enumerate(labels):
                fp.write(f"{axes_origin[idx]:f} {axes_step[idx]:f} {label}\n")
            fp.write('Body Index Data\n')

    def body_path_add_bodys(self, output_folder: str, time: float) -> None:
        """
        Append hypercube-occupancy output for adaptive interpolators.

        :param output_folder: folder to write output to
        :param time: current time
        """
        with open(os.path.join(output_folder, 'body_path.txt'), "a") as fp:
            fp.write(f'T={time:f}\n')
            itor = self.acc_flux_itor[0]
            # Unbounded grid: hypercubes are identified by signed multi-index keys
            # (tuples of ints), not a single packed integer. Skip if the interpolator
            # does not expose them (e.g. GPU variants).
            if not hasattr(itor, "get_hypercube_keys"):
                return
            all_idxs = set(itor.get_hypercube_keys())
            new_idxs = all_idxs - self.processed_body_idxs
            for k in new_idxs:
                fp.write(" ".join(str(i) for i in k) + "\n")
            self.processed_body_idxs = all_idxs

    def evaluate_flash(
        self,
        state_spec: dict = None,
        compositions: dict = None,
        obl_interval_multiplier: float = 1.0,
        plot_flash_results: bool = False,
        region: int = 0,
    ):
        """
        Evaluate (and optionally plot) flash for the given state spec and composition ranges.

        :param state_spec: Dict of state-spec values; ``None`` evaluates full OBL space
        :type state_spec: dict
        :param compositions: Dict of composition ranges; ``None`` evaluates full OBL space
        :type compositions: dict
        :param obl_interval_multiplier: Multiplier to OBL axis intervals if ranges are obtained
                                        from OBL axes, default 1
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

    def __del__(self) -> None:
        # __del__ may fire on a partially-constructed object (an exception in
        # __init__ before self.cache was assigned still triggers cleanup), so
        # read attributes defensively rather than asserting they exist.
        # Use _finalize_cache so SIGTERM-installed handlers + atexit don't double-flush.
        if getattr(self, 'cache', False):
            try:
                self._finalize_cache()
            except Exception:
                pass
        # Now destroy all objects in physics
        for name in list(vars(self).keys()):
            delattr(self, name)
