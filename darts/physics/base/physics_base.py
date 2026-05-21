import abc
import atexit
import hashlib
import os
import pickle
import signal
import tempfile
from enum import Enum
from functools import total_ordering

import numpy as np

from darts.engines import *
from darts.interpolators import *
from darts.physics.base.operators_base import ThermalVarOperator, WellCtrlOperators


class PhysicsBase:
    """
    This is a base class for Physics definition.

    Physics contains all necessary objects to initialize and run the DARTS :class:`engine`.

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

    # Advisory cell count per axis used to derive a legacy (axes_min, axes_max) window
    # for the C++ interpolator constructor and for the integer-keyed pickle-export filter.
    # The adaptive interpolator cache grows past this window on demand, so the value is
    # not load-bearing; it only affects pickle export of in-bounds cells and the legacy
    # `axes_hypercube_mult` packing reach.
    ADVISORY_N_AXES_POINTS = 1024

    def __init__(
        self,
        state_spec: StateSpecification,
        variables: list,
        components: list,
        phases: list,
        n_ops: int,
        timer: timer_node,
        axes_step: list,
        axes_origin: list = None,
        sim_eps: float = None,
        cache: bool = False,
    ):
        """
        Constructor of the PhysicsBase class. Defines the OBL grid by per-axis cell size
        plus an optional origin offset; creates a `simulation` timer node and initializes
        caching.

        Adaptive interpolators key on a signed multi-index, so the grid is fully
        defined by ``axes_step`` (per-axis cell size) and ``axes_origin`` (per-axis
        offset). The cache grows past any prescribed window on demand.

        :param state_spec: State specification - 0) P, 1) PT, 2) PH
        :type state_spec: StateSpecification
        :param variables: Independent variables.
        :param components: Components.
        :param phases: Phases.
        :param n_ops: Number of operators.
        :param timer: Timer object.
        :param axes_step: Per-axis cell size, length n_vars. Required.
        :param axes_origin: Per-axis grid origin (default: zeros).
        :param sim_eps: Epsilon below which the Newton update is clipped to the physical [0,1] simplex.
        :param cache: Switch to cache operator values to disk between runs.
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

        # OBL grid: cell size + origin per axis. With multi-index-keyed adaptive
        # interpolators these two are the only state the cache needs; no max/n_points.
        assert axes_step is not None, "axes_step is required"
        assert len(axes_step) == self.n_vars, (
            f"axes_step must have {self.n_vars} entries, got {len(axes_step)}"
        )
        self.axes_step = [float(s) for s in axes_step]
        if axes_origin is None:
            axes_origin = [0.0] * self.n_vars
        assert len(axes_origin) == self.n_vars, (
            f"axes_origin must have {self.n_vars} entries, got {len(axes_origin)}"
        )
        self.axes_origin = [float(o) for o in axes_origin]

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
        parallel_evaluation: bool = False,
        n_workers: int = None,
        evaluator_factory_hook=None,
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
        # OBL grid is fully defined by (axes_origin, axes_step) — see __init__.
        # No more determine_obl_bounds() call: the adaptive interpolator caches cells
        # on demand wherever the solver lands.

        # set engine, operators and create interpolators
        self.engine = self.set_engine(discr_type, platform)

        # for separate mineral fraction in reactive flow formulations
        if n_solid is not None:
            self.engine.n_solid = n_solid

        # Set state specification in the engine
        self.set_state_spec(state_spec=self.state_spec)

        self.set_operators()
        self.set_interpolators(
            platform,
            itor_type,
            itor_mode,
            itor_precision,
            is_barycentric,
            parallel_evaluation=parallel_evaluation,
            n_workers=n_workers,
            evaluator_factory_hook=evaluator_factory_hook,
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
        parallel_evaluation: bool = False,
        n_workers: int = None,
        evaluator_factory_hook=None,
    ):
        """
        Function to initialize set interpolator objects based on the set of operators.
        It creates timers for each of the interpolators.

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
        :param parallel_evaluation: Enable parallel batch evaluation of supporting points via multiprocessing
        :type parallel_evaluation: bool
        :param n_workers: Number of worker processes for parallel evaluation (default: os.cpu_count())
        :type n_workers: int
        :param evaluator_factory_hook: Callable ``(region: int) -> callable`` that returns a factory
            function for constructing a fresh evaluator per worker process. Required when
            ``parallel_evaluation=True``. Each factory must return an ``operator_set_evaluator_iface``.
        :type evaluator_factory_hook: callable
        """
        # Optionally wrap evaluators with ParallelEvaluator for batch parallelism
        if parallel_evaluation:
            if evaluator_factory_hook is None:
                raise ValueError(
                    "parallel_evaluation=True requires evaluator_factory_hook: "
                    "a callable(region) -> callable that returns a factory for "
                    "constructing a fresh evaluator per worker process."
                )
            from darts.physics.base.parallel_evaluator import ParallelEvaluator

            for region in self.regions:
                factory = evaluator_factory_hook(region)
                self.reservoir_operators[region] = ParallelEvaluator(
                    evaluator_factory=factory,
                    n_workers=n_workers,
                )

        # All interpolators share the same (axes_origin, axes_step) grid for compositional
        # variables. The thermal-var interpolator uses a separate PT-based grid (handled
        # internally by derived physics classes via thermal_var_axes_step/_origin).
        self.acc_flux_itor = {}
        self.property_itor = {}
        for region in self.regions:
            self.acc_flux_itor[region], _ = self.create_interpolator(
                self.reservoir_operators[region],
                n_ops=self.n_ops,
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
            timer_name='well interpolation',
            platform=platform,
            algorithm=itor_type,
            mode=itor_mode,
            precision=itor_precision,
            region='-1',
            is_barycentric=is_barycentric,
        )

        self.well_ctrl_itor, self.n_well_ctrl_itor_ops = self.create_interpolator(
            self.well_ctrl_operators,
            n_ops=self.well_ctrl_operators.n_ops,
            timer_name='well controls interpolation',
            platform=platform,
            algorithm=itor_type,
            mode=itor_mode,
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

    def init_wells(self, wells):
        """
        Function to initialize physics of wells

        :param wells: List of :class:`ms_well` objects
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

    def create_interpolator(
        self,
        evaluator: operator_set_evaluator_iface,
        timer_name: str,
        n_ops: int,
        axes_step: list = None,
        axes_origin: list = None,
        algorithm: str = 'multilinear',
        mode: str = 'adaptive',
        platform: str = 'cpu',
        precision: str = 'd',
        region: str = '',
        is_barycentric: bool = False,
    ):
        """
        Create an interpolator object using (axes_origin, axes_step) to define the grid.

        Defaults to ``self.axes_step`` / ``self.axes_origin`` from PhysicsBase. The
        legacy (axes_min, axes_max, n_axes_points) tuple passed to the C++ constructor
        is derived from these as ``axes_min = origin``, ``axes_max = origin + (N-1)*step``
        for an advisory N = :attr:`ADVISORY_N_AXES_POINTS`; the cache extends past freely.

        :param evaluator: Operator-set evaluator used to materialize supporting points.
        :param timer_name: Name of the timer subnode for this interpolator.
        :param n_ops: Number of operators.
        :param axes_step: Per-axis cell size (defaults to self.axes_step).
        :param axes_origin: Per-axis grid origin (defaults to self.axes_origin).
        :param algorithm: 'multilinear' (default) or 'linear'.
        :param mode: 'adaptive' (default) or 'static'.
        :param platform: 'cpu' (default) or 'gpu'.
        :param precision: 'd' (default) or 's'.
        :param region: Per-region tag used to disambiguate cache file names.
        :param is_barycentric: Enable Delaunay-based barycentric interpolation.
        :returns: (interpolator, effective_n_ops)
        """
        if axes_step is None:
            axes_step = self.axes_step
        if axes_origin is None:
            axes_origin = self.axes_origin
        assert len(axes_step) == self.n_vars, (
            f"axes_step length {len(axes_step)} != n_vars {self.n_vars}"
        )
        assert len(axes_origin) == self.n_vars

        # Derive the (n_axes_points, axes_min, axes_max) tuple required by the C++
        # interpolator ctor. The advisory window sets the integer-key reach for the
        # legacy pickle export; cells past it still cache on demand via multi-index keys.
        n_dims = self.n_vars
        advisory_n = PhysicsBase.ADVISORY_N_AXES_POINTS
        n_axes_points = index_vector([advisory_n] * n_dims)
        axes_min_vec = value_vector(list(axes_origin))
        axes_max_vec = value_vector(
            [axes_origin[i] + (advisory_n - 1) * axes_step[i] for i in range(n_dims)]
        )

        # calculate object name using 32 bit index type (i)
        itor_name = f"{algorithm}_{mode}_{platform}_interpolator_i_{precision}_{n_dims:d}_{n_ops:d}"
        itor = None
        general = False
        cache_loaded = 0
        signature_n_ops = n_ops
        # try to create itor with 32-bit index type first (kinda a bit faster)
        try:
            if algorithm == 'linear':
                itor = eval(itor_name)(
                    evaluator, n_axes_points, axes_min_vec, axes_max_vec, is_barycentric
                )
            else:
                itor = eval(itor_name)(
                    evaluator, n_axes_points, axes_min_vec, axes_max_vec
                )
        except (ValueError, NameError):
            # 32-bit index overflow or this (n_dims, n_ops) pair was not compiled.
            # Fall back to 64-bit; multi-index storage is unaffected.
            itor_name = itor_name.replace('interpolator_i', 'interpolator_l')
            try:
                if algorithm == 'linear':
                    itor = eval(itor_name)(
                        evaluator,
                        n_axes_points,
                        axes_min_vec,
                        axes_max_vec,
                        is_barycentric,
                    )
                else:
                    itor = eval(itor_name)(
                        evaluator, n_axes_points, axes_min_vec, axes_max_vec
                    )
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
                                evaluator, n_axes_points, axes_min_vec, axes_max_vec
                            )
                        elif algorithm == 'linear':
                            itor = selected_cls(
                                evaluator,
                                n_axes_points,
                                axes_min_vec,
                                axes_max_vec,
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
                            axes_min_vec,
                            axes_max_vec,
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
            # Cache identity is (axes_origin, axes_step) per axis — these define WHICH
            # physical points the cache contains, which is what matters for cache reuse.
            # axes_max and n_points are merely advisory in adaptive mode (cache grows
            # past them) so we omit them from the signature; two runs with identical
            # (origin, step) and different (n_points, axes_max) windows can now share
            # a cache. The legacy fmtv1 suffix lets us distinguish the new tuple-keyed
            # pickle format from old integer-keyed caches.
            for dim in range(n_dims):
                itor_cache_signature += (
                    f"_origin={axes_origin[dim]:e}_step={axes_step[dim]:e}"
                )
            itor_cache_signature += "_fmtv2"
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
                    # Prefer the tuple-keyed full export (preserves out-of-window cells)
                    # when available on the interpolator; fall back to the legacy
                    # integer-keyed view otherwise.
                    if (
                        loaded_point_data
                        and hasattr(itor, "point_data_full")
                        and isinstance(next(iter(loaded_point_data)), tuple)
                    ):
                        itor.point_data_full = loaded_point_data
                        print(
                            len(itor.point_data_full.keys()),
                            "points loaded (full multi-index format)",
                        )
                    else:
                        itor.point_data = loaded_point_data
                        print(
                            len(itor.point_data.keys()),
                            "points loaded (legacy integer-key format)",
                        )
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
                # Prefer the tuple-keyed full export to preserve out-of-window cells.
                # Falls back to the legacy integer-keyed view for interpolators that
                # do not expose the full view (e.g. static interpolators).
                if hasattr(itor, "point_data_full"):
                    self._atomic_pickle_dump(itor.point_data_full, filename)
                else:
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
                    f"{self.axes_origin[id]:f} {self.axes_step[id]:f} {self.vars[id]}\n"
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
        # __del__ may fire on a partially-constructed object (an exception in
        # __init__ before self.cache was assigned still triggers cleanup), so
        # read attributes defensively rather than asserting they exist.
        if getattr(self, 'cache', False):
            try:
                self.write_cache()
            except Exception:
                pass
        for name in list(vars(self).keys()):
            delattr(self, name)
