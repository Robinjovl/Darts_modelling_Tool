import abc
import hashlib
import os
import pickle
import atexit
import numpy as np

from darts.engines import *


class PhysicsBase:
    """
    This is a base class for Physics definition.

    Physics contains all necessary objects to initialize and run the DARTS :class:`engine`.

    The Physics object is composed of :class:`PropertyContainer` objects for each of the regions and a set of operators.
    The operators consist of :class:`ReservoirOperators` objects for each of the regions, a :class:`WellOperators`,
    a :class:`RateOperators` and a :class:`PropertyOperators` object.
    For each set of operators (evaluators, etor), an interpolator (itor) object is created for use in the :class:`engine`.

    :ivar engine: Engine object
    :type engine: :class:`engine_base`
    :ivar property_containers: Set of :class:`PropertyContainer` objects for each of the regions for evaluation of reservoir cell properties
    :type property_containers: dict
    :ivar reservoir_operators: Set of :class:`ReservoirOperators` objects for each of the regions for evaluation of reservoir cell states
    :type reservoir_operators: dict
    :ivar property_operators: :class:`PropertyOperators` object for evaluation and interpolation of properties
    :type property_operators: dict
    :ivar wellbore_operators: :class:`WellOperators` object for evaluation of well cell states
    :type wellbore_operators: dict
    :ivar rate_operators: :class:`RateOperators` object for evaluation of fluxes
    :type rate_operators: dict
    :ivar regions: List of property regions
    :type regions: list
    """
    engine: engine_base
    wellbore_operators: operator_set_evaluator_iface
    rate_operators: operator_set_evaluator_iface
    mass_flux_operators: operator_set_evaluator_iface

    def __init__(self, variables: list, nc: int, phases: list, n_ops: int,
                 axes_min: value_vector, axes_max: value_vector, n_axes_points: index_vector,
                 timer: timer_node, cache: bool = False):
        """
        This is the constructor of the PhysicsBase class. It creates a `simulation` timer node and initializes caching.

        :param variables: List of independent variables
        :type variables: list
        :param nc: Number of components
        :type nc: int
        :param phases: List of phases
        :type phases: list
        :param n_ops: Number of operators
        :type n_ops: int
        :param axes_min, axes_max: Minimum, maximum of each OBL axis
        :type axes_min, axes_max: :class:`darts.engines.value_vector`
        :param n_axes_points: Number of OBL points along axes
        :type n_axes_points: index_vector
        :param timer: Timer object
        :type cache: :class:`darts.engines.timer_node`
        :param cache: Switch to cache operator values
        :type cache: bool
        """
        # Define variables and number of operators
        self.vars = variables
        self.n_vars = len(variables)

        self.nc = nc
        self.phases = phases
        self.nph = len(phases)
        self.n_ops = n_ops

        # Define OBL grid
        self.axes_min = axes_min
        self.axes_max = axes_max
        self.n_axes_points = n_axes_points

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
        self.mass_flux_operators = {}

    def init_physics(self, discr_type: str = 'tpfa', platform: str = 'cpu',
                     itor_type: str = 'multilinear', itor_mode: str = 'adaptive',
                     itor_precision: str = 'd', verbose: bool = False, is_barycentric: bool = False):
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
        """
        # Define operators, set engine, set interpolators and define well controls
        self.set_operators()
        self.engine = self.set_engine(discr_type, platform)
        self.set_interpolators(platform, itor_type, itor_mode, itor_precision, is_barycentric)
        self.define_well_controls()
        return

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
    def set_engine(self, discr_type: str = 'tpfa', platform: str = 'cpu') -> engine_base:
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

    def set_interpolators(self, platform='cpu', itor_type='multilinear', itor_mode='adaptive',
                          itor_precision='d', is_barycentric: bool = False):
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
        """
        self.acc_flux_itor = {}
        self.property_itor = {}
        self.mass_flux_itor = {}
        for region in self.regions:
            self.acc_flux_itor[region] = self.create_interpolator(self.reservoir_operators[region], n_ops=self.n_ops,
                                                                  platform=platform, algorithm=itor_type,
                                                                  mode=itor_mode, precision=itor_precision,
                                                                  timer_name='reservoir %d interpolation' % region, region=str(region),
                                                                  is_barycentric=is_barycentric)

            self.property_itor[region] = self.create_interpolator(self.property_operators[region], n_ops=self.n_ops,
                                                                  platform=platform, algorithm=itor_type,
                                                                  mode=itor_mode, precision=itor_precision,
                                                                  timer_name='property %d interpolation' % region, region=str(region))

            self.mass_flux_itor[region] = self.create_interpolator(self.mass_flux_operators[region], n_ops=self.n_ops,
                                                                   platform=platform, algorithm=itor_type,
                                                                   mode=itor_mode, precision=itor_precision,
                                                                   timer_name='Mass flux %d interpolation' % region,
                                                                   region=str(region))

        self.acc_flux_w_itor = self.create_interpolator(self.wellbore_operators, n_ops=self.n_ops,
                                                        timer_name='wellbore interpolation',
                                                        platform=platform, algorithm=itor_type, mode=itor_mode,
                                                        precision=itor_precision, region='-1')

        self.rate_itor = self.create_interpolator(self.rate_operators, n_ops=self.nph,
                                                  timer_name='well controls interpolation',
                                                  platform=platform, algorithm=itor_type, mode=itor_mode,
                                                  precision=itor_precision)
        return

    @abc.abstractmethod
    def define_well_controls(self):
        pass

    def init_wells(self, wells):
        """
        Function to initialize the well rates for each well.

        :param wells: List of :class:`ms_well` objects
        """
        for w in wells:
            assert isinstance(w, ms_well)
            w.init_rate_parameters(self.n_vars, self.n_ops, self.phases, self.rate_itor)

    def create_interpolator(self, evaluator: operator_set_evaluator_iface, timer_name: str, n_ops: int,
                            algorithm: str = 'multilinear', mode: str = 'adaptive',
                            platform: str = 'cpu', precision: str = 'd', region: str = '',
                            is_barycentric: bool = False):
        """
        Create interpolator object according to specified parameters

        :param evaluator: State operators to be interpolated. Evaluator object is used to generate supporting points
        :type evaluator: darts.engines.operator_set_evaluator_iface
        :param timer_name: Name of timer object
        :type timer_name: str
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
        needed to make different filenames for cache as self.wellbore_operators has the same type ReservoirOperators
        :param is_barycentric: Flag which turn on barycentric interpolation on Delaunay simplices
        :type is_barycentric: bool
        """
        # verify then inputs are valid
        assert len(self.n_axes_points) == self.n_vars
        assert len(self.axes_min) == self.n_vars
        assert len(self.axes_max) == self.n_vars
        for n_p in self.n_axes_points:
            assert n_p > 1

        # calculate object name using 32 bit index type (i)
        n_dims = self.n_vars
        itor_name = "%s_%s_%s_interpolator_i_%s_%d_%d" % (algorithm,
                                                          mode,
                                                          platform,
                                                          precision,
                                                          n_dims,
                                                          n_ops)
        itor = None
        general = False
        cache_loaded = 0
        # try to create itor with 32-bit index type first (kinda a bit faster)
        try:
            if algorithm == 'linear':
                itor = eval(itor_name)(evaluator, self.n_axes_points, self.axes_min, self.axes_max, is_barycentric)
            else:
                itor = eval(itor_name)(evaluator, self.n_axes_points, self.axes_min, self.axes_max)
        except (ValueError, NameError):
            # 32-bit index type did not succeed: either total amount of points is out of range or has not been compiled
            # try 64 bit now raising exception this time if goes wrong:
            if np.prod(np.array(self.n_axes_points), dtype=np.float64) < np.iinfo(np.int64).max:
                itor_name = itor_name.replace('interpolator_i', 'interpolator_l')
            else:
                itor_name = itor_name.replace('interpolator_i', 'interpolator_ll')
            try:
                if algorithm == 'linear':
                    itor = eval(itor_name)(evaluator, self.n_axes_points, self.axes_min, self.axes_max, is_barycentric)
                else:
                    itor = eval(itor_name)(evaluator, self.n_axes_points, self.axes_min, self.axes_max)
            except (ValueError, NameError):
                # if 64-bit index also failed, probably the combination of required n_ops and n_dims
                # was not instantiated/exposed. In this case substitute general implementation of interpolator
                itor = eval("multilinear_adaptive_cpu_interpolator_general")(evaluator, self.n_axes_points,
                                                                             self.axes_min, self.axes_max, n_dims, n_ops)
                general = True

        if self.cache:
            # create unique signature for interpolator
            itor_cache_signature = "%s_%s_%s_%d_%d_%s" % (type(evaluator).__name__, mode, precision, n_dims, n_ops, region)
            # geenral itor has a different point_data format
            if general:
                itor_cache_signature += "_general_"
            for dim in range(n_dims):
                itor_cache_signature += "_%d_%e_%e" % (self.n_axes_points[dim], self.axes_min[dim], self.axes_max[dim])
            # compute signature hash to uniquely identify itor parameters and load correct cache
            itor_cache_signature_hash = str(hashlib.md5(itor_cache_signature.encode()).hexdigest())
            itor_cache_filename = 'obl_point_data_' + itor_cache_signature_hash + '.pkl'

            if hasattr(self, 'cache_dir'):
                itor_cache_filename = os.path.join(self.cache_dir, itor_cache_filename)
            # if cache file exists, read it
            if os.path.exists(itor_cache_filename):
                with open(itor_cache_filename, "rb") as fp:
                    print("Reading cached point data for ", type(itor).__name__, 'from', itor_cache_filename)
                    itor.point_data = pickle.load(fp)
                    print(len(itor.point_data.keys()), "points loaded")
                    cache_loaded = 1
            if mode == 'adaptive':
                # for adaptive itors, delay obl data save moment, because
                # during simulations new points will be evaluated.
                # on model destruction (or interpreter exit), itor point data will be written to disk
                self.created_itors.append((itor, itor_cache_filename))

        itor.init()
        # for static itors, save the cache immediately after init, if it has not been already loaded
        # otherwise, there is no point to save the same data over and over
        if self.cache and mode == 'static' and not cache_loaded:
            with open(itor_cache_filename, "wb") as fp:
                print("Writing point data for ", type(itor).__name__)
                pickle.dump(itor.point_data, fp, protocol=4)

        self.create_itor_timers(itor, timer_name)
        return itor

    def create_itor_timers(self, itor: operator_set_gradient_evaluator_iface, timer_name: str):
        """
        Create timers for interpolators.

        :param itor: The object which performs evaluation of operator gradient (interpolators currently, AD-based in future)
        :type itor: operator_set_gradient_evaluator_iface object
        :param timer_name: Timer name to be used for the given interpolator
        :type timer_name: str
        """
        try:
            # in case this is a subsequent call, create only timer node for the given timer
            self.timer.node["jacobian assembly"].node["interpolation"].node[timer_name] = timer_node()
        except:
            # in case this is first call, create first only timer nodes for jacobian assembly and interpolation
            self.timer.node["jacobian assembly"] = timer_node()
            self.timer.node["jacobian assembly"].node["interpolation"] = timer_node()
            self.timer.node["jacobian assembly"].node["interpolation"].node[timer_name] = timer_node()

        # assign created timer to interpolator
        itor.init_timer_node(self.timer.node["jacobian assembly"].node["interpolation"].node[timer_name])

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
                if os.path.basename(fname) == fname: # could already have a folder in fname
                    filename = os.path.join(self.cache_dir, fname)
            with open(filename, "wb") as fp:
                print("Writing point data for ", type(itor).__name__, 'to', filename)
                pickle.dump(itor.point_data, fp, protocol=4)

    def body_path_start(self, output_folder):
        """
        Function that prepare hypercube output demonstrating occupancy of state space (for adaptive interpolators)

        :param output_folder: folder to write output to
        """
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        with open(os.path.join(output_folder, 'body_path.txt'), "w") as fp:
            itor = self.acc_flux_itor[0]
            self.processed_body_idxs = set()
            for id in range(self.n_vars):
                fp.write('%d %lf %lf %s\n' % (self.n_axes_points[id],
                                              self.axes_min[id],
                                              self.axes_max[id],
                                              self.vars[id]))
            fp.write('Body Index Data\n')

    def body_path_add_bodys(self, output_folder, time):
        """
        Function performs hypercube output demonstrating occupancy of state space (for adaptive interpolators)

        :param output_folder: folder to write output to
        :param time: current time
        """
        with open(os.path.join(output_folder, 'body_path.txt'), "a") as fp:
            fp.write('T=%lf\n' % time)
            itor = self.acc_flux_itor[0]
            all_idxs = set(itor.get_hypercube_indexes())
            new_idxs = all_idxs - self.processed_body_idxs
            for i in new_idxs:
                fp.write('%d\n' % i)
            self.processed_body_idxs = all_idxs

    def __del__(self):
        # first write cache
        if self.cache:
            self.write_cache()
        # Now destroy all objects in physics
        for name in list(vars(self).keys()):
            delattr(self, name)
