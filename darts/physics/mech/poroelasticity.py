from darts.engines import *
from darts.physics.super.physics import Compositional

class Poroelasticity(Compositional):
    """
    This is the Physics class for compositional poroelastic simulation.

    It includes:
    - Creating Reservoir, Well, Rate and Property operators and interpolators for P-z or P-T-z compositional simulation
    - Initializing the :class:`super_engine`
    - Setting well controls (rate, bhp)
    - Defining initial and boundary conditions
    """
    def __init__(self, components: list, phases: list, timer: timer_node, n_points: int,
                 min_p: float, max_p: float, min_z: float, max_z: float, min_t: float = None, max_t: float = None,
                 thermal: bool = False, cache: bool = False):
        """
        This is the constructor of the Compositional Physics class.

        It defines the OBL grid for P-z or P-T-z compositional simulation.

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
        :param thermal: Switch for (iso)thermal simulation
        :type thermal: bool
        :param cache: Switch to cache operator values
        :type cache: bool
        """
        super().__init__(components, phases, timer, n_points,
                 min_p, max_p, min_z, max_z, min_t, max_t, thermal, cache)

    def init_physics(self, regions: list = None, output_props=None, discr_type: str = 'tpfa', platform: str = 'cpu',
                     itor_type: str = 'multilinear', itor_mode: str = 'adaptive', itor_precision: str = 'd',
                     discretizer: str = 'new_discretizer'):
        """
        Function to initialize all contained objects within the Physics object.

        :param regions: List of regions. It contains the keys of the `property_containers` and `reservoir_operators` dict
        :type regions: list
        :param output_props: Output property operators object, default is None
        :type output_props:
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
        :param discretizer: name of the discretizer
        :type discretizer: str
        """
        # If no list of regions has been provided, generate it from the keys of self.property_containers dict
        if regions is None:
            regions = [key for key in self.property_containers.keys()]

        # Define operators, set engine, set interpolators and define well controls
        self.n_props = output_props.n_props if output_props is not None else 0
        self.set_operators(regions, output_props)
        self.set_engine(discretizer, platform)
        self.set_interpolators(platform, itor_type, itor_mode, itor_precision)
        self.set_well_controls()
        return


    def set_engine(self, discretizer: str = 'mech_discretizer', platform: str = 'cpu'):
        """
        Function to set :class:`engine_super` object.

        :param discretizer: Which discretizer in use (affect the choice of engine):
        'mech_discretizer' (default) or 'pm_discretizer'
        :type discretizer: str
        :param platform: Switch for CPU/GPU engine, 'cpu' (default) or 'gpu'
        :type platform: str
        """
        self.discretizer_name = discretizer
        if discretizer == 'mech_discretizer':
            if self.thermal:
                self.engine = eval("engine_super_%s%d_%d_t" % (platform, self.nc, self.nph))()
            else:
                self.engine = eval("engine_super_%s%d_%d" % (platform, self.nc, self.nph))()
        elif discretizer == 'pm_discretizer':
            self.engine = eval("engine_pm_%s" % (platform))()

    def init_wells(self, wells):
        """""
        Function to initialize the well rates for each well
        Arguments:
            -wells: well_object array
        """
        for w in wells:
            assert isinstance(w, ms_well)
            # TODO
            # w.init_rate_parameters(self.n_components, self.rate_phases, self.rate_itor)
            w.init_mech_rate_parameters(self.engine.N_VARS, self.engine.P_VAR, self.n_components, self.rate_phases,
                                        self.rate_itor)

    # TODO: add composition
    def set_uniform_initial_conditions(self, mesh, uniform_pressure, uniform_displacement: list):
        assert isinstance(mesh, conn_mesh)
        nb = mesh.n_blocks

        # set initial pressure
        pressure = np.array(mesh.pressure, copy=False)
        pressure.fill(uniform_pressure)
        # set initial displacements
        displacement = np.array(mesh.displacement, copy=False)
        for i in range(self.n_dim):
            displacement[i::self.n_dim] = uniform_displacement[i]

    # TODO: add composition
    def set_nonuniform_initial_conditions(self, mesh, initial_pressure, initial_displacement: list):
        assert isinstance(mesh, conn_mesh)
        nb = mesh.n_blocks
        n_res_blocks = mesh.n_res_blocks

        # set initial pressure
        pressure = np.array(mesh.pressure, copy=False)
        pressure[:n_res_blocks] = initial_pressure
        # set initial displacements
        displacement = np.array(mesh.displacement, copy=False)
        for i in range(self.n_dim):
            displacement[i:self.n_dim * n_res_blocks:self.n_dim] = initial_displacement[i]
