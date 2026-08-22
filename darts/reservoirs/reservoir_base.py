import abc
import atexit
from math import pi

import numpy as np

from darts.engines import conn_mesh, ms_well, ms_well_vector, timer_node, value_vector
from darts.pipes.define_pipe_geometry import PipeGeometry


class BoundaryVolumeDict(dict):
    """``face name -> boundary volume`` map that latches once the engine ran.

    The "huge boundary volume" trick (an open / constant-state far field, see
    :class:`~darts.models.conditions.ConstantStateBC`) is expressed by writing
    a very large volume into the boundary cells. The engine caches the pore
    volume ``PV = volume * poro`` ONCE, inside ``engine.init()``, so a volume
    written afterwards is silently ignored. This dict therefore refuses writes
    after :meth:`ReservoirBase.freeze_pore_volumes` has been called (the
    conditions layer calls it at the end of ``DartsModel.init()``, i.e. right
    after the engine cached ``PV``) instead of letting the change disappear.
    """

    #: class-level default, set per-instance by ``freeze_pore_volumes()``
    frozen = False

    def _check_mutable(self, key):
        if self.frozen:
            raise RuntimeError(
                f"boundary_volumes['{key}'] was assigned after the engine was "
                "initialized. The engine caches the pore volume "
                "PV = volume * poro once, in engine.init() (DartsModel.reset(), "
                "called from DartsModel.init()), so this change would be "
                "SILENTLY IGNORED. Set the boundary volumes before "
                "model.init() -- typically in set_reservoir(), since the "
                "reservoir applies them inside discretize(). If the engine is "
                "deliberately re-initialized afterwards, call "
                "reservoir.allow_pore_volume_updates() first."
            )

    def __setitem__(self, key, value):
        self._check_mutable(key)
        super().__setitem__(key, value)

    def update(self, *args, **kwargs):
        for key in dict(*args, **kwargs):
            self._check_mutable(key)
        super().update(*args, **kwargs)


class ReservoirBase:
    """
    Base class for generating a mesh
    """

    #: Latched by :meth:`freeze_pore_volumes` once the engine has cached
    #: ``PV = volume * poro``; class-level default so reservoirs that do not
    #: call ``ReservoirBase.__init__`` still behave.
    pore_volumes_frozen = False

    mesh: conn_mesh
    wells: list[ms_well]

    def __init__(self, timer: timer_node, cache: bool = False):
        # Initialize timer for initialization and caching
        self.timer = timer.node["initialization"]

        self.cache = cache
        self.mesh = None
        self.wells = []

        self.poro, self.permx, self.permy, self.permz = [], [], [], []
        self.hcap, self.rcond = [], []

        # Gravitational acceleration used for potential energy calculation in m/s^2
        self.grav_acceleration_for_spe = 0.0  # in m/s^2

        self.vtk_initialized = False

        # is used on destruction to save cache data
        if self.cache:
            self.created_itors = []
            atexit.register(self.write_cache)

    def init_reservoir(self, verbose: bool = False):
        """
        Generic function to initialize reservoir.

        It calls discretize() to generate mesh object and adds the wells with perforations to the mesh.
        """
        # if block is used to avoid double execution when call init_reservoir explicitly in model and DARTSModel.init()
        if self.mesh is None:
            self.mesh = self.discretize(verbose)
        return

    @abc.abstractmethod
    def discretize(self, verbose: bool = False) -> conn_mesh:
        """
        Function to generate discretized mesh

        This function is virtual, needs to be overloaded in derived Reservoir classes

        :param verbose: Switch for verbose
        :type verbose: bool
        :rtype: conn_mesh
        """
        pass

    def set_layer_properties(self) -> None:
        """
        Function to set properties for different layers, will be called in Reservoir.discretize()

        This function is empty by default, can be overloaded by child classes
        """
        pass

    def set_wells(self, verbose: bool = False):
        """
        Function to predefine wells inside Reservoir class, will be called in DartsModel.set_wells()

        This function is empty by default, can be overloaded by child classes
        """
        pass

    def freeze_pore_volumes(self) -> None:
        """Latch the cell volumes: the engine has cached ``PV = volume * poro``.

        Called once per model by
        :meth:`darts.models.conditions.ConditionSet.compile` at the end of
        ``DartsModel.init()``, which is the single point that is guaranteed to
        run after ``engine.init()``. From here on, any write to the boundary
        volumes would be silently ignored by the engine, so the user-callable
        entry points refuse it (see :meth:`assert_pore_volumes_mutable`).
        """
        self.pore_volumes_frozen = True
        boundary_volumes = getattr(self, "boundary_volumes", None)
        if isinstance(boundary_volumes, BoundaryVolumeDict):
            boundary_volumes.frozen = True

    def allow_pore_volume_updates(self) -> None:
        """Release the latch set by :meth:`freeze_pore_volumes`.

        Only meaningful when the engine is going to be re-initialized (a fresh
        ``engine.init()`` re-caches ``PV`` from ``mesh.volume``).
        """
        self.pore_volumes_frozen = False
        boundary_volumes = getattr(self, "boundary_volumes", None)
        if isinstance(boundary_volumes, BoundaryVolumeDict):
            boundary_volumes.frozen = False

    def assert_pore_volumes_mutable(self, api_name: str) -> None:
        """Raise when cell volumes are written after the engine cached ``PV``.

        :param api_name: name of the entry point being guarded, quoted in the error
        """
        if self.pore_volumes_frozen:
            raise RuntimeError(
                f"{type(self).__name__}.{api_name}() was called after the engine "
                "was initialized. The engine caches the pore volume "
                "PV = volume * poro once, in engine.init() (DartsModel.reset(), "
                "called from DartsModel.init()), so this volume change would be "
                "SILENTLY IGNORED. Set boundary volumes before model.init() -- "
                "typically in set_reservoir(). If the engine is deliberately "
                "re-initialized afterwards, call "
                "reservoir.allow_pore_volume_updates() first."
            )

    @abc.abstractmethod
    def set_boundary_volume(self, boundary_volumes: dict):
        """
        Function to set size of volume for boundary cells

        This is the "huge boundary volume" open / constant-state far field (see
        :class:`~darts.models.conditions.ConstantStateBC`). ORDERING: the engine
        caches ``PV = volume * poro`` once in ``engine.init()``, so the volumes
        must be in ``mesh.volume`` before ``DartsModel.init()`` initializes the
        engine; a later write is silently ignored.

        :param boundary_volumes: Dictionary that contains boundary cells with assigned volume
        :type boundary_volumes: dict
        """
        pass

    def add_well(
        self,
        well_name: str,
        ms_well_type: ms_well.MS_Type = ms_well.MS_Type.EPM,
        well_diameter: float = 0.15,
        well_geometry: PipeGeometry = None,
    ) -> None:
        """
        Function to create an ms_well object and add it to the list of wells

        :param well_name: Well name
        :type well_name: str
        :param ms_well_type: Type of the multi-segment well model:
                             ms_well.MS_Type.EPM: For the Equivalent Porous Medium model (default well type)
                             ms_well.MS_Type.DFM: For the Drift-Flux model
        :type ms_well_type: ms_well.MS_Type
        :param well_diameter: Well inside diameter. If ms_well_type is EPM, this input argument is needed. If
                              ms_well_type is DFM, this will be extracted from well_geometry.
        :type well_diameter: float
        :param well_geometry: Geometry of the well. If ms_well_type is DFM, this input argument must be specified.
        :type well_geometry: PipeGeometry
        """
        well = ms_well()
        well.name = well_name
        well.ms_type = ms_well_type

        if well.ms_type == ms_well.MS_Type.EPM:
            assert well_geometry is None, (
                "For EPM, well_geometry must not be specified!"
            )
            # Large well trans for EPM well model
            well.well_transmissibility = 1e5
            # First put only area in segment_volume to be multiplied by segment length later. segment_volume is the volume of
            # the segment in front of the reservoir cell which is perforated.
            well.segment_volume = pi / 4 * well_diameter**2
            # will be updated in add_perforation
            well.well_head_depth = 0
            well.well_body_depth = 0
            well.segment_depth_increment = 0

        elif well.ms_type == ms_well.MS_Type.DFM:
            assert well_geometry is not None, (
                "For DFM, well_geometry must be specified!"
            )
            # segment_volumes are the volumes of all the segments of the wellbore from the wellhead segment to
            # the lowermost perforated or non-perforated segment.
            well.segment_volumes = value_vector(well_geometry.segment_volumes)
            well.well_transmissibility = well_geometry.pipe_internal_A
            well.segment_depths = value_vector(well_geometry.TVD_segments)
            well.num_segments = well_geometry.num_segments

        self.wells.append(well)

        return

    @abc.abstractmethod
    def add_perforation(
        self,
        well_name: str,
        res_cell_idx: int | tuple,
        well_seg_idx: int = None,
        well_diameter: float = 0.3048,
        well_index: float = None,
        well_indexD: float = None,
        segment_direction: str = "z_axis",
        skin: float = 0.0,
        ms_epm: bool = False,
        flow_law=None,
        verbose: bool = False,
    ):
        """
        Function to add a perforation to the well

        :param well_name: Name of well to add perforation to
        :type well_name: str
        :param res_cell_idx: Index of reservoir cell to be perforated
        :type res_cell_idx: int or tuple
        :param well_seg_idx: Index of well segment to be perforated. Required for DFM wells,
                             where indexing is 1-based and index 1 is the wellhead ghost segment,
                             which cannot be perforated; perforable segments are 2..num_segments.
                             StructReservoir additionally accepts the DFM-only keyword
                             with_peaceman_for_dfm_well to compute the well index with the
                             Peaceman model.
        :type well_seg_idx: int
        :param well_diameter: Internal diameter of the wellbore
        :type well_diameter: float
        :param well_index: Well index, default is calculated inside
        :type well_index: float
        :param well_indexD: Thermal well index, default is calculated inside
        :type well_indexD: float
        :param segment_direction: X-, Y- or Z-direction
        :type segment_direction: str
        :param skin: Skin factor
        :type skin: float
        :param ms_epm: Whether the EPM well model uses a separate well segment per perforation or not (a single well segment for all perforations).
        :type ms_epm: bool
        :param flow_law: Optional flow law computed ENGINE-SIDE for this perforation
                         instead of the Darcy/Peaceman flux, e.g.
                         :class:`~darts.pipes.linear_dfm_well_ipr.LinearIPR`. Any
                         object with a ``to_engine()`` returning a
                         ``darts.engines.perforation_flow_law`` is accepted, as is
                         such a ``perforation_flow_law`` itself. A non-Darcy law
                         requires ``well_index=0.0``: the engine would otherwise
                         assemble the Peaceman flux across the very interface the
                         law carries.
        :type flow_law: object or None
        :param verbose: Switch to set verbose level
        :type verbose: bool
        """
        pass

    @staticmethod
    def _attach_perforation_flow_law(well, perforation_index: int, flow_law):
        """Attach ``flow_law`` to one perforation of ``well``.

        Shared by the concrete ``add_perforation`` implementations so that the
        translation and the validation live in one place. ``flow_law`` is either
        a ``darts.engines.perforation_flow_law`` or anything exposing
        ``to_engine()`` (which is what
        :class:`~darts.pipes.linear_dfm_well_ipr.LinearIPR` provides) -- the
        reservoir package deliberately does not import the flow-law classes, so
        a new law needs no change here.

        :param well: the ``ms_well`` the perforation belongs to
        :param perforation_index: index into ``well.perforations``
        :param flow_law: the law to attach
        """
        engine_law = (
            flow_law.to_engine() if hasattr(flow_law, "to_engine") else flow_law
        )
        well.set_perforation_flow_law(perforation_index, engine_law)

    @abc.abstractmethod
    def find_cell_index(self, coord: list | np.ndarray) -> int:
        """
        Function to find index of cell centre closest to given xyz-coordinates.

        :returns: Global index
        :rtype: int
        """
        pass

    def get_well(self, well_name: str):
        """
        Find well by name

        :param well_name: Well name
        :returns: :class:`ms_well` object
        """
        for w in self.wells:
            if w.name == well_name:
                return w

    def init_wells(self):
        """
        Function to
        - add well blocks to mesh
        - reverse and sort mesh
        - initialize the gravity coefficient used in the Darcy's law
        - initialize specific potential energy
        """
        for w in self.wells:
            assert len(w.perforations) > 0, (
                f"Well {w.name} does not perforate any active reservoir blocks"
            )
        self.mesh.add_wells(ms_well_vector(self.wells))

        # connect perforations of wells (for example, for closed loop geothermal)
        # dictionary: key is a pair of 2 well names; value is a list of well perforation indices to connect
        # example {(well_1.name, well_2.name): [(w1_perf_1, w2_perf_1),(w1_perf_2, w2_perf_2)]}
        if hasattr(self, "connected_well_segments"):
            for well_pair in self.connected_well_segments.keys():
                well_1 = self.get_well(well_pair[0])
                well_2 = self.get_well(well_pair[1])
                for perf_pair in self.connected_well_segments[well_pair]:
                    self.mesh.connect_segments(
                        well_1, well_2, perf_pair[0], perf_pair[1], 1
                    )

        # allocate mesh arrays
        self.mesh.reverse_and_sort()
        self.mesh.init_grav_coef()
        # Initialize specific potential energy at cell centroids and connections for both reservoir and wells
        self.mesh.init_spe(grav_acceleration_for_spe=self.grav_acceleration_for_spe)

    @abc.abstractmethod
    def init_vtk(self, output_directory: str, export_grid_data: bool = True):
        """
        Method to initialize objects required for output into `.vtk` format.
        This method can also export the mesh properties, e.g. porosity, permeability, etc.

        :param output_directory: Path for output
        :type output_directory: str
        :param export_grid_data: Switch for mesh properties output, default is True
        :type export_grid_data: bool
        """
        pass

    @abc.abstractmethod
    def output_to_vtk(
        self,
        ith_step: int,
        t: float,
        output_directory: str,
        prop_names: list,
        data: dict,
    ):
        """
        Function to export results at timestamp t into `.vtk` format.

        :param ith_step: i'th reporting step
        :type ith_step: int
        :param t: Current time [days]
        :type t: float
        :param output_directory: Path to save .vtk file
        :type output_directory: str
        :param prop_names: List of keys for properties
        :type prop_names: list
        :param data: Data for output
        :type data: dict
        """
        pass

    def write_cache(self):
        return

    def __del__(self):
        # first write cache
        if self.cache:
            self.write_cache()
        # Now destroy all objects in Reservoir
        for name in list(vars(self).keys()):
            delattr(self, name)
