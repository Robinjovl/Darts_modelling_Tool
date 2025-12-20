import abc
import atexit
from math import pi

import numpy as np

from darts.engines import conn_mesh, ms_well, ms_well_vector, timer_node, value_vector
from darts.pipes.define_pipe_geometry import PipeGeometry


class ReservoirBase:
    """
    Base class for generating a mesh
    """

    mesh: conn_mesh
    wells: ms_well_vector = []

    def __init__(self, timer: timer_node, cache: bool = False):
        # Initialize timer for initialization and caching
        self.timer = timer.node["initialization"]

        self.cache = cache
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
        if not hasattr(self, "mesh"):
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

    @abc.abstractmethod
    def set_boundary_volume(self, boundary_volumes: dict):
        """
        Function to set size of volume for boundary cells

        :param boundary_volumes: Dictionary that contains boundary cells with assigned volume
        :type boundary_volumes: dict
        """
        pass

    def add_well(
        self,
        well_name: str,
        ms_well_type: ms_well.MS_Type,
        well_diameter: float = 0.15,
        well_geometry: PipeGeometry = None,
    ) -> None:
        """
        Function to add :class:`ms_well` object to list of wells and generate list of perforations

        :param well_name: Well name
        :type well_name: str
        :param ms_well_type: Type of the multi-segment well model:
        ms_well.MS_Type.EPM: For the Equivalent Porous Medium model
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
            # First put only area here, to be multiplied by segment length later. segment_volume is the volume of
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
        multi_segment: bool = False,
        verbose: bool = False,
    ):
        """
        Function to add a perforation to the well

        :param well_name: Name of well to add perforation to
        :type well_name: str
        :param res_cell_idx: Index of reservoir cell to be perforated
        :type res_cell_idx: int or tuple
        :param well_seg_idx: Index of well segment to be perforated
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
        :param multi_segment: Whether the EPM well model uses a separate well segment per perforation, or a single
        well segment for all perforations of the wellbore.
        :type multi_segment: bool
        :param verbose: Switch to set verbose level
        :type verbose: bool
        """
        pass

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
    def output_to_plt(
        self,
        data: dict,
        output_props: list = None,
        lims: dict = None,
        fig=None,
        figsize: tuple = None,
        axs_shape: tuple = None,
        aspect_ratio: str = "equal",
        logx: bool = False,
        plot_zeros: bool = True,
        cmap: str = "jet",
        colorbar_loc: str = "right",
    ):
        """
        Method for plotting output using matplotlib library.
        Implementation is specific to inherited Reservoir classes

        :param data: Data for output
        :type data: dict
        :param output_props: List of properties to plot
        :type output_props: list
        :param lims: Dictionary of lists with [lower, upper] limits for output variables, will default to [None, None]
        :type lims: dict
        :param fig: Figure object, default is None
        :param figsize: Tuple of (width, height) for figure
        :param axs_shape: Tuple of (rows, columns) for figure
        :param aspect_ratio: Aspect ratio ('equal', 'auto', or float), default is 'equal'
        :param logx: Bool to plot x-axis in logscale, default is False
        :param plot_zeros: Bool to plot zero values, default is True
        :param cmap: plt.Colourmap, default is 'jet'
        :param colorbar_loc: Location of colorbar ('right' or 'bottom'), default is 'right'
        """
        pass

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
