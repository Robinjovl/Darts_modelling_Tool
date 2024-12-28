import abc
from math import pi
import numpy as np
import pickle
import atexit
from typing import Union

from darts.engines import conn_mesh, timer_node, ms_well_vector, ms_well


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
        if not hasattr(self, 'mesh'):  # to avoid double execution when call init_reservoir explicitly in model and DARTSModel.init()
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

    def add_well(self, well_name: str, wellbore_diameter: float = 0.15) -> None:
        """
        Function to add :class:`ms_well` object to list of wells and generate list of perforations

        :param well_name: Well name
        :param wellbore_diameter:
        """
        well = ms_well()
        well.name = well_name

        # first put only area here, to be multiplied by segment length later
        well.segment_volume = pi * wellbore_diameter ** 2 / 4

        # will be updated  in add_perforation
        well.well_head_depth = 0
        well.well_body_depth = 0
        well.segment_depth_increment = 0
        self.wells.append(well)

        return

    @abc.abstractmethod
    def add_perforation(self, well_name: str, cell_index: Union[int, tuple], well_radius: float = 0.1524,
                        well_index: float = None, well_indexD: float = None, segment_direction: str = 'z_axis',
                        skin: float = 0, multi_segment: bool = False, verbose: bool = False):
        """
        Function to add perforations to well objects.

        :param well_name: Name of well to add perforation to
        :type well_name: str
        :param cell_index: Index of cell to be perforated
        :type cell_index: int or tuple
        :param well_radius: Radius of well, default is 0.1524
        :param well_index: Well index, default is calculated inside
        :param well_indexD: Thermal well index, default is calculated inside
        :param segment_direction: X-, Y- or Z-direction, default is `z_axis`
        :param skin: default is 0
        :param multi_segment: default is False
        :param verbose: Switch to set verbose level
        """
        pass

    @abc.abstractmethod
    def find_cell_index(self, coord: Union[list, np.ndarray]) -> int:
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

    def init_wells(self, verbose: bool = False):
        """
        Function to initialize wells.

        Adds perforations to the wells, adds well objects to the mesh object
        and prepares mesh object for running simulation

        :param mesh: conn_mesh object
        :param verbose: Switch to set verbose level
        """
        for w in self.wells:
            assert (len(w.perforations) > 0), "Well %s does not perforate any active reservoir blocks" % w.name
        self.mesh.add_wells(ms_well_vector(self.wells))
        
        # connect perforations of wells (for example, for closed loop geothermal)
        # dictionary: key is a pair of 2 well names; value is a list of well perforation indices to connect
        # example {(well_1.name, well_2.name): [(w1_perf_1, w2_perf_1),(w1_perf_2, w2_perf_2)]}
        if hasattr (self, 'connected_well_segments'):
            for well_pair in self.connected_well_segments.keys():
                well_1 = self.get_well(well_pair[0])
                well_2 = self.get_well(well_pair[1])
                for perf_pair in self.connected_well_segments[well_pair]:
                    self.mesh.connect_segments(well_1, well_2, perf_pair[0], perf_pair[1], 1)
        
        # allocate mesh arrays
        self.mesh.reverse_and_sort()
        self.mesh.init_grav_coef()

    @abc.abstractmethod
    def plot(self, output_idxs: dict, data: np.ndarray, fig=None, lims: dict = None):
        """
        Method for plotting output using matplotlib library.
        Implementation is specific to inherited Reservoir classes

        :param output_idxs: Dictionary of properties with data array indices for output
        :type output_idxs: dict
        :param data: Data for output
        :type data: np.ndarray
        :param fig: Figure object, default is None
        :param lims: Dictionary of lists with [lower, upper] limits for output variables, will default to [None, None]
        :type lims: dict
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
    def output_to_vtk(self, ith_step: int, t: float, output_directory: str, prop_names: list, data: dict):
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
