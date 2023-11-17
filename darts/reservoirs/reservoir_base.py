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

        # is used on destruction to save cache data
        if self.cache:
            self.created_itors = []
            atexit.register(self.write_cache)

    def init_reservoir(self, verbose: bool = False):
        """
        Generic function to initialize reservoir.

        It calls discretize() to generate mesh object and adds the wells with perforations to the mesh.
        """
        self.discretize()
        return

    @abc.abstractmethod
    def discretize(self, verbose: bool = False):
        """
        Function to generate discretized mesh

        This function is virtual, needs to be overloaded in derived Reservoir classes

        :param cache: Option to cache mesh discretization
        :type cache: bool
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

        # also to be filled up when the first perforation is made
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

    def init_wells(self, verbose: bool = False) -> ms_well_vector:
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

        self.mesh.reverse_and_sort()
        self.mesh.init_grav_coef()
        return self.wells

    @abc.abstractmethod
    def output_to_vtk(self, output_directory, output_filename, property_data, ith_step):
        pass

    def write_cache(self):
        return

    def __del__(self):
        # first write cache
        if self.cache:
            self.write_cache()
        # Now destroy all objects in physics
        for name in list(vars(self).keys()):
            delattr(self, name)
