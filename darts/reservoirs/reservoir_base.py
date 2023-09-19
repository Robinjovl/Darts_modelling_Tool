import abc
from math import pi
import numpy as np
import pickle
import atexit
from typing import Union

from darts.engines import conn_mesh, timer_node, ms_well_vector, ms_well

from dataclasses import dataclass


class ReservoirBase:
    """
    Base class for generating a mesh
    """
    @dataclass
    class Perforation:
        well_name: str
        cell_index: Union[tuple, int]
        well_radius: float
        well_index: float
        well_indexD: float
        segment_direction: str = 'z_axis'
        skin: float = 0.
        multi_segment: bool = False

    wells: ms_well_vector = []
    perforations: list = []

    def __init__(self, timer: timer_node, cache: bool = False):
        # Initialize timer for initialization and caching
        self.timer = timer.node["initialization"]
        self.cache = cache

        # is used on destruction to save cache data
        if self.cache:
            self.created_itors = []
            atexit.register(self.write_cache)

    def init_reservoir(self, verbose: bool = False) -> conn_mesh:
        """
        Generic function to initialize reservoir.

        It calls discretize() to generate mesh object and adds the wells with perforations to the mesh.
        """
        mesh = self.discretize()
        return mesh

    @abc.abstractmethod
    def discretize(self) -> conn_mesh:
        """
        Function to generate discretized mesh

        This function is virtual, needs to be overloaded in derived Reservoir classes

        :returns: Mesh object
        :rtype: conn_mesh
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

    def add_well(self, name: str, perf_list: list, well_radius: float = 0.1524,
                 wellbore_diameter: float = 0.15, well_index: float = None, well_indexD: float = None,
                 segment_direction: str = 'z_axis', skin: float = 0, multi_segment: bool = False) -> None:
        """
        Function to add :class:`ms_well` object to list of wells and generate list of perforations

        :param name: Well name
        :param perf_cell_idxs: Set of cells to perforate, (i, j, k)
        :param well_radius:
        :param wellbore_diameter:
        :param well_index:
        :param well_indexD:
        :param segment_direction:
        :param skin:
        :param multi_segment:
        """
        well = ms_well()
        well.name = name

        # first put only area here, to be multiplied by segment length later
        well.segment_volume = pi * wellbore_diameter ** 2 / 4

        # also to be filled up when the first perforation is made
        well.well_head_depth = 0
        well.well_body_depth = 0
        well.segment_depth_increment = 0
        self.wells.append(well)

        if isinstance(perf_list, (tuple, int, np.ndarray)):
            perf_list = [perf_list]

        for p, perf_idx in enumerate(perf_list):
            self.perforations.append(self.Perforation(well_name=name, cell_index=perf_idx, well_radius=well_radius,
                                                      well_index=well_index, well_indexD=well_indexD,
                                                      segment_direction=segment_direction, skin=skin,
                                                      multi_segment=multi_segment))
        return

    @abc.abstractmethod
    def add_perforations(self, mesh: conn_mesh, verbose: bool = False) -> None:
        """
        Function to add perforations to well objects.

        :param mesh: Mesh object
        :type mesh: conn_mesh
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

    def init_wells(self, mesh: conn_mesh, verbose: bool = False) -> ms_well_vector:
        """
        Function to initialize wells.

        Adds perforations to the wells, adds well objects to the mesh object
        and prepares mesh object for running simulation

        :param mesh: conn_mesh object
        :param verbose: Switch to set verbose level
        """
        self.add_perforations(mesh, verbose)

        for w in self.wells:
            assert (len(w.perforations) > 0), "Well %s does not perforate any active reservoir blocks" % w.name
        mesh.add_wells(ms_well_vector(self.wells))

        mesh.reverse_and_sort()
        mesh.init_grav_coef()
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
