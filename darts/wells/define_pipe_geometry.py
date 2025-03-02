import math
from darts.wells.units import *

class PipeGeometry:
    def __init__(self, pipe_name: str, segments_lengths, pipe_ID: float, inclination_angle: float = 0,
                 wall_roughness: float = 5e-5*meter(), verbose: bool = False):
        """
        Class constructor to define the geometry of a pipe
        Assumptions:
        The pipe has constant diameter, inclination angle, and wall roughness.

        :param pipe_name: Name of the pipe
        :type pipe_name: str
        :param segments_lengths: Lengths of the segments from bottom to top [meter]
        :type segments_lengths: list or numpy.ndarray
        :param pipe_ID: Internal diameter of the pipe [meter]
        :type pipe_ID: float
        :param inclination_angle: Inclination angle of the pipe relative to vertical direction [degree]
        :type inclination_angle: float
        :param wall_roughness: Wall roughness of the pipe [meter]
        :type wall_roughness: float
        :param verbose: Whether to display extra info about PipeGeometry
        :type verbose: boolean
        """

        self.pipe_name = pipe_name

        if isinstance(segments_lengths, list):
            self.segments_lengths = np.array(segments_lengths)
        elif isinstance(segments_lengths, np.ndarray):
            self.segments_lengths = segments_lengths
        else:
            raise TypeError(f"segments_lengths of the pipe {pipe_name} is neither a list nor a numpy array!")

        self.inclination_angle_degree = inclination_angle   # 0 for a vertical pipe, 90 for a horizontal pipe for now
        self.pipe_ID = pipe_ID
        self.wall_roughness = wall_roughness

        # Calculate additional geometry properties
        self.pipe_length = sum(segments_lengths)
        self.pipe_IR = self.pipe_ID / 2
        self.pipe_internal_A = math.pi * self.pipe_IR ** 2
        self.perimeter = 2 * math.pi * self.pipe_IR
        self.segments_volumes = self.pipe_internal_A * self.segments_lengths
        self.inclination_angle_radian = math.radians(self.inclination_angle_degree)
        self.num_segments = len(self.segments_lengths)
        self.num_interfaces = self.num_segments - 1

        # Get segments centroids
        z = []
        current_z = 0
        for length in self.segments_lengths:
            centroid = current_z + length/2
            z.append(centroid)
            # Move to the starting point of the next segment
            current_z += length
        self.z = np.array(z)

        self.z_m = self.z[0:-1:1]
        self.z_p = self.z[1::1]

        self.D = self.z_p - self.z_m   # Distances between the centroids of neighboring interfaces

        # Duplicate the first and last values of the array, which will be used for exterfaces
        self.D = np.insert(self.D, 0, self.D[0])
        self.D = np.append(self.D, self.D[-1])

        # Get interfaces positions
        self.z_interfaces = np.cumsum(self.segments_lengths)[:-1]

        # Get segments centroids and interfaces positions together
        self.z_seg_interfaces = np.zeros(self.num_segments + self.num_interfaces)
        self.z_seg_interfaces[0::2] = self.z
        self.z_seg_interfaces[1::2] = self.z_interfaces

        if verbose:
            print("** Geometry of the pipe \"%s\" is defined!" % self.pipe_name)
