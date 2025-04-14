import math
import pandas as pd
import numpy

from darts.pipes.units import *

class PipeGeometry:
    def __init__(self, pipe_name: str, segments_lengths, pipe_ID: float, inclination_angle=0,
                 wall_roughness: float = 5e-5*meter(), verbose: bool = False):
        """
        Class constructor to define the geometry of a pipe
        Assumptions:
        The pipe has constant diameter, inclination angle, and wall roughness.

        :param pipe_name: Name of the pipe
        :type pipe_name: str
        :param segments_lengths: Lengths of the segments from top to bottom [meter]
        :type segments_lengths: list or numpy.ndarray
        :param pipe_ID: Internal diameter of the pipe [meter]
        :type pipe_ID: float
        :param inclination_angle: Inclination angle of the pipe relative to vertical direction [degree]
        :type inclination_angle: float or list or numpy.ndarray
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

        if isinstance(inclination_angle, list):
            self.inclination_angle = np.array(inclination_angle)
        elif isinstance(inclination_angle, (float, numpy.ndarray)):
            self.inclination_angle = inclination_angle
        else:
            raise TypeError(f"inclination_angle of the pipe {pipe_name} is neither a list nor a numpy array nor a float!")

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
        self.z_interfaces = np.cumsum(self.segments_lengths)[:-1]   # [:-1] removes the last exterface position

        # Get segments centroids and interfaces positions together
        self.z_seg_interfaces = np.zeros(self.num_segments + self.num_interfaces)
        self.z_seg_interfaces[0::2] = self.z
        self.z_seg_interfaces[1::2] = self.z_interfaces

        if verbose:
            print("** Geometry of the pipe \"%s\" is defined!" % self.pipe_name)

class PETREL_PipeGeometry(PipeGeometry):
    def __init__(self, pipe_name: str, csv_file_name: str, num_segments: int, pipe_ID: float,
                 wall_roughness: float = 5e-5*meter(), verbose: bool = False):
        df = pd.read_csv(csv_file_name, delim_whitespace=True, comment="#")

        # Extract the MD column and get min and max MDs
        min_MD = df["MD"].min()
        max_MD = df["MD"].max()

        # Compute segment length
        segments_length = np.abs(max_MD - min_MD) / num_segments
        segments_lengths = segments_length * np.ones(num_segments)

        # Initialize list for inclination angles
        inclination_angles = []

        # Loop through segments
        for i in range(num_segments):
            start_MD = min_MD + i * segments_length
            end_MD = start_MD + segments_length

            # Filter points within the segment range
            segment_df = df[(df["MD"] >= start_MD) & (df["MD"] <= end_MD)]
            if segment_df.empty:
                inclination_angles.append(np.nan)
                continue

            # Get the start and end point
            start_point = segment_df.iloc[0]
            end_point = segment_df.iloc[-1]

            # Calculate displacement vector components
            dx = end_point["X"] - start_point["X"]
            dy = end_point["Y"] - start_point["Y"]
            dz = end_point["Z"] - start_point["Z"]

            # Inclination angle calculation
            vector_magnitude = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
            cos_theta = dz / vector_magnitude if vector_magnitude != 0 else np.nan
            theta_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # avoid domain errors
            theta_deg = np.degrees(theta_rad)

            inclination_angles.append(theta_deg)

            inclination_angles = np.array(inclination_angles)
            conn_inclination_angles = (inclination_angles[:-1] + inclination_angles[1:]) / 2

        # Create the result DataFrame. This is not used in any part of the code.
        self.segments_info = pd.DataFrame({
                 "Segment": range(1, num_segments + 1),
                 "Start_MD": [min_MD + i * segments_length for i in range(num_segments)],
                 "End_MD": [min_MD + (i + 1) * segments_length for i in range(num_segments)],
                 "Inclination_Degrees": inclination_angles
        })

        super().__init__(pipe_name, segments_lengths, pipe_ID, conn_inclination_angles, wall_roughness, verbose)
