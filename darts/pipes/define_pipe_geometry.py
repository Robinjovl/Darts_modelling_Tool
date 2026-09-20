import math

import numpy
import pandas as pd

from darts.pipes.units import *


class PipeGeometry:
    def __init__(
        self,
        pipe_name: str,
        segment_lengths,
        pipe_ID: float,
        inclination_angle=0.0,
        wall_roughness: float = 5e-5 * meter(),
        verbose: bool = False,
    ):
        """
        Class constructor to define the geometry of a pipe
        Assumptions:
        The pipe has constant diameter and wall roughness.

        :param pipe_name: Name of the pipe
        :type pipe_name: str
        :param segment_lengths: Lengths of the segments from top to bottom [meter]
        :type segment_lengths: list or numpy.ndarray
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

        if isinstance(segment_lengths, list):
            self.segment_lengths = np.array(segment_lengths)
        elif isinstance(segment_lengths, np.ndarray):
            self.segment_lengths = segment_lengths
        else:
            raise TypeError(
                f"segment_lengths of the pipe {pipe_name} is neither a list nor a numpy array!"
            )

        if isinstance(inclination_angle, list):
            self.inclination_angle = np.array(inclination_angle)
        elif isinstance(inclination_angle, float | numpy.ndarray):
            self.inclination_angle = inclination_angle
        else:
            raise TypeError(
                f"inclination_angle of the pipe {pipe_name} is neither a list nor a numpy array nor a float!"
            )

        self.inclination_angle_degree = (
            inclination_angle  # 0 for a vertical pipe, 90 for a horizontal pipe for now
        )
        self.pipe_ID = pipe_ID
        self.wall_roughness = wall_roughness

        # Calculate additional geometry properties
        self.pipe_length = sum(segment_lengths)
        self.pipe_IR = self.pipe_ID / 2
        self.pipe_internal_A = math.pi * self.pipe_IR**2
        self.perimeter = 2 * math.pi * self.pipe_IR
        self.segment_volumes = self.pipe_internal_A * self.segment_lengths
        self.inclination_angle_radian = np.radians(self.inclination_angle_degree)
        self.num_segments = len(self.segment_lengths)
        self.num_interfaces = self.num_segments - 1

        # Get segments centroids
        z = []
        current_z = 0
        for length in self.segment_lengths:
            centroid = current_z + length / 2
            z.append(centroid)
            # Move to the starting point of the next segment
            current_z += length
        self.z = np.array(z)

        self.z_m = self.z[0:-1:1]
        self.z_p = self.z[1::1]

        self.D = (
            self.z_p - self.z_m
        )  # Distances between the centroids of neighboring interfaces

        # Duplicate the first and last values of the array, which will be used for exterfaces
        self.D = np.insert(self.D, 0, self.D[0])
        self.D = np.append(self.D, self.D[-1])

        # Get interfaces positions
        self.z_interfaces = np.cumsum(self.segment_lengths)[
            :-1
        ]  # [:-1] removes the last exterface position

        # Get segments centroids and interfaces positions together
        self.z_seg_interfaces = np.zeros(self.num_segments + self.num_interfaces)
        self.z_seg_interfaces[0::2] = self.z
        self.z_seg_interfaces[1::2] = self.z_interfaces

        if isinstance(self.inclination_angle_radian, float):
            self.TVD_segments = self.z * np.cos(self.inclination_angle_radian)
            self.TVD_interfaces = self.z_interfaces * np.cos(
                self.inclination_angle_radian
            )
        elif isinstance(self.inclination_angle_radian, numpy.ndarray):
            # This condition is satisfied when class PETREL_PipeGeometry is used where we could have multiple
            # inclination angles and these variables are evaluated in the constructor of that class.
            pass

        self.TVD_seg_interfaces = np.zeros(self.num_segments + self.num_interfaces)
        self.TVD_seg_interfaces[0::2] = self.TVD_segments
        self.TVD_seg_interfaces[1::2] = self.TVD_interfaces

        if verbose:
            print(f'** Geometry of the pipe "{self.pipe_name}" is defined!')


class PETREL_PipeGeometry(PipeGeometry):
    """
    This class is used to get the geometry of the well from a PETREL well trajectory file. The number of segments
    of the well is specified by the user and the lengths of all the segments are considered equal.
    """

    def __init__(
        self,
        pipe_name: str,
        well_traj_file_name: str,
        num_segments: int,
        pipe_ID: float,
        wall_roughness: float = 5e-5 * meter(),
        verbose: bool = False,
    ):
        """
        :param pipe_name: Name of the pipe
        :type pipe_name: str
        :param well_traj_file_name: Name of the PETREL well trajectory file
        :type well_traj_file_name: str
        :param num_segments: Number of segments of the pipe
        :type num_segments: int
        :param pipe_ID: Internal diameter of the pipe [meter]
        :type pipe_ID: float
        :param wall_roughness: Wall roughness of the pipe [meter]
        :type wall_roughness: float
        :param verbose: Whether to display extra info about PipeGeometry
        :type verbose: boolean
        """
        df = pd.read_csv(well_traj_file_name, sep=r"\s+", comment="#")

        # Get min and max MDs
        min_MD = df["MD"].min()
        max_MD = df["MD"].max()

        # Compute segment length
        segments_length = np.abs(max_MD - min_MD) / num_segments
        segment_lengths = segments_length * np.ones(num_segments)

        # Initialize list for inclination angles
        inclination_angles_deg = []

        # Loop through segments
        for i in range(num_segments):
            start_MD = min_MD + i * segments_length
            end_MD = start_MD + segments_length

            # Get the start and end point
            start_point = self._interpolate_point(df, start_MD)
            end_point = self._interpolate_point(df, end_MD)

            if start_point is None or end_point is None:
                raise Exception("start_point or end_point is None!")

            # Calculate displacement vector components
            dx = end_point["X"] - start_point["X"]
            dy = end_point["Y"] - start_point["Y"]
            dz = end_point["Z"] - start_point["Z"]

            # Inclination angle calculation
            vector_magnitude = np.sqrt(dx**2 + dy**2 + dz**2)
            cos_theta = dz / vector_magnitude if vector_magnitude != 0 else np.nan
            theta_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # avoid domain errors
            theta_deg = np.degrees(theta_rad)

            inclination_angles_deg.append(theta_deg)

        inclination_angles_deg = np.array(inclination_angles_deg)
        conn_inclination_angles_deg = (
            inclination_angles_deg[:-1] + inclination_angles_deg[1:]
        ) / 2

        # Segments vertical lengths
        vertical_lengths_segments = segments_length * np.cos(
            np.radians(inclination_angles_deg)
        )

        # TVD at each interface: cumulative sum starting from the top
        TVD_faces = np.zeros(num_segments + 1)
        TVD_faces[1:] = np.cumsum(vertical_lengths_segments)
        # Only internal faces: exclude top (0) and bottom (-1)
        self.TVD_interfaces = TVD_faces[1:-1]

        self.TVD_segments = 0.5 * (TVD_faces[:-1] + TVD_faces[1:])

        # Create the result DataFrame. This is not used in any part of the code.
        self.segments_info = pd.DataFrame(
            {
                "Segment": range(1, num_segments + 1),
                "Start_MD": [min_MD + i * segments_length for i in range(num_segments)],
                "End_MD": [
                    min_MD + (i + 1) * segments_length for i in range(num_segments)
                ],
                "Inclination_Degrees": inclination_angles_deg,
            }
        )

        super().__init__(
            pipe_name,
            segment_lengths,
            pipe_ID,
            conn_inclination_angles_deg,
            wall_roughness,
            verbose,
        )

    def _interpolate_point(self, df, target_MD):
        lower = df[df["MD"] <= target_MD].tail(1)
        upper = df[df["MD"] >= target_MD].head(1)

        if lower.empty or upper.empty:
            return None  # Cannot interpolate outside bounds

        if lower["MD"].values[0] == upper["MD"].values[0]:
            return lower.iloc[0]  # Exact match

        # Linear interpolation
        frac = (target_MD - lower["MD"].values[0]) / (
            upper["MD"].values[0] - lower["MD"].values[0]
        )
        interpolated = {}
        for col in ["X", "Y", "Z"]:
            interpolated[col] = lower[col].values[0] + frac * (
                upper[col].values[0] - lower[col].values[0]
            )
        interpolated["MD"] = target_MD
        return pd.Series(interpolated)
