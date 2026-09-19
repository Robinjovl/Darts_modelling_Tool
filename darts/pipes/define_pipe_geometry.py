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

        if not isinstance(segment_lengths, list | np.ndarray):
            raise TypeError(
                f"segment_lengths of the pipe {pipe_name} is neither a list nor a numpy array!"
            )
        segment_lengths = np.array(segment_lengths, dtype=float, copy=True)
        if segment_lengths.ndim != 1 or segment_lengths.size < 2:
            raise ValueError("A pipe must contain at least two segments")
        if not np.all(np.isfinite(segment_lengths)) or np.any(segment_lengths <= 0):
            raise ValueError("All pipe segment lengths must be finite and positive")
        self.segment_lengths = segment_lengths

        if isinstance(inclination_angle, list):
            self.inclination_angle = np.array(inclination_angle)
        elif isinstance(inclination_angle, float | numpy.ndarray):
            self.inclination_angle = inclination_angle
        else:
            raise TypeError(
                f"inclination_angle of the pipe {pipe_name} is neither a list nor a numpy array nor a float!"
            )
        if not np.all(np.isfinite(self.inclination_angle)):
            raise ValueError("All pipe inclination angles must be finite")

        self.inclination_angle_degree = (
            inclination_angle  # 0 for a vertical pipe, 90 for a horizontal pipe
        )
        self.pipe_ID = float(pipe_ID)
        self.wall_roughness = float(wall_roughness)
        if not np.isfinite(self.pipe_ID) or self.pipe_ID <= 0:
            raise ValueError("pipe_ID must be finite and positive")
        if not np.isfinite(self.wall_roughness) or self.wall_roughness < 0:
            raise ValueError("wall_roughness must be finite and non-negative")

        # Calculate additional geometry properties
        self.pipe_length = np.sum(self.segment_lengths)
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
            measured_depth_faces = np.concatenate(
                ([0.0], np.cumsum(self.segment_lengths))
            )
            self.TVD_faces = measured_depth_faces * np.cos(
                self.inclination_angle_radian
            )
            self.TVD_segments = 0.5 * (self.TVD_faces[:-1] + self.TVD_faces[1:])
            self.TVD_interfaces = self.TVD_faces[1:-1]
            self.xyz_nodes = np.column_stack(
                (
                    measured_depth_faces * np.sin(self.inclination_angle_radian),
                    np.zeros_like(measured_depth_faces),
                    self.TVD_faces,
                )
            )
        elif isinstance(self.inclination_angle_radian, numpy.ndarray):
            # This condition is satisfied when class PETREL_PipeGeometry is used where we could have multiple
            # inclination angles and these variables are evaluated in the constructor of that class.
            if self.inclination_angle_radian.shape != (self.num_interfaces,):
                raise ValueError(
                    "An inclination-angle array must contain one value per internal interface"
                )

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
        if not isinstance(num_segments, int) or num_segments < 2:
            raise ValueError("num_segments must be an integer greater than one")

        df = pd.read_csv(well_traj_file_name, sep=r"\s+", comment="#")
        required_columns = {"MD", "X", "Y", "Z"}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(
                f"Trajectory file is missing columns: {sorted(missing_columns)}"
            )
        if len(df) < 2 or not np.all(
            np.isfinite(df[list(required_columns)].to_numpy(dtype=float))
        ):
            raise ValueError(
                "Trajectory must contain at least two finite survey points"
            )
        if np.any(np.diff(df["MD"].to_numpy(dtype=float)) <= 0):
            raise ValueError("Trajectory MD values must be strictly increasing")

        # Get min and max MDs
        min_MD = df["MD"].min()
        max_MD = df["MD"].max()

        # Compute segment length
        segments_length = np.abs(max_MD - min_MD) / num_segments
        segment_lengths = segments_length * np.ones(num_segments)

        face_MDs = np.linspace(min_MD, max_MD, num_segments + 1)
        face_points = np.array(
            [
                [point["X"], point["Y"], point["Z"]]
                for point in (self._interpolate_point(df, md) for md in face_MDs)
            ],
            dtype=float,
        )

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
        self.TVD_faces = TVD_faces
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

        # Store the sampled trajectory relative to the first survey point. The
        # simulator uses positive Z/TVD downward; VTP output flips that sign.
        self.xyz_nodes = face_points - face_points[0]

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
