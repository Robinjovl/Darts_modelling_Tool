import numpy as np
import math
import warnings
# from dataclasses import dataclass, is_dataclass, field
from darts.reservoirs.mesh.geometry.shapes import *
from darts.reservoirs.mesh.geometry.wells import *

import gmsh


class Geometry:
    def __init__(self, dim, axs=[0, 1, 2]):
        self.dim = dim
        self.axs = axs

        self.points = []
        self.curves = []
        self.surfaces = []
        self.volumes = []
        self.physical_points = {}
        self.physical_curves = {}
        self.physical_surfaces = {}
        self.physical_volumes = {}

        self.holes = []
        self.lc = []

        self.points_list = []
        self.curves_list = []
        self.surfaces_list = []
        self.volumes_list = []

    def add_shape(self, shape: Shape):
        """
        Function to add shapes to Geometry class
        Finds or creates new Points, Curves, Surfaces and Volumes
        """

        if not self.points:
            self.points = shape.points
            for point in self.points:
                self.points_list.append(point.xyz)
            self.curves = shape.curves
            for curve in self.curves:
                self.curves_list.append(curve.points)
            self.surfaces = shape.surfaces
            for surface in self.surfaces:
                self.surfaces_list.append(surface.curves)
            self.volumes = shape.volumes
            for volume in self.volumes:
                self.volumes_list.append(volume.surfaces)

            self.holes = shape.holes
            self.lc = shape.lc
            self.physical_points = shape.physical_points
            self.physical_curves = shape.physical_curves
            self.physical_surfaces = shape.physical_surfaces
            self.physical_volumes = shape.physical_volumes
        else:
            points_map = {}
            curves_map = {}
            surfaces_map = {}
            volumes_map = {}

            lc0 = len(self.lc)
            self.lc += shape.lc

            # Check if Points are already in self.points; if not, add point. Map index
            for i, point in enumerate(shape.points):
                if point.xyz in self.points_list:
                    point_idx = self.points_list.index(point.xyz) + 1
                    points_map[point.idx] = point_idx
                else:
                    p = Point(len(self.points) + 1, point.xyz, lc0 + point.lc, embed=point.embed)
                    self.points.append(p)
                    self.points_list.append(point.xyz)
                    points_map[point.idx] = len(self.points_list)

            # Check if Curves are already in self.curves; if not, add curve. Map index
            for i, curve in enumerate(shape.curves):
                new_curve = []
                for j, point in enumerate(curve.points):
                    point_idx_from_map = points_map[point]
                    new_curve.append(point_idx_from_map)
                reverse = []
                for j in range(len(new_curve)):
                    reverse.append(new_curve[len(new_curve) - 1 - j])

                if new_curve in self.curves_list:
                    curve_idx = self.curves_list.index(new_curve) + 1
                    curves_map[curve.idx] = curve_idx
                elif reverse in self.curves_list:
                    curve_idx = self.curves_list.index(reverse) + 1
                    curves_map[curve.idx] = -curve_idx
                else:
                    c = Curve(len(self.curves) + 1, curve_type=curve.curve_type, points=new_curve, embed=curve.embed)
                    self.curves.append(c)
                    self.curves_list.append(new_curve)
                    curves_map[curve.idx] = len(self.curves_list)

            # Check if Surfaces are already in self.surfaces; if not, add surface. Map index
            for i, surface in enumerate(shape.surfaces):
                # Find curve idxs from curves_map
                curves = []
                for curve_idx in surface.curves:
                    if curve_idx > 0:
                        curve_idx_from_map = curves_map[curve_idx]
                        curves.append(curve_idx_from_map)
                    else:
                        curve_idx_from_map = curves_map[-curve_idx]
                        curves.append(-curve_idx_from_map)

                # Find reverse curve idxs
                reverse = []
                for j in range(len(curves)):
                    reverse.append(-curves[len(curves) - 1 - j])

                # Check if curve loop exists in surfaces_list
                exists = False
                for k, existing_surface in enumerate(self.surfaces_list):
                    if curves[0] in existing_surface:
                        if all(elem in existing_surface for elem in curves):
                            surfaces_map[surface.idx] = k + 1
                            exists = True
                            break
                    elif reverse[0] in existing_surface:
                        if all(elem in existing_surface for elem in reverse):
                            surfaces_map[surface.idx] = k + 1
                            exists = True
                            break

                if not exists:
                    surface_idx = len(self.surfaces) + 1
                    points = []
                    for point in surface.points:
                        point_idx_from_map = points_map[point]
                        points.append(point_idx_from_map)
                    s = Surface(surface_idx, points, curves, plane=surface.plane, embed=surface.embed)
                    self.surfaces.append(s)
                    self.surfaces_list.append(curves)
                    surfaces_map[surface.idx] = len(self.surfaces_list)

                    if surface.in_surfaces:
                        for other_surface_idx in surface.in_surfaces:
                            surface_idx_from_map = surfaces_map[surface.idx]
                            self.surfaces[other_surface_idx-1].holes.append(surface_idx_from_map)
                        self.holes.append(surface_idx_from_map)

            # Check if Volumes are already in self.volumes; if not, add volume. Map index
            for i, volume in enumerate(shape.volumes):
                # Find curve idxs from curves_map
                surfaces = []
                for surface_idx in volume.surfaces:
                    surface_idx_from_map = surfaces_map[surface_idx]
                    surfaces.append(surface_idx_from_map)

                # Check if surface loop exists in volumes_list
                exists = False
                for k, existing_volume in enumerate(self.volumes_list):
                    if surfaces[0] in existing_volume:
                        if all(elem in existing_volume for elem in surfaces):
                            volumes_map[volume.idx] = k + 1
                            exists = True
                            break

                if not exists:
                    volume_idx = len(self.volumes) + 1
                    v = Volume(volume_idx, surfaces)
                    self.volumes.append(v)
                    self.volumes_list.append(surfaces)
                    volumes_map[volume.idx] = len(self.volumes_list)

            if 0:
                self.physical_points = shape.physical_points
                self.physical_curves = shape.physical_curves
                self.physical_surfaces = shape.physical_surfaces
                self.physical_volumes = shape.physical_volumes
        return

    def add_boundary(self, shape: Shape):

        return

    def add_well(self, well: Well):
        # Check if Well type and location are suitable, and add points in curves for refinements close to well
        self.points, self.curves, self.surfaces, self.volumes = well.check_location(self.points, self.curves, self.surfaces, self.volumes)
        well.define_well()

        # Update lists
        self.points_list = []
        for point in self.points:
            self.points_list.append(point.xyz)
        self.curves_list = []
        for curve in self.curves:
            self.curves_list.append(curve.points)
        self.surfaces_list = []
        for surface in self.surfaces:
            self.surfaces_list.append(surface.curves)
        self.volumes_list = []
        for volume in self.volumes:
            self.volumes_list.append(volume.surfaces)

        # Add Well
        self.add_shape(well)

        return

    def add_fractures(self):

        return

    def find_surface(self, point):
        # Find which polygon the well is in and add this polygon to circle_in_polygon list
        for s, surface in enumerate(self.surfaces):
            intersections = 0
            for c in surface.curves:  # segment in polygon:
                # check if well y coordinate is in y range of segment
                curve = self.curves[np.abs(c) - 1]

                point0 = curve.points[0] - 1
                point1 = curve.points[1] - 1
                x1 = self.points[point0].xyz[self.axs[0]]  # x coordinate of first point in segment
                x2 = self.points[point1].xyz[self.axs[0]]
                y1 = self.points[point0].xyz[self.axs[1]]
                y2 = self.points[point1].xyz[self.axs[1]]
                dy1 = y1 - point[self.axs[1]]
                dy2 = y2 - point[self.axs[1]]

                if np.sign(dy1) != np.sign(dy2):
                    # find if point is left (above) or right (below) segment
                    if point[self.axs[0]] > x1 and point[self.axs[0]] > x2:  # both x1 and x2 left of segment
                        intersections += 1
                    elif point[self.axs[0]] > x1 or point[self.axs[0]] > x2:  # one of two points left of segment
                        b = y1 - (y2 - y1) / (x2 - x1) * x1  # construct line segment
                        y0 = (y2 - y1) / (x2 - x1) * point[self.axs[0]] + b

                        if np.sign(x1 - x2) == np.sign(y1 - y2):
                            if y0 > point[self.axs[1]]:
                                intersections += 1
                        else:
                            if y0 < point[self.axs[1]]:
                                intersections += 1

            if (intersections % 2) != 0:
                return [s+1]
        warnings.warn("Didn't find surface for point", point)
        return []
