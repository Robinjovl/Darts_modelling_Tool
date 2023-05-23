import numpy as np
import math
import warnings
# from dataclasses import dataclass, is_dataclass, field
from .shapes import *
from .wells import *

import gmsh


class Geometry:
    def __init__(self, dim, axs=[0, 1, 2]):
        self.dim = dim
        self.axs = axs

        self.points = []
        self.curves = []
        self.surfaces = []
        self.volumes = []
        self.physical_points = []
        self.physical_curves = []
        self.physical_surfaces = []
        self.physical_volumes = []

        self.holes = []

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

            self.physical_points = shape.physical_points
            self.physical_curves = shape.physical_curves
            self.physical_surfaces = shape.physical_surfaces
            self.physical_volumes = shape.physical_volumes
        else:
            points_map = {}
            curves_map = {}
            surfaces_map = {}
            volumes_map = {}

            # Check if Points are already in self.points; if not, add point. Map index
            for i, point in enumerate(shape.points):
                if point.xyz in self.points_list:
                    point_idx = self.points_list.index(point.xyz) + 1
                    points_map[point.idx] = point_idx
                else:
                    p = Point(len(self.points) + 1, point.xyz, point.lc, embed=point.embed)
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
        raise Exception("Didn't find well surface")

    def refine_around_point(self, point, radius, lc):
        # Check if radius of cylinder is overlapping or intersecting with any points or curves
        # to make sure mesh around the well is refined enough if close to a layer boundary,

        curves_map = {}
        curves_temp = self.curves[:]  # need new list to avoid infinite loop with appending segments

        # Find segments that are intersected by circle
        for c, curve in enumerate(curves_temp):
            if curve.active and not curve.curve_type == 'circle':  # and not segment.index in new_segments:
                # check if circle intersects segment
                # first construct equation for line segment
                point0 = curve.points[0] - 1
                point1 = curve.points[1] - 1
                x1 = self.points[point0].xyz[self.axs[0]]  # x coordinate of first point in segment
                x2 = self.points[point1].xyz[self.axs[0]]
                y1 = self.points[point0].xyz[self.axs[1]]
                y2 = self.points[point1].xyz[self.axs[1]]
                pointa_index = self.points[point0].idx
                pointb_index = self.points[point1].idx

                intersecting = [False, False]

                if x1 == x2:  # vertical segment
                    # line segment: x = x1
                    if np.abs(x1 - point[self.axs[0]]) == radius:  # touching circle
                        X1 = point[self.axs[0]]
                        Y1 = point[self.axs[1]]
                        if np.sign(Y1 - y1) != np.sign(Y1 - y2):
                            intersecting[0] = True
                    elif np.abs(x1 - point[self.axs[0]]) < radius:  # intersecting circle
                        X1 = x1
                        X2 = X1
                        Y1 = point[self.axs[1]] + np.sqrt(radius ** 2 - (x1 - point[self.axs[0]]) ** 2)
                        Y2 = point[self.axs[1]] - np.sqrt(radius ** 2 - (x1 - point[self.axs[0]]) ** 2)
                        if np.sign(Y1 - y1) != np.sign(Y1 - y2):  # upper intersection is between y1 and y2
                            intersecting[0] = True
                        if np.sign(Y2 - y1) != np.sign(Y2 - y2):  # lower intersection is between y1 and y2
                            intersecting[1] = True

                else:  # non-vertical segment
                    # construct line segment y = ax + b
                    a = (y2 - y1) / (x2 - x1)
                    b = y1 - a * x1

                    # intersection of circle (x-xc)**2 + (y-yc)**2 = r**2 and y = ax + b
                    # (1 + a**2)*x**2 + (2*a*(b-yc) - 2*xc)*x + xc**2 + (b-yc)**2 - r**2 = 0
                    # if D < 0: no intersection, if D = 0: touching, if D > 0: intersection
                    A = 1 + a ** 2
                    B = 2 * a * (b - point[self.axs[1]]) - 2 * point[self.axs[0]]
                    C = point[self.axs[0]] ** 2 + (b - point[self.axs[1]]) ** 2 - radius ** 2
                    D = B ** 2 - 4 * A * C
                    if D == 0:  # touching circle
                        X1 = -B / (2 * A)
                        Y1 = a * X1 + b
                        if np.sign(Y1 - y1) != np.sign(Y1 - y2):
                            intersecting[0] = True
                    elif D > 0:  # intersecting circle
                        X1 = (-B + np.sqrt(D)) / (2 * A)
                        Y1 = a * X1 + b
                        X2 = (-B - np.sqrt(D)) / (2 * A)
                        Y2 = a * X2 + b
                        if np.sign(Y1 - y1) != np.sign(Y1 - y2):
                            intersecting[0] = True
                        if np.sign(Y2 - y1) != np.sign(Y2 - y2):
                            intersecting[1] = True

                if intersecting[0] and intersecting[1]:
                    xyz = [0., 0., 0.]  # create point at intersection
                    xyz[self.axs[0]] = X1
                    xyz[self.axs[1]] = Y1
                    point1_index = len(self.points) + 1
                    self.points.append(Point(point1_index, xyz, lc=lc))
                    self.points_list.append(xyz)

                    xyz = [0., 0., 0.]  # create point at intersection
                    xyz[self.axs[0]] = X2
                    xyz[self.axs[1]] = Y2
                    point2_index = len(self.points) + 1
                    self.points.append(Point(point2_index, xyz, lc=lc))
                    self.points_list.append(xyz)

                    # determine order of points in segment
                    if np.sqrt((X1 - x1) ** 2 + (Y1 - y1) ** 2) < np.sqrt(
                            (X2 - x1) ** 2 + (Y2 - y1) ** 2):  # which is closer to x1, y1
                        segment1 = [pointa_index, point1_index]
                        segment2 = [point1_index, point2_index]
                        segment3 = [point2_index, pointb_index]
                    else:
                        segment1 = [pointa_index, point2_index]
                        segment2 = [point2_index, point1_index]
                        segment3 = [point1_index, pointb_index]

                    self.curves[c].active = False  # remove curve c from active curves
                    index1 = len(self.curves) + 1
                    self.curves.append(Curve(index1, curve_type='line', points=segment1))
                    self.curves_list.append(segment1)
                    index2 = len(self.curves) + 1
                    self.curves.append(Curve(index2, curve_type='line', points=segment2))
                    self.curves_list.append(segment2)
                    index3 = len(self.curves) + 1
                    self.curves.append(Curve(index3, curve_type='line', points=segment3))
                    self.curves_list.append(segment3)

                    # Add curve c to curves map to put correct curve in surfaces
                    curves_map[c + 1] = [index1, index2, index3]

                elif intersecting[0] or intersecting[1]:  # circle is intersecting segment only once
                    xyz = [0., 0., 0.]
                    if intersecting[0]:  # and not intersecting[1]:
                        xyz[self.axs[0]] = X1
                        xyz[self.axs[1]] = Y1
                    else:  # if intersecting[1]:  # and not intersecting[0]:
                        xyz[self.axs[0]] = X2
                        xyz[self.axs[1]] = Y2
                    point_index = len(self.points) + 1
                    self.points.append(Point(point_index, xyz, lc=lc))
                    self.points_list.append(xyz)

                    self.curves[c].active = False  # remove curve c from active curves
                    segment1 = [pointa_index, point_index]
                    index1 = len(self.curves) + 1
                    self.curves.append(Curve(index1, curve_type='line', points=segment1))
                    self.curves_list.append(segment1)
                    segment2 = [point_index, pointb_index]
                    index2 = len(self.curves) + 1
                    self.curves.append(Curve(index2, curve_type='line', points=segment2))
                    self.curves_list.append(segment2)

                    # Add curve c to curves map to put correct curve in surfaces
                    curves_map[c + 1] = [index1, index2]

        # Put subsegments in surfaces
        for curve in curves_map:
            new_segments = curves_map[curve]
            for surface in self.surfaces:
                if curve in surface.curves:
                    new_curves = []
                    # Replace intersected segments by subsegments
                    for old_curve in surface.curves:
                        if old_curve == curve:
                            for seg in new_segments:
                                new_curves.append(seg)
                        else:
                            new_curves.append(old_curve)
                    surface.curves = new_curves
                elif -curve in surface.curves:
                    new_curves = []
                    # Replace intersected segments by subsegments
                    for old_curve in surface.curves:
                        if old_curve == -curve:
                            for seg in range(len(new_segments)):
                                seg_index = len(new_segments) - 1 - seg
                                new_curves.append(-1 * new_segments[seg_index])
                        else:
                            new_curves.append(old_curve)
                    surface.curves = new_curves

        return
