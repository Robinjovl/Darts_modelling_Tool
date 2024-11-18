import numpy as np
import math
from darts.reservoirs.mesh.geometry.shapes import *


class Well(Shape):
    curves_map = {}

    def check_location(self, points, curves, surfaces, volumes):
        pass

    def refine_location(self, points, curves):
        pass


class WellCell(Well):
    def __init__(self, center: list, lc: float, orientation: str, in_surfaces: list):
        self.lc = [lc]

        re = lc/np.sqrt(3) * 0.999  # The radius of the circumscribed circle is R = side/sqrt(3)

        self.points = [Point(1, center),
                       Point(2, self.calc_radial_points(center, re, orientation, math.radians(60)), embed=in_surfaces),
                       Point(3, self.calc_radial_points(center, re, orientation, math.radians(180)), embed=in_surfaces),
                       Point(4, self.calc_radial_points(center, re, orientation, math.radians(300)), embed=in_surfaces),
                       ]


class CircularWell(Well):
    def __init__(self, center: list, re: float, lc: float, orientation: str, in_surfaces: list):
        self.lc = [lc]

        self.points = [Point(1, center),
                       Point(2, self.calc_radial_points(center, re, orientation, math.radians(0))),
                       Point(3, self.calc_radial_points(center, re, orientation, math.radians(120))),
                       Point(4, self.calc_radial_points(center, re, orientation, math.radians(240))),
                       ]

        self.curves = [Curve(1, curve_type='circle', points=[2, 1, 3], embed=in_surfaces),
                       Curve(2, curve_type='circle', points=[3, 1, 4], embed=in_surfaces),
                       Curve(3, curve_type='circle', points=[4, 1, 2], embed=in_surfaces),
                       ]

        self.surfaces = [Surface(1, points=[2, 3, 4, 2], curves=[1, 2, 3])]


# class Cylindrical2D(Well):
#     def __init__(self, xyz: list, rw: float, re: float, RE: float, orientation: int, lc_well: int):
#         self.xyz = xyz
#         self.rw = rw
#         self.re = re
#         self.RE = RE
#         self.orientation = orientation
#         self.lc_well = lc_well
#
#         # Define axes perpendicular to orientation for circle calculations: axis1 is "x-axis" of this plane, axis2 is y-axis
#         if self.orientation == 0:  # well in x-dir
#             self.axis1 = 1
#             self.axis2 = 2
#         elif self.orientation == 1:  # y-dir
#             self.axis1 = 0
#             self.axis2 = 2
#         else:  # z-dir
#             self.axis1 = 0
#             self.axis2 = 1
#
#         super().__init__(xyz, rw, re, RE, orientation, lc_well)
#
#     def check_location(self, points, curves, surfaces, volumes):
#         # Check if radius of cylinder is overlapping or intersecting with any points or curves
#         # do this twice; first time to check if well cylinder is "free", second time for a larger circle around the well
#         # to make sure mesh around the well is refined enough if close to a layer boundary,
#
#         radii = [self.re, self.RE]
#
#         for i in range(2):
#             curves_temp = curves[:]  # need new list to avoid infinite loop with appending segments
#
#             # Find segments that are intersected by circle
#             for c, curve in enumerate(curves_temp):
#                 if curve.active and not curve.curve_type == 'circle':  # and not segment.index in new_segments:
#                     # check if circle intersects segment
#                     # first construct equation for line segment
#                     point0 = curve.points[0]-1
#                     point1 = curve.points[1]-1
#                     x1 = points[point0].xyz[self.axis1]  # x coordinate of first point in segment
#                     x2 = points[point1].xyz[self.axis1]
#                     y1 = points[point0].xyz[self.axis2]
#                     y2 = points[point1].xyz[self.axis2]
#                     pointa_index = points[point0].idx
#                     pointb_index = points[point1].idx
#
#                     intersecting = [False, False]
#
#                     if x1 == x2:  # vertical segment
#                         # line segment: x = x1
#                         if np.abs(x1 - self.xyz[self.axis1]) == radii[i]:  # touching circle
#                             X1 = self.xyz[self.axis1]
#                             Y1 = self.xyz[self.axis2]
#                             if np.sign(Y1 - y1) != np.sign(Y1 - y2):
#                                 intersecting[0] = True
#                         elif np.abs(x1 - self.xyz[self.axis1]) < radii[i]:  # intersecting circle
#                             X1 = x1
#                             X2 = X1
#                             Y1 = self.xyz[self.axis2] + np.sqrt(radii[i] ** 2 - (x1 - self.xyz[self.axis1]) ** 2)
#                             Y2 = self.xyz[self.axis2] - np.sqrt(radii[i] ** 2 - (x1 - self.xyz[self.axis1]) ** 2)
#                             if np.sign(Y1 - y1) != np.sign(Y1 - y2):  # upper intersection is between y1 and y2
#                                 intersecting[0] = True
#                             if np.sign(Y2 - y1) != np.sign(Y2 - y2):  # lower intersection is between y1 and y2
#                                 intersecting[1] = True
#
#                     else:  # non-vertical segment
#                         # construct line segment y = ax + b
#                         a = (y2 - y1) / (x2 - x1)
#                         b = y1 - a * x1
#
#                         # intersection of circle (x-xc)**2 + (y-yc)**2 = r**2 and y = ax + b
#                         # (1 + a**2)*x**2 + (2*a*(b-yc) - 2*xc)*x + xc**2 + (b-yc)**2 - r**2 = 0
#                         # if D < 0: no intersection, if D = 0: touching, if D > 0: intersection
#                         A = 1 + a ** 2
#                         B = 2 * a * (b - self.xyz[self.axis2]) - 2 * self.xyz[self.axis1]
#                         C = self.xyz[self.axis1] ** 2 + (b - self.xyz[self.axis2]) ** 2 - radii[i] ** 2
#                         D = B ** 2 - 4 * A * C
#                         if D == 0:  # touching circle
#                             X1 = -B / (2 * A)
#                             Y1 = a * X1 + b
#                             if np.sign(Y1 - y1) != np.sign(Y1 - y2):
#                                 intersecting[0] = True
#                         elif D > 0:  # intersecting circle
#                             X1 = (-B + np.sqrt(D)) / (2 * A)
#                             Y1 = a * X1 + b
#                             X2 = (-B - np.sqrt(D)) / (2 * A)
#                             Y2 = a * X2 + b
#                             if np.sign(Y1 - y1) != np.sign(Y1 - y2):
#                                 intersecting[0] = True
#                             if np.sign(Y2 - y1) != np.sign(Y2 - y2):
#                                 intersecting[1] = True
#
#                     if i == 0:
#                         if np.any(intersecting):
#                             raise Exception("Well cylinder intersecting existing curve: radial flow assumption not valid. "
#                                             "Choice of well location and/or type not suitable")
#                     else:  # i==1
#                         if intersecting[0] and intersecting[1]:
#                             xyz = [0., 0., 0.]  # create point at intersection
#                             xyz[self.axis1] = X1
#                             xyz[self.axis2] = Y1
#                             point1_index = len(points) + 1
#                             points.append(Point(point1_index, xyz, lc=self.lc_well))
#
#                             xyz = [0., 0., 0.]  # create point at intersection
#                             xyz[self.axis1] = X2
#                             xyz[self.axis2] = Y2
#                             point2_index = len(points) + 1
#                             points.append(Point(point2_index, xyz, lc=self.lc_well))
#
#                             # determine order of points in segment
#                             if np.sqrt((X1 - x1) ** 2 + (Y1 - y1) ** 2) < np.sqrt(
#                                     (X2 - x1) ** 2 + (Y2 - y1) ** 2):  # which is closer to x1, y1
#                                 segment1 = [pointa_index, point1_index]
#                                 segment2 = [point1_index, point2_index]
#                                 segment3 = [point2_index, pointb_index]
#                             else:
#                                 segment1 = [pointa_index, point2_index]
#                                 segment2 = [point2_index, point1_index]
#                                 segment3 = [point1_index, pointb_index]
#
#                             curves[c].active = False  # remove curve c from active curves
#                             index1 = len(curves) + 1
#                             curves.append(Curve(index1, curve_type='line', points=segment1))
#                             index2 = len(curves) + 1
#                             curves.append(Curve(index2, curve_type='line', points=segment2))
#                             index3 = len(curves) + 1
#                             curves.append(Curve(index3, curve_type='line', points=segment3))
#
#                             # Add curve c to curves map to put correct curve in surfaces
#                             self.curves_map[c + 1] = [index1, index2, index3]
#
#                         elif intersecting[0] or intersecting[1]:  # circle is intersecting segment only once
#                             xyz = [0., 0., 0.]
#                             if intersecting[0]:  # and not intersecting[1]:
#                                 xyz[self.axis1] = X1
#                                 xyz[self.axis2] = Y1
#                             else:  # if intersecting[1]:  # and not intersecting[0]:
#                                 xyz[self.axis1] = X2
#                                 xyz[self.axis2] = Y2
#                             point_index = len(points) + 1
#                             points.append(Point(point_index, xyz, lc=self.lc_well))
#
#                             curves[c].active = False  # remove curve c from active curves
#                             segment1 = [pointa_index, point_index]
#                             index1 = len(curves) + 1
#                             curves.append(Curve(index1, curve_type='line', points=segment1))
#                             segment2 = [point_index, pointb_index]
#                             index2 = len(curves) + 1
#                             curves.append(Curve(index2, curve_type='line', points=segment2))
#
#                             # Add curve c to curves map to put correct curve in surfaces
#                             self.curves_map[c + 1] = [index1, index2]
#
#             # Put subsegments in surfaces
#             for curve in self.curves_map:
#                 new_segments = self.curves_map[curve]
#                 for surface in surfaces:
#                     if curve in surface.curves:
#                         new_curves = []
#                         # Replace intersected segments by subsegments
#                         for old_curve in surface.curves:
#                             if old_curve == curve:
#                                 for seg in new_segments:
#                                     new_curves.append(seg)
#                             else:
#                                 new_curves.append(old_curve)
#                         surface.curves = new_curves
#                     elif -curve in surface.curves:
#                         new_curves = []
#                         # Replace intersected segments by subsegments
#                         for old_curve in surface.curves:
#                             if old_curve == -curve:
#                                 for seg in range(len(new_segments)):
#                                     seg_index = len(new_segments) - 1 - seg
#                                     new_curves.append(-1 * new_segments[seg_index])
#                             else:
#                                 new_curves.append(old_curve)
#                         surface.curves = new_curves
#         self.in_surfaces_ = self.in_surfaces(points, curves, surfaces)
#         return points, curves, surfaces, volumes
#
#     def in_surfaces(self, points, curves, surfaces):
#         # Find which polygon the well is in and add this polygon to circle_in_polygon list
#         for s, surface in enumerate(surfaces):
#             intersections = 0
#             for c in surface.curves:  # segment in polygon:
#                 # check if well y coordinate is in y range of segment
#                 curve = curves[np.abs(c) - 1]
#
#                 point0 = curve.points[0] - 1
#                 point1 = curve.points[1] - 1
#                 x1 = points[point0].xyz[self.axis1]  # x coordinate of first point in segment
#                 x2 = points[point1].xyz[self.axis1]
#                 y1 = points[point0].xyz[self.axis2]
#                 y2 = points[point1].xyz[self.axis2]
#                 dy1 = y1 - self.xyz[self.axis2]
#                 dy2 = y2 - self.xyz[self.axis2]
#
#                 if np.sign(dy1) != np.sign(dy2):
#                     # find if point is left (above) or right (below) segment
#                     if self.xyz[self.axis1] > x1 and self.xyz[self.axis1] > x2:  # both x1 and x2 left of segment
#                         intersections += 1
#                     elif self.xyz[self.axis1] > x1 or self.xyz[self.axis1] > x2:  # one of two points left of segment
#                         b = y1 - (y2 - y1) / (x2 - x1) * x1  # construct line segment
#                         y0 = (y2 - y1) / (x2 - x1) * self.xyz[self.axis1] + b
#
#                         if np.sign(x1 - x2) == np.sign(y1 - y2):
#                             if y0 > self.xyz[self.axis2]:
#                                 intersections += 1
#                         else:
#                             if y0 < self.xyz[self.axis2]:
#                                 intersections += 1
#             print("Intersections", intersections)
#
#             if (intersections % 2) != 0:
#                 return [s+1]
#         raise Exception("Didn't find well surface")
#
#     def define_well(self):
#         # Define small triangle for well cell
#         point1 = self.xyz[:]
#         point1[self.axis1] += 0.01 * self.re
#         point2 = self.xyz[:]
#         point2[self.axis2] += 0.01 * self.re
#
#         self.points = [Point(1, self.xyz),
#                        Point(2, self.calc_radial_points(self.xyz, self.re, self.orientation, math.radians(0)), lc=self.lc_well),
#                        Point(3, self.calc_radial_points(self.xyz, self.re, self.orientation, math.radians(120)), lc=self.lc_well),
#                        Point(4, self.calc_radial_points(self.xyz, self.re, self.orientation, math.radians(240)), lc=self.lc_well),
#                        Point(5, point1),
#                        Point(6, point2)]
#
#         self.curves = [Curve(1, curve_type='circle', points=[2, 1, 3]),
#                        Curve(2, curve_type='circle', points=[3, 1, 4]),
#                        Curve(3, curve_type='circle', points=[4, 1, 2]),
#                        Curve(4, curve_type='line', points=[1, 5]),
#                        Curve(5, curve_type='line', points=[5, 6]),
#                        Curve(6, curve_type='line', points=[6, 1])]
#
#         self.surfaces = [Surface(1, points=[2, 3, 4, 2], curves=[1, 2, 3], in_surfaces=self.in_surfaces_),
#                          Surface(2, points=[1, 5, 6, 1], curves=[4, 5, 6])]
#
#         self.physical_curves = [Physical(tag='radial', idxs=[1, 2, 3])]
#
#         self.physical_surfaces = [Physical(tag='well_cell', idxs=[2])]
#         return


# class Cylindrical3D(Well):
#     def __init__(self):
#         super().__init__()
#
#
# class BCWell(Well):
#     def __init__(self):
#         super().__init__()
#
#
# class WellTrajectory(Well):
#     def __int__(self):
#         super().__init__()