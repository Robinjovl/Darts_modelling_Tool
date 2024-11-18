import numpy as np
import math
import warnings
import matplotlib.pyplot as plt
from darts.reservoirs.mesh.geometry.geometry import Geometry

import numba
from numba import jit, njit


@jit(nopython=True)
def _find_surface(xyz, ax1, ax2, points, curves, surfaces):
    """
    Function to find surface of coordinate
    """
    for s, surface in enumerate(surfaces):
        intersections = 0
        for c in surface:
            # check if well y coordinate is in y range of segment
            curve = curves[np.abs(c) - 1]

            point0 = curve[0] - 1
            point1 = curve[1] - 1
            x1 = points[point0][ax1]  # x coordinate of first point in segment
            x2 = points[point1][ax1]
            y1 = points[point0][ax2]
            y2 = points[point1][ax2]
            dy1 = y1 - xyz[ax2]
            dy2 = y2 - xyz[ax2]

            if np.sign(dy1) != np.sign(dy2):
                # find if point is left (above) or right (below) segment
                if xyz[ax1] > x1 and xyz[ax1] > x2:  # both x1 and x2 left of segment
                    intersections += 1
                elif xyz[ax1] > x1 or xyz[ax1] > x2:  # one of two points left of segment
                    b = y1 - (y2 - y1) / (x2 - x1) * x1  # construct line segment
                    y0 = (y2 - y1) / (x2 - x1) * xyz[ax1] + b

                    if np.sign(x1 - x2) == np.sign(y1 - y2):
                        if y0 > xyz[ax2]:
                            intersections += 1
                    else:
                        if y0 < xyz[ax2]:
                            intersections += 1

        if (intersections % 2) != 0:
            return s + 1, True
    # warnings.warn("Didn't find surface. ACTNUM=0")
    return 0, False


# @jit(nopython=True)
# def _find_volume(xyz, points, curves, surfaces, volumes):
#     """
#     Function to find surface of coordinate
#     """
#     for s, surface in enumerate(surfaces):
#         intersections = 0
#         for c in surface.curves:
#             # check if well y coordinate is in y range of segment
#             curve = curves[np.abs(c) - 1]
#
#             point0 = curve.points[0] - 1
#             point1 = curve.points[1] - 1
#             x1 = points[point0].xyz[ax1]  # x coordinate of first point in segment
#             x2 = points[point1].xyz[ax1]
#             y1 = points[point0].xyz[ax2]
#             y2 = points[point1].xyz[ax2]
#             dy1 = y1 - xyz[ax2]
#             dy2 = y2 - xyz[ax2]
#
#             if np.sign(dy1) != np.sign(dy2):
#                 # find if point is left (above) or right (below) segment
#                 if xyz[ax1] > x1 and xyz[ax1] > x2:  # both x1 and x2 left of segment
#                     intersections += 1
#                 elif xyz[ax1] > x1 or xyz[ax1] > x2:  # one of two points left of segment
#                     b = y1 - (y2 - y1) / (x2 - x1) * x1  # construct line segment
#                     y0 = (y2 - y1) / (x2 - x1) * xyz[ax1] + b
#
#                     if np.sign(x1 - x2) == np.sign(y1 - y2):
#                         if y0 > xyz[ax2]:
#                             intersections += 1
#                     else:
#                         if y0 < xyz[ax2]:
#                             intersections += 1
#
#         if (intersections % 2) != 0:
#             return s + 1, True
#     warnings.warn("Didn't find volume. ACTNUM=0")
#     return 0, False


class Structured(Geometry):
    def generate_mesh2(self, nx=1, ny=1, nz=1):
        # Create lists of points, curves, surfaces
        points = np.asarray(self.points_list)
        curves = np.array(self.curves_list)
        surfaces = numba.typed.List(self.surfaces_list)

        # Calculate bounds of grid
        minx, maxx = min(points[:, 0]), max(points[:, 0])
        miny, maxy = min(points[:, 1]), max(points[:, 1])
        minz, maxz = min(points[:, 2]), max(points[:, 2])

        # Calculate x, y and z
        dx, dy, dz = (maxx-minx)/nx, (maxy-miny)/ny, (maxz-minz)/nz
        x, y, z = np.linspace(minx+dx/2, maxx-dx/2, nx), np.linspace(miny+dy/2, maxy-dy/2, ny), np.linspace(minz+dz/2, maxz-dz/2, nz)

        # Define layers and actnum
        cell_to_layer = np.zeros((nx, ny, nz), dtype=int)
        actnum = np.ones((nx, ny, nz), dtype=int)

        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    xyz = np.array([x[i]+dx/2, y[j]+dy/2, z[k]+dz/2])
                    (surface, act) = _find_surface(xyz, self.axs[0], self.axs[1], points, curves, surfaces)
                    cell_to_layer[i, j, k] = surface
                    actnum[i, j, k] = act

        return x, y, z, cell_to_layer, actnum

    # def generate_mesh3(self, nx, ny, nz):
    #     return
