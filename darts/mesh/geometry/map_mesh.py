import numpy as np
from math import pi, asin

from numba import jit


@jit(nopython=True)
def _find_cells_index(xyz, centroids):
    """
    Function to find index of nearest centroid for coordinate xyz
    """
    dist0 = None
    cell_index = None
    for l, centroid in enumerate(centroids):
        dist1 = np.sqrt((xyz[0] - centroid[0]) ** 2 + (xyz[1] - centroid[1]) ** 2 + (xyz[2] - centroid[2]) ** 2)
        if dist0 is None or dist1 < dist0:
            cell_index = l
            dist0 = dist1

    return cell_index


@jit(nopython=True)
def _find_struct_cell_index(centroid: list, x, y, z):
    # Centroid to structured mesh
    for i, X in enumerate(x):
        if X >= centroid[0]:
            break
    for j, Y in enumerate(y):
        if Y >= centroid[1]:
            break
    for k, Z in enumerate(z):
        if Z >= centroid[2]:
            break
    return [i, j, k]


@jit(nopython=True)
def _translate_curvature(centroids):
    """
    Function to translate the centroid coordinates of the curved interface back to flat
    """
    xyz = np.zeros((len(centroids), 3))
    for l, centroid in enumerate(centroids):
        r = np.sqrt(centroid[0] ** 2 + centroid[1] ** 2)  # radius of circle at centroid
        theta = pi / 8 + asin(centroid[0] / r)  # angle from left end of domain

        X = r * theta  # arc length L == x
        Y = 0.
        Z = centroid[2]
        xyz[l, :] = np.array([X, Y, Z])
    return xyz


@jit(nopython=True)
def _find_connections(cells_idxs, conn_0, conn_1):
    connections = []
    for i, cell in enumerate(cells_idxs):
        conn = []
        for j, cell_0 in enumerate(conn_0):
            cell_1 = conn_1[j]
            if cell_0 == cell:
                conn.append(cell_1)
            elif cell_1 == cell:
                conn.append(cell_0)
        connections.append(conn)
    return connections


@jit(nopython=True)
def _find_cells_in_polygon(centroids, points, segments, polygon):
    """
    Function to find cells inside polygon
    """
    cells_idxs = []

    for ith_cell, centroid in enumerate(centroids):
        X, Y, Z = centroid[0], centroid[1], centroid[2]

        intersections = 0
        for s in polygon:
            # check if well z coordinate is in z range of segment
            segment = segments[int(np.abs(s))]
            x1 = points[segment[0]][0]  # x coordinate of first point in segment
            x2 = points[segment[1]][0]
            # y1 = points[segment[0]][1]
            # y2 = points[segment[1]][1]
            z1 = points[segment[0]][2]
            z2 = points[segment[1]][2]
            dz1 = z1 - Z
            dz2 = z2 - Z
            if np.sign(dz1) != np.sign(dz2):
                # find if point is left (above) or right (below) segment
                if X > x1 and X > x2:  # both x1 and x2 right of segment
                    intersections += 1
                elif X > x1 or X > x2:  # one of two points left of segment
                    b = z1 - (z2 - z1) / (x2 - x1) * x1  # construct line segment
                    z0 = (z2 - z1) / (x2 - x1) * X + b

                    if np.sign(x1 - x2) == np.sign(z1 - z2):
                        if z0 > Z:
                            intersections += 1
                    else:
                        if z0 < Z:
                            intersections += 1

        if (intersections % 2) != 0:
            cells_idxs.append(ith_cell)

    return cells_idxs


class MapMesh:
    def __init__(self, centroids):
        self.centroids = centroids
        # self.centroids = _translate_curvature(centroids)

    def find_struct_cell_index(self, centroid: list, x, y, z):
        # Centroid to structured mesh
        return _find_struct_cell_index(centroid, x, y, z)

    def map_from_structured(self, x, y, z):
        # Find structured cell indices for mesh
        idxs = np.zeros((len(self.centroids), 3), dtype=int)
        for ith_cell, centroid in enumerate(self.centroids):
            idx = _find_struct_cell_index(centroid, x, y, z)
            idxs[ith_cell, :] = np.array(idx)
        return idxs

    def find_cells_in_domain(self, xrange: list, yrange: list, zrange: list):
        # Find cells in certain x-y-z range
        idxs = []
        for i, centroid in enumerate(self.centroids):
            if xrange[0] <= centroid[0] <= xrange[1]:
                if yrange[0] <= centroid[1] <= yrange[1]:
                    if zrange[0] <= centroid[2] <= zrange[1]:
                        idxs.append(i)
        return idxs

    def find_nearest_cell_index(self, centroid: np.ndarray):
        # Find nearest cell index
        return _find_cells_index(centroid, self.centroids)

    def map_to_structured(self, x, y, z):
        # Find cell indices corresponding to structured mesh
        """
        Function to translate mesh to structured mesh of certain x-y-z
        """
        structured_mesh_size = (len(x), len(y), len(z))
        idxs = np.zeros(structured_mesh_size)
        for i, xx in enumerate(x):
            for j, yy in enumerate(y):
                for k, zz in enumerate(z):
                    idxs[i, j, k] = _find_cells_index(np.array([xx, yy, zz]), self.centroids)

        return idxs

    def find_nearest_cells(self, centroids):
        idxs = np.zeros(len(centroids))
        for i, centroid in enumerate(centroids):
            idxs[i] = self.find_nearest_cell_index(centroid)
        return idxs

    def find_connections(self, cell_idxs, conn_0, conn_1):
        return _find_connections(cell_idxs, conn_0, conn_1)

    def find_cells_in_polygon(self, points, segments, polygon):
        return _find_cells_in_polygon(self.centroids, points, segments, polygon)
