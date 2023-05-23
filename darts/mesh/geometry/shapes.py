import numpy as np
import math
from dataclasses import dataclass, is_dataclass, field


# region geometry dataclasses
@dataclass
class Point:
    idx: int
    xyz: list
    lc: int = 0
    active: bool = True
    embed: list = field(default_factory=list)


@dataclass
class Curve:
    idx: int
    curve_type: str
    points: list
    active: bool = True
    embed: list = field(default_factory=list)


@dataclass
class Surface:
    idx: int
    points: list
    curves: list = field(default_factory=list)
    in_surfaces: list = field(default_factory=list)
    holes: list = field(default_factory=list)
    plane: bool = True
    active: bool = True
    embed: list = field(default_factory=list)


@dataclass
class Volume:
    idx: int
    surfaces: list
    active: bool = True


@dataclass
class Physical:
    tag: str
    idxs: list
# endregion


class Shape:
    def __init__(self):
        self.points = []
        self.curves = []
        self.surfaces = []
        self.volumes = []

        self.physical_points = []
        self.physical_curves = []
        self.physical_surfaces = []
        self.physical_volumes = []

        # self.physical_points = {}
        # self.physical_curves = {}
        # self.physical_surfaces = {}
        # self.physical_volumes = {}

        self.boundary_tag = 90000

        self.define_shapes()

    def define_shapes(self):
        pass

    def connect_points(self):
        """
        Function to connect points in surfaces specified. Points in surfaces must be defined in order
        """
        curves = []
        for curve in self.curves:
            # If self.curves is not empty
            curves.append(curve.points)

        for j, surface in enumerate(self.surfaces):
            points = surface.points
            surface_curves = []
            for i in range(len(points)-1):
                curve_points = [points[i], points[i+1]]
                reverse = [points[i+1], points[i]]

                if curve_points not in curves and reverse not in curves:
                    curve_idx = len(curves) + 1
                    curves.append(curve_points)
                    curve_type = 'line'
                    self.curves.append(Curve(curve_idx, curve_type, curve_points))
                    surface_curves.append(curve_idx)
                else:
                    if curve_points in curves:
                        curve_idx = curves.index(curve_points) + 1
                        surface_curves.append(curve_idx)
                    else:
                        curve_idx = curves.index(reverse) + 1
                        surface_curves.append(-curve_idx)
            self.surfaces[j].curves = surface_curves

        return

    def add_boundary(self, boundary):
        pass

    def plot_shape_2D(self):
        """
        Function to plot point and curves
        """
        import matplotlib.pyplot as plt
        plt.figure(dpi=400, figsize=(10, 5))

        for point in self.points:
            plt.scatter(point.xyz[0], point.xyz[2], c='k', s=0.5)

        for curve in self.curves:
            idx1 = curve.points[0]-1
            idx2 = curve.points[1]-1

            p1 = self.points[idx1].xyz
            p2 = self.points[idx2].xyz

            x = [p1[0], p2[0]]
            z = [p1[2], p2[2]]

            plt.plot(x, z, c='k', linewidth=1)

        # Find limits
        xmin, xmax = self.points[0].xyz[0], self.points[0].xyz[0]
        zmin, zmax = self.points[0].xyz[2], self.points[0].xyz[2]
        for point in self.points:
            xi = point.xyz[0]
            if xi < xmin:
                xmin = xi
            elif xi > xmax:
                xmax = xi

            zi = point.xyz[2]
            if zi < zmin:
                zmin = zi
            elif zi > zmax:
                zmax = zi

        plt.axis('scaled')
        plt.xlim([xmin, xmax])
        plt.ylim([zmin, zmax])
        plt.xlabel("x [m]")
        plt.ylabel("z [m]")

        return


class Box(Shape):
    def __init__(self, xdim: list, ydim: list, zdim: list):
        self.xdim, self.ydim, self.zdim = xdim, ydim, zdim

        super().__init__()

        self.connect_points()

        surface_idxs = [surface.idx for surface in self.surfaces]
        self.volumes = [Volume(1, surface_idxs)]

    def define_shapes(self):
        self.points = [Point(1, [self.xdim[0], self.ydim[0], self.zdim[0]]),
                       Point(2, [self.xdim[1], self.ydim[0], self.zdim[0]]),
                       Point(3, [self.xdim[1], self.ydim[1], self.zdim[0]]),
                       Point(4, [self.xdim[0], self.ydim[1], self.zdim[0]]),
                       Point(5, [self.xdim[0], self.ydim[0], self.zdim[1]]),
                       Point(6, [self.xdim[1], self.ydim[0], self.zdim[1]]),
                       Point(7, [self.xdim[1], self.ydim[1], self.zdim[1]]),
                       Point(8, [self.xdim[0], self.ydim[1], self.zdim[1]])]
        self.surfaces = [Surface(1, points=[1, 4, 8, 5, 1]),  # yz_min
                         Surface(2, points=[2, 3, 7, 6, 2]),  # yz_plus
                         Surface(3, points=[1, 2, 6, 5, 1]),  # xz_min
                         Surface(4, points=[4, 3, 7, 8, 4]),  # xz_plus
                         Surface(5, points=[1, 2, 3, 4, 1]),  # xy_min
                         Surface(6, points=[5, 6, 7, 8, 5])]  # xy_plus
        return

    def add_boundary(self, boundary):
        # xy_min, xy_plus, xz_min, xz_plus, yz_min, yz_plus
        if boundary == "xy_min":
            idxs = [5]
        elif boundary == "xy_plus":
            idxs = [6]
        elif boundary == "xz_min":
            idxs = [3]
        elif boundary == "xz_plus":
            idxs = [4]
        elif boundary == "yz_min":
            idxs = [1]
        elif boundary == "yz_plus":
            idxs = [2]
        else:
            raise Exception("Not a valid boundary")

        surface = Physical(boundary, idxs)
        self.physical_surfaces.append(surface)

        self.boundary_tag += 1
        return self.boundary_tag


class Cylinder(Shape):
    def __init__(self, c0: list, radius: float, length: float, orientation: int, angle=360.):
        self.c0, self.radius, self.length, self.orientation, self.angle = c0, radius, length, orientation, angle

        super().__init__()

    def define_shapes(self):
        if self.angle == 360.:  # need 3 curved surfaces
            self.points, self.curves, self.surfaces, self.volumes = self.calc_faces_360(self.c0, self.radius, self.length, self.orientation)
        elif self.angle >= 180.:  # need 2 curved surfaces
            self.points, self.curves, self.surfaces, self.volumes = self.calc_faces_180(self.c0, self.radius, self.length, self.orientation)
        else:  # need only 1 curved surface
            self.points, self.curves, self.surfaces, self.volumes = self.calc_faces_180_(self.c0, self.radius, self.length, self.orientation)

        return

    def calc_faces_360(self, c0, r, l, orientation):
        """Function to calculate points, curves and surfaces for a full cylinder"""

        """Calculate points"""
        segments = 3
        curve_angle = 120

        points = [Point(1, c0)]
        for i in range(segments):
            radians = math.radians(i*curve_angle)
            point0 = self.calc_radial_points(c0, r, orientation, radians)
            points.append(Point(i+2, point0))

        for i, point in enumerate(points[:]):
            point1 = point.xyz[:]
            point1[orientation] += l
            points.append(Point(i+segments+2, point1))

        """Define curves and surfaces"""
        curves = [Curve(1, 'circle', [2, 1, 3]),
                  Curve(2, 'circle', [3, 1, 4]),
                  Curve(3, 'circle', [4, 1, 2]),
                  Curve(4, 'circle', [6, 5, 7]),
                  Curve(5, 'circle', [7, 5, 8]),
                  Curve(6, 'circle', [8, 5, 6]),
                  Curve(7, 'line', [1, 5]),
                  Curve(8, 'line', [2, 6]),
                  Curve(9, 'line', [3, 7]),
                  Curve(10, 'line', [4, 8])]

        surfaces = [Surface(1, points=[2, 3, 4, 2], curves=[1, 2, 3]),  # face0
                    Surface(2, [6, 7, 8, 6], [4, 5, 6]),  # face1
                    Surface(3, [2, 3, 7, 6, 2], [1, 9, -4, -8], plane=False),  # side12
                    Surface(4, [3, 4, 8, 7, 3], [2, 10, -5, -9], plane=False),  # side23
                    Surface(5, [4, 2, 6, 8, 4], [3, 8, -6, -10], plane=False)]  # side31

        surface_idxs = [surface.idx for surface in surfaces]
        volumes = [Volume(1, surface_idxs)]

        return points, curves, surfaces, volumes

    def calc_faces_180(self, c0, r, l, orientation):
        """Function to calculate points, curves and surfaces for a half cylinder"""

        """Calculate points"""
        segments = 2
        curve_angle = self.angle/2

        points = [Point(1, c0)]
        for i in range(segments + 1):
            radians = math.radians(i*curve_angle)
            point0 = self.calc_radial_points(c0, r, orientation, radians)
            points.append(Point(i+2, point0))

        for i, point in enumerate(points[:]):
            point1 = point.xyz[:]
            point1[orientation] += l
            points.append(Point(i+segments+3, point1))

        """Define curves and surfaces"""
        curves = [Curve(1, 'line', [1, 2]),
                  Curve(2, 'circle', [2, 1, 3]),
                  Curve(3, 'circle', [3, 1, 4]),
                  Curve(4, 'line', [4, 1]),
                  Curve(5, 'line', [5, 6]),
                  Curve(6, 'circle', [6, 5, 7]),
                  Curve(7, 'circle', [7, 5, 8]),
                  Curve(8, 'line', [8, 5]),
                  Curve(9, 'line', [1, 5]),
                  Curve(10, 'line', [2, 6]),
                  Curve(11, 'line', [3, 7]),
                  Curve(12, 'line', [4, 8])]

        surfaces = [Surface(1, [1, 2, 3, 4, 1], [1, 2, 3, 4]),  # face0
                    Surface(2, [5, 6, 7, 8, 5], [5, 6, 7, 8]),  # face1
                    Surface(3, [1, 2, 6, 5, 1], [1, 10, -5, -9]),  # side centre1
                    Surface(4, [2, 3, 7, 6, 2], [2, 11, -6, -10], plane=False),  # side12
                    Surface(5, [3, 4, 8, 7, 3], [3, 12, -7, -11], plane=False),  # side23
                    Surface(6, [1, 5, 8, 4, 1], [9, -8, -12, 4])]  # side centre2

        surface_idxs = [surface.idx for surface in surfaces]
        volumes = [Volume(1, surface_idxs)]

        return points, curves, surfaces, volumes

    def calc_faces_180_(self, c0, r, l, orientation):
        """Function to calculate points, curves and surfaces for a less than half cylinder"""

        """Calculate points"""
        segments = 1
        curve_angle = self.angle

        points = [Point(1, c0)]
        for i in range(segments+1):
            radians = math.radians(i*curve_angle)
            point0 = self.calc_radial_points(c0, r, orientation, radians)
            points.append(Point(i+2, point0))

        for i, point in enumerate(points[:]):
            point1 = point.xyz[:]
            point1[orientation] += l
            points.append(Point(i+segments+3, point1))

        """Define curves and surfaces"""
        curves = [Curve(1, 'line', [1, 2]),
                  Curve(2, 'circle', [2, 1, 3]),
                  Curve(3, 'line', [3, 1]),
                  Curve(4, 'line', [4, 5]),
                  Curve(5, 'circle', [5, 4, 6]),
                  Curve(6, 'line', [6, 4]),
                  Curve(7, 'line', [1, 4]),
                  Curve(8, 'line', [2, 5]),
                  Curve(9, 'line', [3, 6])]

        surfaces = [Surface(1, [1, 2, 3, 1], [1, 2, 3]),  # face0
                    Surface(2, [4, 5, 6, 4], [4, 5, 6]),  # face1
                    Surface(3, [1, 2, 5, 4, 1], [1, 8, -4, -7]),  # side centre1
                    Surface(4, [2, 3, 6, 5, 2], [2, 9, -5, -8], plane=False),  # side12
                    Surface(5, [1, 4, 6, 3, 1], [7, -6, -9, 3])]  # side centre2

        surface_idxs = [surface.idx for surface in surfaces]
        volumes = [Volume(1, surface_idxs)]

        return points, curves, surfaces, volumes

    def calc_radial_points(self, c, r, orientation, angle):
        point = c[:]
        if orientation == 0:  # x-dir
            point[1] += np.round(r * np.sin(angle), 5)
            point[2] += np.round(r * np.cos(angle), 5)
        elif orientation == 1:  # y-dir
            point[0] += np.round(r * np.sin(angle), 5)
            point[2] += np.round(r * np.cos(angle), 5)
        else:  # z-dir
            point[0] += np.round(r * np.sin(angle), 5)
            point[1] += np.round(r * np.cos(angle), 5)

        return point

    def add_boundary(self, boundary):
        # bottom, top, outer
        if boundary == "top":
            idxs = [1]
        elif boundary == "bottom":
            idxs = [2]
        elif boundary == "outer":
            if self.angle < 180.:
                idxs = [4]
            elif self.angle < 360.:
                idxs = [4, 5]
            else:
                idxs = [3, 4, 5]
        else:
            raise Exception("Not a valid boundary")

        surface = Physical(boundary, idxs)
        self.physical_surfaces.append(surface)

        self.boundary_tag += 1
        return self.boundary_tag


class CylinderWithHole(Cylinder):
    def __init__(self, c0: list, c1: list, radius: float, rw: float, length: float, orientation: int, angle=360.):
        self.rw = rw

        super().__init__(c0, radius, length, orientation, angle)

    def calc_faces_360(self, c0, r, l, orientation):
        """Function to calculate points, curves and surfaces for a full cylinder"""

        """Calculate points"""
        segments = 3
        curve_angle = 120

        points = [Point(1, c0)]
        for i in range(segments):
            radians = math.radians(i * curve_angle)
            point0 = self.calc_radial_points(c0, r, orientation, radians)
            points.append(Point(i + 2, point0))

        for i, point in enumerate(points[:]):
            point1 = point.xyz[:]
            point1[orientation] += l
            points.append(Point(i + segments + 2, point1))

        """Define curves and surfaces"""
        curves = [Curve(1, 'circle', [2, 1, 3]),
                  Curve(2, 'circle', [3, 1, 4]),
                  Curve(3, 'circle', [4, 1, 2]),
                  Curve(4, 'circle', [5, 1, 6]),
                  Curve(5, 'circle', [6, 1, 7]),
                  Curve(6, 'circle', [7, 1, 8]),
                  Curve(7, 'line', [1, 5]),
                  Curve(8, 'line', [2, 6]),
                  Curve(9, 'line', [3, 7]),
                  Curve(10, 'line', [4, 8])]

        surfaces = [Surface(1, points=[2, 3, 4, 2], curves=[1, 2, 3]),  # face0
                    Surface(2, [6, 7, 8, 6], [4, 5, 6]),  # face1
                    Surface(3, [2, 3, 7, 6, 2], [1, 9, -4, -8], plane=False),  # side12
                    Surface(4, [3, 4, 8, 7, 3], [2, 10, -5, -9], plane=False),  # side23
                    Surface(5, [4, 2, 6, 8, 4], [3, 8, -6, -10], plane=False)]  # side31

        surface_idxs = [surface.idx for surface in surfaces]
        volumes = [Volume(1, surface_idxs)]

        return points, curves, surfaces, volumes

    def calc_faces_180(self, c0, r, l, orientation):
        """Function to calculate points, curves and surfaces for a half cylinder"""

        """Calculate points"""
        segments = 2
        curve_angle = self.angle / 2

        points = [Point(1, c0)]
        for i in range(segments + 1):
            radians = math.radians(i * curve_angle)
            point0 = self.calc_radial_points(c0, r, orientation, radians)
            points.append(Point(i + 2, point0))

        for i, point in enumerate(points[:]):
            point1 = point.xyz[:]
            point1[orientation] += l
            points.append(Point(i + segments + 3, point1))

        """Define curves and surfaces"""
        curves = [Curve(1, 'line', [1, 2]),
                  Curve(2, 'circle', [2, 1, 3]),
                  Curve(3, 'circle', [3, 1, 4]),
                  Curve(4, 'line', [4, 1]),
                  Curve(5, 'line', [5, 6]),
                  Curve(6, 'circle', [6, 5, 7]),
                  Curve(7, 'circle', [7, 5, 8]),
                  Curve(8, 'line', [8, 5]),
                  Curve(9, 'line', [1, 5]),
                  Curve(10, 'line', [2, 6]),
                  Curve(11, 'line', [3, 7]),
                  Curve(12, 'line', [4, 8])]

        surfaces = [Surface(1, [1, 2, 3, 4, 1], [1, 2, 3, 4]),  # face0
                    Surface(2, [5, 6, 7, 8, 5], [5, 6, 7, 8]),  # face1
                    Surface(3, [1, 2, 6, 5, 1], [1, 10, -5, -9]),  # side centre1
                    Surface(4, [2, 3, 7, 6, 2], [2, 11, -6, -10], plane=False),  # side12
                    Surface(5, [3, 4, 8, 7, 3], [3, 12, -7, -11], plane=False),  # side23
                    Surface(6, [1, 5, 8, 4, 1], [9, -8, -12, 4])]  # side centre2

        surface_idxs = [surface.idx for surface in surfaces]
        volumes = [Volume(1, surface_idxs)]

        return points, curves, surfaces, volumes

    def calc_faces_180_(self, c0, r, l, orientation):
        """Function to calculate points, curves and surfaces for a less than half cylinder"""

        """Calculate points"""
        segments = 1
        curve_angle = self.angle

        points = [Point(1, c0)]
        for i in range(segments + 1):
            radians = math.radians(i * curve_angle)
            point0 = self.calc_radial_points(c0, r, orientation, radians)
            points.append(Point(i + 2, point0))

        for i, point in enumerate(points[:]):
            point1 = point.xyz[:]
            point1[orientation] += l
            points.append(Point(i + segments + 3, point1))

        """Define curves and surfaces"""
        curves = [Curve(1, 'line', [1, 2]),
                  Curve(2, 'circle', [2, 1, 3]),
                  Curve(3, 'line', [3, 1]),
                  Curve(4, 'line', [4, 5]),
                  Curve(5, 'circle', [5, 4, 6]),
                  Curve(6, 'line', [6, 4]),
                  Curve(7, 'line', [1, 4]),
                  Curve(8, 'line', [2, 5]),
                  Curve(9, 'line', [3, 6])]

        surfaces = [Surface(1, [1, 2, 3, 1], [1, 2, 3]),  # face0
                    Surface(2, [4, 5, 6, 4], [4, 5, 6]),  # face1
                    Surface(3, [1, 2, 5, 4, 1], [1, 8, -4, -7]),  # side centre1
                    Surface(4, [2, 3, 6, 5, 2], [2, 9, -5, -8], plane=False),  # side12
                    Surface(5, [1, 4, 6, 3, 1], [7, -6, -9, 3])]  # side centre2

        surface_idxs = [surface.idx for surface in surfaces]
        volumes = [Volume(1, surface_idxs)]

        return points, curves, surfaces, volumes

    def add_boundary(self, boundary):
        # bottom, top, outer
        if boundary == "top":
            idxs = [1]
        elif boundary == "bottom":
            idxs = [2]
        elif boundary == "outer":
            if self.angle < 180.:
                idxs = [4]
            elif self.angle < 360.:
                idxs = [4, 5]
            else:
                idxs = [3, 4, 5]
        else:
            raise Exception("Not a valid boundary")

        surface = Physical(boundary, idxs)
        self.physical_surfaces.append(surface)

        self.boundary_tag += 1
        return self.boundary_tag

class Circle(Shape):
    def __init__(self, c0: list, radius: float, orientation: int = 1, angle=360.):
        self.c0, self.radius, self.orientation, self.angle = c0, radius, orientation, angle

        super().__init__()

    def define_shapes(self):
        if self.angle == 360.:  # need 3 curved lines
            self.points = [Point(1, self.c0),
                           Point(2, self.calc_radial_points(self.c0, self.radius, self.orientation, math.radians(0))),
                           Point(3, self.calc_radial_points(self.c0, self.radius, self.orientation, math.radians(120))),
                           Point(4, self.calc_radial_points(self.c0, self.radius, self.orientation, math.radians(240)))]

            self.curves = [Curve(1, curve_type='circle', points=[2, 1, 3]),
                           Curve(2, curve_type='circle', points=[3, 1, 4]),
                           Curve(3, curve_type='circle', points=[4, 1, 2])]

            self.surfaces = [Surface(1, points=[2, 3, 4, 2], curves=[1, 2, 3])]
        elif self.angle >= 180.:  # need 2 curved lines
            self.points = [Point(1, self.c0),
                           Point(2, self.calc_radial_points(self.c0, self.radius, self.orientation, math.radians(0))),
                           Point(3, self.calc_radial_points(self.c0, self.radius, self.orientation, math.radians(self.angle/2))),
                           Point(4, self.calc_radial_points(self.c0, self.radius, self.orientation, math.radians(self.angle)))]

            self.curves = [Curve(1, curve_type='line', points=[1, 2]),
                           Curve(2, curve_type='circle', points=[2, 1, 3]),
                           Curve(3, curve_type='circle', points=[3, 1, 4]),
                           Curve(4, curve_type='line', points=[4, 1])]

            self.surfaces = [Surface(1, points=[1, 2, 3, 4, 1], curves=[1, 2, 3, 4])]
        else:  # need only 1 curved line
            self.points = [Point(1, self.c0),
                           Point(2, self.calc_radial_points(self.c0, self.radius, self.orientation, math.radians(0))),
                           Point(3, self.calc_radial_points(self.c0, self.radius, self.orientation, math.radians(self.angle)))]

            self.curves = [Curve(1, curve_type='line', points=[1, 2]),
                           Curve(2, curve_type='circle', points=[2, 1, 3]),
                           Curve(3, curve_type='line', points=[3, 1])]

            self.surfaces = [Surface(1, points=[1, 2, 3, 1], curves=[1, 2, 3])]

        return

    def calc_radial_points(self, c, r, orientation, angle):
        point = c[:]
        if orientation == 0:  # x-dir
            point[1] += np.round(r * np.sin(angle), 5)
            point[2] += np.round(r * np.cos(angle), 5)
        elif orientation == 1:  # y-dir
            point[0] += np.round(r * np.sin(angle), 5)
            point[2] += np.round(r * np.cos(angle), 5)
        else:  # z-dir
            point[0] += np.round(r * np.sin(angle), 5)
            point[1] += np.round(r * np.cos(angle), 5)

        return point


class CircleWithHole(Circle):
    def __init__(self, c0: list, radius: float, rw: float, orientation: int = 1, angle=360.):
        self.rw = rw

        super().__init__(c0, radius, orientation, angle)

    def define_shapes(self):
        if self.angle == 360.:  # need 3 curved lines
            print("DOESN'T WORK YET")
            # self.points = [Point(1, self.c0),
            #                Point(2, self.calc_radial_points(self.c0, self.radius, self.orientation, math.radians(0))),
            #                Point(3, self.calc_radial_points(self.c0, self.radius, self.orientation, math.radians(120))),
            #                Point(4, self.calc_radial_points(self.c0, self.radius, self.orientation, math.radians(240))),
            #                Point(5, self.calc_radial_points(self.c0, self.rw, self.orientation, math.radians(0)), lc=1),
            #                Point(6, self.calc_radial_points(self.c0, self.rw, self.orientation, math.radians(120)), lc=1),
            #                Point(7, self.calc_radial_points(self.c0, self.rw, self.orientation, math.radians(240)), lc=1)]
            #
            # self.curves = [Curve(1, curve_type='circle', points=[2, 1, 3]),
            #                Curve(2, curve_type='circle', points=[3, 1, 4]),
            #                Curve(3, curve_type='circle', points=[4, 1, 2]),
            #                Curve(4, curve_type='circle', points=[5, 1, 6]),
            #                Curve(5, curve_type='circle', points=[6, 1, 7]),
            #                Curve(6, curve_type='circle', points=[7, 1, 5])
            #                ]
            #
            # self.surfaces = [Surface(1, points=[2, 3, 4, 2, 5, 6, 7, 5], curves=[1, 2, 3, 4, 5, 6])]
            #
            # self.physical_curves = [Physical('inner', idxs=[4, 5, 6]),
            #                         Physical('outer', idxs=[1, 2, 3])
            #                         ]
        elif self.angle >= 180.:  # need 2 curved lines
            self.points = [Point(1, self.c0),
                           Point(2, self.calc_radial_points(self.c0, self.rw, self.orientation, math.radians(0)), lc=1),
                           Point(3, self.calc_radial_points(self.c0, self.radius, self.orientation, math.radians(0))),
                           Point(4, self.calc_radial_points(self.c0, self.radius, self.orientation, math.radians(self.angle/2))),
                           Point(5, self.calc_radial_points(self.c0, self.radius, self.orientation, math.radians(self.angle))),
                           Point(6, self.calc_radial_points(self.c0, self.rw, self.orientation, math.radians(self.angle)), lc=1),
                           Point(7, self.calc_radial_points(self.c0, self.rw, self.orientation, math.radians(self.angle/2)), lc=1),
                           ]

            self.curves = [Curve(1, curve_type='line', points=[2, 3]),
                           Curve(2, curve_type='circle', points=[3, 1, 4]),
                           Curve(3, curve_type='circle', points=[4, 1, 5]),
                           Curve(4, curve_type='line', points=[5, 6]),
                           Curve(5, curve_type='circle', points=[6, 1, 7]),
                           Curve(6, curve_type='circle', points=[7, 1, 2]),]

            self.surfaces = [Surface(1, points=[2, 3, 4, 5, 6, 7, 2], curves=[1, 2, 3, 4, 5, 6])]

            self.physical_curves = [Physical('inner', idxs=[5, 6]),
                                    Physical('outer', idxs=[2, 3])
                                    ]

            # self.physical_curves['inner'] = Physical('inner', idxs=[5, 6])
            # self.physical_curves['outer'] = Physical('outer', idxs=[2, 3])

        else:  # need only 1 curved line
            self.points = [Point(1, self.c0),
                           Point(2, self.calc_radial_points(self.c0, self.rw, self.orientation, math.radians(0)), lc=1),
                           Point(3, self.calc_radial_points(self.c0, self.radius, self.orientation, math.radians(0))),
                           Point(4, self.calc_radial_points(self.c0, self.radius, self.orientation, math.radians(self.angle))),
                           Point(5, self.calc_radial_points(self.c0, self.rw, self.orientation, math.radians(self.angle)), lc=1)]

            self.curves = [Curve(1, curve_type='line', points=[2, 3]),
                           Curve(2, curve_type='circle', points=[3, 1, 4]),
                           Curve(3, curve_type='line', points=[4, 5]),
                           Curve(4, curve_type='circle', points=[5, 1, 2])]

            self.surfaces = [Surface(1, points=[2, 3, 4, 5, 2], curves=[1, 2, 3, 4])]

            self.physical_curves = [Physical('inner', idxs=[4]),
                                    Physical('outer', idxs=[2])
                                    ]
        return
