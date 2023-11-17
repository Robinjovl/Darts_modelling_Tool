import numpy as np
import math
from dataclasses import dataclass, field


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
class MeshProperties:
    square: bool
    center: list
    lc: list
    xlen: float = 0.
    ylen: float = 0.
    zlen: float = 0.
    orientation: str = 'xy'
    radii: list = field(default_factory=list)
    hole: bool = False
    extrude: bool = False
    extrude_length: float = 0.
    extrude_layers: int = 1
    extrude_axis: int = 2
    extrude_recombine: bool = False
# endregion


class Shape:
    points = []
    curves = []
    surfaces = []
    volumes = []

    holes = []
    lc = []

    physical_points = {}
    physical_curves = {}
    physical_surfaces = {}
    physical_volumes = {}

    boundary_tag: int = 90000

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

    def calc_radial_points(self, center: list, radius: float, orientation: str, angle: float):
        point = np.copy(center)
        if orientation == 'xy':
            point[0] += np.round(radius * np.sin(angle), 5)
            point[1] += np.round(radius * np.cos(angle), 5)
        elif orientation == 'xz':
            point[0] += np.round(radius * np.sin(angle), 5)
            point[2] += np.round(radius * np.cos(angle), 5)
        else:  # 'yz'
            point[1] += np.round(radius * np.sin(angle), 5)
            point[2] += np.round(radius * np.cos(angle), 5)

        return list(point)

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


class Square(Shape):
    def __init__(self, center: list, lc: list, xlen: float = 0., ylen: float = 0., zlen: float = 0.,
                 orientation: str = 'xy', radii: list = None, hole: bool = False):
        num_radii = len(radii) if radii is not None else 0
        assert len(lc) > num_radii, "List of lc is of insufficient length"
        self.lc = lc

        if orientation == 'xy':
            xmin, xmax = center[0] - xlen*0.5, center[0] + xlen*0.5,
            ymin, ymax = center[1] - ylen*0.5, center[1] + ylen*0.5,
            self.points = [Point(1, center),
                           Point(2, [xmin, ymin, center[2]]),
                           Point(3, [xmax, ymin, center[2]]),
                           Point(4, [xmax, ymax, center[2]]),
                           Point(5, [xmin, ymax, center[2]])]
        elif orientation == 'xz':
            xmin, xmax = center[0] - xlen * 0.5, center[0] + xlen * 0.5,
            zmin, zmax = center[2] - zlen * 0.5, center[2] + zlen * 0.5,
            self.points = [Point(1, center),
                           Point(2, [xmin, center[1], zmin]),
                           Point(3, [xmax, center[1], zmin]),
                           Point(4, [xmax, center[1], zmax]),
                           Point(5, [xmin, center[1], zmax])]
        else:  # 'yz'
            ymin, ymax = center[1] - ylen * 0.5, center[1] + ylen * 0.5,
            zmin, zmax = center[2] - zlen * 0.5, center[2] + zlen * 0.5,
            self.points = [Point(1, center),
                           Point(2, [center[0], ymin, zmin]),
                           Point(3, [center[0], ymax, zmin]),
                           Point(4, [center[0], ymax, zmax]),
                           Point(5, [center[0], ymin, zmax])]

        self.curves = [Curve(1, curve_type='line', points=[5, 2]),
                       Curve(2, curve_type='line', points=[2, 3]),
                       Curve(3, curve_type='line', points=[3, 4]),
                       Curve(4, curve_type='line', points=[4, 5])]
        self.surfaces = [Surface(1, points=[2, 3, 4, 5, 2], curves=[1, 2, 3, 4])]

        for i, radius in enumerate(radii):
            p0 = len(self.points)
            p1, p2, p3 = p0 + 1, p0 + 2, p0 + 3
            self.points += [Point(p1, self.calc_radial_points(center, radius, orientation, math.radians(0)), lc=i+1),
                            Point(p2, self.calc_radial_points(center, radius, orientation, math.radians(120)), lc=i+1),
                            Point(p3, self.calc_radial_points(center, radius, orientation, math.radians(240)), lc=i+1)]

            c0 = len(self.curves)
            c1, c2, c3 = c0 + 1, c0 + 2, c0 + 3
            self.curves += [Curve(c1, curve_type='circle', points=[p1, 1, p2]),
                            Curve(c2, curve_type='circle', points=[p2, 1, p3]),
                            Curve(c3, curve_type='circle', points=[p3, 1, p1])]

            s0 = len(self.surfaces)
            self.surfaces += [Surface(s0 + 1, points=[p1, p2, p3, p1], curves=[c1, c2, c3])]

            self.surfaces[i].holes = [s0 + 1]

        if hole:
            self.holes = [len(self.surfaces)]

            c3 = len(self.curves)
            c1, c2 = c3 - 2, c3 - 1
            self.physical_curves['inner'] = [c1, c2, c3]


class Circle(Shape):
    def __init__(self, center: list, lc: list, radii: list, orientation: str = 'xy', angle=360., hole: bool = False):
        assert len(radii) > 0, "No radii provided"
        assert len(lc) >= len(radii), "List of lc is of insufficient length"
        self.lc = lc

        self.physical_curves['outer'] = []
        if hole:
            self.physical_curves['inner'] = []

        if angle == 360.:  # need 3 curved lines
            self.points = [Point(1, center)]

            for i, radius in enumerate(radii):
                p0 = len(self.points)
                p1, p2, p3 = p0 + 1, p0 + 2, p0 + 3
                self.points += [Point(p1, self.calc_radial_points(center, radius, orientation, math.radians(0)), lc=i),
                                Point(p2, self.calc_radial_points(center, radius, orientation, math.radians(120)), lc=i),
                                Point(p3, self.calc_radial_points(center, radius, orientation, math.radians(240)), lc=i)]

                c0 = len(self.curves)
                c1, c2, c3 = c0 + 1, c0 + 2, c0 + 3
                self.curves += [Curve(c1, curve_type='circle', points=[p1, 1, p2]),
                                Curve(c2, curve_type='circle', points=[p2, 1, p3]),
                                Curve(c3, curve_type='circle', points=[p3, 1, p1])]

                s0 = len(self.surfaces)
                self.surfaces += [Surface(s0 + 1, points=[p1, p2, p3, p1], curves=[c1, c2, c3])]

                if i == 0:
                    self.physical_curves['outer'] += [c1, c2, c3]
                else:
                    self.surfaces[i-1].holes = [s0 + 1]
                    if i == len(radii) - 1 and hole:
                        self.holes += [s0 + 1]

            # if not hole:
            #     self.holes = self.holes[:-1]
            if hole:
                c3 = len(self.curves)
                c1, c2 = c3-2, c3-1
                self.physical_curves['inner'] += [c1, c2, c3]

        elif angle >= 180.:  # need 2 curved lines
            self.points = [Point(1, center)]

            # Start from outermost surface
            for i, radius in enumerate(radii[:-1]):
                p0 = len(self.points)
                p1, p2, p3, p4, p5, p6 = p0 + 1, p0 + 2, p0 + 3, p0 + 4, p0 + 5, p0 + 6
                self.points += [Point(p1, self.calc_radial_points(center, radius, orientation, math.radians(0)), lc=i),
                                Point(p2, self.calc_radial_points(center, radius, orientation, math.radians(angle*0.5)), lc=i),
                                Point(p3, self.calc_radial_points(center, radius, orientation, math.radians(angle)), lc=i)]

                c0 = len(self.curves)
                c1, c2, c3, c4, c6, c7 = c0 + 1, c0 + 2, c0 + 3, c0 + 4, c0 + 6, c0 + 7
                self.curves += [Curve(c1, curve_type='line', points=[p4, p1]),
                                Curve(c2, curve_type='circle', points=[p1, 1, p2]),
                                Curve(c3, curve_type='circle', points=[p2, 1, p3]),
                                Curve(c4, curve_type='line', points=[p3, p6])]

                s0 = len(self.surfaces)
                self.surfaces += [Surface(s0 + 1, points=[p4, p1, p2, p3, p6, p5, p4], curves=[c1, c2, c3, c4, -c7, -c6])]

                if i == 0:
                    self.physical_curves['outer'] += [c2, c3]

            # Add innermost surface
            ii = len(radii) - 1
            p0 = len(self.points)
            p1, p2, p3 = p0 + 1, p0 + 2, p0 + 3
            self.points += [Point(p1, self.calc_radial_points(center, radii[-1], orientation, math.radians(0)), lc=ii),
                            Point(p2, self.calc_radial_points(center, radii[-1], orientation, math.radians(angle*0.5)), lc=ii),
                            Point(p3, self.calc_radial_points(center, radii[-1], orientation, math.radians(angle)), lc=ii)]

            c0 = len(self.curves)
            c1, c2, c3, c4 = c0 + 1, c0 + 2, c0 + 3, c0 + 4
            self.curves += [Curve(c2, curve_type='circle', points=[p1, 1, p2]),
                            Curve(c3, curve_type='circle', points=[p2, 1, p3])]

            if len(radii) == 1 or not hole:
                self.curves += [Curve(c1, curve_type='line', points=[1, p1]),
                                Curve(c4, curve_type='line', points=[p3, 1])]
                s0 = len(self.surfaces)
                self.surfaces += [Surface(s0 + 1, points=[p0, p1, p2, p3, p0], curves=[c1, c2, c3, c4])]
            else:
                self.physical_curves['inner'] += [c2, c3]

        else:  # need only 1 curved line
            self.points = [Point(1, center)]

            # Start from outermost surface
            for i, radius in enumerate(radii[:-1]):
                p0 = len(self.points)
                p1, p2, p3, p4 = p0 + 1, p0 + 2, p0 + 3, p0 + 4
                self.points += [Point(p1, self.calc_radial_points(center, radius, orientation, math.radians(0)), lc=i),
                                Point(p2, self.calc_radial_points(center, radius, orientation, math.radians(angle)), lc=i),]

                c0 = len(self.curves)
                c1, c2, c3, c5 = c0 + 1, c0 + 2, c0 + 3, c0 + 5
                self.curves += [Curve(c1, curve_type='line', points=[p3, p1]),
                                Curve(c2, curve_type='circle', points=[p1, 1, p2]),
                                Curve(c3, curve_type='line', points=[p2, p4])]

                s0 = len(self.surfaces)
                self.surfaces += [Surface(s0 + 1, points=[p3, p1, p2, p4, p3], curves=[c1, c2, c3, -c5])]

                if i == 0:
                    self.physical_curves['outer'] += [c2]

            # Add innermost surface
            ii = len(radii) - 1
            p0 = len(self.points)
            p1, p2 = p0 + 1, p0 + 2
            self.points += [Point(p1, self.calc_radial_points(center, radii[-1], orientation, math.radians(0)), lc=ii),
                            Point(p2, self.calc_radial_points(center, radii[-1], orientation, math.radians(angle)), lc=ii)]

            c0 = len(self.curves)
            c1, c2, c3 = c0 + 1, c0 + 2, c0 + 3
            self.curves += [Curve(c2, curve_type='circle', points=[p1, 1, p2])]

            if len(radii) == 1 or not hole:
                self.curves += [Curve(c1, curve_type='line', points=[1, p1]),
                                Curve(c3, curve_type='line', points=[p2, 1])]
                s0 = len(self.surfaces)
                self.surfaces += [Surface(s0 + 1, points=[p0, p1, p2, p0], curves=[c1, c2, c3])]
            else:
                self.physical_curves['inner'] += [c2]


class Cylinder(Shape):
    def __init__(self, center: list, lc: list, radii: list, length: float, orientation: str = 'xy', angle=360., hole: bool = False):
        """Function to calculate points, curves and surfaces for a full cylinder"""
        assert len(radii) > 0, "No radii provided"
        assert len(lc) >= len(radii), "List of lc is of insufficient length"
        self.lc = lc

        self.physical_surfaces['top'] = []
        self.physical_surfaces['bottom'] = []
        self.physical_surfaces['outer'] = []
        if hole:
            self.physical_surfaces['inner'] = []

        center = np.array(center)
        L = np.array([0., 0., 0.])
        if orientation == 'xy':
            L[2] = length
        elif orientation == 'xz':
            L[1] = length
        else:
            L[0] = length

        if angle == 360.:  # need 3 curved surfaces
            self.points = [Point(1, center), Point(2, center + L)]
            self.curves = [Curve(1, 'line', points=[1, 2])]

            # Start from outermost surface
            for i, radius in enumerate(radii):
                p0 = len(self.points)
                p1, p2, p3, p4, p5, p6 = p0 + 1, p0 + 2, p0 + 3, p0 + 4, p0 + 5, p0 + 6
                P1 = self.calc_radial_points(center, radius, orientation, math.radians(0))
                P2 = self.calc_radial_points(center, radius, orientation, math.radians(120))
                P3 = self.calc_radial_points(center, radius, orientation, math.radians(240))

                self.points += [Point(p1, P1, lc=i), Point(p2, P2, lc=i), Point(p3, P3, lc=i),
                                Point(p4, list(P1 + L), lc=i), Point(p5, list(P2 + L), lc=i), Point(p6, list(P3 + L), lc=i)]

                c0 = len(self.curves)
                c1, c2, c3, c4, c5, c6 = c0 + 1, c0 + 2, c0 + 3, c0 + 4, c0 + 5, c0 + 6
                c7, c8, c9 = c0 + 7, c0 + 8, c0 + 9
                self.curves += [Curve(c1, curve_type='circle', points=[p1, 1, p2]),
                                Curve(c2, 'circle', points=[p2, 1, p3]),
                                Curve(c3, 'circle', points=[p3, 1, p1]),
                                Curve(c4, 'circle', points=[p4, 2, p5]),
                                Curve(c5, 'circle', points=[p5, 2, p6]),
                                Curve(c6, 'circle', points=[p6, 2, p4]),
                                Curve(c7, 'line', [p1, p4]),
                                Curve(c8, 'line', [p2, p5]),
                                Curve(c9, 'line', [p3, p6]),
                                ]

                s0 = len(self.surfaces)
                s1, s2, s3, s4, s5 = s0 + 1, s0 + 2, s0 + 3, s0 + 4, s0 + 5
                s8, s9, s10 = s0 + 8, s0 + 9, s0 + 10
                surfaces = [Surface(s1, points=[p1, p2, p3, p1], curves=[c1, c2, c3],
                                    holes=[s1 + 5*(ii+1) for ii, _ in enumerate(radii[i+1:])]),  # face0
                            Surface(s2, [p4, p5, p6, p4], [c4, c5, c6],
                                    holes=[s2 + 5*(ii+1) for ii, _ in enumerate(radii[i+1:])]),  # face1
                            Surface(s3, [p1, p2, p5, p4, p1], [c1, c8, -c4, -c7], plane=False),  # side12
                            Surface(s4, [p2, p3, p6, p4, p2], [c2, c9, -c5, -c8], plane=False),  # side23
                            Surface(s5, [p3, p4, p1, p5, p3], [c3, c7, -c6, -c9], plane=False)  # side31
                            ]
                self.surfaces += surfaces

                surface_idxs = [s1, s2, s3, s4, s5] if i == len(radii)-1 else [s1, s2, s3, s4, s5, s8, s9, s10]
                v0 = len(self.volumes)
                self.volumes += [Volume(v0 + 1, surface_idxs)]

                self.physical_surfaces['top'] += [s1]
                self.physical_surfaces['bottom'] += [s2]
                if i == 0:
                    self.physical_surfaces['outer'] += [s3, s4, s5]
                elif i == len(radii) - 1 and hole:
                    self.physical_surfaces['inner'] += [s3, s4, s5]

            if len(radii) > 1 and hole:
                self.surfaces[s0].active = False
                self.surfaces[s0+1].active = False
                self.volumes = self.volumes[:-1]

        elif angle >= 180.:  # need 2 curved surfaces
            self.points = [Point(1, center), Point(2, center + L)]
            self.curves = [Curve(1, 'line', points=[1, 2])]

            # Start from outermost surface
            for i, radius in enumerate(radii[:-1]):
                p0 = len(self.points)
                p1, p2, p3, p4, p5, p6 = p0 + 1, p0 + 2, p0 + 3, p0 + 4, p0 + 5, p0 + 6
                p7, p8, p9, p10, p11, p12 = p0 + 7, p0 + 8, p0 + 9, p0 + 10, p0 + 11, p0 + 12
                P1 = self.calc_radial_points(center, radius, orientation, math.radians(0))
                P2 = self.calc_radial_points(center, radius, orientation, math.radians(angle*0.5))
                P3 = self.calc_radial_points(center, radius, orientation, math.radians(angle))

                self.points += [Point(p1, P1, lc=i), Point(p2, P2, lc=i), Point(p3, P3, lc=i),
                                Point(p4, P1 + L, lc=i), Point(p5, P2 + L, lc=i), Point(p6, P3 + L, lc=i)]

                c0 = len(self.curves)
                c1, c2, c3, c4, c5, c6 = c0 + 1, c0 + 2, c0 + 3, c0 + 4, c0 + 5, c0 + 6
                c7, c8, c9, c10, c11 = c0 + 7, c0 + 8, c0 + 9, c0 + 10, c0 + 11
                c13, c14, c17, c18, c20, c21, c22 = c0 + 13, c0 + 14, c0 + 17, c0 + 18, c0 + 20, c0 + 21, c0 + 22
                self.curves += [Curve(c1, curve_type='line', points=[p7, p1]),
                                Curve(c2, 'circle', [p1, 1, p2]),
                                Curve(c3, 'circle', [p2, 1, p3]),
                                Curve(c4, 'line', [p3, p9]),
                                Curve(c5, 'line', [p10, p4]),
                                Curve(c6, 'circle', [p4, 2, p5]),
                                Curve(c7, 'circle', [p5, 2, p6]),
                                Curve(c8, 'line', [p6, p12]),
                                Curve(c9, 'line', [p1, p4]),
                                Curve(c10, 'line', [p2, p5]),
                                Curve(c11, 'line', [p3, p6]),
                                ]

                s0 = len(self.surfaces)
                s1, s2, s3, s4, s5, s6 = s0 + 1, s0 + 2, s0 + 3, s0 + 4, s0 + 5, s0 + 6
                s10, s11 = s0 + 10, s0 + 11

                self.surfaces += [Surface(s1, [p7, p1, p2, p3, p9, p8, p7], [c1, c2, c3, c4, -c14, -c13]),  # face0
                                  Surface(s2, [p10, p4, p5, p6, p12, p11, p10], [c5, c6, c7, c8, -c18, -c17]),  # face1
                                  Surface(s3, [p7, p1, p4, p10, p7], [c1, c9, -c5, -c20]),  # side centre1
                                  Surface(s4, [p1, p2, p5, p4, p1], [c2, c10, -c6, -c9], plane=False),  # side12
                                  Surface(s5, [p2, p3, p6, p5, p2], [c3, c11, -c7, -c10], plane=False),  # side23
                                  Surface(s6, [p9, p3, p6, p12, p9], [-c4, c11, c8, -c22])  # side centre2
                                  ]

                surface_idxs = [s1, s2, s3, s4, s5, s6] if i == len(radii)-1 else [s1, s2, s3, s4, s5, s6, s10, s11]
                v0 = len(self.volumes)
                self.volumes += [Volume(v0 + 1, surface_idxs)]

                self.physical_surfaces['top'] += [s1]
                self.physical_surfaces['bottom'] += [s2]
                if i == 0:
                    self.physical_surfaces['outer'] += [s4, s5]

            # Add innermost surface
            i = len(radii)-1
            p0 = len(self.points)
            p1, p2, p3, p4, p5, p6 = p0 + 1, p0 + 2, p0 + 3, p0 + 4, p0 + 5, p0 + 6
            P1 = self.calc_radial_points(center, radii[-1], orientation, math.radians(0))
            P2 = self.calc_radial_points(center, radii[-1], orientation, math.radians(angle * 0.5))
            P3 = self.calc_radial_points(center, radii[-1], orientation, math.radians(angle))

            self.points += [Point(p1, P1, lc=i), Point(p2, P2, lc=i), Point(p3, P3, lc=i),
                            Point(p4, P1 + L, lc=i), Point(p5, P2 + L, lc=i), Point(p6, P3 + L, lc=i)]

            c0 = len(self.curves)
            c1, c2, c3, c4, c5, c6 = c0 + 1, c0 + 2, c0 + 3, c0 + 4, c0 + 5, c0 + 6
            c7, c8, c9, c10, c11 = c0 + 7, c0 + 8, c0 + 9, c0 + 10, c0 + 11
            self.curves += [Curve(c2, 'circle', [p1, 1, p2]),
                            Curve(c3, 'circle', [p2, 1, p3]),
                            Curve(c6, 'circle', [p4, 2, p5]),
                            Curve(c7, 'circle', [p5, 2, p6]),
                            Curve(c9, 'line', [p1, p4]),
                            Curve(c10, 'line', [p2, p5]),
                            Curve(c11, 'line', [p3, p6]),
                            ]

            s0 = len(self.surfaces)
            s1, s2, s3, s4, s5, s6 = s0 + 1, s0 + 2, s0 + 3, s0 + 4, s0 + 5, s0 + 6
            self.surfaces += [Surface(s4, [p1, p2, p5, p4, p1], [c2, c10, -c6, -c9], plane=False),  # side12
                              Surface(s5, [p2, p3, p6, p5, p2], [c3, c11, -c7, -c10], plane=False),  # side23]
                              ]

            if len(radii) == 1 or not hole:
                self.curves += [Curve(c1, curve_type='line', points=[1, p1]),
                                Curve(c4, 'line', [p3, 1]),
                                Curve(c5, 'line', [2, p4]),
                                Curve(c8, 'line', [p6, 2]),
                                ]

                self.surfaces += [Surface(s1, [p1, p2, p3, 1, p1], [c2, c3, c4, c1]),  # face0
                                  Surface(s2, [p4, p5, p6, 2, p4], [c6, c7, c8, c5]),  # face1
                                  Surface(s3, [1, p1, p4, 2, 1], [c1, c9, -c5, -1]),  # side centre1
                                  Surface(s6, [1, p3, p6, 2, 1], [-c4, c11, c8, -1]),  # side centre1
                                  ]
                surface_idxs = [s1, s2, s3, s4, s5, s6]
                v0 = len(self.volumes)
                self.volumes += [Volume(v0 + 1, surface_idxs)]
            else:
                self.physical_surfaces['inner'] += [s4, s5]

        else:  # angle < 180 need 1 curved surface
            self.points = [Point(1, center), Point(2, center + L)]
            self.curves = [Curve(1, 'line', points=[1, 2])]

            # Start from outermost surface
            for i, radius in enumerate(radii[:-1]):
                p0 = len(self.points)
                p1, p2, p3, p4 = p0 + 1, p0 + 2, p0 + 3, p0 + 4
                p5, p6, p7, p8 = p0 + 5, p0 + 6, p0 + 7, p0 + 8
                P1 = self.calc_radial_points(center, radius, orientation, math.radians(0))
                P2 = self.calc_radial_points(center, radius, orientation, math.radians(angle))

                self.points += [Point(p1, P1, lc=i), Point(p2, P2, lc=i),
                                Point(p3, P1 + L, lc=i), Point(p4, P2 + L, lc=i)]

                c0 = len(self.curves)
                c1, c2, c3, c4, c5, c6, c7, c8 = c0 + 1, c0 + 2, c0 + 3, c0 + 4, c0 + 5, c0 + 6, c0 + 7, c0 + 8
                c10, c13, c15, c16 = c0 + 10, c0 + 13, c0 + 15, c0 + 16
                self.curves += [Curve(c1, curve_type='line', points=[p5, p1]),
                                Curve(c2, 'circle', [p1, 1, p2]),
                                Curve(c3, 'line', [p2, p6]),
                                Curve(c4, 'line', [p7, p3]),
                                Curve(c5, 'circle', [p3, 2, p4]),
                                Curve(c6, 'line', [p4, p8]),
                                Curve(c7, 'line', [p1, p3]),
                                Curve(c8, 'line', [p2, p4]),
                                ]

                s0 = len(self.surfaces)
                s1, s2, s3, s4, s5 = s0 + 1, s0 + 2, s0 + 3, s0 + 4, s0 + 5
                s9 = s0 + 9
                self.surfaces += [Surface(s1, [p5, p1, p2, p6, p5], [c1, c2, c3, -c10]),  # face0
                                  Surface(s2, [p7, p3, p4, p8, p7], [c4, c5, c6, -c13]),  # face1
                                  Surface(s3, [p5, p1, p3, p7, p5], [c1, c7, -c4, -c15]),  # side centre1
                                  Surface(s4, [p1, p2, p4, p3, p1], [c2, c8, -c5, -c7], plane=False),  # side12
                                  Surface(s5, [p6, p2, p4, p8, p6], [-c3, c8, c6, -c16])  # side centre2
                                  ]

                surface_idxs = [s1, s2, s3, s4, s5] if i == len(radii) - 1 else [s1, s2, s3, s4, s5, s9]
                v0 = len(self.volumes)
                self.volumes += [Volume(v0 + 1, surface_idxs)]

                self.physical_surfaces['top'] += [s1]
                self.physical_surfaces['bottom'] += [s2]
                if i == 0:
                    self.physical_surfaces['outer'] += [s4]

            # Add innermost surface
            i = len(radii) - 1
            p0 = len(self.points)
            p1, p2, p3, p4 = p0 + 1, p0 + 2, p0 + 3, p0 + 4
            P1 = self.calc_radial_points(center, radii[-1], orientation, math.radians(0))
            P2 = self.calc_radial_points(center, radii[-1], orientation, math.radians(angle))

            self.points += [Point(p1, P1, lc=i), Point(p2, P2, lc=i),
                            Point(p3, P1 + L, lc=i), Point(p4, P2 + L, lc=i)]

            c0 = len(self.curves)
            c1, c2, c3, c4, c5, c6, c7, c8 = c0 + 1, c0 + 2, c0 + 3, c0 + 4, c0 + 5, c0 + 6, c0 + 7, c0 + 8
            self.curves += [Curve(c2, 'circle', [p1, 1, p2]),
                            Curve(c5, 'circle', [p3, 2, p4]),
                            Curve(c7, 'line', [p1, p3]),
                            Curve(c8, 'line', [p2, p4]),
                            ]

            s0 = len(self.surfaces)
            s1, s2, s3, s4, s5 = s0 + 1, s0 + 2, s0 + 3, s0 + 4, s0 + 5
            self.surfaces += [Surface(s4, [p1, p2, p4, p3, p1], [c2, c8, -c5, -c7], plane=False),  # side12
                              ]

            if len(radii) == 1 or not hole:
                self.curves += [Curve(c1, curve_type='line', points=[1, p1]),
                                Curve(c3, 'line', [p2, 1]),
                                Curve(c4, 'line', [2, p3]),
                                Curve(c6, 'line', [p4, 2]),
                                ]

                self.surfaces += [Surface(s1, [1, p1, p2, 1], [c1, c2, c3]),  # face0
                                  Surface(s2, [2, p3, p4, 2], [c4, c5, c6]),  # face1
                                  Surface(s3, [1, p1, p3, 2, 1], [c1, c7, -c4, -1]),  # side centre1
                                  Surface(s5, [1, p2, p4, 2, 1], [-c3, c8, c6, -1])  # side centre2
                                  ]
                surface_idxs = [s1, s2, s3, s4, s5]
                v0 = len(self.volumes)
                self.volumes += [Volume(v0 + 1, surface_idxs)]
            else:
                self.physical_surfaces['inner'] += [s4]


class Box(Shape):
    def __init__(self, center: list, lc: list, xlen: float, ylen: float, zlen: float,
                 orientation: str = 'xy', radii: list = None, hole: bool = False):
        num_radii = len(radii) if radii is not None else 0
        assert len(lc) > num_radii, "List of lc is of insufficient length"
        self.lc = lc

        xmin, xmax = center[0] - xlen * 0.5, center[0] + xlen * 0.5,
        ymin, ymax = center[1] - ylen * 0.5, center[1] + ylen * 0.5,
        zmin, zmax = center[2] - zlen * 0.5, center[2] + zlen * 0.5,

        self.points = [Point(1, [xmin, ymin, zmin]),
                       Point(2, [xmax, ymin, zmin]),
                       Point(3, [xmax, ymax, zmin]),
                       Point(4, [xmin, ymax, zmin]),
                       Point(5, [xmin, ymin, zmax]),
                       Point(6, [xmax, ymin, zmax]),
                       Point(7, [xmax, ymax, zmax]),
                       Point(8, [xmin, ymax, zmax])]
        self.curves = [Curve(1, curve_type='line', points=[1, 2]),
                       Curve(2, curve_type='line', points=[2, 3]),
                       Curve(3, curve_type='line', points=[3, 4]),
                       Curve(4, curve_type='line', points=[4, 1]),
                       Curve(5, curve_type='line', points=[5, 6]),
                       Curve(6, curve_type='line', points=[6, 7]),
                       Curve(7, curve_type='line', points=[7, 8]),
                       Curve(8, curve_type='line', points=[8, 5]),
                       Curve(9, curve_type='line', points=[1, 5]),
                       Curve(10, curve_type='line', points=[2, 6]),
                       Curve(11, curve_type='line', points=[3, 7]),
                       Curve(12, curve_type='line', points=[4, 8]),
                       ]
        self.surfaces = [Surface(1, points=[1, 2, 3, 4, 1], curves=[1, 2, 3, 4]),  # xy_min
                         Surface(2, points=[5, 6, 7, 8, 5], curves=[5, 6, 7, 8]),  # xy_plus
                         Surface(3, points=[1, 2, 6, 5, 1], curves=[1, 10, -5, -9]),  # xz_min
                         Surface(4, points=[4, 3, 7, 8, 4], curves=[-3, 11, 7, -12]),  # xz_plus
                         Surface(5, points=[1, 4, 8, 5, 1], curves=[-4, 12, 8, -9]),  # yz_min
                         Surface(6, points=[2, 3, 7, 6, 2], curves=[2, 11, -6, -10]),  # yz_plus
                         ]

        self.physical_surfaces['xy_min'] = [1]
        self.physical_surfaces['xy_plus'] = [2]
        self.physical_surfaces['xz_min'] = [3]
        self.physical_surfaces['xz_plus'] = [4]
        self.physical_surfaces['yz_min'] = [5]
        self.physical_surfaces['yz_plus'] = [6]

        surface_idxs = [surface.idx for surface in self.surfaces]
        self.volumes = [Volume(1, surface_idxs)]

        if radii:
            assert radii[0] < 0.5*xlen and radii[0] < 0.5*ylen and radii[0] < 0.5*zlen, "Largest radius too large"
            point1, point2 = [center[0], center[1], zmin], [center[0], center[1], zmax]
            L = np.array([0., 0., zlen])

            self.points += [Point(9, point1), Point(10, point2)]

            for i, radius in enumerate(radii):
                p0 = len(self.points)
                p1, p2, p3, p4, p5, p6 = p0 + 1, p0 + 2, p0 + 3, p0 + 4, p0 + 5, p0 + 6

                P1 = self.calc_radial_points(point1, radius, 'xy', math.radians(0))
                P2 = self.calc_radial_points(point1, radius, 'xy', math.radians(120))
                P3 = self.calc_radial_points(point1, radius, 'xy', math.radians(240))

                self.points += [Point(p1, P1, lc=i+1), Point(p2, P2, lc=i+1), Point(p3, P3, lc=i+1),
                                Point(p4, list(P1 + L), lc=i+1), Point(p5, list(P2 + L), lc=i+1), Point(p6, list(P3 + L), lc=i+1)]

                c0 = len(self.curves)
                c1, c2, c3, c4, c5, c6 = c0 + 1, c0 + 2, c0 + 3, c0 + 4, c0 + 5, c0 + 6
                c7, c8, c9 = c0 + 7, c0 + 8, c0 + 9
                self.curves += [Curve(c1, curve_type='circle', points=[p1, 9, p2]),
                                Curve(c2, 'circle', points=[p2, 9, p3]),
                                Curve(c3, 'circle', points=[p3, 9, p1]),
                                Curve(c4, 'circle', points=[p4, 10, p5]),
                                Curve(c5, 'circle', points=[p5, 10, p6]),
                                Curve(c6, 'circle', points=[p6, 10, p4]),
                                Curve(c7, 'line', [p1, p4]),
                                Curve(c8, 'line', [p2, p5]),
                                Curve(c9, 'line', [p3, p6]),
                                ]

                s0 = len(self.surfaces)
                s1, s2, s3, s4, s5 = s0 + 1, s0 + 2, s0 + 3, s0 + 4, s0 + 5
                s8, s9, s10 = s0 + 8, s0 + 9, s0 + 10
                surfaces = [Surface(s1, points=[p1, p2, p3, p1], curves=[c1, c2, c3],
                                    holes=[s1 + 5 * (ii + 1) for ii, _ in enumerate(radii[i + 1:])]),  # face0
                            Surface(s2, [p4, p5, p6, p4], [c4, c5, c6],
                                    holes=[s2 + 5 * (ii + 1) for ii, _ in enumerate(radii[i + 1:])]),  # face1
                            Surface(s3, [p1, p2, p5, p4, p1], [c1, c8, -c4, -c7], plane=False),  # side12
                            Surface(s4, [p2, p3, p6, p4, p2], [c2, c9, -c5, -c8], plane=False),  # side23
                            Surface(s5, [p3, p4, p1, p5, p3], [c3, c7, -c6, -c9], plane=False)  # side31
                            ]
                self.surfaces += surfaces

                if i == 0:
                    self.surfaces[0].holes += [s1]
                    self.surfaces[1].holes += [s2]
                    self.volumes[0].surfaces += [s3, s4, s5]

                surface_idxs = [s1, s2, s3, s4, s5] if i == len(radii) - 1 else [s1, s2, s3, s4, s5, s8, s9, s10]
                v0 = len(self.volumes)
                self.volumes += [Volume(v0 + 1, surface_idxs)]

                self.physical_surfaces[orientation + '_min'] += [s1]
                self.physical_surfaces[orientation + '_plus'] += [s2]
                if i == len(radii) - 1 and hole:
                    self.physical_surfaces['inner'] = [s3, s4, s5]

            if len(radii) > 1 and hole:
                self.surfaces[s0].active = False
                self.surfaces[s0 + 1].active = False
                self.volumes = self.volumes[:-1]
