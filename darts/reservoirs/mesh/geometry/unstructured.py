import numpy as np
import math
import warnings
from darts.reservoirs.mesh.geometry.geometry import Geometry
from darts.reservoirs.mesh.geometry.shapes import Point, Curve, Surface, Volume

import gmsh


class Unstructured(Geometry):
    extrude: dict = None
    physical_groups = {'matrix': {}, 'boundary': {}}
    tags = [901, 9001, 90001, 900001]  # tags for physical points, lines, surfaces, volumes

    def calc_radial_points(self, center: list, radius: float, orientation: str, angle: float):
        point = center[:]
        if orientation == 'xy':
            point[0] += np.round(radius * np.sin(angle), 5)
            point[1] += np.round(radius * np.cos(angle), 5)
        elif orientation == 'xz':
            point[0] += np.round(radius * np.sin(angle), 5)
            point[2] += np.round(radius * np.cos(angle), 5)
        else:  # 'yz'
            point[1] += np.round(radius * np.sin(angle), 5)
            point[2] += np.round(radius * np.cos(angle), 5)

        return point

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

    def embed_point(self, point: list, lc: int):
        point_index = len(self.points) + 1
        self.points.append(Point(point_index, point, lc=lc, embed=self.find_surface(point)))
        return

    def embed_circle(self, center: list, radius: float, surface: int = 1,
                     orientation: str = 'xy', lc: int = 0, angle: float = 360.):
        p0_index = len(self.points) + 1

        if angle == 360:
            angles = [0., 120., 240.]
        elif angle >= 180.:
            angles = [0., angle*0.5, angle]
        else:
            angles = [0., angle]

        self.points.append(Point(p0_index, center))
        for i, a in enumerate(angles):
            self.points.append(Point(p0_index + i + 1,
                                     self.calc_radial_points(center, radius, orientation, math.radians(a)), lc=lc))

        c0_index = len(self.curves) + 1
        for i, a in enumerate(angles[:-1]):
            self.curves.append(Curve(c0_index + i, curve_type='circle',
                                     points=[p0_index + i + 1, p0_index, p0_index + i + 2], embed=[surface]))

        if angle == 360.:
            self.curves.append(Curve(c0_index + 3, curve_type='circle',
                                     points=[p0_index + 3, p0_index, p0_index + 1], embed=[surface]))

        return

    def embed_curve(self, points: list, lc: int = 0):
        in_surfaces = self.find_surface(points[0])
        p0_index = len(self.points) + 1
        c0_index = len(self.curves) + 1

        for i, p in enumerate(points):
            self.points.append(Point(p0_index + i, p, lc=lc))

        for i, p in enumerate(points[:-1]):
            self.curves.append(Curve(c0_index + i, curve_type='line',
                                     points=[p0_index + i, p0_index + i + 1], embed=in_surfaces))

        return

    def embed_surface(self, points: list):
        return

    def extrude_mesh(self, length: float, layers: int, axis: int, recombine: bool):
        self.extrude = {
            'length': length,
            'layers': layers,
            'axis': axis,
            'recombine': recombine
        }

    def write_geo(self, filename):
        # Create geo-file:
        f = open(filename + '.geo', "w+")

        for i, l in enumerate(self.lc):
            f.write('lc_{:d} = {:f};\n'.format(i, l))
        f.write('\n')

        """Write all points"""
        f.write('// POINTS\n')
        for point in self.points:
            local_text = 'Point({:d}) = {{{:8.5f}, {:8.5f}, {:8.5f}, lc_{:d} }};\n'.format(point.idx, point.xyz[0], point.xyz[1], point.xyz[2], point.lc)
            f.write(local_text)
        f.write('\n')

        """Write curves"""
        f.write('// CURVES\n')
        for curve in self.curves:
            if curve.active:
                if curve.curve_type == 'circle':
                    f.write('Circle({:d}) = {{{:d}, {:d}, {:d}}};\n'.format(curve.idx, curve.points[0], curve.points[1], curve.points[2]))
                else:
                    f.write('Line({:d}) = {{{:d}, {:d}}};\n'.format(curve.idx, curve.points[0], curve.points[1]))
        f.write('\n')

        """Write Curve Loops"""
        f.write('// CURVE LOOPS\n')
        for surface in self.surfaces:
            local_text = 'Curve Loop({:d}) = {{'.format(surface.idx)
            for curve_idx in surface.curves:
                if curve_idx >= 0:  # +1 segment index for p >= 0
                    local_text += '{:d}, '.format(curve_idx)
                else:  # -1 segment index for p < 0
                    local_text += '{:d}, '.format(curve_idx)
            local_text = local_text[:-2]
            local_text += '};\n'
            f.write(local_text)
        f.write('\n')

        "Write Surfaces"
        f.write('// SURFACES\n')
        for surface in self.surfaces:
            if surface.active:
                if self.dim == 3 or surface.idx not in self.holes:
                    local_text = 'Plane ' if surface.plane else ''
                    local_text += 'Surface({:d}) = {{{:d}, '.format(surface.idx, surface.idx)

                    # Add holes to existing surface
                    for hole_idx in surface.holes:
                        local_text += '{:d}, '.format(hole_idx)
                    local_text = local_text[:-2]
                    local_text += '};\n'
                    f.write(local_text)
        f.write('\n')

        """Write Embedded"""
        f.write('// EMBEDDED\n')
        for point in self.points:
            for surface_idx in point.embed:
                f.write('Point{{{:d}}} In Surface{{{:d}}};\n'.format(point.idx, surface_idx))
        for curve in self.curves:
            for surface_idx in curve.embed:
                f.write('Curve{{{:d}}} In Surface{{{:d}}};\n'.format(curve.idx, surface_idx))
        for surface in self.surfaces:
            for volume_idx in surface.embed:
                f.write('Surface{{{:d}}} In Volume{{{:d}}};\n'.format(surface.idx, volume_idx))
        f.write('\n')

        """Write Physical Groups and volumes"""
        f.write('// PHYSICAL GROUPS, EXTRUSIONS AND VOLUMES\n')
        if self.dim == 2:
            assert self.extrude is not None, "Define extrusion"
            if self.extrude is not None:
                # Extruded 2D geometry
                self.dim = 3

                extrusion = [0, 0, 0]
                extrusion[self.extrude['axis']] = self.extrude['length']

                # Extrude surface
                # Write Physical Volume: out[1]
                # Write Physical Curve: out[0], out[2], ...
                # Physical Point: ??
                for i, (name, idxs) in enumerate(self.physical_points.items()):
                    # self.physical_groups['edge'][name] = i + self.tags[0]
                    f.write('Physical Point("{:s}", {:d}) = {{}};\n'.format(name, i + self.tags[0]))
                    f.write('\n')

                # for i, physical_point in enumerate(self.physical_points):
                #     f.write('Physical Curve("{:s}", {:d}) = {{}};\n'.format(physical_point.tag, i + self.tags[1]))
                #     for point_idx in physical_point.idxs:
                #         local_text = 'out[] = Extrude {{{:f}, {:f}, {:f}}}{{ Point'.format(extrusion[0], extrusion[1], extrusion[2]) \
                #                      + '{{{:d}}};'.format(point_idx) + ' Layers{{{:d}}};\n'.format(self.extrude['layers'])
                #         local_text += 'Physical Curve("{:s}", {:d}) += {{out[1]}};\n'.format(physical_point.tag, i + self.tags[1])
                #         f.write(local_text)
                #     f.write('\n')

                # Extrude physical curves and write as Physical Surfaces
                for i, (name, idxs) in enumerate(self.physical_curves.items()):
                    self.physical_groups['boundary'][name] = i + self.tags[2]
                    f.write('Physical Surface("{:s}", {:d}) = {{}};\n'.format(name, i + self.tags[2]))
                    for curve_idx in idxs:
                        local_text = 'out[] = Extrude {{{:f}, {:f}, {:f}}}{{ Curve'.format(extrusion[0], extrusion[1], extrusion[2]) \
                                     + '{{{:d}}};'.format(curve_idx) + ' Layers{{{:d}}};}};\n'.format(self.extrude['layers'])
                        local_text += 'Physical Surface("{:s}", {:d}) += {{out[1]}};\n'.format(name, i + self.tags[2])
                        f.write(local_text)
                    f.write('\n')

                # Extrude physical surfaces and write as Physical Volumes
                surfaces_seen = []
                for i, (name, idxs) in enumerate(self.physical_surfaces.items()):
                    self.physical_groups['matrix'][name] = i + self.tags[3]
                    f.write('Physical Volume("{:s}", {:d}) = {{}};\n'.format(name, i + self.tags[3]))
                    for surface_idx in idxs:
                        local_text = 'out[] = Extrude {{{:f}, {:f}, {:f}}}{{ Surface'.format(extrusion[0], extrusion[1], extrusion[2]) \
                                     + '{{{:d}}};'.format(surface_idx) + ' Layers{{{:d}}};'.format(self.extrude['layers'])
                        if self.extrude['recombine']:
                            local_text += ' Recombine;'
                        local_text += '};\n'
                        local_text += 'Physical Volume("{:s}", {:d}) += {{out[1]}};\n'.format(name, i + self.tags[3])
                        f.write(local_text)
                        surfaces_seen.append(surface_idx)
                    f.write('\n')

                # Extrude non-Physical surfaces that haven't been extruded yet
                nps = len(self.physical_surfaces)  # add length of physical surfaces to physical volume index
                for i, surface in enumerate(self.surfaces):
                    if surface.idx not in surfaces_seen and surface.idx not in self.holes:
                        local_text = 'out[] = Extrude {{{:f}, {:f}, {:f}}}{{ Surface'.format(extrusion[0], extrusion[1], extrusion[2]) \
                                     + '{{{:d}}};'.format(surface.idx) + ' Layers{{{:d}}};'.format(self.extrude['layers'])
                        if self.extrude['recombine']:
                            local_text += ' Recombine;'
                        local_text += '};\n'
                        f.write(local_text)

                        # Add Physical Volume
                        self.physical_groups['matrix']['Volume_' + str(i+1)] = i + nps + self.tags[3]
                        f.write('Physical Volume("Volume_{:d}", {:d}) = {{out[1]}};\n'.format(i + 1, i + nps + self.tags[3]))
                f.write('\n')
        else:
            # Write 3D
            # Add Physical Points, Curves, Surfaces
            for i, (name, idxs) in enumerate(self.physical_points.items()):
                # self.physical_groups['matrix'][name] = i + self.tags[0]
                local_text = 'Physical Point("{:s}", {:d}) = {{'.format(name, i + self.tags[0])
                for point_idx in idxs:
                    local_text += '{:d}, '.format(point_idx)
                local_text = local_text[:-2]
                local_text += '};\n'
                f.write(local_text)

            for i, (name, idxs) in enumerate(self.physical_curves.items()):
                # self.physical_groups['matrix'][name] = i + self.tags[1]
                local_text = 'Physical Curve("{:s}", {:d}) = {{'.format(name, i + self.tags[1])
                for curve_idx in idxs:
                    local_text += '{:d}, '.format(curve_idx)
                local_text = local_text[:-2]
                local_text += '};\n'
                f.write(local_text)

            for i, (name, idxs) in enumerate(self.physical_surfaces.items()):
                self.physical_groups['boundary'][name] = i + self.tags[2]
                local_text = 'Physical Surface("{:s}", {:d}) = {{'.format(name, i + self.tags[2])
                for surface_idx in idxs:
                    local_text += '{:d}, '.format(surface_idx)
                local_text = local_text[:-2]
                local_text += '};\n'
                f.write(local_text)

            # Add Volumes
            for i, volume in enumerate(self.volumes):
                surfaces = volume.surfaces
                local_text = 'Surface Loop({:d}) = {{'.format(volume.idx)
                for surface in surfaces:
                    local_text += '{:d}, '.format(surface)
                local_text = local_text[:-2]
                local_text += '};\n'
                f.write(local_text)

                if volume.idx not in self.holes:
                    f.write('Volume({:d}) = {{{:d}}};\n'.format(volume.idx, volume.idx))
            f.write('\n')

            # Add Physical Volumes
            volumes_seen = []
            for i, (name, idxs) in enumerate(self.physical_volumes.items()):
                self.physical_groups['matrix'][name] = i + self.tags[3]
                local_text = 'Physical Volume("{:s}", {:d}) = {{'.format(name, i + self.tags[3])
                for volume_idx in idxs:
                    local_text += '{:d}, '.format(volume_idx)
                    volumes_seen.append(volume_idx)
                local_text = local_text[:-2]
                local_text += '};\n'
                f.write(local_text)
            f.write('\n')

            # Create Physical Volumes for each non-Physical volume
            npv = len(self.physical_volumes)  # add length of physical volumes to physical volume index
            for i, volume in enumerate(self.volumes):
                if volume.idx not in volumes_seen:
                    # Add Physical Volume
                    self.physical_groups['matrix']['Volume_' + str(i+1)] = i + npv + self.tags[3]
                    f.write('Physical Volume("Volume_{:d}", {:d}) = {{{:d}}};\n'.format(i + 1, i + npv + self.tags[3], i + 1))
            f.write('\n')

        # Find well surfaces and turn into physical surfaces
        f.write('Mesh {:d};  // Generate {:d}D mesh\n'.format(self.dim, self.dim))
        f.write('Coherence Mesh;  // Remove duplicate entities\n')
        f.close()
        # https://gmsh.info/doc/texinfo/gmsh.html#File-formats

        return

    def generate_msh(self, filename):
        # Gmsh API
        gmsh.initialize()
        gmsh.option.setNumber('General.Terminal', 0)
        gmsh.open(filename + '.geo')
        gmsh.model.mesh.generate(self.dim)

        gmsh.write(filename + '.msh')
        gmsh.finalize()

        return
