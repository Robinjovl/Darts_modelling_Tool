import numpy as np
import math
import warnings
from .geometry import Geometry

import gmsh


class Unstructured(Geometry):
    def __init__(self, dim, axs=[0, 1, 2]):
        super().__init__(dim, axs)

        self.lc = []
        self.extrude: dict = None

        self.tags = [901, 9001, 90001, 900001]  # tags for physical points, lines, surfaces, volumes
        # self.physical_tags = {
        #     "matrix": [],
        #     "boundary": [],
        #     "fracture": [],
        #     "fracture_shape": [],
        #     "well": [],
        #     "output": []
        # }

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
            if surface.idx not in self.holes:
                if surface.plane:
                    local_text = 'Plane Surface({:d}) = {{{:d}, '.format(surface.idx, surface.idx)
                else:
                    local_text = 'Surface({:d}) = {{{:d}, '.format(surface.idx, surface.idx)

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
            if self.extrude is not None:
                # Extruded 2D geometry
                self.dim = 3

                extrusion = [0, 0, 0]
                extrusion[self.extrude['axis']] = self.extrude['length']

                # Extrude surface
                # Write Physical Volume: out[1]
                # Write Physical Curve: out[0], out[2], ...
                # Physical Point: ??
                for i, physical_point in enumerate(self.physical_points):
                    f.write('Physical Point("{:s}", {:d}) = {{}};\n'.format(physical_point.tag, i + self.tags[0]))
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
                for i, physical_curve in enumerate(self.physical_curves):
                    f.write('Physical Surface("{:s}", {:d}) = {{}};\n'.format(physical_curve.tag, i + self.tags[2]))
                    for curve_idx in physical_curve.idxs:
                        local_text = 'out[] = Extrude {{{:f}, {:f}, {:f}}}{{ Curve'.format(extrusion[0], extrusion[1], extrusion[2]) \
                                     + '{{{:d}}};'.format(curve_idx) + ' Layers{{{:d}}};}};\n'.format(self.extrude['layers'])
                        local_text += 'Physical Surface("{:s}", {:d}) += {{out[1]}};\n'.format(physical_curve.tag, i + self.tags[2])
                        f.write(local_text)
                    f.write('\n')

                # Extrude physical surfaces and write as Physical Volumes
                surfaces_seen = []
                for i, physical_surface in enumerate(self.physical_surfaces):
                    f.write('Physical Volume("{:s}", {:d}) = {{}};\n'.format(physical_surface.tag, i + self.tags[3]))
                    for surface_idx in physical_surface.idxs:
                        local_text = 'out[] = Extrude {{{:f}, {:f}, {:f}}}{{ Surface'.format(extrusion[0], extrusion[1], extrusion[2]) \
                                     + '{{{:d}}};'.format(surface_idx) + ' Layers{{{:d}}};'.format(self.extrude['layers'])
                        if self.extrude['recombine']:
                            local_text += ' Recombine;'
                        local_text += '};\n'
                        local_text += 'Physical Volume("{:s}", {:d}) += {{out[1]}};\n'.format(physical_surface.tag, i + self.tags[3])
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
                        f.write('Physical Volume("Volume_{:d}", {:d}) = {{out[1]}};\n'.format(i + 1, i + nps + self.tags[3]))
                f.write('\n')
        else:
            # Write 3D
            # Add Physical Points, Curves, Surfaces
            for i, physical_point in enumerate(self.physical_points):
                local_text = 'Physical Point("{:s}", {:d}) = {{'.format(physical_point.tag, i + self.tags[0])
                for point_idx in physical_point.idxs:
                    local_text += '{:d}, '.format(point_idx)
                local_text = local_text[:-2]
                local_text += '};\n'
                f.write(local_text)

            for i, physical_curve in enumerate(self.physical_curves):
                local_text = 'Physical Curve("{:s}", {:d}) = {{'.format(physical_curve.tag, i + self.tags[1])
                for curve_idx in physical_curve.idxs:
                    local_text += '{:d}, '.format(curve_idx)
                local_text = local_text[:-2]
                local_text += '};\n'
                f.write(local_text)

            for i, physical_surface in enumerate(self.physical_surfaces):
                local_text = 'Physical Surface("{:s}", {:d}) = {{'.format(physical_surface.tag, i + self.tags[2])
                for surface_idx in physical_surface.idxs:
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

                f.write('Volume({:d}) = {{{:d}}};\n'.format(volume.idx, volume.idx))
            f.write('\n')

            # Add Physical Volumes
            volumes_seen = []
            for i, physical_volume in enumerate(self.physical_volumes):
                local_text = 'Physical Volume("{:s}", {:d}) = {{'.format(physical_volume.tag, i + self.tags[3])
                for volume_idx in physical_volume.idxs:
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
