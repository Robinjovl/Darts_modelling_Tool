import numpy as np
from structured import Structured
from unstructured import Unstructured
from geometry import Geometry
from shapes import *
from wells import *


if __name__ == "__main__":
    geo = 'fluidflower'
    if geo == 'cylinder':
        filename = geo

        dim = 3
        thickness = 3
        orientation = 1
        angle = 360

        c0 = [0., 0., 0.]
        c = Cylinder(c0, radius=1, length=thickness, orientation=orientation, angle=angle)
        c.add_boundary("top")
        c.add_boundary("bottom")
        c.add_boundary("outer")
        # w = Cylinder(c0, radius=1, length=thickness, orientation=orientation, angle=angle)

        u = Unstructured(dim)
        u.add_shape(c)
        u.add_shape(w)

        u.write_geo(filename, lc=[0.2])
        u.generate_msh(filename)
    elif geo == 'cylinder_with_well':
        filename = geo

        dim = 3
        thickness = 20
        orientation = 2
        angle = 30

        c0 = [0., 0., 0.]
        c = CylinderWithHole(c0, radius=1000, rw=1, length=thickness, orientation=orientation, angle=angle)

        g = Geometry(dim)
        g.add_shape(c)

        g.write_geo(filename + '.geo', lc=[40, 2])
    elif geo == 'extruded_circle':
        filename = geo

        dim = 2
        thickness = 10
        orientation = 2
        angle = 30

        c0 = [0., 0., 0.]
        c = Circle(c0, radius=1000, orientation=orientation, angle=angle)
        # w = Circle(c0, radius=1, orientation=orientation, angle=angle)

        g = Geometry(dim)
        g.add_shape(c)
        # g.add_shape(w)

        g.extrusion['Axis'] = orientation
        g.extrusion['Distance'] = thickness
        g.extrusion['Layers'] = 1
        g.extrusion['Recombine'] = True

        g.write_geo(filename + '.geo', lc=[40])
    elif geo == 'extruded_circle_with_well':
        filename = geo

        dim = 2
        thickness = 20
        orientation = 2
        angle = 30

        c0 = [0., 0., 0.]
        c = CircleWithHole(c0, radius=1000, rw=0.075, orientation=orientation, angle=angle)

        g = Geometry(dim)
        g.add_shape(c)

        g.extrusion['Axis'] = orientation
        g.extrusion['Distance'] = thickness
        g.extrusion['Layers'] = 1
        g.extrusion['Recombine'] = True

        g.write_geo(filename + '.geo', lc=[40, 0.05])
    elif geo == 'box':
        filename = geo

        dim = 3

        b = Box([0, 1], [0, 1], [0, 1])

        g = Geometry(dim)
        g.add_shape(b)

        g.write_geo(filename + '.geo', lc=[0.1])
    elif geo == 'layered':
        filename = geo

        dim = 3

        b1 = Box([0, 1], [0, 1], [0, 1])
        b2 = Box([0, 1], [0, 1], [1, 2])

        c_0 = [0., 0., 0.]
        c1 = Cylinder(c_0, radius=1, length=2, orientation=0, angle=350)
        c_1 = [2., 0., 0.]
        c2 = Cylinder(c_1, radius=1, length=2, orientation=0, angle=30)

        g = Geometry(dim)
        # g.add_shape(b1)
        # g.add_shape(b2)
        g.add_shape(c1)
        g.add_shape(c2)
        g.write_geo(filename + '.geo', lc=[0.1])
    elif geo == 'fluidflower':
        from fluidflower import FluidFlower
        filename = geo

        f = FluidFlower()
        f.plot_shape_2D()

        dim = 2
        structured = False

        well_centers = [[0.925, 0.0, 0.32942],
                        [1.72806, 0.0, 0.72757]]

        if structured:
            s = Structured(dim=2, axs=[0, 2])
            s.add_shape(FluidFlower())

            s.generate_mesh2(nx=400, ny=1, nz=200)

        else:
            u = Unstructured(dim=2, axs=[0, 2])
            u.add_shape(FluidFlower())

            for i, center in enumerate(well_centers):
                in_surfaces = u.find_surface(center)
                u.refine_around_point(center, radius=0.015, lc=1)

                w = CircularWell(center, re=0.01, lc_well=1, axs=[0, 2], in_surfaces=in_surfaces)
                u.add_shape(w)

            u.extrude_mesh(length=0.025, layers=1, axis=1, recombine=True)
            u.lc = [0.025, 0.005]

            u.write_geo(filename)
            u.generate_msh(filename)

    elif geo == 'geo_file':
        filename = 'extruded_circle_with_well'
    else:
        raise Exception("geo not implemented")
