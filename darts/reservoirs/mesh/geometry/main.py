import numpy as np
from structured import Structured
from unstructured import Unstructured
from geometry import Geometry
from shapes import *
from wells import *


if __name__ == "__main__":
    geo = 'fluidflower'
    if geo == 'cylinder':
        c = Cylinder(center=[0., 0., 0.], radii=[1, 0.3], lc=[0.5, 0.1], length=3,
                     orientation='xz', angle=300, hole=1)

        m = Unstructured(dim=3)
        m.add_shape(c)

    elif geo == 'circle':
        c = Circle(center=[0., 0., 0.], radii=[10, 4], lc=[2, 1], orientation='yz', angle=300, hole=0)

        m = Unstructured(dim=2, axs=[1, 2])
        m.add_shape(c)

        m.extrude_mesh(length=100, layers=10, axis=0, recombine=True)

    elif geo == 'box':
        b = Box(center=[0., 0., 0.], lc=[0.1, 0.1, 0.05], xlen=1., ylen=1., zlen=1., orientation='yz',
                radii=[0.3, 0.1], hole=1)

        m = Unstructured(dim=3)
        m.add_shape(b)

    elif geo == 'square':
        s = Square(center=[0., 0., 0.], lc=[0.1, 0.1, 0.05], xlen=1., ylen=1., zlen=1., orientation='yz',
                   radii=[0.3, 0.1], hole=1)

        m = Unstructured(dim=2)
        m.add_shape(s)

        m.extrude_mesh(length=1, layers=10, axis=0, recombine=True)
    elif geo == 'layered':
        b1 = Box(center=[0., 0., 0.], lc=[0.1, 0.1, 0.05], xlen=1., ylen=1., zlen=1., orientation='xy',
                 radii=[], hole=1)
        b2 = Box(center=[0., 0., 1.], lc=[0.1, 0.1, 0.05], xlen=1., ylen=1., zlen=1., orientation='xy',
                 radii=[], hole=1)

        # c_0 = [0., 0., 0.]
        # c1 = Cylinder(c_0, radius=1, length=2, orientation=0, angle=350)
        # c_1 = [2., 0., 0.]
        # c2 = Cylinder(c_1, radius=1, length=2, orientation=0, angle=30)

        m = Unstructured(dim=3)
        m.add_shape(b1)
        m.add_shape(b2)
        # m.add_shape(c1)
        # m.add_shape(c2)
    elif geo == 'fluidflower':
        from fluidflower import FluidFlower
        f = FluidFlower(lc=[0.025])
        f.plot_shape_2D()

        structured = False

        well_centers = {"I1": [0.925, 0.0, 0.32942],
                        "I2": [1.72806, 0.0, 0.72757]
                        }

        if structured:
            m = Structured(dim=2, axs=[0, 2])
            m.add_shape(f)

            m.generate_mesh2(nx=400, ny=1, nz=200)

            exit()

        else:
            m = Unstructured(dim=2, axs=[0, 2])
            m.add_shape(f)

            for i, (name, center) in enumerate(well_centers.items()):
                in_surfaces = m.find_surface(center)
                m.refine_around_point(center, radius=0.015, lc=len(m.lc))

                w = WellCell(center, lc=0.005, orientation='xz', in_surfaces=in_surfaces)
                # w = CircularWell(center, re=0.01, lc_well=1, axs=[0, 2], in_surfaces=in_surfaces)
                m.add_shape(w)

            m.extrude_mesh(length=0.025, layers=1, axis=1, recombine=True)

    else:
        center = [0., 0., 0.]
        angle = 80.

        if 1:
            m = Unstructured(dim=2, axs=[0, 1])
            # m.add_shape(Circle(center=center, radii=[1., 0.2], angle=angle, hole=True))
            m.add_shape(Square(center=center, ax0_len=2, ax1_len=2, orientation='xy', radii=[0.5, 0.2, 0.1], hole=True))
            # m.add_shape(CircularWell([0.5, 0.5, 0.], re=0.01, lc_well=1, axs=[0, 1], in_surfaces=[1]))
            m.extrude_mesh(length=0.1, layers=3, axis=2, recombine=True)
        else:
            m = Unstructured(dim, axs=[0, 1])
            m.add_shape(Box(xdim=[-1., 1.], ydim=[-1., 1.], zdim=[-1., 1.], radii=[]))
            # m.add_shape(Cylinder(center=center, radii=[1., 0.5], length=1., angle=angle, hole=True))

        # m.embed_point([0.5, 0.5, 0.], lc=1)
        # m.embed_circle([0.5, 0.5, 0.], radius=0.25, orientation='xy', lc=0)
        # m.embed_circle([0.5, 0.5, 0.], radius=0.05, orientation='xy', lc=1)
        # m.embed_circle(center=center, radius=0.5, angle=angle, lc=1, orientation='xy')

        m.lc = [0.25, 0.1, 0.05, 0.01]

    filename = 'mesh'
    m.write_geo(filename)
    m.generate_msh(filename)
