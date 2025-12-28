import numpy as np

from darts.engines import conn_mesh, index_vector, value_vector, timer_node

from darts.reservoirs.struct_reservoir import StructReservoir
from darts.reservoirs.unstruct_reservoir import UnstructReservoir
from darts.reservoirs.mesh.unstruct_discretizer import UnstructDiscretizer

from darts.reservoirs.mesh.geometry.unstructured import Unstructured
from darts.reservoirs.mesh.geometry.shapes import Square, Circle, MeshProperties

import meshio
import os


class RadialStruct(StructReservoir):
    boundary_cells = {'top': [], 'bottom': [], 'inner': [], 'outer': []}

    def __init__(self, timer: timer_node, nr: int, nz: int, dr, dz, permr, permz, poro, logspace: bool = False,
                 R0: float = 0., R1: float = None, angle: float = 360., top_depth=0, rcond=181.44, hcap=2200, op_num=0,
                 boundary_volume: float = None):
        """
        Structured radial reservoir class (1D/2D). Has option to create logarithmically increasing element size.

        :param timer: Timer from DartsModel class
        :type timer: timer_node
        :param nr: Number of elements in radial direction
        :type nr: int
        :param nz: Number of elements in vertical direction
        :type nz: int
        :param dr: Element size in radial direction
        :param dz: Element size in vertical direction
        :param permr: Permeability in radial direction
        :param permz: Permeability in vertical direction
        :param poro: Porosity
        :param logspace: Switch for logarithmic element sizes in radial direction
        :type logspace: bool
        :param R0: Inner radius [m], default is 0
        :type R0: float
        :param R1: Outer radius [m], only needs to be specified with logspace
        :type R1: float
        :param angle: Angle of radial slice [degrees], default is 360
        :type angle: float
        :param top_depth: Depth of the upper exterface of the uppermost layer [m]
        :param rcond: Rock conductivity [kJ/m.K.day]
        :param hcap: Rock heat capacity [kJ/m3.K]
        :param op_num: Operator numbers
        :param boundary_volume: Volume of outer boundary cells
        """
        depth_upper = top_depth
        if logspace:
            '''
            Near-centre refined grid
            '''
            # Find dr distribution such that outer radius is R1
            from scipy.optimize import fsolve
            f = lambda dr1: np.sum(np.logspace(np.log10(dr), np.log10(dr1), num=nr)) - R1
            dr1 = fsolve(f, dr)

            dr = np.logspace(start=np.log10(dr), stop=np.log10(dr1), num=nr)
        elif isinstance(dr, (int, float)):
            '''
            Uniform grid size in radial direction
            '''
            dr = np.ones(nr) * dr
        else:
            '''
            Pre-defined cell sizes
            '''
            assert nr == len(dr)
        R1 = R0 + np.sum(dr)

        # Calculate distance in radial direction
        r = [R0 + 0.5 * dr[0]]
        for i in range(1, nr):
            r.append(r[i-1] + 0.5*dr[i-1] + 0.5*dr[i])

        # Calculate area and corresponding dy for approximation of radial grid
        A_r = np.pi * np.array([np.abs((r[i] + 0.5 * dr[i]) ** 2 - (r[i] - 0.5 * dr[i]) ** 2) for i in range(nr)])
        dy = angle / (360. * dr) * A_r

        # If number of cells in vertical direction is larger than 1, adjust arrays of dr, dy, dz and r
        # if nz > 1:
        if isinstance(dz, (int, float)):
            '''
            Uniform grid size in vertical direction
            '''
            dz = np.ones(nz) * dz
        else:
            '''
            Pre-defined cell sizes
            '''
            assert nz == len(dz)

        dr = np.tile(dr, nz)
        dy = np.tile(dy, nz)

        dr = dr.reshape(nr, 1, nz)
        dy = dy.reshape(nr, 1, nz)

        DZ, Z = np.zeros((nr, 1, nz)), np.zeros((nr, 1, nz))

        for i in range(nr):
            DZ[i, 0, :] = dz
            Z[i, 0, :] = top_depth + np.cumsum(dz) - 0.5*dz[0]
        dz, depth = DZ, Z

        super().__init__(timer, nx=nr, ny=1, nz=nz, dx=dr, dy=dy, dz=dz, permx=permr, permy=permr, permz=permz,
                         poro=poro, depth=depth, rcond=rcond, hcap=hcap, op_num=op_num)

        # Fill boundary cells
        self.boundary_cells['top'] = [i for i in range(self.nx)]
        self.boundary_cells['bottom'] = [(self.nz-1) * self.nx + i for i in range(self.nx)]

        self.boundary_cells['inner'] = [k * self.nx for k in range(self.nz)]
        self.boundary_cells['outer'] = [(k+1) * self.nx - 1 for k in range(self.nz)]

        self.boundary_volumes['yz_plus'] = boundary_volume
        self.boundary_volumes['yz_minus'] = boundary_volume
        self.boundary_volumes['xy_plus'] = boundary_volume
        self.boundary_volumes['xy_minus'] = boundary_volume

        # radial mesh generation for VTK output
        r_vertices, z_vertices = R0 * np.ones(nr + 1), depth_upper * np.ones(nz + 1)
        if len(dr.shape) == 1:
            r_vertices[1:] = R0 + np.cumsum(dr[:nr])
        else:
            r_vertices[1:] = R0 +np.cumsum(dr[:nr, 0, 0])
        z_vertices[1:] = depth_upper + np.cumsum(dz[0, 0, :])
        self.generate_quarter_radial_grid(r_vert=r_vertices, z_vert=z_vertices, filename='quater_radial_grid')

    def generate_quarter_radial_grid(self, r_vert, z_vert, filename):
        nr, nz = r_vert.size, z_vert.size
        nphi = 10 + 1
        self.nphi = nphi - 1
        phi_vert = np.linspace(0, np.pi / 2, nphi)

        if r_vert[0] == 0.0:
            r_vert += 0.1
        phi, z, r = np.meshgrid(phi_vert, z_vert, r_vert, indexing='ij')

        cells = []
        cell_data = { 'cell_id': [np.zeros((nr - 1) * (nz - 1) * (nphi - 1))]}
        for k in range(nphi - 1):
            for j in range(nz - 1):
                for i in range(nr - 1):
                    pt1 = i + nr * (j + k * nz)
                    pt2 = i + nr * (j + k * nz) + 1
                    pt3 = i + nr * (j + 1 + k * nz) + 1
                    pt4 = i + nr * (j + 1 + k * nz)
                    pt5 = i + nr * (j + (k + 1) * nz)
                    pt6 = i + nr * (j + (k + 1) * nz) + 1
                    pt7 = i + nr * (j + 1 + (k + 1) * nz) + 1
                    pt8 = i + nr * (j + 1 + (k + 1) * nz)
                    cells.append([pt1, pt2, pt3, pt4, pt5, pt6, pt7, pt8])
                    cell_data['cell_id'][0][i + j * (nr - 1)] = i + j * (nr - 1) + k * (nr - 1) * (nz - 1)

        cells = [('hexahedron', np.array(cells))]

        points = np.vstack((r.flatten(), phi.flatten(), z.flatten())).T
        self.output_points = np.copy(points)
        self.output_points[:, 0] = points[:, 0] * np.cos(points[:, 1])
        self.output_points[:, 1] = points[:, 0] * np.sin(points[:, 1])
        self.output_cells = cells

    def output_to_vtk(self, ith_step: int, t: float, output_directory: str, prop_names: list, data: dict):
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        geometries = ['hexahedron']
        cell_data = {prop: [[] for geometry in geometries] for prop in prop_names}
        for prop in prop_names:
            cell_data[prop][0] += data[prop][0].tolist()

        mesh = meshio.Mesh(points=self.output_points, cells=self.output_cells, cell_data=cell_data)
        meshio.write("{:s}/solution{:d}.vtk".format(output_directory, ith_step), mesh)


class RadialUnstruct(UnstructReservoir):
    r = []

    def __init__(self, timer: timer_node, mesh_properties: MeshProperties, angle: float,
                 permx, permy, permz, poro, hcap=2200, rcond=181.44):
        """
        Class constructor for NearWellboreReservoir class

        :param mesh_properties:
        :param permx: Matrix permeability in the x-direction (scalar or vector)
        :param permy: Matrix permeability in the y-direction (scalar or vector)
        :param permz: Matrix permeability in the z-direction (scalar or vector)
        :param poro: Matrix (and fracture?) porosity (scalar or vector)
        """
        filename = 'mesh'

        m = Unstructured(dim=2, axs=[0, 1])
        if mesh_properties.square:
            m.add_shape(Square(center=mesh_properties.center, xlen=mesh_properties.xlen, ylen=mesh_properties.ylen,
                               zlen=mesh_properties.zlen, orientation=mesh_properties.orientation,
                               lc=mesh_properties.lc, radii=mesh_properties.radii, hole=mesh_properties.hole))
        else:
            m.add_shape(Circle(center=mesh_properties.center, orientation=mesh_properties.orientation, angle=angle,
                               lc=mesh_properties.lc, radii=mesh_properties.radii, hole=mesh_properties.hole))

        if mesh_properties.extrude:
            m.extrude_mesh(length=mesh_properties.extrude_length, layers=mesh_properties.extrude_layers,
                           axis=mesh_properties.extrude_axis, recombine=mesh_properties.extrude_recombine)

        m.write_geo(filename)
        m.generate_msh(filename)

        self.physical_groups = m.physical_groups

        self.physical_tags['matrix'] += [tag for name, tag in m.physical_groups['matrix'].items()]
        self.physical_tags['boundary'] += [tag for name, tag in m.physical_groups['boundary'].items()]

        super().__init__(timer=timer, mesh_file=filename + '.msh',
                         permx=permx, permy=permy, permz=permz, poro=poro, hcap=hcap, rcond=rcond)

    def discretize(self, verbose: bool = False):
        # Construct instance of Unstructured Discretization class:
        self.discretizer = UnstructDiscretizer(mesh_file=self.mesh_file, physical_tags=self.physical_tags,
                                               verbose=verbose)

        self.discretizer.n_dim = 3

        # Use class method load_mesh to load the GMSH file specified above:
        self.discretizer.load_mesh(permx=self.permx, permy=self.permy, permz=self.permz, frac_aper=0, cache=False)

        # Store volumes and depth to single numpy arrays:
        self.discretizer.store_volume_all_cells()
        self.discretizer.store_depth_all_cells()
        self.discretizer.store_centroid_all_cells()

        # Assign layer properties
        self.set_layer_properties()

        # Perform discretization:
        self.cell_m, self.cell_p, self.tran, self.tran_thermal = self.discretizer.calc_connections_all_cells()

        # Initialize mesh using built connection list
        self.mesh = conn_mesh()
        self.mesh.init(index_vector(self.cell_m), index_vector(self.cell_p),
                       value_vector(self.tran), value_vector(self.tran_thermal))

        # Create numpy arrays wrapped around mesh data (no copying, this will severely slow down the process!)
        np.array(self.mesh.poro, copy=False)[:] = self.poro
        np.array(self.mesh.rock_cond, copy=False)[:] = self.rcond
        np.array(self.mesh.heat_capacity, copy=False)[:] = self.hcap
        np.array(self.mesh.op_num, copy=False)[:] = self.op_num
        np.array(self.mesh.depth, copy=False)[:] = self.discretizer.depth_all_cells
        np.array(self.mesh.volume, copy=False)[:] = self.discretizer.volume_all_cells

        coord = self.discretizer.centroid_all_cells
        self.r = np.zeros(self.mesh.n_res_blocks)
        for ith_cell, xyz in enumerate(coord):
            self.r[ith_cell] = np.sqrt(xyz[0] ** 2 + xyz[1] ** 2)

        return
