import numpy as np

from darts.engines import timer_node
from darts.reservoirs.struct_reservoir import StructReservoir


class StructRadialReservoir(StructReservoir):
    def __init__(
        self,
        timer: timer_node,
        nz: int,
        dr,
        dz,
        poro,
        permr,
        permz,
        logspace: bool = False,
        nr: int = None,
        R0: float = 0.0,
        R1: float = None,
        depth: float = 0,
        rcond=181.44,
        hcap=2200.0,
        op_num=0,
        boundary_volume: float = None,
        innermost_block_volume: float = None,
    ):
        """
        Structured radial reservoir class (1D/2D). Has option to create logarithmically increasing element size.

        :param timer: Timer from DartsModel class
        :type timer: timer_node
        :param nz: Number of elements in vertical direction
        :type nz: int
        :param dr: Element size in radial direction (in logspace case, first grid block size)
        :param dz: Element size in vertical direction
        :param poro: Porosity
        :param permr: Layer permeability in radial direction (length nz)
        :param permz: Layer permeability in vertical direction (length nz)
        :param logspace: Switch for logarithmic element sizes in radial direction
        :type logspace: bool
        :param nr: Number of elements in radial direction (only in case of logspace)
        :type nr: int
        :param R0: Inner radius [m], default is 0
        :type R0: float
        :param R1: Outer radius [m]
        :type R1: float
        :param depth: Depth of centroid of top layer [m]
        :type depth: float
        :param rcond: Rock conductivity [kJ/m.K.day]
        :param hcap: Rock volumetric heat capacity [kJ/m3.K]
        :param op_num: Operator numbers
        :param boundary_volume: Volume of outer boundary cells
        :type boundary_volume: float
        :param innermost_block_volume: Volume of the innermost block. This can be used if the user needs to set a large
        volume to the innermost grid block. This can be useful when using the DFM well model as a standalone wellbore
        model in open-DARTS.
        :type innermost_block_volume: float
        """
        # Get top exterface (external or boundary face) depth for VTK output
        if isinstance(dz, int | float):
            top_exterface_depth = depth - dz / 2
        else:
            top_exterface_depth = depth - dz[0] / 2

        if logspace:
            '''
            Near-centre refined grid
            '''
            assert nr is not None and R1 is not None, (
                "Please provide nr and R1 arguments for logspace reservoir"
            )
            # Find dr distribution such that outer radius is R1
            from scipy.optimize import fsolve

            f = (
                lambda dr1: np.sum(np.logspace(np.log10(dr), np.log10(dr1), num=nr))
                - R1
            )
            dr1 = fsolve(f, dr)[0]

            dr = np.logspace(start=np.log10(dr), stop=np.log10(dr1), num=nr)
        elif isinstance(dr, int | float):
            '''
            Uniform grid size in radial direction
            '''
            dr = np.ones(int((R1 - R0) / dr)) * dr
        else:
            '''
            Pre-defined cell sizes
            '''
            pass

        nr = len(dr)
        R1 = R0 + np.sum(dr)

        # Calculate distance in radial direction
        r = [R0 + 0.5 * dr[0]]
        for i in range(1, nr):
            r.append(r[i - 1] + 0.5 * dr[i - 1] + 0.5 * dr[i])
        self.r = r

        # Calculate area and corresponding dy for approximation of radial grid
        A_r = np.pi * np.array(
            [
                np.abs((r[i] + 0.5 * dr[i]) ** 2 - (r[i] - 0.5 * dr[i]) ** 2)
                for i in range(nr)
            ]
        )
        dy = A_r / dr

        # If number of cells in vertical direction is larger than 1, adjust arrays of dr, dy, dz and r
        if nz > 1:
            if isinstance(dz, int | float):
                '''
                Uniform grid size in vertical direction
                '''
                dz = np.ones(nz) * dz
            else:
                '''
                Pre-defined cell sizes
                '''
                assert nz == len(dz)

            # Create tile of dr, dy and dz to pass to StructReservoir constructor as dx, dy, dz
            dr = np.tile(dr, (nz, 1)).transpose()
            dy = np.tile(dy, (nz, 1)).transpose()
            dz = np.tile(dz, (nr, 1))

            dr = dr.reshape(nr, 1, nz)
            dy = dy.reshape(nr, 1, nz)
            dz = dz.reshape(nr, 1, nz)

            # Depth of exterface of the top block
            z_top_exterface = depth - 0.5 * dz[0, 0, 0]
            # Depth of top faces of all blocks
            z_top_faces = z_top_exterface + np.concatenate(
                ([0.0], np.cumsum(dz[0, 0, :-1]))
            )
            # Depths of centroids of all blocks
            self.z = z_top_faces + dz[0, 0, :] / 2.0

            depth = z_top_faces[None, None, :] + dz / 2.0

        else:
            self.z = np.array([depth])

        # Create poro and perm arrays
        # Poro and perm arrays can be scalar, 2D (nr, nz) or 1D (len nr or nz, which means homogeneous in the other direction)
        if hasattr(poro, "__len__"):
            assert (
                np.shape(poro) == (nr, nz)
                if poro.ndim == 2
                else (len(poro) == nr or len(poro) == nz)
            ), ("Length of poro array is incompatible with nr/nz: ", poro.shape, nr, nz)
        else:
            poro = np.ones((nr, nz)) * poro
        poro = (
            poro
            if poro.ndim == 2
            else (
                np.tile(poro, (nr, 1))
                if len(poro) == nz
                else np.tile(poro, (nz, 1)).transpose()
            )
        )
        poro = poro.flatten(order="F")

        if hasattr(permr, "__len__"):
            assert (
                np.shape(permr) == (nr, nz)
                if permr.ndim == 2
                else (len(permr) == nr or len(permr) == nz)
            ), (
                "Length of permr array is incompatible with nr/nz: ",
                permr.shape,
                nr,
                nz,
            )
        else:
            permr = np.ones((nr, nz)) * permr
        permr = (
            permr
            if permr.ndim == 2
            else (
                np.tile(permr, (nr, 1))
                if len(permr) == nz
                else np.tile(permr, (nz, 1)).transpose()
            )
        )
        permr = permr.flatten(order="F")

        if hasattr(permz, "__len__"):
            assert (
                np.shape(permz) == (nr, nz)
                if permz.ndim == 2
                else (len(permz) == nr or len(permz) == nz)
            ), (
                "Length of permz array is incompatible with nr/nz: ",
                permz.shape,
                nr,
                nz,
            )
        else:
            permz = np.ones((nr, nz)) * permz
        permz = (
            permz
            if permz.ndim == 2
            else (
                np.tile(permz, (nr, 1))
                if len(permz) == nz
                else np.tile(permz, (nz, 1)).transpose()
            )
        )
        permz = permz.flatten(order="F")

        super().__init__(
            timer,
            nx=nr,
            ny=1,
            nz=nz,
            dx=dr,
            dy=dy,
            dz=dz,
            permx=permr,
            permy=permr,
            permz=permz,
            poro=poro,
            depth=depth,
            rcond=rcond,
            hcap=hcap,
            op_num=op_num,
        )

        # Fill boundary cells
        self.boundary_cells = {'top': [], 'bottom': [], 'inner': [], 'outer': []}
        self.boundary_cells['top'] = [i for i in range(self.nx)]
        self.boundary_cells['bottom'] = [
            (self.nz - 1) * self.nx + i for i in range(self.nx)
        ]

        self.boundary_cells['inner'] = [k * self.nx for k in range(self.nz)]
        self.boundary_cells['outer'] = [(k + 1) * self.nx - 1 for k in range(self.nz)]

        self.boundary_volumes['yz_plus'] = boundary_volume
        self.boundary_volumes['xy_plus'] = boundary_volume
        self.boundary_volumes['xy_minus'] = boundary_volume

        if innermost_block_volume is not None:
            self.boundary_volumes['yz_minus'] = innermost_block_volume

        # radial mesh generation for VTK output
        self.r_vertices, self.z_vertices = (
            R0 * np.ones(nr + 1),
            top_exterface_depth * np.ones(nz + 1),
        )
        if len(dr.shape) == 1:
            self.r_vertices[1:] = R0 + np.cumsum(dr[:nr])
        else:
            self.r_vertices[1:] = R0 + np.cumsum(dr[:nr, 0, 0])
        self.z_vertices[1:] = (
            top_exterface_depth + np.cumsum(dz[0, 0, :])
            if nz > 1
            else top_exterface_depth + dz
        )
        self.generate_quarter_radial_grid(
            r_vert=self.r_vertices,
            z_vert=self.z_vertices,
            filename='quater_radial_grid',
        )

    def set_wells(self, verbose: bool = False):
        for well_name, cell_idxs in self.well_dict.items():
            # Add production well:
            self.add_well(well_name)

            # Perforate all boundary cells:
            for idxs in cell_idxs:
                # the parameter is res_cell_idx on every add_perforation
                # signature; the historical cell_index= spelling raised TypeError
                self.add_perforation(well_name, res_cell_idx=idxs)
                # self.add_perforation(well_name, res_cell_idx=idxs, well_index=100, well_indexD=100)

        return

    def generate_quarter_radial_grid(self, r_vert, z_vert, filename):
        nr, nz = r_vert.size, z_vert.size
        nphi = 10 + 1
        self.nphi = nphi - 1
        phi_vert = np.linspace(0, np.pi / 2, nphi)

        if r_vert[0] == 0.0:
            r_vert += 0.1
        phi, z, r = np.meshgrid(phi_vert, z_vert, r_vert, indexing='ij')

        # Flip depth sign for VTK (positive z in DARTS is downward, while negative z in ParaView is downward)
        z = -z

        cells = []
        cell_data = {'cell_id': [np.zeros((nr - 1) * (nz - 1) * (nphi - 1))]}
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
                    cell_data['cell_id'][0][i + j * (nr - 1)] = (
                        i + j * (nr - 1) + k * (nr - 1) * (nz - 1)
                    )

        cells = [('hexahedron', np.array(cells))]

        points = np.vstack((r.flatten(), phi.flatten(), z.flatten())).T
        self.output_points = np.copy(points)
        self.output_points[:, 0] = points[:, 0] * np.cos(points[:, 1])
        self.output_points[:, 1] = points[:, 0] * np.sin(points[:, 1])
        self.output_cells = cells

    def populate_data_for_radial_vtk_output(self, data, prop_names: list):
        new_data = {}
        for ith_prop, prop in enumerate(prop_names):
            # populate r-z data to all angles
            new_data[prop] = np.tile(data[ith_prop, :], self.nphi)

        return new_data

    def output_to_vtk(
        self,
        ith_step: int,
        t: float,
        output_directory: str,
        prop_names: list,
        data: dict,
    ):
        import os

        import meshio

        from darts.tools.vtk_io import write_pvd

        os.makedirs(output_directory, exist_ok=True)

        data = self.populate_data_for_radial_vtk_output(data, prop_names)

        geometries = ['hexahedron']
        cell_data = {prop: [[] for geometry in geometries] for prop in prop_names}
        for prop in prop_names:
            cell_data[prop][0] += data[prop].tolist()

        mesh = meshio.Mesh(
            points=self.output_points, cells=self.output_cells, cell_data=cell_data
        )
        vtk_filename = f"solution_ts{ith_step:d}.vtu"
        meshio.write(os.path.join(output_directory, vtk_filename), mesh)

        if not hasattr(self, 'vtk_filenames_and_times'):
            self.vtk_filenames_and_times = {}
        self.vtk_filenames_and_times[vtk_filename] = t

        write_pvd(
            os.path.join(output_directory, "solution.pvd"),
            [(sim_t, fname) for fname, sim_t in self.vtk_filenames_and_times.items()],
        )
