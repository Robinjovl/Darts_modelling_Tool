import numpy as np

from darts.engines import conn_mesh, index_vector, timer_node, value_vector
from darts.reservoirs.mesh.geometry.shapes import Circle, MeshProperties, Square
from darts.reservoirs.mesh.geometry.unstructured import Unstructured
from darts.reservoirs.mesh.unstruct_discretizer import UnstructDiscretizer
from darts.reservoirs.unstruct_reservoir import UnstructReservoir


class UnstructRadialReservoir(UnstructReservoir):
    def __init__(
        self,
        timer: timer_node,
        mesh_properties: MeshProperties,
        poro,
        permr,
        permz,
        rcond=181.44,
        hcap=2200.0,
    ):
        """
        Unstructured radial reservoir class

        :param timer: Timer from DartsModel class
        :type timer: timer_node
        :param mesh_properties:
        :type mesh_properties: MeshProperties
        :param poro: Matrix (and fracture?) porosity
        :type poro: float or vector
        :param permx: Matrix permeability in the x-direction
        :type permx: float or vector
        :param permy: Matrix permeability in the y-direction
        :type permy: float or vector
        :param permz: Matrix permeability in the z-direction
        :type permz: float or vector
        :param rcond: Rock conductivity [kJ/m.K.day]
        :param hcap: Rock volumetric heat capacity [kJ/m3.K]
        """
        filename = 'mesh'

        m = Unstructured(dim=2, axs=[0, 1])
        if mesh_properties.square:
            m.add_shape(
                Square(
                    center=mesh_properties.center,
                    xlen=mesh_properties.xlen,
                    ylen=mesh_properties.ylen,
                    zlen=mesh_properties.zlen,
                    orientation=mesh_properties.orientation,
                    lc=mesh_properties.lc,
                    radii=mesh_properties.radii,
                    hole=mesh_properties.hole,
                )
            )
        else:
            m.add_shape(
                Circle(
                    center=mesh_properties.center,
                    orientation=mesh_properties.orientation,
                    angle=360.0,
                    lc=mesh_properties.lc,
                    radii=mesh_properties.radii,
                    hole=mesh_properties.hole,
                )
            )

        if mesh_properties.extrude:
            m.extrude_mesh(
                length=mesh_properties.extrude_length,
                layers=mesh_properties.extrude_layers,
                axis=mesh_properties.extrude_axis,
                recombine=mesh_properties.extrude_recombine,
            )

        m.write_geo(filename)
        m.generate_msh(filename)

        self.physical_groups = m.physical_groups

        self.physical_tags['matrix'] += [
            tag for name, tag in m.physical_groups['matrix'].items()
        ]
        self.physical_tags['boundary'] += [
            tag for name, tag in m.physical_groups['boundary'].items()
        ]

        self.r = []

        super().__init__(
            timer=timer,
            mesh_file=filename + '.msh',
            permx=permr,
            permy=permr,
            permz=permz,
            poro=poro,
            hcap=hcap,
            rcond=rcond,
        )

    def discretize(self, verbose: bool = False):
        # Construct instance of Unstructured Discretization class:
        self.discretizer = UnstructDiscretizer(
            mesh_file=self.mesh_file, physical_tags=self.physical_tags, verbose=verbose
        )

        self.discretizer.n_dim = 3

        # Use class method load_mesh to load the GMSH file specified above:
        self.discretizer.load_mesh(
            permx=self.permx,
            permy=self.permy,
            permz=self.permz,
            frac_aper=0,
            cache=False,
        )

        # Store volumes and depth to single numpy arrays:
        self.discretizer.store_volume_all_cells()
        self.discretizer.store_depth_all_cells()
        self.discretizer.store_centroids_all_cells()

        # Assign layer properties
        self.set_layer_properties()

        # Perform discretization:
        self.cell_m, self.cell_p, self.tran, self.tran_thermal = (
            self.discretizer.calc_connections_all_cells()
        )

        # Initialize mesh using built connection list
        self.mesh = conn_mesh()
        self.mesh.init(
            index_vector(self.cell_m),
            index_vector(self.cell_p),
            value_vector(self.tran),
            value_vector(self.tran_thermal),
        )

        # Create numpy arrays wrapped around mesh data (no copying, this will severely slow down the process!)
        np.array(self.mesh.poro, copy=False)[:] = self.poro
        np.array(self.mesh.rock_cond, copy=False)[:] = self.rcond
        np.array(self.mesh.heat_capacity, copy=False)[:] = self.hcap
        np.array(self.mesh.op_num, copy=False)[:] = self.op_num
        np.array(self.mesh.depth, copy=False)[:] = self.discretizer.depth_all_cells
        np.array(self.mesh.volume, copy=False)[:] = self.discretizer.volume_all_cells

        coord = self.discretizer.centroids_all_cells
        self.r = np.zeros(self.mesh.n_res_blocks)
        for ith_cell, xyz in enumerate(coord):
            self.r[ith_cell] = np.sqrt(xyz[0] ** 2 + xyz[1] ** 2)

        return

    def set_wells(self, verbose: bool = False):
        # Add production well:
        self.add_well(well_name="P1")

        # Perforate all boundary cells:
        boundary_cells = self.discretizer.find_cells(
            self.physical_groups['boundary']['inner'], 'face'
        )
        for _nth_perf, cell_index in enumerate(boundary_cells):
            self.add_perforation(
                well_name="P1",
                res_cell_idx=cell_index,
                well_index=100,
                well_indexD=100,
            )

        return
