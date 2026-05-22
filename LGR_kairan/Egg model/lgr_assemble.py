import numpy as np
import os
import meshio
from pyevtk.vtk import VtkGroup
from types import SimpleNamespace
from darts.reservoirs.reservoir_base import ReservoirBase
from darts.engines import(
    conn_mesh,
    index_vector,
    value_vector,
    ms_well,
)
from lgr_scaled_fc import build_flow_based_scaled_fc

def assemble_lgr_connections(self):
    # STEP 1 discretize Level 0
    self.level0.discretize()
    disc0 = self.level0.discretizer
    g2l0 = disc0.global_to_local
    l2g0 = disc0.local_to_global
    n0_act = len(l2g0)

    cm0 = self.level0.cell_m
    cp0 = self.level0.cell_p
    T0 = self.level0.tran
    T0_therm = self.level0.tran_thermal

    # STEP 2: discretize real fine grids and assign offsets
    # -----------------------------
    lgr_orders = list(self.lgrs.keys())
    lgr_offsets = {}

    ff_cm = []
    ff_cp = []
    ff_T = []
    ff_Tt = []

    offset = n0_act
    for name in lgr_orders:
        self.level1[name].discretize()
        disc1 = self.level1[name].discretizer

        cm1, cp1, T1, T1_therm = disc1.calc_structured_discr()

        cm1 = np.asarray(cm1, dtype=int)
        cp1 = np.asarray(cp1, dtype=int)
        T1 = np.asarray(T1, dtype=float)
        T1_therm = np.asarray(T1_therm, dtype=float)

        lgr_offsets[name] = offset

        ff_cm.append(cm1 + offset)
        ff_cp.append(cp1 + offset)
        ff_T.append(T1)
        ff_Tt.append(T1_therm)

        offset += self.level1[name].n

    # flatten fine-fine
    ff_cm = np.concatenate(ff_cm) if ff_cm else np.array([], dtype=int)
    ff_cp = np.concatenate(ff_cp) if ff_cp else np.array([], dtype=int)
    ff_T = np.concatenate(ff_T) if ff_T else np.array([], dtype=float)
    ff_Tt = np.concatenate(ff_Tt) if ff_Tt else np.array([], dtype=float)


    # step 3: extract fine-coarse from imaginary grid

    fc_cm, fc_cp, fc_T, fc_Tt, fc_debug_df = build_flow_based_scaled_fc(
        model=self,
        lgr_orders=lgr_orders,
        lgr_offsets=lgr_offsets,
        verbose=True,
    )
    self.lgr_fc_debug_df = fc_debug_df

    # fc_cm = []
    # fc_cp = []
    # fc_T = []
    # fc_Tt = []

    # for name in lgr_orders:
    #     cfg = self.lgrs[name]
    #     ic = cfg['lgr_coords_in_parent_grid']['i_range'][0]
    #     jc = cfg['lgr_coords_in_parent_grid']['j_range'][0]
    #     k1, k2 = cfg['lgr_coords_in_parent_grid']['k_range']
    #     rx,ry,rz = cfg['lgr_coords_in_parent_grid']['refine']
    #     nk = k2 - k1 + 1

    #     self.level1_imag[name].discretize()
    #     disc_im = self.level1_imag[name].discretizer
    #     cmi, cpi, Ti, Ti_therm = disc_im.calc_structured_discr()
    #     # customized transmissibility scaling for fine-coarse connections, to better match the original model's behavior.
    #     # alpha = 3.624043958155789 / 2.8422382239720037
    #     # Ti = Ti * alpha

    #     nxim = rx +2
    #     nyim = ry +2
    #     nxy_im = nxim * nyim
    #     plane_size = rx * ry
    #     fine_global_offset = lgr_offsets[name]

    #     # build per layer fine/ halo index map
    #     fine_im_set = set()
    #     coarse_im_set = set()
    #     imag_fine_to_global = {} # this global means consider actnum and concatenate lgr to the end of level0
    #     image_ring_to_level0 = {}

    #     for kk, k in enumerate(range(k1, k2 +1)):
    #         nbr_g = {
    #             "left" : self.convert_ijk_to_gindex_1based(ic-1, jc, k, self.level0.nx, self.level0.ny),
    #             "right" : self.convert_ijk_to_gindex_1based(ic+1, jc, k, self.level0.nx, self.level0.ny),
    #             "up" : self.convert_ijk_to_gindex_1based(ic, jc-1, k, self.level0.nx, self.level0.ny),
    #             "down" : self.convert_ijk_to_gindex_1based(ic, jc+1, k, self.level0.nx, self.level0.ny),
    #         }
    #         nbr_l = {k: int(g2l0[v]) for k, v in nbr_g.items()} # local indices in level0 after actnum squeeze

    #         base_im = kk * nxy_im
    #         base_fine = kk * plane_size

    #         # central fine cells in imaginary grid

    #         for j in range(1, ry+1):
    #             for i in range(1, rx+1):
    #                 imag_idx = base_im + j*nxim + i # index of 5*5 grid
    #                 fine_local = base_fine + (j-1)*rx + (i-1) # local fine index in level1
    #                 fine_global = fine_global_offset + fine_local # global index of fine cell

    #                 imag_fine_to_global[imag_idx] = fine_global
    #                 fine_im_set.add(imag_idx)

    #         # halo cells
    #         for jj in range(1,ry+1):
    #             left_idx = base_im + jj*nxim + 0
    #             right_idx = base_im + jj*nxim + (rx+1)
    #             image_ring_to_level0[left_idx] = nbr_l['left']
    #             image_ring_to_level0[right_idx] = nbr_l['right']
    #             coarse_im_set.add(left_idx)
    #             coarse_im_set.add(right_idx)
    #         for ii in range(1,rx+1):
    #             up_idx = base_im + 0*nxim + ii
    #             down_idx = base_im + (ry+1)*nxim + ii
    #             image_ring_to_level0[up_idx] = nbr_l['up']
    #             image_ring_to_level0[down_idx] = nbr_l['down']
    #             coarse_im_set.add(up_idx)
    #             coarse_im_set.add(down_idx)

    #     for cm, cp, t, tt in zip(cmi, cpi, Ti, Ti_therm):
    #         if (cm // nxy_im) != (cp // nxy_im):
    #             continue
    #         cm_is_fine = cm in fine_im_set
    #         cp_is_fine = cp in fine_im_set
    #         cm_is_coarse = cm in coarse_im_set
    #         cp_is_coarse = cp in coarse_im_set

    #         # fine-coarse connection
    #         if cm_is_fine and cp_is_coarse:
    #             fine_global = imag_fine_to_global[cm]
    #             coarse_local = image_ring_to_level0[cp]
    #             fc_cm.append(coarse_local)
    #             fc_cp.append(fine_global)
    #             fc_T.append(t)
    #             fc_Tt.append(tt)

    #         elif cp_is_fine and cm_is_coarse:
    #             fine_global = imag_fine_to_global[cp]
    #             coarse_local = image_ring_to_level0[cm]
    #             fc_cm.append(coarse_local)
    #             fc_cp.append(fine_global)
    #             fc_T.append(t)
    #             fc_Tt.append(tt)


    # step 4: extract overburden and underburden connections from imaginary grid
    cm_burden, cp_burden, T_burden, Tt_burden = [], [], [], []

    for name in lgr_orders:
        lgr = self.lgrs[name]["lgr_coords_in_parent_grid"]
        ic = lgr['i_range'][0]   # 1-based
        jc = lgr['j_range'][0]   # 1-based
        k1  = lgr['k_range'][0]   # 1-based
        k2  = lgr['k_range'][1]   # 1-based
        rx, ry, rz = lgr['refine']
        nk = k2 - k1 + 1

        plane_size = rx * ry
        fine_global_offset = lgr_offsets[name]

        # top
        self.level1_imag_z_top[name].discretize()
        disc_top = self.level1_imag_z_top[name].discretizer
        cmi_top, cpi_top, Ti_top, Ti_top_therm = disc_top.calc_structured_discr()

        over_coarse_local  = int(g2l0[self.convert_ijk_to_gindex_1based(ic, jc, k1-1,  self.level0.nx, self.level0.ny)])
        nx_im_top = self.level1_imag_z_top[name].nx
        ny_im_top = self.level1_imag_z_top[name].ny
        nxy_im_top = nx_im_top * ny_im_top

        for cm, cp, t, tt in zip(cmi_top, cpi_top, Ti_top, Ti_top_therm):

            # vertical filter
            if abs((cm // nxy_im_top) - (cp // nxy_im_top)) != 1:
                continue

            if cm // nxy_im_top ==1:
                fine_local_2d = cm % nxy_im_top
            elif cp // nxy_im_top ==1:
                fine_local_2d = cp % nxy_im_top
            else:
                continue

            top_fine_global = fine_global_offset + fine_local_2d
            # --- overburden ↔ LGR TOP ---
            cm_burden.append(over_coarse_local)
            cp_burden.append(top_fine_global)
            T_burden.append(t)
            Tt_burden.append(tt)

        # bottom
        self.level1_imag_z_bot[name].discretize()
        disc_bot = self.level1_imag_z_bot[name].discretizer
        cmi_bot, cpi_bot, Ti_bot, Ti_bot_therm = disc_bot.calc_structured_discr()
        under_coarse_local = int(g2l0[self.convert_ijk_to_gindex_1based(ic, jc, k2+1, self.level0.nx, self.level0.ny)])

        nx_im_bot = self.level1_imag_z_bot[name].nx
        ny_im_bot = self.level1_imag_z_bot[name].ny
        nxy_im_bot = nx_im_bot * ny_im_bot

        for cm, cp, t, tt in zip(cmi_bot, cpi_bot, Ti_bot, Ti_bot_therm):
            if abs(cm//nxy_im_bot - cp//nxy_im_bot) != 1:
                continue

            # fine layer is layer 0 in bot imag
            if cm // nxy_im_bot == 0:
                fine_local_2d = cm % nxy_im_bot
            elif cp // nxy_im_bot == 0:
                fine_local_2d = cp % nxy_im_bot
            else:
                continue
            bot_fine_global = fine_global_offset + (nk - 1) * plane_size + fine_local_2d

            cm_burden.append(under_coarse_local)
            cp_burden.append(bot_fine_global)
            T_burden.append(t)
            Tt_burden.append(tt)



    # assemble all connections
    ## coarse-coarse connections
    cm_parts = [cm0]
    cp_parts = [cp0]
    T_parts = [T0]
    Tt_parts = [T0_therm]
    ## fine-fine connections
    cm_parts.append(np.asarray(ff_cm, dtype= int))
    cp_parts.append(np.asarray(ff_cp, dtype= int))
    T_parts.append(np.asarray(ff_T, dtype= float))
    Tt_parts.append(np.asarray(ff_Tt, dtype= float))
    ## fine-coarse connections
    cm_parts.append(np.asarray(fc_cm, dtype= int))
    cp_parts.append(np.asarray(fc_cp, dtype= int))
    T_parts.append(np.asarray(fc_T, dtype= float))
    Tt_parts.append(np.asarray(fc_Tt, dtype= float))
    # overburden and underburden connections
    cm_parts.append(np.asarray(cm_burden, dtype= int))
    cp_parts.append(np.asarray(cp_burden, dtype= int))
    T_parts.append(np.asarray(T_burden, dtype= float))
    Tt_parts.append(np.asarray(Tt_burden, dtype= float))


    cm_all = np.concatenate(cm_parts)
    cp_all = np.concatenate(cp_parts)
    T_all = np.concatenate(T_parts)
    T_all_therm = np.concatenate(Tt_parts)

    # only assume all LGRs have the same refine ratio and are centered in the parent grid, then the global index of local center is
    center_2d = (ry//2) * rx + (rx//2)

    print(f'cm is {cm_all}')
    print(f'cp is {cp_all}')

    meta = {
        "lgr_orders": lgr_orders,
        "lgr_offsets": lgr_offsets,
        "n0_act": n0_act,
        "well_local_center": center_2d,
    }
    return cm_all, cp_all, T_all, T_all_therm, meta

class LGRReservoir(ReservoirBase):
    """
    A simple reservoir wrapper that takes pre-assembled
    connection lists, Transmissibility and cell properties
    , then build a new conn_mesh
    """
    def __init__(self, timer, cell_m, cell_p, tran, tran_thermal,
                 poro, rcond, hcap, depth, volume, dx = None, dy = None,
                 dz = None, kx=None,ky = None, kz =None,
                 op_num=None, cache: bool= False):

        super().__init__(timer, cache)
        self.vtk_initialized = False
        self.discretizer = SimpleNamespace(frac_cells_tot=0)
        # connectivity
        self.cell_m = np.asarray(cell_m, dtype= int)
        self.cell_p = np.asarray(cell_p, dtype= int)
        self.tran = np.asarray(tran, dtype= float)
        self.tran_thermal = np.asarray(tran_thermal, dtype= float)

        # properties
        self.poro = np.asarray(poro, dtype=float)
        self.rcond = np.asarray(rcond, dtype=float)
        self.hcap = np.asarray(hcap, dtype=float)
        self.depth = np.asarray(depth, dtype=float)
        self.volume = np.asarray(volume, dtype=float)
        self.n = int(len(self.poro))
        if op_num is None:
            self.op_num = np.zeros_like(self.poro, dtype= int)
        else:
            self.op_num = np.asarray(op_num, dtype= int)


        self.n = self.poro.size
        self.ndims = 3
        self.actnum = np.ones(self.n, dtype= bool)
        self.global_data = {}

        self.dx = np.asarray(dx, dtype=float) if dx is not None else None
        self.dy = np.asarray(dy, dtype=float) if dy is not None else None
        self.dz = np.asarray(dz, dtype=float) if dz is not None else None
        self.kx = np.asarray(kx, dtype=float) if kx is not None else None
        self.ky = np.asarray(ky, dtype=float) if ky is not None else None
        self.kz = np.asarray(kz, dtype=float) if kz is not None else None


    def discretize(self, cache: bool = False, verbose: bool = False) -> conn_mesh:

        """build conn_mesh from pre-assembled data"""
        cm = np.asarray(self.cell_m, dtype=np.int32)
        cp = np.asarray(self.cell_p, dtype=np.int32)
        mesh = conn_mesh()
        mesh.init(
            index_vector(cm),
            index_vector(cp),
            value_vector(self.tran),
            value_vector(self.tran_thermal))

        # Create numpy arrays wrapped around mesh data
        np.array(mesh.poro, copy=False)[:] = self.poro
        np.array(mesh.rock_cond, copy=False)[:] = self.rcond
        np.array(mesh.heat_capacity, copy=False)[:] = self.hcap
        np.array(mesh.depth, copy=False)[:] = self.depth

        np.array(mesh.volume, copy=False)[:] = self.volume
        np.array(mesh.op_num, copy=False)[:] = self.op_num

        self.mesh = mesh

        return mesh

    def cal_well_index_from_cell(self,cell_index: int, well_radius: float = 0.0762,segment_direction='z_axis',skin=0) -> float:
        dx = float(self.dx[cell_index])
        dy = float(self.dy[cell_index])
        dz = float(self.dz[cell_index])
        kx = float(self.kx[cell_index])
        ky = float(self.ky[cell_index])
        kz = float(self.kz[cell_index])
        darcy_constant = 0.0085267146719160104986876640419948
        if segment_direction == 'z_axis':
            if kx * ky != 0:
                peaceman_rad = (
                        0.28
                        * np.sqrt(np.sqrt(ky / kx) * dx**2 + np.sqrt(kx / ky) * dy**2)
                        / ((ky / kx) ** (1 / 4) + (kx / ky) ** (1 / 4))
                    )
                well_index = (
                        2
                        * np.pi
                        * dz
                        * np.sqrt(kx * ky)
                        / (np.log(peaceman_rad / well_radius) + skin)
                    )

                conduction_rad = 0.28 * np.sqrt(dx**2 + dy**2) / 2.0
                well_indexD = (
                    2 * np.pi * dz / (np.log(conduction_rad / well_radius) + skin))
        elif segment_direction == 'x_axis':
            if kz * ky != 0:
                peaceman_rad = (
                        0.28
                        * np.sqrt(np.sqrt(ky / kz) * dz**2 + np.sqrt(kz / ky) * dy**2)
                        / ((ky / kz) ** (1 / 4) + (kz / ky) ** (1 / 4))
                    )
                well_index = (
                        2
                        * np.pi
                        * dx
                        * np.sqrt(kz * ky)
                        / (np.log(peaceman_rad / well_radius) + skin)
                    )

                conduction_rad = 0.28 * np.sqrt(dz**2 + dy**2) / 2.0
                well_indexD = (
                        2 * np.pi * dx / (np.log(conduction_rad / well_radius) + skin)
                    )
        elif segment_direction == 'y_axis':
            if kx * kz != 0:
                peaceman_rad = (
                        0.28
                        * np.sqrt(np.sqrt(kz / kx) * dx**2 + np.sqrt(kx / kz) * dz**2)
                        / ((kz / kx) ** (1 / 4) + (kx / kz) ** (1 / 4))
                    )
                well_index = (
                        2
                        * np.pi
                        * dy
                        * np.sqrt(kx * kz)
                        / (np.log(peaceman_rad / well_radius) + skin)
                    )

                conduction_rad = 0.28 * np.sqrt(dz**2 + dx**2) / 2.0
                well_indexD = (
                        2 * np.pi * dy / (np.log(conduction_rad / well_radius) + skin)
                    )

        well_index = well_index * darcy_constant
        return well_index, well_indexD


    def add_perforation(
        self,
        well_name: str,
        cell_index: int,
        well_seg_idx: int = None,
        well_radius: float = 0.0762,
        well_index: float = None,
        well_indexD: float = 0.0,
        segment_direction: str = 'z_axis',
        skin: float = 0,
        ms_epm: bool = False,
        with_peaceman_for_coupled_well_reservoir: bool = False,
        verbose: bool = False,
    ):
        well = self.get_well(well_name)

        if well.ms_type == ms_well.MS_Type.EPM:
            assert well_seg_idx is None, (
                "If the well is of the EPM type, well_seg_idx must not be specified!"
            )
            wi, wid = self.cal_well_index_from_cell(
                cell_index, well_radius, segment_direction, skin
            )
        elif well.ms_type == ms_well.MS_Type.DFM:
            raise NotImplementedError(
                "DFM type wells are not supported in this method. "
                "Only EPM multi-segment wells are supported."
            )

        if well_index is None:
            well_index = wi

        if well_indexD is None:
            well_indexD = wid

        if well.ms_type == ms_well.MS_Type.EPM:
            if ms_epm:
                well_block = len(well.perforations)
            else:
                well_block = 0
        elif well.ms_type == ms_well.MS_Type.DFM:
            well_block = well_seg_idx - 2

        if well.ms_type == ms_well.MS_Type.EPM:
            if len(well.perforations) == 0:  # if adding the first perforation
                well.well_head_depth = np.array(self.mesh.depth, copy=False)[
                    cell_index
                ]
                well.well_body_depth = well.well_head_depth

                # well.segment_depth_increment = self.discretizer.len_cell_zdir[
                #     i - 1, j - 1, k - 1
                # ]
                # dz is constant in the current implementation, so I directly use dz here
                well.segment_depth_increment = self.dz[cell_index]

                well.segment_volume *= well.segment_depth_increment
            else:  # update well depth
                well.well_head_depth = min(
                    well.well_head_depth,
                    np.array(self.mesh.depth, copy=False)[cell_index],
                )
                well.well_body_depth = well.well_head_depth


        for p in well.perforations:
            if p[0] == well_block and p[1] == cell_index:
                print(
                    f'Neglected duplicate perforation for well {well.name} to block {cell_index}'
                )
                return

        well.perforations = well.perforations + [
            (well_block, cell_index, well_index, well_indexD)
        ]

        if verbose:
            print(
                f'Added perforation for well {well.name} to block {cell_index} '
                f'with WI={well_index}, WID={well_indexD}'
            )

        assert well_index >= 0
        assert well_indexD >= 0

        return

    def init_vtk(self, output_directory: str, export_grid_data: bool = True):
        """
        Initialize VTK output for LGR reservoir.

        The LGR reservoir is exported as an unstructured hexahedral VTU mesh,
        because the final grid is not a single conforming structured tensor grid.
        """
        os.makedirs(output_directory, exist_ok=True)

        self.vtk_initialized = True
        self.vtk_filenames_and_times = {}

        points, cells = self._build_lgr_vtk_geometry()
        self._vtk_points = points
        self._vtk_cells = cells

        if export_grid_data:
            static_data = self._build_lgr_static_cell_data()

            mesh = meshio.Mesh(
                points=points,
                cells=[("hexahedron", cells)],
                cell_data={key: [value] for key, value in static_data.items()},
            )

            meshio.write(os.path.join(output_directory, "mesh.vtu"), mesh)
            print(f"Writing LGR mesh data to {os.path.join(output_directory, 'mesh.vtu')}")


    def _build_lgr_vtk_geometry(self):
        """
        Build an unstructured hexahedral mesh from cell centers and cell dimensions.

        Returns
        -------
        points : ndarray, shape (8*n_cells, 3)
            VTK point coordinates.
        cells : ndarray, shape (n_cells, 8)
            Hexahedron connectivity.
        """
        x = np.asarray(self.cell_center_x, dtype=float)
        y = np.asarray(self.cell_center_y, dtype=float)
        z = np.asarray(self.cell_center_z, dtype=float)

        dx = np.asarray(self.dx, dtype=float)
        dy = np.asarray(self.dy, dtype=float)
        dz = np.asarray(self.dz, dtype=float)

        n = int(self.mesh.n_res_blocks)

        if not (len(x) >= n and len(y) >= n and len(z) >= n):
            raise ValueError(
                f"LGR VTK geometry error: center arrays are shorter than n_res_blocks={n}."
            )

        if not (len(dx) >= n and len(dy) >= n and len(dz) >= n):
            raise ValueError(
                f"LGR VTK geometry error: dx/dy/dz arrays are shorter than n_res_blocks={n}."
            )

        points = np.empty((8 * n, 3), dtype=float)
        cells = np.empty((n, 8), dtype=np.int64)

        for c in range(n):
            xm = x[c] - 0.5 * dx[c]
            xp = x[c] + 0.5 * dx[c]
            ym = y[c] - 0.5 * dy[c]
            yp = y[c] + 0.5 * dy[c]
            zm = z[c] - 0.5 * dz[c]
            zp = z[c] + 0.5 * dz[c]

            base = 8 * c

            # Hexahedron point order:
            # bottom face: 0-1-2-3
            # top face:    4-5-6-7
            points[base + 0] = [xm, ym, zm]
            points[base + 1] = [xp, ym, zm]
            points[base + 2] = [xp, yp, zm]
            points[base + 3] = [xm, yp, zm]
            points[base + 4] = [xm, ym, zp]
            points[base + 5] = [xp, ym, zp]
            points[base + 6] = [xp, yp, zp]
            points[base + 7] = [xm, yp, zp]

            cells[c] = np.arange(base, base + 8, dtype=np.int64)

        return points, cells


    def _build_lgr_static_cell_data(self):
        """
        Static cell data written to mesh.vtu.

        This is mainly for debugging geometry and checking whether coarse/LGR cells
        are placed correctly in ParaView.
        """
        n = int(self.mesh.n_res_blocks)

        static_data = {
            "cell_id": np.arange(n, dtype=np.int32),
            "center_x": np.asarray(self.cell_center_x, dtype=float)[:n],
            "center_y": np.asarray(self.cell_center_y, dtype=float)[:n],
            "center_z": np.asarray(self.cell_center_z, dtype=float)[:n],
            "dx": np.asarray(self.dx, dtype=float)[:n],
            "dy": np.asarray(self.dy, dtype=float)[:n],
            "dz": np.asarray(self.dz, dtype=float)[:n],
            "poro": np.asarray(self.mesh.poro, dtype=float)[:n],
            "volume": np.asarray(self.mesh.volume, dtype=float)[:n],
            "depth": np.asarray(self.mesh.depth, dtype=float)[:n],
            "op_num": np.asarray(self.mesh.op_num, dtype=np.int32)[:n],
        }

        # Optional properties if your LGRReservoir has them.
        optional_names = {
            "kx": "permx",
            "ky": "permy",
            "kz": "permz",
            "rcond": "rcond",
            "hcap": "hcap",
        }

        for attr_name, vtk_name in optional_names.items():
            if hasattr(self, attr_name):
                arr = np.asarray(getattr(self, attr_name), dtype=float)
                if len(arr) >= n:
                    static_data[vtk_name] = arr[:n]

        # Useful visual classification: coarse vs refined cells.
        # In your current LGR construction, fine cells usually have dx/dy smaller than max dx/dy.
        dx = static_data["dx"]
        dy = static_data["dy"]
        max_dx = np.nanmax(dx)
        max_dy = np.nanmax(dy)
        static_data["is_lgr_cell"] = (
            (~np.isclose(dx, max_dx)) | (~np.isclose(dy, max_dy))
        ).astype(np.int32)

        return static_data


    def output_to_vtk(
        self,
        ith_step: int,
        t: float,
        output_directory: str,
        prop_names: dict,
        data,
    ):
        """
        Export LGR simulation results to VTU.

        This method is called by DARTS Output.output_to_vtk().
        """
        os.makedirs(output_directory, exist_ok=True)

        if not getattr(self, "vtk_initialized", False):
            self.init_vtk(output_directory, export_grid_data=True)

        n = int(self.mesh.n_res_blocks)

        cell_data = {}
        for i, prop in enumerate(prop_names):
            arr = np.asarray(data[i], dtype=float)

            if arr.ndim != 1:
                arr = arr.reshape(-1)

            if len(arr) < n:
                raise ValueError(
                    f"VTK property '{prop}' has length {len(arr)}, "
                    f"but n_res_blocks={n}."
                )

            cell_data[prop_names[prop]] = arr[:n]
        cell_data["cell_id"] = np.arange(n, dtype=np.int32)
        cell_data["depth"] = np.asarray(self.mesh.depth, dtype=float)[:n]
        cell_data["poro"] = np.asarray(self.mesh.poro, dtype=float)[:n]
        cell_data["volume"] = np.asarray(self.mesh.volume, dtype=float)[:n]
        cell_data["dx"] = np.asarray(self.dx, dtype=float)[:n]
        cell_data["dy"] = np.asarray(self.dy, dtype=float)[:n]
        cell_data["dz"] = np.asarray(self.dz, dtype=float)[:n]

        if hasattr(self, "kx"):
            cell_data["permx"] = np.asarray(self.kx, dtype=float)[:n]
        if hasattr(self, "ky"):
            cell_data["permy"] = np.asarray(self.ky, dtype=float)[:n]
        if hasattr(self, "kz"):
            cell_data["permz"] = np.asarray(self.kz, dtype=float)[:n]
        if hasattr(self, "rcond"):
            cell_data["rcond"] = np.asarray(self.rcond, dtype=float)[:n]
        if hasattr(self, "hcap"):
            cell_data["hcap"] = np.asarray(self.hcap, dtype=float)[:n]

        # Add useful diagnostics at every timestep
        cell_data["cell_id"] = np.arange(n, dtype=np.int32)
        if hasattr(self, "dx") and hasattr(self, "dy"):
            dx = np.asarray(self.dx, dtype=float)[:n]
            dy = np.asarray(self.dy, dtype=float)[:n]
            cell_data["is_lgr_cell"] = (
                (~np.isclose(dx, np.nanmax(dx))) | (~np.isclose(dy, np.nanmax(dy)))
            ).astype(np.int32)

        vtk_file_name = os.path.join(output_directory, f"solution_ts{ith_step}.vtu")

        mesh = meshio.Mesh(
            points=self._vtk_points,
            cells=[("hexahedron", self._vtk_cells)],
            cell_data={key: [value] for key, value in cell_data.items()},
        )

        print(f"Writing LGR VTK file for reporting step {ith_step}: {vtk_file_name}")
        meshio.write(vtk_file_name, mesh)

        self.vtk_filenames_and_times[vtk_file_name] = float(t)

        vtk_group = VtkGroup(os.path.join(output_directory, "solution"))
        for fname, time in self.vtk_filenames_and_times.items():
            vtk_group.addFile(fname, time)
        vtk_group.save()
