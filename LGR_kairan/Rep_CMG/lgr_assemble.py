import numpy as np

from darts.reservoirs.reservoir_base import ReservoirBase
from darts.engines import(
    conn_mesh,
    index_vector,
    value_vector,
    ms_well,
)

def assemble_lgr_connections_eclipse(self):
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


    # step 2 discretize each fine LGR and build global offsets
    lgr_orders = list(self.lgrs.keys())
    lgr_offsets = {} # name -> start global index in assembled system

    fine_conn_parts = []
    fine_T_parts = []
    fine_Tt_parts = []

    offset = n0_act
    for name in lgr_orders:
        self.level1[name].discretize()
        disc1 = self.level1[name].discretizer
        cm1, cp1, T1, T1_therm = disc1.calc_structured_discr()
        lgr_offsets[name] = offset
        fine_conn_parts.append((cm1 + offset, cp1 + offset))
        fine_T_parts.append(T1)
        fine_Tt_parts.append(T1_therm)
        offset += self.level1[name].n

    # step3 fine-coarse connections via imaginary grids
    fc_cm = []
    fc_cp = []
    fc_T = []
    fc_Tt = []

    for name in lgr_orders:
        cfg = self.lgrs[name]
        ic = cfg['lgr_coords_in_parent_grid']['i_range'][0]
        jc = cfg['lgr_coords_in_parent_grid']['j_range'][0]
        k1, k2 = cfg['lgr_coords_in_parent_grid']['k_range']
        rx,ry,rz = cfg['lgr_coords_in_parent_grid']['refine']
        
        self.level1_imag[name].discretize()
        disc_im = self.level1_imag[name].discretizer
        cmi, cpi, Ti, Ti_therm = disc_im.calc_structured_discr()

        nxim = rx +2
        nyim = ry +2

        # identify fine and coarse cells in imaginary grid for per layer connections
        fine_im = []
        for j in range(1,ry+1):
            for i in range(1,rx+1):
                fine_im.append(j*nxim + i)
        fine_im_set = set(fine_im)

        left_im = [jj * nxim for jj in range(1,ry+1)]
        right_im = [rx +1 + jj*nxim for jj in range(1,ry+1)]
        up_im = [ii for ii in range(1,rx+1)]
        down_im = [ (ry +1)*nxim + ii for ii in range(1,rx+1)]
        coarse_im_set = set(left_im + right_im + up_im + down_im)

        # mapping: imaginary fine index (5*5) -> local fine index (3*3)
        imag_fine_to_local = {}
        k = 0
        for j in range(1,ry+1):
            for i in range(1,rx+1):
                imag_idx = j*nxim + i # index of 5*5 grid
                imag_fine_to_local[imag_idx] = k # index of 3*3 grid
                k += 1
        # filter connections that are fine-coarse
        is_fine_coarse = np.array([(cm in fine_im_set and cp in coarse_im_set)
                                    or (cp in fine_im_set and cm in coarse_im_set)
                                    for cm, cp in zip(cmi, cpi)], dtype= bool)
        # boundary connection (fine-coarse)
        cmi_fc = cmi[is_fine_coarse]
        cpi_fc = cpi[is_fine_coarse]
        Ti_fc = Ti[is_fine_coarse]
        Ti_therm_fc = Ti_therm[is_fine_coarse]

        # For each k layer in the column, 
        # map halo to the corresponding coarse neighbor at that k
        nx0 = self.level0.nx
        ny0 = self.level0.ny
        fine_global_offset = lgr_offsets[name]
        # compute 4 neighboring coarse cell global indices (including inactive cells) 
        # at each k layer at level0
        for kk_local, k_layer in enumerate(range(k1, k2 +1)):
            nbr_g = {
                "left" :self.convert_ijk_to_gindex_1based(ic-1, jc, k_layer, nx0, ny0),
                "right" :self.convert_ijk_to_gindex_1based(ic+1, jc, k_layer, nx0, ny0),
                "up" :self.convert_ijk_to_gindex_1based(ic, jc-1, k_layer, nx0, ny0),
                "down" :self.convert_ijk_to_gindex_1based(ic, jc+1, k_layer, nx0, ny0),
                
            }


            nbr_l = {k: int(g2l0[v]) for k, v in nbr_g.items()} # local indices in level0 after actnum squeeze

            # map imag coarse ring to lvel0 local index
            image_ring_to_level0 = {}
            for idx in left_im:
                image_ring_to_level0[idx] = nbr_l['left']
            for idx in right_im:
                image_ring_to_level0[idx] = nbr_l['right']
            for idx in up_im:
                image_ring_to_level0[idx] = nbr_l['up']
            for idx in down_im:
                image_ring_to_level0[idx] = nbr_l['down']

            # build FC connnections for this k layer
            plane_size = rx * ry
            for cm, cp, t, tt in zip(cmi_fc,cpi_fc, Ti_fc, Ti_therm_fc):
                # imag_fine_to_local is a dict: imag index -> local fine index (3*3) of level1
                # image_ring_to_level0 is a dict: imag index -> local coarse index of level0 (after actnum squeeze)
                if cm in imag_fine_to_local and cp in image_ring_to_level0:
                    fine_2d = imag_fine_to_local[cm]
                    coarse_local = image_ring_to_level0[cp]
                elif cp in imag_fine_to_local and cm in image_ring_to_level0:
                    fine_2d = imag_fine_to_local[cp]
                    coarse_local = image_ring_to_level0[cm]
                else:
                    continue

                fine_3d = fine_2d + plane_size * kk_local
                fine_global = fine_3d + fine_global_offset # global index of fine cell
                fc_cm.append(coarse_local)
                fc_cp.append(fine_global)
                fc_T.append(t)
                fc_Tt.append(tt)

    # add new z direction connections between overburden and underburden
    if self.cfg.get("burden", None) is not None:
    
        cm_burden, cp_burden, T_burden, Tt_burden = [], [], [], []

        for name in lgr_orders:
            self.level1_imag_z[name].discretize()
            disc_im = self.level1_imag_z[name].discretizer
            cmi, cpi, Ti, Ti_therm = disc_im.calc_structured_discr()

            lgr = self.lgrs[name]["lgr_coords_in_parent_grid"]
            ic = lgr['i_range'][0]   # 1-based
            jc = lgr['j_range'][0]   # 1-based
            k1  = lgr['k_range'][0]   # 1-based
            k2  = lgr['k_range'][1]   # 1-based

            k_over  = k1 - 1
            k_under = k2 + 1
            # the local index of the burden coarse cell in level0 (this local means of the level0 system after squeezing out inactive cells)
            over_coarse_local  = int(g2l0[self.convert_ijk_to_gindex_1based(ic, jc, k_over,  self.level0.nx, self.level0.ny)])
            under_coarse_local = int(g2l0[self.convert_ijk_to_gindex_1based(ic, jc, k_under, self.level0.nx, self.level0.ny)])

            nx_im = self.level1_imag_z[name].nx
            ny_im = self.level1_imag_z[name].ny
            nxy_im = nx_im * ny_im

            # build sets/maps for imag grid
            fine_set = set()
            coarse_set = set()
            top_map = {}
            bot_map = {}

            for j in range(ny_im):
                for i in range(nx_im):
                    fine_local = self.ijk0_to_lin(i, j, 1, nx_im, ny_im)   # k=1 fine layer in imag
                    coarse_local = self.ijk0_to_lin(i, j, 0, nx_im, ny_im) # k=0 coarse/burden layer in imag
                    fine_set.add(fine_local)
                    coarse_set.add(coarse_local)

                    # map imag fine-local (k=1) -> real LGR fine global (top/bottom layer)
                    fine2d = j * nx_im + i
                    top_fine_global = fine2d + lgr_offsets[name] 
                    bot_fine_global = top_fine_global + (self.level1[name].n - self.level1[name].nx * self.level1[name].ny)
                    top_map[fine_local] = top_fine_global
                    bot_map[fine_local] = bot_fine_global

            
            def k_of(local_id: int) -> int:
                return local_id // nxy_im

            for cm, cp, t, tt in zip(cmi, cpi, Ti, Ti_therm):

                # vertical filter
                if abs(k_of(cm) - k_of(cp)) != 1:
                    continue

                if cm in fine_set and cp in coarse_set:
                    fine_local = cm
                elif cp in fine_set and cm in coarse_set:
                    fine_local = cp
                else:
                    continue

                # --- overburden ↔ LGR TOP ---
                cm_burden.append(over_coarse_local)
                cp_burden.append(top_map[fine_local])
                T_burden.append(t)
                Tt_burden.append(tt)

                # --- underburden ↔ LGR BOTTOM ---
                cm_burden.append(under_coarse_local)
                cp_burden.append(bot_map[fine_local])
                T_burden.append(t)
                Tt_burden.append(tt)


    # assemble all connections
    ## coarse-coarse connections
    cm_parts = [cm0]
    cp_parts = [cp0]
    T_parts = [T0]
    Tt_parts = [T0_therm]
    ## fine-fine connections
    for (cmg, cpg), t, tt in zip(fine_conn_parts, fine_T_parts, fine_Tt_parts):
        cm_parts.append(cmg)
        cp_parts.append(cpg)
        T_parts.append(t)
        Tt_parts.append(tt)
    ## fine-coarse connections
    cm_parts.append(np.asarray(fc_cm, dtype= int))
    cp_parts.append(np.asarray(fc_cp, dtype= int))
    T_parts.append(np.asarray(fc_T, dtype= float))
    Tt_parts.append(np.asarray(fc_Tt, dtype= float))
    # overburden and underburden connections
    if self.cfg.get("burden", None) is not None:
        cm_parts.append(np.asarray(cm_burden, dtype= int))
        cp_parts.append(np.asarray(cp_burden, dtype= int))
        T_parts.append(np.asarray(T_burden, dtype= float))
        Tt_parts.append(np.asarray(Tt_burden, dtype= float))


    cm_all = np.concatenate(cm_parts)
    cp_all = np.concatenate(cp_parts)
    T_all = np.concatenate(T_parts)
    T_all_therm = np.concatenate(Tt_parts)

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

    # The issue with this section lies in the fact that once the LGR is located，
    # the global index of the cell containing the well
    # currently requires the user to manually compute both the global index and
    # cell index before passing them to the well block.
    # def add_perforation(
    #     self,
    #     well_name: str,
    #     cell_index: int,
    #     well_radius: float = 0.0762,
    #     well_index: float = None,
    #     well_indexD: float = 0.0,
    #     segment_direction: str = 'z_axis',
    #     skin: float = 0,
    #     multi_segment: bool = False,
    #     verbose: bool = False,
    # ):
    #     well = self.get_well(well_name)
    #     wi, wid = self.cal_well_index_from_cell(
    #         cell_index, well_radius, segment_direction, skin
    #     )

    #     if well_index is None:
    #         well_index = wi

    #     if well_indexD is None:
    #         well_indexD = wid

    #     if multi_segment:
    #         well_block = len(well.perforations)
    #     else:
    #         well_block = 0

    #     if len(well.perforations) == 0:
    #         well.well_head_depth = np.array(self.mesh.depth, copy=False)[cell_index]
    #         well.well_body_depth = well.well_head_depth
    #         # I don't consider CPG type
    #     else:
    #         well.well_head_depth = min(
    #             well.well_head_depth,
    #             np.array(self.mesh.depth, copy=False)[cell_index],
    #         )
    #         well.well_body_depth = well.well_body_depth

    #     for p in well.perforations:
    #         if p[0] == well_block and p[1] == cell_index:
    #             raise ValueError(
    #                 f'Perforation for well {well.name} in block {cell_index} already exists.'
    #             )
    #             return

    #     well.perforations = well.perforations + [
    #         (well_block, cell_index, well_index, well_indexD)
    #     ]

    #     if verbose:
    #         print(
    #             f'Added perforation for well {well.name} to block {cell_index} '
    #             f'with WI={well_index}, WID={well_indexD}'
    #         )

    #     assert well_index >= 0
    #     assert well_indexD >= 0

    #     return
    
    # Try to set multi-segment well controls
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