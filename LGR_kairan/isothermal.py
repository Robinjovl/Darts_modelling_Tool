from darts.reservoirs.struct_reservoir import StructReservoir
from darts.models.cicd_model import DartsModel
from darts.engines import sim_params, well_control_iface
from darts.engines import (
    conn_mesh,
    index_vector,
    ms_well,
    ms_well_vector,
    timer_node,
    value_vector,
)
import numpy as np

from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer

from darts.physics.properties.flash import ConstantK
from darts.physics.properties.basic import ConstFunc, PhaseRelPerm
from darts.physics.properties.density import DensityBasic, Garcia2001
from darts.reservoirs.reservoir_base import ReservoirBase
from darts.physics.properties.viscosity import Fenghour1998, Islam2012  
from darts.physics.properties.eos_properties import EoSDensity, EoSEnthalpy

from dartsflash.libflash import NegativeFlash
from dartsflash.libflash import CubicEoS, AQEoS, FlashParams, InitialGuess
from dartsflash.components import CompData


class Model(DartsModel):
    def __init__(self):
        # Call base class constructor
        super().__init__()

        # Measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        self.set_reservoir()
        self.zero = 1e-8
        self.set_physics()

        self.set_sim_params(first_ts=0.01, mult_ts=2, max_ts=10, runtime=1000, tol_newton=1e-3, tol_linear=1e-3,
                            it_newton=10, it_linear=50)

        self.timer.node["initialization"].stop()

    def build_lgr_definition(self):
        """
        -parent name
        -i,j,k range in parent grid (1-based)
        -local name
        """
        lgrs = {}
        # Define lgr0
        parent_grid_name = 'global'
        lgr_coords_in_parent_grid = {
        "i_range": [20, 20],
        "j_range": [40, 40],
        "k_range": [1, 40],
        "refine": [3, 3, 1],
        "tag" : "inj"
        }

        lgrs['lgr0'] = {
            'parent_grid_name': parent_grid_name,
            'lgr_coords_in_parent_grid': lgr_coords_in_parent_grid,
        }
        # Define lgr1
        parent_grid_name = 'global'
        lgr_coords_in_parent_grid = {
        "i_range": [60, 60],
        "j_range": [40, 40],
        "k_range": [1, 40],
        "refine": [3, 3, 1],
        "tag" : "prd"
        }
        lgrs['lgr1'] = {
            'parent_grid_name': parent_grid_name,
            'lgr_coords_in_parent_grid': lgr_coords_in_parent_grid,
        }
        return lgrs
    
    def build_well_completion(self):
        return {
            "I1": {"lgr": "lgr0", "k_from": 1, "k_to": 40},  # coarse k (1-based)
            "P1": {"lgr": "lgr1", "k_from": 1, "k_to": 40},
        }    

    def convert_ijk_to_gindex_1based(self, i_1based: int, j_1based: int, k_1based:int, nx:int, ny:int) -> int:
        """
        Convert (i,j,k) in 1-based to global index in 0-based
        """
        g_index_0based = (k_1based - 1) * nx * ny + (j_1based - 1) * nx + (i_1based - 1)
        return g_index_0based
    
    def ijk0_to_lin(self, i0: int, j0: int, k0: int, nx: int, ny: int) -> int:
        """Convert 0-based (i,j,k) to linear index"""
        return k0 * nx * ny + j0 * nx + i0

    def create_actnum_with_lgr(self, nx0, ny0, nz0, refined_ij_list):
        actnum0 = np.ones(nx0 * ny0 * nz0, dtype=np.int32)
        for (i_c,j_c,k_c) in refined_ij_list:
            g = self.convert_ijk_to_gindex_1based(i_c, j_c, k_c, nx0, ny0)
            actnum0[g] = 0  # deactivate the coarse cell that will be refined
        return actnum0

    def auto_image_grid_dx_dy(self, dx_parent, dy_parent, rx, ry):
       
        dx_image = np.r_[dx_parent, np.full(rx, dx_parent/rx), dx_parent]
        dy_image = np.r_[dy_parent, np.full(ry, dy_parent/ry), dy_parent]

        return dx_image, dy_image
    
    def build_dz (self, dz0:float, total_thickness: float, ratio: float = 2.0):
        layers = []
        s = 0
        dz = float(dz0)
        while dz + s < total_thickness:
            layers.append(dz)
            s += dz
            dz*= ratio
        layers.append(total_thickness - s) # add last layer
        return np.asarray(layers, dtype=float)
    
    # build cell center coordinates for visualization
    def build_cell_center(self):
        if not hasattr(self, 'reservoir') or self.reservoir is None:
            raise RuntimeError("Reservoir not built yet.")
        n = self.reservoir.n
        x = np.empty(n, dtype=float)
        y = np.empty(n, dtype=float)
        z = np.asarray(self.reservoir.depth, dtype = float).copy()
        # level 0 
        self.level0.discretize()
        disc0 = self.level0.discretizer
        l2g0 = np.asarray(disc0.local_to_global, dtype=int)
        
        n0 = len(l2g0)
        nx0= int(self.level0.nx)
        ny0= int(self.level0.ny)
        nz0= int(self.level0.nz)
        dx0 =  float(np.asarray(self.level0.global_data["dx"]).flat[0])
        dy0 = float(np.asarray(self.level0.global_data["dy"]).flat[0])
        for local_id, global_id in enumerate(l2g0):
            k = global_id // (nx0 * ny0) # 0 based
            j = (global_id % (nx0 * ny0)) // nx0
            i = global_id % nx0
            x[local_id] = (i + 0.5) * dx0
            y[local_id] = (j + 0.5) * dy0
            
        # lgr blocks
        meta = self.lgr_meta
        for name in meta['lgr_orders']:
            offset = meta['lgr_offsets'][name]
            grid = self.level1[name]
            nx1, ny1, nz1 = int(grid.nx), int(grid.ny), int(grid.nz)
            cfg = self.lgrs[name]['lgr_coords_in_parent_grid']
            i_start, j_start = cfg['i_range'][0], cfg['j_range'][0] # 1-based
            rx, ry, rz = cfg['refine']
            x0 = (i_start - 1) * dx0 # left corner of lgr block in global coordinate
            y0 = (j_start - 1) * dy0
            dx1 = dx0 / rx
            dy1 = dy0 / ry
            n_lgr = int(grid.n)
            nxy1 = nx1 * ny1
            for lid in range(n_lgr): # local id in lgr block
                k = lid // nxy1
                r = lid - k*nxy1
                j = r//nx1
                i = r % nx1

                glid = offset + lid
                x[glid] = x0 + (i+0.5) * dx1
                y[glid] = y0 + (j+0.5) * dy1

        self.reservoir.cell_center_x = x
        self.reservoir.cell_center_y = y
        self.reservoir.cell_center_z = z
        return x,y,z
                
    
    def set_reservoir(self):
        self.lgrs = self.build_lgr_definition()

        #Build Level0 with actnum + overburden and underburden
        nx0, ny0 = 80, 80 # global grid size
        dx0, dy0 = 100, 100
        nz_res = 40
        dz_res = 5

        over_thickness = 2000
        under_thickness = 2000
        dz_over = self.build_dz(dz0=dz_res, total_thickness=over_thickness)
        dz_over = dz_over[::-1]  # reverse for overburden

        dz_under = self.build_dz(dz0=dz_res,total_thickness=under_thickness)
        nz_over = len(dz_over)
        nz_under = len(dz_under)
        nz0 = nz_over + nz_res + nz_under
        dz0_layers = np.concatenate([dz_over, np.full(nz_res, dz_res, dtype=float), dz_under])
        permx0, permy0, permz0 = 50, 50, 50
        poro0 = 0.1
        poro_burden = 0.01
        perm_burden = 0.001

        # thermal properties
        rcond_res = 181.44 # KJ/m/day/k
        hcap_res = 2650 # kJ/m3/K assume reservoir density here
        rcond_over, rcond_under = 149.54, 149.54
        hcap_over, hcap_under = 2347.29, 2347.29

        reservoir_top = 2000
        
       # update lgr k range after adding overburden layer
        for lname, _cfg in self.lgrs.items():
             _cfg['lgr_coords_in_parent_grid']['k_range'] = [nz_over + 1, nz_over + nz_res]


        refined_cells_ijk = []
        for name, cfg in self.lgrs.items():
            i1, i2 = cfg['lgr_coords_in_parent_grid']['i_range']
            j1, j2 = cfg['lgr_coords_in_parent_grid']['j_range']
            k1, k2 = cfg['lgr_coords_in_parent_grid']['k_range']
            assert i1 == i2 and j1 == j2, "This script only assume column LGR"  
            for k in range(k1, k2 + 1):
                refined_cells_ijk.append( (i1, j1, k) )  

        self.refined_cells_ijk = refined_cells_ijk

        actnum0 = self.create_actnum_with_lgr(nx0, ny0, nz0, refined_cells_ijk)


        self.level0 = StructReservoir(self.timer, nx=nx0, ny=ny0, nz=nz0, dx=dx0, dy=dy0, dz=dz0_layers,
                                      permx=permx0, permy=permy0, permz=permz0, poro=poro0,depth= None, 
                                      start_z=0,actnum=actnum0, rcond=rcond_res, hcap=hcap_res,)

        boundary_factor = 1e6
        base_vol = float(dx0 * dy0 * dz_res)
        v_big = base_vol * boundary_factor
        self.level0.boundary_volumes = {
            "xy_minus": None,
            "xy_plus": None,
            "yz_minus": v_big,
            "yz_plus": v_big,
            "xz_minus": v_big,
            "xz_plus": v_big,
        }

        # Build Level1 grids and imaginary grids
        self.level1 = {}
        self.level1_imag = {}
        self.level1_imag_z = {}
        
        for name, cfg in self.lgrs.items():
            rx, ry, rz = cfg['lgr_coords_in_parent_grid']['refine']
            k1, k2 = cfg['lgr_coords_in_parent_grid']['k_range']
            nk = k2 - k1 + 1

            nx1,ny1,nz1 = rx, ry, nk
            dx1 = dx0 / rx
            dy1 = dy0 / ry
            dz1 = dz_res

            self.level1[name] = StructReservoir(self.timer, nx=nx1, ny=ny1, nz=nz1, dx=dx1, dy=dy1, dz=dz1,
                                        permx=permx0, permy=permy0, permz=permz0, poro=poro0, depth= None, start_z=2000, rcond=rcond_res, hcap=hcap_res)

            dx_imag, dy_imag = self.auto_image_grid_dx_dy(dx0, dy0, rx, ry)

            # 2D imaginary grid per layer
            self.level1_imag[name] = StructReservoir(self.timer, nx=rx+2, ny=ry+2, nz=1, dx=dx_imag, dy=dy_imag, dz=dz_res,
                                        permx=permx0, permy=permy0, permz=permz0, poro=poro0, depth= None, start_z=2000,rcond=rcond_res, hcap=hcap_res,)

            #create a new imaginary grid for overburden and underburden connections
            permz_x = np.full((rx, ry, 2), permx0, dtype=float) 
            permz_x [:,:,0] = perm_burden
            permz_y = np.full((rx, ry, 2), permy0, dtype=float)
            permz_y [:,:,0] = perm_burden
            permz_z = np.full((rx, ry, 2), permz0, dtype=float)
            permz_z [:,:,0] = perm_burden
        

            self.level1_imag_z[name] = StructReservoir(self.timer, nx=rx, ny=ry,nz=2, dx =dx1, dy=dy1, dz= dz1,
                                                  permx=permz_x, permy=permz_y, permz=permz_z, poro=poro0, depth= None, start_z=2000,rcond=rcond_res, hcap=hcap_res,)
            
        cm_all, cp_all, T_all, T_all_therm, meta = self.assemble_lgr_connections_eclipse()

       # assemble properties
        self.level0.discretize()
        disc0 = self.level0.discretizer
        l2g0 = disc0.local_to_global          # l2g0 has already eliminated inactive cells

        dx0_arr = disc0.convert_to_flat_array(self.level0.global_data['dx'], 'dx')[l2g0]
        dy0_arr = disc0.convert_to_flat_array(self.level0.global_data['dy'], 'dy')[l2g0]
        dz0_arr = disc0.convert_to_flat_array(self.level0.global_data['dz'], 'dz')[l2g0]
        
        depth0_arr = disc0.convert_to_flat_array(self.level0.global_data['depth'], 'depth')[l2g0]
        volume0_arr = np.array(self.level0.volume, copy=False).astype(float)
        # thermal properties for level 0 varying with ob and res
        k_index0 = np.arange(self.level0.n, dtype=np.int32) // (nx0 * ny0)

        rcon0_full = np.empty(self.level0.n, dtype=float)
        hcap0_full = np.empty(self.level0.n, dtype=float)
        poro0_full = np.full(self.level0.n, poro_burden, dtype=float)
        kx0_full = np.full(self.level0.n, perm_burden, dtype=float)
        ky0_full = np.full(self.level0.n, perm_burden, dtype=float)
        kz0_full = np.full(self.level0.n, perm_burden, dtype=float)
  
        mask_over = k_index0 < nz_over
        mask_res  = (k_index0 >= nz_over) & (k_index0 < nz_over + nz_res)
        mask_under = k_index0 >= (nz_over + nz_res)

        rcon0_full[mask_over] = rcond_over
        rcon0_full[mask_res] = rcond_res
        rcon0_full[mask_under] = rcond_under
        hcap0_full[mask_over] = hcap_over
        hcap0_full[mask_res] = hcap_res
        hcap0_full[mask_under] =hcap_under
        poro0_full[mask_res] = poro0
        kx0_full [mask_res] = permx0
        ky0_full [mask_res] = permy0
        kz0_full [mask_res] = permz0

        rcon0_arr = rcon0_full[l2g0]
        hcap0_arr = hcap0_full[l2g0]
        kx0_arr = kx0_full[l2g0]
        ky0_arr = ky0_full[l2g0]
        kz0_arr = kz0_full[l2g0]
        poro0_arr = poro0_full[l2g0]
        

        dx_list = [dx0_arr]; dy_list = [dy0_arr]; dz_list = [dz0_arr]
        kx_list = [kx0_arr]; ky_list = [ky0_arr]; kz_list = [kz0_arr]
        poro_list = [poro0_arr]; depth_list = [depth0_arr]; volume_list = [volume0_arr]
        rcond_list = [rcon0_arr]; hcap_list = [hcap0_arr]

        for name in meta['lgr_orders']:
            rx, ry, rz = self.lgrs[name]['lgr_coords_in_parent_grid']['refine']
            self.level1[name].discretize()
            disc1 = self.level1[name].discretizer

            dx_list.append(disc1.convert_to_flat_array(self.level1[name].global_data["dx"], "dx"))
            dy_list.append(disc1.convert_to_flat_array(self.level1[name].global_data["dy"], "dy"))
            dz_list.append(disc1.convert_to_flat_array(self.level1[name].global_data["dz"], "dz"))
            kx_list.append(disc1.convert_to_flat_array(self.level1[name].global_data["permx"], "permx"))
            ky_list.append(disc1.convert_to_flat_array(self.level1[name].global_data["permy"], "permy"))
            kz_list.append(disc1.convert_to_flat_array(self.level1[name].global_data["permz"], "permz"))

            poro_list.append(disc1.convert_to_flat_array(self.level1[name].global_data["poro"], "poro"))
            depth_list.append(disc1.convert_to_flat_array(self.level1[name].global_data["depth"], "depth"))
            volume_list.append(np.ones(self.level1[name].n, dtype=float) * (dx0/rx) * (dy0/ry) * dz_res)
            rcond_list.append(np.ones(self.level1[name].n, dtype=float)*rcond_res)
            hcap_list.append(np.ones(self.level1[name].n, dtype=float) * hcap_res)

        dx = np.concatenate(dx_list)
        dy = np.concatenate(dy_list)
        dz = np.concatenate(dz_list)
        kx = np.concatenate(kx_list)
        ky = np.concatenate(ky_list)
        kz = np.concatenate(kz_list)
        poro = np.concatenate(poro_list)
        depth = np.concatenate(depth_list)
        volume = np.concatenate(volume_list)
        rcon = np.concatenate(rcond_list)
        hcap = np.concatenate(hcap_list)



        self.reservoir = LGRReservoir(self.timer,
                                        cell_m=cm_all,
                                        cell_p=cp_all,
                                        tran=T_all,
                                        tran_thermal=T_all_therm,
                                        poro=poro,
                                        rcond=rcon,
                                        hcap=hcap,
                                        depth=depth,
                                        volume=volume,
                                        dx=dx,
                                        dy=dy,
                                        dz=dz,
                                        kx=kx,
                                        ky=ky,
                                        kz=kz,
                                        )
        self.lgr_meta = meta
        self.build_cell_center()
        return

    # uncompleted, the new connections between lgr and overburden and underburden haven't done
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



    def set_wells(self):
        self.reservoir.add_well("I1")
        self.reservoir.add_well("P1")

        center_2d = self.lgr_meta['well_local_center']

        comp = self.build_well_completion()
        
        for wname, cfg in comp.items():
            lgr_name = cfg["lgr"]
            per_from = cfg["k_from"]
            per_to = cfg["k_to"]
            
            for k in range(per_from -1, per_to):  
                rx,ry,_ = self.lgrs[lgr_name]['lgr_coords_in_parent_grid']['refine']
                inj_local = center_2d + k * (rx * ry)
                inj_global = inj_local + self.lgr_meta['lgr_offsets'][comp[wname]["lgr"]]
                self.reservoir.add_perforation(wname, cell_index=inj_global)
  

    def set_physics(self):
            """Physical properties"""
            # Create property containers:
            components = ['CO2', 'H2O']
            phases = ['CO2_rich', 'aqueous']
            Mw = [44.01, 18.015]

            temperature = 80+273.15
            property_container = PropertyContainer(phases_name=phases, components_name=components,
                                                Mw=Mw, min_z=self.zero / 10, temperature=temperature)

            """ properties correlations """
            property_container.flash_ev = ConstantK(len(components), [4, 1e-1], self.zero)
            property_container.density_ev = dict([('CO2_rich', DensityBasic(compr=1e-3, dens0=200)),
                                                ('aqueous', DensityBasic(compr=1e-5, dens0=600))])
            property_container.viscosity_ev = dict([('CO2_rich', ConstFunc(0.05)),
                                                    ('aqueous', ConstFunc(0.5))])
            property_container.rel_perm_ev = dict([('CO2_rich', PhaseRelPerm("gas")),
                                                ('aqueous', PhaseRelPerm("wat"))])

            """ Activate physics """
            thermal = False
            state_spec = Compositional.StateSpecification.PT if thermal else Compositional.StateSpecification.P
            self.physics = Compositional(components, phases, self.timer, state_spec=state_spec,
                                        n_points=200, min_p=1, max_p=300, min_z=self.zero/10, max_z=1-self.zero/10)
            # property_container.output_props = {
            #     "sat0": lambda: property_container.sat[0],
            #     "dens0": lambda: property_container.dens[0],
            #     "nu0": lambda: property_container.nu[0],
            #     "x00": lambda: property_container.x[0,0]
            #     }

            self.physics.add_property_region(property_container)

            return

    def set_initial_conditions(self):
        input_distribution = {self.physics.vars[0]: 195,
                                self.physics.vars[1]: self.zero,
                                }
        return self.physics.set_initial_conditions_from_array(mesh=self.reservoir.mesh,
                                                                input_distribution=input_distribution)

    
        # self.reservoir.mesh.volume[0:3] = 1e20

    def set_well_controls(self):
        inj_composition = [1.0 - self.zero]
        for i, w in enumerate(self.reservoir.wells):
            if i == 0:
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.BHP,
                                                is_inj=True, target=250., inj_composition=inj_composition)
            else:
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.BHP,
                                                is_inj=False, target=50.)

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
    def add_perforation(
        self,
        well_name: str,
        cell_index: int,
        well_radius: float = 0.0762,
        well_index: float = None,
        well_indexD: float = 0.0,
        segment_direction: str = 'z_axis',
        skin: float = 0,
        multi_segment: bool = False,
        verbose: bool = False,
    ):
        well = self.get_well(well_name)
        wi, wid = self.cal_well_index_from_cell(
            cell_index, well_radius, segment_direction, skin
        )

        if well_index is None:
            well_index = wi

        if well_indexD is None:
            well_indexD = wid

        if multi_segment:
            well_block = len(well.perforations)
        else:
            well_block = 0

        if len(well.perforations) == 0:
            well.well_head_depth = np.array(self.mesh.depth, copy=False)[cell_index]
            well.well_body_depth = well.well_head_depth
            # I don't consider CPG type
        else:
            well.well_head_depth = min(
                well.well_head_depth,
                np.array(self.mesh.depth, copy=False)[cell_index],
            )
            well.well_body_depth = well.well_body_depth

        for p in well.perforations:
            if p[0] == well_block and p[1] == cell_index:
                raise ValueError(
                    f'Perforation for well {well.name} in block {cell_index} already exists.'
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
