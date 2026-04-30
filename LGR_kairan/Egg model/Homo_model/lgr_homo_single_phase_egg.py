import os
from pathlib import Path

from lgr_assemble import assemble_lgr_connections, LGRReservoir
from darts.reservoirs.struct_reservoir import StructReservoir
from darts.models.cicd_model import DartsModel
from darts.engines import sim_params, well_control_iface
import numpy as np

from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer

from darts.physics.properties.flash import ConstantK, SinglePhase
from darts.physics.properties.basic import ConstFunc, PhaseRelPerm
from darts.physics.properties.density import DensityBasic, Garcia2001

from darts.physics.properties.viscosity import Fenghour1998, Islam2012
from darts.physics.properties.eos_properties import EoSDensity, EoSEnthalpy

from dartsflash.libflash import NegativeFlash
from dartsflash.libflash import CubicEoS, AQEoS, FlashParams, InitialGuess
from dartsflash.components import CompData
from darts.physics.super.initialize import Initialize
from darts.tools.keyword_file_tools import load_single_keyword


class Model(DartsModel):
    def __init__(self, cfg:dict):
        # Call base class constructor
        super().__init__()

        # Measure time spend on reading/initialization
        self.timer.node["initialization"].start()
        self.cfg = cfg
        self.set_reservoir()
        self.zero = 1e-8
        self.set_physics()


        self.set_sim_params(first_ts=1e-6, mult_ts=2, max_ts=30, runtime=1000,
                            tol_newton=1e-3, tol_linear=1e-3,
                            it_newton=10, it_linear=50)

        self.timer.node["initialization"].stop()

    def define_lgr(self):
        lgrs = self.cfg["lgrs"]
        return lgrs

    def build_well_completion(self):
        return self.cfg["wells"]

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

     # build cell center coordinates for visualization
    def build_cell_center(self):
        """
        """
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
            cfg = self.lgrs[name]['lgr_coords_in_parent_grid']
            i_start, j_start = cfg['i_range'][0], cfg['j_range'][0] # 1-based
            k1, k2 = cfg['k_range']
            nk = k2 - k1 + 1
            rx, ry, rz = cfg['refine']

            dx_3d = np.asarray(self.level1[name].global_data["dx"], dtype=float)
            dy_3d = np.asarray(self.level1[name].global_data["dy"], dtype=float)

            dx_vec = dx_3d[:,0,0].copy()
            dy_vec = dy_3d[0,:,0].copy()

            assert len(dx_vec) == rx, f"len(dx_vec)={len(dx_vec)} does not match rx={rx}"
            assert len(dy_vec) == ry, f"len(dy_vec)={len(dy_vec)} does not match ry={ry}"
            x0 = (i_start - 1) * dx0 # left corner of lgr block in global coordinate
            y0 = (j_start - 1) * dy0

            x_centers_local = x0 + np.cumsum(dx_vec) - 0.5 * dx_vec
            y_centers_local = y0 + np.cumsum(dy_vec) - 0.5 * dy_vec
            local_counter = 0
            for kk in range(nk):
                for jj in range(ry):
                    for ii in range(rx):
                        global_id = offset + local_counter
                        x[global_id] = x_centers_local[ii]
                        y[global_id] = y_centers_local[jj]
                        local_counter += 1

        self.reservoir.cell_center_x = x
        self.reservoir.cell_center_y = y
        self.reservoir.cell_center_z = z
        return x,y,z

    def set_reservoir(self):
        # set heterogeneous egg model
        self.lgrs = self.define_lgr()
        (nx,ny,nz) = (self.cfg["reservoir"]["nx"], self.cfg["reservoir"]["ny"], self.cfg["reservoir"]["nz"])
        nb = nx*ny*nz

        dx = self.cfg["reservoir"]["dx"]
        dy = self.cfg["reservoir"]["dy"]
        # dz0 = np.array([10,70,10])
        dz = self.cfg["reservoir"]["dz"]

        burden = self.cfg["burden"]


        poro_burden = burden["poro_burden"]
        perm_burden = burden["perm_burden"]

        rcond_over, rcond_under = burden["rcond_over"], burden["rcond_under"]
        hcap_over, hcap_under = burden["hcap_over"], burden["hcap_under"]
        # --- layer masks: 9 layers total = 1 overburden + 7 reservoir + 1 underburden ---
        nz_over = 1
        nz_res = 7
        # nz_res = 1 # assume 1 layer reservoir
        nz_under = 1

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

        k_index0 = np.arange(nb, dtype=np.int32) // (nx * ny)

        # --- initialize full-domain properties with burden defaults ---
        kx0_full = np.full(nb, perm_burden, dtype=float)
        ky0_full = np.full(nb, perm_burden, dtype=float)
        kz0_full = np.full(nb, perm_burden, dtype=float)

        rcon0_full = np.empty(nb, dtype=float)
        hcap0_full = np.empty(nb, dtype=float)
        poro0_full = np.full(nb, poro_burden, dtype=float)

        assert nz == nz_over + nz_res + nz_under, (
            f"nz={nz} does not match nz_over+nz_res+nz_under={nz_over+nz_res+nz_under}"
        )

        mask_over = k_index0 < nz_over
        mask_res = (k_index0 >= nz_over) & (k_index0 < nz_over + nz_res)
        mask_under = k_index0 >= (nz_over + nz_res)

        # --- reshape Egg model permeability to reservoir part only ---
        permx_res = 800
        permy_res = 800
        permz_res = 800



        # assign reservoir permeability into middle 7 layers
        kx0_full[mask_res] = permx_res
        ky0_full[mask_res] = permy_res
        kz0_full[mask_res] = permz_res

        # --- thermal properties ---
        rcon0_full[mask_over] = rcond_over
        rcon0_full[mask_res] = 500
        rcon0_full[mask_under] = rcond_under
        hcap0_full[mask_over] = hcap_over
        hcap0_full[mask_res] = 2200
        hcap0_full[mask_under] = hcap_under

        poro0_full[mask_res] =self.cfg["reservoir"]["poro"]



        actnum0 = self.create_actnum_with_lgr(nx, ny, nz, refined_cells_ijk)
        self.level0 = StructReservoir(self.timer, nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz,
                                      permx=kx0_full, permy=ky0_full, permz=kz0_full, poro=poro0_full, depth=None, start_z=490,
                                      hcap=hcap0_full, rcond=rcon0_full, actnum=actnum0)

        v_big = 1e20

        self.level0.boundary_volumes = {
            "xy_minus": v_big,
            "xy_plus": v_big,
            "yz_minus": None,
            "yz_plus": None,
            "xz_minus": None,
            "xz_plus": None,
        }


        # Build Level1 grids and imaginary grids
        self.level1 = {}
        self.level1_imag = {}
        self.level1_imag_z_top = {}
        self.level1_imag_z_bot = {}

        for name, cfg in self.lgrs.items():
            i1 = cfg['lgr_coords_in_parent_grid']['i_range'][0]
            j1 = cfg['lgr_coords_in_parent_grid']['j_range'][0]
            ip = i1 - 1 # 0-based index of the coarse cell in parent grid that will be refined
            jp = j1 - 1 # 0-based index of the coarse cell in parent grid that will be refined
            k1, k2 = cfg['lgr_coords_in_parent_grid']['k_range']
            nk = k2 - k1 + 1
            assert 2 <= i1 <= nx - 2, f"LGR {name} too close to x boundary"
            assert 2 <= j1 <= ny - 2, f"LGR {name} too close to y boundary"


            nx1, ny1,_ = cfg['lgr_coords_in_parent_grid']['refine']
            nz1 = nk
            dx_vec = cfg['lgr_coords_in_parent_grid']['dx_vec']
            dx_vec = np.array(dx_vec, dtype=float)
            dx1 = np.broadcast_to(dx_vec[:, None, None], (nx1, ny1, nz1)).copy()
            dy_vec = cfg['lgr_coords_in_parent_grid']['dy_vec']
            dy_vec = np.array(dy_vec, dtype=float)
            dy1 = np.broadcast_to(dy_vec[None, :, None], (nx1, ny1, nz1)).copy()
            assert len(dx_vec) == nx1, f"len(dx_vec)={len(dx_vec)} != nx1={nx1}"
            assert len(dy_vec) == ny1, f"len(dy_vec)={len(dy_vec)} != ny1={ny1}"

            permx_f = np.empty((nx1, ny1, nz1), dtype=float)
            permy_f = np.empty((nx1, ny1, nz1), dtype=float)
            permz_f = np.empty((nx1, ny1, nz1), dtype=float)
            for kk, k in enumerate(range(k1, k2+1)):
                pk = k - 1
                permx_f[:,:,kk] = self.level0.global_data['permx'][ip,jp,pk]
                permy_f[:,:,kk] = self.level0.global_data['permy'][ip,jp,pk]
                permz_f[:,:,kk] = self.level0.global_data['permz'][ip,jp,pk]

            poro_f = np.ones((nx1, ny1, nz1), dtype=float) * 0.2

            self.level1[name] = StructReservoir(
                self.timer,
                nx=nx1, ny=ny1, nz=nz1,
                dx=dx1, dy=dy1, dz=dz,
                permx=permx_f, permy=permy_f, permz=permz_f,
                poro=poro_f,
                depth=None,
                start_z=500,
                rcond=500,
                hcap=2200,
            )

            # 1) lateral imaginary grid
            nx_im = nx1 + 2
            ny_im = ny1 + 2
            nz_im = nk
            dx_imag_vec = np.r_[dx, dx_vec, dx].astype(float)
            dy_imag_vec = np.r_[dy, dy_vec, dy].astype(float)
            dx_imag = np.broadcast_to(dx_imag_vec[:, None, None], (nx_im, ny_im, nz_im)).copy()
            dy_imag = np.broadcast_to(dy_imag_vec[None, :, None], (nx_im, ny_im, nz_im)).copy()
            dz_imag = np.full((nx_im, ny_im, nz_im), dz, dtype=float)
            permx_im = np.empty((nx_im, ny_im, nz_im), dtype=float)
            permy_im = np.empty((nx_im, ny_im, nz_im), dtype=float)
            permz_im = np.empty((nx_im, ny_im, nz_im), dtype=float)
            poro_im = np.full((nx_im, ny_im, nz_im), 0.2, dtype=float)

            # Find the properties of the refined cells in the parent grid to assign to the LGR grid and imaginary grids
            for kk, k in enumerate(range(k1, k2 + 1)):
                pk = k - 1 # 0-based
                 # center fine region
                permx_im[1:-1, 1:-1, kk] = self.level0.global_data['permx'][ip, jp, pk]
                permy_im[1:-1, 1:-1, kk] = self.level0.global_data['permy'][ip, jp, pk]
                permz_im[1:-1, 1:-1, kk] = self.level0.global_data['permz'][ip, jp, pk]



                # left halo
                permx_im[0,    1:-1, kk] = self.level0.global_data['permx'][ip-1, jp,   pk]
                permy_im[0,    1:-1, kk] = self.level0.global_data['permy'][ip-1, jp,   pk]
                permz_im[0,    1:-1, kk] = self.level0.global_data['permz'][ip-1, jp,   pk]


                # right halo
                permx_im[-1,   1:-1, kk] = self.level0.global_data['permx'][ip+1, jp,   pk]
                permy_im[-1,   1:-1, kk] = self.level0.global_data['permy'][ip+1, jp,   pk]
                permz_im[-1,   1:-1, kk] = self.level0.global_data['permz'][ip+1, jp,   pk]



                # upper halo (j-1)
                permx_im[1:-1, 0,    kk] = self.level0.global_data['permx'][ip,   jp-1, pk]
                permy_im[1:-1, 0,    kk] = self.level0.global_data['permy'][ip,   jp-1, pk]
                permz_im[1:-1, 0,    kk] = self.level0.global_data['permz'][ip,   jp-1, pk]



                # lower halo (j+1)
                permx_im[1:-1, -1,   kk] = self.level0.global_data['permx'][ip,   jp+1, pk]
                permy_im[1:-1, -1,   kk] = self.level0.global_data['permy'][ip,   jp+1, pk]
                permz_im[1:-1, -1,   kk] = self.level0.global_data['permz'][ip,   jp+1, pk]


                # corners
                permx_im[0,  0,  kk] = self.level0.global_data['permx'][ip-1, jp-1, pk]
                permy_im[0,  0,  kk] = self.level0.global_data['permy'][ip-1, jp-1, pk]
                permz_im[0,  0,  kk] = self.level0.global_data['permz'][ip-1, jp-1, pk]


                permx_im[0, -1,  kk] = self.level0.global_data['permx'][ip-1, jp+1, pk]
                permy_im[0, -1,  kk] = self.level0.global_data['permy'][ip-1, jp+1, pk]
                permz_im[0, -1,  kk] = self.level0.global_data['permz'][ip-1, jp+1, pk]


                permx_im[-1, 0,  kk] = self.level0.global_data['permx'][ip+1, jp-1, pk]
                permy_im[-1, 0,  kk] = self.level0.global_data['permy'][ip+1, jp-1, pk]
                permz_im[-1, 0,  kk] = self.level0.global_data['permz'][ip+1, jp-1, pk]


                permx_im[-1, -1, kk] = self.level0.global_data['permx'][ip+1, jp+1, pk]
                permy_im[-1, -1, kk] = self.level0.global_data['permy'][ip+1, jp+1, pk]
                permz_im[-1, -1, kk] = self.level0.global_data['permz'][ip+1, jp+1, pk]


            self.level1_imag[name] = StructReservoir(self.timer, nx=nx1+2, ny=ny1+2, nz=nz1, dx=dx_imag, dy=dy_imag, dz=dz_imag,
                                        permx=permx_im, permy=permy_im, permz=permz_im, poro=poro_im, depth= None, start_z=500, rcond=500, hcap=2200)


            # top vertical imaginary grid for overburden connection(5,5,2)
            dx_top = np.broadcast_to(dx_vec[:, None, None], (nx1, ny1, 2)).copy()
            dy_top = np.broadcast_to(dy_vec[None, :, None], (nx1, ny1, 2)).copy()
            dz_top = np.full((nx1, ny1, 2), dz, dtype=float)

            permx_top = np.empty((nx1, ny1, 2), dtype=float)
            permy_top = np.empty((nx1, ny1, 2), dtype=float)
            permz_top = np.empty((nx1, ny1, 2), dtype=float)

            rcond_top = np.empty((nx1, ny1, 2), dtype=float)
            hcap_top  = np.empty((nx1, ny1, 2), dtype=float)

            # top burden coarse layer: k = k1-1 in 1-based, so pk = k1-2
            pk_over = k1 - 2
            pk_top_res = k1 - 1

            permx_top[:, :, 0] = self.level0.global_data['permx'][ip, jp, pk_over]
            permy_top[:, :, 0] = self.level0.global_data['permy'][ip, jp, pk_over]
            permz_top[:, :, 0] = self.level0.global_data['permz'][ip, jp, pk_over]

            rcond_top[:, :, 0] = self.cfg["burden"]["rcond_over"]
            hcap_top[:, :, 0]  = self.cfg["burden"]["hcap_over"]

            permx_top[:, :, 1] = self.level0.global_data['permx'][ip, jp, pk_top_res]
            permy_top[:, :, 1] = self.level0.global_data['permy'][ip, jp, pk_top_res]
            permz_top[:, :, 1] = self.level0.global_data['permz'][ip, jp, pk_top_res]

            rcond_top[:, :, 1] = 500
            hcap_top[:, :, 1]  = 2200
            self.level1_imag_z_top[name] = StructReservoir(
                self.timer,
                nx=nx1, ny=ny1, nz=2,
                dx=dx_top, dy=dy_top, dz=dz_top,
                permx=permx_top, permy=permy_top, permz=permz_top,
                poro=0.2,
                depth=None,
                start_z=490,
                rcond=rcond_top,
                hcap=hcap_top
            )

            # 3) BOTTOM VERTICAL IMAGINARY GRID: (5, 5, 2)
            # layer 0 = reservoir bottom fine layer
            # layer 1 = underburden coarse layer

            dx_bot = np.broadcast_to(dx_vec[:, None, None], (nx1, ny1, 2)).copy()
            dy_bot = np.broadcast_to(dy_vec[None, :, None], (nx1, ny1, 2)).copy()
            dz_bot = np.full((nx1, ny1, 2), dz, dtype=float)

            permx_bot = np.empty((nx1, ny1, 2), dtype=float)
            permy_bot = np.empty((nx1, ny1, 2), dtype=float)
            permz_bot = np.empty((nx1, ny1, 2), dtype=float)
            poro_bot  = np.empty((nx1, ny1, 2), dtype=float)
            rcond_bot = np.empty((nx1, ny1, 2), dtype=float)
            hcap_bot  = np.empty((nx1, ny1, 2), dtype=float)

            pk_bot_res = k2 - 1
            pk_under   = k2

            permx_bot[:, :, 0] = self.level0.global_data['permx'][ip, jp, pk_bot_res]
            permy_bot[:, :, 0] = self.level0.global_data['permy'][ip, jp, pk_bot_res]
            permz_bot[:, :, 0] = self.level0.global_data['permz'][ip, jp, pk_bot_res]
            poro_bot[:, :, 0]  = 0.2
            rcond_bot[:, :, 0] = 500
            hcap_bot[:, :, 0]  = 2200

            permx_bot[:, :, 1] = self.level0.global_data['permx'][ip, jp, pk_under]
            permy_bot[:, :, 1] = self.level0.global_data['permy'][ip, jp, pk_under]
            permz_bot[:, :, 1] = self.level0.global_data['permz'][ip, jp, pk_under]
            poro_bot[:, :, 1]  = self.cfg["burden"]["poro_burden"]
            rcond_bot[:, :, 1] = self.cfg["burden"]["rcond_under"]
            hcap_bot[:, :, 1]  = self.cfg["burden"]["hcap_under"]

            self.level1_imag_z_bot[name] = StructReservoir(
                self.timer,
                nx=nx1, ny=ny1, nz=2,
                dx=dx_bot, dy=dy_bot, dz=dz_bot,
                permx=permx_bot, permy=permy_bot, permz=permz_bot,
                poro=poro_bot,
                depth=None,
                start_z=490 + (nz_over + nz_res - 1) * dz,
                rcond=rcond_bot,
                hcap=hcap_bot
            )

        cm_all, cp_all, T_all, T_all_therm, meta = assemble_lgr_connections(self)
        self.lgr_meta = meta

       # assemble properties
        self.level0.discretize()
        disc0 = self.level0.discretizer
        l2g0 = disc0.local_to_global          # l2g0 has already eliminated inactive cells

        dx0_arr = disc0.convert_to_flat_array(self.level0.global_data['dx'], 'dx')[l2g0]
        dy0_arr = disc0.convert_to_flat_array(self.level0.global_data['dy'], 'dy')[l2g0]
        dz0_arr = disc0.convert_to_flat_array(self.level0.global_data['dz'], 'dz')[l2g0]

        depth0_arr = disc0.convert_to_flat_array(self.level0.global_data['depth'], 'depth')[l2g0]
        volume0_arr = np.array(self.level0.volume, copy=False).astype(float) # self.level0.volume is already filtered

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

        for name in self.lgr_meta["lgr_orders"]:
            self.level1[name].discretize()
            disc1 = self.level1[name].discretizer
            dx_f = disc1.convert_to_flat_array(self.level1[name].global_data["dx"], "dx")
            dy_f = disc1.convert_to_flat_array(self.level1[name].global_data["dy"], "dy")
            dz_f = disc1.convert_to_flat_array(self.level1[name].global_data["dz"], "dz")

            dx_list.append(dx_f)
            dy_list.append(dy_f)
            dz_list.append(dz_f)

            kx_list.append(disc1.convert_to_flat_array(self.level1[name].global_data["permx"], "permx"))
            ky_list.append(disc1.convert_to_flat_array(self.level1[name].global_data["permy"], "permy"))
            kz_list.append(disc1.convert_to_flat_array(self.level1[name].global_data["permz"], "permz"))

            poro_list.append(disc1.convert_to_flat_array(self.level1[name].global_data["poro"], "poro"))
            depth_list.append(disc1.convert_to_flat_array(self.level1[name].global_data["depth"], "depth"))

            # volume: structured fine grid, direct from discretized volume is safer
            volume_list.append(dx_f * dy_f * dz_f)

            rcond_list.append(np.ones(self.level1[name].n, dtype=float)*500)
            hcap_list.append(np.ones(self.level1[name].n, dtype=float) * 2200)


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

        self.build_cell_center()
        return



    def set_wells(self):
        wells= self.cfg["wells"]
        for key, value in wells.items():
            self.reservoir.add_well(key)

        center_2d = self.lgr_meta['well_local_center']

        comp = self.build_well_completion()
        for wname, cfg in comp.items():
            lgr_name = cfg["lgr"]
            per_from = cfg["k_from"]
            per_to = cfg["k_to"]
            # note:
            # k_from / k_to in cfg["wells"] are LOCAL layer indices inside the LGR subgrid,
            # not global layer indices in the parent grid.
            for k in range(per_from -1, per_to):
                rx,ry,_ = self.lgrs[lgr_name]['lgr_coords_in_parent_grid']['refine']
                inj_local = center_2d + k * (rx * ry)
                inj_global = inj_local + self.lgr_meta['lgr_offsets'][comp[wname]["lgr"]]
                self.reservoir.add_perforation(wname, cell_index=inj_global,ms_epm=True, well_radius=0.0762)

    # """single phase- single component model"""
    # def set_physics(self):
    #     components = ['CO2']

    #     self.components = components
    #     comp_data = CompData(components, setprops=True)
    #     pr = CubicEoS(comp_data, CubicEoS.PR)

    #     self.zero = 1e-12
    #     epsilon = self.zero / 10
    #     phases = ['CO2_rich']

    #     property_container = ModelProperties(phases_name=phases, components_name=components, eps_z=epsilon, Mw=comp_data.Mw)

    #     # Define property evaluators based on custom properties
    #     property_container.density_ev = dict([('CO2_rich', EoSDensity(eos=pr,Mw=comp_data.Mw))])
    #     property_container.viscosity_ev = dict([('CO2_rich', Fenghour1998())])

    #     property_container.enthalpy_ev = dict([('CO2_rich', EoSEnthalpy(eos=pr))])
    #     property_container.conductivity_ev = dict([('CO2_rich', ConstFunc(10.)) ])


    #     """ Activate physics """
    #     thermal = True
    #     state_spec = Compositional.StateSpecification.PT if thermal else Compositional.StateSpecification.P
    #     self.physics = Compositional(components, phases, self.timer, state_spec=state_spec,
    #                                  n_points=400, min_p=1, max_p=1000, min_z=self.zero/10, max_z=1-self.zero/10,
    #                                  epsilon_z=epsilon, min_t=273.15, max_t=373.15+200)


    #     property_container.output_props = {
    #         "satG": lambda: property_container.sat[0],
    #         "rhoG": lambda: property_container.dens[0],
    #          "muG": lambda: property_container.mu[0],
    #         }

    #     self.physics.add_property_region(property_container)

    #     return
    def set_physics(self):
        components_names = ['CO2']
        phases_names = ['CO2_rich']
        self.components = components_names
        comp_data = CompData(components_names, setprops=True)
        pr = CubicEoS(comp_data, CubicEoS.PR)
        epsilon = self.zero / 10
        state_spec = Compositional.StateSpecification.PT
        min_t = 150
        max_t = 200 + 273.15
        self.physics = Compositional(
            components_names,
            phases_names,
            self.timer,
            state_spec=state_spec,
            n_points=10000,
            min_p=1,
            max_p=500,
            min_z=0.0,
            max_z=1.0,
            epsilon_z=epsilon,
            min_t=min_t,
            max_t=max_t,
        )

        property_container = PropertyContainer(
            phases_name=phases_names,
            components_name=components_names,
            Mw=comp_data.Mw,
            eps_z=epsilon,
            temperature=None,
            rock_comp=0,
        )
        property_container.flash_ev = SinglePhase(nc=1)

        property_container.density_ev = {
            'CO2_rich': EoSDensity(eos=pr, Mw=comp_data.Mw),
        }
        property_container.enthalpy_ev = {
            'CO2_rich': EoSEnthalpy(eos=pr),
        }
        property_container.viscosity_ev = {
            'CO2_rich': Fenghour1998(),
        }
        property_container.conductivity_ev = {
            'CO2_rich': ConstFunc(2.2),
        }
        property_container.rel_perm_ev = {
            'CO2_rich': ConstFunc(1.0),
        }

        self.physics.add_property_region(property_container)

        property_container.output_props = {
            'temperature': lambda: property_container.temperature,
            'sat_CO2_rich': lambda: property_container.sat[0],
            'rho_CO2_rich': lambda: property_container.dens[0],
            'miu_CO2_rich': lambda: property_container.mu[0],
            'enth_CO2_rich': lambda: property_container.enthalpy[0],
            'CO2_in_CO2_rich': lambda: property_container.x[0, 0],
        }

        return


    def set_initial_conditions(self):
        # input_distribution = {self.physics.vars[0]: 200, # pressure
        #                       self.physics.vars[1]: 353.15 # temperature
        #                       }
        # return self.physics.set_initial_conditions_from_array(mesh=self.reservoir.mesh,
        #                                                       input_distribution=input_distribution)

        depths = np.asarray(self.reservoir.mesh.depth)
        min_depth = np.min(depths)
        max_depth = np.max(depths)
        nb = int(self.level0.nz) # number of depth values
        depths = np.linspace(min_depth,max_depth,nb)

        init = Initialize(self.physics)

        primary_specs = {}
        for comp in self.physics.components[:-1]:
            primary_specs[comp] = 1.0

        boundary_state = {"pressure" :50}
        for comp in self.physics.components[:-1]:
            boundary_state[comp] = primary_specs[comp]
        boundary_state["temperature"] = 32 +273.15

        dTdh = 34/1000 #k/m

        X = init.solve_up_and_downwards(depth_bottom=max_depth, depth_top=min_depth, depth_known=500,
                                        boundary_state=boundary_state, primary_specs=primary_specs, nb=nb,
                                        dTdh=dTdh)

        # self.physics.set_initial_conditions_from_depth_table(mesh=self.reservoir.mesh,
        #                                                      input_depth= init.depths,
        #                                                     input_distribution={v:X[:,i] for i, v in enumerate(self.physics.vars)})
        
        self.physics.set_initial_conditions_from_depth_table(
                mesh=self.reservoir.mesh,
                input_depth=init.depths,
                input_distribution={
                    "pressure": X[:, init.var_idxs["pressure"]],
                    "temperature": X[:, init.var_idxs["temperature"]],
                },
            )
        return



    def set_well_controls(self):
        inj_composition = [1.0]  # pure CO2 injection
        for i, w in enumerate(self.reservoir.wells):
            if i == 0:
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.MASS_RATE,
                                            is_inj=True,
                                            phase_name='CO2_rich',
                                            target=4.32e6, inj_composition=inj_composition, inj_temp=16+273.15)

            else:
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.BHP,
                                               is_inj=False, target=40.)
