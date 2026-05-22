import os
from lgr_assemble import assemble_lgr_connections, LGRReservoir
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
import pandas as pd

from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer
from darts.physics.properties.flash import Flash, RR2

from darts.physics.properties.flash import ConstantK
from darts.physics.properties.basic import ConstFunc, PhaseRelPerm
from darts.physics.properties.density import DensityBasic, Spivey2004, Garcia2001

from darts.physics.properties.viscosity import Fenghour1998, Islam2012
from darts.physics.properties.eos_properties import EoSDensity, EoSEnthalpy

from dartsflash.libflash import NegativeFlash
from dartsflash.libflash import CubicEoS, AQEoS, FlashParams, InitialGuess
from dartsflash.components import CompData
from darts.physics.super.initialize import Initialize
from darts.tools.interpolation import TableInterpolation
from darts.tools.keyword_file_tools import *


class WatRelPerm:
    def __init__(self, pvt):
        super().__init__()
        self.pvt = pvt
        self.SGAF = get_table_keyword(self.pvt, 'SGAF')

    def evaluate(self, wat_sat):
        gas_index = 0
        krwg_index = 2
        gas_sat = 1 - wat_sat
        Table = TableInterpolation()
        if gas_sat < self.SGAF[0][0] or gas_sat > self.SGAF[len(self.SGAF) - 1][0]:
            krwg = Table.SCALExtraP(self.SGAF, gas_sat, gas_index, krwg_index)
        else:
            krwg = Table.LinearInterP(self.SGAF, gas_sat, gas_index, krwg_index)
        return krwg


class GasRelPerm:
    def __init__(self, pvt):
        super().__init__()
        self.pvt = pvt
        self.SGAF = get_table_keyword(self.pvt, 'SGAF')

    def evaluate(self, gas_sat):
        gas_index = 0
        krg_index = 1

        Table = TableInterpolation()
        if gas_sat < self.SGAF[0][0] or gas_sat > self.SGAF[len(self.SGAF) - 1][0]:
            krg = Table.SCALExtraP(self.SGAF, gas_sat, gas_index, krg_index)
        else:
            krg = Table.LinearInterP(self.SGAF, gas_sat, gas_index, krg_index)

        return krg

class Garcia2001(Spivey2004):
    """
    Correlation for brine density with dissolved CO2: Garcia (2001) - Density of aqueous solutions of CO2
    """

    def __init__(self, components: list, ions: list = None, combined_ions: list = None):
        super().__init__(components, ions, combined_ions)

        self.CO2_idx = components.index("CO2") if "CO2" in components else None

    def evaluate(self, pressure, temperature, x):
        """"""
        # simplification: use basic density for water because there is no salt and then apply correction if CO2 is present
        rho_b = DensityBasic(dens0=1020, compr=4.5e-5, p0=1.01325).evaluate(pressure, temperature, x)
        # If CO2 is present, correct density
        if self.CO2_idx is not None:
            # Apparent molar volume of dissolved CO2
            tc = temperature - 273.15  # Temp in [Celcius]
            V_app = (
                37.51 - 9.585e-2 * tc + 8.740e-4 * tc**2 - 5.044e-7 * tc**3
            ) * 1e-6  # in [m3 / mol]

            mCO2 = 55.509 * x[self.CO2_idx] / (x[self.H2O_idx])
            MW = 44.01  # molecular weight of CO2
            rho = (1.0 + mCO2 * MW * 1e-3) / (
                mCO2 * V_app + 1.0 / rho_b
            )  # in [kg / m3]
        else:
            rho = rho_b

        return rho

class TableKFlash(Flash):
    def __init__(self, nc, table_path, p_axis=None, t_axis=None, eps=1e-11):
        super().__init__(nph=2, nc=nc)

        self.rr_eps = eps
        df = pd.read_csv(table_path)
        self.p_axis = np.sort(df["P_bar"].unique())
        self.t_axis = np.sort(df["T_K"].unique())
        n_p = len(self.p_axis)
        n_t = len(self.t_axis)
        self.K_co2_table = np.zeros((n_p, n_t), dtype=float)
        self.K_h2o_table = np.zeros((n_p, n_t), dtype=float)
        for i, p in enumerate(self.p_axis):
            df_p = df[df["P_bar"] == p].sort_values("T_K")
            self.K_co2_table[i, :] = df_p["K_CO2"].values
            self.K_h2o_table[i, :] = df_p["K_H2O"].values

    def evaluate(self, pressure, temperature, zc):
        self.K_values = self.get_k_values(pressure, temperature)
        self.nu, self.X = RR2(self.K_values, zc, self.rr_eps)
        if self.nu[0] < 0.:
            self.nu = [0., 1.]
            self.X = [[0., 0.], zc]
        elif self.nu[0] > 1.:
            self.nu = [1., 0.]
            self.X = [zc, [0., 0.]]
        self.temperature = temperature

        return 0

    def get_k_values(self, pressure, temperature):
        p = pressure
        t = temperature
        i1 = np.searchsorted(self.p_axis, p)
        j1 = np.searchsorted(self.t_axis, t)
        # find location in table and corner points for interpolation
        if i1 == 0:
            i0 = i1 = 0
        elif i1 >= len(self.p_axis):
            i0 = i1 = len(self.p_axis) - 1
        else:
            i0 = i1 - 1

        if j1 == 0:
            j0 = j1 = 0
        elif j1 >= len(self.t_axis):
            j0 = j1 = len(self.t_axis) - 1
        else:
            j0 = j1 - 1

        p0, p1 = self.p_axis[i0], self.p_axis[i1]
        t0, t1 = self.t_axis[j0], self.t_axis[j1]

        # four corners for bilinear interpolation
        kco2_00 = self.K_co2_table[i0, j0]
        kco2_10 = self.K_co2_table[i1, j0]
        kco2_01 = self.K_co2_table[i0, j1]
        kco2_11 = self.K_co2_table[i1, j1]

        kh2o_00 = self.K_h2o_table[i0, j0]
        kh2o_10 = self.K_h2o_table[i1, j0]
        kh2o_01 = self.K_h2o_table[i0, j1]
        kh2o_11 = self.K_h2o_table[i1, j1]

        # if PT is coincident with table point, return directly to avoid interpolation error
        if i0 == i1 and j0 == j1:
            return np.array([kco2_00, kh2o_00])
        if i0 == i1:
            wt = (t - t0) / (t1 - t0)
            kco2 = kco2_00 * (1 - wt) + kco2_01 * wt
            kh2o = kh2o_00 * (1 - wt) + kh2o_01 * wt
            return np.array([kco2, kh2o])

        if j0 == j1:
            wp = (p - p0) / (p1 - p0)
            kco2 = kco2_00 * (1 - wp) + kco2_10 * wp
            kh2o = kh2o_00 * (1 - wp) + kh2o_10 * wp
            return np.array([kco2, kh2o])

        wp = (p - p0) / (p1 - p0)
        wt = (t - t0) / (t1 - t0)

        kco2 = (
            kco2_00 * (1 - wp) * (1 - wt) +
            kco2_10 * wp * (1 - wt) +
            kco2_01 * (1 - wp) * wt +
            kco2_11 * wp * wt
        )

        kh2o = (
            kh2o_00 * (1 - wp) * (1 - wt) +
            kh2o_10 * wp * (1 - wt) +
            kh2o_01 * (1 - wp) * wt +
            kh2o_11 * wp * wt
        )

        return np.array([kco2, kh2o])


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


        self.set_sim_params(first_ts=1e-6, mult_ts=2, max_ts=2, runtime=1000,
                            tol_newton=2e-3, tol_linear=1e-3,
                            it_newton=10, it_linear=50,
                            well_rate_ctrl_absolute_residual_scale=10.0,
                            well_rate_ctrl_relative_residual_scale=1e-5)

        self.timer.node["initialization"].stop()

    def build_lgr_definition(self):
        """
        -parent name
        -i,j,k range in parent grid (1-based)
        -local name
        """
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
        dx0_data = np.asarray(self.level0.global_data["dx"], dtype=float)
        dy0_data = np.asarray(self.level0.global_data["dy"], dtype=float)
        if dx0_data.ndim <= 1:
            dx0_vec = np.full(nx0, float(dx0_data.flat[0]))
        else:
            dx0_vec = dx0_data[:, 0, 0]
        if dy0_data.ndim <= 1:
            dy0_vec = np.full(ny0, float(dy0_data.flat[0]))
        else:
            dy0_vec = dy0_data[0, :, 0]
        x_edges0 = np.r_[0.0, np.cumsum(dx0_vec)]
        y_edges0 = np.r_[0.0, np.cumsum(dy0_vec)]
        for local_id, global_id in enumerate(l2g0):
            k = global_id // (nx0 * ny0) # 0 based
            j = (global_id % (nx0 * ny0)) // nx0
            i = global_id % nx0
            x[local_id] = 0.5 * (x_edges0[i] + x_edges0[i + 1])
            y[local_id] = 0.5 * (y_edges0[j] + y_edges0[j + 1])

        # lgr blocks
        meta = self.lgr_meta
        for name in meta['lgr_orders']:
            offset = meta['lgr_offsets'][name]
            grid = self.level1[name]
            nx1, ny1, nz1 = int(grid.nx), int(grid.ny), int(grid.nz)
            cfg = self.lgrs[name]['lgr_coords_in_parent_grid']
            i_start, j_start = cfg['i_range'][0], cfg['j_range'][0] # 1-based
            dx1_data = np.asarray(grid.global_data["dx"], dtype=float)
            dy1_data = np.asarray(grid.global_data["dy"], dtype=float)
            if dx1_data.ndim <= 1:
                dx1_vec = np.full(nx1, float(dx1_data.flat[0]))
            else:
                dx1_vec = dx1_data[:, 0, 0]
            if dy1_data.ndim <= 1:
                dy1_vec = np.full(ny1, float(dy1_data.flat[0]))
            else:
                dy1_vec = dy1_data[0, :, 0]
            x0 = x_edges0[i_start - 1] # left corner of lgr block in global coordinate
            y0 = y_edges0[j_start - 1]
            x_edges1 = x0 + np.r_[0.0, np.cumsum(dx1_vec)]
            y_edges1 = y0 + np.r_[0.0, np.cumsum(dy1_vec)]
            n_lgr = int(grid.n)
            nxy1 = nx1 * ny1
            for lid in range(n_lgr): # local id in lgr block
                k = lid // nxy1
                r = lid - k*nxy1
                j = r//nx1
                i = r % nx1

                glid = offset + lid
                x[glid] = 0.5 * (x_edges1[i] + x_edges1[i + 1])
                y[glid] = 0.5 * (y_edges1[j] + y_edges1[j + 1])

        self.reservoir.cell_center_x = x
        self.reservoir.cell_center_y = y
        self.reservoir.cell_center_z = z
        return x,y,z

    def level0_value(self, prop, i0, j0, k0):
        data = np.asarray(self.level0.global_data[prop])
        nx, ny, nz = int(self.level0.nx), int(self.level0.ny), int(self.level0.nz)
        i0 = min(max(int(i0), 0), nx - 1)
        j0 = min(max(int(j0), 0), ny - 1)
        k0 = min(max(int(k0), 0), nz - 1)
        if data.ndim == 0:
            return float(data)
        if data.ndim == 1:
            return float(data[i0 + nx * (j0 + ny * k0)])
        return float(data[i0, j0, k0])

    def make_lgr_array(self, prop, i_parent, j_parent, k_range, shape_xy):
        rx, ry = shape_xy
        k1, k2 = k_range
        nk = k2 - k1 + 1
        out = np.empty((rx, ry, nk), dtype=float)
        for kk, k in enumerate(range(k1, k2 + 1)):
            out[:, :, kk] = self.level0_value(prop, i_parent, j_parent, k - 1)
        return out

    def make_lateral_imag_array(self, prop, i_parent, j_parent, k_range, shape_xy):
        rx, ry = shape_xy
        k1, k2 = k_range
        nk = k2 - k1 + 1
        out = np.empty((rx + 2, ry + 2, nk), dtype=float)
        for kk, k in enumerate(range(k1, k2 + 1)):
            k0 = k - 1
            for ii in range(rx + 2):
                if ii == 0:
                    pi = i_parent - 1
                elif ii == rx + 1:
                    pi = i_parent + 1
                else:
                    pi = i_parent
                for jj in range(ry + 2):
                    if jj == 0:
                        pj = j_parent - 1
                    elif jj == ry + 1:
                        pj = j_parent + 1
                    else:
                        pj = j_parent
                    out[ii, jj, kk] = self.level0_value(prop, pi, pj, k0)
        return out

    def set_reservoir(self):
        self.lgrs = self.build_lgr_definition()
        grid = self.cfg["grid"]
        #Build Level0 with actnum + overburden and underburden
        nx0, ny0 = grid["nx"], grid["ny"] # global grid size
        dx0, dy0 = grid["dx"], grid["dy"]
        nz_res = grid["nz_res"]
        dz_res = grid["dz_res"]
        burden = self.cfg["burden"]
        over_thickness = burden["over_thickness"]
        under_thickness = burden["under_thickness"]
        dz_over = self.build_dz(dz0=dz_res, total_thickness=over_thickness)
        dz_over = dz_over[::-1]  # reverse for overburden

        dz_under = self.build_dz(dz0=dz_res,total_thickness=under_thickness)
        nz_over = len(dz_over)
        self.nz_over = nz_over
        nz_under = len(dz_under)
        nz0 = nz_over + nz_res + nz_under
        dz0_layers = np.concatenate([dz_over, np.full(nz_res, dz_res, dtype=float), dz_under])
        rock = self.cfg["rock"]
        permx0, permy0, permz0 = rock["perm_x"], rock["perm_y"], rock["perm_z"]
        poro0 = rock["poro"]
        poro_burden = burden["poro_burden"]
        perm_burden = burden["perm_burden"]

        # thermal properties
        rcond_res = rock["rcond_res"] # KJ/m/day/k
        hcap_res = rock["hcap_res"] # kJ/m3/K assume reservoir density here
        rcond_over, rcond_under = rock["rcond_over"], rock["rcond_under"]
        hcap_over, hcap_under = rock["hcap_over"], rock["hcap_under"]
        reservoir_top = grid["reservoir_top"]

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


        k_index0 = np.arange(nx0 * ny0 * nz0, dtype=np.int32) // (nx0 * ny0)
        kx0_full = np.full(nx0 * ny0 * nz0, perm_burden, dtype=float)
        ky0_full = np.full(nx0 * ny0 * nz0, perm_burden, dtype=float)
        kz0_full = np.full(nx0 * ny0 * nz0, perm_burden, dtype=float)

        rcon0_full = np.empty(nx0 * ny0 * nz0, dtype=float)
        hcap0_full = np.empty(nx0 * ny0 * nz0, dtype=float)
        poro0_full = np.full(nx0 * ny0 * nz0, poro_burden, dtype=float)

        mask_over = k_index0 < nz_over
        mask_res  = (k_index0 >= nz_over) & (k_index0 < nz_over + nz_res)
        mask_under = k_index0 >= (nz_over + nz_res)
        kx0_full [mask_res] = permx0
        ky0_full [mask_res] = permy0
        kz0_full [mask_res] = permz0

        rcon0_full[mask_over] = rcond_over
        rcon0_full[mask_res] = rcond_res
        rcon0_full[mask_under] = rcond_under
        hcap0_full[mask_over] = hcap_over
        hcap0_full[mask_res] = hcap_res
        hcap0_full[mask_under] =hcap_under
        poro0_full[mask_res] = poro0

        self.level0 = StructReservoir(self.timer, nx=nx0, ny=ny0, nz=nz0, dx=dx0, dy=dy0, dz=dz0_layers,
                                      permx=kx0_full, permy=ky0_full, permz=kz0_full, poro=poro0_full,depth= None,
                                      start_z=0,actnum=actnum0, rcond=rcon0_full, hcap=hcap0_full,)
        boundary_factor = 2000
        base_vol = float(dx0 * dy0 * dz_res)
        v_big = base_vol * boundary_factor

        self.level0.boundary_volumes = {
            "xy_minus": 1e20,
            "xy_plus": 1e20,
            "yz_minus": v_big,
            "yz_plus": v_big,
            "xz_minus": v_big,
            "xz_plus": v_big,
        }


        # Build Level1 grids and imaginary grids
        self.level1, self.level1_imag, self.level1_imag_z_top, self.level1_imag_z_bot = {}, {}, {}, {}

        for name, entry in self.lgrs.items():
            cfg = entry['lgr_coords_in_parent_grid']
            rx, ry, _ = cfg['refine']
            k1, k2 = cfg['k_range']
            nk = k2 - k1 + 1
            i_parent = cfg['i_range'][0] - 1
            j_parent = cfg['j_range'][0] - 1

            dx_vec = np.asarray(cfg.get('dx_vec', np.full(rx, dx0 / rx)), dtype=float)
            dy_vec = np.asarray(cfg.get('dy_vec', np.full(ry, dy0 / ry)), dtype=float)
            dx_f = np.broadcast_to(dx_vec[:, None, None], (rx, ry, nk)).copy()
            dy_f = np.broadcast_to(dy_vec[None, :, None], (rx, ry, nk)).copy()
            dz_f = np.full((rx, ry, nk), dz_res)
            prop_shape = (rx, ry)

            self.level1[name] = StructReservoir(self.timer, nx=rx, ny=ry, nz=nk, dx=dx_f, dy=dy_f, dz=dz_f,
                                        permx=self.make_lgr_array("permx", i_parent, j_parent, (k1, k2), prop_shape),
                                        permy=self.make_lgr_array("permy", i_parent, j_parent, (k1, k2), prop_shape),
                                        permz=self.make_lgr_array("permz", i_parent, j_parent, (k1, k2), prop_shape),
                                        poro=self.make_lgr_array("poro", i_parent, j_parent, (k1, k2), prop_shape),
                                        depth= None, start_z=reservoir_top,
                                        rcond=self.make_lgr_array("rcond", i_parent, j_parent, (k1, k2), prop_shape),
                                        hcap=self.make_lgr_array("hcap", i_parent, j_parent, (k1, k2), prop_shape))

            dx_im = np.broadcast_to(np.r_[dx0, dx_vec, dx0][:, None, None], (rx + 2, ry + 2, nk)).copy()
            dy_im = np.broadcast_to(np.r_[dy0, dy_vec, dy0][None, :, None], (rx + 2, ry + 2, nk)).copy()
            dz_im = np.full((rx + 2, ry + 2, nk), dz_res)

            self.level1_imag[name] = StructReservoir(self.timer, nx=rx+2, ny=ry+2, nz=nk,
                                        dx=dx_im, dy=dy_im, dz=dz_im,
                                        permx=self.make_lateral_imag_array("permx", i_parent, j_parent, (k1, k2), prop_shape),
                                        permy=self.make_lateral_imag_array("permy", i_parent, j_parent, (k1, k2), prop_shape),
                                        permz=self.make_lateral_imag_array("permz", i_parent, j_parent, (k1, k2), prop_shape),
                                        poro=self.make_lateral_imag_array("poro", i_parent, j_parent, (k1, k2), prop_shape),
                                        depth= None, start_z=reservoir_top,
                                        rcond=self.make_lateral_imag_array("rcond", i_parent, j_parent, (k1, k2), prop_shape),
                                        hcap=self.make_lateral_imag_array("hcap", i_parent, j_parent, (k1, k2), prop_shape))

            dx_z = np.broadcast_to(dx_vec[:, None, None], (rx, ry, 2)).copy()
            dy_z = np.broadcast_to(dy_vec[None, :, None], (rx, ry, 2)).copy()
            dz_z = np.full((rx, ry, 2), dz_res)
            top_range = (k1 - 1, k1)
            bot_range = (k2, k2 + 1)

            self.level1_imag_z_top[name] = StructReservoir(self.timer, nx=rx, ny=ry, nz=2, dx=dx_z, dy=dy_z, dz=dz_z,
                                                  permx=self.make_lgr_array("permx", i_parent, j_parent, top_range, prop_shape),
                                                  permy=self.make_lgr_array("permy", i_parent, j_parent, top_range, prop_shape),
                                                  permz=self.make_lgr_array("permz", i_parent, j_parent, top_range, prop_shape),
                                                  poro=self.make_lgr_array("poro", i_parent, j_parent, top_range, prop_shape),
                                                  depth= None, start_z=reservoir_top - float(dz_over[-1]),
                                                  rcond=self.make_lgr_array("rcond", i_parent, j_parent, top_range, prop_shape),
                                                  hcap=self.make_lgr_array("hcap", i_parent, j_parent, top_range, prop_shape))

            self.level1_imag_z_bot[name] = StructReservoir(self.timer, nx=rx, ny=ry, nz=2, dx=dx_z, dy=dy_z, dz=dz_z,
                                                  permx=self.make_lgr_array("permx", i_parent, j_parent, bot_range, prop_shape),
                                                  permy=self.make_lgr_array("permy", i_parent, j_parent, bot_range, prop_shape),
                                                  permz=self.make_lgr_array("permz", i_parent, j_parent, bot_range, prop_shape),
                                                  poro=self.make_lgr_array("poro", i_parent, j_parent, bot_range, prop_shape),
                                                  depth= None, start_z=reservoir_top + (nz_res - 1) * dz_res,
                                                  rcond=self.make_lgr_array("rcond", i_parent, j_parent, bot_range, prop_shape),
                                                  hcap=self.make_lgr_array("hcap", i_parent, j_parent, bot_range, prop_shape))

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

        for name in meta['lgr_orders']:
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
            volume_list.append(dx_f * dy_f * dz_f)
            rcond_list.append(disc1.convert_to_flat_array(self.level1[name].global_data["rcond"], "rcond"))
            hcap_list.append(disc1.convert_to_flat_array(self.level1[name].global_data["hcap"], "hcap"))

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

        # print(f'final volume list is {volume}')

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

            for k in range(per_from -1, per_to):
                rx,ry,_ = self.lgrs[lgr_name]['lgr_coords_in_parent_grid']['refine']
                inj_local = center_2d + k * (rx * ry)
                inj_global = inj_local + self.lgr_meta['lgr_offsets'][comp[wname]["lgr"]]
                self.reservoir.add_perforation(wname, cell_index=inj_global, ms_epm=True)

        wat_wells = self.cfg.get("water_inj", {})
        for key, value in wat_wells.items():
            self.reservoir.add_well(key)
        # add four water injectors
        for wname, cfg in wat_wells.items():
            k_from = cfg["k_from"]
            k_to = cfg["k_to"]
            i0 = cfg["i0"]
            j0 = cfg["j0"]
            for k in range(self.nz_over + k_from , self.nz_over + k_to + 1):
                id0 = self.convert_ijk_to_gindex_1based(i0, j0, k, self.level0.nx, self.level0.ny)
                idx_global = self.level0.discretizer.global_to_local[id0]
                self.reservoir.add_perforation(wname, cell_index=idx_global, ms_epm=True)


    def set_physics(self):
        components = ['CO2', 'H2O']
        nc = len(components)
        self.components = components
        epsilon = self.zero / 10
        comp_data = CompData(components, setprops=True)
        phases = ['CO2_rich', 'aqueous']
        base_dir = os.path.dirname(os.path.abspath(__file__))
        pvt = os.path.join(base_dir, "physics.in")
        # pvt = 'physics.in'
        pr = CubicEoS(comp_data, CubicEoS.PR)
        aq = AQEoS(comp_data, {AQEoS.water: AQEoS.Jager2003,
                                AQEoS.solute: AQEoS.Ziabakhsh2012,
                                })
        # EoS-related parameters
        flash_params = FlashParams(comp_data)
        flash_params.add_eos("PR", pr)
        flash_params.add_eos("AQ", aq)
        flash_params.eos_order = ["PR", "AQ"]

        # Flash-related parameters
        flash_params.split_tol = 1e-12

        property_container = PropertyContainer(phases_name=phases, components_name=components,
                                               Mw=comp_data.Mw, eps_z=epsilon)

        """ properties correlations """
        table_path = r"K_values.csv"
        property_container.flash_ev = TableKFlash(len(components), table_path, self.zero)
        property_container.density_ev = dict([('CO2_rich', EoSDensity(eos=pr,Mw=comp_data.Mw)),
                                              ('aqueous', Garcia2001(components))])
        property_container.viscosity_ev = dict([('CO2_rich', Fenghour1998()),
                                                ('aqueous', Islam2012(components))])
        # property_container.rel_perm_ev = dict([('CO2_rich', PhaseRelPerm("gas", swc=0.30, sgr=0.1, kre=1.0, n=4.2)),
        #                                        ('aqueous', PhaseRelPerm("oil", swc=0.30, sgr=0.1, kre=1.0, n=1.9))])
        property_container.rel_perm_ev = dict([('CO2_rich', GasRelPerm(pvt)),
                                               ('aqueous', WatRelPerm(pvt))])
        property_container.enthalpy_ev = dict([('CO2_rich', EoSEnthalpy(eos=pr)),
                                                ('aqueous', EoSEnthalpy(eos=aq))])
        property_container.conductivity_ev = dict([('CO2_rich', ConstFunc(181.44)),
                                                   ('aqueous', ConstFunc(181.44)), ])
        """ Activate physics """
        thermal = True
        state_spec = Compositional.StateSpecification.PT if thermal else Compositional.StateSpecification.P
        self.physics = Compositional(components, phases, self.timer, state_spec=state_spec,
                                     n_points=400, min_p=1, max_p=1000, min_z=self.zero/10, max_z=1-self.zero/10,
                                     epsilon_z=epsilon, min_t=273.15, max_t=373.15+200)


        property_container.output_props = {
            "satG": lambda: property_container.sat[0],
            "XCO2": lambda: property_container.x[1, 1],
            "rhoG": lambda: property_container.dens[0],
            "rhoAq": lambda: property_container.dens[1],
            "muG": lambda: property_container.mu[0],
            "muAq": lambda: property_container.mu[1],
            }

        self.physics.add_property_region(property_container)

        return

    def set_initial_conditions(self):
        depths = np.asarray(self.reservoir.mesh.depth)
        min_depth = np.min(depths)
        max_depth = np.max(depths)
        nb = int(self.level0.nz)
        depths = np.linspace(min_depth,max_depth,nb)

        init = Initialize(self.physics)

        primary_specs = {}
        for comp in self.physics.components[:-1]:
            primary_specs[comp] = self.zero

        boundary_state = {"pressure" :195}
        for comp in self.physics.components[:-1]:
            boundary_state[comp] = primary_specs[comp]
        boundary_state["temperature"] = 80 + 273.15

        dTdh = 40/1000 #k/m

        X = init.solve_up_and_downwards(depth_bottom=max_depth, depth_top=min_depth, depth_known=2000,
                                        boundary_state=boundary_state, primary_specs=primary_specs, nb=nb,
                                        dTdh=dTdh)

        self.physics.set_initial_conditions_from_depth_table(mesh=self.reservoir.mesh,
                                                             input_depth= init.depths,
                                                            input_distribution={v:X[:,i] for i, v in enumerate(self.physics.vars)})
        return


    def set_well_controls(self):

        inj_composition = [1.0 - self.zero]  # pure CO2 injection
        for i, w in enumerate(self.reservoir.wells):
            if "I" in w.name:
                        # injector: MASS_RATE with BHP constraint
                        self.physics.set_well_controls(
                            wctrl=w.control,
                            control_type=well_control_iface.MASS_RATE,
                            # control_type=well_control_iface.VOLUMETRIC_RATE,
                            is_inj=True,
                            # target=4.32e4,
                            target=3.3264e7,
                            inj_composition=[1.0 - self.zero],
                            phase_name="CO2_rich",
                            inj_temp=314.15
                        )
                        self.physics.set_well_controls(
                            wctrl=w.constraint,
                            control_type=well_control_iface.BHP,
                            is_inj=True,
                            target=306.90,
                            inj_composition=[1.0 - self.zero],
                            inj_temp=314.15
                        )
            if "W" in w.name:
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.MASS_RATE,
                                                  is_inj=True, target=0,
                                                  inj_composition=[self.zero], phase_name="aqueous", inj_temp=288.15)
            if "P" in w.name:
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.MASS_RATE,
                                               is_inj=False, target=0
                                               )
