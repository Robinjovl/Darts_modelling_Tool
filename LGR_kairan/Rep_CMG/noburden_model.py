import os
from lgr_assemble import assemble_lgr_connections_eclipse, LGRReservoir
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
                            tol_newton=1e-3, tol_linear=1e-3,
                            it_newton=10, it_linear=50)

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
        grid = self.cfg["grid"]
        #Build Level0 with actnum + overburden and underburden
        nx0, ny0 = grid["nx"], grid["ny"] # global grid size
        dx0, dy0 = grid["dx"], grid["dy"]
        nz_res = grid["nz_res"]
        dz_res = grid["dz_res"]

        nz0 = nz_res
        dz0_layers = np.full(nz_res, dz_res, dtype=float)
        rock = self.cfg["rock"]
        permx0, permy0, permz0 = rock["perm_x"], rock["perm_y"], rock["perm_z"]
        poro0 = rock["poro"]

        # thermal properties
        rcond_res = rock["rcond_res"] # KJ/m/day/k
        hcap_res = rock["hcap_res"] # kJ/m3/K assume reservoir density here
        reservoir_top = grid["reservoir_top"]

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
        kx0_full = np.full(nx0 * ny0 * nz0, permx0, dtype=float)
        ky0_full = np.full(nx0 * ny0 * nz0, permy0, dtype=float)
        kz0_full = np.full(nx0 * ny0 * nz0, permz0, dtype=float)

        rcon0_full = np.full(nx0 * ny0 * nz0, rcond_res, dtype=float)
        hcap0_full = np.full(nx0 * ny0 * nz0, hcap_res, dtype=float)
        poro0_full = np.full(nx0 * ny0 * nz0, poro0, dtype=float)

        self.level0 = StructReservoir(self.timer, nx=nx0, ny=ny0, nz=nz0, dx=dx0, dy=dy0, dz=dz0_layers,
                                      permx=kx0_full, permy=ky0_full, permz=kz0_full, poro=poro0_full,depth= None,
                                      start_z=2000,actnum=actnum0, rcond=rcon0_full, hcap=hcap0_full,)
        boundary_factor = 2000
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
                                        permx=permx0, permy=permy0, permz=permz0, poro=poro0, depth= None, start_z=reservoir_top, rcond=rcond_res, hcap=hcap_res)

            dx_imag, dy_imag = self.auto_image_grid_dx_dy(dx0, dy0, rx, ry)

            # 2D imaginary grid per layer
            self.level1_imag[name] = StructReservoir(self.timer, nx=rx+2, ny=ry+2, nz=1, dx=dx_imag, dy=dy_imag, dz=dz_res,
                                        permx=permx0, permy=permy0, permz=permz0, poro=poro0, depth= None, start_z=reservoir_top,rcond=rcond_res, hcap=hcap_res,)


        cm_all, cp_all, T_all, T_all_therm, meta = assemble_lgr_connections_eclipse(self)
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
                self.reservoir.add_perforation(wname, cell_index=inj_global, ms_epm=False)

        wat_wells = self.cfg["water_inj"]
        for key, value in wat_wells.items():
            self.reservoir.add_well(key)
        # add four water injectors
        for wname, cfg in wat_wells.items():
            k_from = cfg["k_from"]
            k_to = cfg["k_to"]
            i0 = cfg["i0"]
            j0 = cfg["j0"]
            for k in range(k_from , k_to + 1):
                id0 = self.convert_ijk_to_gindex_1based(i0, j0, k, self.level0.nx, self.level0.ny)
                idx_global = self.level0.discretizer.global_to_local[id0]
                self.reservoir.add_perforation(wname, cell_index=idx_global, ms_epm=False)


    def set_physics(self):
        components = ['CO2', 'H2O']
        nc = len(components)
        self.components = components
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
                                               Mw=comp_data.Mw, min_z=self.zero / 10,)

        """ properties correlations """
        table_path = r"E:\repo_2\open-darts\LGR_kairan\Rep_CMG\K_values.csv"
        property_container.flash_ev = TableKFlash(len(components), table_path, self.zero)
        property_container.density_ev = dict([('CO2_rich', EoSDensity(eos=pr,Mw=comp_data.Mw)),
                                              ('aqueous', Garcia2001(components))])
        property_container.viscosity_ev = dict([('CO2_rich', Fenghour1998()),
                                                ('aqueous', Islam2012(components))])
        property_container.rel_perm_ev = dict([('CO2_rich', PhaseRelPerm("gas", swc=0.30, sgr=0.1, kre=1.0, n=4.2)),
                                               ('aqueous', PhaseRelPerm("oil", swc=0.30, sgr=0.1, kre=1.0, n=1.9))])
        # property_container.rel_perm_ev = dict([('CO2_rich', GasRelPerm(pvt)),
        #                                        ('aqueous', WatRelPerm(pvt))])
        property_container.enthalpy_ev = dict([('CO2_rich', EoSEnthalpy(eos=pr)),
                                                ('aqueous', EoSEnthalpy(eos=aq))])
        property_container.conductivity_ev = dict([('CO2_rich', ConstFunc(181.44)),
                                                   ('aqueous', ConstFunc(181.44)), ])
        """ Activate physics """
        thermal = True
        state_spec = Compositional.StateSpecification.PT if thermal else Compositional.StateSpecification.P
        self.physics = Compositional(components, phases, self.timer, state_spec=state_spec,
                                     n_points=400, min_p=1, max_p=1000, min_z=self.zero/10, max_z=1-self.zero/10,
                                     min_t=273.15, max_t=373.15+200)


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
        boundary_state["temperature"] = 80 +273.15

        dTdh = 40/1000 #k/m

        X = init.solve_up_and_downwards(depth_bottom=2200, depth_top=2000, depth_known=2000,
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
                            is_inj=True,
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
                                               ,phase_name="aqueous"
                                               )
