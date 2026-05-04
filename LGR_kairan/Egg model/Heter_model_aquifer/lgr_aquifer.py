from pathlib import Path
import sys

import numpy as np
import pandas as pd

EGG_DIR = Path(__file__).resolve().parent.parent
if str(EGG_DIR) not in sys.path:
    sys.path.insert(0, str(EGG_DIR))

from lgr_assemble import LGRReservoir, assemble_lgr_connections  # noqa: E402

from darts.engines import redirect_darts_output, well_control_iface
from darts.models.cicd_model import DartsModel
from darts.physics.properties.basic import ConstFunc, PhaseRelPerm
from darts.physics.properties.density import DensityBasic, Spivey2004, Garcia2001
from darts.physics.properties.eos_properties import EoSDensity, EoSEnthalpy
from darts.physics.properties.flash import Flash, RR2
from darts.physics.properties.viscosity import Fenghour1998, Islam2012
from darts.physics.super.initialize import Initialize
from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer
from darts.reservoirs.struct_reservoir import StructReservoir
from darts.tools.keyword_file_tools import load_single_keyword
from dartsflash.components import CompData
from dartsflash.libflash import AQEoS, CubicEoS

class TableKFlash(Flash):
    def __init__(self, nc, table_path, eps=1e-11):
        super().__init__(nph=2, nc=nc)
        self.rr_eps = eps
        df = pd.read_csv(table_path)
        self.p_axis = np.sort(df["P_bar"].unique())
        self.t_axis = np.sort(df["T_K"].unique())
        self.k_co2 = np.zeros((len(self.p_axis), len(self.t_axis)))
        self.k_h2o = np.zeros_like(self.k_co2)
        for i, p in enumerate(self.p_axis):
            df_p = df[df["P_bar"] == p].sort_values("T_K")
            self.k_co2[i, :] = df_p["K_CO2"].values
            self.k_h2o[i, :] = df_p["K_H2O"].values

    def evaluate(self, pressure, temperature, zc):
        self.K_values = self.get_k_values(pressure, temperature)
        self.nu, self.X = RR2(self.K_values, zc, self.rr_eps)
        self.temperature = temperature
        return 0

    def get_k_values(self, pressure, temperature):
        pi0, pi1, wp = self._bounds(self.p_axis, pressure)
        ti0, ti1, wt = self._bounds(self.t_axis, temperature)
        return np.array(
            [
                self._interp(self.k_co2, pi0, pi1, ti0, ti1, wp, wt),
                self._interp(self.k_h2o, pi0, pi1, ti0, ti1, wp, wt),
            ]
        )

    @staticmethod
    def _bounds(axis, value):
        i1 = np.searchsorted(axis, value)
        if i1 == 0:
            return 0, 0, 0.0
        if i1 >= len(axis):
            i = len(axis) - 1
            return i, i, 0.0
        i0 = i1 - 1
        return i0, i1, (value - axis[i0]) / (axis[i1] - axis[i0])

    @staticmethod
    def _interp(table, pi0, pi1, ti0, ti1, wp, wt):
        if pi0 == pi1 and ti0 == ti1:
            return float(table[pi0, ti0])
        if pi0 == pi1:
            return float(table[pi0, ti0] * (1.0 - wt) + table[pi0, ti1] * wt)
        if ti0 == ti1:
            return float(table[pi0, ti0] * (1.0 - wp) + table[pi1, ti0] * wp)
        return float(
            table[pi0, ti0] * (1.0 - wp) * (1.0 - wt)
            + table[pi1, ti0] * wp * (1.0 - wt)
            + table[pi0, ti1] * (1.0 - wp) * wt
            + table[pi1, ti1] * wp * wt
        )

class Model(DartsModel):
    def __init__(self, cfg: dict, perm_file_name: str):
        # Call base class constructor
        super().__init__()

        # Measure time spend on reading/initialization
        self.timer.node["initialization"].start()
        self.cfg = cfg
        self.perm_file_name = perm_file_name
        self.set_reservoir()
        self.zero = 1e-12
        self.set_physics()
        self.set_sim_params(
            first_ts=1e-6,
            mult_ts=2,
            max_ts=2,
            runtime=1000,
            tol_newton=1e-3,
            tol_linear=1e-3,
            it_newton=10,
            it_linear=50,
            well_rate_ctrl_absolute_residual_scale=1.0,
            well_rate_ctrl_relative_residual_scale=1e-5,
        )
        self.timer.node["initialization"].stop()

    def define_lgr(self):
        return self.cfg["lgrs"]

    def build_well_completion(self):
        return self.cfg["wells"]

    def _resolve_perm_file(self):
        perm_file = Path(self.perm_file_name)
        if perm_file.is_absolute():
            return perm_file

        candidates = [
            Path.cwd() / perm_file,
            Path(__file__).resolve().parent / perm_file,
            Path(__file__).resolve().parent.parent / "Heter_model_pure_co2" / perm_file,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def convert_ijk_to_gindex_1based(self, i, j, k, nx, ny):
        return (k - 1) * nx * ny + (j - 1) * nx + (i - 1)

    def ijk0_to_lin(self, i0: int, j0: int, k0: int, nx: int, ny: int) -> int:
        """Convert 0-based (i,j,k) to linear index"""
        return k0 * nx * ny + j0 * nx + i0

    def create_actnum_with_lgr(self, nx, ny, nz, refined_cells):
        actnum = np.ones(nx * ny * nz, dtype=np.int32)
        for i, j, k in refined_cells:
            actnum[self.convert_ijk_to_gindex_1based(i, j, k, nx, ny)] = 0
        return actnum

    def build_cell_center(self):
        n = self.reservoir.n
        x = np.empty(n)
        y = np.empty(n)
        z = np.asarray(self.reservoir.depth, dtype=float).copy()
        self.level0.discretize()
        l2g0 = np.asarray(self.level0.discretizer.local_to_global, dtype=int)
        nx0, ny0 = int(self.level0.nx), int(self.level0.ny)
        dx0 = float(np.asarray(self.level0.global_data["dx"]).flat[0])
        dy0 = float(np.asarray(self.level0.global_data["dy"]).flat[0])
        for local_id, global_id in enumerate(l2g0):
            j = (global_id % (nx0 * ny0)) // nx0
            i = global_id % nx0
            x[local_id] = (i + 0.5) * dx0
            y[local_id] = (j + 0.5) * dy0
        for name in self.lgr_meta["lgr_orders"]:
            grid = self.level1[name]
            offset = self.lgr_meta["lgr_offsets"][name]
            cfg = self.lgrs[name]["lgr_coords_in_parent_grid"]
            i0 = cfg["i_range"][0] - 1
            j0 = cfg["j_range"][0] - 1
            nxy = grid.nx * grid.ny
            for lid in range(grid.n):
                kk = lid // nxy
                rem = lid - kk * nxy
                jj = rem // grid.nx
                ii = rem % grid.nx
                x[offset + lid] = i0 * dx0 + np.sum(cfg["dx_vec"][:ii]) + 0.5 * cfg["dx_vec"][ii]
                y[offset + lid] = j0 * dy0 + np.sum(cfg["dy_vec"][:jj]) + 0.5 * cfg["dy_vec"][jj]
        self.reservoir.cell_center_x = x
        self.reservoir.cell_center_y = y
        self.reservoir.cell_center_z = z

    def level0_value(self, prop, i0, j0, k0):
        data = np.asarray(self.level0.global_data[prop], dtype=float)
        if data.ndim == 0:
            return float(data)
        nx, ny, nz = int(self.level0.nx), int(self.level0.ny), int(self.level0.nz)
        i = min(max(i0, 0), nx - 1)
        j = min(max(j0, 0), ny - 1)
        k = min(max(k0, 0), nz - 1)
        if data.ndim == 1:
            return float(data[k * nx * ny + j * nx + i])
        return float(data[i, j, k])

    def make_lgr_array(self, prop, i0, j0, k_range, shape_xy):
        rx, ry = shape_xy
        k1, k2 = k_range
        arr = np.empty((rx, ry, k2 - k1 + 1), dtype=float)
        for kk, k in enumerate(range(k1, k2 + 1)):
            arr[:, :, kk] = self.level0_value(prop, i0, j0, k - 1)
        return arr

    def make_lateral_imag_array(self, prop, i0, j0, k_range, shape_xy):
        rx, ry = shape_xy
        k1, k2 = k_range
        arr = np.empty((rx + 2, ry + 2, k2 - k1 + 1), dtype=float)
        for kk, k in enumerate(range(k1, k2 + 1)):
            pk = k - 1
            for ii in range(rx + 2):
                for jj in range(ry + 2):
                    di = -1 if ii == 0 else 1 if ii == rx + 1 else 0
                    dj = -1 if jj == 0 else 1 if jj == ry + 1 else 0
                    arr[ii, jj, kk] = self.level0_value(prop, i0 + di, j0 + dj, pk)
        return arr

    def set_reservoir(self):
        # set heterogeneous egg model
        self.lgrs = self.define_lgr()
        nx, ny, nz = self.cfg["reservoir"]["nx"], self.cfg["reservoir"]["ny"], self.cfg["reservoir"]["nz"]
        dx, dy, dz = self.cfg["reservoir"]["dx"], self.cfg["reservoir"]["dy"], self.cfg["reservoir"]["dz"]
        nb = nx * ny * nz
        nz_over = 1
        nz_res = 7
        nz_under = 1
        nb_res = nx * ny * nz_res

        perm_file = self._resolve_perm_file()
        permx_res = load_single_keyword(str(perm_file), "PERMX", nb_res)
        permy_res = load_single_keyword(str(perm_file), "PERMY", nb_res)
        permz_res = load_single_keyword(str(perm_file), "PERMZ", nb_res)

        burden = self.cfg["burden"]
        poro_burden = burden["poro_burden"]
        perm_burden = burden["perm_burden"]
        rcond_over, rcond_under = burden["rcond_over"], burden["rcond_under"]
        hcap_over, hcap_under = burden["hcap_over"], burden["hcap_under"]

        for cfg in self.lgrs.values():
            cfg["lgr_coords_in_parent_grid"]["k_range"] = [nz_over + 1, nz_over + nz_res]

        refined_cells_ijk = []
        for cfg in self.lgrs.values():
            c = cfg["lgr_coords_in_parent_grid"]
            i1, i2 = c["i_range"]
            j1, j2 = c["j_range"]
            assert i1 == i2 and j1 == j2, "This script only assume column LGR"
            for k in range(c["k_range"][0], c["k_range"][1] + 1):
                refined_cells_ijk.append((i1, j1, k))
        self.refined_cells_ijk = refined_cells_ijk

        k_index = np.arange(nb) // (nx * ny)
        assert nz == nz_over + nz_res + nz_under, (
            f"nz={nz} does not match nz_over+nz_res+nz_under={nz_over + nz_res + nz_under}"
        )
        mask_res = (k_index >= nz_over) & (k_index < nz_over + nz_res)
        mask_over = k_index < nz_over
        mask_under = k_index >= nz_over + nz_res

        kx = np.full(nb, perm_burden, dtype=float)
        ky = np.full(nb, perm_burden, dtype=float)
        kz = np.full(nb, perm_burden, dtype=float)
        poro = np.full(nb, poro_burden, dtype=float)
        rcond = np.empty(nb, dtype=float)
        hcap = np.empty(nb, dtype=float)

        permx_res = np.asarray(permx_res, dtype=float).reshape(nx * ny * nz_res, order="F")
        permy_res = np.asarray(permy_res, dtype=float).reshape(nx * ny * nz_res, order="F")
        permz_res = np.asarray(permz_res, dtype=float).reshape(nx * ny * nz_res, order="F")
        assert len(permx_res) == nb_res, f"PERMX length {len(permx_res)} != {nb_res}"
        assert len(permy_res) == nb_res, f"PERMY length {len(permy_res)} != {nb_res}"
        assert len(permz_res) == nb_res, f"PERMZ length {len(permz_res)} != {nb_res}"

        kx[mask_res] = permx_res.reshape(-1, order="F")
        ky[mask_res] = permy_res.reshape(-1, order="F")
        kz[mask_res] = permz_res.reshape(-1, order="F")
        poro[mask_res] = self.cfg["reservoir"]["poro"]
        rcond[mask_over], rcond[mask_res], rcond[mask_under] = rcond_over, 500.0, rcond_under
        hcap[mask_over], hcap[mask_res], hcap[mask_under] = hcap_over, 2200.0, hcap_under

        actnum = self.create_actnum_with_lgr(nx, ny, nz, refined_cells_ijk)
        self.level0 = StructReservoir(
            self.timer,
            nx=nx,
            ny=ny,
            nz=nz,
            dx=dx,
            dy=dy,
            dz=dz,
            permx=kx,
            permy=ky,
            permz=kz,
            poro=poro,
            depth=None,
            start_z=1990.0,
            rcond=rcond,
            hcap=hcap,
            actnum=actnum,
        )
        self.level0.boundary_volumes = {
            "xy_minus": 1e20,
            "xy_plus": 1e20,
            "yz_minus": None,
            "yz_plus": None,
            "xz_minus": None,
            "xz_plus": None,
        }

        self.level1, self.level1_imag, self.level1_imag_z_top, self.level1_imag_z_bot = {}, {}, {}, {}
        for name, entry in self.lgrs.items():
            c = entry["lgr_coords_in_parent_grid"]
            rx, ry, _ = c["refine"]
            k1, k2 = c["k_range"]
            nk = k2 - k1 + 1
            i_parent = c["i_range"][0] - 1
            j_parent = c["j_range"][0] - 1
            dx_vec = np.asarray(c["dx_vec"], dtype=float)
            dy_vec = np.asarray(c["dy_vec"], dtype=float)
            dx_f = np.broadcast_to(dx_vec[:, None, None], (rx, ry, nk)).copy()
            dy_f = np.broadcast_to(dy_vec[None, :, None], (rx, ry, nk)).copy()
            dz_f = np.full((rx, ry, nk), dz)
            prop_shape = (rx, ry)
            kx_f = self.make_lgr_array("permx", i_parent, j_parent, (k1, k2), prop_shape)
            ky_f = self.make_lgr_array("permy", i_parent, j_parent, (k1, k2), prop_shape)
            kz_f = self.make_lgr_array("permz", i_parent, j_parent, (k1, k2), prop_shape)
            self.level1[name] = StructReservoir(
                self.timer,
                nx=rx,
                ny=ry,
                nz=nk,
                dx=dx_f,
                dy=dy_f,
                dz=dz_f,
                permx=kx_f,
                permy=ky_f,
                permz=kz_f,
                poro=self.make_lgr_array("poro", i_parent, j_parent, (k1, k2), prop_shape),
                depth=None,
                start_z=2000.0,
                rcond=self.make_lgr_array("rcond", i_parent, j_parent, (k1, k2), prop_shape),
                hcap=self.make_lgr_array("hcap", i_parent, j_parent, (k1, k2), prop_shape),
            )

            dx_im = np.broadcast_to(np.r_[dx, dx_vec, dx][:, None, None], (rx + 2, ry + 2, nk)).copy()
            dy_im = np.broadcast_to(np.r_[dy, dy_vec, dy][None, :, None], (rx + 2, ry + 2, nk)).copy()
            dz_im = np.full((rx + 2, ry + 2, nk), dz)
            kx_im = self.make_lateral_imag_array("permx", i_parent, j_parent, (k1, k2), prop_shape)
            ky_im = self.make_lateral_imag_array("permy", i_parent, j_parent, (k1, k2), prop_shape)
            kz_im = self.make_lateral_imag_array("permz", i_parent, j_parent, (k1, k2), prop_shape)
            self.level1_imag[name] = StructReservoir(
                self.timer,
                nx=rx + 2,
                ny=ry + 2,
                nz=nk,
                dx=dx_im,
                dy=dy_im,
                dz=dz_im,
                permx=kx_im,
                permy=ky_im,
                permz=kz_im,
                poro=self.make_lateral_imag_array("poro", i_parent, j_parent, (k1, k2), prop_shape),
                depth=None,
                start_z=2000.0,
                rcond=self.make_lateral_imag_array("rcond", i_parent, j_parent, (k1, k2), prop_shape),
                hcap=self.make_lateral_imag_array("hcap", i_parent, j_parent, (k1, k2), prop_shape),
            )

            dx_z = np.broadcast_to(dx_vec[:, None, None], (rx, ry, 2)).copy()
            dy_z = np.broadcast_to(dy_vec[None, :, None], (rx, ry, 2)).copy()
            dz_z = np.full((rx, ry, 2), dz)
            top_range = (k1 - 1, k1)
            bot_range = (k2, k2 + 1)
            self.level1_imag_z_top[name] = StructReservoir(
                self.timer,
                nx=rx,
                ny=ry,
                nz=2,
                dx=dx_z,
                dy=dy_z,
                dz=dz_z,
                permx=self.make_lgr_array("permx", i_parent, j_parent, top_range, prop_shape),
                permy=self.make_lgr_array("permy", i_parent, j_parent, top_range, prop_shape),
                permz=self.make_lgr_array("permz", i_parent, j_parent, top_range, prop_shape),
                poro=self.make_lgr_array("poro", i_parent, j_parent, top_range, prop_shape),
                depth=None,
                start_z=1990.0,
                rcond=self.make_lgr_array("rcond", i_parent, j_parent, top_range, prop_shape),
                hcap=self.make_lgr_array("hcap", i_parent, j_parent, top_range, prop_shape),
            )
            self.level1_imag_z_bot[name] = StructReservoir(
                self.timer,
                nx=rx,
                ny=ry,
                nz=2,
                dx=dx_z,
                dy=dy_z,
                dz=dz_z,
                permx=self.make_lgr_array("permx", i_parent, j_parent, bot_range, prop_shape),
                permy=self.make_lgr_array("permy", i_parent, j_parent, bot_range, prop_shape),
                permz=self.make_lgr_array("permz", i_parent, j_parent, bot_range, prop_shape),
                poro=self.make_lgr_array("poro", i_parent, j_parent, bot_range, prop_shape),
                depth=None,
                start_z=2050.0,
                rcond=self.make_lgr_array("rcond", i_parent, j_parent, bot_range, prop_shape),
                hcap=self.make_lgr_array("hcap", i_parent, j_parent, bot_range, prop_shape),
            )

        cm, cp, tran, tran_thermal, meta = assemble_lgr_connections(self)
        self.lgr_meta = meta
        self.level0.discretize()
        disc0 = self.level0.discretizer
        l2g0 = np.asarray(disc0.local_to_global, dtype=int)
        dx_list = [disc0.convert_to_flat_array(self.level0.global_data["dx"], "dx")[l2g0]]
        dy_list = [disc0.convert_to_flat_array(self.level0.global_data["dy"], "dy")[l2g0]]
        dz_list = [disc0.convert_to_flat_array(self.level0.global_data["dz"], "dz")[l2g0]]
        depth_list = [disc0.convert_to_flat_array(self.level0.global_data["depth"], "depth")[l2g0]]
        volume_list = [np.asarray(self.level0.volume, dtype=float)]
        kx_list, ky_list, kz_list = [kx[l2g0]], [ky[l2g0]], [kz[l2g0]]
        poro_list, rcond_list, hcap_list = [poro[l2g0]], [rcond[l2g0]], [hcap[l2g0]]
        for name in meta["lgr_orders"]:
            self.level1[name].discretize()
            d = self.level1[name].discretizer
            dx_f = d.convert_to_flat_array(self.level1[name].global_data["dx"], "dx")
            dy_f = d.convert_to_flat_array(self.level1[name].global_data["dy"], "dy")
            dz_f = d.convert_to_flat_array(self.level1[name].global_data["dz"], "dz")
            dx_list.append(dx_f)
            dy_list.append(dy_f)
            dz_list.append(dz_f)
            depth_list.append(d.convert_to_flat_array(self.level1[name].global_data["depth"], "depth"))
            volume_list.append(dx_f * dy_f * dz_f)
            kx_list.append(d.convert_to_flat_array(self.level1[name].global_data["permx"], "permx"))
            ky_list.append(d.convert_to_flat_array(self.level1[name].global_data["permy"], "permy"))
            kz_list.append(d.convert_to_flat_array(self.level1[name].global_data["permz"], "permz"))
            poro_list.append(d.convert_to_flat_array(self.level1[name].global_data["poro"], "poro"))
            rcond_list.append(d.convert_to_flat_array(self.level1[name].global_data["rcond"], "rcond"))
            hcap_list.append(d.convert_to_flat_array(self.level1[name].global_data["hcap"], "hcap"))
        self.reservoir = LGRReservoir(
            self.timer,
            cell_m=cm,
            cell_p=cp,
            tran=tran,
            tran_thermal=tran_thermal,
            poro=np.concatenate(poro_list),
            rcond=np.concatenate(rcond_list),
            hcap=np.concatenate(hcap_list),
            depth=np.concatenate(depth_list),
            volume=np.concatenate(volume_list),
            dx=np.concatenate(dx_list),
            dy=np.concatenate(dy_list),
            dz=np.concatenate(dz_list),
            kx=np.concatenate(kx_list),
            ky=np.concatenate(ky_list),
            kz=np.concatenate(kz_list),
        )
        self.build_cell_center()

    def set_wells(self):
        well_completion = self.build_well_completion()
        for name in well_completion:
            self.reservoir.add_well(name)
        center = self.lgr_meta["well_local_center"]
        for wname, cfg in well_completion.items():
            lgr = cfg["lgr"]
            rx, ry, _ = self.lgrs[lgr]["lgr_coords_in_parent_grid"]["refine"]
            for k in range(cfg["k_from"] - 1, cfg["k_to"]):
                cell = self.lgr_meta["lgr_offsets"][lgr] + center + k * rx * ry
                self.reservoir.add_perforation(wname, cell_index=cell, ms_epm=True, well_radius=0.0762)

    def set_physics(self):
        components = ["CO2", "H2O"]
        phases = ["CO2_rich", "aqueous"]
        self.components = components
        eps = self.zero / 10.0
        comp_data = CompData(components, setprops=True)
        pr = CubicEoS(comp_data, CubicEoS.PR)
        aq = AQEoS(comp_data, {AQEoS.water: AQEoS.Jager2003, AQEoS.solute: AQEoS.Ziabakhsh2012})
        pc = PropertyContainer(phases_name=phases, components_name=components, Mw=comp_data.Mw, eps_z=eps)
        pc.flash_ev = TableKFlash(2, Path(__file__).resolve().parent / "K_values.csv", eps)
        pc.density_ev = {"CO2_rich": EoSDensity(eos=pr, Mw=comp_data.Mw),
                         "aqueous": Garcia2001(components)}
        pc.viscosity_ev = {"CO2_rich": Fenghour1998(),
                           "aqueous": Islam2012(components)}
        pc.rel_perm_ev = {
            "CO2_rich": PhaseRelPerm("gas", swc=0.30, sgr=0.10, kre=1.0, n=4.2),
            "aqueous": PhaseRelPerm("oil", swc=0.30, sgr=0.10, kre=1.0, n=1.9),
        }
        pc.enthalpy_ev = {"CO2_rich": EoSEnthalpy(eos=pr),
                          "aqueous": EoSEnthalpy(eos=aq)}
        pc.conductivity_ev = {"CO2_rich": ConstFunc(10),
                              "aqueous": ConstFunc(181.44)}
        self.physics = Compositional(
            components,
            phases,
            self.timer,
            state_spec=Compositional.StateSpecification.PT,
            n_points=400,
            min_p=1,
            max_p=1000,
            min_z=eps,
            max_z=1.0 - eps,
            epsilon_z=eps,
            min_t=273.15,
            max_t=573.15,
        )
        pc.output_props = {
            "satG": lambda: pc.sat[0],
            "rhoG": lambda: pc.dens[0],
            "rhoAq": lambda: pc.dens[1],
            "muG": lambda: pc.mu[0],
            "muAq": lambda: pc.mu[1],
        }
        self.physics.add_property_region(pc)

    def set_initial_conditions(self):
        depths = np.asarray(self.reservoir.mesh.depth)
        init = Initialize(self.physics)
        primary_specs = {"CO2": self.zero}
        boundary_state = {"pressure": 200.0, "CO2": self.zero, "temperature": 356.15}

        # Xi = [boundary_state['pressure']]
        # for c in init.physics.components[:-1]:
        #     Xi += [boundary_state[c]]
        # if 'temperature' in boundary_state.keys():
        #     Xi += [boundary_state['temperature']]

        # X0 = init.solve_state(Xi=Xi, specs=boundary_state)
        # # Initialize depth table
        # X, bc_idx = init.init_depth_table(
        #     depth_bottom=float(np.max(depths)),
        #     depth_top=float(np.min(depths)),
        #     depth_known=2000,
        #     X0=X0,
        #     nb=int(self.level0.nz),
        #     dTdh=34/1000,
        # )
        # # Solve vertical equilibrium
        # X = init.solve(
        #     X=X, bc_idx=bc_idx, specs=primary_specs, downward=False
        # )  # solve above
        # X = init.solve(
        #     X=X, bc_idx=bc_idx, specs=primary_specs, downward=True
        # )  # solve below

        # x = X
        x = init.solve_up_and_downwards(
            depth_bottom=float(np.max(depths)),
            depth_top=float(np.min(depths)),
            depth_known=2000.0,
            boundary_state=boundary_state,
            primary_specs=primary_specs,
            nb=int(self.level0.nz),
            dTdh=34.0 / 1000.0,
        )
        self.physics.set_initial_conditions_from_depth_table(
            mesh=self.reservoir.mesh,
            input_depth=init.depths,
            input_distribution={v: x[:, i] for i, v in enumerate(self.physics.vars)},
        )

    def set_well_controls(self):
        for well in self.reservoir.wells:
            if well.name.startswith("I"):
                self.physics.set_well_controls(
                    wctrl=well.control,
                    control_type=well_control_iface.MASS_RATE,
                    is_inj=True,
                    target=4.32e6,
                    phase_name="CO2_rich",
                    inj_composition=[1.0 - self.zero],
                    inj_temp=40+273.15,
                )
                self.physics.set_well_controls(
                    wctrl=well.constraint,
                    control_type=well_control_iface.BHP,
                    is_inj=True,
                    target=300.0,
                    phase_name="CO2_rich",
                    inj_composition=[1.0 - self.zero],
                    inj_temp=40+273.15,
                )
            else:
                self.physics.set_well_controls(
                    wctrl=well.control,
                    control_type=well_control_iface.BHP,
                    is_inj=False,
                    target=190.0,
                )
