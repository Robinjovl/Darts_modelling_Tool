import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


def prolongate_piecewise(arr_2d_coarse: np.ndarray, refine_x: int, refine_y: int) -> np.ndarray:
    out = np.repeat(arr_2d_coarse, refine_x, axis=0)
    return np.repeat(out, refine_y, axis=1)


def build_patch_perm_from_egg(
    perm_file: str,
    center_ij_1b=(42, 30),
    coarse_patch_size=5,
    refine=(5, 5),
    layer_1b=7,
):
    nx_c, ny_c, nz_c = 60, 60, 7
    nb_res = nx_c * ny_c * nz_c

    permx = np.asarray(load_single_keyword(str(perm_file), "PERMX", nb_res), dtype=float)
    permy = np.asarray(load_single_keyword(str(perm_file), "PERMY", nb_res), dtype=float)
    permz = np.asarray(load_single_keyword(str(perm_file), "PERMZ", nb_res), dtype=float)

    permx = permx.reshape((nx_c, ny_c, nz_c), order="F")
    permy = permy.reshape((nx_c, ny_c, nz_c), order="F")
    permz = permz.reshape((nx_c, ny_c, nz_c), order="F")

    ic_1b, jc_1b = center_ij_1b
    k0 = layer_1b - 1
    half = coarse_patch_size // 2

    i0_min = ic_1b - 1 - half
    i0_max = ic_1b - 1 + half
    j0_min = jc_1b - 1 - half
    j0_max = jc_1b - 1 + half

    if i0_min < 0 or j0_min < 0 or i0_max >= nx_c or j0_max >= ny_c:
        raise ValueError("Requested coarse patch exceeds Egg grid boundary.")

    kx_patch_c = permx[i0_min:i0_max + 1, j0_min:j0_max + 1, k0]
    ky_patch_c = permy[i0_min:i0_max + 1, j0_min:j0_max + 1, k0]
    kz_patch_c = permz[i0_min:i0_max + 1, j0_min:j0_max + 1, k0]

    rx, ry = refine
    kx_f = prolongate_piecewise(kx_patch_c, rx, ry)[:, :, None]
    ky_f = prolongate_piecewise(ky_patch_c, rx, ry)[:, :, None]
    kz_f = prolongate_piecewise(kz_patch_c, rx, ry)[:, :, None]
    return kx_f, ky_f, kz_f


# class Garcia2001(Spivey2004):
#     def __init__(self, components, ions=None, combined_ions=None):
#         super().__init__(components, ions, combined_ions)
#         self.CO2_idx = components.index("CO2")

#     def evaluate(self, pressure, temperature, x):
#         rho_b = DensityBasic(dens0=1020.0, compr=4.5e-5, p0=1.01325).evaluate(pressure, temperature, x)
#         tc = temperature - 273.15
#         v_app = (37.51 - 9.585e-2 * tc + 8.740e-4 * tc**2 - 5.044e-7 * tc**3) * 1e-6
#         m_co2 = 55.509 * x[self.CO2_idx] / x[self.H2O_idx]
#         return (1.0 + m_co2 * 44.01e-3) / (m_co2 * v_app + 1.0 / rho_b)


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
        return np.array([
            self._interp(self.k_co2, pi0, pi1, ti0, ti1, wp, wt),
            self._interp(self.k_h2o, pi0, pi1, ti0, ti1, wp, wt),
        ])

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


class SinglePhaseCO2Properties(PropertyContainer):
    def __init__(self, phases_name, components_name, eps_z, Mw):
        super().__init__(phases_name, components_name, Mw, eps_z=eps_z, temperature=None)

    def evaluate(self, state):
        state_np = np.asarray(state, dtype=float)
        self.pressure = state_np[0]
        self.temperature = state_np[-1] if self.thermal else self.temperature
        zc = np.append(state_np[1:self.nc], 1.0 - np.sum(state_np[1:self.nc]))

        self.clean_arrays()
        self.x[0, :] = zc
        self.ph = np.array([0], dtype=np.intp)
        m = np.sum(self.x[0, :] * self.Mw)
        self.dens[0] = self.density_ev[self.phases_name[0]].evaluate(self.pressure, self.temperature, [1.0])
        self.dens_m[0] = self.dens[0] / m
        self.mu[0] = self.viscosity_ev[self.phases_name[0]].evaluate(
            pressure=self.pressure,
            temperature=self.temperature,
            x=[1.0],
            rho=self.dens[0],
        )
        self.sat[0] = 1.0
        self.kr[0] = 1.0
        self.pc[0] = 0.0


class PatchBaseModel(DartsModel):
    def __init__(
        self,
        perm_file: str,
        egg_center_ij_1b=(42, 30),
        layer_1b=7,
        injection_rate=1000.0,
        output_name="patch",
    ):
        super().__init__()
        self.perm_file = perm_file
        self.egg_center_ij_1b = egg_center_ij_1b
        self.layer_1b = int(layer_1b)
        self.injection_rate = float(injection_rate)
        self.output_name = output_name

        self.nx = 25
        self.ny = 25
        self.nz = 1
        self.dx = 6.0
        self.dy = 6.0
        self.dz = 10.0
        self.patch_size = 5
        self.patch_center_1b = (13, 13)
        self.zero = 1e-8

        self.timer.node["initialization"].start()
        self.set_reservoir()
        self.set_physics()
        self.set_sim_params(
            first_ts=1e-6,
            mult_ts=2.0,
            max_ts=30.0,
            runtime=1000.0,
            tol_newton=1e-3,
            tol_linear=1e-3,
            it_newton=10,
            it_linear=50,
            well_rate_ctrl_absolute_residual_scale=1.0,
            well_rate_ctrl_relative_residual_scale=1e-5,
        )
        self.timer.node["initialization"].stop()

    def set_reservoir(self):
        kx, ky, kz = build_patch_perm_from_egg(
            perm_file=self.perm_file,
            center_ij_1b=self.egg_center_ij_1b,
            coarse_patch_size=5,
            refine=(5, 5),
            layer_1b=self.layer_1b,
        )
        self.reservoir = StructReservoir(
            self.timer,
            nx=self.nx,
            ny=self.ny,
            nz=self.nz,
            dx=self.dx,
            dy=self.dy,
            dz=self.dz,
            permx=kx,
            permy=ky,
            permz=kz,
            poro=0.2,
            depth=None,
            start_z=2000.0,
            rcond=181.44,
            hcap=2200.0,
        )
        self.reservoir.discretize()
        self._build_cell_centers()

    def _build_cell_centers(self):
        n = self.reservoir.n
        x = np.empty(n, dtype=float)
        y = np.empty(n, dtype=float)
        z = np.asarray(self.reservoir.global_data["depth"], dtype=float).copy()
        for g in range(n):
            j = (g % (self.nx * self.ny)) // self.nx
            i = g % self.nx
            x[g] = (i + 0.5) * self.dx
            y[g] = (j + 0.5) * self.dy
        self.reservoir.cell_center_x = x
        self.reservoir.cell_center_y = y
        self.reservoir.cell_center_z = z

    def set_wells(self):
        self.reservoir.add_well("I1")
        self.reservoir.add_perforation(
            "I1",
            res_cell_idx=(self.patch_center_1b[0], self.patch_center_1b[1], 1),
            well_diameter=0.1524,
        )


class TwoPhaseAquiferPatchModel(PatchBaseModel):
    def set_physics(self):
        components = ["CO2", "H2O"]
        phases = ["CO2_rich", "aqueous"]
        self.components = components
        eps = self.zero / 10.0
        comp_data = CompData(components, setprops=True)
        pr = CubicEoS(comp_data, CubicEoS.PR)
        aq = AQEoS(comp_data, {AQEoS.water: AQEoS.Jager2003, AQEoS.solute: AQEoS.Ziabakhsh2012})

        pc = PropertyContainer(phases_name=phases, components_name=components, Mw=comp_data.Mw, eps_z=eps)
        pc.flash_ev = TableKFlash(2, Path(__file__).resolve().parents[1] / "Rep_CMG" / "K_values.csv", eps)
        pc.density_ev = {"CO2_rich": EoSDensity(eos=pr, Mw=comp_data.Mw), "aqueous": Garcia2001(components)}
        pc.viscosity_ev = {"CO2_rich": Fenghour1998(), "aqueous": Islam2012(components)}
        pc.rel_perm_ev = {
            "CO2_rich": PhaseRelPerm("gas", swc=0.30, sgr=0.10, kre=1.0, n=4.2),
            "aqueous": PhaseRelPerm("oil", swc=0.30, sgr=0.10, kre=1.0, n=1.9),
        }
        pc.enthalpy_ev = {"CO2_rich": EoSEnthalpy(eos=pr), "aqueous": EoSEnthalpy(eos=aq)}
        pc.conductivity_ev = {"CO2_rich": ConstFunc(181.44), "aqueous": ConstFunc(181.44)}
        pc.output_props = {
            "satG": lambda: pc.sat[0],
            "XCO2_aq": lambda: pc.x[1, 0],
            "rhoG": lambda: pc.dens[0],
            "rhoAq": lambda: pc.dens[1],
            "muG": lambda: pc.mu[0],
            "muAq": lambda: pc.mu[1],
            "krG": lambda: pc.kr[0],
            "krAq": lambda: pc.kr[1],
            "temperature": lambda: pc.temperature,
        }

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
        self.physics.add_property_region(pc)

    def set_initial_conditions(self):
        depths = np.asarray(self.reservoir.mesh.depth)
        init = Initialize(self.physics)
        primary_specs = {"CO2": self.zero}
        boundary_state = {"pressure": 195.0, "CO2": self.zero, "temperature": 356.15}
        x = init.solve_up_and_downwards(
            depth_bottom=float(np.max(depths)),
            depth_top=float(np.min(depths)),
            depth_known=2000.0,
            boundary_state=boundary_state,
            primary_specs=primary_specs,
            nb=int(self.reservoir.nz),
            dTdh=34.0 / 1000.0,
        )
        self.physics.set_initial_conditions_from_depth_table(
            mesh=self.reservoir.mesh,
            input_depth=init.depths,
            input_distribution={v: x[:, i] for i, v in enumerate(self.physics.vars)},
        )

    def set_well_controls(self):
        for well in self.reservoir.wells:
            self.physics.set_well_controls(
                wctrl=well.control,
                control_type=well_control_iface.MASS_RATE,
                is_inj=True,
                target=self.injection_rate,
                phase_name="CO2_rich",
                inj_composition=[1.0 - self.zero],
                inj_temp=40+273.15,
            )



class SinglePhaseCO2PatchModel(PatchBaseModel):
    def set_physics(self):
        components = ["CO2"]
        phases = ["CO2_rich"]
        eps = self.zero / 10.0
        comp_data = CompData(components, setprops=True)
        pr = CubicEoS(comp_data, CubicEoS.PR)

        pc = SinglePhaseCO2Properties(
            phases_name=phases,
            components_name=components,
            eps_z=eps,
            Mw=comp_data.Mw,
        )
        pc.density_ev = {"CO2_rich": EoSDensity(eos=pr, Mw=comp_data.Mw)}
        pc.viscosity_ev = {"CO2_rich": Fenghour1998()}
        pc.enthalpy_ev = {"CO2_rich": EoSEnthalpy(eos=pr)}
        pc.conductivity_ev = {"CO2_rich": ConstFunc(181.44)}
        pc.output_props = {
            "rhoG": lambda: pc.dens[0],
            "muG": lambda: pc.mu[0],
            "temperature": lambda: pc.temperature,
        }

        self.physics = Compositional(
            components=components,
            phases=phases,
            timer=self.timer,
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
        self.physics.add_property_region(pc)

    def set_initial_conditions(self):
        depths = np.asarray(self.reservoir.mesh.depth)
        init = Initialize(self.physics)
        boundary_state = {"pressure": 195.0, "temperature": 356.15}
        primary_specs = {}
        x = init.solve_up_and_downwards(
            depth_bottom=float(np.max(depths)),
            depth_top=float(np.min(depths)),
            depth_known=2000.0,
            boundary_state=boundary_state,
            primary_specs=primary_specs,
            nb=int(self.reservoir.nz),
            dTdh=34.0 / 1000.0,
        )
        self.physics.set_initial_conditions_from_depth_table(
            mesh=self.reservoir.mesh,
            input_depth=init.depths,
            input_distribution={v: x[:, i] for i, v in enumerate(self.physics.vars)},
        )

    def set_well_controls(self):
        for well in self.reservoir.wells:
            self.physics.set_well_controls(
                wctrl=well.control,
                control_type=well_control_iface.MASS_RATE,
                is_inj=True,
                target=self.injection_rate,
                inj_composition=[1.0],
                inj_temp=40+273.15,
            )


class PhaseEffectiveTransAnalyzer:
    VALID_SIDES = ("left", "right", "up", "down")

    def __init__(self, model, nx, ny, nz, patch_size=5, patch_center_1b=(13, 13), n_nb_cols=5):
        self.model = model
        self.nx = int(nx)
        self.ny = int(ny)
        self.nz = int(nz)
        self.patch_size = int(patch_size)
        self.patch_center_1b = tuple(patch_center_1b)
        self.n_nb_cols = int(n_nb_cols)
        self.phase_names = list(model.physics.property_containers[0].phases_name)

    def lin_index(self, i0, j0, k0):
        return i0 + j0 * self.nx + k0 * self.nx * self.ny

    def _patch_bounds_0b(self):
        ic = self.patch_center_1b[0] - 1
        jc = self.patch_center_1b[1] - 1
        half = self.patch_size // 2
        return ic - half, ic + half, jc - half, jc + half

    def get_connection_df(self):
        mesh = self.model.reservoir.mesh
        return pd.DataFrame({
            "block_m": np.asarray(mesh.block_m, dtype=int),
            "block_p": np.asarray(mesh.block_p, dtype=int),
            "tran": np.asarray(mesh.tran, dtype=float),
        })

    def get_state_array(self):
        n_vars = len(self.model.physics.vars)
        x = np.asarray(self.model.physics.engine.X, dtype=float).reshape((-1, n_vars))
        return x[: self.model.reservoir.mesh.n_res_blocks, :].copy()

    def eval_phase_props(self, cell_idx):
        states = self.get_state_array()
        pc = self.model.physics.property_containers[0]
        pc.evaluate(states[cell_idx, :].copy())
        out = {}
        for ip, phase in enumerate(self.phase_names):
            out[phase] = {
                "mu": float(pc.mu[ip]),
                "kr": float(pc.kr[ip]),
                "sat": float(pc.sat[ip]),
                "mobility": float(pc.kr[ip] / pc.mu[ip]) if pc.mu[ip] > 0 else np.nan,
            }
        return out

    def interface_cells(self, side, k0=0):
        i0_min, i0_max, j0_min, j0_max = self._patch_bounds_0b()
        if side == "left":
            return [self.lin_index(i0_min, j0, k0) for j0 in range(j0_min, j0_max + 1)]
        if side == "right":
            return [self.lin_index(i0_max, j0, k0) for j0 in range(j0_min, j0_max + 1)]
        if side == "up":
            return [self.lin_index(i0, j0_min, k0) for i0 in range(i0_min, i0_max + 1)]
        if side == "down":
            return [self.lin_index(i0, j0_max, k0) for i0 in range(i0_min, i0_max + 1)]
        raise ValueError(f"Invalid side: {side}")

    def immediate_neighbor_cells(self, side, k0=0):
        if_cells = self.interface_cells(side, k0)
        if side == "left":
            return [c - 1 for c in if_cells]
        if side == "right":
            return [c + 1 for c in if_cells]
        if side == "up":
            return [c - self.nx for c in if_cells]
        if side == "down":
            return [c + self.nx for c in if_cells]
        raise ValueError(f"Invalid side: {side}")

    def neighbor_support_cells(self, side, k0=0):
        i0_min, i0_max, j0_min, j0_max = self._patch_bounds_0b()
        s = self.n_nb_cols
        ids = []
        if side == "left":
            for j0 in range(j0_min, j0_max + 1):
                for i0 in range(i0_min - s, i0_min):
                    ids.append(self.lin_index(i0, j0, k0))
        elif side == "right":
            for j0 in range(j0_min, j0_max + 1):
                for i0 in range(i0_max + 1, i0_max + 1 + s):
                    ids.append(self.lin_index(i0, j0, k0))
        elif side == "up":
            for j0 in range(j0_min - s, j0_min):
                for i0 in range(i0_min, i0_max + 1):
                    ids.append(self.lin_index(i0, j0, k0))
        elif side == "down":
            for j0 in range(j0_max + 1, j0_max + 1 + s):
                for i0 in range(i0_min, i0_max + 1):
                    ids.append(self.lin_index(i0, j0, k0))
        else:
            raise ValueError(f"Invalid side: {side}")
        return ids

    @staticmethod
    def find_connection(cell_a, cell_b, conn_df):
        mask = (
            ((conn_df["block_m"] == cell_a) & (conn_df["block_p"] == cell_b))
            | ((conn_df["block_m"] == cell_b) & (conn_df["block_p"] == cell_a))
        )
        return conn_df.loc[mask]

    def face_effective_trans(self, side, k0=0, mobility_mode="interface_avg", eps_mobility=1e-30):
        conn_df = self.get_connection_df()
        states = self.get_state_array()
        pressure = states[:, 0]

        if_cells = self.interface_cells(side, k0)
        nb_cells = self.immediate_neighbor_cells(side, k0)
        nb_support = self.neighbor_support_cells(side, k0)

        p_if_avg = float(np.mean(pressure[if_cells]))
        p_nb_avg = float(np.mean(pressure[nb_support]))
        dp_macro = p_if_avg - p_nb_avg
        if abs(dp_macro) < 1e-14:
            raise ZeroDivisionError(f"dp_macro too small on side={side}")

        accum = {
            phase: {"flux": 0.0, "mobility_refs": [], "link_rows": []}
            for phase in self.phase_names
        }

        for c_if, c_nb in zip(if_cells, nb_cells):
            hit = self.find_connection(c_if, c_nb, conn_df)
            if hit.empty:
                raise RuntimeError(f"No connection found between {c_if} and {c_nb}")

            tran = float(hit["tran"].iloc[0])
            p_if = float(pressure[c_if])
            p_nb = float(pressure[c_nb])
            dp_link = p_if - p_nb
            props_if = self.eval_phase_props(c_if)
            props_nb = self.eval_phase_props(c_nb)

            for phase in self.phase_names:
                mob_if = props_if[phase]["mobility"]
                mob_nb = props_nb[phase]["mobility"]
                if mobility_mode == "interface":
                    mob_ref_link = mob_if
                elif mobility_mode == "neighbor":
                    mob_ref_link = mob_nb
                elif mobility_mode == "interface_avg":
                    mob_ref_link = 0.5 * (mob_if + mob_nb)
                else:
                    raise ValueError(f"Unknown mobility_mode: {mobility_mode}")

                q_link = -tran * dp_link * mob_ref_link
                accum[phase]["flux"] += q_link
                accum[phase]["mobility_refs"].append(mob_ref_link)
                accum[phase]["link_rows"].append({
                    "side": side,
                    "phase": phase,
                    "cell_if": c_if,
                    "cell_nb": c_nb,
                    "tran": tran,
                    "p_if": p_if,
                    "p_nb": p_nb,
                    "dp_link": dp_link,
                    "mobility_if": mob_if,
                    "mobility_nb": mob_nb,
                    "mobility_ref_link": mob_ref_link,
                    "kr_if": props_if[phase]["kr"],
                    "kr_nb": props_nb[phase]["kr"],
                    "mu_if": props_if[phase]["mu"],
                    "mu_nb": props_nb[phase]["mu"],
                    "sat_if": props_if[phase]["sat"],
                    "sat_nb": props_nb[phase]["sat"],
                    "flux": q_link,
                })

        rows = []
        detail = {}
        for phase, data in accum.items():
            mobility_ref = float(np.mean(data["mobility_refs"]))
            total_flux = float(data["flux"])
            if abs(mobility_ref) < eps_mobility:
                t_eff_total = np.nan
                t_eff_link = np.nan
            else:
                t_eff_total = -total_flux / (mobility_ref * dp_macro)
                t_eff_link = t_eff_total / len(data["link_rows"])

            row = {
                "side": side,
                "phase": phase,
                "n_links": len(data["link_rows"]),
                "p_if_avg": p_if_avg,
                "p_nb_avg": p_nb_avg,
                "dp_macro": dp_macro,
                "mobility_ref": mobility_ref,
                "total_flux": total_flux,
                "T_eff_total": t_eff_total,
                "T_eff_per_link": t_eff_link,
            }
            rows.append(row)
            detail[phase] = {
                **row,
                "link_df": pd.DataFrame(data["link_rows"]),
            }

        return {"summary_df": pd.DataFrame(rows), "phases": detail}

    def all_faces(self, k0=0, mobility_mode="interface_avg"):
        rows = []
        detail = {}
        for side in self.VALID_SIDES:
            out = self.face_effective_trans(side=side, k0=k0, mobility_mode=mobility_mode)
            rows.append(out["summary_df"])
            detail[side] = out["phases"]
        return {"summary_df": pd.concat(rows, axis=0, ignore_index=True), "faces": detail}


def plot_phase_trans_history(df, save_dir, filename, title):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(9, 5), dpi=150)
    for phase in sorted(df["phase"].unique()):
        for side in ["left", "right", "up", "down"]:
            dfi = df[(df["phase"] == phase) & (df["side"] == side)]
            plt.plot(dfi["time_day"], dfi["T_eff_per_link"], label=f"{phase}-{side}")
    plt.xlabel("time [day]")
    plt.ylabel("phase effective transmissibility per equivalent link")
    plt.title(title)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()


def plot_phase_single_comparison(df, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for phase in sorted(df["phase"].unique()):
        plt.figure(figsize=(9, 5), dpi=150)
        for side in ["left", "right", "up", "down"]:
            dfi = df[(df["phase"] == phase) & (df["side"] == side)]
            plt.plot(dfi["time_day"], dfi["ratio_vs_single"], label=side)
        plt.xlabel("time [day]")
        plt.ylabel("T_phase / T_single_phase")
        plt.title(f"{phase} effective transmissibility ratio to single-phase")
        plt.legend()
        plt.tight_layout()
        safe_phase = phase.replace("/", "_")
        plt.savefig(os.path.join(save_dir, f"{safe_phase}_ratio_vs_single_phase.png"))
        plt.close()


def get_reservoir_average_pressure(model):
    n_res = model.reservoir.mesh.n_res_blocks
    n_vars = len(model.physics.vars)
    x = np.asarray(model.physics.engine.X, dtype=float).reshape((-1, n_vars))
    pressure = x[:n_res, 0]
    poro = np.asarray(model.reservoir.mesh.poro, dtype=float)[:n_res]
    volume = np.asarray(model.reservoir.mesh.volume, dtype=float)[:n_res]
    pv = poro * volume
    return float(np.sum(pressure * pv) / np.sum(pv))


def build_pressure_history(model, avg_pressure_df):
    time_data = pd.DataFrame.from_dict(model.output.store_well_time_data(save_output_files=True))
    bhp_col = "well_I1_BHP"
    if bhp_col not in time_data.columns:
        candidates = [c for c in time_data.columns if "BHP" in c]
        raise KeyError(f"{bhp_col} not found in well time data. BHP candidates: {candidates}")

    bhp_df = (
        time_data[["time", bhp_col]]
        .rename(columns={"time": "time_day", bhp_col: "bhp_bar"})
        .sort_values("time_day")
        .reset_index(drop=True)
    )
    out = pd.merge_asof(
        avg_pressure_df.sort_values("time_day").reset_index(drop=True),
        bhp_df,
        on="time_day",
        direction="nearest",
        tolerance=1e-6,
    )
    out["bhp_minus_pavg"] = out["bhp_bar"] - out["avg_pressure"]
    return out


def plot_pressure_history(pressure_hist, save_dir, phase_label):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(8, 5), dpi=150)
    plt.plot(pressure_hist["time_day"], pressure_hist["bhp_bar"], label="Injector BHP")
    plt.plot(pressure_hist["time_day"], pressure_hist["avg_pressure"], label="PV-average pressure")
    plt.xlabel("time [day]")
    plt.ylabel("pressure [bar]")
    plt.title(f"{phase_label}: BHP and reservoir average pressure")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{phase_label}_bhp_and_avg_pressure.png"))
    plt.close()

    plt.figure(figsize=(8, 5), dpi=150)
    plt.plot(pressure_hist["time_day"], pressure_hist["bhp_minus_pavg"])
    plt.xlabel("time [day]")
    plt.ylabel("BHP - Pavg [bar]")
    plt.title(f"{phase_label}: injection pressure buildup")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{phase_label}_bhp_minus_pavg.png"))
    plt.close()


def plot_two_phase_saturation_xy(model, save_dir, phase_label):
    os.makedirs(save_dir, exist_ok=True)
    pc = model.physics.property_containers[0]
    if len(pc.phases_name) < 2:
        return

    n_res = model.reservoir.mesh.n_res_blocks
    n_vars = len(model.physics.vars)
    states = np.asarray(model.physics.engine.X, dtype=float).reshape((-1, n_vars))[:n_res, :]

    sat = np.empty((n_res, len(pc.phases_name)), dtype=float)
    for cell_idx in range(n_res):
        pc.evaluate(states[cell_idx, :].copy())
        sat[cell_idx, :] = np.asarray(pc.sat[: len(pc.phases_name)], dtype=float)

    for iph, phase in enumerate(pc.phases_name):
        arr = sat[:, iph].reshape((model.nx, model.ny), order="F")
        plt.figure(figsize=(6, 5), dpi=150)
        im = plt.imshow(
            arr.T,
            origin="lower",
            extent=[0.0, model.nx * model.dx, 0.0, model.ny * model.dy],
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
            aspect="equal",
        )
        plt.colorbar(im, label=f"{phase} saturation")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.title(f"{phase_label}: final {phase} saturation")

        half = model.patch_size // 2
        i0 = model.patch_center_1b[0] - 1
        j0 = model.patch_center_1b[1] - 1
        x0 = (i0 - half) * model.dx
        y0 = (j0 - half) * model.dy
        rect = plt.Rectangle(
            (x0, y0),
            model.patch_size * model.dx,
            model.patch_size * model.dy,
            fill=False,
            edgecolor="white",
            linewidth=1.5,
        )
        plt.gca().add_patch(rect)
        safe_phase = phase.replace("/", "_")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{phase_label}_final_{safe_phase}_saturation_xy.png"))
        plt.close()


def compare_with_single_phase(two_phase_hist, single_phase_hist):
    sp = single_phase_hist[["time_day", "side", "T_eff_per_link"]].rename(
        columns={"T_eff_per_link": "T_eff_single_phase_per_link"}
    )
    out = two_phase_hist.merge(sp, on=["time_day", "side"], how="left")
    out["delta_vs_single"] = out["T_eff_per_link"] - out["T_eff_single_phase_per_link"]
    out["ratio_vs_single"] = out["T_eff_per_link"] / out["T_eff_single_phase_per_link"]
    return out


def run_one_model(model, output_dir, nt, dt, phase_label):
    redirect_darts_output(os.path.join(output_dir, f"{phase_label}.log"))
    model.init(platform="cpu")
    model.set_output(output_folder=os.path.join(output_dir, phase_label))
    analyzer = PhaseEffectiveTransAnalyzer(
        model=model,
        nx=model.nx,
        ny=model.ny,
        nz=model.nz,
        patch_size=model.patch_size,
        patch_center_1b=model.patch_center_1b,
        n_nb_cols=5,
    )

    rows = []
    pressure_rows = []
    for _ in range(nt):
        model.run(dt)
        t_end = float(model.physics.engine.t)
        pressure_rows.append({
            "time_day": t_end,
            "avg_pressure": get_reservoir_average_pressure(model),
        })
        res = analyzer.all_faces(k0=0, mobility_mode="interface_avg")
        dfi = res["summary_df"].copy()
        dfi["time_day"] = t_end
        rows.append(dfi)
        print(f"\n[{phase_label}] t={model.physics.engine.t}")
        print(dfi)

    avg_pressure_df = pd.DataFrame(pressure_rows)
    pressure_hist = build_pressure_history(model, avg_pressure_df)
    return pd.concat(rows, axis=0, ignore_index=True), pressure_hist


def run_case():
    output_dir = "flow_upscaling_two_phase_aquifer"
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    perm_file = Path(__file__).resolve().parent / "Heter_model_pure_co2" / "PERM66_ECL.INC"
    center_ij_1b = (42, 30)
    layer_1b = 1
    injection_rate = 80.0
    nt = 50
    dt = 365.0

    two_phase_model = TwoPhaseAquiferPatchModel(
        perm_file=str(perm_file),
        egg_center_ij_1b=center_ij_1b,
        layer_1b=layer_1b,
        injection_rate=injection_rate,
        output_name="two_phase",
    )
    single_phase_model = SinglePhaseCO2PatchModel(
        perm_file=str(perm_file),
        egg_center_ij_1b=center_ij_1b,
        layer_1b=layer_1b,
        injection_rate=injection_rate,
        output_name="single_phase",
    )

    two_phase_hist, two_phase_pressure = run_one_model(two_phase_model, output_dir, nt, dt, "two_phase_aquifer")
    single_phase_hist, single_phase_pressure = run_one_model(single_phase_model, output_dir, nt, dt, "single_phase_co2")

    two_phase_hist.to_excel(os.path.join(output_dir, "two_phase_effective_trans_history.xlsx"), index=False)
    single_phase_hist.to_excel(os.path.join(output_dir, "single_phase_effective_trans_history.xlsx"), index=False)
    two_phase_pressure.to_excel(os.path.join(output_dir, "two_phase_pressure_history.xlsx"), index=False)
    single_phase_pressure.to_excel(os.path.join(output_dir, "single_phase_pressure_history.xlsx"), index=False)

    comparison = compare_with_single_phase(two_phase_hist, single_phase_hist)
    comparison.to_excel(os.path.join(output_dir, "phase_vs_single_phase_trans_comparison.xlsx"), index=False)
    plot_phase_trans_history(
        two_phase_hist,
        fig_dir,
        filename="two_phase_effective_trans_vs_time.png",
        title="Two-phase apparent interface transmissibility",
    )
    plot_phase_trans_history(
        single_phase_hist,
        fig_dir,
        filename="single_phase_effective_trans_vs_time.png",
        title="Single-phase CO2 apparent interface transmissibility",
    )
    plot_phase_single_comparison(comparison, fig_dir)
    plot_pressure_history(two_phase_pressure, fig_dir, "two_phase_aquifer")
    plot_pressure_history(single_phase_pressure, fig_dir, "single_phase_co2")
    plot_two_phase_saturation_xy(two_phase_model, fig_dir, "two_phase_aquifer")
    print(f"Saved results to: {output_dir}")


if __name__ == "__main__":
    run_case()
