import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from darts.models.cicd_model import DartsModel
from darts.engines import well_control_iface
from darts.reservoirs.struct_reservoir import StructReservoir

from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer
from darts.physics.properties.basic import ConstFunc
from darts.physics.properties.viscosity import Fenghour1998
from darts.physics.properties.eos_properties import EoSDensity, EoSEnthalpy

from dartsflash.libflash import CubicEoS
from dartsflash.components import CompData

from darts.physics.super.initialize import Initialize
from darts.tools.keyword_file_tools import load_single_keyword


# ============================================================
# Utilities
# ============================================================

def ijk_to_global_0based(i_1b: int, j_1b: int, k_1b: int, nx: int, ny: int) -> int:
    """Convert 1-based (i,j,k) to 0-based flattened index in F-style structured ordering."""
    return (k_1b - 1) * nx * ny + (j_1b - 1) * nx + (i_1b - 1)


def prolongate_piecewise(arr_2d_coarse: np.ndarray, refine_x: int, refine_y: int) -> np.ndarray:
    """
    Piecewise prolongation from coarse 2D array to fine 2D array by block replication.
    shape: (nx_c, ny_c) -> (nx_c*refine_x, ny_c*refine_y)
    """
    out = np.repeat(arr_2d_coarse, refine_x, axis=0)
    out = np.repeat(out, refine_y, axis=1)
    return out



def build_patch_perm_from_egg(
    perm_file: str,
    center_ij_1b=(46, 30),
    coarse_patch_size=5,
    refine=(5, 5),
    layer_1b=2,
):
    """
    Build a 25x25 fine patch from a 5x5 coarse Egg patch around center_ij_1b.

    Example:
      coarse_patch_size = 5
      refine = (5,5)
      => 25x25 fine grid

    Returns
    -------
    kx_f, ky_f, kz_f : ndarray, shape (25, 25, 1)
    """
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


# ============================================================
# Property container
# ============================================================

class SinglePhaseCO2Properties(PropertyContainer):
    def __init__(self, phases_name, components_name, eps_z, Mw):
        super().__init__(phases_name, components_name, Mw, eps_z=eps_z, temperature=None)

    def evaluate(self, state):
        state_np = np.asarray(state, dtype=float)
        self.pressure = state_np[0]
        self.temperature = state_np[-1] if self.thermal else self.temperature

        zc = np.append(
            state_np[1:self.nc],
            1.0 - np.sum(state_np[1:self.nc]),
        )

        self.clean_arrays()

        j = 0
        self.x[j, :] = zc
        self.ph = np.array([j], dtype=np.intp)

        M = np.sum(self.x[j, :] * self.Mw)

        self.dens[j] = self.density_ev[self.phases_name[j]].evaluate(
            self.pressure, self.temperature, [1.0]
        )
        self.dens_m[j] = self.dens[j] / M

        self.mu[j] = self.viscosity_ev[self.phases_name[j]].evaluate(
            pressure=self.pressure,
            temperature=self.temperature,
            x=[1.0],
            rho=self.dens[j],
        )

        self.sat[j] = 1.0
        self.kr[j] = 1.0
        self.pc[j] = 0.0
        return


# ============================================================
# Model
# ============================================================

class FlowUpscalingExampleModel(DartsModel):
    """
    Minimal single-phase CO2 2D example for flow-based upscaling.

    Geometry
    --------
    - total grid: 25 x 25 x 1
    - each fine block: 6 x 6 x 10 m
    - central 5 x 5 fine block is treated as the "patch" for effective-T analysis
    - permeability comes from a 5 x 5 coarse Egg patch, piecewise prolonged to 25 x 25
    """

    def __init__(self, perm_file: str, egg_center_ij_1b=(46, 30)):
        super().__init__()

        self.perm_file = perm_file
        self.egg_center_ij_1b = egg_center_ij_1b

        self.nx = 25
        self.ny = 25
        self.nz = 1

        self.dx = 6.0
        self.dy = 6.0
        self.dz = 10.0

        self.patch_size = 5
        self.patch_center_1b = (13, 13)

        self.zero = 1e-10

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
            layer_1b=7,
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
            start_z=500.0,
            rcond=500.0,
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
        # only one producer in the center
        self.reservoir.add_well("P1")
        self.reservoir.add_perforation(
            "P1",
            res_cell_idx=(13, 13, 1),
            well_diameter=0.1524,
        )

    def set_physics(self):
        components = ["CO2"]
        phases = ["CO2_rich"]
        epsilon = self.zero / 10

        comp_data = CompData(components, setprops=True)
        pr = CubicEoS(comp_data, CubicEoS.PR)

        pc = SinglePhaseCO2Properties(
            phases_name=phases,
            components_name=components,
            eps_z=epsilon,
            Mw=comp_data.Mw,
        )

        pc.density_ev = {"CO2_rich": EoSDensity(eos=pr, Mw=comp_data.Mw)}
        pc.viscosity_ev = {"CO2_rich": Fenghour1998()}
        pc.enthalpy_ev = {"CO2_rich": EoSEnthalpy(eos=pr)}
        pc.conductivity_ev = {"CO2_rich": ConstFunc(10.0)}

        pc.output_props = {
            "satG": lambda: pc.sat[0],
            "rhoG": lambda: pc.dens[0],
            "muG": lambda: pc.mu[0],
        }

        self.physics = Compositional(
            components=components,
            phases=phases,
            timer=self.timer,
            state_spec=Compositional.StateSpecification.PT,
            n_points=400,
            min_p=1.0,
            max_p=1000.0,
            min_z=self.zero / 10.0,
            max_z=1.0 - self.zero / 10.0,
            epsilon_z=epsilon,
            min_t=273.15,
            max_t=573.15,
        )

        self.physics.add_property_region(pc)

    def set_initial_conditions(self):
        depths = np.asarray(self.reservoir.mesh.depth)
        min_depth = np.min(depths)
        max_depth = np.max(depths)

        init = Initialize(self.physics)

        boundary_state = {
            "pressure": 50.0,          # bar
            "temperature": 32.0 + 273.15,
        }

        primary_specs = {}
        for comp in self.physics.components[:-1]:
            primary_specs[comp] = np.ones(int(self.reservoir.nz))

        X = init.solve_up_and_downwards(depth_bottom=max_depth, depth_top=min_depth, depth_known=500.0,
                                        boundary_state=boundary_state, primary_specs=primary_specs, nb=int(self.reservoir.nz),
                                        dTdh=34.0 / 1000.0)

        self.physics.set_initial_conditions_from_depth_table(
            mesh=self.reservoir.mesh,
            input_depth=init.depths,
            input_distribution={v: X[:, i] for i, v in enumerate(self.physics.vars)},
        )

    def set_well_controls(self):
        for i, w in enumerate(self.reservoir.wells):
            if i == 0:
                self.physics.set_well_controls(
                    wctrl=w.control,
                    control_type=well_control_iface.MASS_RATE,
                    is_inj=False,
                    target=20.0,              # kg/day or simulator-consistent unit in your setup
                )


# ============================================================
# Effective transmissibility analyzer for central 5x5 patch
# ============================================================

class PatchEffectiveTransAnalyzer:
    """
    Apparent/effective transmissibility around a square patch in a fine model.

    For each side:
      T_eff = - Q_total / (lambda_ref * dp_macro)

    with:
      - Q_total: sum of immediate link fluxes across the patch boundary
      - dp_macro: p_if_avg - p_nb_avg
      - lambda_ref: 1 / mu_ref
    """

    VALID_SIDES = ("left", "right", "up", "down")

    def __init__(self, model, nx, ny, nz, patch_size=5, patch_center_1b=(13, 13), n_nb_cols=5):
        self.model = model
        self.nx = int(nx)
        self.ny = int(ny)
        self.nz = int(nz)
        self.patch_size = int(patch_size)
        self.patch_center_1b = tuple(patch_center_1b)
        self.n_nb_cols = int(n_nb_cols)

        if self.patch_size % 2 == 0:
            raise ValueError("patch_size must be odd.")
        if self.n_nb_cols < 1:
            raise ValueError("n_nb_cols must be >= 1.")

    def lin_index(self, i0, j0, k0):
        return i0 + j0 * self.nx + k0 * self.nx * self.ny

    def _patch_bounds_0b(self):
        ic = self.patch_center_1b[0] - 1
        jc = self.patch_center_1b[1] - 1
        half = self.patch_size // 2
        return ic - half, ic + half, jc - half, jc + half

    def get_connection_df(self):
        mesh = self.model.reservoir.mesh
        block_m = np.asarray(mesh.block_m, dtype=int)
        block_p = np.asarray(mesh.block_p, dtype=int)
        tran = np.asarray(mesh.tran, dtype=float)

        return pd.DataFrame({
            "block_m": block_m,
            "block_p": block_p,
            "tran": tran,
        })

    def get_pressure_array(self):
        n_vars = len(self.model.physics.vars)
        X = np.asarray(self.model.physics.engine.X, dtype=float).reshape((-1, n_vars))
        return X[: self.model.reservoir.mesh.n_res_blocks, 0].copy()

    def eval_mu(self, cell_idx):
        n_vars = len(self.model.physics.vars)
        X = np.asarray(self.model.physics.engine.X, dtype=float).reshape((-1, n_vars))
        state = X[cell_idx, :].copy()

        pc = self.model.physics.property_containers[0]
        pc.evaluate(state)
        return float(pc.mu[0])

    def interface_cells(self, side, k0=0):
        i0_min, i0_max, j0_min, j0_max = self._patch_bounds_0b()
        ids = []

        if side == "left":
            i0 = i0_min
            for j0 in range(j0_min, j0_max + 1):
                ids.append(self.lin_index(i0, j0, k0))

        elif side == "right":
            i0 = i0_max
            for j0 in range(j0_min, j0_max + 1):
                ids.append(self.lin_index(i0, j0, k0))

        elif side == "up":
            j0 = j0_min
            for i0 in range(i0_min, i0_max + 1):
                ids.append(self.lin_index(i0, j0, k0))

        elif side == "down":
            j0 = j0_max
            for i0 in range(i0_min, i0_max + 1):
                ids.append(self.lin_index(i0, j0, k0))

        else:
            raise ValueError(f"Invalid side: {side}")

        return ids

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

    def find_connection(self, cell_a, cell_b, conn_df):
        mask = (
            ((conn_df["block_m"] == cell_a) & (conn_df["block_p"] == cell_b)) |
            ((conn_df["block_m"] == cell_b) & (conn_df["block_p"] == cell_a))
        )
        return conn_df.loc[mask]

    def face_effective_trans(self, side, k0=0, mobility_mode="interface_avg"):
        conn_df = self.get_connection_df()
        pressure = self.get_pressure_array()

        if_cells = self.interface_cells(side, k0)
        nb_cells = self.immediate_neighbor_cells(side, k0)
        nb_support = self.neighbor_support_cells(side, k0)

        p_if_avg = float(np.mean(pressure[if_cells]))
        p_nb_avg = float(np.mean(pressure[nb_support]))
        dp_macro = p_if_avg - p_nb_avg

        if abs(dp_macro) < 1e-14:
            raise ZeroDivisionError(f"dp_macro too small on side={side}")

        total_flux = 0.0
        mu_refs = []

        link_rows = []

        for c_if, c_nb in zip(if_cells, nb_cells):
            hit = self.find_connection(c_if, c_nb, conn_df)
            if hit.empty:
                raise RuntimeError(f"No connection found between {c_if} and {c_nb}")

            tran = float(hit["tran"].iloc[0])
            p_if = float(pressure[c_if])
            p_nb = float(pressure[c_nb])
            dp_link = p_if - p_nb

            mu_if = self.eval_mu(c_if)
            mu_nb = self.eval_mu(c_nb)

            if mobility_mode == "interface":
                mu_ref_link = mu_if
            elif mobility_mode == "neighbor":
                mu_ref_link = mu_nb
            elif mobility_mode == "interface_avg":
                mu_ref_link = 0.5 * (mu_if + mu_nb)
            else:
                raise ValueError(f"Unknown mobility_mode: {mobility_mode}")

            lam_link = 1.0 / mu_ref_link
            q_link = - tran * dp_link * lam_link

            total_flux += q_link
            mu_refs.append(mu_ref_link)

            link_rows.append({
                "side": side,
                "cell_if": c_if,
                "cell_nb": c_nb,
                "tran": tran,
                "p_if": p_if,
                "p_nb": p_nb,
                "dp_link": dp_link,
                "mu_ref_link": mu_ref_link,
                "flux": q_link,
            })

        mu_ref = float(np.mean(mu_refs))
        lam_ref = 1.0 / mu_ref

        T_eff_total = - total_flux / (lam_ref * dp_macro)
        T_eff_avglink = T_eff_total / len(link_rows)

        return {
            "side": side,
            "n_links": len(link_rows),
            "p_if_avg": p_if_avg,
            "p_nb_avg": p_nb_avg,
            "dp_macro": dp_macro,
            "mu_ref": mu_ref,
            "lambda_ref": lam_ref,
            "total_flux": total_flux,
            "T_eff_total": T_eff_total,
            "T_eff_avglink": T_eff_avglink,
            "link_df": pd.DataFrame(link_rows),
        }

    def all_faces(self, k0=0, mobility_mode="interface_avg"):
        rows = []
        detail = {}

        for side in self.VALID_SIDES:
            out = self.face_effective_trans(side=side, k0=k0, mobility_mode=mobility_mode)
            detail[side] = out
            rows.append({
                "side": side,
                "n_links": out["n_links"],
                "p_if_avg": out["p_if_avg"],
                "p_nb_avg": out["p_nb_avg"],
                "dp_macro": out["dp_macro"],
                "mu_ref": out["mu_ref"],
                "lambda_ref": out["lambda_ref"],
                "total_flux": out["total_flux"],
                "T_eff_total": out["T_eff_total"],
                "T_eff_avglink": out["T_eff_avglink"],
            })

        return {
            "faces": detail,
            "summary_df": pd.DataFrame(rows),
        }


# ============================================================
# Run + plot
# ============================================================

def plot_effective_trans_history(df, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(8, 5), dpi=150)
    for side in ["left", "right", "up", "down"]:
        dfi = df[df["side"] == side]
        plt.plot(dfi["time_day"], dfi["T_eff_avglink"], label=side)

    plt.xlabel("time [day]")
    plt.ylabel("effective transmissibility")
    plt.title("Interface effective transmissibility vs time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "effective_trans_vs_time.png"))
    plt.close()

def get_reservoir_average_pressure(model):
    n_res = model.reservoir.mesh.n_res_blocks
    n_vars = len(model.physics.vars)

    X = np.asarray(model.physics.engine.X, dtype=float)
    P = X.reshape((-1, n_vars))[:n_res, 0]

    poro = np.array(model.reservoir.mesh.poro, copy=False)[:n_res]
    volume = np.array(model.reservoir.mesh.volume, copy=False)[:n_res]
    pv = poro * volume
    return float(np.sum(P * pv) / np.sum(pv))

def plot_pressure_history(pressure_hist, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # -------- plot 1: BHP and reservoir average pressure --------
    plt.figure(figsize=(8, 5), dpi=150)
    plt.plot(pressure_hist["time_day"], pressure_hist["bhp_bar"], label="Producer BHP")
    plt.plot(pressure_hist["time_day"], pressure_hist["avg_pressure"], label="Reservoir average pressure")

    plt.xlabel("time [day]")
    plt.ylabel("pressure [bar]")
    plt.title("Producer BHP and reservoir average pressure")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "bhp_and_reservoir_avg_pressure.png"))
    plt.close()

    # -------- plot 2: pressure difference only --------
    plt.figure(figsize=(8, 5), dpi=150)
    plt.plot(pressure_hist["time_day"], pressure_hist["delta_p"])

    plt.xlabel("time [day]")
    plt.ylabel("Pavg - BHP [bar]")
    plt.title("Producer pressure drawdown for stabilization check")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pavg_minus_bhp_vs_time.png"))
    plt.close()


def run_case():
    output_dir = "flow_upscaling_producer_example"
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    perm_file = Path(r"E:\repo_2\open-darts\LGR_kairan\Egg model\Heter_model\PERM1_ECL.INC")

    model = FlowUpscalingExampleModel(
        perm_file=str(perm_file),
        egg_center_ij_1b=(46, 30),
    )

    model.init(platform="cpu")
    model.set_output(output_folder=output_dir)

    analyzer = PatchEffectiveTransAnalyzer(
        model=model,
        nx=25,
        ny=25,
        nz=1,
        patch_size=5,
        patch_center_1b=(13, 13),
        n_nb_cols=5,
    )

    Nt = 50
    Dt = 365.0

    rows = []
    time_point = []
    avg_pre = []



    for _ in range(Nt):
        model.run(Dt)
        t_end = float(model.physics.engine.t)
        time_point.append(t_end)
        avg_pre.append(get_reservoir_average_pressure(model))

        res = analyzer.all_faces(k0=0, mobility_mode="interface_avg")
        dfi = res["summary_df"].copy()
        dfi["time_day"] = float(model.physics.engine.t)
        rows.append(dfi)

        print(dfi)
    avg_pre_df = pd.DataFrame({
        "time_day": time_point,
        "avg_pressure": avg_pre,
    }).sort_values("time_day").reset_index(drop=True)
    time_data_dict = model.output.store_well_time_data(save_output_files=True)
    time_data_df = pd.DataFrame.from_dict(time_data_dict)
    bhp_df = time_data_df[["time", "well_P1_BHP"]].copy()
    bhp_df = bhp_df.rename(columns={
        "time":"time_day",
        "well_P1_BHP":"bhp_bar",
    })
    bhp_df = bhp_df.sort_values("time_day").reset_index(drop=True)

    pressure_hist = pd.merge_asof(
        avg_pre_df, bhp_df, on="time_day", direction="nearest", tolerance=1e-6,
    )
    pressure_hist["delta_p"] = pressure_hist["avg_pressure"] - pressure_hist["bhp_bar"]

    pressure_hist.to_excel(
    os.path.join(output_dir, "pressure_stabilization_history.xlsx"),
    index=False,
)

    plot_pressure_history(pressure_hist, fig_dir)

    hist = pd.concat(rows, axis=0, ignore_index=True)
    hist.to_excel(os.path.join(output_dir, "effective_trans_history.xlsx"), index=False)
    plot_effective_trans_history(hist, fig_dir)

    print(f"Saved results to: {output_dir}")


if __name__ == "__main__":
    run_case()
