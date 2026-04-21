import os
from pathlib import Path

import numpy as np
import pandas as pd
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
from darts.engines import redirect_darts_output

class SinglePhaseCO2Properties(PropertyContainer):
    def __init__(self, phases_name, components_name, eps_z, Mw):
        super().__init__(phases_name, components_name, Mw, eps_z=eps_z, temperature=None)

    def evaluate(self, state):
        state_np = np.asarray(state, dtype=float)
        pressure = state_np[0]
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
            pressure, self.temperature, [1.0]
        )
        self.dens_m[j] = self.dens[j] / M

        self.mu[j] = self.viscosity_ev[self.phases_name[j]].evaluate(
            pressure=pressure,
            temperature=self.temperature,
            x=[1.0],
            rho=self.dens[j],
        )

        self.sat[j] = 1.0
        self.kr[j] = 1.0
        self.pc[j] = 0.0
        return



def extract_coarse_patch_from_level0(model, ic_1b, jc_1b, k_1b, coarse_patch_size=5):

    if coarse_patch_size % 2 == 0:
        raise ValueError("coarse_patch_size must be odd.")

    half = coarse_patch_size // 2
    i0_min = ic_1b - 1 - half
    i0_max = ic_1b - 1 + half
    j0_min = jc_1b - 1 - half
    j0_max = jc_1b - 1 + half
    k0 = k_1b - 1

    nx0 = int(model.level0.nx)
    ny0 = int(model.level0.ny)
    nz0 = int(model.level0.nz)

    if not (0 <= i0_min and i0_max < nx0 and 0 <= j0_min and j0_max < ny0 and 0 <= k0 < nz0):
        raise ValueError(
            f"Requested patch exceeds level0 boundary: "
            f"(ic,jc,k)=({ic_1b},{jc_1b},{k_1b}), patch={coarse_patch_size}"
        )

    kx0 = np.asarray(model.level0.global_data["permx"], dtype=float)
    ky0 = np.asarray(model.level0.global_data["permy"], dtype=float)
    kz0 = np.asarray(model.level0.global_data["permz"], dtype=float)

    kx_patch_c = kx0[i0_min:i0_max + 1, j0_min:j0_max + 1, k0].copy()
    ky_patch_c = ky0[i0_min:i0_max + 1, j0_min:j0_max + 1, k0].copy()
    kz_patch_c = kz0[i0_min:i0_max + 1, j0_min:j0_max + 1, k0].copy()

    return kx_patch_c, ky_patch_c, kz_patch_c

class FlowUpscalingModel(DartsModel):
    """
    Minimal single-phase CO2 2D example for flow-based upscaling.

    Geometry
    --------
    - total grid: 25 x 25 x 1
    - each fine block: 6 x 6 x 10 m
    - central 5 x 5 fine block is treated as the "patch" for effective-T analysis
    - permeability comes from a 5 x 5 coarse Egg patch, piecewise prolonged to 25 x 25
    """

    def __init__(self, kx_patch_c, ky_patch_c, kz_patch_c,
                 refine=(5, 5), dx_parent=30.0, dy_parent=30.0, dz_parent=10.0,
                 start_z=1990.0, poro=0.2, rcond=500.0, hcap=2200.0):
        super().__init__()

        self.kx_patch_c = kx_patch_c
        self.ky_patch_c = ky_patch_c
        self.kz_patch_c = kz_patch_c

        self.coarse_patch_size = int(kx_patch_c.shape[0])
        self.refine = tuple(refine)

        rx, ry = self.refine
        nxc = self.kx_patch_c.shape[0]
        nyc = self.kx_patch_c.shape[1]

        self.nx = nxc * rx
        self.ny = nyc * ry
        self.nz = 1

        self.dx = float(dx_parent) / rx
        self.dy = float(dy_parent) / ry
        self.dz = float(dz_parent)

        self.patch_size = rx
        self.patch_center_1b = (self.nx // 2 + 1, self.ny // 2 + 1)
        self.start_z = float(start_z)
        self.poro = float(poro)

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
        )
        self.timer.node["initialization"].stop()

    def set_reservoir(self):
        rx, ry = self.refine

        kx_f = np.repeat(np.repeat(self.kx_patch_c, rx, axis=0), ry, axis=1)[:, :, None]
        ky_f = np.repeat(np.repeat(self.ky_patch_c, rx, axis=0), ry, axis=1)[:, :, None]
        kz_f = np.repeat(np.repeat(self.kz_patch_c, rx, axis=0), ry, axis=1)[:, :, None]

        self.reservoir = StructReservoir(
            self.timer,
            nx=self.nx,
            ny=self.ny,
            nz=self.nz,
            dx=self.dx,
            dy=self.dy,
            dz=self.dz,
            permx=kx_f,
            permy=ky_f,
            permz=kz_f,
            poro=self.poro,
            depth=None,
            start_z=self.start_z,
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
        # only one injector in the center
        ic = self.patch_center_1b[0]
        jc = self.patch_center_1b[1]
        self.reservoir.add_well("I1")
        self.reservoir.add_perforation(
            "I1",
            res_cell_idx=(ic, jc, 1),
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
            "pressure": 200.0,          # bar
            "temperature": 80.0 + 273.15,
        }

        primary_specs = {}
        for comp in self.physics.components[:-1]:
            primary_specs[comp] = np.ones(int(self.reservoir.nz))

        X = init.solve_up_and_downwards(depth_bottom=max_depth, depth_top=min_depth, depth_known=self.start_z,
                                        boundary_state=boundary_state, primary_specs=primary_specs, nb=int(self.reservoir.nz),
                                        dTdh=40.0 / 1000.0)

        self.physics.set_initial_conditions_from_depth_table(
            mesh=self.reservoir.mesh,
            input_depth=init.depths,
            input_distribution={v: X[:, i] for i, v in enumerate(self.physics.vars)},
        )

    def set_well_controls(self):
        inj_composition = [1.0]

        for i, w in enumerate(self.reservoir.wells):
            if i == 0:
                self.physics.set_well_controls(
                    wctrl=w.control,
                    control_type=well_control_iface.MASS_RATE,
                    is_inj=True,
                    target=10.0,               # kg/day or simulator-consistent unit in your setup
                    inj_composition=inj_composition,
                    inj_temp=314.15,
                )


# ============================================================
# Effective transmissibility analyzer for central 5x5 patch
# ============================================================

def compute_eff_tran_for_one_lgr_layer(
    model, lgr_name,
    k_1b, coarse_patch_size=5,
    refine=(5, 5),
    run_days=365.0,
    n_steps=20,
    mobility_mode="interface_avg",
):

    cfg = model.lgrs[lgr_name]["lgr_coords_in_parent_grid"]
    ic = int(cfg['i_range'][0])
    jc = int(cfg['j_range'][0])

    kx_patch_c, ky_patch_c, kz_patch_c = extract_coarse_patch_from_level0(
        model=model,
        ic_1b=ic,
        jc_1b=jc,
        k_1b=k_1b,
        coarse_patch_size=coarse_patch_size,
    )

    effective_2d_model = FlowUpscalingModel(
        kx_patch_c=kx_patch_c,
        ky_patch_c=ky_patch_c,
        kz_patch_c=kz_patch_c,
        refine=refine,
        dx_parent=float(model.level0.global_data["dx"][0,0,0]),
        dy_parent=float(model.level0.global_data["dy"][0,0,0]),
        dz_parent=float(model.level0.global_data["dz"][0,0,0]),
        start_z=float(model.level0.global_data["start_z"]) + k_1b * float(model.level0.global_data["dz"][0,0,0]),
        poro=float(model.level1[lgr_name].global_data["poro"][0,0,0]),
    )

    analyzer = PatchEffectiveTransAnalyzer(
        model=effective_2d_model,
        nx=effective_2d_model.nx,
        ny=effective_2d_model.ny,
        nz=effective_2d_model.nz,
        patch_size=refine[0],
        patch_center_1b=effective_2d_model.patch_center_1b,
        n_nb_cols=refine[0],
    )
    redirect_darts_output("upscaling_2d.log")
    effective_2d_model.init(platform='cpu')
    # code for batch tasks to avoid output collision. Each task writes to its own tmp folder, and the main process can gather results after all tasks are done.
    job_id = os.environ.get("SLURM_JOB_ID", "nojid")
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "notaskid")
    pid = os.getpid()
    tmp_root = Path("tmp_upscaling_results")
    tmp_root.mkdir(parents=True, exist_ok=True)
    tmp_dir = tmp_root / f"tmp_upscaling_{job_id}_{task_id}_{pid}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    effective_2d_model.set_output(output_folder=str(tmp_dir))

    for dt in range(n_steps):
        effective_2d_model.run(run_days)

    res = analyzer.all_faces(k0=0, mobility_mode=mobility_mode)
    df = res["summary_df"].copy()

    out = {}
    for _, row in df.iterrows():
        out[row["side"]] = float(row["T_eff_avglink"])

    return{
        'lgr_name': lgr_name,
        'k_1b': k_1b,
        'T_eff_avglink': out,
        "summary_df": df,
        "effective_2d_model": effective_2d_model,
    }

def compute_eff_tran_map_for_lgrs(
    model,
    lgr_orders,
    coarse_patch_size=5,
    run_days=365.0,
    n_steps=20,
    mobility_mode="interface_avg",
    tag_filter=None,
):
    """
    return:
    eff_tran_map[lgr_name][k_1b][side] = T_eff_total
    """
    eff_tran_map = {}
    detail = {}

    for name in lgr_orders:
        cfg = model.lgrs[name]["lgr_coords_in_parent_grid"]
        tag = cfg.get("tag", None)

        if tag_filter is not None and tag not in tag_filter:
            continue

        rx, ry, rz = map(int, cfg["refine"])
        k1, k2 = map(int, cfg["k_range"])

        eff_tran_map[name] = {}
        detail[name] = {}

        for k_1b in range(k1, k2 + 1):
            out = compute_eff_tran_for_one_lgr_layer(
                model=model,
                lgr_name=name,
                k_1b=k_1b,
                coarse_patch_size=coarse_patch_size,
                refine=(rx, ry),
                run_days=run_days,
                n_steps=n_steps,
                mobility_mode=mobility_mode,
            )
            eff_tran_map[name][k_1b] = out["T_eff_avglink"]
            detail[name][k_1b] = out

            print(f"[eff_tran] {name} layer={k_1b} -> {out['T_eff_avglink']}")

    return eff_tran_map, detail

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
