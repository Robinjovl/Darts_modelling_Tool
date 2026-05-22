from pathlib import Path

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
from dartsflash.libflash import EoS
from dartsflash.mixtures import DARTSFlash, VLAq




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
        if self.nu[0] < 0.:
            self.nu = [0., 1.]
            self.X = [[0., 0.], zc]
        elif self.nu[0] > 1.:
            self.nu = [1., 0.]
            self.X = [zc, [0., 0.]]
        self.temperature = temperature
        return 0

    def get_k_values(self, pressure, temperature):
        pi0, pi1, wp = self._bounds(self.p_axis, pressure)
        ti0, ti1, wt = self._bounds(self.t_axis, temperature)
        return np.array([self._interp(self.k_co2, pi0, pi1, ti0, ti1, wp, wt),
                         self._interp(self.k_h2o, pi0, pi1, ti0, ti1, wp, wt)])

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
        return float(table[pi0, ti0] * (1.0 - wp) * (1.0 - wt)
                     + table[pi1, ti0] * wp * (1.0 - wt)
                     + table[pi0, ti1] * (1.0 - wp) * wt
                     + table[pi1, ti1] * wp * wt)


class Model(DartsModel):
    def __init__(self, refine=(5, 5, 1), perm_file_name: str = "PERM66_ECL.INC", include_producer=True):
        super().__init__()

        self.timer.node["initialization"].start()
        self.refine = tuple(refine)
        self.perm_file_name = perm_file_name
        self.include_producer = include_producer
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

    @staticmethod
    def prolongate_coarse_to_fine(arr_c, fx=5, fy=5, fz=1):
        arr_f = np.repeat(arr_c, fx, axis=0)
        arr_f = np.repeat(arr_f, fy, axis=1)
        arr_f = np.repeat(arr_f, fz, axis=2)
        return arr_f

    def build_cell_center(self):
        if not hasattr(self, "reservoir") or self.reservoir is None:
            raise RuntimeError("Reservoir not built yet.")

        n = self.reservoir.n
        self.reservoir.discretize()

        x = np.empty(n, dtype=float)
        y = np.empty(n, dtype=float)
        z = np.asarray(self.reservoir.global_data["depth"], dtype=float).copy()
        nx, ny = int(self.reservoir.nx), int(self.reservoir.ny)
        dx = float(np.asarray(self.reservoir.global_data["dx"]).flat[0])
        dy = float(np.asarray(self.reservoir.global_data["dy"]).flat[0])
        for idx in range(n):
            j = (idx % (nx * ny)) // nx
            i = idx % nx
            x[idx] = (i + 0.5) * dx
            y[idx] = (j + 0.5) * dy
        self.reservoir.cell_center_x = x
        self.reservoir.cell_center_y = y
        self.reservoir.cell_center_z = z
        return x, y, z

    def set_reservoir(self):
        fx, fy, fz = self.refine
        if fz != 1:
            raise NotImplementedError(
                "Current well/layer setup assumes fz=1. Extend carefully before using vertical refinement."
            )

        nx_c, ny_c, nz_res = 60, 60, 7
        nz_over = 1
        nz_under = 1
        nz = nz_over + nz_res + nz_under

        nx = nx_c * fx
        ny = ny_c * fy
        nb = nx * ny * nz
        nb_res_c = nx_c * ny_c * nz_res

        dx = 30.0 / fx
        dy = 30.0 / fy
        dz = 10.0 / fz

        base_dir = Path(__file__).resolve().parent
        perm_file = base_dir / self.perm_file_name

        permx_c = load_single_keyword(str(perm_file), "PERMX", nb_res_c)
        permy_c = load_single_keyword(str(perm_file), "PERMY", nb_res_c)
        permz_c = load_single_keyword(str(perm_file), "PERMZ", nb_res_c)

        permx_c = np.asarray(permx_c, dtype=float).reshape((nx_c, ny_c, nz_res), order="F")
        permy_c = np.asarray(permy_c, dtype=float).reshape((nx_c, ny_c, nz_res), order="F")
        permz_c = np.asarray(permz_c, dtype=float).reshape((nx_c, ny_c, nz_res), order="F")

        permx_res = self.prolongate_coarse_to_fine(permx_c, fx=fx, fy=fy, fz=fz)
        permy_res = self.prolongate_coarse_to_fine(permy_c, fx=fx, fy=fy, fz=fz)
        permz_res = self.prolongate_coarse_to_fine(permz_c, fx=fx, fy=fy, fz=fz)

        assert permx_res.shape == (nx, ny, nz_res * fz)
        assert permy_res.shape == (nx, ny, nz_res * fz)
        assert permz_res.shape == (nx, ny, nz_res * fz)

        poro_burden = 1e-5
        perm_burden = 1e-9
        rcond_over, rcond_under = 149.54, 149.54
        hcap_over, hcap_under = 2347.29, 2347.29
        rcond_res = 2.1 * 86.4
        hcap_res = 2200.0
        poro_res = 0.2

        k_index = np.arange(nb) // (nx * ny)
        mask_over = k_index < nz_over
        mask_res = (k_index >= nz_over) & (k_index < nz_over + nz_res)
        mask_under = k_index >= nz_over + nz_res

        kx_full = np.full(nb, perm_burden, dtype=float)
        ky_full = np.full(nb, perm_burden, dtype=float)
        kz_full = np.full(nb, perm_burden, dtype=float)
        rcond_full = np.empty(nb, dtype=float)
        hcap_full = np.empty(nb, dtype=float)
        poro_full = np.full(nb, poro_burden, dtype=float)

        kx_full[mask_res] = permx_res.reshape(-1, order="F")
        ky_full[mask_res] = permy_res.reshape(-1, order="F")
        kz_full[mask_res] = permz_res.reshape(-1, order="F")
        rcond_full[mask_over] = rcond_over
        rcond_full[mask_res] = rcond_res
        rcond_full[mask_under] = rcond_under
        hcap_full[mask_over] = hcap_over
        hcap_full[mask_res] = hcap_res
        hcap_full[mask_under] = hcap_under
        poro_full[mask_res] = poro_res

        self.reservoir = StructReservoir(
            self.timer,
            nx=nx,
            ny=ny,
            nz=nz,
            dx=dx,
            dy=dy,
            dz=dz,
            permx=kx_full,
            permy=ky_full,
            permz=kz_full,
            poro=poro_full,
            depth=None,
            start_z=1990.0,
            rcond=rcond_full,
            hcap=hcap_full,
        )

        v_big = 1e20
        self.reservoir.boundary_volumes = {
            "xy_minus": v_big,
            "xy_plus": v_big,
            "yz_minus": v_big,
            "yz_plus": v_big,
            "xz_minus": v_big,
            "xz_plus": v_big,
        }

        self.reservoir.discretize()
        self.build_cell_center()
        return

    def set_wells(self):
        fx, fy, _ = self.refine

        inj_i, inj_j = 208, 148
        prod_i, prod_j = 93, 148

        self.reservoir.add_well("I1")
        for k in range(2, 9):
            self.reservoir.add_perforation(
                "I1",
                res_cell_idx=(inj_i, inj_j, k),
                ms_epm=True,
                well_diameter=0.1524,
            )
        if not self.include_producer:
            return
        self.reservoir.add_well("P1")
        for k in range(2, 9):
            self.reservoir.add_perforation(
                "P1",
                res_cell_idx=(prod_i, prod_j, k),
                ms_epm=True,
                well_diameter=0.1524,
            )

    def set_physics(self):
        components = ["CO2", "H2O"]
        phases = ["CO2_rich", "aqueous"]
        self.components = components
        eps = self.zero / 10.0
        comp_data = CompData(components, setprops=True)
        pc = PropertyContainer(phases_name=phases, components_name=components, Mw=comp_data.Mw, eps_z=eps)
        # pc.flash_ev = TableKFlash(2, Path(__file__).resolve().parent / "K_values.csv", eps)
        pc.flash_ev = VLAq(comp_data, hybrid=True)
        pc.flash_ev.set_vl_eos(
            "PR",
            root_order=[EoS.STABLE],
            trial_comps=[i for i in range(len(components))],
            stability_tol=1e-20,
            switch_tol=1e-2,
            max_iter=50,
            use_gmix=False,
        )
        pc.flash_ev.set_aq_eos("Aq", stability_tol=1e-20, max_iter=10, use_gmix=True)
        pc.flash_ev.init_flash(
            flash_type=DARTSFlash.FlashType.PTFlash,
            eos_order=["VL", "Aq"],
            t_min=270.,
            t_max=700.,
            t_init=300.,
        )
        pr = pc.flash_ev.eos["VL"]
        aq = pc.flash_ev.eos["Aq"]
        pc.density_ev = {"CO2_rich": EoSDensity(eos=pr, Mw=comp_data.Mw),
                         "aqueous": Garcia2001(components)}
        pc.viscosity_ev = {"CO2_rich": Fenghour1998(),
                           "aqueous": Islam2012(components)}
        pc.rel_perm_ev = {"CO2_rich": PhaseRelPerm("gas", swc=0.20, sgr=0.00, kre=0.95, n=5),
                          "aqueous": PhaseRelPerm("oil", swc=0.20, sgr=0.00, kre=1.0, n=6)}
        pc.enthalpy_ev = {"CO2_rich": EoSEnthalpy(eos=pr),
                          "aqueous": EoSEnthalpy(eos=aq)}
        pc.conductivity_ev = {"CO2_rich": ConstFunc(6),
                              "aqueous": ConstFunc(60)}
        self.physics = Compositional(components, phases, self.timer,
                                     state_spec=Compositional.StateSpecification.PT,
                                     n_points=400, min_p=1, max_p=1000, min_z=eps, max_z=1.0 - eps,
                                     epsilon_z=eps, min_t=273.15, max_t=400+273.15)
        pc.output_props = {"satG": lambda: pc.sat[0],
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
        x = init.solve_up_and_downwards(depth_bottom=float(np.max(depths)), depth_top=float(np.min(depths)),
                                        depth_known=2000.0, boundary_state=boundary_state,
                                        primary_specs=primary_specs, nb=int(self.reservoir.nz), dTdh=34.0 / 1000.0)
        self.physics.set_initial_conditions_from_depth_table(
            mesh=self.reservoir.mesh, input_depth=init.depths,
            input_distribution={v: x[:, i] for i, v in enumerate(self.physics.vars)})

    def set_well_controls(self):
        for well in self.reservoir.wells:
            if well.name.startswith("I"):
                self.physics.set_well_controls(wctrl=well.control, control_type=well_control_iface.MASS_RATE,
                                               is_inj=True, target=4.32e6, phase_name="CO2_rich",
                                               inj_composition=[1.0 - self.zero], inj_temp=40+273.15)
                self.physics.set_well_controls(wctrl=well.constraint, control_type=well_control_iface.BHP,
                                               is_inj=True, target=300.0, phase_name="CO2_rich",
                                               inj_composition=[1.0 - self.zero], inj_temp=40+273.15)
            else:
                self.physics.set_well_controls(wctrl=well.control, control_type=well_control_iface.BHP,
                                               is_inj=False, target=190.0)
