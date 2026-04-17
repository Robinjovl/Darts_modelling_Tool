from pathlib import Path

import numpy as np

from darts.reservoirs.struct_reservoir import StructReservoir
from darts.models.cicd_model import DartsModel
from darts.engines import well_control_iface

from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer

from darts.physics.properties.basic import ConstFunc
from darts.physics.properties.viscosity import Fenghour1998
from darts.physics.properties.eos_properties import EoSDensity, EoSEnthalpy

from dartsflash.libflash import CubicEoS
from dartsflash.components import CompData
from darts.physics.super.initialize import Initialize
from darts.tools.keyword_file_tools import load_single_keyword


class Model(DartsModel):
    """
    Fine structured Egg model obtained by prolongating the original
    60x60x7 Egg permeability field to a finer structured grid.

    Key idea:
    - keep total physical size unchanged
    - keep original spatial distribution unchanged at coarse-block level
    - each coarse Egg block is subdivided into fx*fy*fz fine blocks
      and all sub-blocks inherit the same coarse permeability
    """

    def __init__(self, refine=(5, 5, 1), perm_file_name: str = "PERM1_ECL.INC"):
        super().__init__()

        self.timer.node["initialization"].start()
        self.refine = tuple(refine)
        self.perm_file_name = perm_file_name
        self.set_reservoir()
        self.zero = 1e-8
        self.set_physics()

        self.set_sim_params(
            first_ts=1e-6,
            mult_ts=2,
            max_ts=30,
            runtime=1000,
            tol_newton=1e-3,
            tol_linear=1e-3,
            it_newton=10,
            it_linear=50,
        )

        self.timer.node["initialization"].stop()

    @staticmethod
    def prolongate_coarse_to_fine(arr_c, fx=5, fy=5, fz=1):
        """
        Replicate each coarse value into fine subcells.

        Parameters
        ----------
        arr_c : ndarray, shape (nx_c, ny_c, nz_c)
        fx, fy, fz : int
            Refinement factors in x, y, z directions.
        """
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

        nx = int(self.reservoir.nx)
        ny = int(self.reservoir.ny)
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

        # Original Egg reservoir grid (reservoir only)
        nx_c, ny_c, nz_res = 60, 60, 7
        nz_over = 1
        nz_under = 1
        nz = nz_over + nz_res + nz_under

        # Fine structured grid
        nx = nx_c * fx
        ny = ny_c * fy
        nb = nx * ny * nz
        nb_res_c = nx_c * ny_c * nz_res

        # Keep total model size unchanged
        dx = 30.0 / fx
        dy = 30.0 / fy
        dz = 10.0 / fz

        base_dir = Path(__file__).resolve().parent
        perm_file = base_dir / self.perm_file_name

        # Read original Egg permeability defined on 60x60x7 blocks
        permx_c = load_single_keyword(str(perm_file), "PERMX", nb_res_c)
        permy_c = load_single_keyword(str(perm_file), "PERMY", nb_res_c)
        permz_c = load_single_keyword(str(perm_file), "PERMZ", nb_res_c)

        permx_c = np.asarray(permx_c, dtype=float).reshape((nx_c, ny_c, nz_res), order="F")
        permy_c = np.asarray(permy_c, dtype=float).reshape((nx_c, ny_c, nz_res), order="F")
        permz_c = np.asarray(permz_c, dtype=float).reshape((nx_c, ny_c, nz_res), order="F")

        # Prolongate to fine grid by block replication, not interpolation
        permx_res = self.prolongate_coarse_to_fine(permx_c, fx=fx, fy=fy, fz=fz)
        permy_res = self.prolongate_coarse_to_fine(permy_c, fx=fx, fy=fy, fz=fz)
        permz_res = self.prolongate_coarse_to_fine(permz_c, fx=fx, fy=fy, fz=fz)

        assert permx_res.shape == (nx, ny, nz_res * fz)
        assert permy_res.shape == (nx, ny, nz_res * fz)
        assert permz_res.shape == (nx, ny, nz_res * fz)

        poro_burden = 0.0001
        perm_burden = 1e-6

        rcond_over, rcond_under = 149.54, 149.54
        hcap_over, hcap_under = 2347.29, 2347.29
        rcond_res = 500.0
        hcap_res = 2200.0
        poro_res = 0.2

        k_index = np.arange(nb, dtype=np.int32) // (nx * ny)
        mask_over = k_index < nz_over
        mask_res = (k_index >= nz_over) & (k_index < nz_over + nz_res)
        mask_under = k_index >= (nz_over + nz_res)

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
            start_z=1990,
            rcond=rcond_full,
            hcap=hcap_full,
        )

        # Match current heter-model boundary treatment more closely
        boundary_factor = 2000
        base_vol = float(dx * dy * dz)
        v_big = 1e20
        self.reservoir.boundary_volumes = {
            "xy_minus": v_big,
            "xy_plus": v_big,
            "yz_minus": None,
            "yz_plus": None,
            "xz_minus": None,
            "xz_plus": None,
        }

        self.reservoir.discretize()
        self.build_cell_center()
        return

    def set_wells(self):
        fx, fy, _ = self.refine

        # Coarse well locations from your heter/LGR setup
        inj_i, inj_j =  228, 148
        prod_i, prod_j = 78, 148

        self.reservoir.add_well("I1")
        for k in range(2, 9):
            self.reservoir.add_perforation(
                "I1",
                res_cell_idx=(inj_i, inj_j, k),
                ms_epm=True,
                well_diameter=0.1524,
            )

        self.reservoir.add_well("P1")
        for k in range(2, 9):
            self.reservoir.add_perforation(
                "P1",
                res_cell_idx=(prod_i, prod_j, k),
                ms_epm=True,
                well_diameter=0.1524,
            )

    def set_physics(self):
        components = ["CO2"]
        self.components = components

        comp_data = CompData(components, setprops=True)
        pr = CubicEoS(comp_data, CubicEoS.PR)

        self.zero = 1e-12
        phases = ["CO2_rich"]

        property_container = ModelProperties(
            phases_name=phases,
            components_name=components,
            min_z=self.zero,
            Mw=comp_data.Mw,
        )

        property_container.density_ev = {
            "CO2_rich": EoSDensity(eos=pr, Mw=comp_data.Mw)
        }
        property_container.viscosity_ev = {"CO2_rich": Fenghour1998()}
        property_container.enthalpy_ev = {"CO2_rich": EoSEnthalpy(eos=pr)}
        property_container.conductivity_ev = {"CO2_rich": ConstFunc(10.0)}

        thermal = True
        state_spec = (
            Compositional.StateSpecification.PT
            if thermal
            else Compositional.StateSpecification.P
        )
        self.physics = Compositional(
            components,
            phases,
            self.timer,
            state_spec=state_spec,
            n_points=400,
            min_p=1,
            max_p=1000,
            min_z=self.zero / 10,
            max_z=1 - self.zero / 10,
            min_t=273.15 + 10,
            max_t=373.15 + 200,
        )

        property_container.output_props = {
            "satG": lambda: property_container.sat[0],
            "rhoG": lambda: property_container.dens[0],
            "muG": lambda: property_container.mu[0],
        }

        self.physics.add_property_region(property_container)
        return

    def set_initial_conditions(self):
        depths = np.asarray(self.reservoir.mesh.depth)
        min_depth = np.min(depths)
        max_depth = np.max(depths)
        nb = int(self.reservoir.nz)

        init = Initialize(self.physics)

        primary_specs = {}
        for comp in self.physics.components[:-1]:
            primary_specs[comp] = np.ones(nb)

        boundary_state = {"pressure": 200}
        for comp in self.physics.components[:-1]:
            boundary_state[comp] = float(primary_specs[comp][0])
        boundary_state["temperature"] = 80 + 273.15

        dTdh = 40 / 1000  # K/m

        X = init.solve(
            depth_bottom=max_depth,
            depth_top=min_depth,
            depth_known=2000,
            nb=nb,
            boundary_state=boundary_state,
            primary_specs=primary_specs,
            secondary_specs=None,
            dTdh=dTdh,
        ).reshape((nb, self.physics.n_vars))

        self.physics.set_initial_conditions_from_depth_table(
            mesh=self.reservoir.mesh,
            input_depth=init.depths,
            input_distribution={v: X[:, i] for i, v in enumerate(self.physics.vars)},
        )
        return

    def set_well_controls(self):
        inj_composition = [1.0]
        for i, w in enumerate(self.reservoir.wells):
            if i == 0:
                self.physics.set_well_controls(
                    wctrl=w.control,
                    control_type=well_control_iface.MASS_RATE,
                    is_inj=True,
                    target=4.32e6,
                    inj_composition=inj_composition,
                    inj_temp=313.15,
                )
            else:
                self.physics.set_well_controls(
                    wctrl=w.control,
                    control_type=well_control_iface.BHP,
                    is_inj=False,
                    target=190.0,
                )


class ModelProperties(PropertyContainer):
    def __init__(self, phases_name, components_name, min_z, Mw):
        super().__init__(phases_name, components_name, Mw, min_z=min_z, temperature=None)

    def evaluate(self, state):
        vec_state_as_np = np.asarray(state)
        pressure = vec_state_as_np[0]
        self.temperature = vec_state_as_np[-1] if self.thermal else self.temperature

        zc = np.append(
            vec_state_as_np[1 : self.nc],
            1 - np.sum(vec_state_as_np[1 : self.nc]),
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
