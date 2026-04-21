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
    def build_cell_center(self):
        if not hasattr(self, 'reservoir') or self.reservoir is None:
            raise RuntimeError("Reservoir not built yet.")
        n = self.reservoir.n
        self.reservoir.discretize()
        x = np.empty(n, dtype=float)
        y = np.empty(n, dtype=float)
        z = np.asarray(self.reservoir.global_data["depth"], dtype = float).copy()
        # level 0
        nx0= int(self.reservoir.nx)
        ny0= int(self.reservoir.ny)
        nz0= int(self.reservoir.nz)
        dx0 =  float(np.asarray(self.reservoir.global_data["dx"]).flat[0])
        dy0 = float(np.asarray(self.reservoir.global_data["dy"]).flat[0])
        for id in range(n):
            k = id // (nx0 * ny0) # 0 based
            j = (id % (nx0 * ny0)) // nx0
            i = id % nx0
            x[id] = (i + 0.5) * dx0
            y[id] = (j + 0.5) * dy0

        self.reservoir.cell_center_x = x
        self.reservoir.cell_center_y = y
        self.reservoir.cell_center_z = z
        return x,y,z

    def __init__(self):
        # Call base class constructor
        super().__init__()

        # Measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        self.set_reservoir()
        self.zero = 1e-8
        self.set_physics()

        self.set_sim_params(first_ts=1e-6, mult_ts=2, max_ts=2, runtime=1000,
                            tol_newton=1e-3, tol_linear=1e-3,
                            it_newton=10, it_linear=50)

        self.timer.node["initialization"].stop()

    def set_reservoir(self):

        nx0, ny0 = 80, 80 # global grid size
        dx0, dy0 = 100, 100
        nz_res = 10
        dz_res = 20

        over_thickness = 2000.0
        under_thickness = 2000.0
        dz_over = self.build_dz(dz0=dz_res, total_thickness=over_thickness)
        dz_over = dz_over[::-1]  # reverse for overburden

        dz_under = self.build_dz(dz0=dz_res,total_thickness=under_thickness)
        nz_over = len(dz_over)
        nz_under = len(dz_under)
        nz0 = nz_over + nz_res + nz_under
        dz0_layers = np.concatenate([dz_over, np.full(nz_res, dz_res, dtype=float), dz_under])
        permx0, permy0, permz0 = 50, 50, 50
        poro0 = 0.1
        poro_burden = 0.0001
        perm_burden = 1e-6
        self.nz_over = nz_over
        self.nz_res = nz_res

        # thermal properties
        rcond_res = 181.44 # KJ/m/day/k
        hcap_res = 2650 # kJ/m3/K assume reservoir density here
        rcond_over, rcond_under = 149.54, 149.54
        hcap_over, hcap_under = 2347.29, 2347.29

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

        self.reservoir = StructReservoir(self.timer, nx=nx0, ny=ny0, nz=nz0, dx=dx0, dy=dy0, dz=dz0_layers,
                                      permx=kx0_full, permy=ky0_full, permz=kz0_full, poro=poro0_full,depth= None,
                                      start_z=0, rcond=rcon0_full, hcap=hcap0_full,)
        boundary_factor = 2000
        base_vol = float(dx0 * dy0 * dz_res)
        v_big = base_vol * boundary_factor

        self.reservoir.boundary_volumes = {
            "xy_minus": None,
            "xy_plus": None,
            "yz_minus": v_big,
            "yz_plus": v_big,
            "xz_minus": v_big,
            "xz_plus": v_big,
        }
        self.reservoir.discretize()
        self.build_cell_center()
        return


    def set_wells(self):
        self.reservoir.add_well("I1")
        for k in range(self.nz_over, self.nz_over + self.nz_res):
            self.reservoir.add_perforation("I1", res_cell_idx=(41,41,k), ms_epm=False)

        self.reservoir.add_well("P1")
        for k in range(self.nz_over, self.nz_over + 5):
            self.reservoir.add_perforation("P1", res_cell_idx=(36,46,k), ms_epm=False)
        self.reservoir.add_well("P2")
        for k in range(self.nz_over, self.nz_over + 5):
            self.reservoir.add_perforation("P2", res_cell_idx=(46,46,k), ms_epm=False)
        self.reservoir.add_well("P3")
        for k in range(self.nz_over, self.nz_over + 5):
            self.reservoir.add_perforation("P3", res_cell_idx=(36,36,k), ms_epm=False)
        self.reservoir.add_well("P4")
        for k in range(self.nz_over, self.nz_over + 5):
            self.reservoir.add_perforation("P4", res_cell_idx=(46,36,k), ms_epm=False)



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
        nb = int(self.reservoir.nz)
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
