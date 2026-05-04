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

from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer

from darts.physics.properties.flash import ConstantK
from darts.physics.properties.basic import ConstFunc, PhaseRelPerm
from darts.physics.properties.density import DensityBasic, Garcia2001
from darts.reservoirs.reservoir_base import ReservoirBase
from darts.physics.properties.viscosity import Fenghour1998, Islam2012
from darts.physics.properties.eos_properties import EoSDensity, EoSEnthalpy

from dartsflash.libflash import NegativeFlash
from dartsflash.libflash import CubicEoS, AQEoS, FlashParams, InitialGuess
from dartsflash.components import CompData
from darts.physics.super.initialize import Initialize


class Model(DartsModel):

    def __init__(self):
        # Call base class constructor
        super().__init__()

        # Measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        self.set_reservoir()
        self.zero = 1e-8
        self.set_physics()

        self.set_sim_params(first_ts=1e-6, mult_ts=2, max_ts=30, runtime=1000,
                            tol_newton=1e-3, tol_linear=1e-3,
                            it_newton=10, it_linear=50,
                            well_rate_ctrl_absolute_residual_scale=1.0,
                            well_rate_ctrl_relative_residual_scale=1e-5)

        self.timer.node["initialization"].stop()

         # build cell center coordinates for visualization
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


    def set_reservoir(self):

        (nx, ny, nz) = (60,60,9)
        nb = nx * ny * nz
        nb_res = nx * ny * 7
        dx, dy, dz_res = 30., 30., 10.

        permx_res, permy_res, permz_res = 100, 100, 10
        poro0 = 0.2
        poro_burden = 1e-5
        perm_burden = 1e-9

        # thermal properties
        rcond_res = 181.44 # KJ/m/day/k
        hcap_res = 2650 # kJ/m3/K assume reservoir density here
        rcond_over, rcond_under = 149.54, 149.54
        hcap_over, hcap_under = 2347.29, 2347.29

        nz_over = 1
        nz_res = 7
        nz_under = 1

        k_index0 = np.arange(nb, dtype=np.int32) // (nx * ny)

        # --- initialize full-domain properties with burden defaults ---
        kx0_full = np.full(nb, perm_burden, dtype=float)
        ky0_full = np.full(nb, perm_burden, dtype=float)
        kz0_full = np.full(nb, perm_burden, dtype=float)

        rcon0_full = np.empty(nb, dtype=float)
        hcap0_full = np.empty(nb, dtype=float)
        poro0_full = np.full(nb, poro_burden, dtype=float)

        mask_over = k_index0 < nz_over
        mask_res = (k_index0 >= nz_over) & (k_index0 < nz_over + nz_res)
        mask_under = k_index0 >= (nz_over + nz_res)

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

        poro0_full[mask_res] = 0.2

        self.reservoir = StructReservoir(self.timer, nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz_res,
                                      permx=kx0_full, permy=ky0_full, permz=kz0_full, poro=poro0_full,depth= None,
                                      start_z=490, rcond=rcon0_full, hcap=hcap0_full,)
        boundary_factor = 2000
        base_vol = float(dx * dy * dz_res)
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
        self.reservoir.add_well("I1")
        for k in range(2, 9):
            self.reservoir.add_perforation("I1", res_cell_idx=(42,30,k),ms_epm=True, well_diameter=0.1524)

        self.reservoir.add_well("P1")
        for k in range(2, 9):
            self.reservoir.add_perforation("P1", res_cell_idx=(19,30,k),ms_epm=True, well_diameter=0.1524)


    def set_physics(self):
        components = ['CO2']

        self.components = components
        comp_data = CompData(components, setprops=True)
        pr = CubicEoS(comp_data, CubicEoS.PR)

        self.zero = 1e-12
        epsilon = self.zero / 10
        phases = ['CO2_rich']

        property_container = ModelProperties(phases_name=phases, components_name=components, eps_z=epsilon, Mw = comp_data.Mw)

        # Define property evaluators based on custom properties
        property_container.density_ev = dict([('CO2_rich', EoSDensity(eos=pr,Mw=comp_data.Mw))])
        property_container.viscosity_ev = dict([('CO2_rich', Fenghour1998())])

        property_container.enthalpy_ev = dict([('CO2_rich', EoSEnthalpy(eos=pr))])
        property_container.conductivity_ev = dict([('CO2_rich', ConstFunc(10.)) ])


        """ Activate physics """
        thermal = True
        state_spec = Compositional.StateSpecification.PT if thermal else Compositional.StateSpecification.P
        self.physics = Compositional(components, phases, self.timer, state_spec=state_spec,
                                     n_points=400, min_p=1, max_p=1000, min_z=self.zero/10, max_z=1-self.zero/10,
                                     epsilon_z=epsilon, min_t=273.15, max_t=373.15+200)


        property_container.output_props = {
            "satG": lambda: property_container.sat[0],
            "rhoG": lambda: property_container.dens[0],
             "muG": lambda: property_container.mu[0],
            }

        self.physics.add_property_region(property_container)

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
        nb = int(self.reservoir.nz) # number of depth values
        depths = np.linspace(min_depth,max_depth,nb)

        init = Initialize(self.physics)

        primary_specs = {}
        for comp in self.physics.components[:-1]:
            primary_specs[comp] = 1.0

        boundary_state = {"pressure" :50}
        for comp in self.physics.components[:-1]:
            boundary_state[comp] = float(primary_specs[comp])
        boundary_state["temperature"] = 32 +273.15

        dTdh = 34/1000 #k/m

        X = init.solve_up_and_downwards(depth_bottom=max_depth, depth_top=min_depth, depth_known=500,
                                        boundary_state=boundary_state, primary_specs=primary_specs, nb=nb,
                                        dTdh=dTdh)

        self.physics.set_initial_conditions_from_depth_table(mesh=self.reservoir.mesh,
                                                             input_depth= init.depths,
                                                            input_distribution={v:X[:,i] for i, v in enumerate(self.physics.vars)})
        return

    def set_well_controls(self):
        inj_composition = [1.0]  # pure CO2 injection
        for i, w in enumerate(self.reservoir.wells):
            if i == 0:
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.MASS_RATE,
                                            is_inj=True, target=4.32e6, inj_composition=inj_composition, inj_temp=314.15)
            else:
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.BHP,
                                               is_inj=False, target=40.)



# Simplified property evaluation for single-phase model`
class ModelProperties(PropertyContainer):
    def __init__(self, phases_name, components_name, eps_z, Mw):
        # Call base class constructor
        # nc = len(components_name)
        # Mw = np.ones(nc)
        super().__init__(phases_name, components_name, Mw, eps_z=eps_z, temperature=None)

    def evaluate(self, state):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        # Composition vector and pressure from state:
        vec_state_as_np = np.asarray(state)
        pressure = vec_state_as_np[0]

        self.temperature = vec_state_as_np[-1] if self.thermal else self.temperature

        zc = np.append(vec_state_as_np[1:self.nc], 1 - np.sum(vec_state_as_np[1:self.nc]))

        self.clean_arrays()
        j = 0
        # two-phase flash - assume water phase is always present and water component last
        self.x[j, :] = zc

        self.ph = np.array([j], dtype=np.intp)

        # molar weight of mixture
        M = np.sum(self.x[j, :] * self.Mw)
        self.dens[j] = self.density_ev[self.phases_name[j]].evaluate(pressure, self.temperature,[1.0])  # output in [kg/m3]
        self.dens_m[j] = self.dens[j] / M
        self.mu[j] = self.viscosity_ev[self.phases_name[j]].evaluate(pressure=pressure, temperature=self.temperature, x=[1.0],rho=self.dens[j])  # output in [cp]

        self.sat[j] = 1
        self.kr[j] = 1
        self.pc[j] = 0

        return
