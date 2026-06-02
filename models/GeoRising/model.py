from darts.reservoirs.struct_reservoir import StructReservoir
from darts.models.cicd_model import CICDModel
from darts.physics.properties.iapws.iapws_property_vec import _Backward1_T_Ph_vec
from darts.tools.keyword_file_tools import load_single_keyword
import numpy as np
from darts.engines import value_vector, sim_params, ms_well, well_control_iface

from darts.input.input_data import InputData

from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer
from dartsflash.mixtures import DARTSFlash, CompData, EoS, IAPWS
from darts.physics.properties.eos_properties import EoSDensity, EoSEnthalpy
from darts.physics.properties.basic import ConstFunc, PhaseRelPerm
from darts.physics.properties.viscosity import MaoDuan2009


class Model(CICDModel):
    def __init__(self, n_points=128, iapws_physics: bool = True):
        # call base class constructor
        super().__init__()

        self.timer.node["initialization"].start()

        self.set_reservoir()

        self.iapws_physics = iapws_physics
        self.set_input_data(n_points)
        self.set_physics()

        self.set_sim_params(first_ts=1e-4, mult_ts=8, max_ts=365, runtime=3650, tol_newton=1e-2, tol_linear=1e-6,
                            it_newton=20, it_linear=40, newton_type=sim_params.newton_global_chop,
                            newton_params=value_vector([1]))

        self.timer.node["initialization"].stop()

    def set_reservoir(self):
        (nx, ny, nz) = (60, 60, 3)
        nb = nx * ny * nz
        perm = np.ones(nb) * 2000
        perm = load_single_keyword('permXVanEssen.in', 'PERMX')
        perm = perm[:nb]

        poro = np.ones(nb) * 0.2
        dx = 30
        dy = 30
        dz = np.ones(nb) * 30

        # discretize structured reservoir
        self.reservoir = StructReservoir(self.timer, nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz,
                                         permx=perm, permy=perm, permz=perm * 0.1, poro=poro, depth=2000,
                                         hcap=2200, rcond=500)
        self.reservoir.boundary_volumes['yz_minus'] = 1e8
        self.reservoir.boundary_volumes['yz_plus'] = 1e8
        self.reservoir.boundary_volumes['xz_minus'] = 1e8
        self.reservoir.boundary_volumes['xz_plus'] = 1e8

        return

    def set_wells(self):
        # add well's locations
        iw = [30, 30]
        jw = [14, 46]

        # add well
        self.reservoir.add_well("INJ")
        for k in range(1, self.reservoir.nz):
            self.reservoir.add_perforation("INJ", res_cell_idx=(iw[0], jw[0], k + 1),
                                           well_diameter=0.32, ms_epm=True)

        # add well
        self.reservoir.add_well("PRD")
        for k in range(1, self.reservoir.nz):
            self.reservoir.add_perforation("PRD", res_cell_idx=(iw[1], jw[1], k + 1),
                                           well_diameter=0.32, ms_epm=True)

    def set_iapws_physics(self, n_points, min_p, max_p, min_t, max_t, cache=False):
        """Drop-in replacement for legacy Geothermal(...) using Compositional + IAPWS PT-flash.

        State spec is PT so the OBL grid axes are pressure and temperature, matching the
        BaseModels.set_iapws_physics template. PHFlash on IAPWS95 was unstable: the OBL
        sampling extended below 273.15 K, triggering "LIQUID MINIMUM BISECTION not converged"
        crashes during init.
        """
        components = ["H2O"]
        phases = ['V', 'L']  # vapor, liquid (replaces 'steam','water')
        zero = 1e-12
        comp_data = CompData(components=components, setprops=True)

        pc = PropertyContainer(phases_name=phases, components_name=components,
                               Mw=comp_data.Mw, eps_z=zero)

        flash_ev = IAPWS(iapws_ideal=True, ice_phase=False)
        flash_ev.init_flash(flash_type=DARTSFlash.FlashType.PTFlash)
        pc.flash_ev = flash_ev

        pc.density_ev = {
            'V': EoSDensity(eos=flash_ev.eos["IAPWS"], Mw=comp_data.Mw, root_flag=EoS.RootFlag.MAX),
            'L': EoSDensity(eos=flash_ev.eos["IAPWS"], Mw=comp_data.Mw, root_flag=EoS.RootFlag.MIN),
        }
        pc.viscosity_ev = {
            'V': ConstFunc(0.01),                  # cP, steam
            'L': MaoDuan2009(components),          # cP, liquid water (pressure-dependent)
        }
        pc.enthalpy_ev = {
            'V': EoSEnthalpy(eos=flash_ev.eos["IAPWS"], root_flag=EoS.RootFlag.MAX),
            'L': EoSEnthalpy(eos=flash_ev.eos["IAPWS"], root_flag=EoS.RootFlag.MIN),
        }
        pc.rel_perm_ev = {
            'V': PhaseRelPerm("gas", swc=0.0),
            'L': PhaseRelPerm("oil", swc=0.0),
        }
        pc.conductivity_ev = {
            'V': ConstFunc(0.0),
            'L': ConstFunc(172.8),                 # kJ/m/day/K, matches geothermal default
        }
        # output_props exposes derived T (K) via the property interpolator
        pc.output_props = {'temperature': lambda: pc.temperature}

        self.physics = Compositional(
            components, phases, self.timer,
            state_spec=Compositional.StateSpecification.PT,
            n_points=n_points,
            min_p=min_p, max_p=max_p,
            min_z=zero, max_z=1.0 - zero, epsilon_z=zero,
            min_t=min_t, max_t=max_t,
            cache=cache,
        )
        self.physics.add_property_region(pc)
        return pc

    def set_physics(self):
        # Both legacy iapws_physics=True (Geothermal) and iapws_physics=False (GeothermalPH)
        # branches now route through the Compositional + IAPWS PH-flash helper.
        self.set_iapws_physics(n_points=self.idata.obl.n_points,
                               min_p=1., max_p=400.,
                               min_t=273.15, max_t=575.)

    def set_initial_conditions(self):
        input_distribution = {'pressure': 200.,
                              'temperature': 350.
                              }
        return self.physics.set_initial_conditions_from_array(mesh=self.reservoir.mesh,
                                                              input_distribution=input_distribution)


    def set_well_controls(self):
        for i, w in enumerate(self.reservoir.wells):
            if i == 0:
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.VOLUMETRIC_RATE,
                                               is_inj=True, target=8000., phase_name='L', inj_composition=[], inj_temp=300.)
            else:
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.VOLUMETRIC_RATE,
                                               is_inj=False, target=8000., phase_name='L')

    def compute_temperature(self, X):
        nb = self.reservoir.mesh.n_res_blocks
        temp = _Backward1_T_Ph_vec(X[0:2 * nb:2] / 10, X[1:2 * nb:2] / 18.015)
        return temp

    def set_input_data(self, n_points):
        #init_type = 'uniform'
        init_type = 'gradient'
        self.idata = InputData(type_hydr='thermal', type_mech='none', init_type=init_type)

        self.idata.rock.compressibility = 0.  # [1/bars]
        self.idata.rock.compressibility_ref_p = 1.  # [bars]
        self.idata.rock.compressibility_ref_T = 273.15  # [K]

        # Fluid evaluator wiring now lives in set_iapws_physics(); no idata.fluid needed.
        self.compositional = True

        # example - how to change the properties
        # self.idata.fluid.density['water'] = DensityBasic(compr=1e-5, dens0=1014)

        #from darts.physics.properties.basic import ConstFunc
        #self.idata.fluid.conduction_ev['water'] = ConstFunc(172.8)

        # if init_type== 'uniform': # uniform initial conditions
        #     self.idata.initial.initial_pressure = 200.  # bars
        #     self.idata.initial.initial_temperature = 350.  # K
        # elif init_type == 'gradient':         # gradient by depth
        #     self.idata.initial.reference_depth_for_pressure = 0  # [m]
        #     self.idata.initial.pressure_gradient = 100  # [bar/km]
        #     self.idata.initial.pressure_at_ref_depth = 1 # [bars]
        #
        #     self.idata.initial.reference_depth_for_temperature = 0  # [m]
        #     self.idata.initial.temperature_gradient = 30  # [K/km]
        #     self.idata.initial.temperature_at_ref_depth = 273.15 + 20 # [K]

        # # well controls
        # wctrl = self.idata.wells.controls  # short name
        # wctrl.type = 'rate'
        # #wctrl.type = 'bhp'
        # if wctrl.type == 'bhp':
        #     self.idata.wells.controls.inj_bhp = 250 # bars
        #     self.idata.wells.controls.prod_bhp = 100 # bars
        # elif wctrl.type == 'rate':
        #     self.idata.wells.controls.inj_rate = 5500 # m3/day
        #     self.idata.wells.controls.inj_bhp_constraint = 300 # upper limit for bhp, bars
        #     self.idata.wells.controls.prod_rate = 5500 # m3/day
        #     self.idata.wells.controls.prod_bhp_constraint = 70 # lower limit for bhp, bars
        # self.idata.wells.controls.inj_bht = 300  # K

        self.idata.obl.n_points = n_points
        self.idata.obl.min_p = 1.
        self.idata.obl.max_p = 351.
        self.idata.obl.min_e = 1000.  # kJ/kmol, will be overwritten in PHFlash physics
        self.idata.obl.max_e = 10000.  # kJ/kmol, will be overwritten in PHFlash physics
