from darts.reservoirs.struct_reservoir import StructReservoir
from darts.models.cicd_model import CICDModel
from darts.physics.properties.iapws.iapws_property_vec import _Backward1_T_Ph_vec
from darts.tools.keyword_file_tools import load_single_keyword
import numpy as np
from darts.engines import value_vector, sim_params

from darts.input.input_data import InputData


class Model(CICDModel):
    def __init__(self, n_points=128, iapws_physics: bool = True):
        # call base class constructor
        super().__init__()

        self.timer.node["initialization"].start()

        self.set_reservoir()

        self.iapws_physics = iapws_physics
        self.set_input_data(n_points)
        self.set_physics()

        self.set_sim_params(first_ts=1e-3, mult_ts=8, max_ts=365, runtime=3650, tol_newton=1e-2, tol_linear=1e-6,
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
            self.reservoir.add_perforation("INJ", cell_index=(iw[0], jw[0], k + 1),
                                           well_radius=0.16, multi_segment=True)

        # add well
        self.reservoir.add_well("PRD")
        for k in range(1, self.reservoir.nz):
            self.reservoir.add_perforation("PRD", cell_index=(iw[1], jw[1], k + 1),
                                           well_radius=0.16, multi_segment=True)

    def set_physics(self):
        if self.iapws_physics:
            from darts.physics.geothermal.geothermal import Geothermal
            self.physics = Geothermal(self.idata, self.timer)
        else:
            if self.compositional:
                # Define fluid components, phases and Flash object
                from dartsflash.libflash import PXFlash, FlashParams, EoS
                from dartsflash.libflash import CubicEoS, AQEoS
                from dartsflash.components import CompData
                phases = ['water', 'steam']
                components = ["H2O"]
                comp_data = CompData(components=components, setprops=True)
                Mw = comp_data.Mw
                ceos = CubicEoS(comp_data, CubicEoS.PR)
                ceos.set_preferred_roots(0, 0.75, EoS.MAX)
                aq = AQEoS(comp_data, AQEoS.Jager2003)
                aq.set_eos_range(0, [0.6, 1.])

                flash_params = FlashParams(comp_data)

                # EoS-related parameters
                flash_params.add_eos("CEOS", ceos)
                flash_params.add_eos("AQ", aq)
                flash_params.eos_order = ["AQ", "CEOS"]

                flash_params.T_min = 250.
                flash_params.T_max = 575.
                flash_params.phflash_Htol = 1e-3
                flash_params.phflash_Ttol = 1e-8

                # Define PropertyContainer
                from darts.physics.super.property_container import PropertyContainer
                zero = 1e-10
                property_container = PropertyContainer(phases_name=phases, components_name=["H2O"], Mw=Mw, min_z=zero/10)

                property_container.flash_ev = PXFlash(flash_params, PXFlash.ENTHALPY)

                # properties implemented in python
                from darts.physics.properties.eos_properties import EoSDensity, EoSEnthalpy
                from darts.physics.properties.density import Spivey2004
                from darts.physics.properties.viscosity import MaoDuan2009
                from darts.physics.properties.basic import ConstFunc, PhaseRelPerm
                property_container.enthalpy_ev = {'water': EoSEnthalpy(aq),
                                                  'steam': EoSEnthalpy(ceos)}
                property_container.density_ev = {'water': Spivey2004(components),
                                                 'steam': EoSDensity(ceos, comp_data.Mw)}
                property_container.viscosity_ev = {'water': MaoDuan2009(components),
                                                   'steam': ConstFunc(0.01)}
                property_container.conductivity_ev = {'water': ConstFunc(172.8),
                                                      'steam': ConstFunc(0.)}
                property_container.rel_perm_ev = {'water': PhaseRelPerm("water"),
                                                  'steam': PhaseRelPerm("gas")}
                property_container.output_props = {'temperature': lambda: property_container.temperature,
                                                   'satAq': lambda: property_container.sat[0]}

                from darts.physics.super.physics import Compositional
                self.physics = Compositional(components, phases, self.timer, state_spec=Compositional.StateSpecification.PH,
                                             n_points=1001, min_p=1, max_p=400, min_z=zero / 10, max_z=1 - zero / 10,
                                             min_t=273.15, max_t=373.15, cache=False)
                self.physics.add_property_region(property_container)

            else:
                from darts.physics.geothermal.geothermal import GeothermalPH
                self.physics = GeothermalPH(self.idata, self.timer)

    def set_initial_conditions(self):
        input_distribution = {'pressure': 200.,
                              'temperature': 350.
                              }
        return self.physics.set_initial_conditions_from_array(mesh=self.reservoir.mesh,
                                                              input_distribution=input_distribution)


    def set_well_controls(self):
        from darts.engines import well_control_iface
        for i, w in enumerate(self.reservoir.wells):
            if i == 0:
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.VOLUMETRIC_RATE,
                                               is_inj=True, target=8000., phase_name='water', inj_composition=[], inj_temp=300.)
            else:
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.VOLUMETRIC_RATE,
                                               is_inj=False, target=8000., phase_name='water')

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

        if self.iapws_physics:
            from darts.physics.geothermal.geothermal import GeothermalIAPWSFluidProps
            self.idata.fluid = GeothermalIAPWSFluidProps()
            self.compositional = False
        else:
            self.compositional = True
            if self.compositional:
                pass
            else:
                from darts.physics.geothermal.geothermal import GeothermalPHFluidProps
                self.idata.fluid = GeothermalPHFluidProps()

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
