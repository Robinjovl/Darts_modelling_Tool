from darts.reservoirs.struct_reservoir import StructReservoir
from darts.models.cicd_model import CICDModel
import numpy as np

from darts.models.opt.opt_module_settings import OptModuleSettings

from darts.input.input_data import InputData


class Model(CICDModel, OptModuleSettings):
    def __init__(self, T, report_step=120, perm=300, poro=0.2, iapws_physics=False, n_points=128):
        # call base class constructor
        CICDModel.__init__(self)
        OptModuleSettings.__init__(self)

        # measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        self.T = T
        self.report_step = report_step

        self.set_reservoir(perm, poro)
        self.iapws_physics = iapws_physics
        self.set_input_data(n_points=n_points)
        self.set_physics()
        self.set_sim_params(first_ts=0.0001, mult_ts=2, max_ts=5, runtime=1000, tol_newton=1e-3, tol_linear=1e-6)

        self.init_pressure = 200.
        self.init_temperature = 350.

        self.timer.node["initialization"].stop()

    def set_reservoir(self, perm, poro):
        """Reservoir construction"""
        # nx = 20
        # ny = 10
        # nz = 2

        nx = 5
        ny = 5
        nz = 2

        # reservoir geometry： for realistic case, one just needs to load the data and input it
        self.reservoir = StructReservoir(self.timer, nx=nx, ny=ny, nz=nz, dx=30, dy=30, dz=12,
                                         permx=perm, permy=perm, permz=perm, poro=poro, depth=2000)

        return

    def set_wells(self):
        # self.inj_list = [[5, 5]]
        # self.prod_list = [[15, 3], [15, 8]]

        self.inj_list = [[3, 3]]
        self.prod_list = [[1, 1], [5, 5]]

        WI = 200

        n_perf = self.reservoir.nz
        for i, inj in enumerate(self.inj_list):
            self.reservoir.add_well('I' + str(i + 1))

            for k in range(n_perf):
                self.reservoir.add_perforation('I' + str(i + 1), cell_index=(inj[0], inj[1], k + 1),
                                               well_radius=0.1, well_index=WI)

        for p, prod in enumerate(self.prod_list):
            self.reservoir.add_well('P' + str(p + 1))

            for k in range(n_perf):
                self.reservoir.add_perforation('P' + str(p + 1), cell_index=(prod[0], prod[1], k + 1),
                                               well_radius=0.1, well_index=WI)

    def set_physics(self):
        """Physical properties"""
        if self.iapws_physics:
            from darts.physics.geothermal.geothermal import Geothermal
            self.physics = Geothermal(self.idata, self.timer)
        else:
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
            property_container = PropertyContainer(phases_name=phases, components_name=["H2O"], Mw=Mw, min_z=zero / 10)

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

        return

    def set_initial_conditions(self):


        input_distribution = {'pressure': self.init_pressure}
        input_distribution.update({comp: self.ini[i] for i, comp in enumerate(self.physics.components[:-1])})
        if self.physics.thermal:
            input_distribution['temperature'] = self.init_temperature

        return self.physics.set_initial_conditions_from_array(self.reservoir.mesh,
                                                              input_distribution=input_distribution)
    def set_well_controls(self):
        from darts.engines import well_control_iface
        for i, w in enumerate(self.reservoir.wells):
            if i == 0:
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.BHP,
                                               is_inj=True, target=self.init_pressure + 30., inj_composition=[],
                                               inj_temp=308.15)
            else:
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.BHP,
                                               is_inj=False, target=self.init_pressure - 10.)

    def run(self, export_to_vtk=False, file_name='data'):
        output_props = ['pressure', 'temperature', 'enthalpy']
        output_path = 'vtk'
        ith_step = 0
        if export_to_vtk:
            self.output_to_vtk(ith_step=ith_step, output_directory=output_path, output_properties=output_props)

        # now we start to run for the time report--------------------------------------------------------------
        time_step = self.report_step
        even_end = int(self.T / time_step) * time_step
        time_step_arr = np.ones(int(self.T / time_step)) * time_step
        if self.T - even_end > 0:
            time_step_arr = np.append(time_step_arr, self.T - even_end)

        for ts in time_step_arr:
            self.set_well_controls()

            CICDModel.run(self, ts, verbose=export_to_vtk)
            self.physics.engine.report()
            if export_to_vtk:
                ith_step += 1
                self.output_to_vtk(ith_step=ith_step, output_directory=output_path, output_properties=output_props)

    def set_input_data(self, n_points):
        # init_type = 'uniform'
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

        # from darts.physics.properties.basic import ConstFunc
        # self.idata.fluid.conduction_ev['water'] = ConstFunc(172.8)

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
