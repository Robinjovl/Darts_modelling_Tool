from darts.reservoirs.struct_reservoir import StructReservoir
from darts.models.cicd_model import CICDModel
from darts.engines import ms_well
import numpy as np

from darts.models.opt.opt_module_settings import OptModuleSettings


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
        self.n_points = n_points
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
                self.reservoir.add_perforation('I' + str(i + 1), res_cell_idx=(inj[0], inj[1], k + 1),
                                               well_diameter=0.2, well_index=WI)

        for p, prod in enumerate(self.prod_list):
            self.reservoir.add_well('P' + str(p + 1))

            for k in range(n_perf):
                self.reservoir.add_perforation('P' + str(p + 1), res_cell_idx=(prod[0], prod[1], k + 1),
                                               well_diameter=0.2, well_index=WI)

    def set_physics(self):
        """Physical properties"""
        if self.iapws_physics:
            from darts.physics.geothermal.geothermal import Geothermal, GeothermalConfig
            config = GeothermalConfig(state_spec="PH", n_points=self.n_points,
                                      min_p=1., max_p=351., min_e=1000., max_e=10000.,
                                      rock_compressibility=0.,
                                      rock_compressibility_ref_p=1.,
                                      rock_compressibility_ref_T=273.15)
            self.physics = Geothermal(config, self.timer)
        else:
            # Define fluid components, phases and Flash object
            from dartsflash.libflash import EoS
            from dartsflash.components import CompData
            from dartsflash.dartsflash import DARTSFlash
            from dartsflash.mixtures import IAPWS, VLAq
            phases = ['water', 'steam']
            components = ["H2O"]
            comp_data = CompData(components=components, setprops=True)
            Mw = comp_data.Mw

            """ Initialize flash """
            pt = False
            flash_ev = VLAq(comp_data, hybrid=True)

            # Add EoS objects for V/L and Aq phases
            flash_ev.set_vl_eos("PR", root_order=[EoS.MAX])
            flash_ev.set_aq_eos("Aq", use_gmix=True)
            ceos = flash_ev.eos["VL"]  # covers vapour phase
            aq = flash_ev.eos["Aq"]  # aqueous liquid phase

            # Initialize flash object
            flash_ev.init_flash(flash_type=DARTSFlash.FlashType.PTFlash if pt else DARTSFlash.FlashType.PHFlash,
                                eos_order=["Aq", "VL"], t_min=250., t_max=575.,
                                pxflash_switch_ttol=1e-1, pxflash_ftol=1e-10)

            # Define PropertyContainer
            from darts.physics.super.property_container import PropertyContainer
            zero = 1e-10
            epsilon = 1e-11
            property_container = PropertyContainer(phases_name=phases, components_name=["H2O"], Mw=Mw, eps_z=epsilon)

            property_container.flash_ev = flash_ev

            # properties implemented in python
            from darts.physics.properties.eos_properties import EoSDensity, EoSEnthalpy
            from darts.physics.properties.density import Spivey2004
            from darts.physics.properties.viscosity import MaoDuan2009
            from darts.physics.properties.basic import ConstFunc, PhaseRelPerm
            property_container.enthalpy_ev = {'water': EoSEnthalpy(aq),
                                              'steam': EoSEnthalpy(ceos, root_flag=EoS.RootFlag.MAX)}
            property_container.density_ev = {'water': Spivey2004(components),
                                             'steam': EoSDensity(ceos, comp_data.Mw, root_flag=EoS.RootFlag.MAX)}
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
                                         n_points=1001, min_p=1, max_p=400, min_z=0., max_z=1., epsilon_z=epsilon,
                                         min_t=273.15, max_t=373.15, cache=False, extrapolation_flag=True)
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
