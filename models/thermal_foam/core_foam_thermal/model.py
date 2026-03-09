import sys

from darts.reservoirs.struct_reservoir import StructReservoir
from darts.models.cicd_model import CICDModel
from darts.engines import value_vector, redirect_darts_output, sim_params, well_control_iface

import numpy as np

from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer

from darts.physics.properties.flash import ConstantK
from darts.physics.properties.basic import ConstFunc, PhaseRelPerm
from darts.physics.properties.density import DensityBasic, Garcia2001
from darts.physics.properties.enthalpy import EnthalpyBasic
from darts.physics.properties.viscosity import Lee1966
from darts.physics.properties.eos_properties import EoSDensity, EoSEnthalpy

class ThermalFoamRelPerm:
    def __init__(self, phase, foam, swc=0., sgr=0., kre=1., n=2.):
        super().__init__()
        self.phase = phase
        self.Swc = swc
        self.Sgr = sgr
        self.kre = kre
        self.sr = sgr
        self.sr1 = swc
        self.n = n
        self.fmmob = foam['fmmob']
        self.sfdry = foam['sfdry']
        self.sfbet = foam['sfbet']
        self.eptemp = foam['eptemp']
        self.t_gas = foam['t_gas']
        self.t_ref = foam['t_ref']

    def evaluate(self, sat, temperature):

        Ftemp = np.exp(-self.eptemp * (temperature - self.t_ref))
        if temperature >= self.t_gas:
            Ftemp = 0.0
        if temperature <= self.t_ref:
            Ftemp = 1.0

        if sat >= 1 - self.sr1:
            kr = self.kre
        elif sat <= self.sr:
            kr = 0
        else:
            # general Brook-Corey
            kr = self.kre * ((sat - self.sr) / (1 - self.Sgr - self.Swc)) ** self.n
        water_sat = 1 - sat

        Fdry = 0.5 + np.arctan(self.sfbet * (water_sat - self.sfdry)) / np.pi

        ret = kr / (1 + self.fmmob * Fdry * Ftemp)
        return ret

class WaterRelPerm:
    def __init__(self, phase, swc=0.0, sgr=0.0, kre=1.0, n=2.0):
        self.phase = phase
        self.Swc = swc
        self.Sgr = sgr
        self.kre = kre
        self.n = n

    def evaluate(self, sat, temperature=0):
        if sat >= 1 - self.Swc:
            kr = self.kre
        elif sat <= self.Sgr:
            kr = 0
        else:
            # general Brooks-Corey
            kr = self.kre * ((sat - self.Sgr) / (1 - self.Sgr - self.Swc)) ** self.n
        return kr

class Model(CICDModel):
    def __init__(self, t_inj=60+273.15):
        # call base class constructor
        super().__init__()

        # measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        self.P_init = 40  # initial pressure [bar]
        self.T_init = 60+273.15
        self.T_inj = t_inj
        self.fg_inj = 0.8

        self.set_reservoir()
        self.set_physics()

        self.initial_values = {self.physics.vars[0]: self.P_init,
                                self.physics.vars[1]: 1-self.zero,
                                self.physics.vars[2]: self.T_init}
       
        self.inj_stream = [1-self.fg_inj, self.T_inj]

        self.set_sim_params(first_ts=1e-6, mult_ts=2, max_ts=1e-4, runtime=1000, tol_newton=1e-3, tol_linear=1e-6)

        # find saturation corresponding with composition
        # INITIAL CONDITION
        self.sgr = 0.293
        z_range = np.linspace(self.zero, 1 - self.zero, 10000)
        # for z in z_range:
        #     state = [self.P_init, z]
        #     # sat = self.physics.property_containers[0].compute_saturation_full(state)
        #     sat = self.physics.property_containers[0].evaluate(state) 
        #     if self.physics.property_containers[0].sat[self.physics.phases.index("wat")] >= 1. - self.sgr: # sgr = 0.293 -> S_w^0 = 1. - sgr
        #         self.ini = value_vector([z])
        #         break
        self.ini = 0.79628
        print("S_w^0 = ", self.ini)
        # sys.exit(0)
        # self.ini = 0.857286

        # INJECTION CONDITION
        self.fg = 0.1
        self.vol_rate = 0.001436 # [m3/day]

        self.timer.node["initialization"].stop()

    def set_fg_Inj(self, fg):
        self.fg = fg
        self.set_well_controls()

    def set_reservoir(self):
        nx = 500
        L = 0.15
        permx = np.ones(nx)*273.5775
        permx[0] = 1e10
        self.reservoir = StructReservoir(self.timer, nx=nx, ny=1, nz=1, dx=L/nx, dy=0.033853868, dz=0.033853868,
                                    permx=permx, permy=300, permz=300, poro=0.25, hcap=2200, rcond=150, depth=100)
        return

    def set_wells(self):
        self.reservoir.add_well("I1") 
        self.reservoir.add_perforation("I1", res_cell_idx=(1, 1, 1), well_index=1e5)

        # TODO: Isn't possible to define 2 wells at same cell_index? when I set cell_index=(2, 1, 1) it works!
        self.reservoir.add_well("I2") # wat well
        self.reservoir.add_perforation("I2", res_cell_idx=(2, 1, 1), well_index=1e5)

        self.reservoir.add_well("P1")
        self.reservoir.add_perforation("P1", res_cell_idx=(self.reservoir.nx, 1, 1), well_index=1e5)

    def set_physics(self, ph: bool = False, vl_phases: bool = False):
        """Physical properties"""

        # If thermal simulation is activated, set temperature variable
        temperature = None
        self.zero = 1e-13
        epsilon = self.zero/10

        # Flash/EoS
        from dartsflash.libflash import EoS
        from dartsflash.components import CompData
        from dartsflash.mixtures import DARTSFlash, VLAq
        # Fluid components, ions and solid
        components = ["H2O", "N2"]
        self.components = components
        phases = ["Aq", "V", "L"] if vl_phases else ["Aq", "V"]
        comp_data = CompData(components, setprops=True)
        nc = len(components)

        # """ Define flash """
        # flash_ev = VLAq(comp_data, hybrid=True)
        # flash_ev.set_vl_eos("PR", root_order=[EoS.MAX, EoS.MIN] if vl_phases else [EoS.STABLE],
        #                     trial_comps=[i for i in range(nc)],
        #                     stability_tol=1e-20, switch_tol=1e-2, max_iter=50, use_gmix=False
        #                     )
        # flash_ev.set_aq_eos("Aq", stability_tol=1e-20, max_iter=10, use_gmix=True)

        # flash_ev.init_flash(flash_type=DARTSFlash.FlashType.PHFlash if ph else DARTSFlash.FlashType.PTFlash,
        #                     eos_order=["Aq", "VL"],
        #                     t_min=270., t_max=500., t_init=300., t_tol=1e-3,
        #                     verbose=False
        #                     )
        property_container = ModelProperties(phases_name=phases, components_name=components, Mw=comp_data.Mw,
                                            temperature=temperature, eps_z=epsilon)

        ki = np.array([44.5, 2.05e-2])
        property_container.flash_ev = ConstantK(nc=len(components), ki=ki, eps=self.zero)

        # property_container.flash_ev = flash_ev
        # Define property evaluators based on custom properties
        property_container.density_ev = dict([('Aq', DensityBasic(compr=1e-5, dens0=1014)),
                                            ('V', DensityBasic(compr=1e-4, dens0=800)),]) # CO2 supercritical
        # property_container.density_ev = dict([('gas', DensityBasic(compr=1e-4, dens0=22.3)),
        #                                     ('wat', DensityBasic(compr=1e-5, dens0=55.5))])
        property_container.viscosity_ev = dict([('Aq', ConstFunc(0.5)),
                                                # ('V', ConstFunc(0.04)),
                                                ('V', Lee1966(components, Mw=comp_data.Mw)),
                                                ]) # (cP)
        foam_params = {
            'fmmob': 269.34,
            'sfdry': 0.4376,
            'sfbet': 787.4579,
            'eptemp': 0.1,
            't_ref': 20+273.15,
            't_gas': 100+273.15
        }

        property_container.rel_perm_ev = dict([('Aq', WaterRelPerm("Aq", swc=0.4, sgr=0.293, kre=0.302, n=2.98)),
                                            ('V', ThermalFoamRelPerm("V", foam=foam_params, swc=0.4, sgr=0.293, kre=0.04, n=0.96))])
        property_container.enthalpy_ev = dict([('Aq', EnthalpyBasic(hcap=4.18)),
                                            ('V', EnthalpyBasic(hcap=0.84))])
        # property_container.enthalpy_ev = dict([('Aq', EoSEnthalpy(flash_ev.eos['Aq'])),
        #                                        ('V', EoSEnthalpy(flash_ev.eos['VL']))])
        property_container.conductivity_ev = dict([('Aq', ConstFunc(1.0)),
                                                ('V', ConstFunc(0.1))])
        property_container.rock_energy_ev = EnthalpyBasic(hcap=1.0)

        """ activate physics """
        thermal = True
        state_spec = Compositional.StateSpecification.PT if thermal else Compositional.StateSpecification.P
        self.physics = Compositional(components, phases, self.timer, state_spec=state_spec,
                                    n_points=500, min_p=0, max_p=1000, min_z=self.zero, max_z=1-self.zero,
                                    min_t=273.15, max_t=273.15 + 200, epsilon_z=self.zero/10)
        self.physics.add_property_region(property_container)

        return

    def set_initial_conditions(self):
        self.physics.set_initial_conditions_from_array(mesh=self.reservoir.mesh, input_distribution=self.initial_values)

    def set_well_controls(self):
        # print(self.reservoir.wells[2].name)
        wat_inj = 'Aq'
        gas_inj = 'V'
        from darts.engines import well_control_iface
        for i, w in enumerate(self.reservoir.wells):
            if i == 0:   # GAS INJECTION WELL
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.VOLUMETRIC_RATE,
                                               is_inj=True, target=self.fg*self.vol_rate, phase_name=gas_inj,
                                               inj_composition=value_vector([self.zero]), inj_temp=self.inj_stream[-1]) 
            elif i == 1: # WATER INJECTION WELL
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.VOLUMETRIC_RATE,
                                               is_inj=True, target=(1. - self.fg)*self.vol_rate, phase_name=wat_inj, 
                                               inj_composition=value_vector([1.-self.zero]), inj_temp=self.inj_stream[-1])    
            else:      # PRODUCTION WELL
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.BHP,
                                               is_inj=False, target=self.P_init)

class ModelProperties(PropertyContainer):
    def __init__(self, phases_name, components_name, Mw, temperature, eps_z):
        # Call base class constructor
        self.nph = len(phases_name)
        super().__init__(phases_name, components_name, Mw=Mw, eps_z=eps_z, temperature=temperature)

    def evaluate(self, state: value_vector):
        """
        Evaluate the phase properties. Phase properties used only in the energy conservation equation
        are evaluated using a different method.

        :param state: state variables [pres, comp_0, ..., comp_N-1, temperature (optional)]
        :type state: value_vector
        """
        # Composition vector and pressure from state:
        pressure, temperature, zc = self.get_state(state)

        self.clean_arrays()

        # Run flash
        self.ph = self.run_flash(
            pressure, temperature, zc, evaluate_PT=self.evaluate_PT_bool
        )
        self.pressure = pressure
        self.temperature = (
            self.flash_ev.get_flash_results().temperature
            if not isinstance(self.flash_ev, int)
            else self.temperature
        )
        assert self.temperature is not None, (
            "PropertyContainer does not specify self.temperature, should be set to "
            "constant temperature in case of isothermal physics, "
            "self.flash.temperature in case of thermal"
        )

        for j in self.ph:
            M = np.sum(self.Mw[: self.nc_fl] * self.x[j][: self.nc_fl])

            self.dens[j] = self.density_ev[self.phases_name[j]].evaluate(
                pressure, self.temperature, self.x[j, :]
            )  # output in [kg/m3]
            self.dens_m[j] = (
                self.dens[j] / M
            )  # molar density [kg/m3]/[kg/kmol]=[kmol/m3]
            self.mu[j] = self.viscosity_ev[self.phases_name[j]].evaluate(
                pressure, self.temperature, self.x[j, :], self.dens[j]
            )  # output in [cp]
        self.compute_saturation(self.ph)

        self.pc = self.capillary_pressure_ev.evaluate(self.sat)

        for j in self.ph:
            self.kr[j] = self.rel_perm_ev[self.phases_name[j]].evaluate(self.sat[j], self.temperature)  # add temperature parser in rel perm evaluator

        for j in range(self.ns):
            idx = self.np_fl + j
            self.sat[idx] = zc[self.nc_fl + j]
            self.dens[idx] = self.density_ev[self.phases_name[idx]].evaluate(
                pressure, self.temperature
            )
            self.dens_m[idx] = self.dens[idx] / self.Mw[self.nc_fl + j]

        self.mass_source = self.evaluate_mass_source(pressure, self.temperature, zc)

        return