import numpy as np

from darts.input.input_data import InputData
from darts.models.darts_model import DartsModel
from set_case import set_input_data
from darts.engines import value_vector

from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer

from darts.physics.properties.basic import PhaseRelPerm, ConstFunc
from darts.physics.properties.density import Garcia2001
from darts.physics.properties.viscosity import Fenghour1998, Islam2012
from darts.physics.properties.eos_properties import EoSDensity, EoSEnthalpy

from dartsflash.libflash import NegativeFlash
from dartsflash.libflash import CubicEoS, AQEoS, FlashParams, InitialGuess
from dartsflash.components import CompData
from model_cpg import Model_CPG, fmt

class ModelCCS(Model_CPG):
    def __init__(self):
        self.zero = 1e-10
        super().__init__()

    def set_physics(self):
        """Physical properties"""
        # Fluid components, ions and solid
        components = ["H2O", "CO2"]
        phases = ["Aq", "V"]
        nc = len(components)
        comp_data = CompData(components, setprops=True)

        pr = CubicEoS(comp_data, CubicEoS.PR)
        # aq = Jager2003(comp_data)
        aq = AQEoS(comp_data, AQEoS.Ziabakhsh2012)

        flash_params = FlashParams(comp_data)

        # EoS-related parameters
        flash_params.add_eos("PR", pr)
        flash_params.add_eos("AQ", aq)
        flash_params.eos_order = ["AQ", "PR"]

        # Flash-related parameters
        # flash_params.split_switch_tol = 1e-3
        temperature = None
        if temperature is None:  # if None, then thermal=True
            thermal = True
        else:
            thermal = False

        """ properties correlations """
        property_container = PropertyContainer(phases_name=phases, components_name=components, Mw=comp_data.Mw,
                                               temperature=temperature, min_z=self.zero/10)

        property_container.flash_ev = NegativeFlash(flash_params, ["AQ", "PR"], [InitialGuess.Henry_AV])
        property_container.density_ev = dict([('V', EoSDensity(pr, comp_data.Mw)),
                                              ('Aq', Garcia2001(components))])
        property_container.viscosity_ev = dict([('V', Fenghour1998()),
                                                ('Aq', Islam2012(components))])
        property_container.rel_perm_ev = dict([('V', PhaseRelPerm("gas")),
                                               ('Aq', PhaseRelPerm("oil"))])

        property_container.enthalpy_ev = dict([('V', EoSEnthalpy(pr)),
                                               ('Aq', EoSEnthalpy(aq))])
        property_container.conductivity_ev = dict([('V', ConstFunc(10.)),
                                                   ('Aq', ConstFunc(180.)), ])

        property_container.output_props = {"satA": lambda: property_container.sat[0],
                                           "satV": lambda: property_container.sat[1],
                                           "xCO2": lambda: property_container.x[0, 1],
                                           "yH2O": lambda: property_container.x[1, 0]
                                           }

        self.physics = Compositional(components, phases, self.timer, n_points=self.idata.obl.n_points,
                                     min_p=self.idata.obl.min_p, max_p=self.idata.obl.max_p,
                                     min_z=self.idata.obl.min_z, max_z=self.idata.obl.max_z,
                                     min_t=self.idata.obl.min_t, max_t=self.idata.obl.max_t,
                                     thermal=self.idata.thermal, cache=self.idata.obl.cache)
        self.physics.add_property_region(property_container)

        return

    def get_arrays(self, ith_step):
        '''
        :return: dictionary of current unknown arrays (p, T)
        '''
        # Find index of properties to output
        ev_props = self.physics.property_operators[next(iter(self.physics.property_operators))].props_name
        # If output_properties is None, all variables and properties from property_operators will be passed
        props_names = list(ev_props)
        timesteps, property_array = self.output_properties(output_properties=props_names, timestep=ith_step)
        return property_array

    def set_input_data(self, case=''):
        self.idata = InputData(type_hydr='thermal', type_mech='none', init_type='uniform')
        set_input_data(self.idata, case)

        self.idata.geom.burden_layers = 0

        # well controls
        wdata = self.idata.well_data
        wells = wdata.wells  # short name
        # set default injection composition
        #wdata.inj = value_vector([self.zero])  # injection composition - water
        y2d = 365.25
        if 'wbhp' in case:
            for w in wells:
                wdata.add_inj_bhp_control(name=w, bhp=250, comp_index=1, temperature=300)  # kmol/day | bars | K
        elif 'wrate' in case:
            for w in wells:
                wdata.add_inj_rate_control(name=w, rate=1e6, comp_index=1, bhp_constraint=250, temperature=300)  # kmol/day | bars | K

        self.idata.obl.n_points = 1000
        self.idata.obl.zero = 1e-11
        self.idata.obl.min_p = 1.
        self.idata.obl.max_p = 400.
        self.idata.obl.min_t = 273.15
        self.idata.obl.max_t = 373.15
        self.idata.obl.min_z = self.idata.obl.zero
        self.idata.obl.max_z = 1 - self.idata.obl.zero
        self.idata.obl.cache = False
        self.idata.thermal = True

    def set_initial_conditions(self):
        self.temperature_initial_ = 273.15 + 76.85  # K
        self.initial_values = {"pressure": 100.,
                            "H2O": 0.99995,
                            "temperature": self.temperature_initial_
                            }
        super().set_initial_conditions()

        mesh = self.reservoir.mesh
        depth = np.array(mesh.depth, copy=True)
        # set initial pressure
        pressure_grad = 100
        pressure = np.array(mesh.pressure, copy=False)
        pressure[:] = depth[:pressure.size] / 1000 * pressure_grad + 1
        # set initial temperature
        temperature_grad = 30
        temperature = np.array(mesh.temperature, copy=False)
        temperature[:] = depth[:pressure.size] / 1000 * temperature_grad + 273.15 + 20
        temperature[:] = 350

    def set_well_controls(self, time: float = 0., verbose=True):
        '''
        :param time: simulation time, [days]
        :return:
        '''
        inj_stream_base = [self.zero * 100]
        eps_time = 1e-15
        for w in self.reservoir.wells:
            # find next well control in controls list for different timesteps
            wctrl = None
            for wctrl_t in self.idata.well_data.wells[w.name].controls:
                if np.fabs(wctrl_t[0] - time) < eps_time:  # check time
                    wctrl = wctrl_t[1]
                    break
            if wctrl is None:
                continue
            if wctrl.type == 'inj':  # INJ well
                inj_stream = inj_stream_base
                if self.physics.thermal:
                    inj_stream += [wctrl.inj_bht]
                if wctrl.mode == 'rate': # rate control
                    w.control = self.physics.new_rate_inj(wctrl.rate, inj_stream, wctrl.comp_index)
                    w.constraint = self.physics.new_bhp_inj(wctrl.bhp_constraint, inj_stream)
                elif wctrl.mode == 'bhp': # BHP control
                    w.control = self.physics.new_bhp_inj(wctrl.bhp, inj_stream)
                else:
                    print('Unknown well ctrl.mode', wctrl.mode)
                    exit(1)
            elif wctrl.type == 'prod':  # PROD well
                if wctrl.mode == 'rate': # rate control
                    w.control = self.physics.new_rate_prod(wctrl.rate, wctrl.comp_index)
                    w.constraint = self.physics.new_bhp_prod(wctrl.bhp_constraint)
                elif wctrl.mode == 'bhp': # BHP control
                    w.control = self.physics.new_bhp_prod(wctrl.bhp)
                else:
                    print('Unknown well ctrl.mode', wctrl.mode)
                    exit(1)
            else:
                print('Unknown well ctrl.type', wctrl.type)
                exit(1)
            if verbose:
                print('set_well_controls: time=', time, 'well=', w.name, w.control, w.constraint)

        # check
        for w in self.reservoir.wells:
            assert w.control is not None, 'well control is not initialized for the well ' + w.name
            if verbose and w.constraint is not None and 'rate' in str(type(w.control)):
                print('A constraint for the well ' + w.name + ' is not initialized!')

    def print_well_rate(self):
        return