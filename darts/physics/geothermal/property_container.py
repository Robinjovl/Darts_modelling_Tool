import numpy as np
from darts.engines import value_vector
from darts.physics.base.property_base import PropertyBase

from darts.physics.properties.iapws.iapws_property import *
from darts.physics.properties.iapws.custom_rock_property import *
from darts.physics.properties.basic import ConstFunc


class PropertyContainerIAPWS(PropertyBase):
    """
    Class responsible for collecting all needed properties in geothermal simulation
    """
    nc: int = 1
    nph: int = 2

    def __init__(self, property_evaluator='IAPWS'):
        """
        Constructor
        :param property_evaluator: determines what property evaluator is used, input is either 'IAPWS' or 'ADGPRS'
        """
        self.Mw = [18.015]
        self.rock = [value_vector([1, 0, 273.15])]
        self.rock_compaction_ev = custom_rock_compaction_evaluator(self.rock)  # Create rock_compaction object
        self.rock_energy_ev = custom_rock_energy_evaluator(self.rock)  # Create rock_energy object

        # properties implemented in C++
        if property_evaluator == 'ADGPRS':
            sat_enthalpy = {'water': saturated_water_enthalpy_evaluator(),
                            'steam': saturated_steam_enthalpy_evaluator()}
            self.enthalpy_ev = {'water', water_enthalpy_evaluator(sat_enthalpy['water'], sat_enthalpy['steam']),
                                'steam', steam_enthalpy_evaluator(sat_enthalpy['steam'], sat_enthalpy['water'])}
            sat_density = {'water': saturated_water_density_evaluator(sat_enthalpy['water']),
                           'steam': saturated_steam_density_evaluator(sat_enthalpy['steam'])}
            self.density_ev = {'water': water_density_evaluator(sat_density['water'], sat_density['steam']),
                            'steam': steam_density_evaluator(sat_density['steam'], sat_density['water'])}
            self.temperature_ev = temperature_evaluator(sat_enthalpy['water'], sat_enthalpy['steam'])
            self.saturation_ev = {'water': water_saturation_evaluator(sat_density['water'], sat_density['steam'],
                                                                      sat_enthalpy['water'], sat_enthalpy['steam']),
                                  'steam': steam_saturation_evaluator(sat_density['water'], sat_density['steam'],
                                                                      sat_enthalpy['water'], sat_enthalpy['steam'])}
            self.viscosity_ev = {'water': water_viscosity_evaluator(self.temperature_ev),
                                 'steam': steam_viscosity_evaluator(self.temperature_ev)}
            self.relperm_ev = {'water': water_relperm_evaluator(self.saturation_ev['water']),
                               'steam': steam_relperm_evaluator(self.saturation_ev['steam'])}
            self.conduction_ev = {'water': ConstFunc(172.8),
                                  'steam': ConstFunc(0.)}

        elif property_evaluator == 'IAPWS':
            # properties implemented in python (the IAPWS package)
            self.temperature_ev = iapws_temperature_evaluator()                    # Create temperature object
            self.enthalpy_ev = {'water': iapws_water_enthalpy_evaluator(),
                                'steam': iapws_steam_enthalpy_evaluator(),
                                'total': iapws_total_enthalpy_evalutor()}
            self.density_ev = {'water': iapws_water_density_evaluator(),
                               'steam': iapws_steam_density_evaluator()}
            self.saturation_ev = {'water': iapws_water_saturation_evaluator(),
                                  'steam': iapws_steam_saturation_evaluator()}
            self.viscosity_ev = {'water': iapws_water_viscosity_evaluator(),
                                 'steam': iapws_steam_viscosity_evaluator()}
            self.conduction_ev = {'water': ConstFunc(172.8),
                                  'steam': ConstFunc(0.)}
            self.relperm_ev = {'water': iapws_water_relperm_evaluator(),
                               'steam': iapws_steam_relperm_evaluator()}

        self.temperature = 0
        self.enthalpy = np.zeros(2)
        self.density = np.zeros(2)
        self.dens_m = np.zeros(2)
        self.saturation = np.zeros(2)
        self.viscosity = np.zeros(2)
        self.conduction = np.zeros(2)
        self.relperm = np.zeros(2)

        self.output_props = {'temperature': lambda: self.temperature}

    def evaluate(self, state):
        self.temperature = self.temperature_ev.evaluate(state)

        for j, phase in enumerate(['water', 'steam']):
            self.enthalpy[j] = self.enthalpy_ev[phase].evaluate(state)
            self.density[j] = self.density_ev[phase].evaluate(state)
            self.dens_m[j] = self.density[j] / self.Mw[0]
            self.saturation[j] = self.saturation_ev[phase].evaluate(state)
            self.viscosity[j] = self.viscosity_ev[phase].evaluate(state)
            self.conduction[j] = self.conduction_ev[phase].evaluate(state)
            self.relperm[j] = self.relperm_ev[phase].evaluate(state)

        self.ph = np.array([j for j in range(self.nph) if self.saturation[j] > 0])
        return

    def compute_total_enthalpy(self, state, temperature):
        return self.enthalpy_ev['total'].evaluate(state, temperature)


class PropertyContainerPH(PropertyBase):
    """
    Class responsible for collecting all needed properties in geothermal simulation
    """
    def __init__(self):
        self.components = ["H2O"]
        self.phases = ['water', 'steam']
        self.nc = len(self.components)
        self.nc_fl = self.nc
        self.nph = len(self.phases)
        self.np_fl = len(self.phases)

        # PH-flash from DARTS-flash
        from dartsflash.libflash import PHFlash, FlashParams, EoSParams
        from dartsflash.libflash import CubicEoS, AQEoS
        from dartsflash.components import CompData
        comp_data = CompData(components=self.components, setprops=True)
        self.Mw = comp_data.Mw
        pr = CubicEoS(comp_data, CubicEoS.PR)
        aq = AQEoS(comp_data, AQEoS.Jager2003)

        flash_params = FlashParams(comp_data)

        # EoS-related parameters
        flash_params.add_eos("PR", pr)
        flash_params.add_eos("AQ", aq)
        flash_params.eos_order = ["AQ", "PR"]

        self.flash_ev = PHFlash(flash_params)

        # properties implemented in python
        from darts.physics.properties.eos_properties import EoSDensity, EoSEnthalpy
        from darts.physics.properties.density import Spivey2004
        from darts.physics.properties.viscosity import MaoDuan2009
        self.enthalpy_ev = {'water': EoSEnthalpy(aq),
                            'steam': EoSEnthalpy(pr),
                            'total': lambda: np.nansum(self.nu * self.enthalpy)}
        self.density_ev = {'water': Spivey2004(self.components),
                           'steam': EoSDensity(pr, comp_data.Mw)}
        self.viscosity_ev = {'water': MaoDuan2009(self.components),
                             'steam': ConstFunc(0.01)}
        self.conduction_ev = {'water': ConstFunc(172.8),
                              'steam': ConstFunc(0.)}
        self.relperm_ev = {'water': iapws_water_relperm_evaluator(),
                           'steam': iapws_steam_relperm_evaluator()}

        self.rock = [value_vector([1, 0, 273.15])]
        self.rock_compaction_ev = custom_rock_compaction_evaluator(self.rock)  # Create rock_compaction object
        self.rock_energy_ev = custom_rock_energy_evaluator(self.rock)  # Create rock_energy object

        # Initialize arrays
        self.nu = np.zeros(self.np_fl)
        self.x = np.zeros((self.np_fl, self.nc_fl))
        self.density = np.zeros(self.nph)
        self.dens_m = np.zeros(self.nph)
        self.saturation = np.zeros(self.nph)
        self.viscosity = np.zeros(self.np_fl)
        self.relperm = np.zeros(self.np_fl)
        self.pc = np.zeros(self.np_fl)
        self.enthalpy = np.zeros(self.nph)
        self.conduction = np.zeros(self.nph)
        self.dX = []
        self.mass_source = np.zeros(self.nc)
        self.energy_source = 0.
        self.temperature = 0.

        self.phase_props = [self.density, self.dens_m, self.saturation, self.nu, self.viscosity, self.relperm, self.pc,
                            self.enthalpy, self.conduction, self.mass_source]

        self.output_props = {'temperature': lambda: self.temperature}

    def run_flash(self, pressure, enthalpy):
        error_output = self.flash_ev.evaluate(pressure, enthalpy)
        flash_results = self.flash_ev.get_flash_results()
        self.nu = np.array(flash_results.nu)
        self.x = np.array(flash_results.X).reshape(self.np_fl, self.nc_fl)
        self.temperature = flash_results.T

        ph = np.array([j for j in range(self.np_fl) if self.nu[j] > 0])

        return ph

    def compute_total_enthalpy(self, state, temperature):
        _ = self.flash_ev.evaluate_PT(state[0], temperature)
        flash_results = self.flash_ev.get_flash_results()
        nu = np.array(flash_results.nu)
        x = np.array(flash_results.X).reshape(self.nph, self.nc)

        ph = np.array([j for j in range(self.nph) if nu[j] > 0])

        enthalpy = 0.
        for j in ph:
            enthalpy += nu[j] * self.enthalpy_ev[self.phases[j]].evaluate(state[0], temperature, x[j, :])

        return enthalpy


    def compute_saturation(self, ph):
        # Get saturations [volume fraction]
        if len(ph) == 1:
            self.saturation[ph] = 1.
        else:
            Vtot = 0
            for j in ph:
                Vtot += self.nu[j] / self.dens_m[j]

            for j in ph:
                self.saturation[j] = (self.nu[j] / self.dens_m[j]) / Vtot

        return

    def evaluate(self, state):
        # Clean arrays
        for a in self.phase_props:
            a[:] = 0

        # Evaluate flash
        self.ph = self.run_flash(state[0], state[1])

        # Evaluate phase properties
        for j in self.ph:
            phase = self.phases[j]
            Mw = np.sum(self.Mw * self.x[j, :])
            self.density[j] = self.density_ev[phase].evaluate(state[0], self.temperature, self.x[j, :])
            self.dens_m[j] = self.density[j] / Mw
            self.viscosity[j] = self.viscosity_ev[phase].evaluate(state[0], self.temperature, self.x[j, :], self.density[j])
            self.enthalpy[j] = self.enthalpy_ev[phase].evaluate(state[0], self.temperature, self.x[j, :])
            self.conduction[j] = self.conduction_ev[phase].evaluate(state)

        # Compute saturation and saturation-based properties
        self.compute_saturation(self.ph)

        # self.pc = self.capillary_pressure_ev.evaluate(self.sat)
        for j in self.ph:
            self.relperm[j] = self.relperm_ev[self.phases[j]].evaluate(self.saturation[j])

        return
