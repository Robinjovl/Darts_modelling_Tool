import numpy as np
from darts.engines import value_vector
from darts.physics.property_base import PropertyBase

from darts.physics.properties.iapws.iapws_property import *
from darts.physics.properties.iapws.custom_rock_property import *
from darts.physics.properties.basic import ConstFunc


class PropertyContainer(PropertyBase):
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
                                'total': iapws_total_enthalpy_evalutor}
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
            self.saturation[j] = self.saturation_ev[phase].evaluate(state)
            self.viscosity[j] = self.viscosity_ev[phase].evaluate(state)
            self.conduction[j] = self.conduction_ev[phase].evaluate(state)
            self.relperm[j] = self.relperm_ev[phase].evaluate(state)
        return
