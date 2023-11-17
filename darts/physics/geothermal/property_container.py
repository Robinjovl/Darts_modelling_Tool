from darts.engines import value_vector
from darts.physics.properties.iapws.iapws_property import *
from darts.physics.properties.iapws.custom_rock_property import *
from darts.physics.properties.basic import ConstFunc


class PropertyContainer:
    '''
    Class resposible for collecting all needed properties in geothermal simulation
    '''
    def __init__(self, property_evaluator='IAPWS'):
        """
        Constructor
        :param property_evaluator: determines what property evaluator is used, input is either 'IAPWS' or 'ADGPRS'
        """
        self.rock = [value_vector([1, 0, 273.15])]
        # properties implemented in C++
        if property_evaluator == 'ADGPRS':
            self.sat_steam_enthalpy = saturated_steam_enthalpy_evaluator()
            self.sat_water_enthalpy = saturated_water_enthalpy_evaluator()
            self.water_enthalpy = water_enthalpy_evaluator(self.sat_water_enthalpy, self.sat_steam_enthalpy)
            self.steam_enthalpy = steam_enthalpy_evaluator(self.sat_steam_enthalpy, self.sat_water_enthalpy)
            self.sat_water_density = saturated_water_density_evaluator(self.sat_water_enthalpy)
            self.sat_steam_density = saturated_steam_density_evaluator(self.sat_steam_enthalpy)
            self.water_density = water_density_evaluator(self.sat_water_enthalpy, self.sat_steam_enthalpy)
            self.steam_density = steam_density_evaluator(self.sat_water_enthalpy, self.sat_steam_enthalpy)
            self.temperature = temperature_evaluator(self.sat_water_enthalpy, self.sat_steam_enthalpy)
            self.water_saturation = water_saturation_evaluator(self.sat_water_density, self.sat_steam_density,
                                                               self.sat_water_enthalpy, self.sat_steam_enthalpy)
            self.steam_saturation = steam_saturation_evaluator(self.sat_water_density, self.sat_steam_density,
                                                               self.sat_water_enthalpy, self.sat_steam_enthalpy)
            self.water_viscosity = water_viscosity_evaluator(self.temperature)
            self.steam_viscosity = steam_viscosity_evaluator(self.temperature)
            self.water_relperm = water_relperm_evaluator(self.water_saturation)
            self.steam_relperm = steam_relperm_evaluator(self.steam_saturation)
            self.rock_compaction = custom_rock_compaction_evaluator(self.rock)
            self.rock_energy = custom_rock_energy_evaluator(self.rock)
            self.water_conduction = ConstFunc(172.8)
            self.steam_conduction = ConstFunc(0)
            

        elif property_evaluator == 'IAPWS':
            # properties implemented in python (the IAPWS package)
            self.temperature = iapws_temperature_evaluator()                    # Create temperature object
            self.water_enthalpy = iapws_water_enthalpy_evaluator()              # Create water_enthalpy object
            self.steam_enthalpy = iapws_steam_enthalpy_evaluator()              # Create steam_enthalpy object
            self.total_enthalpy = iapws_total_enthalpy_evalutor
            self.water_saturation = iapws_water_saturation_evaluator()          # Create water_saturation object
            self.steam_saturation = iapws_steam_saturation_evaluator()          # Create steam_saturation object
            self.water_relperm = iapws_water_relperm_evaluator()                # Create water_relperm object
            self.steam_relperm = iapws_steam_relperm_evaluator()                # Create steam_relperm object
            self.water_density = iapws_water_density_evaluator()                # Create water_density object
            self.steam_density = iapws_steam_density_evaluator()                # Create steam_density object
            self.water_viscosity = iapws_water_viscosity_evaluator()            # Create water_viscosity object
            self.steam_viscosity = iapws_steam_viscosity_evaluator()            # Create steam_viscosity object
            self.rock_compaction = custom_rock_compaction_evaluator(self.rock)  # Create rock_compaction object
            self.rock_energy = custom_rock_energy_evaluator(self.rock)          # Create rock_energy object
            self.water_conduction = ConstFunc(172.8)
            self.steam_conduction = ConstFunc(0)