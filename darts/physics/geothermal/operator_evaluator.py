from darts.engines import operator_set_evaluator_iface, value_vector
from darts.physics.geothermal.property_container import PropertyContainer


class acc_flux_custom_iapws_evaluator_python(operator_set_evaluator_iface):
    def __init__(self, property_container: PropertyContainer):
        super().__init__()

        self.property = property_container

    def evaluate(self, state, values):

        water_enth = self.property.water_enthalpy.evaluate(state)
        steam_enth = self.property.steam_enthalpy.evaluate(state)
        water_den  = self.property.water_density.evaluate(state)
        steam_den  = self.property.steam_density.evaluate(state)
        water_sat  = self.property.water_saturation.evaluate(state)
        steam_sat  = self.property.steam_saturation.evaluate(state)
        temp       = self.property.temperature.evaluate(state)
        water_rp   = self.property.water_relperm.evaluate(state)
        steam_rp   = self.property.steam_relperm.evaluate(state)
        water_vis  = self.property.water_viscosity.evaluate(state)
        steam_vis  = self.property.steam_viscosity.evaluate(state)
        pore_volume_factor = self.property.rock_compaction.evaluate(state)
        rock_int_energy    = self.property.rock_energy.evaluate(state)
        water_cond = self.property.water_conduction.evaluate(state)
        steam_cond = self.property.steam_conduction.evaluate(state)
        pressure = state[0]

        # mass accumulation
        values[0] = pore_volume_factor * (water_den * water_sat + steam_den * steam_sat)
        # mass flux
        values[1] = water_den * water_rp / water_vis + steam_den * steam_rp / steam_vis
        # fluid internal energy = water_enthalpy + steam_enthalpy - work
        # (in the following expression, 100 denotes the conversion factor from bars to kJ/m3)
        values[2] = pore_volume_factor * (water_den * water_sat * water_enth + steam_den * steam_sat * steam_enth
                                           - 100 * pressure)
        # rock internal energy
        values[3] = rock_int_energy / pore_volume_factor
        # energy flux
        values[4] = (water_enth * water_den * water_rp / water_vis + steam_enth * steam_den * steam_rp / steam_vis)
        # fluid conduction
        values[5] = water_cond * water_sat + steam_cond * steam_sat
        # rock conduction
        values[6] = 1 / pore_volume_factor
        # temperature
        values[7] = temp

        return 0


class acc_flux_custom_iapws_evaluator_python_well(operator_set_evaluator_iface):
    def __init__(self, property_container: PropertyContainer):
        super().__init__()

        self.property = property_container

    def evaluate(self, state, values):

        water_enth = self.property.water_enthalpy.evaluate(state)
        steam_enth = self.property.steam_enthalpy.evaluate(state)
        water_den  = self.property.water_density.evaluate(state)
        steam_den  = self.property.steam_density.evaluate(state)
        water_sat  = self.property.water_saturation.evaluate(state)
        steam_sat  = self.property.steam_saturation.evaluate(state)
        temp       = self.property.temperature.evaluate(state)
        water_rp   = self.property.water_relperm.evaluate(state)
        steam_rp   = self.property.steam_relperm.evaluate(state)
        water_vis  = self.property.water_viscosity.evaluate(state)
        steam_vis  = self.property.steam_viscosity.evaluate(state)
        pore_volume_factor = self.property.rock_compaction.evaluate(state)
        rock_int_energy    = self.property.rock_energy.evaluate(state)
        pressure = state[0]

        # mass accumulation
        values[0] = pore_volume_factor * (water_den * water_sat + steam_den * steam_sat)
        # mass flux
        values[1] = water_den * water_rp / water_vis + steam_den * steam_rp / steam_vis
        # fluid internal energy = water_enthalpy + steam_enthalpy - work
        # (in the following expression, 100 denotes the conversion factor from bars to kJ/m3)
        values[2] = pore_volume_factor * (water_den * water_sat * water_enth + steam_den * steam_sat * steam_enth
                                           - 100 * pressure)
        # rock internal energy
        values[3] = rock_int_energy / pore_volume_factor
        # energy flux
        values[4] = (water_enth * water_den * water_rp / water_vis + steam_enth * steam_den * steam_rp / steam_vis)
        # fluid conduction
        values[5] = 0.0
        # rock conduction
        values[6] = 1 / pore_volume_factor
        # temperature
        values[7] = temp

        return 0


class acc_flux_gravity_evaluator_python(operator_set_evaluator_iface):
    def __init__(self, property_container: PropertyContainer):
        super().__init__()

        self.property = property_container

    def evaluate(self, state, values):
        water_enth = self.property.water_enthalpy.evaluate(state)
        steam_enth = self.property.steam_enthalpy.evaluate(state)
        water_den = self.property.water_density.evaluate(state)
        steam_den = self.property.steam_density.evaluate(state)
        water_sat = self.property.water_saturation.evaluate(state)
        steam_sat = self.property.steam_saturation.evaluate(state)
        temp = self.property.temperature.evaluate(state)
        water_rp = self.property.water_relperm.evaluate(state)
        steam_rp = self.property.steam_relperm.evaluate(state)
        water_vis = self.property.water_viscosity.evaluate(state)
        steam_vis = self.property.steam_viscosity.evaluate(state)
        pore_volume_factor = self.property.rock_compaction.evaluate(state)
        rock_int_energy = self.property.rock_energy.evaluate(state)
        water_cond = self.property.water_conduction.evaluate(state)
        steam_cond = self.property.steam_conduction.evaluate(state)
        pressure = state[0]

        # mass accumulation
        values[0] = pore_volume_factor * (water_den * water_sat + steam_den * steam_sat)
        # mass flux
        values[1] = water_den * water_rp / water_vis
        values[2] = steam_den * steam_rp / steam_vis
        # fluid internal energy = water_enthalpy + steam_enthalpy - work
        # (in the following expression, 100 denotes the conversion factor from bars to kJ/m3)
        values[3] = pore_volume_factor * (water_den * water_sat * water_enth + steam_den * steam_sat * steam_enth
                                          - 100 * pressure)
        # rock internal energy
        values[4] = rock_int_energy / pore_volume_factor
        # energy flux
        values[5] = water_enth * water_den * water_rp / water_vis
        values[6] = steam_enth * steam_den * steam_rp / steam_vis
        # fluid conduction
        values[7] = water_cond * water_sat + steam_cond * steam_sat
        # rock conduction
        values[8] = 1 / pore_volume_factor
        # temperature
        values[9] = temp
        # water density
        values[10] = water_den
        # steam density
        values[11] = steam_den

        return 0


class acc_flux_gravity_evaluator_python_well(operator_set_evaluator_iface):
    def __init__(self, property_container: PropertyContainer):
        super().__init__()

        self.property = property_container

    def evaluate(self, state, values):
        water_enth = self.property.water_enthalpy.evaluate(state)
        steam_enth = self.property.steam_enthalpy.evaluate(state)
        water_den = self.property.water_density.evaluate(state)
        steam_den = self.property.steam_density.evaluate(state)
        water_sat = self.property.water_saturation.evaluate(state)
        steam_sat = self.property.steam_saturation.evaluate(state)
        temp = self.property.temperature.evaluate(state)
        water_rp = self.property.water_relperm.evaluate(state)
        steam_rp = self.property.steam_relperm.evaluate(state)
        water_vis = self.property.water_viscosity.evaluate(state)
        steam_vis = self.property.steam_viscosity.evaluate(state)
        pore_volume_factor = self.property.rock_compaction.evaluate(state)
        rock_int_energy = self.property.rock_energy.evaluate(state)
        pressure = state[0]

        # mass accumulation
        values[0] = pore_volume_factor * (water_den * water_sat + steam_den * steam_sat)
        # mass flux
        values[1] = water_den * water_rp / water_vis
        values[2] = steam_den * steam_rp / steam_vis
        # fluid internal energy = water_enthalpy + steam_enthalpy - work
        # (in the following expression, 100 denotes the conversion factor from bars to kJ/m3)
        values[3] = pore_volume_factor * (water_den * water_sat * water_enth + steam_den * steam_sat * steam_enth
                                          - 100 * pressure)
        # rock internal energy
        values[4] = rock_int_energy / pore_volume_factor
        # energy flux
        values[5] = water_enth * water_den * water_rp / water_vis
        values[6] = steam_enth * steam_den * steam_rp / steam_vis
        # fluid conduction
        values[7] = 0.0
        # rock conduction
        values[8] = 1 / pore_volume_factor
        # temperature
        values[9] = temp
        # water density
        values[10] = water_den
        # steam density
        values[11] = steam_den

        return 0


class geothermal_rate_custom_evaluator_python(operator_set_evaluator_iface):
    def __init__(self, property_container: PropertyContainer):
        super().__init__()

        self.property = property_container

    def evaluate(self, state, values):
        water_den = self.property.water_density.evaluate(state)
        steam_den = self.property.steam_density.evaluate(state)
        water_sat = self.property.water_saturation.evaluate(state)
        steam_sat = self.property.steam_saturation.evaluate(state)
        water_rp  = self.property.water_relperm.evaluate(state)
        steam_rp  = self.property.steam_relperm.evaluate(state)
        water_vis = self.property.water_viscosity.evaluate(state)
        steam_vis = self.property.steam_viscosity.evaluate(state)
        water_enth = self.property.water_enthalpy.evaluate(state)
        steam_enth = self.property.steam_enthalpy.evaluate(state)
        temp = self.property.temperature.evaluate(state)

        total_density = water_sat * water_den + steam_sat * steam_den

        # water volumetric rate
        values[0] = water_sat * (water_den * water_rp / water_vis + steam_den * steam_rp / steam_vis) / total_density
        # steam volumetric rate
        values[1] = steam_sat * (water_den * water_rp / water_vis + steam_den * steam_rp / steam_vis) / total_density
        # temperature
        values[2] = temp
        # energy rate
        values[3] = water_enth * water_den * water_rp / water_vis + steam_enth * steam_den * steam_rp / steam_vis

        return 0


class geothermal_mass_rate_custom_evaluator_python(operator_set_evaluator_iface):
    def __init__(self, property_container: PropertyContainer):
        super().__init__()

        self.property = property_container

    def evaluate(self, state, values):
        water_den = self.property.water_density.evaluate(state)
        steam_den = self.property.steam_density.evaluate(state)
        water_sat = self.property.water_saturation.evaluate(state)
        steam_sat = self.property.steam_saturation.evaluate(state)
        water_rp  = self.property.water_relperm.evaluate(state)
        steam_rp  = self.property.steam_relperm.evaluate(state)
        water_vis = self.property.water_viscosity.evaluate(state)
        steam_vis = self.property.steam_viscosity.evaluate(state)
        temp      = self.property.temperature.evaluate(state)
        water_enth= self.property.water_enthalpy.evaluate(state)
        steam_enth= self.property.steam_enthalpy.evaluate(state)

        total_density = water_sat * water_den + steam_sat * steam_den

        # water mass rate
        values[0] = water_den * water_rp / water_vis + steam_den * steam_rp / steam_vis
        # steam mass rate
        values[1] = steam_sat * (water_den * water_rp / water_vis + steam_den * steam_rp / steam_vis) / total_density
        # temperature
        values[2] = temp
        # energy rate
        values[3] = water_enth * water_den * water_rp / water_vis + steam_enth * steam_den * steam_rp / steam_vis
        
        return 0
