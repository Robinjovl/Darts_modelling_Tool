from darts.engines import operator_set_evaluator_iface, value_vector
from darts.physics.operators_base import OperatorsBase
from darts.physics.geothermal.property_container import PropertyContainer


class OperatorsGeothermal(OperatorsBase):
    def __init__(self, property_container: PropertyContainer, thermal: bool = True):
        super().__init__(property_container, thermal)


class acc_flux_custom_iapws_evaluator_python(OperatorsGeothermal):
    n_ops = 8

    def evaluate(self, state, values):
        pressure = state[0]
        pc = self.property
        pc.evaluate(state)

        pore_volume_factor = pc.rock_compaction_ev.evaluate(state)
        rock_int_energy    = pc.rock_energy_ev.evaluate(state)

        # mass accumulation
        values[0] = pore_volume_factor * (pc.density[0] * pc.saturation[0] + pc.density[1] * pc.saturation[1])
        # mass flux
        values[1] = pc.density[0] * pc.relperm[0] / pc.viscosity[0] + pc.density[1] * pc.relperm[1] / pc.viscosity[1]
        # fluid internal energy = water_enthalpy + steam_enthalpy - work
        # (in the following expression, 100 denotes the conversion factor from bars to kJ/m3)
        values[2] = pore_volume_factor * (pc.density[0] * pc.saturation[0] * pc.enthalpy[0] +
                                          pc.density[1] * pc.saturation[1] * pc.enthalpy[1] - 100 * pressure)
        # rock internal energy
        values[3] = rock_int_energy / pore_volume_factor
        # energy flux
        values[4] = pc.enthalpy[0] * pc.density[0] * pc.relperm[0] / pc.viscosity[0] +\
                    pc.enthalpy[1] * pc.density[1] * pc.relperm[1] / pc.viscosity[1]
        # fluid conduction
        values[5] = pc.conduction[0] * pc.saturation[0] + pc.conduction[1] * pc.saturation[1]
        # rock conduction
        values[6] = 1 / pore_volume_factor
        # temperature
        values[7] = pc.temperature

        return 0


class acc_flux_custom_iapws_evaluator_python_well(OperatorsGeothermal):
    n_ops = 8

    def evaluate(self, state, values):
        pressure = state[0]
        pc = self.property
        pc.evaluate(state)

        pore_volume_factor = pc.rock_compaction_ev.evaluate(state)
        rock_int_energy = pc.rock_energy_ev.evaluate(state)

        # mass accumulation
        values[0] = pore_volume_factor * (pc.density[0] * pc.saturation[0] + pc.density[1] * pc.saturation[1])
        # mass flux
        values[1] = pc.density[0] * pc.relperm[0] / pc.viscosity[0] + pc.density[1] * pc.relperm[1] / pc.viscosity[1]
        # fluid internal energy = water_enthalpy + steam_enthalpy - work
        # (in the following expression, 100 denotes the conversion factor from bars to kJ/m3)
        values[2] = pore_volume_factor * (pc.density[0] * pc.saturation[0] * pc.enthalpy[0] +
                                          pc.density[1] * pc.saturation[1] * pc.enthalpy[1] - 100 * pressure)
        # rock internal energy
        values[3] = rock_int_energy / pore_volume_factor
        # energy flux
        values[4] = pc.enthalpy[0] * pc.density[0] * pc.relperm[0] / pc.viscosity[0] + \
                    pc.enthalpy[1] * pc.density[1] * pc.relperm[1] / pc.viscosity[1]
        # fluid conduction
        values[5] = 0.0
        # rock conduction
        values[6] = 1 / pore_volume_factor
        # temperature
        values[7] = pc.temperature

        return 0


class acc_flux_gravity_evaluator_python(OperatorsGeothermal):
    n_ops = 12

    def evaluate(self, state, values):
        pressure = state[0]
        pc = self.property
        pc.evaluate(state)

        pore_volume_factor = pc.rock_compaction_ev.evaluate(state)
        rock_int_energy = pc.rock_energy_ev.evaluate(state)

        # mass accumulation
        values[0] = pore_volume_factor * (pc.density[0] * pc.saturation[0] + pc.density[1] * pc.saturation[1])
        # mass flux
        values[1] = pc.density[0] * pc.relperm[0] / pc.viscosity[0]
        values[2] = pc.density[1] * pc.relperm[1] / pc.viscosity[1]
        # fluid internal energy = water_enthalpy + steam_enthalpy - work
        # (in the following expression, 100 denotes the conversion factor from bars to kJ/m3)
        values[3] = pore_volume_factor * (pc.density[0] * pc.saturation[0] * pc.enthalpy[0] +
                                          pc.density[1] * pc.saturation[1] * pc.enthalpy[1] - 100 * pressure)
        # rock internal energy
        values[4] = rock_int_energy / pore_volume_factor
        # energy flux
        values[5] = pc.enthalpy[0] * pc.density[0] * pc.relperm[0] / pc.viscosity[0]
        values[6] = pc.enthalpy[1] * pc.density[1] * pc.relperm[1] / pc.viscosity[1]
        # fluid conduction
        values[7] = pc.conduction[0] * pc.saturation[0] + pc.conduction[1] * pc.saturation[1]
        # rock conduction
        values[8] = 1 / pore_volume_factor
        # temperature
        values[9] = pc.temperature
        # water density
        values[10] = pc.density[0]
        # steam density
        values[11] = pc.density[1]

        return 0


class acc_flux_gravity_evaluator_python_well(OperatorsGeothermal):
    n_ops = 12

    def evaluate(self, state, values):
        pressure = state[0]
        pc = self.property
        pc.evaluate(state)

        pore_volume_factor = pc.rock_compaction_ev.evaluate(state)
        rock_int_energy = pc.rock_energy_ev.evaluate(state)

        # mass accumulation
        values[0] = pore_volume_factor * (pc.density[0] * pc.saturation[0] + pc.density[1] * pc.saturation[1])
        # mass flux
        values[1] = pc.density[0] * pc.relperm[0] / pc.viscosity[0]
        values[2] = pc.density[1] * pc.relperm[1] / pc.viscosity[1]
        # fluid internal energy = water_enthalpy + steam_enthalpy - work
        # (in the following expression, 100 denotes the conversion factor from bars to kJ/m3)
        values[3] = pore_volume_factor * (pc.density[0] * pc.saturation[0] * pc.enthalpy[0] +
                                          pc.density[1] * pc.saturation[1] * pc.enthalpy[1] - 100 * pressure)
        # rock internal energy
        values[4] = rock_int_energy / pore_volume_factor
        # energy flux
        values[5] = pc.enthalpy[0] * pc.density[0] * pc.relperm[0] / pc.viscosity[0]
        values[6] = pc.enthalpy[1] * pc.density[1] * pc.relperm[1] / pc.viscosity[1]
        # fluid conduction
        values[7] = 0.0
        # rock conduction
        values[8] = 1 / pore_volume_factor
        # temperature
        values[9] = pc.temperature
        # water density
        values[10] = pc.density[0]
        # steam density
        values[11] = pc.density[1]

        return 0


class geothermal_rate_custom_evaluator_python(OperatorsGeothermal):
    n_ops = 4

    def evaluate(self, state, values):
        pc = self.property
        pc.evaluate(state)

        total_density = pc.saturation[0] * pc.density[0] + pc.saturation[1] * pc.density[1]
        total_flux = (pc.density[0] * pc.relperm[0] / pc.viscosity[0] + pc.density[1] * pc.relperm[1] / pc.viscosity[1]) / total_density

        # water volumetric rate
        values[0] = pc.saturation[0] * total_flux
        # steam volumetric rate
        values[1] = pc.saturation[1] * total_flux
        # temperature
        values[2] = pc.temperature
        # energy rate
        values[3] = pc.enthalpy[0] * pc.density[0] * pc.relperm[0] / pc.viscosity[0] +\
                    pc.enthalpy[1] * pc.density[1] * pc.relperm[1] / pc.viscosity[1]

        return 0


class geothermal_mass_rate_custom_evaluator_python(OperatorsGeothermal):
    n_ops = 4

    def evaluate(self, state, values):
        pc = self.property
        pc.evaluate(state)

        total_density = pc.saturation[0] * pc.density[0] + pc.saturation[1] * pc.density[1]

        # water mass rate
        values[0] = pc.density[0] * pc.relperm[0] / pc.viscosity[0] + pc.density[1] * pc.relperm[1] / pc.viscosity[1]
        # steam mass rate
        values[1] = pc.saturation[1] * (pc.density[0] * pc.relperm[0] / pc.viscosity[0]
                                        + pc.density[0] * pc.relperm[0] / pc.viscosity[0]) / total_density
        # temperature
        values[2] = pc.temperature
        # energy rate
        values[3] = pc.enthalpy[0] * pc.density[0] * pc.relperm[0] / pc.viscosity[0] + \
                    pc.enthalpy[1] * pc.density[1] * pc.relperm[1] / pc.viscosity[1]
        
        return 0

class MassFluxOperators(OperatorsGeothermal):
    n_ops = 1

    def evaluate(self, state, values):
        pc = self.property
        pc.evaluate(state)

        """ Beta operator here represents mass flux term: """
        values[0] = pc.density[0] * pc.relperm[0] / pc.viscosity[0] + pc.density[1] * pc.relperm[1] / pc.viscosity[1]

        return 0
