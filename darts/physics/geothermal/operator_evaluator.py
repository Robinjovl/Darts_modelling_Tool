from darts.engines import operator_set_evaluator_iface, value_vector
from darts.physics.operators_base import OperatorsBase
from darts.physics.geothermal.property_container import PropertyContainer


class OperatorsGeothermal(OperatorsBase):
    def __init__(self, property_container: PropertyContainer, thermal: bool = True):
        super().__init__(property_container, thermal)


class ReservoirOperators(OperatorsGeothermal):
    n_ops = 10

    def evaluate(self, state, values):
        pressure = state[0]
        pc = self.property
        pc.evaluate(state)

        pore_volume_factor = pc.rock_compaction_ev.evaluate(state)

        # mass accumulation
        values[0] = pore_volume_factor * (pc.density[0] * pc.saturation[0] + pc.density[1] * pc.saturation[1])
        # mass flux
        values[1] = pc.density[0] * pc.relperm[0] / pc.viscosity[0]
        values[2] = pc.density[1] * pc.relperm[1] / pc.viscosity[1]
        # fluid internal energy = water_enthalpy + steam_enthalpy - work
        # (in the following expression, 100 denotes the conversion factor from bars to kJ/m3)
        values[3] = pore_volume_factor * (pc.density[0] * pc.saturation[0] * pc.enthalpy[0] +
                                          pc.density[1] * pc.saturation[1] * pc.enthalpy[1] - 100 * pressure)
        # energy flux
        values[4] = pc.enthalpy[0] * pc.density[0] * pc.relperm[0] / pc.viscosity[0]
        values[5] = pc.enthalpy[1] * pc.density[1] * pc.relperm[1] / pc.viscosity[1]
        # fluid conduction
        values[6] = pc.conduction[0] * pc.saturation[0] + pc.conduction[1] * pc.saturation[1]
        # water density
        values[7] = pc.density[0]
        # steam density
        values[8] = pc.density[1]
        # temperature
        values[9] = pc.temperature

        return 0


class WellOperators(OperatorsGeothermal):
    n_ops = 10

    def evaluate(self, state, values):
        pressure = state[0]
        pc = self.property
        pc.evaluate(state)

        pore_volume_factor = pc.rock_compaction_ev.evaluate(state)

        # mass accumulation
        values[0] = pore_volume_factor * (pc.density[0] * pc.saturation[0] + pc.density[1] * pc.saturation[1])
        # mass flux
        values[1] = pc.density[0] * pc.relperm[0] / pc.viscosity[0]
        values[2] = pc.density[1] * pc.relperm[1] / pc.viscosity[1]
        # fluid internal energy = water_enthalpy + steam_enthalpy - work
        # (in the following expression, 100 denotes the conversion factor from bars to kJ/m3)
        values[3] = pore_volume_factor * (pc.density[0] * pc.saturation[0] * pc.enthalpy[0] +
                                          pc.density[1] * pc.saturation[1] * pc.enthalpy[1] - 100 * pressure)
        # energy flux
        values[4] = pc.enthalpy[0] * pc.density[0] * pc.relperm[0] / pc.viscosity[0]
        values[5] = pc.enthalpy[1] * pc.density[1] * pc.relperm[1] / pc.viscosity[1]
        # fluid conduction
        values[6] = 0.0
        # water density
        values[7] = pc.density[0]
        # steam density
        values[8] = pc.density[1]
        # temperature
        values[9] = pc.temperature

        return 0


class RateOperators(OperatorsGeothermal):
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


class MassRateOperators(OperatorsGeothermal):
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
