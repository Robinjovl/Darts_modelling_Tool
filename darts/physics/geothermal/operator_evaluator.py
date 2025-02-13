import numpy as np
from darts.physics.base.operators_base import OperatorsBase


class OperatorsGeothermal(OperatorsBase):
    def __init__(self, property_container, thermal: bool = True):
        super().__init__(property_container, thermal)


class acc_flux_custom_iapws_evaluator_python(OperatorsGeothermal):
    n_ops = 6

    def evaluate(self, state, values):
        pressure = state[0]
        pc = self.property
        pc.evaluate(state)

        pore_volume_factor = pc.rock_compaction_ev.evaluate(state)

        # mass accumulation
        values[0] = pore_volume_factor * np.sum(pc.dens_m[pc.ph] * pc.saturation[pc.ph])
        # mass flux
        values[1] = np.sum(pc.dens_m[pc.ph] * pc.kr[pc.ph] / pc.mu[pc.ph])
        # fluid internal energy = water_enthalpy + steam_enthalpy - work
        # (in the following expression, 100 denotes the conversion factor from bars to kJ/m3)
        values[2] = pore_volume_factor * (np.sum(pc.dens_m[pc.ph] * pc.saturation[pc.ph] * pc.enthalpy[pc.ph])
                                          - 100 * pressure)
        # energy flux
        values[3] = np.sum(pc.enthalpy[pc.ph] * pc.dens_m[pc.ph] * pc.kr[pc.ph] / pc.mu[pc.ph])
        # fluid conduction
        values[4] = np.sum(pc.conduction[pc.ph] * pc.saturation[pc.ph])
        # temperature
        values[5] = pc.temperature

        return 0


class acc_flux_custom_iapws_evaluator_python_well(OperatorsGeothermal):
    n_ops = 6

    def evaluate(self, state, values):
        pressure = state[0]
        pc = self.property
        pc.evaluate(state)

        pore_volume_factor = pc.rock_compaction_ev.evaluate(state)

        # mass accumulation
        values[0] = pore_volume_factor * np.sum(pc.dens_m[pc.ph] * pc.saturation[pc.ph])
        # mass flux
        values[1] = np.sum(pc.dens_m[pc.ph] * pc.kr[pc.ph] / pc.mu[pc.ph])
        # fluid internal energy = water_enthalpy + steam_enthalpy - work
        # (in the following expression, 100 denotes the conversion factor from bars to kJ/m3)
        values[2] = pore_volume_factor * (np.sum(pc.dens_m[pc.ph] * pc.saturation[pc.ph] * pc.enthalpy[pc.ph])
                                          - 100 * pressure)
        # energy flux
        values[3] = np.sum(pc.enthalpy[pc.ph] * pc.dens_m[pc.ph] * pc.kr[pc.ph] / pc.mu[pc.ph])
        # fluid conduction
        values[4] = 0.0
        # temperature
        values[5] = pc.temperature

        return 0


class acc_flux_gravity_evaluator_python(OperatorsGeothermal):
    n_ops = 10

    def evaluate(self, state, values):
        pressure = state[0]
        pc = self.property
        pc.evaluate(state)

        pore_volume_factor = pc.rock_compaction_ev.evaluate(state)

        # mass accumulation
        values[0] = pore_volume_factor * np.sum(pc.dens_m[pc.ph] * pc.saturation[pc.ph])
        # mass flux
        values[1] = pc.dens_m[0] * pc.kr[0] / pc.mu[0] if 0 in pc.ph else 0.
        values[2] = pc.dens_m[1] * pc.kr[1] / pc.mu[1] if 1 in pc.ph else 0.
        # fluid internal energy = water_enthalpy + steam_enthalpy - work
        # (in the following expression, 100 denotes the conversion factor from bars to kJ/m3)
        values[3] = pore_volume_factor * (np.sum(pc.dens_m[pc.ph] * pc.saturation[pc.ph] * pc.enthalpy[pc.ph])
                                          - 100 * pressure)
        # energy flux
        values[4] = pc.enthalpy[0] * pc.dens_m[0] * pc.kr[0] / pc.mu[0] if 0 in pc.ph else 0.
        values[5] = pc.enthalpy[1] * pc.dens_m[1] * pc.kr[1] / pc.mu[1] if 1 in pc.ph else 0.
        # fluid conduction
        values[6] = np.sum(pc.conduction[pc.ph] * pc.saturation[pc.ph])
        # water density
        values[7] = pc.dens_m[0] if 0 in pc.ph else 0.
        # steam density
        values[8] = pc.dens_m[1] if 1 in pc.ph else 0.
        # temperature
        values[9] = pc.temperature

        return 0


class acc_flux_gravity_evaluator_python_well(OperatorsGeothermal):
    n_ops = 10

    def evaluate(self, state, values):
        pressure = state[0]
        pc = self.property
        pc.evaluate(state)

        pore_volume_factor = pc.rock_compaction_ev.evaluate(state)

        # mass accumulation
        values[0] = pore_volume_factor * np.sum(pc.dens_m[pc.ph] * pc.saturation[pc.ph])
        # mass flux
        values[1] = pc.dens_m[0] * pc.kr[0] / pc.mu[0] if 0 in pc.ph else 0.
        values[2] = pc.dens_m[1] * pc.kr[1] / pc.mu[1] if 1 in pc.ph else 0.
        # fluid internal energy = water_enthalpy + steam_enthalpy - work
        # (in the following expression, 100 denotes the conversion factor from bars to kJ/m3)
        values[3] = pore_volume_factor * (np.sum(pc.dens_m[pc.ph] * pc.saturation[pc.ph] * pc.enthalpy[pc.ph])
                                          - 100 * pressure)
        # energy flux
        values[4] = pc.enthalpy[0] * pc.dens_m[0] * pc.kr[0] / pc.mu[0] if 0 in pc.ph else 0.
        values[5] = pc.enthalpy[1] * pc.dens_m[1] * pc.kr[1] / pc.mu[1] if 1 in pc.ph else 0.
        # fluid conduction
        values[6] = 0.0
        # water density
        values[7] = pc.dens_m[0] if 0 in pc.ph else 0.
        # steam density
        values[8] = pc.dens_m[1] if 1 in pc.ph else 0.
        # temperature
        values[9] = pc.temperature

        return 0


class MassFluxOperators(OperatorsGeothermal):
    n_ops = 1

    def evaluate(self, state, values):
        pc = self.property
        pc.evaluate(state)

        """ Beta operator here represents mass flux term: """
        values[0] = np.sum(pc.dens_m[pc.ph] * pc.kr[pc.ph] / pc.mu[pc.ph])

        return 0
