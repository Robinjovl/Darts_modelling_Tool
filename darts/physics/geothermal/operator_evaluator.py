import numpy as np

from darts.physics.base.operators_base import OperatorsBase


class OperatorsGeothermal(OperatorsBase):
    def __init__(self, property_container, thermal: bool = True):
        super().__init__(property_container, thermal)


class acc_flux_custom_iapws_evaluator_python(OperatorsGeothermal):
    n_ops = 6

    def evaluate(self, state, values):
        # State and Values vectors to numpy:
        vec_state_as_np = state.to_numpy()
        vec_values_as_np = values.to_numpy()
        vec_values_as_np[:] = 0

        pressure = vec_state_as_np[0]
        pc = self.property
        pc.evaluate(vec_state_as_np)

        pore_volume_factor = pc.rock_compaction_ev.evaluate(vec_state_as_np)

        # mass accumulation
        vec_values_as_np[0] = pore_volume_factor * np.sum(
            pc.dens_m[pc.ph] * pc.saturation[pc.ph]
        )
        # mass flux
        vec_values_as_np[1] = np.sum(pc.dens_m[pc.ph] * pc.kr[pc.ph] / pc.mu[pc.ph])
        # fluid internal energy = water_enthalpy + steam_enthalpy - work
        # (in the following expression, 100 denotes the conversion factor from bars to kJ/m3)
        vec_values_as_np[2] = pore_volume_factor * (
            np.sum(pc.dens_m[pc.ph] * pc.saturation[pc.ph] * pc.enthalpy[pc.ph])
            - 100 * pressure
        )
        # energy flux
        vec_values_as_np[3] = np.sum(
            pc.enthalpy[pc.ph] * pc.dens_m[pc.ph] * pc.kr[pc.ph] / pc.mu[pc.ph]
        )
        # fluid conduction
        vec_values_as_np[4] = np.sum(pc.conduction[pc.ph] * pc.saturation[pc.ph])
        # temperature
        vec_values_as_np[5] = pc.temperature

        return 0


class acc_flux_custom_iapws_evaluator_python_well(OperatorsGeothermal):
    n_ops = 6

    def evaluate(self, state, values):
        # State and Values vectors to numpy:
        vec_state_as_np = state.to_numpy()
        vec_values_as_np = values.to_numpy()
        vec_values_as_np[:] = 0

        pressure = vec_state_as_np[0]
        pc = self.property
        pc.evaluate(vec_state_as_np)

        pore_volume_factor = pc.rock_compaction_ev.evaluate(vec_state_as_np)

        # mass accumulation
        vec_values_as_np[0] = pore_volume_factor * np.sum(
            pc.dens_m[pc.ph] * pc.saturation[pc.ph]
        )
        # mass flux
        vec_values_as_np[1] = np.sum(pc.dens_m[pc.ph] * pc.kr[pc.ph] / pc.mu[pc.ph])
        # fluid internal energy = water_enthalpy + steam_enthalpy - work
        # (in the following expression, 100 denotes the conversion factor from bars to kJ/m3)
        vec_values_as_np[2] = pore_volume_factor * (
            np.sum(pc.dens_m[pc.ph] * pc.saturation[pc.ph] * pc.enthalpy[pc.ph])
            - 100 * pressure
        )
        # energy flux
        vec_values_as_np[3] = np.sum(
            pc.enthalpy[pc.ph] * pc.dens_m[pc.ph] * pc.kr[pc.ph] / pc.mu[pc.ph]
        )
        # fluid conduction
        vec_values_as_np[4] = 0.0
        # temperature
        vec_values_as_np[5] = pc.temperature

        return 0


class acc_flux_gravity_evaluator_python(OperatorsGeothermal):
    n_ops = 10

    def evaluate(self, state, values):
        # State and Values vectors to numpy:
        vec_state_as_np = state.to_numpy()
        vec_values_as_np = values.to_numpy()
        vec_values_as_np[:] = 0

        pressure = vec_state_as_np[0]
        pc = self.property
        pc.evaluate(vec_state_as_np)

        pore_volume_factor = pc.rock_compaction_ev.evaluate(vec_state_as_np)

        # mass accumulation
        vec_values_as_np[0] = pore_volume_factor * np.sum(
            pc.dens_m[pc.ph] * pc.saturation[pc.ph]
        )
        # mass flux
        vec_values_as_np[1] = pc.dens_m[0] * pc.kr[0] / pc.mu[0] if 0 in pc.ph else 0.0
        vec_values_as_np[2] = pc.dens_m[1] * pc.kr[1] / pc.mu[1] if 1 in pc.ph else 0.0
        # fluid internal energy = water_enthalpy + steam_enthalpy - work
        # (in the following expression, 100 denotes the conversion factor from bars to kJ/m3)
        vec_values_as_np[3] = pore_volume_factor * (
            np.sum(pc.dens_m[pc.ph] * pc.saturation[pc.ph] * pc.enthalpy[pc.ph])
            - 100 * pressure
        )
        # energy flux
        vec_values_as_np[4] = (
            pc.enthalpy[0] * pc.dens_m[0] * pc.kr[0] / pc.mu[0] if 0 in pc.ph else 0.0
        )
        vec_values_as_np[5] = (
            pc.enthalpy[1] * pc.dens_m[1] * pc.kr[1] / pc.mu[1] if 1 in pc.ph else 0.0
        )
        # fluid conduction
        vec_values_as_np[6] = np.sum(pc.conduction[pc.ph] * pc.saturation[pc.ph])
        # water density
        vec_values_as_np[7] = pc.dens_m[0] if 0 in pc.ph else 0.0
        # steam density
        vec_values_as_np[8] = pc.dens_m[1] if 1 in pc.ph else 0.0
        # temperature
        vec_values_as_np[9] = pc.temperature

        return 0


class acc_flux_gravity_evaluator_python_well(OperatorsGeothermal):
    n_ops = 10

    def evaluate(self, state, values):
        # State and Values vectors to numpy:
        vec_state_as_np = state.to_numpy()
        vec_values_as_np = values.to_numpy()
        vec_values_as_np[:] = 0

        pressure = vec_state_as_np[0]
        pc = self.property
        pc.evaluate(vec_state_as_np)

        pore_volume_factor = pc.rock_compaction_ev.evaluate(vec_state_as_np)

        # mass accumulation
        vec_values_as_np[0] = pore_volume_factor * np.sum(
            pc.dens_m[pc.ph] * pc.saturation[pc.ph]
        )
        # mass flux
        vec_values_as_np[1] = pc.dens_m[0] * pc.kr[0] / pc.mu[0] if 0 in pc.ph else 0.0
        vec_values_as_np[2] = pc.dens_m[1] * pc.kr[1] / pc.mu[1] if 1 in pc.ph else 0.0
        # fluid internal energy = water_enthalpy + steam_enthalpy - work
        # (in the following expression, 100 denotes the conversion factor from bars to kJ/m3)
        vec_values_as_np[3] = pore_volume_factor * (
            np.sum(pc.dens_m[pc.ph] * pc.saturation[pc.ph] * pc.enthalpy[pc.ph])
            - 100 * pressure
        )
        # energy flux
        vec_values_as_np[4] = (
            pc.enthalpy[0] * pc.dens_m[0] * pc.kr[0] / pc.mu[0] if 0 in pc.ph else 0.0
        )
        vec_values_as_np[5] = (
            pc.enthalpy[1] * pc.dens_m[1] * pc.kr[1] / pc.mu[1] if 1 in pc.ph else 0.0
        )
        # fluid conduction
        vec_values_as_np[6] = 0.0
        # water density
        vec_values_as_np[7] = pc.dens_m[0] if 0 in pc.ph else 0.0
        # steam density
        vec_values_as_np[8] = pc.dens_m[1] if 1 in pc.ph else 0.0
        # temperature
        vec_values_as_np[9] = pc.temperature

        return 0
