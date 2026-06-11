from dataclasses import dataclass

import numpy as np
from model_rate import Model as RateModel
from model_rate import Stage1ThermosiphonRateConfig

from darts.physics.properties.basic import ConstFunc, PhaseRelPerm
from darts.physics.properties.density import DensityBasic
from darts.physics.properties.enthalpy import EnthalpyBasic
from darts.physics.properties.flash import ConstantK
from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer
from darts.pipes.interfacial_tension import IFT_multicomponent_MCM


@dataclass
class Stage1ThermosiphonSimplePhysicsConfig(Stage1ThermosiphonRateConfig):
    trace_fraction: float = 1e-8


class Model(RateModel):
    def __init__(self, config: Stage1ThermosiphonSimplePhysicsConfig | None = None):
        super().__init__(config or Stage1ThermosiphonSimplePhysicsConfig())

    def _trace_fraction(self):
        return float(getattr(self.config, "trace_fraction", self.zero))

    def set_physics(self):
        epsilon = 1e-9
        molecular_weights = [44.01, 18.015]

        pc = PropertyContainer(
            phases_name=self.phases,
            components_name=self.components,
            Mw=molecular_weights,
            eps_z=epsilon,
            temperature=None,
        )
        pc.flash_ev = ConstantK(
            len(self.components),
            [4.0, 0.1],
            self.zero,
        )
        pc.density_ev = {
            "G": DensityBasic(compr=1e-3, dens0=200.0),
            "L": DensityBasic(compr=1e-5, dens0=600.0),
        }
        pc.viscosity_ev = {
            "G": ConstFunc(0.05),
            "L": ConstFunc(0.5),
        }
        pc.rel_perm_ev = {
            "G": PhaseRelPerm("gas"),
            "L": PhaseRelPerm("oil"),
        }
        pc.enthalpy_ev = {
            "G": EnthalpyBasic(hcap=37.0),
            "L": EnthalpyBasic(hcap=75.3),
        }
        pc.conductivity_ev = {
            "G": ConstFunc(3.5),
            "L": ConstFunc(75.0),
        }
        pc.rock_energy_ev = EnthalpyBasic(hcap=1.0)
        pc.IFT_ev = IFT_multicomponent_MCM(self.components)

        pc.output_props = {"temperature": lambda: pc.temperature}
        for j, ph in enumerate(self.phases):
            pc.output_props["s" + ph] = lambda jj=j: pc.sat[jj]
            pc.output_props["rho" + ph] = lambda jj=j: pc.dens[jj]
            pc.output_props["miu" + ph] = lambda jj=j: pc.mu[jj]
            for i, comp in enumerate(self.components):
                pc.output_props[f"x{comp}_in_{ph}_mass"] = lambda jj=j, ii=i: pc.x_mass[
                    jj, ii
                ]

        self.physics = Compositional(
            self.components,
            self.phases,
            self.timer,
            state_spec=Compositional.StateSpecification.PT,
            n_points=200,
            min_p=1.0,
            max_p=400.0,
            min_z=0.0,
            max_z=1.0,
            epsilon_z=epsilon,
            min_t=273.15,
            max_t=473.15,
            extrapolation_flag=True,
        )
        self.physics.add_property_region(pc)

    def set_wells(self):
        cfg = self.config
        trace = self._trace_fraction()
        gas_rich_composition = [[1.0 - trace, trace]]

        self.wells = {}
        self._add_dfm_well(
            well_name="I1",
            lgr_name="inj_lgr",
            lgr_cell_ij=self._well_lgr_cell_ij(),
            pipe_head_temperature=cfg.injector_head_temperature,
            pipe_head_pressure=cfg.initial_bhp,
            temp_grad=cfg.injector_temp_grad,
            initial_phase_name=["G"],
            initial_phase_composition=gas_rich_composition,
        )
        self._add_dfm_well(
            well_name="P1",
            lgr_name="prod_lgr",
            lgr_cell_ij=self._well_lgr_cell_ij(),
            pipe_head_temperature=cfg.producer_head_temperature,
            pipe_head_pressure=cfg.initial_whp,
            temp_grad=cfg.producer_temp_grad,
            initial_phase_name=["G"],
            initial_phase_composition=gas_rich_composition,
        )

    def _injection_source_composition(self):
        trace = self._trace_fraction()
        return np.array([1.0 - trace, trace], dtype=float)
