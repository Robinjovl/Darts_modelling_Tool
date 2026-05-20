import os
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from darts.physics.properties.black_oil import *
from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer


class BlackOilConfig(BaseModel):
    """Configuration for BlackOil physics (PVT-driven)."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "kind": "black_oil",
                    "pvt_path": "pvt_data.in",
                    "thermal": False,
                    "n_points": 5001,
                    "min_p": 1.0,
                    "max_p": 450.0,
                }
            ]
        },
    )

    kind: Literal["black_oil"] = Field(
        default="black_oil",
        description="Physics discriminator for ModelConfig.physics union",
    )
    pvt_path: Annotated[
        str, Field(description="Path to PVT data file (may be relative)")
    ]
    thermal: Annotated[
        bool, Field(description="Whether to run in thermal (PT) mode")
    ] = False
    type_hydr: Annotated[
        Literal["isothermal", "thermal"],
        Field(description="Hydraulic problem type"),
    ] = "isothermal"
    type_mech: Annotated[
        Literal["none", "poroelasticity", "thermoporoelasticity"],
        Field(description="Mechanical coupling type"),
    ] = "none"
    init_type: Annotated[
        Literal["uniform"], Field(description="Initialization type")
    ] = "uniform"
    n_points: Annotated[int, Field(ge=2, description="OBL table resolution")] = 5001
    zero: Annotated[float, Field(ge=0, description="OBL zero offset")] = 1e-12
    epsilon_z: Annotated[float, Field(ge=0, description="OBL composition epsilon")] = (
        1e-13
    )
    min_p: Annotated[float, Field(ge=0, description="Minimum OBL pressure [bar]")] = 1.0
    max_p: Annotated[float, Field(ge=0, description="Maximum OBL pressure [bar]")] = (
        450.0
    )
    min_t: Annotated[float, Field(description="Minimum OBL temperature [°C]")] = -10.0
    max_t: Annotated[float, Field(description="Maximum OBL temperature [°C]")] = 100.0
    min_z: Annotated[float, Field(ge=0, description="Minimum OBL composition")] = 0.0
    max_z: Annotated[float, Field(ge=0, description="Maximum OBL composition")] = 1.0
    components: Annotated[
        list[str] | None,
        Field(
            description="Component names; metadata-only on BlackOilConfig "
            "(BlackOil.__init__ derives the canonical components from the PVT "
            "file). Lets a JSON preset ship the names so PhysicsSpec.components "
            "doesn't need to be repeated by the caller.",
        ),
    ] = None
    phases: Annotated[
        list[str] | None,
        Field(description="Phase names; same semantics as ``components``."),
    ] = None


class BlackOil(Compositional):
    """Black-oil physics constructed directly from a :class:`BlackOilConfig`.

    Reads the PVT file referenced by ``config.pvt_path`` to build the
    density/viscosity/rel-perm/capillary-pressure evaluator bundle, then
    forwards OBL parameters to :class:`Compositional`.
    """

    def __init__(
        self,
        config: "BlackOilConfig",
        timer: Any,
        *,
        components: list[str] | None = None,
        phases: list[str] | None = None,
    ):
        """
        :param config: validated physics configuration
        :type config: BlackOilConfig
        :param timer: DARTS timer node passed to Compositional
        :type timer: darts.engines.timer_node
        :param components: optional override of component names (must match PVT)
        :type components: list[str] | None
        :param phases: optional override of phase names (must match PVT)
        :type phases: list[str] | None
        """
        pvt_path = os.path.expanduser(config.pvt_path)
        evaluators = _load_black_oil_evaluators(pvt_path)

        if components and components != evaluators["components"]:
            raise ValueError(
                f"BlackOil components mismatch: {components} vs "
                f"{evaluators['components']}"
            )
        if phases and phases != evaluators["phases"]:
            raise ValueError(
                f"BlackOil phases mismatch: {phases} vs {evaluators['phases']}"
            )

        thermal = config.thermal
        state_spec = (
            Compositional.StateSpecification.PT
            if thermal
            else Compositional.StateSpecification.P
        )
        super().__init__(
            components=evaluators["components"],
            phases=evaluators["phases"],
            timer=timer,
            n_points=config.n_points,
            min_p=config.min_p,
            max_p=config.max_p,
            min_z=config.min_z,
            max_z=config.max_z,
            epsilon_z=config.epsilon_z,
            min_t=config.min_t,
            max_t=config.max_t,
            state_spec=state_spec,
            extrapolation_flag=True,
        )

        temperature = None if thermal else 1.0
        property_container = BlackOilProperties(
            phases_name=evaluators["phases"],
            components_name=evaluators["components"],
            Mw=evaluators["Mw"],
            eps_z=config.epsilon_z,
            temperature=temperature,
        )

        property_container.flash_ev = evaluators["flash_ev"]
        property_container.density_ev = evaluators["density"]
        property_container.viscosity_ev = evaluators["viscosity"]
        property_container.rel_perm_ev = evaluators["rel_perm"]
        property_container.capillary_pressure_ev = evaluators["capillary_pressure"]
        property_container.rock_compress_ev = RockCompactionEvaluator(pvt_path)

        self.add_property_region(property_container)

    @classmethod
    def from_config(
        cls,
        config: "BlackOilConfig",
        *,
        components: list[str] | None = None,
        phases: list[str] | None = None,
        timer: Any,
    ) -> "BlackOil":
        """Build a BlackOil directly from its Config — thin alias for
        ``BlackOil(config, timer, components=..., phases=...)``.
        """
        return cls(config, timer, components=components, phases=phases)


def _load_black_oil_evaluators(pvt_path: str) -> dict:
    """Load PVT and build the dict of evaluators consumed by BlackOil.

    Replaces the BlackOilFluidProps facade — the PVT loading side-effect
    now lives directly alongside BlackOil construction instead of being
    attached to a free-form ``idata.fluid`` attribute.

    :param pvt_path: filesystem path to a DARTS PVT input file
    :type pvt_path: str
    :return: dict with keys ``components``, ``phases``, ``Mw``, ``flash_ev``,
        ``density``, ``viscosity``, ``rel_perm``, ``capillary_pressure``
    :rtype: dict
    """
    components = ["g", "o", "w"]
    phases = ["gas", "oil", "water"]
    return {
        "components": components,
        "phases": phases,
        "Mw": np.ones(len(components)),
        "flash_ev": flash_black_oil(pvt_path),
        "density": {
            "gas": DensityGas(pvt_path),
            "oil": DensityOil(pvt_path),
            "water": DensityWat(pvt_path),
        },
        "viscosity": {
            "gas": ViscGas(pvt_path),
            "oil": ViscOil(pvt_path),
            "water": ViscWat(pvt_path),
        },
        "rel_perm": {
            "gas": GasRelPerm(pvt_path),
            "oil": OilRelPerm(pvt_path),
            "water": WatRelPerm(pvt_path),
        },
        "capillary_pressure": {
            "pcow": CapillaryPressurePcow(pvt_path),
            "pcgo": CapillaryPressurePcgo(pvt_path),
        },
    }


class BlackOilProperties(PropertyContainer):
    def __init__(
        self,
        phases_name,
        components_name,
        Mw,
        eps_z: float = 1e-11,
        temperature: float = None,
    ):
        # Call base class constructor
        super().__init__(phases_name, components_name, Mw, eps_z=eps_z, temperature=1.0)
        # self.surf_dens = get_table_keyword(idata.fluid.pvt, 'DENSITY')[0]
        # self.surf_oil_dens = self.surf_dens[0]
        # self.surf_wat_dens = self.surf_dens[1]
        # self.surf_gas_dens = self.surf_dens[2]

    def evaluate(self, state):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        # Composition vector and pressure from state:
        vec_state_as_np = np.asarray(state)
        pressure = vec_state_as_np[0]
        self.temperature = vec_state_as_np[-1] if self.thermal else self.temperature

        zc = np.append(vec_state_as_np[1:], 1 - np.sum(vec_state_as_np[1:]))

        if zc[-1] < 0:
            # print(zc)
            zc = self.comp_out_of_bounds(zc)

        self.clean_arrays()
        # two-phase flash - assume water phase is always present and water component last
        xgo, V, pbub = self.flash_ev.evaluate(pressure, zc)
        for i in range(self.nph):
            self.x[i, i] = 1

        if V < 0:
            self.ph = np.array([1, 2])
        else:  # assume oil and water are always exists
            self.x[1][0] = xgo
            self.x[1][1] = 1 - xgo
            self.ph = np.array([0, 1, 2])

        for j in self.ph:
            M = 0
            # molar weight of mixture
            for i in range(self.nc):
                M += self.Mw[i] * self.x[j][i]
            self.dens[j] = self.density_ev[self.phases_name[j]].evaluate(
                pressure, pbub, xgo
            )  # output in [kg/m3]
            self.dens_m[j] = self.dens[j] / M
            self.mu[j] = self.viscosity_ev[self.phases_name[j]].evaluate(
                pressure, pbub
            )  # output in [cp]

        self.nu[2] = zc[2]
        # two phase undersaturated condition
        if pressure > pbub:
            self.nu[0] = 0
            self.nu[1] = zc[1]
        else:
            self.nu[1] = zc[1] / (1 - xgo)
            self.nu[0] = 1 - self.nu[1] - self.nu[2]

        self.compute_saturation(self.ph)

        for j in self.ph:
            self.kr[j] = self.rel_perm_ev[self.phases_name[j]].evaluate(
                self.sat[0], self.sat[2]
            )

        pcow = self.capillary_pressure_ev['pcow'].evaluate(self.sat[2])
        pcgo = self.capillary_pressure_ev['pcgo'].evaluate(self.sat[0])

        self.pc = np.array([-pcgo, 0, pcow])

        return

    def evaluate_at_cond(self, pressure, zc):
        self.sat[:] = 0

        if zc[-1] < 0:
            # print(zc)
            zc = self.comp_out_of_bounds(zc)

        self.ph = []
        for j in range(self.nph):
            if zc[j] > self.eps_z:
                self.ph.append(j)
            self.dens_m[j] = self.density_ev[self.phases_name[j]].dens_sc

        self.nu = zc
        self.compute_saturation(self.ph)

        return self.sat, self.dens_m
