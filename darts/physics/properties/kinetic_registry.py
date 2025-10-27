"""
Kinetic reactions registry for supported minerals and their kinetic mechanisms.

Entries:
- k: Arrhenius pre-exponential factor [mol/m2/s]
- Ea: Activation energy [J/mol]
- T_ref_K: Reference temperature [K]
- n: reaction order w.r.t. mechanism activity
- p, q: chemical affinity parameters in (1 - SR^p)^q term
"""

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

KINETIC_REGISTRY = {
    'CaCO3': {
        'acidic': {
            'k': 10 ** (-0.3),
            'Ea': 14400,
            'n': 1,
            'p': 1,
            'q': 1,
            'T_ref_K': 273.15 + 25,
        },
        'neutral': {
            'k': 10 ** (-5.81),
            'Ea': 23500,
            'n': 0,
            'p': 1,
            'q': 1,
            'T_ref_K': 273.15 + 25,
        },
        'carbonate': {
            'k': 10 ** (-3.48),
            'Ea': 35400,
            'n': 1,
            'p': 1,
            'q': 1,
            'T_ref_K': 273.15 + 25,
        },
    },
    'CaMg(CO3)2': {
        'acidic': {
            'k': 10 ** (-3.19),
            'Ea': 36100,
            'n': 0.5,
            'p': 1,
            'q': 1,
            'T_ref_K': 273.15 + 25,
        },
        'neutral': {
            'k': 10 ** (-7.53),
            'Ea': 52200,
            'n': 0,
            'p': 1,
            'q': 1,
            'T_ref_K': 273.15 + 25,
        },
        'carbonate': {
            'k': 10 ** (-5.11),
            'Ea': 34800,
            'n': 0.5,
            'p': 1,
            'q': 1,
            'T_ref_K': 273.15 + 25,
        },
    },
    'MgCO3': {
        'acidic': {
            'k': 10 ** (-6.38),
            'Ea': 14400,
            'n': 1,
            'p': 1,
            'q': 1,
            'T_ref_K': 273.15 + 25,
        },
        'neutral': {
            'k': 10 ** (-9.34),
            'Ea': 23500,
            'n': 0,
            'p': 1,
            'q': 1,
            'T_ref_K': 273.15 + 25,
        },
        'carbonate': {
            'k': 10 ** (-5.22),
            'Ea': 62800,
            'n': 1,
            'p': 1,
            'q': 1,
            'T_ref_K': 273.15 + 25,
        },
    },
}


class MineralSpec(BaseModel):
    """Pydantic schema for a single-mineral kinetic configuration.

    Example:
        mineral_name='CaCO3', mechanisms=['acidic', 'neutral', 'carbonate']
    """

    mineral_name: str
    mechanisms: list[str]

    model_config = ConfigDict(extra='forbid')

    @field_validator('mineral_name')
    @classmethod
    def validate_mineral_name(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("'mineral_name' must be a non-empty string")
        mineral_key = value.strip()
        if mineral_key not in KINETIC_REGISTRY:
            raise ValueError(
                f"Unsupported mineral '{value}'. Supported: {sorted(KINETIC_REGISTRY.keys())}"
            )
        return mineral_key

    @field_validator('mechanisms')
    @classmethod
    def validate_mechanisms(cls, value: list[str], info) -> list[str]:
        if not isinstance(value, list | tuple) or len(value) == 0:
            raise ValueError("'mechanisms' must be a non-empty list of strings")
        # Access already-validated mineral_name from model fields
        # Pydantic v2 passes validated fields via 'data' on the model instance after init,
        # but here we re-validate against the full registry after normalization below.
        clean: list[str] = []
        for mech in value:
            if not isinstance(mech, str):
                raise ValueError("Mechanism names must be strings")
            mk = mech.strip().lower()
            if mk not in {'acidic', 'neutral', 'carbonate'}:
                raise ValueError(
                    "Unsupported mechanism '{mk}'. Supported: ['acidic','neutral','carbonate']"
                )
            if mk not in clean:
                clean.append(mk)
        return clean


class ReactionSurfaceAreaSpec(BaseModel):
    """Spec for reaction surface area evaluator used by KineticRate.

    Currently supports a linear evaluator only.
    """

    model_config = ConfigDict(extra='forbid')

    kind: str = Field(default='linear', pattern='^(linear)$')
    initial_area_per_mol: float = Field(gt=0)


class KineticRateSpec(MineralSpec):
    """Pydantic spec for constructing a KineticRate instance.

    Inherits mineral/mechanism validation; adds `min_z` and surface area spec.
    """

    min_z: float = Field(gt=0)
    surface_area: ReactionSurfaceAreaSpec


class KineticRate:
    """Evaluate mineral kinetic rates.

    Input is validated against the kinetics registry via Pydantic.
    """

    def __init__(self, min_z, mineral_name, mechanisms, surface_area_ev):
        """Create a kinetic rate evaluator for a single mineral.

        Parameters
        - min_z: Minimum composition value (kept for API symmetry)
        - mineral_name: Mineral identifier, e.g. 'CaCO3'
        - mechanisms: List of mechanism names, e.g. ['acidic','neutral']
        - surface_area_ev: Surface area evaluator, e.g. LinearReactionSurfaceArea
        """
        self.min_z = min_z
        self.surface_area_ev = surface_area_ev
        try:
            spec = MineralSpec(mineral_name=mineral_name, mechanisms=mechanisms)
        except ValidationError as exc:
            raise exc
        self.mineral = spec.mineral_name
        mech_defs = KINETIC_REGISTRY[self.mineral]
        self.mechanisms = [
            ReactionMechanism(
                name=mech,
                temperature_ref=mech_defs[mech]['T_ref_K'],
                k=mech_defs[mech]['k'],
                Ea=mech_defs[mech]['Ea'],
                n=mech_defs[mech]['n'],
                p=mech_defs[mech]['p'],
                q=mech_defs[mech]['q'],
            )
            for mech in mechanisms
        ]

    def evaluate(self, kin_state, solid_saturation, rho_s, temperature):
        """Compute kinetic rate [kmol/d/m3] for the configured mineral.

        Parameters
        - kin_state: Dict-like with activities/saturation ratios from PHREEQC; expects keys
          'Act(H+)', 'Act(CO2)', and 'SR_<mineral>'
        - solid_saturation: Solid saturation (volume fraction) of the mineral
        - rho_s: Solid molar density [kmol/m3]
        - temperature: Temperature [K]
        """
        if not self.mechanisms:
            return 0.0

        # TODO: avoid dependence on PHREEQC format of kin_state
        # gather activities by mechanism
        activity_by_mech = {
            'acidic': kin_state['Act(H+)'],
            'neutral': 1.0,
            'carbonate': kin_state['Act(CO2)'],
        }

        # calculate rates by mechanism
        sr_key = 'SR_' + self.mineral
        rates = [
            mech.evaluate(
                temperature=temperature,
                activity=activity_by_mech[mech.name],
                SR=kin_state[sr_key],
            )
            for mech in self.mechanisms
        ]

        # calculate surface area [m2/mol]
        surface_area = self.surface_area_ev.evaluate(solid_saturation)

        # calculate kinetic rate [mol/s/m3]
        kinetic_rate = -surface_area * (rho_s * 1000) * sum(rates)

        # convert to [kmol/d/m3]
        kinetic_rate = kinetic_rate * 60 * 60 * 24 / 1000
        return kinetic_rate

    @classmethod
    def from_spec(cls, spec: KineticRateSpec) -> 'KineticRate':
        """Construct a KineticRate instance from a validated KineticRateSpec."""
        # Build surface area evaluator (extensible by kind)
        if spec.surface_area.kind == 'linear':
            sa_ev = LinearReactionSurfaceArea(spec.surface_area.initial_area_per_mol)
        else:
            raise ValueError(f"Unsupported surface area kind: {spec.surface_area.kind}")

        return cls(
            min_z=spec.min_z,
            mineral_name=spec.mineral_name,
            mechanisms=spec.mechanisms,
            surface_area_ev=sa_ev,
        )


class LinearReactionSurfaceArea:
    def __init__(self, initial_area_per_mol: float):
        """
        Initialize the reaction surface area evaluator.
        :param initial_area_per_mol: initial area per mol [m2/mol]
        :type initial_area_per_mol: float
        """
        self.s_init = initial_area_per_mol

    def evaluate(self, vol_fraction):
        return self.s_init * vol_fraction


class ReactionMechanism:
    """
    Class representing an Arrhenius-type reaction mechanism
    with chemical affinity term (1-SR**p)**q.
    """

    def __init__(self, name, temperature_ref, k, Ea, n, p=1, q=1):
        # maximum saturation ratio threshold for chemical affinity term
        self.SR_threshold = 100
        # universal gas constant [J/mol/K]
        self.R = 8.314472
        # name of the mechanism
        self.name = name
        # reference temperature when rate parameters are given [K]
        self.temperature_ref = temperature_ref
        # pre-exponential factor [mol/m2/s]
        self.k = k
        # activation energy [J/mol]
        self.Ea = Ea
        # reaction order with respect to given activity/anything
        self.n = n
        # chemical affinity parameter in (1-SR**p)**q term
        self.p = p
        # chemical affinity parameter in (1-SR**p)**q term
        self.q = q

    def evaluate(self, temperature, activity, SR):
        """
        Evaluate the reaction rate for a given temperature, activity, and saturation ratio.
        :param temperature: temperature [K]
        :param activity: activity of the reactant relevant to the mechanism
        :param SR: saturation ratio
        :return: reaction rate [mol/s/m2]
        """
        # calculate the Arrhenius factor
        k_arr = self.k * np.exp(
            (-self.Ea / self.R) * (1 / temperature - 1 / self.temperature_ref)
        )
        # impose maximum saturation ratio threshold on chemical affinity term
        SR_bound = min(SR, self.SR_threshold)
        # calculate the chemical affinity factor
        k_aff = (1 - SR_bound**self.p) ** self.q
        # calculate the reaction rate
        rate = k_arr * k_aff * activity**self.n
        return rate
