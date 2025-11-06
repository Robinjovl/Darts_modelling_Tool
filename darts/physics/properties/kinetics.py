import abc
import json
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class Kinetics:
    def __init__(self, stoich: list):
        self.stoich = stoich

    @abc.abstractmethod
    def evaluate(self, pressure, temperature, x, sat):
        pass

    def evaluate_enthalpy(self, pressure, temperature, x, sat):
        return 0.0


class KineticBasic:
    def __init__(self, equi_prod, kin_rate_cte, ne, combined_ions=True):
        self.equi_prod = equi_prod
        self.kin_rate_cte = kin_rate_cte
        self.kinetic_rate = np.zeros(ne)
        self.combined_ions = combined_ions

    def evaluate(self, pressure, temperature, x, nu_sol):
        if self.combined_ions:
            ion_prod = (x[1][1] / 2) ** 2
            dQ = 1 - ion_prod / self.equi_prod
            self.kinetic_rate[1] = -self.kin_rate_cte * dQ * nu_sol
            self.kinetic_rate[-1] = -0.5 * self.kinetic_rate[1]
        else:
            ion_prod = x[1][1] * x[1][2]
            dQ = 1 - ion_prod / self.equi_prod
            self.kinetic_rate[1] = -self.kin_rate_cte * dQ * nu_sol
            self.kinetic_rate[2] = -self.kin_rate_cte * dQ * nu_sol
            self.kinetic_rate[-1] = -self.kinetic_rate[1]

        return self.kinetic_rate, dQ


class LawOfMassAction(Kinetics):
    """
    Law of Mass Action for kinetic reaction
    For reaction aA + bB <-> cC: rate = c * (1 - Q/K) with Q = [C]^c / [A]^a [B]^b
    """

    def __init__(
        self,
        stoich: list,
        nc_fl: int,
        fl_idx: int,
        equi_prod: float,
        kin_rate_cte: float,
    ):
        super().__init__(stoich)

        self.nc_fl = nc_fl
        self.fl_idx = fl_idx
        self.equi_prod = equi_prod
        self.kin_rate_cte = kin_rate_cte

    def evaluate(self, pressure, temperature, x, sat_sol):
        # For reaction aA + bB <-> cC
        # Calculate activity product Q = [C]^c / [A]^a [B]^b
        prod = 1.0
        for i in range(self.nc_fl):
            prod = (
                prod * x[self.fl_idx, i] ** self.stoich[i]
                if self.stoich[i] != 0
                else prod
            )

        # Calculate rate = c * As * (1-Q/K)
        dQ = 1.0 - prod / self.equi_prod
        rate = self.kin_rate_cte * sat_sol * dQ

        return [stoich * rate for stoich in self.stoich], dQ


class HydrateKinetics(Kinetics):
    def __init__(
        self,
        components: list,
        phases: list,
        Mw,
        hydrate_eos,
        fluid_eos: list,
        stoich: list = None,
        perm: float = 300.0,
        poro: float = 0.2,
        k: float = None,
        F_a=1.0,
        moridis: bool = True,
        enthalpy: bool = False,
    ):
        super().__init__(stoich)

        self.hydrate_eos = hydrate_eos
        self.fluid_eos = fluid_eos

        self.water_idx = components.index("H2O")
        self.guest_idx = 0 if self.water_idx == 1 else 1
        # self.hydrate_idx = components.index("H")

        self.a_idx = phases.index("Aq")
        self.v_idx = phases.index("V")
        self.h_idx = phases.index("sI")

        self.stoich = stoich
        if stoich is not None:
            self.nH = stoich[self.water_idx] / stoich[self.guest_idx]
        else:
            self.nH = None
        self.Mw = Mw

        self.K = (
            k if k is not None else 3.6e6 * 86400
        )  # reaction constant [kmol/(m^2 bar day)]
        self.F_a = F_a
        self.perm = perm * 1e-15  # mD to m2
        self.poro = poro

        if moridis:
            r_p = np.sqrt(45.0 * self.perm * (1 - self.poro) ** 2 / self.poro**3)
            self.A_s = (
                lambda sat: 0.879
                * self.F_a
                * (1 - self.poro)
                / r_p
                * sat[self.h_idx] ** (2.0 / 3.0)
            )
        else:
            self.K = 8.06 / Mw[-1] * 1e5 * 86400  # kg/m2.Pa.s
            r_p = 3.75e-4
            beta = 2.0 / 3.0
            self.A_s = lambda sat: (
                0.879
                * (1 - self.poro)
                / r_p
                * sat[self.v_idx] ** (2.0 / 3.0)
                * sat[self.a_idx] ** beta
                * (1 - sat[self.h_idx]) ** beta
            )

        self.enthalpy = enthalpy

    def calc_df(self, pressure, temperature, x):
        # Calculate fugacity difference between water in fluid phases and water in hydrate phase
        if x[0, 0] != 0.0:
            f0 = self.fluid_eos[0].fugacity(pressure, temperature, x[0, :])
        else:
            f0 = self.fluid_eos[1].fugacity(pressure, temperature, x[1, :])

        fwH = self.hydrate_eos.fw(pressure, temperature, f0)

        df = fwH - f0[self.water_idx]  # if df < 0 formation, if df > 0 dissociation
        xH = self.hydrate_eos.xH()

        return df, xH

    def evaluate(self, pressure, temperature, x, sat: list):
        df, xH = self.calc_df(pressure, temperature, x)

        # Reaction rate following Yin (2018)
        # surface area
        A_s = self.A_s(sat)

        # Thermodynamic parameters
        dE = -81e3  # activation energy [J/mol]
        R = 8.3145  # gas constant [J/(K.mol)]

        # K is reaction cons, A_s hydrate surface area, dE activation energy, driving force is fugacity difference
        self.rate = self.K * A_s * np.exp(dE / (R * temperature)) * df

        return [stoich * self.rate for stoich in self.stoich], df

    def evaluate_enthalpy(self, pressure, temperature, x, sat):
        if self.enthalpy:
            # Enthalpy change with dissociation (-ive, rate of hydrate component +ive)
            Cf = 33.72995  # J/kg cal/gmol
            if temperature - 273.15 > 0:
                (C1, C2) = (13521, -4.02)
            else:
                (C1, C2) = (6534, -11.97)
            en = Cf * (C1 + C2 / temperature) * 1e-3  # kJ/kg

            # Calculate molar weight of hydrate component
            if self.nH is None:
                mH = np.sum(x * self.Mw)
            else:
                mH = self.Mw[self.water_idx] * self.nH + self.Mw[self.guest_idx]

            H_diss = -en * mH  # kJ/kg * kg/kmol -> kJ/kmol

            return self.rate * H_diss  # rate [kmol/day] * [kJ/kmol] = [kJ/day]
        else:
            return 0.0


# ---------------------------
# Mineral kinetics working with databases
# ---------------------------


# Default databases directory; resolved relative to this module
_DEFAULT_DB_DIR = Path(__file__).parent / 'databases'


def _resolve_kinetic_db_path(kinetic_database: str | Path) -> Path:
    """Resolve path to kinetics JSON database.

    If `kinetic_database` is an existing path, use it. Otherwise, treat it as
    a name within the default databases directory and append .json.
    """
    candidate = Path(kinetic_database)
    if candidate.exists():
        return candidate
    named = _DEFAULT_DB_DIR / f"{str(kinetic_database)}.json"
    if named.exists():
        return named
    raise FileNotFoundError(
        f"Kinetics database not found: '{kinetic_database}'. "
        f"Checked '{candidate}' and '{named}'."
    )


def _load_kinetic_registry(db_path: Path) -> dict:
    with db_path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    # Normalize: compute pre-exponential factor k from logk when needed
    for mineral_name, mechanisms in data.items():
        if not isinstance(mechanisms, dict):
            raise ValueError(f"Invalid mechanisms block for {mineral_name}")
        for mechanism_name, params in mechanisms.items():
            if isinstance(params, dict) and 'k' not in params and 'logk' in params:
                try:
                    params['k'] = float(10 ** float(params['logk']))
                except Exception as exc:
                    raise ValueError(
                        f"Invalid logk for {mineral_name}/{mechanism_name}"
                    ) from exc
    return data


# Global reference for validators; set by KineticRate at construction time
KINETIC_REGISTRY: dict | None = None


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
        if (
            not isinstance(KINETIC_REGISTRY, dict)
            or mineral_key not in KINETIC_REGISTRY
        ):
            supported = (
                sorted(KINETIC_REGISTRY.keys())
                if isinstance(KINETIC_REGISTRY, dict)
                else []
            )
            raise ValueError(f"Unsupported mineral '{value}'. Supported: {supported}")
        return mineral_key

    @field_validator('mechanisms')
    @classmethod
    def validate_mechanisms(cls, value: list[str], info) -> list[str]:
        if not isinstance(value, list | tuple) or len(value) == 0:
            raise ValueError("'mechanisms' must be a non-empty list of strings")
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
    kinetic_database: str | Path = 'PalandriKharaka'


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


class KineticRate:
    """Evaluate mineral kinetic rates loaded from a kinetics database.

    Database may be a path or a named JSON in the default databases folder.
    """

    def __init__(
        self,
        min_z,
        mineral_name,
        mechanisms,
        surface_area_ev,
        kinetic_database: str | Path = 'PalandriKharaka',
    ):
        """Create a kinetic rate evaluator for a single mineral.

        Parameters
        - min_z: Minimum composition value (kept for API symmetry)
        - mineral_name: Mineral identifier, e.g. 'CaCO3'
        - mechanisms: List of mechanism names, e.g. ['acidic','neutral']
        - surface_area_ev: Surface area evaluator, e.g. LinearReactionSurfaceArea
        - kinetic_database: Path or name of kinetics DB (without .json)
        """
        self.min_z = min_z
        self.surface_area_ev = surface_area_ev

        # Resolve and load database
        db_path = _resolve_kinetic_db_path(kinetic_database)
        global KINETIC_REGISTRY
        KINETIC_REGISTRY = _load_kinetic_registry(db_path)

        # Validate and normalize inputs against the loaded registry
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
            kinetic_database=spec.kinetic_database,
        )
