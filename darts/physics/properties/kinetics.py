import abc
import json
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field

from darts.physics.properties.evaluator_base import (
    EvaluatorBase,
    EvaluatorConfigBase,
    register_evaluator,
)


class KineticBasicConfig(EvaluatorConfigBase):
    """Configuration for KineticBasic evaluator."""

    kind: Literal["kinetic_basic"] = "kinetic_basic"
    equi_prod: float
    rate: float = Field(1.0, ge=0)
    ne: int = Field(ge=1)


class LawOfMassActionConfig(EvaluatorConfigBase):
    """Configuration for LawOfMassAction kinetic evaluator."""

    kind: Literal["law_of_mass_action"] = "law_of_mass_action"
    stoich: list[float]
    nc_fl: int = Field(ge=1)
    fl_idx: int = Field(ge=0)
    equi_prod: float
    kin_rate_cte: float


class HydrateKineticsConfig(EvaluatorConfigBase):
    """Configuration for HydrateKinetics evaluator (EOS objects passed via context)."""

    kind: Literal["hydrate_kinetics"] = "hydrate_kinetics"
    components: list[str]
    phases: list[str]
    Mw: list[float]
    stoich: list[float] | None = None
    perm: float = 300.0
    poro: float = 0.2
    k: float | None = None
    F_a: float = 1.0
    moridis: bool = True
    enthalpy: bool = False


class LinearReactionSurfaceAreaConfig(EvaluatorConfigBase):
    """Configuration for LinearReactionSurfaceArea evaluator."""

    kind: Literal["linear_reaction_surface_area"] = "linear_reaction_surface_area"
    initial_area_per_mol: float = Field(ge=0)


class KineticRateConfig(EvaluatorConfigBase):
    """Configuration for KineticRate mineral kinetic rate evaluator."""

    kind: Literal["kinetic_rate"] = "kinetic_rate"
    min_z: float
    mineral_name: str
    mechanisms: list[str]
    surface_area_ev: LinearReactionSurfaceAreaConfig
    kinetic_database: str = "PalandriKharaka"


class Kinetics(EvaluatorBase):
    """Family ABC for kinetic-rate evaluators.

    Concrete subclasses implement :meth:`evaluate` returning a per-component
    rate vector and the driving force (``dQ`` or fugacity difference).
    """

    def __init__(self, stoich: list):
        """
        :param stoich: stoichiometric coefficients per component
        :type stoich: list[float] | None
        """
        self.stoich = stoich

    @abc.abstractmethod
    def evaluate(self, pressure, temperature, x, sat):
        """
        :param pressure: pressure [bar]
        :type pressure: float
        :param temperature: temperature [K]
        :type temperature: float
        :param x: per-phase composition matrix (molar fractions)
        :type x: np.ndarray
        :param sat: per-phase saturations
        :type sat: list[float] | np.ndarray
        :return: (rate vector per component, driving force)
        :rtype: tuple[list[float], float]
        """
        pass

    def evaluate_enthalpy(self, pressure, temperature, x, sat):
        """Optional enthalpy-of-reaction hook; default is no enthalpy effect.

        :param pressure: pressure [bar]
        :type pressure: float
        :param temperature: temperature [K]
        :type temperature: float
        :param x: per-phase composition matrix
        :type x: np.ndarray
        :param sat: per-phase saturations
        :type sat: list[float] | np.ndarray
        :return: enthalpy contribution
        :rtype: float
        """
        return 0.0


class KineticBasic(EvaluatorBase):
    """Simple precipitation/dissolution kinetic for ions."""

    def __init__(self, equi_prod, kin_rate_cte, ne, combined_ions=True):
        """
        :param equi_prod: equilibrium ion product
        :type equi_prod: float
        :param kin_rate_cte: kinetic rate constant
        :type kin_rate_cte: float
        :param ne: number of components in the rate vector
        :type ne: int
        :param combined_ions: whether ion species are combined into one column
        :type combined_ions: bool
        """
        self.equi_prod = equi_prod
        self.kin_rate_cte = kin_rate_cte
        self.kinetic_rate = np.zeros(ne)
        self.combined_ions = combined_ions

    def to_config(self) -> KineticBasicConfig:
        """Build Config explicitly because Config field ``rate`` maps to
        attribute ``kin_rate_cte`` and ``ne`` is derived from
        ``len(self.kinetic_rate)``.

        :return: serialized config
        :rtype: KineticBasicConfig
        """
        return KineticBasicConfig(
            equi_prod=self.equi_prod,
            rate=self.kin_rate_cte,
            ne=len(self.kinetic_rate),
        )

    @classmethod
    def from_config(cls, config: KineticBasicConfig) -> "KineticBasic":
        """Build instance explicitly because the constructor uses
        ``kin_rate_cte`` rather than the Config's ``rate``.

        :param config: validated config
        :type config: KineticBasicConfig
        :return: KineticBasic instance
        :rtype: KineticBasic
        """
        return cls(config.equi_prod, config.rate, config.ne)

    def evaluate(self, pressure, temperature, x, nu_sol):
        """
        :param pressure: pressure [bar] (unused)
        :type pressure: float
        :param temperature: temperature [K] (unused)
        :type temperature: float
        :param x: per-phase composition matrix
        :type x: np.ndarray
        :param nu_sol: solid phase fraction
        :type nu_sol: float
        :return: (kinetic rate vector, driving force ``dQ``)
        :rtype: tuple[np.ndarray, float]
        """
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


register_evaluator("kinetic_basic", KineticBasic, KineticBasicConfig)


class LawOfMassAction(Kinetics):
    """
    Law of Mass Action for kinetic reaction.
    For reaction aA + bB <-> cC: rate = c * (1 - Q/K) with Q = [C]^c / [A]^a [B]^b.
    """

    def __init__(
        self,
        stoich: list,
        nc_fl: int,
        fl_idx: int,
        equi_prod: float,
        kin_rate_cte: float,
    ):
        """
        :param stoich: stoichiometric coefficients per component
        :type stoich: list[float]
        :param nc_fl: number of fluid components
        :type nc_fl: int
        :param fl_idx: phase index used to read activities
        :type fl_idx: int
        :param equi_prod: equilibrium product
        :type equi_prod: float
        :param kin_rate_cte: kinetic rate constant
        :type kin_rate_cte: float
        """
        super().__init__(stoich)

        self.nc_fl = nc_fl
        self.fl_idx = fl_idx
        self.equi_prod = equi_prod
        self.kin_rate_cte = kin_rate_cte

    def evaluate(self, pressure, temperature, x, sat_sol):
        """
        :param pressure: pressure [bar] (unused)
        :type pressure: float
        :param temperature: temperature [K] (unused)
        :type temperature: float
        :param x: per-phase composition matrix
        :type x: np.ndarray
        :param sat_sol: solid saturation
        :type sat_sol: float
        :return: (per-component rate vector, driving force ``dQ``)
        :rtype: tuple[list[float], float]
        """
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


register_evaluator("law_of_mass_action", LawOfMassAction, LawOfMassActionConfig)


class HydrateKinetics(Kinetics):
    """Kinetics of methane-hydrate formation/dissociation following Yin (2018)."""

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
        """
        :param components: fluid component names
        :type components: list[str]
        :param phases: phase names (must include ``'Aq'``, ``'V'`` and ``'sI'``)
        :type phases: list[str]
        :param Mw: molecular weights per component [kg/kmol]
        :type Mw: list[float] | np.ndarray
        :param hydrate_eos: hydrate-phase EOS object (passed at construction time only)
        :type hydrate_eos: object
        :param fluid_eos: list of fluid-phase EOS objects (aqueous, vapor)
        :type fluid_eos: list[object]
        :param stoich: stoichiometric coefficients per component
        :type stoich: list[float] | None
        :param perm: permeability [mD]
        :type perm: float
        :param poro: porosity
        :type poro: float
        :param k: reaction rate constant; falls back to a default when ``None``
        :type k: float | None
        :param F_a: Moridis surface-area scale factor
        :type F_a: float
        :param moridis: whether to use Moridis surface-area model
        :type moridis: bool
        :param enthalpy: whether to compute enthalpy of dissociation
        :type enthalpy: bool
        """
        super().__init__(stoich)

        # Preserve raw constructor args for round-trip serialization (the
        # default to_config picks these up via the _init_<field> fallback).
        self._init_components = list(components)
        self._init_phases = list(phases)
        self._init_perm = perm
        self._init_poro = poro
        self._init_k = k
        self._init_F_a = F_a
        self._init_moridis = moridis

        self.hydrate_eos = hydrate_eos
        self.fluid_eos = fluid_eos

        self.water_idx = components.index("H2O")
        self.guest_idx = 0 if self.water_idx == 1 else 1

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
        """Calculate fugacity difference between water in fluid phases and
        water in hydrate phase.

        :param pressure: pressure [bar]
        :type pressure: float
        :param temperature: temperature [K]
        :type temperature: float
        :param x: per-phase composition matrix
        :type x: np.ndarray
        :return: ``(df, xH)`` — fugacity difference and hydrate composition
        :rtype: tuple[float, np.ndarray]
        """
        if x[0, 0] != 0.0:
            f0 = self.fluid_eos[0].fugacity(pressure, temperature, x[0, :])
        else:
            f0 = self.fluid_eos[1].fugacity(pressure, temperature, x[1, :])

        fwH = self.hydrate_eos.fw(pressure, temperature, f0)

        df = fwH - f0[self.water_idx]  # if df < 0 formation, if df > 0 dissociation
        xH = self.hydrate_eos.xH()

        return df, xH

    def evaluate(self, pressure, temperature, x, sat: list):
        """
        :param pressure: pressure [bar]
        :type pressure: float
        :param temperature: temperature [K]
        :type temperature: float
        :param x: per-phase composition matrix
        :type x: np.ndarray
        :param sat: per-phase saturations
        :type sat: list[float]
        :return: (per-component rate vector, fugacity difference)
        :rtype: tuple[list[float], float]
        """
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
        """
        :param pressure: pressure [bar] (unused)
        :type pressure: float
        :param temperature: temperature [K]
        :type temperature: float
        :param x: per-phase composition matrix
        :type x: np.ndarray
        :param sat: per-phase saturations
        :type sat: list[float]
        :return: enthalpy contribution (kJ/day) when ``self.enthalpy`` is True; ``0`` otherwise
        :rtype: float
        """
        if self.enthalpy:
            # Enthalpy change with dissociation (-ive, rate of hydrate component +ive)
            Cf = 33.72995  # J/kg cal/gmol
            if temperature - 273.15 > 0:
                C1, C2 = (13521, -4.02)
            else:
                C1, C2 = (6534, -11.97)
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

    @classmethod
    def from_config(
        cls,
        config: HydrateKineticsConfig,
        *,
        hydrate_eos,
        fluid_eos: list,
    ) -> "HydrateKinetics":
        """Build instance explicitly because non-serializable EOS objects
        (``hydrate_eos``, ``fluid_eos``) must be supplied via context.

        :param config: validated config
        :type config: HydrateKineticsConfig
        :param hydrate_eos: hydrate-phase EOS object
        :type hydrate_eos: object
        :param fluid_eos: list of fluid-phase EOS objects
        :type fluid_eos: list[object]
        :return: HydrateKinetics instance
        :rtype: HydrateKinetics
        """
        return cls(
            components=list(config.components),
            phases=list(config.phases),
            Mw=list(config.Mw),
            hydrate_eos=hydrate_eos,
            fluid_eos=fluid_eos,
            stoich=list(config.stoich) if config.stoich is not None else None,
            perm=config.perm,
            poro=config.poro,
            k=config.k,
            F_a=config.F_a,
            moridis=config.moridis,
            enthalpy=config.enthalpy,
        )


register_evaluator("hydrate_kinetics", HydrateKinetics, HydrateKineticsConfig)


# ---------------------------
# Mineral kinetics working with databases
# ---------------------------


# Default databases directory; resolved relative to this module
_DEFAULT_DB_DIR = Path(__file__).parent / 'databases'


def _resolve_kinetic_db_path(kinetic_database: str | Path) -> Path:
    """Resolve path to kinetics JSON database.

    If ``kinetic_database`` is an existing path, use it. Otherwise, treat it as
    a name within the default databases directory and append ``.json``.

    :param kinetic_database: path or name of the kinetics database
    :type kinetic_database: str | Path
    :return: resolved path to the database file
    :rtype: Path
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
    """Load and normalize a kinetics database file.

    :param db_path: path to JSON kinetics database
    :type db_path: Path
    :return: mapping mineral_name -> mechanism_name -> parameters
    :rtype: dict
    """
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


_SUPPORTED_MECHANISMS = {'acidic', 'neutral', 'carbonate'}


class LinearReactionSurfaceArea(EvaluatorBase):
    """Reaction surface area linear in mineral volume fraction."""

    def __init__(self, initial_area_per_mol: float):
        """
        :param initial_area_per_mol: initial area per mol [m2/mol]
        :type initial_area_per_mol: float
        """
        self.s_init = initial_area_per_mol

    def evaluate(self, vol_fraction):
        """
        :param vol_fraction: mineral volume fraction
        :type vol_fraction: float
        :return: surface area [m2/mol]
        :rtype: float
        """
        return self.s_init * vol_fraction

    def to_config(self) -> LinearReactionSurfaceAreaConfig:
        """Build Config explicitly because the storage attribute ``s_init``
        diverges from the Config field name ``initial_area_per_mol``.

        :return: serialized config
        :rtype: LinearReactionSurfaceAreaConfig
        """
        return LinearReactionSurfaceAreaConfig(initial_area_per_mol=self.s_init)


register_evaluator(
    "linear_reaction_surface_area",
    LinearReactionSurfaceArea,
    LinearReactionSurfaceAreaConfig,
)


class ReactionMechanism:
    """Class representing an Arrhenius-type reaction mechanism with chemical
    affinity term ``(1 - SR**p)**q``.
    """

    def __init__(self, name, temperature_ref, k, Ea, n, p=1, q=1):
        """
        :param name: mechanism name (``'acidic'``, ``'neutral'`` or ``'carbonate'``)
        :type name: str
        :param temperature_ref: reference temperature [K]
        :type temperature_ref: float
        :param k: pre-exponential factor [mol/m2/s]
        :type k: float
        :param Ea: activation energy [J/mol]
        :type Ea: float
        :param n: reaction order with respect to activity
        :type n: float
        :param p: chemical affinity parameter (exponent on ``SR``)
        :type p: float
        :param q: chemical affinity parameter (outer exponent)
        :type q: float
        """
        # maximum saturation ratio threshold for chemical affinity term
        self.SR_threshold = 100
        # universal gas constant [J/mol/K]
        self.R = 8.314472
        self.name = name
        self.temperature_ref = temperature_ref
        self.k = k
        self.Ea = Ea
        self.n = n
        self.p = p
        self.q = q

    def evaluate(self, temperature, activity, SR):
        """Evaluate the reaction rate for a given temperature, activity, and
        saturation ratio.

        :param temperature: temperature [K]
        :type temperature: float
        :param activity: activity of the reactant relevant to the mechanism
        :type activity: float
        :param SR: saturation ratio
        :type SR: float
        :return: reaction rate [mol/s/m2]
        :rtype: float
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


class KineticRate(EvaluatorBase):
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

        :param min_z: minimum composition value (kept for API symmetry)
        :type min_z: float
        :param mineral_name: mineral identifier, e.g. ``'CaCO3'``
        :type mineral_name: str
        :param mechanisms: list of mechanism names (e.g. ``['acidic', 'neutral']``)
        :type mechanisms: list[str]
        :param surface_area_ev: surface area evaluator
        :type surface_area_ev: LinearReactionSurfaceArea
        :param kinetic_database: path or name of kinetics DB (without ``.json``)
        :type kinetic_database: str | Path
        """
        self.min_z = min_z
        self.surface_area_ev = surface_area_ev
        self.kinetic_database = kinetic_database

        db_path = _resolve_kinetic_db_path(kinetic_database)
        registry = _load_kinetic_registry(db_path)

        self.mineral = self._validate_mineral(mineral_name, registry)
        mech_defs = registry[self.mineral]
        normalized_mechs = self._validate_mechanisms(mechanisms, mech_defs)
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
            for mech in normalized_mechs
        ]

    def evaluate(self, kin_state, solid_saturation, rho_s, temperature):
        """Compute kinetic rate [kmol/d/m3] for the configured mineral.

        :param kin_state: dict with activities/saturation ratios from PHREEQC;
            expects keys ``'Act(H+)'``, ``'Act(CO2)'`` and ``'SR_<mineral>'``
        :type kin_state: dict
        :param solid_saturation: solid saturation (volume fraction) of the mineral
        :type solid_saturation: float
        :param rho_s: solid molar density [kmol/m3]
        :type rho_s: float
        :param temperature: temperature [K]
        :type temperature: float
        :return: kinetic rate [kmol/d/m3]
        :rtype: float
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

    @staticmethod
    def _validate_mineral(mineral_name: str, registry: dict) -> str:
        """Validate mineral name against the registry.

        :param mineral_name: mineral identifier
        :type mineral_name: str
        :param registry: loaded kinetics registry
        :type registry: dict
        :return: validated mineral key
        :rtype: str
        """
        if not isinstance(mineral_name, str) or not mineral_name.strip():
            raise ValueError("'mineral_name' must be a non-empty string")
        mineral_key = mineral_name.strip()
        if mineral_key not in registry:
            supported = sorted(registry.keys())
            raise ValueError(
                f"Unsupported mineral '{mineral_name}'. Supported: {supported}"
            )
        return mineral_key

    @staticmethod
    def _validate_mechanisms(mechanisms, mech_defs: dict) -> list[str]:
        """Validate, normalize and de-duplicate the requested mechanism list.

        :param mechanisms: input mechanism names
        :type mechanisms: list[str] | tuple[str, ...]
        :param mech_defs: mechanism definitions for the chosen mineral
        :type mech_defs: dict
        :return: normalized list of mechanism keys
        :rtype: list[str]
        """
        if not isinstance(mechanisms, list | tuple) or len(mechanisms) == 0:
            raise ValueError("'mechanisms' must be a non-empty list of strings")
        clean: list[str] = []
        for mech in mechanisms:
            if not isinstance(mech, str):
                raise ValueError("Mechanism names must be strings")
            mk = mech.strip().lower()
            if mk not in _SUPPORTED_MECHANISMS:
                raise ValueError(
                    f"Unsupported mechanism '{mech}'. "
                    f"Supported: {sorted(_SUPPORTED_MECHANISMS)}"
                )
            if mk not in mech_defs:
                raise ValueError(
                    f"Mechanism '{mk}' is not available for this mineral. "
                    f"Available: {sorted(mech_defs.keys())}"
                )
            if mk not in clean:
                clean.append(mk)
        return clean

    def to_config(self) -> KineticRateConfig:
        """Build Config explicitly to recursively serialize the nested
        surface-area evaluator and to translate stored attributes
        (``mineral`` -> Config field ``mineral_name``, list of
        ``ReactionMechanism`` objects -> list of names).

        :return: serialized config
        :rtype: KineticRateConfig
        """
        if not isinstance(self.surface_area_ev, EvaluatorBase):
            raise TypeError(
                "KineticRate.surface_area_ev must be an EvaluatorBase to serialize; "
                f"got {type(self.surface_area_ev).__name__}"
            )
        return KineticRateConfig(
            min_z=self.min_z,
            mineral_name=self.mineral,
            mechanisms=[m.name for m in self.mechanisms],
            surface_area_ev=self.surface_area_ev.to_config(),
            kinetic_database=str(self.kinetic_database),
        )

    @classmethod
    def from_config(cls, config: KineticRateConfig) -> "KineticRate":
        """Build instance explicitly: materialize the nested surface-area
        Config into a live evaluator before constructing.

        :param config: validated config
        :type config: KineticRateConfig
        :return: KineticRate instance
        :rtype: KineticRate
        """
        surface_area_ev = LinearReactionSurfaceArea.from_config(config.surface_area_ev)
        return cls(
            min_z=config.min_z,
            mineral_name=config.mineral_name,
            mechanisms=list(config.mechanisms),
            surface_area_ev=surface_area_ev,
            kinetic_database=config.kinetic_database,
        )


register_evaluator("kinetic_rate", KineticRate, KineticRateConfig)
