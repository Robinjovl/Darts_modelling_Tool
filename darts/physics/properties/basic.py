from typing import Any, Literal

import numpy as np
from pydantic import Field

from darts.physics.properties.evaluator_base import (
    EvaluatorBase,
    EvaluatorConfigBase,
    register_evaluator,
)


class ConstFuncConfig(EvaluatorConfigBase):
    """Configuration for ConstFunc evaluator."""

    kind: Literal["const_func"] = "const_func"
    value: Any


class PhaseRelPermConfig(EvaluatorConfigBase):
    """Configuration for PhaseRelPerm evaluator."""

    kind: Literal["phase_rel_perm"] = "phase_rel_perm"
    phase: str
    swc: float = Field(0.0, ge=0)
    sgr: float = Field(0.0, ge=0)
    kre: float = Field(1.0, ge=0)
    n: float = Field(2.0, ge=0)


class PhaseRelPermVGConfig(EvaluatorConfigBase):
    """Configuration for PhaseRelPerm_VG (Van Genuchten) evaluator."""

    kind: Literal["phase_rel_perm_vg"] = "phase_rel_perm_vg"
    phase: str
    swc: float = Field(0.20, ge=0)
    sgr: float = Field(0.0, ge=0)
    kre: float = Field(1.0, ge=0)
    n: float = Field(4.367, gt=1)


class CapillaryPressureConfig(EvaluatorConfigBase):
    """Configuration for CapillaryPressure evaluator."""

    kind: Literal["capillary_pressure"] = "capillary_pressure"
    nph: int = Field(2, ge=1)
    p_entry: float = 0.0
    swc: float = Field(0.0, ge=0)
    labda: float = 2.0


class CapillaryPressureVGConfig(EvaluatorConfigBase):
    """Configuration for CapillaryPressure_VG (Van Genuchten) evaluator."""

    kind: Literal["capillary_pressure_vg"] = "capillary_pressure_vg"
    nph: int = Field(2, ge=1)
    p_entry: float = 0.0
    swc: float = Field(0.0, ge=0)
    labda: float = 3.3e-6
    n: float = Field(4.367, gt=1)


class RockCompactionEvaluatorConfig(EvaluatorConfigBase):
    """Configuration for RockCompactionEvaluator."""

    kind: Literal["rock_compaction"] = "rock_compaction"
    pref: float = 1.0
    compres: float = 1.45e-5


class ConstFunc(EvaluatorBase):
    """Trivial evaluator that always returns a fixed value, regardless of inputs."""

    def __init__(self, value):
        """
        :param value: constant value to return from :meth:`evaluate`
        :type value: Any
        """
        self.value = value

    def evaluate(self, dummy1=0, dummy2=0, dummy3=0, dummy4=0):
        """
        :param dummy1: ignored
        :param dummy2: ignored
        :param dummy3: ignored
        :param dummy4: ignored
        :return: the stored constant value
        :rtype: Any
        """
        return self.value


register_evaluator("const_func", ConstFunc, ConstFuncConfig)


class PhaseRelPerm(EvaluatorBase):
    """Brooks-Corey relative permeability per phase."""

    def __init__(self, phase, swc=0.0, sgr=0.0, kre=1.0, n=2.0):
        """
        :param phase: phase name (``"oil"``, ``"gas"`` or ``"water"``)
        :type phase: str
        :param swc: connate water saturation
        :type swc: float
        :param sgr: residual gas saturation
        :type sgr: float
        :param kre: end-point relative permeability
        :type kre: float
        :param n: Brooks-Corey exponent
        :type n: float
        """
        self.phase = phase

        self.Swc = swc
        self.Sgr = sgr
        if phase == "oil":
            self.kre = kre
            self.sr = swc
            self.sr1 = sgr
            self.n = n
        elif phase == 'gas':
            self.kre = kre
            self.sr = sgr
            self.sr1 = swc
            self.n = n
        else:  # water
            self.kre = kre
            self.sr = sgr
            self.sr1 = swc
            self.n = n

    def evaluate(self, sat):
        """
        :param sat: phase saturation
        :type sat: float
        :return: relative permeability
        :rtype: float
        """
        if sat >= 1 - self.sr1:
            kr = self.kre
        elif sat <= self.sr:
            kr = 0
        else:
            # general Brooks-Corey
            kr = self.kre * ((sat - self.sr) / (1 - self.Sgr - self.Swc)) ** self.n

        return kr

    def to_config(self) -> PhaseRelPermConfig:
        """Build Config explicitly because the ``swc`` / ``sgr`` fields are
        stored under capitalized attribute names (``Swc`` / ``Sgr``).

        :return: serialized config
        :rtype: PhaseRelPermConfig
        """
        return PhaseRelPermConfig(
            phase=self.phase,
            swc=self.Swc,
            sgr=self.Sgr,
            kre=self.kre,
            n=self.n,
        )

    @classmethod
    def from_config(cls, config: PhaseRelPermConfig) -> "PhaseRelPerm":
        """Build instance explicitly because the constructor uses lowercase
        ``swc``/``sgr`` arguments while the class stores them capitalized.

        :param config: validated config
        :type config: PhaseRelPermConfig
        :return: PhaseRelPerm instance
        :rtype: PhaseRelPerm
        """
        return cls(config.phase, config.swc, config.sgr, config.kre, config.n)


register_evaluator("phase_rel_perm", PhaseRelPerm, PhaseRelPermConfig)


class PhaseRelPerm_VG(EvaluatorBase):  # Van Genuchten
    """Van Genuchten relative permeability per phase."""

    def __init__(self, phase, swc=0.20, sgr=0, kre=1.0, n=4.367):
        """
        :param phase: phase name (``"oil"``, ``"gas"`` or ``"water"``)
        :type phase: str
        :param swc: connate water saturation
        :type swc: float
        :param sgr: residual gas saturation
        :type sgr: float
        :param kre: end-point relative permeability
        :type kre: float
        :param n: Van Genuchten exponent (``> 1``)
        :type n: float
        """
        self.phase = phase
        self.Swc = swc
        self.Sgr = sgr
        if phase == "oil":
            self.kre = kre
            self.sr = swc
            self.sr1 = sgr
            self.n = n
            self.m = 1 - 1 / n
        elif phase == 'gas':
            self.kre = kre
            self.sr = sgr
            self.sr1 = swc
            self.n = n
            self.m = 1 - 1 / n
        else:  # water
            self.kre = 1
            self.sr = 0
            self.sr1 = 0
            self.n = n
            self.m = 1 - 1 / n

    def evaluate(self, sat):
        """
        :param sat: phase saturation
        :type sat: float
        :return: relative permeability
        :rtype: float
        """
        if self.phase == "oil":
            Se = (sat - self.Swc) / (1 - self.Sgr - self.Swc)
            if sat >= 1 - self.sr1:
                kr = self.kre
            elif sat <= self.sr:
                kr = 0
            else:
                kr = np.sqrt(Se) * (1 - (1 - Se ** (1 / self.m)) ** self.m) ** 2

        elif self.phase == 'gas':
            Se = ((1 - sat) - self.Swc) / (1 - self.Sgr - self.Swc)
            if sat >= 1 - self.sr1:
                kr = self.kre
            elif sat <= self.sr:
                kr = 0
            else:
                kr = (1 - Se) ** (1 / 3) * (1 - Se ** (1 / self.m)) ** (2 * self.m)
        return kr

    def to_config(self) -> PhaseRelPermVGConfig:
        """Build Config explicitly because the ``swc`` / ``sgr`` fields are
        stored under capitalized attribute names (``Swc`` / ``Sgr``).

        :return: serialized config
        :rtype: PhaseRelPermVGConfig
        """
        return PhaseRelPermVGConfig(
            phase=self.phase,
            swc=self.Swc,
            sgr=self.Sgr,
            kre=self.kre,
            n=self.n,
        )

    @classmethod
    def from_config(cls, config: PhaseRelPermVGConfig) -> "PhaseRelPerm_VG":
        """Build instance explicitly to bridge lowercase Config field names and
        capitalized ``Swc``/``Sgr`` attributes.

        :param config: validated config
        :type config: PhaseRelPermVGConfig
        :return: PhaseRelPerm_VG instance
        :rtype: PhaseRelPerm_VG
        """
        return cls(config.phase, config.swc, config.sgr, config.kre, config.n)


register_evaluator("phase_rel_perm_vg", PhaseRelPerm_VG, PhaseRelPermVGConfig)


class CapillaryPressure(EvaluatorBase):
    """Brooks-Corey capillary pressure model."""

    def __init__(self, nph=2, p_entry=0, swc=0, labda=2):
        """
        :param nph: number of phases
        :type nph: int
        :param p_entry: entry pressure
        :type p_entry: float
        :param swc: connate water saturation
        :type swc: float
        :param labda: pore-size distribution index
        :type labda: float
        """
        self.nph = nph
        self.swc = swc
        self.p_entry = p_entry
        self.labda = labda
        self.eps = 1e-3

    def evaluate(self, sat):
        """Default evaluator of capillary pressure Pc based on pow.

        :param sat: saturation array
        :type sat: list[float] | np.ndarray
        :return: Pc array of length ``nph``
        :rtype: np.ndarray
        """
        if self.nph > 1:
            Se = (sat[1] - self.swc) / (1 - self.swc)
            if Se < self.eps:
                Se = self.eps
            pc = self.p_entry * Se ** (-1 / self.labda)

            Pc = np.zeros(self.nph, dtype=object)
            Pc[1] = pc
        else:
            Pc = [0.0]
        return Pc


register_evaluator("capillary_pressure", CapillaryPressure, CapillaryPressureConfig)


class CapillaryPressure_VG(EvaluatorBase):  # Van Genuchten
    """Van Genuchten capillary pressure model."""

    def __init__(
        self, nph=2, p_entry=0, swc=0, labda=3.3e-6, n=4.367
    ):  # [labda [Pa^-1]]
        """
        :param nph: number of phases
        :type nph: int
        :param p_entry: entry pressure
        :type p_entry: float
        :param swc: connate water saturation
        :type swc: float
        :param labda: pore-size distribution parameter [Pa^-1]
        :type labda: float
        :param n: Van Genuchten exponent (``> 1``)
        :type n: float
        """
        self.nph = nph
        self.swc = swc
        self.p_entry = p_entry
        self.labda = labda
        self.eps = 1e-3
        self.n = n
        self.m = 1 - 1 / n

    def evaluate(self, sat):
        """Default evaluator of capillary pressure Pc based on pow.

        :param sat: saturation
        :type sat: float
        :return: Pc array of length ``nph``
        :rtype: np.ndarray
        """
        Se = (sat - self.swc) / (1 - self.swc)
        if Se < self.eps:
            Se = self.eps
        pc = 1 / self.labda * (Se ** (-1 / self.m) - 1) ** (1 / self.n)
        Pc = np.zeros(self.nph, dtype=object)
        Pc[1] = pc
        return Pc


register_evaluator(
    "capillary_pressure_vg", CapillaryPressure_VG, CapillaryPressureVGConfig
)


class RockCompactionEvaluator(EvaluatorBase):
    """Linear rock-compaction multiplier:
    ``mult = 1 + compres * (p - pref)``.
    """

    def __init__(self, pref=1.0, compres=1.45e-5):
        """
        :param pref: reference pressure [bar]
        :type pref: float
        :param compres: rock compressibility [1/bar]
        :type compres: float
        """
        self.Pref = pref
        self.compres = compres

    def evaluate(self, pressure):
        """
        :param pressure: pressure [bar]
        :type pressure: float
        :return: rock-compaction multiplier
        :rtype: float
        """
        return 1.0 + self.compres * (pressure - self.Pref)

    def to_config(self) -> RockCompactionEvaluatorConfig:
        """Build Config explicitly because the ``pref`` field is stored under
        the capitalized attribute name ``Pref``.

        :return: serialized config
        :rtype: RockCompactionEvaluatorConfig
        """
        return RockCompactionEvaluatorConfig(
            pref=self.Pref,
            compres=self.compres,
        )

    @classmethod
    def from_config(
        cls, config: RockCompactionEvaluatorConfig
    ) -> "RockCompactionEvaluator":
        """Build instance explicitly to bridge lowercase Config ``pref`` and
        capitalized ``Pref`` attribute.

        :param config: validated config
        :type config: RockCompactionEvaluatorConfig
        :return: RockCompactionEvaluator instance
        :rtype: RockCompactionEvaluator
        """
        return cls(pref=config.pref, compres=config.compres)


register_evaluator(
    "rock_compaction", RockCompactionEvaluator, RockCompactionEvaluatorConfig
)
