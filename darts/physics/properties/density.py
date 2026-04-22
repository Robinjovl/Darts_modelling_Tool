import abc
import warnings
from typing import Literal

import numpy as np
from pydantic import Field

from darts.physics.properties.evaluator_base import (
    EvaluatorBase,
    EvaluatorConfigBase,
    register_evaluator,
)


class DensityBasicConfig(EvaluatorConfigBase):
    """Configuration for DensityBasic evaluator."""

    kind: Literal["density_basic"] = "density_basic"
    compr: float = Field(ge=0)
    dens0: float = Field(gt=0)


class DensityBrineCO2Config(EvaluatorConfigBase):
    """Configuration for DensityBrineCO2 evaluator."""

    kind: Literal["density_brine_co2"] = "density_brine_co2"
    dens0: float = Field(gt=0)
    compr: float = Field(0.0, ge=0)
    p0: float = Field(1.0)
    co2_mult: float = Field(0.0)
    ions_mult: float = Field(0.0)


class Density4IonsConfig(EvaluatorConfigBase):
    """Configuration for Density4Ions evaluator."""

    kind: Literal["density_4ions"] = "density_4ions"
    density: float = Field(gt=0)
    compressibility: float = Field(0.0, ge=0)
    p_ref: float = Field(1.0)
    ions_fac: float = Field(0.0)


class Spivey2004Config(EvaluatorConfigBase):
    """Configuration for Spivey2004 brine density correlation."""

    kind: Literal["spivey_2004"] = "spivey_2004"


class Garcia2001Config(EvaluatorConfigBase):
    """Configuration for Garcia2001 brine+CO2 density correlation."""

    kind: Literal["garcia_2001"] = "garcia_2001"


class Density(EvaluatorBase):
    """Family ABC for density evaluators.

    Concrete subclasses implement :meth:`evaluate` returning a phase density
    [kg/m3] for given pressure, temperature, and composition.
    """

    def __init__(self, components: list = None):
        """
        :param components: list of fluid component names (used only to compute ``self.nc``)
        :type components: list[str] | None
        """
        self.nc = len(components) if components is not None else 0

    @abc.abstractmethod
    def evaluate(self, pressure, temperature, x):
        """
        :param pressure: pressure [bar]
        :type pressure: float
        :param temperature: temperature [K]
        :type temperature: float
        :param x: phase composition (molar fractions)
        :type x: list[float] | np.ndarray
        :return: phase density [kg/m3]
        :rtype: float
        """
        pass


class DensityBasic(Density):
    """Constant density with first-order compressibility:
    ``rho = dens0 * (1 + compr * (p - p0))``.
    """

    def __init__(self, dens0, compr=0.0, p0=1.0):
        """
        :param dens0: reference density [kg/m3]
        :type dens0: float
        :param compr: linear compressibility [1/bar]
        :type compr: float
        :param p0: reference pressure [bar]
        :type p0: float
        """
        super().__init__()
        self.dens0 = dens0
        self.compr = compr
        self.p0 = p0

    def evaluate(self, pressure, temperature: float = None, x: list = None):
        """
        :param pressure: pressure [bar]
        :type pressure: float
        :param temperature: temperature [K] (unused)
        :type temperature: float | None
        :param x: composition (unused)
        :type x: list[float] | None
        :return: density [kg/m3]
        :rtype: float
        """
        return self.dens0 * (1 + self.compr * (pressure - self.p0))


register_evaluator("density_basic", DensityBasic, DensityBasicConfig)


class DensityBrineCO2(DensityBasic):
    """Linear-compressibility density with optional CO2 enrichment offset."""

    def __init__(
        self, components, dens0=1000.0, compr=0.0, p0=1.0, co2_mult=0.0, ions_mult=0.0
    ):
        """
        :param components: list of fluid component names
        :type components: list[str]
        :param dens0: reference density [kg/m3]
        :type dens0: float
        :param compr: linear compressibility [1/bar]
        :type compr: float
        :param p0: reference pressure [bar]
        :type p0: float
        :param co2_mult: CO2 enrichment multiplier
        :type co2_mult: float
        :param ions_mult: ion enrichment multiplier
        :type ions_mult: float
        """
        super().__init__(dens0, compr, p0)
        self.co2_mult = co2_mult
        self.ions_mult = ions_mult
        self._components = list(components)

        if "CO2" in components:
            self.CO2_idx = components.index("CO2")
        else:
            self.CO2_idx = None

    def evaluate(self, pressure, temperature: float, x: list):
        """
        :param pressure: pressure [bar]
        :type pressure: float
        :param temperature: temperature [K]
        :type temperature: float
        :param x: phase composition (molar fractions)
        :type x: list[float] | np.ndarray
        :return: density [kg/m3]
        :rtype: float
        """
        if self.CO2_idx is not None:
            x_co2 = x[self.CO2_idx]
        else:
            x_co2 = 0.0

        density = (self.dens0 + x_co2 * self.co2_mult) * (
            1 + self.compr * (pressure - self.p0)
        )
        return density


register_evaluator("density_brine_co2", DensityBrineCO2, DensityBrineCO2Config)


class Density4Ions(EvaluatorBase):
    """Density evaluator with first-order compressibility plus ion correction
    (Taylor expansion).
    """

    def __init__(self, density, compressibility=0, p_ref=1, ions_fac=0):
        """
        :param density: reference density [kg/m3]
        :type density: float
        :param compressibility: linear compressibility [1/bar]
        :type compressibility: float
        :param p_ref: reference pressure [bar]
        :type p_ref: float
        :param ions_fac: ion correction factor
        :type ions_fac: float
        """
        super().__init__()
        self.density_rc = density
        self.cr = compressibility
        self.p_ref = p_ref
        self.ions_fac = ions_fac

    def evaluate(self, pres, ion_liq_molefrac):
        """
        :param pres: pressure [bar]
        :type pres: float
        :param ion_liq_molefrac: ion liquid molar fraction
        :type ion_liq_molefrac: float
        :return: density [kg/m3]
        :rtype: float
        """
        return self.density_rc * (
            1 + self.cr * (pres - self.p_ref) + self.ions_fac * ion_liq_molefrac
        )

    def to_config(self) -> Density4IonsConfig:
        """Build Config explicitly because attribute names diverge from Config
        field names (``density_rc`` vs ``density``, ``cr`` vs
        ``compressibility``).

        :return: serialized config
        :rtype: Density4IonsConfig
        """
        return Density4IonsConfig(
            density=self.density_rc,
            compressibility=self.cr,
            p_ref=self.p_ref,
            ions_fac=self.ions_fac,
        )

    @classmethod
    def from_config(cls, config: Density4IonsConfig) -> "Density4Ions":
        """Build instance explicitly because Config field names diverge from
        constructor argument names (``density``/``compressibility`` are stored
        under different attribute names).

        :param config: validated config
        :type config: Density4IonsConfig
        :return: Density4Ions instance
        :rtype: Density4Ions
        """
        return cls(
            density=config.density,
            compressibility=config.compressibility,
            p_ref=config.p_ref,
            ions_fac=config.ions_fac,
        )


register_evaluator("density_4ions", Density4Ions, Density4IonsConfig)


class Spivey2004(Density):
    """
    Correlation for brine density: Spivey et al. (2004) - Estimating Density, Formation Volume Factor, Compressibility,
                                                        Methane Solubility, and Viscosity for Oilfield Brines at
                                                        Temperatures From 0 to 275˚ C, Pressures to 200 MPa, and
                                                        Salinities to 5.7 mole/kg
    """

    aw = [
        [-0.127213, 0.645486, 1.03265, -0.070291, 0.639589],
        [4.221, -3.478, 6.221, 0.5182, -0.4405],
        [-11.403, 29.932, 27.952, 0.20684, 0.3768],
    ]
    ab = [
        [-7.925e-5, -1.93e-6, -3.4254e-4, 0.0, 0.0],
        [1.0998e-3, -2.8755e-3, -3.5819e-3, -0.72877, 1.92016],
        [-7.6402e-3, 3.6963e-2, 4.36083e-2, -0.333661, 1.185685],
        [3.746e-4, -3.328e-4, -3.346e-4, 0.0, 0.0],
        [0.0, 0.0, 0.1353, 0.0, 0.0],
        [-1.409, -0.361, -0.2532, 0.0, 9.216],
        [0.0, 5.614, 4.6782, -0.307, 2.6069],
        [-0.1127, 0.2047, -0.0452, 0.0, 0.0],
    ]

    def __init__(self, components: list, ions: list = None, combined_ions: list = None):
        """
        :param components: fluid component names
        :type components: list[str]
        :param ions: ion species names
        :type ions: list[str] | None
        :param combined_ions: optional combined-ion weights
        :type combined_ions: list[float] | None
        """
        super().__init__(components)

        self.H2O_idx = components.index("H2O") if "H2O" in components else None
        if self.H2O_idx is None:
            warnings.warn("H2O not present", stacklevel=2)

        self._components = list(components)
        self.ions = ions
        self.ni = len(ions) if ions is not None else 0
        self.combined_ions = combined_ions

    def evaluate(self, pressure, temperature, x):
        """
        :param pressure: pressure [bar]
        :type pressure: float
        :param temperature: temperature [K]
        :type temperature: float
        :param x: composition (molar fractions)
        :type x: list[float] | np.ndarray
        :return: density [kg/m3]
        :rtype: float
        """
        tc = temperature - 273.15  # Temp in [Celcius]
        tc_100 = tc / 100  # needed many times
        p0 = 700  # reference pressure of 70 MPa

        # Pure water density
        a_w = np.empty(3)
        for i in range(3):
            a_w[i] = (
                self.aw[i][0] * tc_100**2 + self.aw[i][1] * tc_100 + self.aw[i][2]
            ) / (self.aw[i][3] * tc_100**2 + self.aw[i][4] * tc_100 + 1.0)

        rho_w0 = a_w[0]
        Ew = a_w[1]
        Fw = a_w[2]

        if self.ni == 0:
            # Pure water
            Iw = (1.0 / Ew) * np.log(np.abs(Ew * (pressure / p0) + Fw))
            Iw0 = (1.0 / Ew) * np.log(np.abs(Ew * (p0 / p0) + Fw))
            rho = 1000 * rho_w0 * np.exp(Iw - Iw0)
        else:
            # Brine density
            if self.combined_ions is not None:
                ion_weights = self.combined_ions / np.sum(self.combined_ions)
                x_ion = ion_weights[0] * x[self.nc]
            else:
                x_ion = x[self.nc]
            Cm = 55.509 * x_ion / x[self.H2O_idx]

            a_b = np.empty(8)
            for i in range(8):
                a_b[i] = (
                    self.ab[i][0] * tc_100**2 + self.ab[i][1] * tc_100 + self.ab[i][2]
                ) / (self.ab[i][3] * tc_100**2 + self.ab[i][4] * tc_100 + 1.0)
            rho_b0 = (
                rho_w0
                + a_b[0] * Cm**2
                + a_b[1] * Cm**1.5
                + a_b[2] * Cm
                + a_b[3] * np.sqrt(Cm)
            )
            Eb = Ew + a_b[4] * Cm
            Fb = Fw + a_b[5] * Cm**1.5 + a_b[6] * Cm + a_b[7] * np.sqrt(Cm)

            Ib = (1.0 / Eb) * np.log(np.abs(Eb * (pressure / p0) + Fb))
            Ib0 = (1.0 / Eb) * np.log(np.abs(Eb * (p0 / p0) + Fb))
            rho = 1000 * rho_b0 * np.exp(Ib - Ib0)

        return rho  # kg/m3


register_evaluator("spivey_2004", Spivey2004, Spivey2004Config)


class Garcia2001(Spivey2004):
    """
    Correlation for brine density with dissolved CO2: Garcia (2001) - Density of aqueous solutions of CO2
    """

    def __init__(self, components: list, ions: list = None, combined_ions: list = None):
        """
        :param components: fluid component names
        :type components: list[str]
        :param ions: ion species names
        :type ions: list[str] | None
        :param combined_ions: optional combined-ion weights
        :type combined_ions: list[float] | None
        """
        super().__init__(components, ions, combined_ions)

        self.CO2_idx = components.index("CO2") if "CO2" in components else None

    def evaluate(self, pressure, temperature, x):
        """
        :param pressure: pressure [bar]
        :type pressure: float
        :param temperature: temperature [K]
        :type temperature: float
        :param x: composition (molar fractions)
        :type x: list[float] | np.ndarray
        :return: density [kg/m3]
        :rtype: float
        """
        rho_b = super().evaluate(pressure, temperature, x)

        # If CO2 is present, correct density
        if self.CO2_idx is not None:
            # Apparent molar volume of dissolved CO2
            tc = temperature - 273.15  # Temp in [Celcius]
            V_app = (
                37.51 - 9.585e-2 * tc + 8.740e-4 * tc**2 - 5.044e-7 * tc**3
            ) * 1e-6  # in [m3 / mol]

            mCO2 = 55.509 * x[self.CO2_idx] / (x[self.H2O_idx])
            MW = 44.01  # molecular weight of CO2
            rho = (1.0 + mCO2 * MW * 1e-3) / (
                mCO2 * V_app + 1.0 / rho_b
            )  # in [kg / m3]
        else:
            rho = rho_b

        return rho


register_evaluator("garcia_2001", Garcia2001, Garcia2001Config)
