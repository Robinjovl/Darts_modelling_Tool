import abc
from typing import Annotated, Literal

import numpy as np
from numba import jit
from pydantic import Field, SerializeAsAny

from darts.physics.properties.evaluator_base import (
    EvaluatorBase,
    EvaluatorConfigBase,
    dispatch_evaluator_config,
    register_evaluator,
)


class ConstantKConfig(EvaluatorConfigBase):
    """Configuration for ConstantK flash evaluator."""

    kind: Literal["constant_k"] = "constant_k"
    K: list[float]
    epsilon: float = Field(1e-8, ge=0)


class SinglePhaseConfig(EvaluatorConfigBase):
    """Configuration for SinglePhase flash evaluator."""

    kind: Literal["single_phase"] = "single_phase"


class SolidFlashConfig(EvaluatorConfigBase):
    """Configuration for SolidFlash wrapper evaluator."""

    kind: Literal["solid_flash"] = "solid_flash"
    flash: Annotated[SerializeAsAny[EvaluatorConfigBase], dispatch_evaluator_config]
    nc_fl: int = Field(ge=1)
    np_fl: int = Field(ge=1)
    ni: int = Field(0, ge=0)
    nc_sol: int = Field(0, ge=0)
    np_sol: int = Field(0, ge=0)


class IonFlashConfig(EvaluatorConfigBase):
    """Configuration for IonFlash wrapper evaluator."""

    kind: Literal["ion_flash"] = "ion_flash"
    flash: Annotated[SerializeAsAny[EvaluatorConfigBase], dispatch_evaluator_config]
    nph: int = Field(ge=1)
    nc: int = Field(ge=1)
    ni: int = Field(ge=0)
    combined_ions: list[float] | None = None


class Flash(EvaluatorBase):
    """Family ABC for vapour-liquid (and solid) equilibrium evaluators.

    Concrete subclasses implement :meth:`evaluate` populating ``self.nu``
    (phase fractions) and ``self.X`` (per-phase compositions).
    """

    def __init__(self, nph, nc, ni=0):
        """
        :param nph: number of phases
        :type nph: int
        :param nc: number of fluid components
        :type nc: int
        :param ni: number of ion species
        :type ni: int
        """
        self.nph = nph
        self.nc = nc
        self.ni = ni
        self.ns = nc + ni

        self.nu = []
        self.X = []
        self.temperature: float = 0.0

    @abc.abstractmethod
    def evaluate(self, pressure, temperature, zc):
        """
        :param pressure: pressure [bar]
        :type pressure: float
        :param temperature: temperature [K]
        :type temperature: float
        :param zc: overall composition (molar fractions)
        :type zc: list[float] | np.ndarray
        :return: error code (``0`` on success)
        :rtype: int
        """
        pass

    def get_flash_results(self):
        """Return the object holding the latest flash result fields
        (``nu``, ``X``, ``temperature``).

        :return: this evaluator instance (default implementation)
        :rtype: Flash
        """
        return self


class SinglePhase(Flash):
    """Trivial single-phase flash: assigns the entire mixture to one phase."""

    def __init__(self, nc):
        """
        :param nc: number of fluid components
        :type nc: int
        """
        super().__init__(nph=1, nc=nc)

    def evaluate(self, pressure, temperature, zc):
        """
        :param pressure: pressure [bar] (unused)
        :type pressure: float
        :param temperature: temperature [K]
        :type temperature: float
        :param zc: overall composition
        :type zc: list[float] | np.ndarray
        :return: error code (always ``0``)
        :rtype: int
        """
        self.nu, self.X = np.array([1.0]), np.array([zc])
        self.temperature = temperature
        return 0


register_evaluator("single_phase", SinglePhase, SinglePhaseConfig)


class ConstantK(Flash):
    """Two-phase Rachford-Rice flash with constant K-values."""

    def __init__(self, nc, ki, eps=1e-11):
        """
        :param nc: number of fluid components
        :type nc: int
        :param ki: K-values (length ``nc``)
        :type ki: list[float] | np.ndarray
        :param eps: convergence epsilon for Rachford-Rice solve
        :type eps: float
        """
        super().__init__(nph=2, nc=nc)

        self.rr_eps = eps
        self.K_values = np.array(ki)

    def evaluate(self, pressure, temperature, zc):
        """
        :param pressure: pressure [bar] (unused)
        :type pressure: float
        :param temperature: temperature [K]
        :type temperature: float
        :param zc: overall composition
        :type zc: list[float] | np.ndarray
        :return: error code (always ``0``)
        :rtype: int
        """
        self.nu, self.X = RR2(self.K_values, zc, self.rr_eps)
        self.temperature = temperature
        return 0

    def to_config(self) -> ConstantKConfig:
        """Build Config explicitly because attribute names diverge from Config
        field names (``K_values`` vs ``K``, ``rr_eps`` vs ``epsilon``).

        :return: serialized config
        :rtype: ConstantKConfig
        """
        return ConstantKConfig(
            K=self.K_values.tolist(),
            epsilon=self.rr_eps,
        )

    @classmethod
    def from_config(cls, config: ConstantKConfig, *, nc: int) -> "ConstantK":
        """Build instance explicitly because the constructor uses ``ki``/``eps``
        argument names rather than the Config's ``K``/``epsilon``.

        :param config: validated flash configuration
        :type config: ConstantKConfig
        :param nc: number of fluid components (must match ``len(config.K)``)
        :type nc: int
        :return: ConstantK instance
        :rtype: ConstantK
        """
        assert len(config.K) == nc, "Length of K must equal number of components"
        return cls(nc, config.K, config.epsilon)


register_evaluator("constant_k", ConstantK, ConstantKConfig)


@jit(nopython=True)
def RR2(k, zc, eps):
    a = 1 / (1 - np.max(k)) + eps
    b = 1 / (1 - np.min(k)) - eps
    k_minus_1 = k - 1

    max_iter = 200  # use enough iterations for V to converge
    tol = 1e-12  # convergence tolerance

    for _i in range(1, max_iter):
        V = 0.5 * (a + b)
        r = np.sum(zc * k_minus_1 / (V * k_minus_1 + 1))
        if abs(r) < tol:
            break

        if r > 0:
            a = V
        else:
            b = V

    if _i >= max_iter:
        print("Flash warning!!!")

    x = zc / (V * k_minus_1 + 1)
    y = k * x

    return [V, 1 - V], [y, x]


class SolidFlash(Flash):
    """
    SolidFlash class is a wrapper around a flash of fluid components/phases and normalized solid that reacts kinetically.
    This is used in a formulation where the solid is a regular component with mole fractions, just does not flow.
    It is a composition of a Flash object.
    During evaluate(), it normalizes fluid composition, evaluates Flash and renormalizes.
    """

    def __init__(
        self,
        flash: Flash,
        nc_fl: int,
        np_fl: int,
        ni: int = 0,
        nc_sol: int = 0,
        np_sol: int = 0,
    ):
        """
        :param flash: inner Flash evaluator for fluid components/phases
        :type flash: Flash
        :param nc_fl: number of fluid components
        :type nc_fl: int
        :param np_fl: number of fluid phases
        :type np_fl: int
        :param ni: number of ions
        :type ni: int
        :param nc_sol: number of solid components
        :type nc_sol: int
        :param np_sol: number of solid phases
        :type np_sol: int
        """
        super().__init__(np_fl, nc_fl, ni)
        self.flash = flash

        self.nc_fl = self.ns
        self.np_fl = self.nph
        self.nc_sol = nc_sol
        self.np_sol = np_sol

    def to_config(self) -> SolidFlashConfig:
        """Build Config explicitly to recursively serialize the wrapped inner
        flash and to derive ``nc_fl``/``np_fl``/``ni`` from it.

        :return: serialized config
        :rtype: SolidFlashConfig
        """
        if not isinstance(self.flash, EvaluatorBase):
            raise TypeError(
                f"SolidFlash.flash must be an EvaluatorBase to serialize; got "
                f"{type(self.flash).__name__}"
            )
        return SolidFlashConfig(
            flash=self.flash.to_config(),
            nc_fl=self.flash.nc,
            np_fl=self.flash.nph,
            ni=self.flash.ni,
            nc_sol=self.nc_sol,
            np_sol=self.np_sol,
        )

    @classmethod
    def from_config(cls, config: SolidFlashConfig) -> "SolidFlash":
        """Build instance explicitly: materialize the nested flash Config into
        a live Flash before constructing.

        :param config: validated config
        :type config: SolidFlashConfig
        :return: SolidFlash instance
        :rtype: SolidFlash
        """
        from darts.physics.properties.evaluator_base import materialize_evaluator

        inner = materialize_evaluator(config.flash, Flash, nc=config.nc_fl)
        return cls(
            flash=inner,
            nc_fl=config.nc_fl,
            np_fl=config.np_fl,
            ni=config.ni,
            nc_sol=config.nc_sol,
            np_sol=config.np_sol,
        )

    def evaluate(self, pressure, temperature, zc):
        """Evaluate flash normalized for solids.

        :param pressure: pressure [bar]
        :type pressure: float
        :param temperature: temperature [K]
        :type temperature: float
        :param zc: overall composition
        :type zc: list[float] | np.ndarray
        :return: error code (``0`` on success)
        :rtype: int
        """
        # Normalize compositions
        zc_sol = zc[self.nc_fl :]
        zc_sol_tot = np.sum(zc_sol)
        zc_norm = zc[: self.nc_fl] / (1.0 - zc_sol_tot)

        # Evaluate flash for normalized composition
        error_output = self.flash.evaluate(pressure, temperature, zc_norm)
        flash_results = self.flash.get_flash_results()
        nu = np.array(flash_results.nu)
        try:
            x = np.array(flash_results.X).reshape(self.np_fl, self.nc_fl)
        except ValueError as e:
            print(e.args[0], pressure, temperature, zc)
            error_output += 1

        # Re-normalize solids and append to nu, x
        NU = np.zeros(self.np_fl + self.np_sol)
        X = np.zeros((self.np_fl + self.np_sol, self.nc_fl + self.nc_sol))
        for j in range(self.np_fl):
            NU[j] = nu[j] * (1.0 - zc_sol_tot)
            X[j, : self.nc_fl] = x[j, :]

        for j in range(self.np_sol):
            NU[self.np_fl + j] = zc_sol[j]
            X[self.np_fl + j, self.nc_fl + j] = 1.0

        self.nu = NU
        self.X = X
        self.temperature = flash_results.temperature

        return error_output


register_evaluator("solid_flash", SolidFlash, SolidFlashConfig)


class IonFlash(Flash):
    """Flash wrapper that handles combined-ion expansion before delegating to
    an inner flash evaluator.
    """

    def __init__(
        self, flash_ev: Flash, nph: int, nc: int, ni: int, combined_ions: list = None
    ):
        """
        :param flash_ev: inner flash evaluator
        :type flash_ev: Flash
        :param nph: number of phases
        :type nph: int
        :param nc: number of regular fluid components
        :type nc: int
        :param ni: number of ion species
        :type ni: int
        :param combined_ions: optional combined-ion stoichiometry weights
        :type combined_ions: list[float] | None
        """
        super().__init__(nph, nc, ni)
        self.flash_ev = flash_ev
        self.combined_ions = combined_ions

    def to_config(self) -> IonFlashConfig:
        """Build Config explicitly to recursively serialize the wrapped inner
        flash; the constructor argument ``flash_ev`` also maps to the Config
        field ``flash``.

        :return: serialized config
        :rtype: IonFlashConfig
        """
        if not isinstance(self.flash_ev, EvaluatorBase):
            raise TypeError(
                f"IonFlash.flash_ev must be an EvaluatorBase to serialize; got "
                f"{type(self.flash_ev).__name__}"
            )
        ci = self.combined_ions
        return IonFlashConfig(
            flash=self.flash_ev.to_config(),
            nph=self.nph,
            nc=self.nc,
            ni=self.ni,
            combined_ions=list(ci) if ci is not None else None,
        )

    @classmethod
    def from_config(cls, config: IonFlashConfig) -> "IonFlash":
        """Build instance explicitly: materialize the nested flash Config into
        a live Flash before constructing.

        :param config: validated config
        :type config: IonFlashConfig
        :return: IonFlash instance
        :rtype: IonFlash
        """
        from darts.physics.properties.evaluator_base import materialize_evaluator

        inner = materialize_evaluator(config.flash, Flash, nc=config.nc)
        return cls(
            flash_ev=inner,
            nph=config.nph,
            nc=config.nc,
            ni=config.ni,
            combined_ions=config.combined_ions,
        )

    def evaluate(self, pressure, temperature, zc):
        """
        :param pressure: pressure [bar]
        :type pressure: float
        :param temperature: temperature [K]
        :type temperature: float
        :param zc: overall composition (with combined ions when applicable)
        :type zc: list[float] | np.ndarray
        :return: error code (always ``0``)
        :rtype: int
        """
        # Uncombine ions into Na+ and Cl- mole fractions
        if self.combined_ions is not None:
            ion_weights = self.combined_ions / np.sum(self.combined_ions)
            zc = np.append(zc[:-1], [ion_weights[0] * zc[-1], ion_weights[1] * zc[-1]])
        nc_tot = len(zc)

        # Evaluates flash, then uses getter for nu and x - for compatibility with DARTS-flash
        self.flash_ev.evaluate(pressure, temperature, zc)
        flash_results = self.flash_ev.get_flash_results()
        self.nu = np.array(flash_results.nu)
        self.X = np.empty(
            (
                self.nph,
                self.nc + 1 if self.combined_ions is not None else self.nc + self.ni,
            )
        )
        self.temperature = flash_results.temperature

        for j in range(self.nph):
            Xj = flash_results.X[j * nc_tot : (j + 1) * nc_tot]

            if self.combined_ions is not None:
                # Normal components +
                self.X[j, : self.nc] = Xj[: self.nc]

                # Sum ions
                self.X[j, self.nc] = np.sum(ion_weights * Xj[self.nc :])
            else:
                self.X[j, :] = Xj

        return 0


register_evaluator("ion_flash", IonFlash, IonFlashConfig)
