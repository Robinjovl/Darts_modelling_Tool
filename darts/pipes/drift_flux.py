"""
Drift-flux momentum closures for :class:`darts.pipes.pipe.Pipe`.

A drift-flux closure supplies the three model-specific ingredients of the pipe
momentum equation:

* the **profile parameter** ``C0`` (distribution coefficient), through
  :meth:`DriftFluxClosure.profile_parameter`;
* the **drift velocity** ``vD``, through :meth:`DriftFluxClosure.drift_velocity`;
* the **wall friction factor**, through
  :meth:`DriftFluxClosure.fanning_friction_factor`.

Each closure family is one class that owns its own parameters, its own
inclination-angle (``theta``) convention, its own sign convention and its own
default friction model. In particular the historical ``self.vD0 = -vD0`` flip
that used to live at the bottom of ``Pipe.update_drift_velocity`` -- and which
the Tang and Bhagwat-Ghajar branches had to jump over with an early return --
is now internal to :class:`ShiT2WellClosure`: every closure returns the FINAL,
SIGNED drift velocity in the pipe frame (positive from the wellhead downwards).

The closures consume the previous-timestep interface state through the
:class:`FaceProps` dataclass instead of the positional lists with magic indices
(``iter_phases_props0_face[6]`` was the gas viscosity, ``[7]`` the liquid one)
that used to cross this boundary.

Supported closures:

* ``"shi_t2well"``    -- :class:`ShiT2WellClosure`, the historical Holmes/Shi/T2Well closure;
* ``"tang_2019"``     -- :class:`Tang2019Closure`, the unified all-inclination Tang et al. (2019) closure;
* ``"bai_2023"``      -- :class:`Bai2023Closure`, the CO2-specific Bai et al. (2023) variant of Bhagwat-Ghajar;
* ``"bhagwat_ghajar_2014"`` -- :class:`BhagwatGhajar2014Closure`, the original Bhagwat and Ghajar (2014) correlation.
"""

import math
import warnings
from dataclasses import dataclass, field

import numpy as np

from darts.pipes.units import meter, second

# Gravitational acceleration used by every closure (and by Pipe itself).
GRAVITY_ACCELERATION = 9.80665 * meter() / second() ** 2


@dataclass(frozen=True)
class AdjustmentFuncParams:
    # The following parameters are used in the adjustment function f(G,X) used for calculating the drift velocity
    Xm1 = 1.0
    Xm2 = 0.94
    Gm1 = 300.0
    Gm2 = 700.0
    alpha = 0.001
    lambdaa = 199.0  # lambdaa is used because lambda is a reserved keyword in Python


@dataclass(frozen=True)
class TangUnifiedDFParams:
    A: float
    B: float
    a1: float
    a2: float
    N1: float
    N2: float
    N3: float
    N4: float
    m1: float
    m2: float
    m3: float


TANG_PARAMETER_SETS = {
    "olgas": TangUnifiedDFParams(
        A=1.000,
        B=0.773,
        a1=0.591,
        a2=0.786,
        N1=1.968,
        N2=1.759,
        N3=0.574,
        N4=0.477,
        m1=1.000,
        m2=2.300,
        m3=1.000,
    ),
    "tuffp": TangUnifiedDFParams(
        A=1.088,
        B=0.833,
        a1=0.577,
        a2=0.769,
        N1=1.981,
        N2=1.759,
        N3=0.574,
        N4=0.477,
        m1=1.017,
        m2=2.303,
        m3=1.000,
    ),
}


def _linear_interp_extrap(x, x0, x1, y0, y1):
    return y0 + (y1 - y0) / (x1 - x0) * (x - x0)


# --------------------------------------------------------------------------- #
# Face state
# --------------------------------------------------------------------------- #
@dataclass
class FaceProps:
    """
    Previous-timestep pipe-interface ("face") state consumed by a drift-flux closure.

    Every array holds one entry per interface, except :attr:`ift`, which holds one
    entry per *two-phase* interface (i.e. it is aligned with :attr:`indices`)
    because the interfacial tension is only evaluated where both phases are present.

    :param sG: Gas volume fraction (saturation) at the interfaces.
    :param sL: Mobile-liquid volume fraction at the interfaces.
    :param rhoG: Gas density [kg/m3].
    :param rhoL: Liquid density [kg/m3].
    :param muG: Gas dynamic viscosity [Pa.s].
    :param muL: Liquid dynamic viscosity [Pa.s].
    :param rhoM: Mixture density [kg/m3].
    :param rhoM_vM: Mixture mass flux, density times mixture velocity [kg/m2/s].
    :param vM: Mixture velocity [m/s].
    :param vG: Gas velocity [m/s].
    :param vL: Liquid velocity [m/s].
    :param ift: Gas-liquid interfacial tension [N/m] on the two-phase interfaces.
    """

    sG: np.ndarray
    sL: np.ndarray
    rhoG: np.ndarray
    rhoL: np.ndarray
    muG: np.ndarray
    muL: np.ndarray
    rhoM: np.ndarray
    rhoM_vM: np.ndarray
    vM: np.ndarray
    vG: np.ndarray
    vL: np.ndarray
    ift: np.ndarray | None = None
    indices: np.ndarray = field(init=False)

    def __post_init__(self):
        # Interfaces carrying both phases; the closures are only evaluated there.
        self.indices = np.where((self.sG > 0) & (self.sL > 0))[0]

    @classmethod
    def from_pipe_arrays(cls, phases_props_face, velocities, rhoM_face, ift=None):
        """
        Build the face state from the positional arrays ``Pipe`` keeps internally.

        This is the single place that knows the layout of the historical
        positional lists: ``phases_props_face`` is
        ``[xG_mass, xL_mass, sG, sL, rhoG, rhoL, muG, muL]`` and ``velocities`` is
        ``[rhoM*vM, vM, vG, vL]``.

        :param phases_props_face: Interface phase properties, see the layout above.
        :param velocities: Interface velocities, see the layout above.
        :param rhoM_face: Mixture density at the interfaces [kg/m3].
        :param ift: Interfacial tension on the two-phase interfaces [N/m], if known.
        """
        _, _, sG, sL, rhoG, rhoL, muG, muL = phases_props_face
        rhoM_vM, vM, vG, vL = velocities
        return cls(
            sG=sG,
            sL=sL,
            rhoG=rhoG,
            rhoL=rhoL,
            muG=muG,
            muL=muL,
            rhoM=rhoM_face,
            rhoM_vM=rhoM_vM,
            vM=vM,
            vG=vG,
            vL=vL,
            ift=ift,
        )


# --------------------------------------------------------------------------- #
# Friction models
# --------------------------------------------------------------------------- #
#: Reynolds number above which the Colebrook-White branch replaces 16/Re.
LAMINAR_TURBULENT_REYNOLDS = 2400.0

#: Initial guess of the Fanning friction factor for the Colebrook-White solve.
COLEBROOK_INITIAL_GUESS = 0.005

_LN10 = math.log(10.0)


def colebrook(f, Re, relative_roughness):
    """
    Calculate the friction factor using the Colebrook-White correlation (implicit method)

    This is the RESIDUAL of the correlation, kept as the historical scalar entry
    point (it is also exposed as ``Pipe.colebrook``). The friction factor itself
    comes from :func:`colebrook_fanning_factor`, which solves this residual for
    every interface at once.

    :param f: Guessed Fanning friction factor
    :param Re: Reynolds number
    :param relative_roughness: Pipe relative roughness
    """
    f = f[0]  # Make sure f passed to math.sqrt is a float
    # Ensure the friction factor doesn't go negative or zero
    # Return a large value to prevent sqrt of negative number
    if f <= 0:
        return 1e6

    sqrt_f = math.sqrt(f)
    return 1.0 / sqrt_f + 4.0 * math.log10(
        relative_roughness / 3.7065 + 1.2613 / (Re * sqrt_f)
    )


def colebrook_fanning_factor(
    Re, relative_roughness: float, tolerance: float = 1e-13, max_iterations: int = 50
) -> np.ndarray:
    r"""
    Solve the implicit Colebrook-White correlation for EVERY interface at once.

    The correlation

    .. math:: 1/\sqrt{f} + 4 \log_{10}(\epsilon/3.7065 + 1.2613/(Re\sqrt{f})) = 0

    is solved in :math:`x = 1/\sqrt{f}`, where it becomes

    .. math:: g(x) = x + 4 \log_{10}(a + b x),\quad a = \epsilon/3.7065,\ b = 1.2613/Re.

    On ``x > 0`` and for ``a >= 0``, ``g`` is strictly increasing
    (``g' = 1 + 4b/(\ln 10 (a + bx)) > 0``) and concave
    (``g'' = -4b^2/(\ln 10 (a + bx)^2) < 0``), so Newton's method converges
    monotonically to the root from ANY positive starting point: after the first
    step every iterate sits on the same side of the root and no damping,
    bracketing or line search is needed. Five iterations reach machine
    precision from the historical ``f = 0.005`` guess.

    Each entry is frozen as soon as its own Newton step falls below
    ``tolerance`` (relative), so the answer does not depend on how many extra
    sweeps the other entries need.

    This replaces a ``scipy.optimize.fsolve`` call per turbulent interface per
    Newton iteration (review item E4). It is both ~100x faster and ~100x more
    accurate: measured against a 40-digit reference root on the Reynolds numbers
    the DFM CI models actually visit, ``fsolve`` (MINPACK ``hybrd``, forward
    difference Jacobian, ``xtol = 1.49e-8``) lands up to 5.4e2 ULP (7.6e-14
    relative) away from the root, this iteration up to 4 ULP (~5e-16), which is
    the resolution of evaluating the residual in double precision rather than a
    property of the tolerance.

    :param Re: Reynolds number of every interface to solve for. Must be
               positive; callers select the turbulent interfaces first.
    :param relative_roughness: Wall roughness divided by the pipe diameter.
    :param tolerance: Relative size of the Newton step below which an entry is
                      converged. The quadratic convergence turns it into an
                      error of ``tolerance ** 2``, so the default is already at
                      machine precision.
    :param max_iterations: Safety cap on the number of sweeps.
    :return: Fanning friction factor, one entry per Reynolds number.
    """
    Re = np.asarray(Re, dtype=float)
    a = relative_roughness / 3.7065
    b = 1.2613 / Re

    x = np.full(Re.shape, 1.0 / math.sqrt(COLEBROOK_INITIAL_GUESS))
    unconverged = np.ones(Re.shape, dtype=bool)
    for _ in range(max_iterations):
        # a + b*x > 0 for every physical input; the floor only keeps a
        # pathological roughness from turning the logarithm into a NaN.
        u = np.maximum(a + b * x, np.finfo(float).tiny)
        g = x + 4.0 * np.log10(u)
        dx = g / (1.0 + 4.0 * b / (_LN10 * u))
        x = np.where(unconverged, x - dx, x)
        unconverged &= np.abs(dx) > tolerance * np.abs(x)
        if not unconverged.any():
            break
    else:
        warnings.warn(
            f"the Colebrook-White solve did not converge on "
            f"{int(unconverged.sum())} of {Re.size} interfaces in "
            f"{max_iterations} iterations (relative roughness "
            f"{relative_roughness:g}); the friction factor there is only "
            "accurate to the last Newton step.",
            RuntimeWarning,
            stacklevel=2,
        )

    return 1.0 / (x * x)


def wang_darcy_friction_factor(Re: float, relative_roughness: float) -> float:
    """
    Calculate the Darcy friction factor with the Wang et al. (2014)
    supercritical-CO2 correlation adopted by Bai et al. (2023).

    Bai et al. use this correlation in their two-phase pure-CO2
    pressure-gradient model because it was developed for CO2 pipe
    flow. The returned value is a Darcy friction factor; callers that
    use the historical open-DARTS Fanning-friction convention must
    divide this value by 4.

    Scalar entry point, kept because it is exposed as
    ``Pipe.wang_darcy_friction_factor``; :meth:`Wang2014Friction.fanning`
    evaluates the same correlation on a whole array.

    :param Re: Mixture Reynolds number.
    :param relative_roughness: Pipe relative roughness, wall roughness divided by pipe diameter.
    :return: Darcy friction factor.
    """
    return float(
        4.0 * Wang2014Friction().fanning(np.array([float(Re)]), relative_roughness)[0]
    )


class FrictionModel:
    """Base class of the wall-friction closures used with the drift-flux models."""

    name = ""

    def fanning(self, Re, relative_roughness: float) -> np.ndarray:
        """
        Fanning friction factor on every pipe interface.

        :param Re: Mixture Reynolds number on every interface.
        :param relative_roughness: Wall roughness divided by the pipe diameter.
        """
        raise NotImplementedError

    def fanning_pointwise(self, Re, relative_roughness: float) -> np.ndarray:
        """
        Fanning friction factor of every entry of ``Re``, evaluated as the
        correlation itself defines it.

        This is the vectorized form of :meth:`fanning_scalar`, and the entry
        point the Bhagwat-Ghajar family uses for its C0,1 term, where the
        friction factor comes from a correlation-specific Reynolds number
        rather than from the wall Reynolds number.

        It is deliberately NOT the same function as :meth:`fanning`: the two
        disagree at exactly ``Re == 2400``, where the wall-friction path leaves
        the friction factor at zero (a pre-existing quirk of the array path,
        preserved rather than silently changed).

        :param Re: Reynolds numbers to evaluate the correlation at.
        :param relative_roughness: Wall roughness divided by the pipe diameter.
        """
        raise NotImplementedError

    def fanning_scalar(self, Re: float, relative_roughness: float) -> float:
        """
        Fanning friction factor of a single interface.

        A convenience wrapper over :meth:`fanning_pointwise`; it must not be
        called from a per-interface loop (review item E4) -- evaluate the whole
        array in one call instead.
        """
        return float(
            self.fanning_pointwise(np.array([float(Re)]), relative_roughness)[0]
        )


class ColebrookWhiteFriction(FrictionModel):
    """Laminar 16/Re below Re = 2400, implicit Colebrook-White above it."""

    name = "colebrook_white"

    def fanning(self, Re, relative_roughness: float) -> np.ndarray:
        Re = np.asarray(Re, dtype=float)
        ff = np.zeros(len(Re))

        # Laminar connections (Re==0 stays 0)
        lam = (Re > 0.0) & (Re < LAMINAR_TURBULENT_REYNOLDS)
        ff[lam] = 16.0 / Re[lam]

        # Turbulent connections, all solved in one vectorized Newton iteration.
        # Re == 2400 exactly stays at zero here; see fanning_pointwise.
        turb = Re > LAMINAR_TURBULENT_REYNOLDS
        ff[turb] = colebrook_fanning_factor(Re[turb], relative_roughness)

        # # T2Well
        # ff = ((1 / (-4 * np.log10(2 * relative_roughness / 3.7 - 5.02 / Re
        #                 * np.log10(2 * relative_roughness / 3.7 + 13 / Re)))) ** 2)

        # # Chen's correlation (The explicit form of Colebrook-White's correlation)
        # ff = (1 / (-4 * np.log10(relative_roughness / 3.7065 - 5.0452 / Re
        #                 * np.log10(relative_roughness ** 1.1098 / 2.8257 + (7.149 / Re) ** 0.8981)))) ** 2

        return ff

    def fanning_pointwise(self, Re, relative_roughness: float) -> np.ndarray:
        Re = np.asarray(Re, dtype=float)
        ff = np.zeros(Re.shape)
        lam = (Re > 0.0) & (Re < LAMINAR_TURBULENT_REYNOLDS)
        ff[lam] = 16.0 / Re[lam]
        turb = Re >= LAMINAR_TURBULENT_REYNOLDS
        ff[turb] = colebrook_fanning_factor(Re[turb], relative_roughness)
        return ff


class Wang2014Friction(FrictionModel):
    """Wang et al. (2014) supercritical-CO2 friction factor, as adopted by Bai et al. (2023)."""

    name = "wang_2014"

    def fanning(self, Re, relative_roughness: float) -> np.ndarray:
        Re = np.asarray(Re, dtype=float)
        ff = np.zeros(Re.shape)

        lam = (Re > 0.0) & (Re < LAMINAR_TURBULENT_REYNOLDS)
        ff[lam] = 16.0 / Re[lam]

        turb = Re >= LAMINAR_TURBULENT_REYNOLDS
        Re_turb = Re[turb]
        inner = (relative_roughness / 29.36) ** 0.95 + (18.35 / Re_turb) ** 1.108
        argument = relative_roughness / 1.72 - (9.26 / Re_turb) * np.log10(inner)
        # The correlation is only defined for a positive argument. It stays
        # positive for every physical (Re >= 2400, roughness >= 0) input, so
        # this is a guard, not a branch the models take.
        degenerate = argument <= 0.0
        regular = ~degenerate
        ff_turb = np.empty(Re_turb.shape)
        ff_turb[regular] = (-2.34 * np.log10(argument[regular])) ** -2.0 / 4.0
        if degenerate.any():
            # Fall back to the Colebrook-White Fanning form.
            ff_turb[degenerate] = colebrook_fanning_factor(
                Re_turb[degenerate], relative_roughness
            )
        ff[turb] = ff_turb
        return ff

    # The Wang correlation has no separate wall/pointwise convention: the
    # laminar branch ends at Re = 2400 in both.
    fanning_pointwise = fanning


FRICTION_MODELS = {
    "colebrook_white": ColebrookWhiteFriction,
    "wang_2014": Wang2014Friction,
}


def make_friction_model(friction_model) -> FrictionModel:
    """
    Resolve a friction-model name (or instance) into a :class:`FrictionModel`.

    :param friction_model: ``"colebrook_white"``, ``"wang_2014"`` or a FrictionModel instance.
    """
    if isinstance(friction_model, FrictionModel):
        return friction_model
    if friction_model not in FRICTION_MODELS:
        raise ValueError(
            "friction_model must be either 'colebrook_white' or 'wang_2014'."
        )
    return FRICTION_MODELS[friction_model]()


# --------------------------------------------------------------------------- #
# Drift-flux closures
# --------------------------------------------------------------------------- #
class DriftFluxClosure:
    """
    Base class of the drift-flux closures.

    A closure is created from its own parameters, then bound to a pipe geometry
    (and to a friction model) with :meth:`bind`. After that it answers three
    questions for a given previous-timestep :class:`FaceProps` state:
    :meth:`profile_parameter`, :meth:`drift_velocity` and
    :meth:`fanning_friction_factor`.
    """

    #: Public name of the closure, i.e. the ``drift_flux_model`` string.
    name = ""
    #: Friction model used when the caller does not ask for a specific one.
    default_friction_model = "colebrook_white"
    #: ``Pipe`` constructor keywords this closure consumes, mapped to its own keywords.
    pipe_keywords = {}

    g = GRAVITY_ACCELERATION

    def __init__(self):
        self.geometry = None
        self.friction = None

    def __repr__(self):
        return f"{type(self).__name__}(name={self.name!r})"

    # ----------------------------------------------------------------- binding
    def bind(self, pipe_geometry, friction_model=None):
        """
        Bind the closure to a pipe geometry and a friction model.

        A closure caches per-interface parameters of the geometry it is bound to,
        so one closure instance belongs to one pipe. Re-binding it to a different
        geometry (e.g. passing the same instance to two Pipes) is refused rather
        than silently re-pointing the first pipe's closure.

        :param pipe_geometry: :class:`darts.pipes.define_pipe_geometry.PipeGeometry`.
        :param friction_model: Friction-model name or instance; ``None`` selects
                               :attr:`default_friction_model`.
        """
        if self.geometry is not None and self.geometry is not pipe_geometry:
            raise ValueError(
                f"{type(self).__name__} is already bound to pipe "
                f"'{self.geometry.pipe_name}'; build one closure per pipe "
                "instead of sharing an instance."
            )
        self.check_geometry(pipe_geometry)
        self.geometry = pipe_geometry
        self.friction = make_friction_model(
            self.default_friction_model if friction_model is None else friction_model
        )
        self._bind(pipe_geometry)
        return self

    def check_geometry(self, pipe_geometry):
        """Reject geometries the correlation is not valid for. Default: accept all."""

    def _bind(self, pipe_geometry):
        """Precompute the per-interface closure parameters. Default: nothing to do."""

    @property
    def relative_roughness(self) -> float:
        return self.geometry.wall_roughness / self.geometry.pipe_ID

    # ------------------------------------------------------------- the closure
    def profile_parameter(self, face: FaceProps) -> np.ndarray:
        """
        Profile parameter (distribution coefficient) C0 on every pipe interface.

        Interfaces that do not carry both phases get C0 = 1.

        :param face: Previous-timestep interface state.
        :return: C0, one entry per interface.
        """
        raise NotImplementedError

    def drift_velocity(self, face: FaceProps, C0_filtered) -> np.ndarray:
        """
        Drift velocity vD on every pipe interface, in the pipe sign convention.

        The returned value is FINAL and SIGNED: positive is from the wellhead
        downwards, which is the direction ``Pipe`` treats as positive. Callers
        must not post-process the sign.

        :param face: Previous-timestep interface state.
        :param C0_filtered: Profile parameter on the two-phase interfaces
                            (``face.indices``), which the caller may have forced
                            to one when the profile parameter is disabled.
        :return: vD, one entry per interface.
        """
        indices = face.indices
        if indices.size == 0:
            # Single-phase everywhere. The historical implementation produced this
            # array through the shi_t2well sign flip, so keep the negative zeros.
            return -np.zeros(self.geometry.num_interfaces)
        return self._drift_velocity(face, indices, C0_filtered)

    def _drift_velocity(self, face, indices, C0_filtered) -> np.ndarray:
        raise NotImplementedError

    def fanning_friction_factor(self, Re) -> np.ndarray:
        """
        Fanning friction factor at the pipe wall for every interface.

        :param Re: Mixture Reynolds number on every interface.
        """
        return self.friction.fanning(Re, self.relative_roughness)


class KutateladzeDriftFluxClosure(DriftFluxClosure):
    """
    Shared skeleton of the flooding-velocity closures (``shi_t2well``, ``tang_2019``).

    Both build C0 from a Kutateladze-number based flooding velocity

    ``beta = max(sG, Fv * sG * |vM| / v_sgf)``, ``eta = clip((beta - B) / (1 - B), 0, 1)``,
    ``C0 = A / (1 + (A - 1) * eta^2)``

    and blend the drift velocity between the bubble-rise (K = 1.53) and the
    film-flooding (K = C0 * Ku) stages over the gas-fraction window [a1, a2].
    Subclasses supply the Kutateladze number, the flooding-fraction multiplier
    and the drift-velocity expression itself.
    """

    #: Profile-parameter parameters, set by :meth:`_bind` or the constructor.
    profile_A = None
    B = None
    a1 = None
    a2 = None

    def __init__(self):
        super().__init__()
        # Workspace shared between profile_parameter() and drift_velocity():
        # the Kutateladze number and the characteristic velocity are needed by both.
        self.Ku0_filtered = None
        self.vC0_filtered = None

    def kutateladze_number(self, NB0):
        """Kutateladze number from the Bond number NB0, on the two-phase interfaces."""
        raise NotImplementedError

    def flooding_fraction(self, sG, vM, v_sgf, eps):
        """Gas fraction the flooding velocity implies, on the two-phase interfaces."""
        raise NotImplementedError

    def profile_parameter(self, face: FaceProps) -> np.ndarray:
        num_interfaces = self.geometry.num_interfaces
        indices = face.indices
        if indices.size == 0:
            self.Ku0_filtered = None
            self.vC0_filtered = None
            return np.ones(num_interfaces)

        sG_f = face.sG[indices]
        rhoG_f = face.rhoG[indices]
        rhoL_f = face.rhoL[indices]
        rhoM_vM_f = face.rhoM_vM[indices]
        rhoM_f = face.rhoM[indices]
        ift_f = face.ift

        # Calculate C00 from the solution of the previous time step
        eps = np.finfo(float).eps
        # Phase densities merging => no slip, physically consistent: the
        # correlations below take the square root of (rhoL - rhoG), so skip
        # them on such faces and use the homogeneous limit C0 = 1 there.
        drho0 = rhoL_f - rhoG_f
        no_slip0 = drho0 <= eps
        safe_drho0 = np.where(no_slip0, eps, drho0)

        vM0 = rhoM_vM_f / rhoM_f
        NB0 = (self.geometry.pipe_ID**2) * (self.g * safe_drho0 / ift_f)
        self.Ku0_filtered = self.kutateladze_number(NB0)
        self.vC0_filtered = (self.g * ift_f * safe_drho0 / rhoL_f**2) ** 0.25
        v_sgf0 = self.Ku0_filtered * np.sqrt(rhoL_f / rhoG_f) * self.vC0_filtered

        beta0 = np.maximum(sG_f, self.flooding_fraction(sG_f, vM0, v_sgf0, eps))
        beta0 = np.clip(beta0, 0, 1)  # T2Well imposes 0 <= beta0 <= 1
        eta0 = (beta0 - self.B) / (1 - self.B)
        # Shi et al. impose 0 <= eta <= 1; clamping eta**2 alone lets a
        # negative eta grow the denominator, making C0 non-monotonic in beta.
        eta0 = np.clip(eta0, 0.0, 1.0)
        C00_filtered = self.profile_A / (1 + (self.profile_A - 1) * eta0**2)

        # Homogeneous no-slip limit on faces with merging phase densities.
        C00_filtered[no_slip0] = 1.0

        C00 = np.ones(num_interfaces)  # C00 all ones first
        C00[indices] = C00_filtered
        return C00

    def bubble_to_film_transition(self, face, indices, C0_filtered):
        """
        K function smoothing the drift velocity between bubble rise and film flooding.

        :param face: Previous-timestep interface state.
        :param indices: Two-phase interfaces.
        :param C0_filtered: Profile parameter on those interfaces.
        """
        sG = face.sG
        K0_filtered = np.zeros(len(C0_filtered))
        for index, value in enumerate(indices):
            if sG[value] <= self.a1:
                K0_filtered[index] = 1.53
            elif self.a1 < sG[value] < self.a2:
                K0_filtered[index] = 1.53 + (
                    C0_filtered[index] * self.Ku0_filtered[index] - 1.53
                ) / 2 * (
                    1 - np.cos(math.pi * (sG[value] - self.a1) / (self.a2 - self.a1))
                )
            elif sG[value] >= self.a2:
                K0_filtered[index] = C0_filtered[index] * self.Ku0_filtered[index]
        return K0_filtered


class ShiT2WellClosure(KutateladzeDriftFluxClosure):
    """
    Historical Holmes/Shi/T2Well drift-flux closure (the open-DARTS default).

    Owns the maximum profile parameter ``Cmax`` (with the Shi et al. anchor
    parameter sets at Cmax = 1.0 and 1.2 and a linear inter-/extrapolation in
    between), the flooding-velocity multiplier ``Fv``, the inclination factor
    ``m`` and the mist-flow adjustment function f(G, X).

    Sign convention: the correlation is written for a well whose positive
    direction points upwards, while ``Pipe`` counts positive from the wellhead
    downwards, so the returned drift velocity is negated.
    """

    name = "shi_t2well"
    default_friction_model = "colebrook_white"
    pipe_keywords = {"Cmax": "Cmax", "Fv": "Fv"}

    Cku = 142
    Cw = 0.008
    adjustment_func_params = AdjustmentFuncParams()

    def __init__(self, Cmax: float = 1.2, Fv: float = 1.0):
        """
        :param Cmax: A user-specified maximum profile parameter that can be tuned to match the observations and
                     could have a value between 1.0 and 1.5.
        :param Fv: A multiplier on the flooding velocity fraction.
        """
        super().__init__()
        # I did not see anywhere to tell if I can use linear interp and extra here or not.
        if Cmax == 1:
            a1 = 0.06
            a2 = 0.21
            m0 = 1.85
            n1 = 0.21
            n2 = 0.95
        elif 1 < Cmax <= 1.5:
            # Linear interpolation/extrapolation from the two anchor points Cmax=1 and Cmax=1.2.
            # Shi et al. provide anchor sets only at Cmax = 1.0 and 1.2; beyond that the
            # parameters are this code's linear extrapolation.
            a1 = 0.06
            a2 = _linear_interp_extrap(Cmax, 1.0, 1.2, 0.21, 0.12)
            m0 = _linear_interp_extrap(Cmax, 1.0, 1.2, 1.85, 1.27)
            n1 = _linear_interp_extrap(Cmax, 1.0, 1.2, 0.21, 0.24)
            n2 = _linear_interp_extrap(Cmax, 1.0, 1.2, 0.95, 1.08)
            if a2 < a1 + 0.02:
                warnings.warn(
                    f"Cmax={Cmax} extrapolates the Shi et al. drift-velocity "
                    f"transition bound a2 to {a2:.4f}, below a1={a1}; the "
                    "bubble-rise/film-flooding transition window would be "
                    "empty or inverted. Clamping a2 to a1 + 0.02.",
                    stacklevel=2,
                )
                a2 = a1 + 0.02
        else:
            raise ValueError("Cmax value is out of the allowed range [1 to 1.5]")

        self.Cmax = Cmax
        self.Fv = Fv
        self.profile_A = Cmax
        self.B = 2 / Cmax - 1.0667
        self.a1 = a1
        self.a2 = a2
        self._m0 = m0
        self._n1 = n1
        self._n2 = n2
        self.m = None

    def _bind(self, pipe_geometry):
        m0, n1, n2 = self._m0, self._n1, self._n2
        if isinstance(pipe_geometry.inclination_angle_radian, float):
            self.m = (
                m0
                * (np.cos(pipe_geometry.inclination_angle_radian) ** n1)
                * (1 + np.sin(pipe_geometry.inclination_angle_radian)) ** n2
                * np.ones(pipe_geometry.num_interfaces)
            )
        elif isinstance(pipe_geometry.inclination_angle_radian, np.ndarray):
            self.m = (
                m0
                * (np.cos(pipe_geometry.inclination_angle_radian) ** n1)
                * (1 + np.sin(pipe_geometry.inclination_angle_radian)) ** n2
            )

    def kutateladze_number(self, NB0):
        return np.sqrt(
            self.Cku / np.sqrt(NB0) * (np.sqrt(1 + NB0 / (self.Cku**2 * self.Cw)) - 1)
        )

    def flooding_fraction(self, sG, vM, v_sgf, eps):
        return self.Fv * sG * abs(vM) / np.maximum(v_sgf, eps)

    def mist_flow_adjustment(self, X0, G0):
        """
        Adjustment function f(G, X) damping the drift velocity in the mist-flow regime.

        :param X0: Gas mass fraction [-].
        :param G0: Total mass flux [kg/m2/s].
        """
        Xm1 = self.adjustment_func_params.Xm1
        Xm2 = self.adjustment_func_params.Xm2
        Gm1 = self.adjustment_func_params.Gm1
        Gm2 = self.adjustment_func_params.Gm2
        alpha = self.adjustment_func_params.alpha
        lambdaa = self.adjustment_func_params.lambdaa

        # Determinant of the matrix
        numerator = alpha * (
            X0 * (Gm1 - Gm2) - G0 * (Xm1 - Xm2) + (Xm1 * Gm2 - Xm2 * Gm1)
        )

        denominator = np.sqrt((Xm2 - Xm1) ** 2 + (alpha * Gm2 - alpha * Gm1) ** 2)
        Dm = numerator / denominator
        return np.maximum(
            0.0, 1 - np.minimum(1, G0 / Gm1) * np.exp(-lambdaa * Dm * abs(Dm))
        )

    def _drift_velocity(self, face, indices, C0_filtered):
        num_interfaces = self.geometry.num_interfaces
        sG_f = face.sG[indices]
        sL_f = face.sL[indices]
        rhoG_f = face.rhoG[indices]
        rhoL_f = face.rhoL[indices]
        rhoM_f = face.rhoM[indices]
        vM_f = face.vM[indices]

        # Phase densities merging => no slip, physically consistent: the
        # drift-velocity contribution is zero on such faces and the
        # correlations (square roots of rhoL - rhoG) are skipped.
        no_slip0 = (rhoL_f - rhoG_f) <= np.finfo(float).eps

        # Calculate the K function to make a smooth transition of drift velocity between
        # the bubble-rise and film-flooding stages
        K0_filtered = self.bubble_to_film_transition(face, indices, C0_filtered)

        # Calculate the adjustment function for the mist flow regime
        # I'm not sure if X should be multiplied by C0 or not.
        # Calculate gas mass fraction [dimensionless]
        X0 = sG_f * rhoG_f / (sG_f * rhoG_f + sL_f * rhoL_f)
        # Calculate G0: the total mass flux (or total mass flow rate per unit cross-sectional area) [kg/m2/s]
        G0 = rhoM_f * abs(vM_f)
        f0 = self.mist_flow_adjustment(X0, G0)

        # Calculate drift velocity
        vD0 = np.zeros(num_interfaces)  # vD0 all zeros first
        for index, value in enumerate(indices):
            vD0[value] = (
                (1 - C0_filtered[index] * sG_f[index])
                * self.vC0_filtered[index]
                * K0_filtered[index]
                * self.m[value]
                * f0[index]
                / (
                    C0_filtered[index]
                    * sG_f[index]
                    * np.sqrt(rhoG_f[index] / rhoL_f[index])
                    + 1
                    - C0_filtered[index] * sG_f[index]
                )
            )
        vD0[indices[no_slip0]] = 0.0
        # The drift velocity is multiplied by -1 because the positive direction of
        # the well is from the top to the bottom.
        return -vD0


class Tang2019Closure(KutateladzeDriftFluxClosure):
    """
    Unified all-inclination drift-flux closure of Tang et al. (2019).

    Owns the OLGA-S / TUFFP parameter sets, the diameter-dependent Kutateladze
    number (clipped to [1e-6, 3.2]) and the vertical/horizontal drift-velocity
    blend. Its inclination angle theta is measured from the horizontal, so the
    sign of the drift velocity is already carried by ``sin(theta)`` and
    ``cos(theta)``: the returned value needs no further sign flip.
    """

    name = "tang_2019"
    default_friction_model = "colebrook_white"
    pipe_keywords = {"tang_parameter_set": "parameter_set"}

    def __init__(self, parameter_set: str = "olgas"):
        """
        :param parameter_set: Parameterization of the unified model, ``"olgas"`` or ``"tuffp"``.
        """
        super().__init__()
        if parameter_set not in TANG_PARAMETER_SETS:
            raise ValueError("tang_parameter_set must be either 'olgas' or 'tuffp'.")
        self.parameter_set = parameter_set
        self.params = TANG_PARAMETER_SETS[parameter_set]
        self.profile_A = self.params.A
        self.B = self.params.B
        self.a1 = self.params.a1
        self.a2 = self.params.a2
        self.theta = None

    def _bind(self, pipe_geometry):
        # inclination_angle_radian is measured from the vertical direction, theta from the horizontal.
        if isinstance(pipe_geometry.inclination_angle_radian, float):
            self.theta = (
                pipe_geometry.inclination_angle_radian - math.pi / 2.0
            ) * np.ones(pipe_geometry.num_interfaces)
        else:
            self.theta = pipe_geometry.inclination_angle_radian - math.pi / 2.0

    def kutateladze_number(self, NB0):
        Dhat0 = np.sqrt(NB0)
        return np.clip(3.587 - 19.105 / (Dhat0 + 3.333), 1.0e-6, 3.2)

    def flooding_fraction(self, sG, vM, v_sgf, eps):
        return sG * abs(vM) / np.maximum(v_sgf, eps)

    def _drift_velocity(self, face, indices, C0_filtered):
        geom = self.geometry
        num_interfaces = geom.num_interfaces
        sG_f = face.sG[indices]
        sL_f = face.sL[indices]
        rhoG_f = face.rhoG[indices]
        rhoL_f = face.rhoL[indices]
        vM_f = face.vM[indices]

        no_slip0 = (rhoL_f - rhoG_f) <= np.finfo(float).eps

        K0_filtered = self.bubble_to_film_transition(face, indices, C0_filtered)

        eps = np.finfo(float).eps
        safe_rhoG_face = np.maximum(rhoG_f, eps)
        safe_rhoL_face = np.maximum(rhoL_f, eps)
        safe_muL_face = np.maximum(face.muL[indices], eps)

        tang_params = self.params
        theta_filtered = self.theta[indices]

        denominator_vd = (
            C0_filtered * sG_f * np.sqrt(safe_rhoG_face / safe_rhoL_face)
            + 1
            - C0_filtered * sG_f
        )
        vDv = (
            (1 - C0_filtered * sG_f)
            * self.vC0_filtered
            * K0_filtered
            / np.maximum(denominator_vd, eps)
        )

        safe_drho0 = np.where(no_slip0, eps, safe_rhoL_face - safe_rhoG_face)
        N_l = safe_muL_face / np.maximum(
            safe_drho0 * np.power(geom.pipe_ID, 1.5) * math.sqrt(self.g),
            eps,
        )
        N_Eo = self.g * safe_drho0 * (geom.pipe_ID**2) / np.maximum(face.ift, eps)
        vDh = (
            np.sqrt(self.g * geom.pipe_ID)
            * (
                tang_params.N1
                - tang_params.N2
                * (N_l**tang_params.N4)
                / np.maximum(
                    N_Eo**tang_params.N3,
                    eps,
                )
            )
            * sG_f
            * sL_f
        )

        transition_angle = theta_filtered + np.deg2rad(tang_params.m2 * vM_f)
        transition_argument = 50.0 * np.sin(transition_angle)
        transition = 1 - 2 / (1 + np.exp(np.clip(transition_argument, -700.0, 700.0)))
        Re_L = np.abs(vM_f) * safe_rhoL_face * geom.pipe_ID / safe_muL_face
        low_re_multiplier = (1 + 1000.0 / (Re_L + 1000.0)) ** tang_params.m3

        vD_filtered = (
            tang_params.m1 * vDv * np.sin(theta_filtered)
            + transition * vDh * np.cos(theta_filtered)
        ) * low_re_multiplier
        vD_filtered[no_slip0] = 0.0

        vD0 = np.zeros(num_interfaces)
        vD0[indices] = vD_filtered
        return vD0


class BhagwatGhajar2014Closure(DriftFluxClosure):
    """
    Bhagwat and Ghajar (2014) drift-flux correlation with Colebrook-White friction.

    The closure is evaluated interface by interface from the superficial phase
    velocities. Its inclination angle theta is measured from the horizontal and
    already carries the sign of the drift velocity, so no sign flip is applied.

    The correlation signs theta by the FLOW direction while this implementation
    evaluates it in the geometry frame; the two coincide only for vertical
    pipes, which is why :meth:`check_geometry` rejects inclined geometries.
    """

    name = "bhagwat_ghajar_2014"
    default_friction_model = "colebrook_white"
    pipe_keywords = {}

    def __init__(self):
        super().__init__()
        self.theta = None

    def check_geometry(self, pipe_geometry):
        # inclination_angle_radian is measured from the vertical direction.
        inclination_from_vertical = np.atleast_1d(
            np.asarray(pipe_geometry.inclination_angle_radian, dtype=float)
        )
        if np.any(np.abs(inclination_from_vertical) > 1e-8):
            raise ValueError(
                f"drift_flux_model='{self.name}' supports only "
                "vertical pipes: the Bhagwat and Ghajar correlation signs its "
                "inclination angle theta by the FLOW direction, while this "
                "implementation evaluates it in the geometry frame; the two "
                "are equivalent only for vertical wells. Inclined support "
                "requires a flow-aware theta (planned)."
            )

    def _bind(self, pipe_geometry):
        if isinstance(pipe_geometry.inclination_angle_radian, float):
            self.theta = (
                pipe_geometry.inclination_angle_radian - math.pi / 2.0
            ) * np.ones(pipe_geometry.num_interfaces)
        else:
            self.theta = pipe_geometry.inclination_angle_radian - math.pi / 2.0

    # ------------------------------------------------------- family-specific hooks
    def c4_downward_low_froude_condition(self, theta_deg: float, Fr_sg: float) -> bool:
        """
        Check the near-horizontal downward-flow condition used for C4.
        """
        return -50.0 <= theta_deg < 0.0 and Fr_sg <= 0.1

    def profile_reynolds_number(self, sG, sL, mixture_velocity, rhoG, rhoL, muG, muL):
        """
        Reynolds number the C0 correlation uses; the original uses liquid properties.

        Elementwise: every argument is either a scalar or an array of one entry
        per interface, so :meth:`profile_parameter` can evaluate it (and the
        friction factor that follows from it) for the whole pipe in one call.
        """
        return (
            rhoL
            * np.abs(mixture_velocity)
            * self.geometry.pipe_ID
            / np.maximum(muL, np.finfo(float).eps)
        )

    def laplace_factor(self, La: float) -> float:
        """C3, the Laplace-number correction of the drift velocity."""
        return (La / 0.025) ** 0.9 if La < 0.025 else 1.0

    def liquid_holdup_factor(self, sL: float) -> float:
        """Liquid-holdup factor of the drift velocity."""
        return math.sqrt(sL)

    # ------------------------------------------------------------ dimensionless
    def gas_froude_number(
        self, j_g: float, rhoG: float, rhoL: float, theta: float
    ) -> float:
        """
        Calculate the gas superficial Froude number used by Bhagwat and Ghajar.

        :param j_g: Gas superficial velocity [m/s].
        :param rhoG: Gas density [kg/m3].
        :param rhoL: Liquid density [kg/m3].
        :param theta: Pipe inclination angle [rad], measured from horizontal.
        :return: Gas superficial Froude number.
        """
        density_difference = rhoL - rhoG
        cos_theta = math.cos(theta)
        if density_difference <= 0.0 or cos_theta <= np.finfo(float).eps:
            return math.inf
        return (
            math.sqrt(rhoG / density_difference)
            * abs(j_g)
            / math.sqrt(self.g * self.geometry.pipe_ID * cos_theta)
        )

    @staticmethod
    def gas_volumetric_fraction(j_g: float, j_l: float) -> float:
        """
        Calculate gas volumetric flow fraction from superficial velocities.

        :param j_g: Gas superficial velocity [m/s].
        :param j_l: Liquid superficial velocity [m/s].
        :return: Gas volumetric flow fraction.
        """
        total = abs(j_g) + abs(j_l)
        return 0.0 if total <= 0.0 else abs(j_g) / total

    @staticmethod
    def gas_quality(j_g: float, j_l: float, rhoG: float, rhoL: float) -> float:
        """
        Calculate gas mass quality from superficial velocities and densities.

        :param j_g: Gas superficial velocity [m/s].
        :param j_l: Liquid superficial velocity [m/s].
        :param rhoG: Gas density [kg/m3].
        :param rhoL: Liquid density [kg/m3].
        :return: Gas mass quality.
        """
        gas_mass_flux = rhoG * abs(j_g)
        liquid_mass_flux = rhoL * abs(j_l)
        total = gas_mass_flux + liquid_mass_flux
        return 0.0 if total <= 0.0 else gas_mass_flux / total

    @staticmethod
    def c01_downward_low_froude_condition(theta_deg: float, Fr_sg: float) -> bool:
        """
        Check the near-horizontal downward-flow condition used for C0,1.
        """
        return -50.0 <= theta_deg <= 0.0 and Fr_sg <= 0.1

    # ----------------------------------------------------------- per-interface
    def interface_profile_parameter(
        self,
        sG: float,
        sL: float,
        j_g: float,
        j_l: float,
        mixture_velocity: float,
        rhoG: float,
        rhoL: float,
        muG: float,
        muL: float,
        theta: float,
        fanning_f: float = None,
    ) -> float:
        """
        Calculate the B&G-family distribution coefficient on a single interface.

        :param sG: Gas volume fraction.
        :param sL: Liquid volume fraction.
        :param j_g: Gas superficial velocity [m/s].
        :param j_l: Liquid superficial velocity [m/s].
        :param mixture_velocity: Mixture velocity magnitude [m/s].
        :param rhoG: Gas density [kg/m3].
        :param rhoL: Liquid density [kg/m3].
        :param muG: Gas viscosity [Pa.s].
        :param muL: Liquid viscosity [Pa.s].
        :param theta: Pipe inclination angle [rad], measured from horizontal.
        :param fanning_f: Fanning friction factor of the C0,1 term, already
                          evaluated at this interface's profile Reynolds
                          number. :meth:`profile_parameter` passes the whole
                          array in one friction call; ``None`` falls back to a
                          single-interface evaluation, which must not be used
                          from a per-interface loop (review item E4).
        :return: Distribution coefficient C0.
        """
        if rhoL - rhoG <= np.finfo(float).eps:
            # Phase densities merging => no slip, physically consistent:
            # skip the correlation and use the homogeneous limit C0 = 1.
            return 1.0
        Re = self.profile_reynolds_number(
            sG, sL, mixture_velocity, rhoG, rhoL, muG, muL
        )

        if fanning_f is None:
            fanning_f = self.friction.fanning_scalar(Re, self.relative_roughness)
        beta = self.gas_volumetric_fraction(j_g, j_l)
        quality = self.gas_quality(j_g, j_l, rhoG, rhoL)
        Fr_sg = self.gas_froude_number(j_g, rhoG, rhoL, theta)
        theta_deg = math.degrees(theta)

        if self.c01_downward_low_froude_condition(theta_deg, Fr_sg):
            C01 = 0.0
        else:
            C01 = (
                0.2
                * (1.0 - math.sqrt(rhoG / rhoL))
                * ((2.6 - beta) ** 0.15 - math.sqrt(max(fanning_f, 0.0)))
                * (1.0 - quality) ** 1.5
            )

        density_ratio = rhoG / rhoL
        cos_theta = math.cos(theta)
        denominator = max(1.0 + cos_theta, np.finfo(float).eps)
        base = math.sqrt(
            max(
                (1.0 + density_ratio**2 * cos_theta) / denominator,
                np.finfo(float).eps,
            )
        )
        # In Equation 10 of Bhagwat,and Ghajar (2014) paper, sL and 2/5 need to be multiplied, but
        # in Equation 13 of Bai et al. (2023) paper, sL is powered by 2/5. I tried both for the
        # two CI tests comparing isothermal two-phase flow in wellbore with OLGA. The results using
        # power are much closer and make more sense.
        exponent = sL ** (2.0 / 5.0)
        low_re_term = (2.0 - density_ratio**2) / (1.0 + (Re / 1000.0) ** 2)
        high_re_term = (base**exponent + C01) / (
            1.0 + (1000.0 / max(Re, np.finfo(float).eps)) ** 2
        )
        return low_re_term + high_re_term

    def interface_drift_velocity(
        self,
        sL: float,
        j_g: float,
        rhoG: float,
        rhoL: float,
        muL: float,
        sigma: float,
        theta: float,
    ) -> float:
        """
        Calculate the B&G-family drift velocity on a single interface.

        :param sL: Liquid volume fraction.
        :param j_g: Gas superficial velocity [m/s].
        :param rhoG: Gas density [kg/m3].
        :param rhoL: Liquid density [kg/m3].
        :param muL: Liquid dynamic viscosity [Pa.s].
        :param sigma: Gas-liquid surface tension [N/m].
        :param theta: Pipe inclination angle [rad], measured from horizontal.
        :return: Drift velocity [m/s] in the pipe coordinate system.
        """
        density_difference = rhoL - rhoG
        if density_difference <= np.finfo(float).eps:
            # Phase densities merging => no slip, physically consistent:
            # the correlation takes the square root of (rhoL - rhoG).
            return 0.0

        viscosity_ratio = muL / 0.001
        if viscosity_ratio > 10.0:
            C2 = (0.434 / math.log10(viscosity_ratio)) ** 0.15
        else:
            C2 = 1.0

        La = math.sqrt(sigma / (self.g * density_difference)) / self.geometry.pipe_ID
        C3 = self.laplace_factor(La)
        theta_deg = math.degrees(theta)
        Fr_sg = self.gas_froude_number(j_g, rhoG, rhoL, theta)
        C4 = -1.0 if self.c4_downward_low_froude_condition(theta_deg, Fr_sg) else 1.0
        return (
            (0.35 * math.sin(theta) + 0.45 * math.cos(theta))
            * math.sqrt(self.g * self.geometry.pipe_ID * density_difference / rhoL)
            * self.liquid_holdup_factor(sL)
            * C2
            * C3
            * C4
        )

    @staticmethod
    def _superficial_velocities(face, value):
        """Gas and liquid superficial velocities on interface ``value``."""
        jG0 = face.sG[value] * face.vG[value]
        jL0 = face.sL[value] * face.vL[value]
        if jG0 == 0.0 and jL0 == 0.0:
            jG0 = face.sG[value] * face.vM[value]
            jL0 = face.sL[value] * face.vM[value]
        return jG0, jL0

    # ---------------------------------------------------------------- closure
    def profile_parameter(self, face: FaceProps) -> np.ndarray:
        num_interfaces = self.geometry.num_interfaces
        indices = face.indices
        if indices.size == 0:
            return np.ones(num_interfaces)

        eps = np.finfo(float).eps
        sG_f = face.sG[indices]
        # Phase densities merging => no slip, use the homogeneous limit C0 = 1.
        no_slip0 = (face.rhoL[indices] - face.rhoG[indices]) <= eps

        # The C0,1 friction factor of every two-phase interface, in ONE call:
        # its Reynolds number is elementwise in the face state, and the
        # Colebrook-White solve behind it is vectorized (review item E4).
        fanning_f = self.friction.fanning_pointwise(
            self.profile_reynolds_number(
                sG_f,
                face.sL[indices],
                np.abs(face.vM[indices]),
                face.rhoG[indices],
                face.rhoL[indices],
                face.muG[indices],
                face.muL[indices],
            ),
            self.relative_roughness,
        )

        C00_filtered = np.zeros(len(indices))
        for i, idx in enumerate(indices):
            jG0, jL0 = self._superficial_velocities(face, idx)
            C00_filtered[i] = self.interface_profile_parameter(
                face.sG[idx],
                face.sL[idx],
                jG0,
                jL0,
                abs(face.vM[idx]),
                face.rhoG[idx],
                face.rhoL[idx],
                face.muG[idx],
                face.muL[idx],
                self.theta[idx],
                fanning_f=fanning_f[i],
            )
        # The drift-flux velocity reconstruction divides by
        # rhoM_adjusted = C0*sG*rhoG + (1 - C0*sG)*rhoL and requires
        # C0*sG < 1 (Shi enforces this structurally via the eta ramp;
        # B&G does not).
        C00_filtered = np.minimum(
            C00_filtered,
            (1.0 - 1e-8) / np.maximum(sG_f, eps),
        )

        C00_filtered[no_slip0] = 1.0

        C00 = np.ones(num_interfaces)
        C00[indices] = C00_filtered
        return C00

    def _drift_velocity(self, face, indices, C0_filtered):
        vD0 = np.zeros(self.geometry.num_interfaces)
        for index, value in enumerate(indices):
            jG0, _ = self._superficial_velocities(face, value)
            vD0[value] = self.interface_drift_velocity(
                face.sL[value],
                jG0,
                face.rhoG[value],
                face.rhoL[value],
                face.muL[value],
                face.ift[index],
                self.theta[value],
            )
        return vD0


class Bai2023Closure(BhagwatGhajar2014Closure):
    """
    Bai et al. (2023) CO2-specific pipe-flow model.

    It adopts Bhagwat and Ghajar (2014) with Bai-specific Reynolds number
    (mixture instead of liquid properties), friction factor (Wang et al. 2014
    supercritical-CO2), Laplace-number branch, liquid-holdup exponent and C4
    condition.
    """

    name = "bai_2023"
    default_friction_model = "wang_2014"
    pipe_keywords = {}

    def c4_downward_low_froude_condition(self, theta_deg: float, Fr_sg: float) -> bool:
        return -50.0 <= theta_deg <= 0.0 and Fr_sg <= 0.1

    def profile_reynolds_number(self, sG, sL, mixture_velocity, rhoG, rhoL, muG, muL):
        rho_m = (sG * rhoG + sL * rhoL) / (sG + sL)
        mu_m = (sG * muG + sL * muL) / (sG + sL)
        return (
            rho_m
            * np.abs(mixture_velocity)
            * self.geometry.pipe_ID
            / np.maximum(mu_m, np.finfo(float).eps)
        )

    def laplace_factor(self, La: float) -> float:
        return (La / 0.025) ** 0.9 if La > 0.025 else 1.0

    def liquid_holdup_factor(self, sL: float) -> float:
        return sL


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
#: Closure families that share the Bhagwat-Ghajar (2014) skeleton.
BHAGWAT_GHAJAR_DRIFT_FLUX_MODELS = ("bai_2023", "bhagwat_ghajar_2014")

DRIFT_FLUX_CLOSURES = {
    "shi_t2well": ShiT2WellClosure,
    "tang_2019": Tang2019Closure,
    "bai_2023": Bai2023Closure,
    "bhagwat_ghajar_2014": BhagwatGhajar2014Closure,
}

SUPPORTED_DRIFT_FLUX_MODELS = tuple(DRIFT_FLUX_CLOSURES)

#: ``Pipe`` constructor keywords that only some closures accept.
CLOSURE_SPECIFIC_PIPE_KEYWORDS = ("Cmax", "Fv", "tang_parameter_set")


def make_drift_flux_closure(drift_flux_model, **pipe_keywords) -> DriftFluxClosure:
    """
    Build the drift-flux closure named by ``drift_flux_model``.

    Keyword arguments are the ``Pipe`` constructor keywords that belong to one
    closure family only (``Cmax`` and ``Fv`` for ``shi_t2well``,
    ``tang_parameter_set`` for ``tang_2019``). ``None`` means "not given";
    passing a keyword the selected closure does not accept raises a ValueError
    instead of being silently ignored.

    :param drift_flux_model: Closure name, or an already-built
                             :class:`DriftFluxClosure` instance.
    :param pipe_keywords: Closure-specific ``Pipe`` constructor keywords.
    """
    unknown = set(pipe_keywords) - set(CLOSURE_SPECIFIC_PIPE_KEYWORDS)
    if unknown:
        raise TypeError(
            f"make_drift_flux_closure() got unexpected keyword(s) {sorted(unknown)}"
        )
    given = {name: value for name, value in pipe_keywords.items() if value is not None}

    if isinstance(drift_flux_model, DriftFluxClosure):
        if given:
            raise ValueError(
                f"{sorted(given)} cannot be combined with a drift-flux closure "
                "instance; configure the closure object itself instead."
            )
        return drift_flux_model

    if (
        not isinstance(drift_flux_model, str)
        or drift_flux_model not in DRIFT_FLUX_CLOSURES
    ):
        raise ValueError(
            "drift_flux_model must be one of "
            f"{', '.join(repr(model) for model in SUPPORTED_DRIFT_FLUX_MODELS)}."
        )

    closure_class = DRIFT_FLUX_CLOSURES[drift_flux_model]
    accepted = closure_class.pipe_keywords
    irrelevant = sorted(set(given) - set(accepted))
    if irrelevant:
        owners = {
            keyword: sorted(
                name
                for name, cls in DRIFT_FLUX_CLOSURES.items()
                if keyword in cls.pipe_keywords
            )
            for keyword in irrelevant
        }
        details = "; ".join(
            f"{keyword} only applies to drift_flux_model in "
            f"{', '.join(repr(owner) for owner in owners[keyword])}"
            for keyword in irrelevant
        )
        raise ValueError(
            f"{irrelevant} cannot be used with drift_flux_model="
            f"{drift_flux_model!r}: {details}. Drop the argument or switch the "
            "drift-flux model."
        )

    return closure_class(**{accepted[name]: value for name, value in given.items()})
