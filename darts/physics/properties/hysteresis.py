from __future__ import annotations

import abc
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from scipy.interpolate import interp1d


class PhaseKind(str, Enum):
    """Canonical classification of a phase for hysteresis evaluators.

    Free-form phase labels in open-DARTS vary ("V"/"Aq", "gas"/"water", "G"/"L", ...).
    Hysteresis evaluators must dispatch on physical role rather than string, so all
    Killough evaluators resolve their phase label through :func:`phase_kind` once at
    construction and store the result here.
    """

    GAS = "gas"
    AQUEOUS = "aqueous"
    OLEIC = "oleic"


# Canonical aliases, lower-cased on lookup. DFM-style multi-liquid labels
# (``"L_a"``, ``"L_b"``) are intentionally absent: they are not classifiable as a
# single oleic phase here, and the current Killough hysteresis model is only defined
# for two-phase gas/water systems.
_PHASE_ALIASES: dict[str, PhaseKind] = {
    "g": PhaseKind.GAS,
    "v": PhaseKind.GAS,
    "gas": PhaseKind.GAS,
    "vapor": PhaseKind.GAS,
    "vapour": PhaseKind.GAS,
    "aq": PhaseKind.AQUEOUS,
    "w": PhaseKind.AQUEOUS,
    "water": PhaseKind.AQUEOUS,
    "brine": PhaseKind.AQUEOUS,
    "h2o": PhaseKind.AQUEOUS,
    "o": PhaseKind.OLEIC,
    "l": PhaseKind.OLEIC,
    "oil": PhaseKind.OLEIC,
    "oleic": PhaseKind.OLEIC,
}


def phase_kind(name: str) -> PhaseKind:
    """Map a free-form phase name to its canonical :class:`PhaseKind`.

    Used by hysteresis evaluators to decide gas vs wetting branches without each class
    re-encoding its own alias set. Raises :class:`ValueError` for unknown names so
    misconfiguration fails at construction, not silently at runtime evaluation.

    :param name: Phase label as supplied by the user (e.g. ``"V"``, ``"Aq"``, ``"gas"``)
    :type name: str
    :returns: Canonical phase kind
    :rtype: PhaseKind
    :raises ValueError: If ``name`` does not match any known alias
    """
    key = name.strip().lower()
    try:
        return _PHASE_ALIASES[key]
    except KeyError as exc:
        raise ValueError(
            f"Unknown phase name {name!r} for hysteresis evaluator. "
            f"Expected one of: {sorted(_PHASE_ALIASES)}. "
            f"DFM-style multi-liquid labels (e.g. 'L_a', 'L_b') are not yet supported."
        ) from exc


class HistoryAwareRelPerm(abc.ABC):
    """
    Abstract base for relative-permeability evaluators that consume OBL history variables.

    :class:`~darts.physics.base.property_container.PropertyContainer` uses
    ``isinstance(..., HistoryAwareRelPerm)`` to decide whether to forward history values from
    the OBL state. All history variables the physics declared in ``history_fields`` are
    unpacked as keyword arguments into :meth:`evaluate`, so concrete subclasses can pick the
    ones they care about (e.g. ``sg_max=...``) and ignore the rest via ``**_``. Plain
    evaluators without this base class are called with saturation only.
    """

    @abc.abstractmethod
    def evaluate(self, sat: float, **history: float) -> float:
        """
        Return relative permeability at a given saturation and history state.

        :param sat: Phase saturation in ``[0, 1]``
        :type sat: float
        :param history: Per-label history values extracted from the OBL state (e.g.
                        ``sg_max=0.42``). Subclasses should name the kwargs they consume;
                        extras should be accepted via ``**_`` to stay forward-compatible.
        :type history: dict[str, float]
        :returns: Relative permeability value
        :rtype: float
        """


class HistoryAwareCapPressure(abc.ABC):
    """
    Abstract base for capillary-pressure evaluators that consume OBL history variables.

    Dispatch contract mirrors :class:`HistoryAwareRelPerm`: the property container unpacks
    ``{label: value}`` for every declared history field as kwargs to :meth:`evaluate`.
    """

    @abc.abstractmethod
    def evaluate(self, sat: float, **history: float) -> float:
        """
        Return capillary pressure at a given saturation and history state.

        :param sat: Phase saturation in ``[0, 1]``
        :type sat: float
        :param history: Per-label history values from the OBL state (see
                        :meth:`HistoryAwareRelPerm.evaluate`).
        :type history: dict[str, float]
        :returns: Capillary pressure value, in the caller's unit convention
        :rtype: float
        """


@dataclass
class KilloughLandModel:
    """
    Killough-Land trapping model for gas hysteresis in the water/gas system.

    Provides the residual (trapped) gas saturation as a function of the historical maximum gas
    saturation, and advances the state variable ``sg_max`` after each converged timestep. The
    Land constant ``C = 1/sgrmax - 1/(1 - swc)`` is computed from connate water and maximum
    residual gas; see Killough (1976), "Reservoir simulation with history-dependent saturation
    functions".

    :ivar swc: Connate water saturation (dimensionless)
    :type swc: float
    :ivar sgrmax: Maximum residual gas saturation when ``sg_max = 1 - swc`` (dimensionless)
    :type sgrmax: float
    """

    swc: float
    sgrmax: float

    @property
    def land_constant(self) -> float:
        """
        Killough-Land trapping constant ``C = 1/sgrmax - 1/(1 - swc)``.

        :returns: Land constant used to compute residual gas saturation
        :rtype: float
        """
        return 1.0 / self.sgrmax - 1.0 / (1.0 - self.swc)

    def residual_gas_saturation(self, sg_max: float) -> float:
        """
        Trapped gas saturation for a given historical maximum.

        Uses the standard Killough-Land form
        ``sgr = sg_max / (1 + C * sg_max)`` after clipping ``sg_max`` to ``[0, 1 - swc]``.

        :param sg_max: Historical maximum gas saturation (dimensionless)
        :type sg_max: float
        :returns: Trapped gas saturation in ``[0, sgrmax]``
        :rtype: float
        """
        sg_max = float(np.clip(sg_max, 0.0, 1.0 - self.swc))
        return sg_max / (1.0 + self.land_constant * sg_max)

    def dissolution_feedback_sgmax(self, sgr_new: float) -> float:
        """
        Map a post-dissolution residual gas saturation back to ``sg_max``.

        This applies the inverse Killough-Land relation so that, after gas
        dissolution reduces the trapped/residual gas saturation, the stored
        history variable remains consistent with that new residual state.

        The input is clipped to ``[0, sgrmax]`` — the algebraic range of
        :meth:`residual_gas_saturation` — because the inverse
        ``sgr / (1 - C * sgr)`` only stays finite and positive for
        ``sgr <= sgrmax`` (the denominator reaches zero at
        ``sgr = 1 / C = sgrmax * (1 - swc) / (1 - swc - sgrmax)``,
        which equals ``sgrmax`` only in the degenerate ``swc = 0``,
        ``sgrmax = 1`` case).

        :param sgr_new: Updated residual gas saturation after dissolution
        :type sgr_new: float
        :returns: Equivalent historical maximum gas saturation in ``[0, 1 - swc]``
        :rtype: float
        """
        sgr_new = float(np.clip(sgr_new, 0.0, self.sgrmax))
        return sgr_new / (1.0 - self.land_constant * sgr_new)

    def update_sg_max(self, sg: float, sg_max: float) -> float:
        """
        Advance ``sg_max`` given the current gas saturation.

        Three cases: (i) ``sg >= sg_max`` — drainage, lift the historical maximum to ``sg``;
        (ii) ``sg < sgr(sg_max)`` — imbibition past trapping, clip to residual; (iii) otherwise
        the historical maximum is preserved.

        :param sg: Current gas saturation
        :type sg: float
        :param sg_max: Previous historical maximum gas saturation
        :type sg_max: float
        :returns: Updated historical maximum gas saturation, in ``[0, 1]``
        :rtype: float
        """
        sg = float(np.clip(sg, 0.0, 1.0))
        sg_max = float(np.clip(sg_max, 0.0, 1.0))
        if sg >= sg_max:
            return sg

        sgr = self.residual_gas_saturation(sg_max)
        if sg < sgr:
            sg_max_new = self.dissolution_feedback_sgmax(sg)
            return float(np.clip(sg_max_new, 0.0, 1.0))
        return sg_max


class _LookupTableMixin:
    @staticmethod
    def normalize_saturation_axis(axis: str | None) -> str | None:
        if axis is None:
            return None

        normalized = axis.strip().lower().replace("-", "_").replace(" ", "_")
        alias_map = {
            "sw": "wetting",
            "sg": "nonwetting",
            "wetting": "wetting",
            "nonwetting": "nonwetting",
            "wetting_saturation": "wetting",
            "nonwetting_saturation": "nonwetting",
        }
        if normalized not in alias_map:
            raise ValueError(f"Unsupported saturation axis label '{axis}'")
        return alias_map[normalized]

    @classmethod
    def parse_section_axis_metadata(cls, line: str) -> str | None:
        stripped = line.lstrip("#").strip()
        if ":" in stripped:
            key, value = stripped.split(":", 1)
        elif "=" in stripped:
            key, value = stripped.split("=", 1)
        else:
            return None

        key = key.strip().lower().replace("-", "_").replace(" ", "_")
        if key not in {"x_axis", "axis", "saturation_axis"}:
            return None
        return cls.normalize_saturation_axis(value)

    @staticmethod
    def infer_section_axis(section: str) -> str:
        # Backward-compatible fallback for legacy tables without explicit metadata.
        del section
        return "wetting"

    @classmethod
    def load_lookup_table(
        cls,
        filename: str,
        section: str,
        target_axis: str | None = None,
    ):
        sat_list = []
        value_list = []
        in_section = False
        source_axis = None
        target_axis = cls.normalize_saturation_axis(target_axis)
        with open(filename, encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith("--"):
                    in_section = line[2:].strip() == section
                    source_axis = None
                    continue
                if not in_section:
                    continue
                if line.startswith("#"):
                    parsed_axis = cls.parse_section_axis_metadata(line)
                    if parsed_axis is not None:
                        source_axis = parsed_axis
                    continue

                parts = line.split()
                if len(parts) < 2:
                    continue
                sat = float(parts[0])
                value = float(parts[1])

                source_axis_eff = source_axis or cls.infer_section_axis(section)
                target_axis_eff = target_axis or source_axis_eff
                if source_axis_eff != target_axis_eff:
                    sat = 1.0 - sat

                sat_list.append(sat)
                value_list.append(value)

        return np.asarray(sat_list, dtype=float), np.asarray(value_list, dtype=float)


class _KilloughRelPermBase(HistoryAwareRelPerm):
    """
    Shared plumbing for Killough relative-permeability evaluators.

    Subclasses implement the drainage curve and the scanning curves; this base dispatches
    between them based on whether the current saturation exceeds ``sg_max`` (drainage path)
    or lies below it (imbibition path with hysteretic scanning curves). Water phase always
    uses the drainage curve.
    """

    def __init__(self, corey: Any, phase: str):
        """
        Bind a Corey parameter object and build the underlying Land model.

        :param corey: Corey parameter container exposing at least ``swc`` and ``sgrmax``
        :type corey: Any
        :param phase: Phase label resolved through :func:`phase_kind`. Killough relperm
                      hysteresis is only defined for the gas / water two-phase system;
                      other phase kinds raise :class:`ValueError`.
        :type phase: str
        """
        self.phase = phase.lower()
        self.phase_kind = phase_kind(phase)
        if self.phase_kind not in (PhaseKind.GAS, PhaseKind.AQUEOUS):
            raise ValueError(
                f"Killough relperm hysteresis only supports gas / aqueous phases; "
                f"got {phase!r} → {self.phase_kind}."
            )
        self.corey = corey
        self.history_model = KilloughLandModel(
            swc=corey.swc,
            sgrmax=corey.sgrmax,
        )

    def evaluate(self, sat: float, sg_max: float = 0.0, **_: float) -> float:
        """
        Return relative permeability, dispatching between drainage and scanning curves.

        Accepts any extra history kwargs the physics may declare via ``history_fields`` (they
        are swallowed by ``**_``) so the generic
        :class:`~darts.physics.base.property_container.PropertyContainer` dispatch stays
        forward-compatible.

        :param sat: Phase saturation in ``[0, 1]``
        :type sat: float
        :param sg_max: Historical maximum gas saturation; ignored on the water phase
        :type sg_max: float
        :returns: Relative permeability value
        :rtype: float
        """
        sat = float(sat)
        if self.phase_kind is PhaseKind.AQUEOUS:
            return float(self.evaluate_drainage(sat))

        if sat >= sg_max:
            return float(self.evaluate_drainage(sat))
        return float(self.evaluate_scanning(sat, sg_max))

    def evaluate_drainage(self, sat: float) -> float:
        """
        Return the drainage (primary, non-hysteretic) relative permeability.

        :param sat: Phase saturation
        :type sat: float
        :returns: Drainage relative permeability
        :rtype: float
        :raises NotImplementedError: Concrete subclass must provide the model
        """
        raise NotImplementedError

    def evaluate_scanning(self, sat: float, sg_max: float) -> float:
        """
        Return the imbibition scanning-curve relative permeability.

        :param sat: Phase saturation
        :type sat: float
        :param sg_max: Historical maximum gas saturation that anchors the scanning curve
        :type sg_max: float
        :returns: Scanning-curve relative permeability
        :rtype: float
        :raises NotImplementedError: Concrete subclass must provide the model
        """
        raise NotImplementedError


class KilloughRelPermCorey(_KilloughRelPermBase):
    """
    Analytical Killough-Corey relative permeability with hysteretic scanning curves.

    Drainage is the standard Corey form parameterised by ``krwe``, ``krge``, ``nw``, ``ng``;
    scanning follows Killough (1976) Eqn. 4 with ``a`` as the shape exponent.
    """

    def __init__(self, corey: Any, phase: str):
        """
        Instantiate the Corey-form evaluator.

        :param corey: Corey parameter container; expected to expose ``swc``, ``sgrmax``, ``sgc``,
                      ``krwe``, ``krge``, ``nw``, ``ng`` and ``a``
        :type corey: Any
        :param phase: Phase label, see :meth:`_KilloughRelPermBase.__init__`
        :type phase: str
        """
        super().__init__(corey, phase)
        self.land_constant = self.history_model.land_constant

    def evaluate_drainage(self, sat: float) -> float:
        if self.phase_kind is PhaseKind.AQUEOUS:
            se = (sat - self.corey.swc) / (1.0 - self.corey.swc - self.corey.sgc)
            se = np.clip(se, 0.0, 1.0)
            return self.corey.krwe * se**self.corey.nw

        se = (sat - self.corey.sgc) / (1.0 - self.corey.swc - self.corey.sgc)
        se = np.clip(se, 0.0, 1.0)
        return self.corey.krge * se**self.corey.ng

    def evaluate_scanning(self, sat: float, sg_max: float) -> float:
        sg_max = min(1.0 - self.corey.swc, float(sg_max))
        sgr = self.history_model.residual_gas_saturation(sg_max)
        sat = min(1.0 - self.corey.swc, float(sat))
        mobile_sg = max(sat - sgr, 0.0)
        denom = max(sg_max - sgr, 1e-12)

        kr_inflection = self.evaluate_drainage(sg_max)
        factor = (mobile_sg / denom) ** self.corey.a
        return kr_inflection * factor


class KilloughRelPermTable(_LookupTableMixin, _KilloughRelPermBase):
    """
    Table-driven Killough relative permeability with cached scanning interpolants.

    Drainage / imbibition / scanning curves are read from a single lookup file split into
    named sections. Scanning interpolants are built lazily per unique ``sg_max`` and cached
    in a bounded LRU to keep memory finite on long runs.
    """

    def __init__(self, corey: Any, phase: str, lookup_file: str = "LookupTable.txt"):
        """
        Load drainage and imbibition tables and prepare the scanning-curve cache.

        :param corey: Corey parameter container; expected to expose ``swc``, ``sgrmax``,
                      ``wetting_type``, ``nowetting_d`` (drainage section) and
                      ``nowetting_i`` (imbibition section)
        :type corey: Any
        :param phase: Phase label, see :meth:`_KilloughRelPermBase.__init__`
        :type phase: str
        :param lookup_file: Path to the lookup-table file with the named sections referenced
                            through ``corey``
        :type lookup_file: str
        """
        super().__init__(corey, phase)
        self.kind = "linear"
        is_wetting = self.phase_kind is PhaseKind.AQUEOUS
        table_section = corey.wetting_type if is_wetting else corey.nowetting_d
        target_axis = "wetting" if is_wetting else "nonwetting"
        self.sat_data, self.kr_data = self.load_lookup_table(
            lookup_file,
            table_section,
            target_axis=target_axis,
        )
        idx = np.argsort(self.sat_data)
        self.sat_data = self.sat_data[idx]
        self.kr_data = self.kr_data[idx]
        self.kr_interpolator = interp1d(
            self.sat_data,
            self.kr_data,
            kind=self.kind,
            bounds_error=False,
            fill_value=(self.kr_data[0], self.kr_data[-1]),
        )

        self.sat_dr, self.kr_dr = self.load_lookup_table(
            lookup_file,
            corey.nowetting_d,
            target_axis="nonwetting",
        )
        idx = np.argsort(self.sat_dr)
        self.sat_dr = self.sat_dr[idx]
        self.kr_dr = self.kr_dr[idx]
        self.sg_max_limit = self.sat_dr[-2]

        self.sat_im, self.kr_im = self.load_lookup_table(
            lookup_file,
            corey.nowetting_i,
            target_axis="nonwetting",
        )
        idx = np.argsort(self.sat_im)
        self.sat_im = self.sat_im[idx]
        self.kr_im = self.kr_im[idx]
        self.sgci_max = self.sat_im[1]
        # Bounded LRU cache over the sg_max -> scanning-interpolator mapping.
        # Without a bound, cells freely populate new sg_max keys each Newton step
        # and memory grows without limit.
        self._scan_cache: OrderedDict[float, interp1d] = OrderedDict()
        self._scan_cache_max = 1024

    def evaluate_drainage(self, sat: float) -> float:
        """
        Return the drainage relative permeability from the primary lookup interpolator.

        :param sat: Phase saturation, in the axis convention of the loaded section
        :type sat: float
        :returns: Drainage relative permeability value
        :rtype: float
        """
        return float(self.kr_interpolator(sat))

    def _make_scanning_interp(self, sg_max: float) -> interp1d:
        """
        Build and cache the scanning-curve interpolator for a given ``sg_max``.

        Constructs a piecewise interpolant that blends the imbibition table (rescaled to
        ``[sgr, sg_max]``) with the drainage tail above ``sg_max``. Degenerate cases where
        ``sg_max == sgr`` fall back to the drainage interpolator.

        :param sg_max: Historical maximum gas saturation, clipped to the table limit
        :type sg_max: float
        :returns: One-dimensional interpolator mapping gas saturation to relative permeability
        :rtype: scipy.interpolate.interp1d
        """
        sg_max = min(float(sg_max), self.sg_max_limit)
        sgr = self.history_model.residual_gas_saturation(sg_max)
        if abs(sg_max - sgr) < 1e-12:
            self._cache_put(sg_max, self.kr_interpolator)
            return self.kr_interpolator

        # TODO: check performance of this interpolation inside evaluator for bigger models
        s1 = np.linspace(sgr, sg_max, 1000)
        s_star = self.sgci_max + (s1 - sgr) * (self.sg_max_limit - self.sgci_max) / (
            sg_max - sgr
        )
        kr_inflection = np.interp(sg_max, self.sat_dr, self.kr_dr)
        kr_ref = np.interp(self.sg_max_limit, self.sat_dr, self.kr_dr)
        kr1 = np.interp(s_star, self.sat_im, self.kr_im) * kr_inflection / kr_ref

        mask = self.sat_dr > sg_max
        s2 = self.sat_dr[mask]
        kr2 = self.kr_dr[mask]

        sats = np.concatenate([s1, s2])
        krs = np.concatenate([kr1, kr2])
        interp = interp1d(
            sats,
            krs,
            kind=self.kind,
            bounds_error=False,
            fill_value=(krs[0], krs[-1]),
        )
        self._cache_put(sg_max, interp)
        return interp

    def _cache_put(self, key: float, value: interp1d) -> None:
        """
        Insert or refresh an entry in the bounded LRU scanning-interpolant cache.

        :param key: Cache key — the ``sg_max`` value that anchors the interpolator
        :type key: float
        :param value: Interpolator produced by :meth:`_make_scanning_interp`
        :type value: scipy.interpolate.interp1d
        :returns: None
        """
        cache = self._scan_cache
        if key in cache:
            cache.move_to_end(key)
        cache[key] = value
        while len(cache) > self._scan_cache_max:
            cache.popitem(last=False)

    def evaluate_scanning(self, sat: float, sg_max: float) -> float:
        """
        Return scanning-curve relative permeability, building the interpolator on first use.

        :param sat: Gas saturation at which to evaluate the scanning curve
        :type sat: float
        :param sg_max: Historical maximum gas saturation anchoring the scanning curve
        :type sg_max: float
        :returns: Relative permeability along the scanning curve at ``sat``
        :rtype: float
        """
        interp = self._scan_cache.get(sg_max)
        if interp is None:
            interp = self._make_scanning_interp(sg_max)
        else:
            self._scan_cache.move_to_end(sg_max)
        return float(interp(sat))


class _KilloughCapillaryPressureBase(HistoryAwareCapPressure):
    """
    Shared plumbing for Killough capillary-pressure evaluators.

    Subclasses provide the drainage and imbibition curves; this base blends them with a
    Killough-form weighting that collapses to the drainage curve when ``sg >= sg_max`` and
    tends to imbibition as ``sg`` approaches the residual gas saturation.
    """

    def __init__(self, corey: Any, phase: str, epsilon: float = 0.1):
        """
        Bind Corey parameters and instantiate the underlying Land model.

        :param corey: Corey parameter container; expected to expose ``swc`` and ``sgrmax``
        :type corey: Any
        :param phase: Phase label resolved through :func:`phase_kind`. Killough Pc
                      hysteresis is only defined for the gas / water two-phase system;
                      other phase kinds raise :class:`ValueError`.
        :type phase: str
        :param epsilon: Regularisation constant in the Killough blending weights
        :type epsilon: float
        """
        self.phase = phase.lower()
        self.phase_kind = phase_kind(phase)
        if self.phase_kind not in (PhaseKind.GAS, PhaseKind.AQUEOUS):
            raise ValueError(
                f"Killough Pc hysteresis only supports gas / aqueous phases; "
                f"got {phase!r} → {self.phase_kind}."
            )
        self.corey = corey
        self.epsilon = epsilon
        self.history_model = KilloughLandModel(
            swc=corey.swc,
            sgrmax=corey.sgrmax,
        )

    def evaluate(self, sat: float, sg_max: float = 0.0, **_: float) -> float:
        """
        Return capillary pressure with Killough scanning behaviour.

        Gas-phase calls short-circuit to zero; water-phase calls blend drainage and
        imbibition curves according to the current ``sg`` and the historical ``sg_max``.
        Extra history kwargs declared by the physics are accepted and ignored via ``**_``.

        :param sat: Wetting-phase saturation
        :type sat: float
        :param sg_max: Historical maximum gas saturation
        :type sg_max: float
        :returns: Capillary pressure in the caller's unit convention
        :rtype: float
        """
        if self.phase_kind is PhaseKind.GAS:
            return 0.0

        sg = float(np.clip(1.0 - sat, 0.0, self.sg_max_limit))
        pc_dr = self.evaluate_drainage(sg)

        sg_max = float(np.clip(sg_max, 0.0, self.sg_max_limit))
        if sg >= sg_max:
            return pc_dr

        sgr = self.history_model.residual_gas_saturation(sg_max)
        numerator = (
            1.0 / (1.0 - sg - (1.0 - sg_max) + self.epsilon) - 1.0 / self.epsilon
        )
        denominator = (
            1.0 / ((1.0 - sgr) - (1.0 - sg_max) + self.epsilon) - 1.0 / self.epsilon
        )
        fraction = numerator / denominator if abs(denominator) > 0 else 0.0
        fraction = float(np.clip(fraction, 0.0, 1.0))
        pc_im = self.evaluate_imbibition(sg)
        return pc_dr + fraction * (pc_im - pc_dr)

    def evaluate_drainage(self, sg: float) -> float:
        """
        Return capillary pressure on the drainage (primary) curve.

        :param sg: Gas saturation
        :type sg: float
        :returns: Drainage capillary pressure
        :rtype: float
        :raises NotImplementedError: Concrete subclass must provide the model
        """
        raise NotImplementedError

    def evaluate_imbibition(self, sg: float) -> float:
        """
        Return capillary pressure on the imbibition curve.

        :param sg: Gas saturation
        :type sg: float
        :returns: Imbibition capillary pressure
        :rtype: float
        :raises NotImplementedError: Concrete subclass must provide the model
        """
        raise NotImplementedError


class KilloughCapillaryPressureTable(_LookupTableMixin, _KilloughCapillaryPressureBase):
    """
    Table-driven Killough capillary pressure with drainage + imbibition sections.

    Both curves are read from the same lookup file, with residual gas saturation derived
    from the Land model. Pressure values are returned in bars (the loaded values are in Pa
    and scaled by 1e-5).
    """

    def __init__(
        self,
        corey: Any,
        phase: str,
        lookup_file: str = "LookupTable.txt",
        epsilon: float = 0.1,
    ):
        """
        Load drainage and imbibition capillary-pressure tables.

        :param corey: Corey parameter container; expected to expose ``swc``, ``sgrmax``,
                      ``Pc_drainage_section`` and ``Pc_imbibition_section``
        :type corey: Any
        :param phase: Phase label, see :meth:`_KilloughCapillaryPressureBase.__init__`
        :type phase: str
        :param lookup_file: Path to the lookup-table file containing the two sections
        :type lookup_file: str
        :param epsilon: Regularisation constant used in the Killough blending weights
        :type epsilon: float
        """
        super().__init__(corey, phase, epsilon=epsilon)

        sat_dr, pc_dr = self.load_lookup_table(
            lookup_file,
            corey.Pc_drainage_section,
            target_axis="nonwetting",
        )
        idx = np.argsort(sat_dr)
        self.sat_dr = sat_dr[idx]
        self.pc_dr = pc_dr[idx]
        self.pc_drainage = interp1d(
            self.sat_dr,
            self.pc_dr,
            kind="linear",
            bounds_error=False,
            fill_value=(self.pc_dr[0], self.pc_dr[-1]),
        )
        # TODO: needed to extract maximum sg from lookup table, apparently it was one before the last element
        self.sg_max_limit = self.sat_dr[-2]

        sat_im, pc_im = self.load_lookup_table(
            lookup_file,
            corey.Pc_imbibition_section,
            target_axis="nonwetting",
        )
        idx = np.argsort(sat_im)
        self.sat_im = sat_im[idx]
        self.pc_im = pc_im[idx]
        self.pc_imbibition = interp1d(
            self.sat_im,
            self.pc_im,
            kind="linear",
            bounds_error=False,
            fill_value=(self.pc_im[0], self.pc_im[-1]),
        )

    def evaluate_drainage(self, sg: float) -> float:
        """
        Evaluate the drainage capillary-pressure interpolator.

        :param sg: Gas saturation
        :type sg: float
        :returns: Drainage capillary pressure in bar (Pa value scaled by 1e-5)
        :rtype: float
        """
        return float(self.pc_drainage(sg)) * 1e-5

    def evaluate_imbibition(self, sg: float) -> float:
        """
        Evaluate the imbibition capillary-pressure interpolator.

        :param sg: Gas saturation
        :type sg: float
        :returns: Imbibition capillary pressure in bar (Pa value scaled by 1e-5)
        :rtype: float
        """
        return float(self.pc_imbibition(sg)) * 1e-5


# Backward-compatible aliases while the rest of the codebase migrates.
KilloughRelPerm = KilloughRelPermTable
KilloughCapillaryPressure = KilloughCapillaryPressureTable

__all__ = [
    "HistoryAwareRelPerm",
    "HistoryAwareCapPressure",
    "KilloughLandModel",
    "KilloughRelPerm",
    "KilloughRelPermCorey",
    "KilloughRelPermTable",
    "KilloughCapillaryPressure",
    "KilloughCapillaryPressureTable",
    "PhaseKind",
    "phase_kind",
]
