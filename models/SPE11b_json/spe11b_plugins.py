"""Custom DARTS plugins for the schema-first SPE11b JSON model.

The SPE11b benchmark uses physics-grade evaluators (DartsFlash PT flash,
Peng-Robinson EoS density/enthalpy, Garcia2001 aqueous density, Fenghour1998
gas viscosity, Islam2012 aqueous viscosity, modified Brooks-Corey rel-perm)
that are not in the built-in :data:`darts.api.type_registry.TYPE_REGISTRY`.
This sidecar registers them via the ``plugin_registry.modules`` mechanism
so the JSON config can name them by ``type_id``.

The flash + EoS instances are cached at module load so the seven property
regions of SPE11b share a single DartsFlash setup rather than each
constructing its own (which would multiply the EoS data structures 7x).

Plugin type_ids exposed:

* ``flash/SPE11bVLAq@v1``         — DartsFlash VLAq PT flash for CO2-H2O
* ``density/SPE11bGasPR@v1``      — Peng-Robinson EoS density (gas phase)
* ``density/SPE11bAqGarcia@v1``   — Garcia2001 aqueous density (CO2-H2O)
* ``viscosity/SPE11bGasFenghour@v1`` — Fenghour1998 supercritical CO2 viscosity
* ``viscosity/SPE11bAqIslam@v1``  — Islam2012 aqueous viscosity (CO2-H2O)
* ``enthalpy/SPE11bGasPR@v1``     — PR EoS enthalpy (gas phase)
* ``enthalpy/SPE11bAqEoS@v1``     — Aq activity-model enthalpy
* ``relperm/SPE11bModBC@v1``      — modified Brooks-Corey relperm (per-phase)

The module is loaded by the schema-first JSON model via::

    {"plugin_registry": {"modules": ["spe11b_plugins.py"]}}
"""

from __future__ import annotations

from typing import Any, Literal

from dartsflash.components import CompData
from dartsflash.libflash import EoS, InitialGuess
from dartsflash.mixtures import DARTSFlash, VLAq

from darts.physics.properties.density import Garcia2001
from darts.physics.properties.eos_properties import EoSDensity, EoSEnthalpy
from darts.physics.properties.viscosity import Fenghour1998, Islam2012

# ---------------------------------------------------------------------------
# Shared flash + EoS state — built once, reused across property regions.
# ---------------------------------------------------------------------------

_COMPONENTS = ("H2O", "CO2")

_FLASH_CACHE: dict[tuple[str, ...], dict[str, Any]] = {}


def _get_flash_bundle(components: tuple[str, ...] = _COMPONENTS) -> dict[str, Any]:
    """Return a cached ``{flash_ev, pr, aq, Mw}`` bundle for ``components``.

    Mirrors the SPE11b ``model_b.set_physics`` flash setup: VLAq mixture with
    Peng-Robinson gas EoS (root order ``[STABLE]``, Wilson initial guesses,
    stability_tol=1e-20) and an Aq activity-model EoS, initialised for a
    PT flash on the 270-500 K window.
    """
    key = tuple(components)
    if key in _FLASH_CACHE:
        return _FLASH_CACHE[key]

    comp_data = CompData(list(components), setprops=True)
    flash_ev = VLAq(comp_data, hybrid=True)
    flash_ev.set_vl_eos(
        "PR",
        root_order=[EoS.STABLE],
        trial_comps=[InitialGuess.Yi.Wilson],
        stability_tol=1e-20,
        switch_tol=1e-2,
        max_iter=50,
        use_gmix=False,
    )
    flash_ev.set_aq_eos("Aq", stability_tol=1e-20, max_iter=10, use_gmix=True)
    pr = flash_ev.eos["VL"]
    aq = flash_ev.eos["Aq"]
    flash_ev.init_flash(
        flash_type=DARTSFlash.FlashType.PTFlash,
        eos_order=["VL", "Aq"],
        t_min=270.0,
        t_max=500.0,
        t_init=300.0,
    )
    bundle = {"flash_ev": flash_ev, "pr": pr, "aq": aq, "Mw": list(comp_data.Mw)}
    _FLASH_CACHE[key] = bundle
    return bundle


# ---------------------------------------------------------------------------
# Plugin constructors — each takes a config dict (via _wrap_constructor) and
# returns a live evaluator instance.
# ---------------------------------------------------------------------------


def make_flash_vlaq(
    components: list[str] | tuple[str, ...] = _COMPONENTS, **_: Any
):
    """Return the shared VLAq PT-flash evaluator.

    :param components: component names (currently must be ``["H2O", "CO2"]``)
    :type components: list[str] | tuple[str, ...]
    :returns: the cached :class:`dartsflash.mixtures.VLAq` flash instance
    """
    bundle = _get_flash_bundle(tuple(components))
    return bundle["flash_ev"]


def make_density_pr_gas(
    components: list[str] | tuple[str, ...] = _COMPONENTS, **_: Any
) -> EoSDensity:
    """Build an :class:`EoSDensity` against the shared PR EoS (gas phase).

    :param components: component names; used to pick the right cached bundle
    :type components: list[str] | tuple[str, ...]
    :returns: PR EoS density evaluator
    :rtype: darts.physics.properties.eos_properties.EoSDensity
    """
    bundle = _get_flash_bundle(tuple(components))
    return EoSDensity(eos=bundle["pr"], Mw=bundle["Mw"])


def make_density_garcia_aq(
    components: list[str] | tuple[str, ...] = _COMPONENTS, **_: Any
) -> Garcia2001:
    """Build a :class:`Garcia2001` aqueous-density evaluator.

    :param components: component names passed verbatim to Garcia2001
    :type components: list[str] | tuple[str, ...]
    :returns: aqueous-density evaluator for CO2-H2O
    :rtype: darts.physics.properties.density.Garcia2001
    """
    return Garcia2001(components=list(components))


def make_viscosity_fenghour(**_: Any) -> Fenghour1998:
    """Build the parameter-free Fenghour-1998 supercritical CO2 viscosity.

    :returns: Fenghour-1998 viscosity evaluator
    :rtype: darts.physics.properties.viscosity.Fenghour1998
    """
    return Fenghour1998()


def make_viscosity_islam(
    components: list[str] | tuple[str, ...] = _COMPONENTS, **_: Any
) -> Islam2012:
    """Build an Islam-2012 aqueous viscosity evaluator.

    :param components: component names passed to Islam2012
    :type components: list[str] | tuple[str, ...]
    :returns: aqueous-phase viscosity evaluator
    :rtype: darts.physics.properties.viscosity.Islam2012
    """
    return Islam2012(components=list(components))


def make_enthalpy_pr_gas(
    components: list[str] | tuple[str, ...] = _COMPONENTS, **_: Any
) -> EoSEnthalpy:
    """Build an EoS enthalpy evaluator against the PR (gas) EoS.

    :param components: component names; used to pick the right cached bundle
    :type components: list[str] | tuple[str, ...]
    :returns: gas-phase enthalpy evaluator
    :rtype: darts.physics.properties.eos_properties.EoSEnthalpy
    """
    bundle = _get_flash_bundle(tuple(components))
    return EoSEnthalpy(eos=bundle["pr"])


def make_enthalpy_aq(
    components: list[str] | tuple[str, ...] = _COMPONENTS, **_: Any
) -> EoSEnthalpy:
    """Build an EoS enthalpy evaluator against the Aq activity-model EoS.

    :param components: component names; used to pick the right cached bundle
    :type components: list[str] | tuple[str, ...]
    :returns: aqueous-phase enthalpy evaluator
    :rtype: darts.physics.properties.eos_properties.EoSEnthalpy
    """
    bundle = _get_flash_bundle(tuple(components))
    return EoSEnthalpy(eos=bundle["aq"])


# ---------------------------------------------------------------------------
# Modified Brooks-Corey relative permeability (per-phase).
# Lifted from models/SPE11b/model_b.py:ModBrooksCorey so the JSON config does
# not have to import the original model.
# ---------------------------------------------------------------------------


class ModBrooksCorey:
    """Per-phase modified Brooks-Corey relative permeability.

    Matches the implementation in :mod:`models.SPE11b.model_b`: for the
    aqueous phase, ``Se = (sat - swc) / (1 - swc - sgc)`` and
    ``k_r = krwe * Se ** nw``; for the gas/vapor phase ``Se = (sat - sgc) /
    (1 - swc - sgc)`` and ``k_r = krge * Se ** ng``.  Saturation is clipped
    to ``[0, 1]`` before exponentiation.

    :param phase: phase tag — ``"Aq"`` for water-wetting, anything else for
        the gas/vapor phase
    :type phase: str
    :param nw: aqueous exponent
    :type nw: float
    :param ng: gas exponent
    :type ng: float
    :param swc: connate water saturation
    :type swc: float
    :param sgc: critical gas saturation
    :type sgc: float
    :param krwe: end-point water relative permeability
    :type krwe: float
    :param krge: end-point gas relative permeability
    :type krge: float
    """

    def __init__(
        self,
        phase: str,
        nw: float = 1.5,
        ng: float = 1.5,
        swc: float = 0.32,
        sgc: float = 0.10,
        krwe: float = 1.0,
        krge: float = 1.0,
        **_: Any,
    ) -> None:
        """Initialise per-phase Brooks-Corey parameters.

        :param phase: phase identifier
        :type phase: str
        :param nw: water-phase exponent
        :type nw: float
        :param ng: gas-phase exponent
        :type ng: float
        :param swc: connate water saturation
        :type swc: float
        :param sgc: critical gas saturation
        :type sgc: float
        :param krwe: end-point water relative permeability
        :type krwe: float
        :param krge: end-point gas relative permeability
        :type krge: float
        """
        self.phase = phase
        if phase == "Aq":
            self.k_rw_e = krwe
            self.swc = swc
            self.sgc = 0.0
            self.nw = nw
        else:
            self.k_rg_e = krge
            self.sgc = sgc
            self.swc = 0.0
            self.ng = ng

    def evaluate(self, sat: float) -> float:
        """Evaluate phase relative permeability at saturation ``sat``.

        :param sat: phase saturation
        :type sat: float
        :returns: relative permeability for this phase
        :rtype: float
        """
        if self.phase == "Aq":
            se = (sat - self.swc) / (1.0 - self.swc - self.sgc)
            se = max(0.0, min(1.0, se))
            return self.k_rw_e * se**self.nw
        se = (sat - self.sgc) / (1.0 - self.swc - self.sgc)
        se = max(0.0, min(1.0, se))
        return self.k_rg_e * se**self.ng


def make_relperm_modbc(
    phase: Literal["Aq", "V"],
    nw: float = 1.5,
    ng: float = 1.5,
    swc: float = 0.32,
    sgc: float = 0.10,
    krwe: float = 1.0,
    krge: float = 1.0,
    **_: Any,
) -> ModBrooksCorey:
    """Build a :class:`ModBrooksCorey` evaluator from a flat config dict.

    Defined as a free function (not directly using the class as constructor)
    so :func:`darts.api.type_registry._wrap_constructor` can introspect a
    keyword-only signature instead of guessing at the dataclass.

    :returns: a per-phase modified Brooks-Corey evaluator
    """
    return ModBrooksCorey(
        phase=phase, nw=nw, ng=ng, swc=swc, sgc=sgc, krwe=krwe, krge=krge
    )


# ---------------------------------------------------------------------------
# DARTS plugin registry contract.
# ---------------------------------------------------------------------------


DARTS_PLUGIN_ENTRIES = [
    {
        "type_id": "flash/SPE11bVLAq@v1",
        "kind": "flash",
        "constructor": "spe11b_plugins.py:make_flash_vlaq",
        "doc": "DartsFlash VLAq PT-flash for CO2-H2O (PR + Aq activity model)",
    },
    {
        "type_id": "density/SPE11bGasPR@v1",
        "kind": "density",
        "constructor": "spe11b_plugins.py:make_density_pr_gas",
        "doc": "Peng-Robinson EoS gas density (uses shared SPE11b flash EoS)",
    },
    {
        "type_id": "density/SPE11bAqGarcia@v1",
        "kind": "density",
        "constructor": "spe11b_plugins.py:make_density_garcia_aq",
        "doc": "Garcia2001 aqueous density for CO2-H2O",
    },
    {
        "type_id": "viscosity/SPE11bGasFenghour@v1",
        "kind": "viscosity",
        "constructor": "spe11b_plugins.py:make_viscosity_fenghour",
        "doc": "Fenghour-1998 supercritical CO2 viscosity",
    },
    {
        "type_id": "viscosity/SPE11bAqIslam@v1",
        "kind": "viscosity",
        "constructor": "spe11b_plugins.py:make_viscosity_islam",
        "doc": "Islam-2012 aqueous viscosity for CO2-H2O",
    },
    {
        "type_id": "enthalpy/SPE11bGasPR@v1",
        "kind": "enthalpy",
        "constructor": "spe11b_plugins.py:make_enthalpy_pr_gas",
        "doc": "Peng-Robinson EoS enthalpy (gas phase)",
    },
    {
        "type_id": "enthalpy/SPE11bAqEoS@v1",
        "kind": "enthalpy",
        "constructor": "spe11b_plugins.py:make_enthalpy_aq",
        "doc": "Aq activity-model EoS enthalpy",
    },
    {
        "type_id": "relperm/SPE11bModBC@v1",
        "kind": "relperm",
        "constructor": "spe11b_plugins.py:make_relperm_modbc",
        "doc": "Modified Brooks-Corey relative permeability (per-phase, SPE11b)",
    },
]
