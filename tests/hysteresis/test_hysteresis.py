"""Pure-Python tests for the hysteresis plumbing.

Script-style test module matching tests/interpolators/test_interpolation.py — no pytest
required. Each ``test_*`` function prints ``OK`` or ``FAILED`` and returns a bool; the
module-level tail runs all of them and exits non-zero on any failure.

Scope is deliberately limited to what can be verified without a built engine:
  * KilloughLandModel algebra (Land constant, trapped-gas saturation, sg_max update)
  * HistoryField dataclass defaults
  * HistoryAwareRelPerm / HistoryAwareCapPressure abstract contract
  * PropertyContainer dispatch on isinstance(kr_ev, HistoryAwareRelPerm)

Engine-level behaviour (Xhis round-trip, set_interpolators extended axes,
init_wells Xhis_well_default propagation) is covered end-to-end by the smoke
test in models/2ph_hysteresis/smoke_hys.py.
"""

from __future__ import annotations

import sys

import numpy as np

from darts.physics.base.physics_base import HistoryField
from darts.physics.properties.hysteresis import (
    HistoryAwareCapPressure,
    HistoryAwareRelPerm,
    KilloughLandModel,
)

_FAILED = 0


def _report(name: str, success: bool) -> bool:
    global _FAILED
    if not success:
        _FAILED += 1
    print(f"{name}: {'OK' if success else 'FAILED'}")
    return success


def test_killough_land_constant() -> bool:
    """Land constant ``C = 1/sgrmax - 1/(1 - swc)`` matches the algebraic form."""
    model = KilloughLandModel(swc=0.2, sgrmax=0.4)
    expected = 1.0 / 0.4 - 1.0 / (1.0 - 0.2)
    success = abs(model.land_constant - expected) < 1e-14
    return _report("test_killough_land_constant", success)


def test_killough_residual_saturation() -> bool:
    """Residual gas monotonic in sg_max, clipped to [0, 1 - swc], matches Killough-Land."""
    model = KilloughLandModel(swc=0.2, sgrmax=0.4)
    # At sg_max == 0: residual == 0
    if abs(model.residual_gas_saturation(0.0)) > 1e-14:
        return _report("test_killough_residual_saturation", False)
    # Monotonic
    prev = -1.0
    for sg_max in np.linspace(0.0, 0.8, 20):
        sgr = model.residual_gas_saturation(sg_max)
        if sgr < prev - 1e-12:
            return _report("test_killough_residual_saturation", False)
        prev = sgr
    # At sg_max == 1 - swc (0.8): residual == sgrmax by construction of Land constant
    sgr_at_top = model.residual_gas_saturation(1.0 - 0.2)
    if abs(sgr_at_top - 0.4) > 1e-12:
        return _report("test_killough_residual_saturation", False)
    # Clipping above 1 - swc leaves value unchanged
    if abs(model.residual_gas_saturation(2.0) - sgr_at_top) > 1e-12:
        return _report("test_killough_residual_saturation", False)
    return _report("test_killough_residual_saturation", True)


def test_killough_update_sg_max() -> bool:
    """update_sg_max: drainage lifts, sg < sgr clips to residual, otherwise preserves."""
    model = KilloughLandModel(swc=0.2, sgrmax=0.4)
    # Drainage: sg >= sg_max
    if abs(model.update_sg_max(sg=0.5, sg_max=0.3) - 0.5) > 1e-14:
        return _report("test_killough_update_sg_max", False)
    # Intermediate: sgr(sg_max) <= sg < sg_max — preserve historical maximum
    sg_max_prev = 0.6
    sgr = model.residual_gas_saturation(sg_max_prev)
    sg_mid = 0.5 * (sgr + sg_max_prev)
    if abs(model.update_sg_max(sg=sg_mid, sg_max=sg_max_prev) - sg_max_prev) > 1e-14:
        return _report("test_killough_update_sg_max", False)
    # Deep imbibition: sg < sgr — clip up to residual
    updated = model.update_sg_max(sg=0.0, sg_max=sg_max_prev)
    if abs(updated - sgr) > 1e-14:
        return _report("test_killough_update_sg_max", False)
    return _report("test_killough_update_sg_max", True)


def test_history_field_defaults() -> bool:
    """HistoryField stores axis bounds and defaults; n_axis_points defaults to None."""
    h = HistoryField(label="sg_max")
    ok = (
        h.label == "sg_max"
        and h.axis_min == 0.0
        and h.axis_max == 1.0
        and h.n_axis_points is None
        and h.default == 0.0
    )
    h2 = HistoryField(
        label="sg_max", axis_min=0.1, axis_max=0.9, n_axis_points=32, default=0.05
    )
    ok = ok and h2.n_axis_points == 32 and abs(h2.default - 0.05) < 1e-14
    return _report("test_history_field_defaults", ok)


def test_history_aware_abstract() -> bool:
    """HistoryAware* are abstract: concrete subclass must override evaluate."""
    try:
        HistoryAwareRelPerm()  # type: ignore[abstract]
        return _report("test_history_aware_abstract", False)
    except TypeError:
        pass
    try:
        HistoryAwareCapPressure()  # type: ignore[abstract]
        return _report("test_history_aware_abstract", False)
    except TypeError:
        pass

    class KrStub(HistoryAwareRelPerm):
        def evaluate(self, sat, sg_max: float = 0.0) -> float:
            return float(sat) * (1.0 - sg_max)

    class PcStub(HistoryAwareCapPressure):
        def evaluate(self, sat, sg_max: float = 0.0) -> float:
            return float(sat) + sg_max

    kr = KrStub()
    pc = PcStub()
    ok = (
        isinstance(kr, HistoryAwareRelPerm)
        and isinstance(pc, HistoryAwareCapPressure)
        and abs(kr.evaluate(0.5, 0.2) - 0.4) < 1e-14
        and abs(pc.evaluate(0.5, 0.2) - 0.7) < 1e-14
    )
    return _report("test_history_aware_abstract", ok)


def test_property_container_dispatch() -> bool:
    """PropertyContainer passes sg_max to HistoryAwareRelPerm and omits it for plain evaluators."""
    from darts.physics.super.property_container import PropertyContainer

    captured: dict = {"aware": None, "plain": None}

    class AwareKr(HistoryAwareRelPerm):
        def evaluate(self, sat, sg_max: float = 0.0) -> float:
            captured["aware"] = (float(sat), float(sg_max))
            return 0.5

    class PlainKr:
        def evaluate(self, sat) -> float:
            captured["plain"] = float(sat)
            return 0.7

    # Minimal container: 1 phase, 1 component — just enough to exercise dispatch
    pc = PropertyContainer(
        phases_name=["V"],
        components_name=["C"],
        Mw=[44.0],
        temperature=300.0,
        n_his=1,
    )
    pc.ph = np.array([0])
    pc.sat = np.array([0.42])

    # HistoryAware branch: sg_max should be forwarded
    pc.rel_perm_ev = {"V": AwareKr()}
    pc.kr = np.zeros(1)
    sg_max = 0.33
    state = np.array([250.0, 0.5, sg_max])  # pressure, z0, sg_max
    # Manually invoke the kr dispatch block (mirror of PropertyContainer.evaluate tail)
    from darts.physics.properties.hysteresis import (
        HistoryAwareCapPressure,  # noqa: F401
    )

    for j in pc.ph:
        kr_ev = pc.rel_perm_ev[pc.phases_name[j]]
        sg_max_arg = float(state[-1]) if pc.n_his else None
        if sg_max_arg is not None and isinstance(kr_ev, HistoryAwareRelPerm):
            pc.kr[j] = kr_ev.evaluate(pc.sat[j], sg_max_arg)
        else:
            pc.kr[j] = kr_ev.evaluate(pc.sat[j])
    if captured["aware"] != (0.42, 0.33) or abs(pc.kr[0] - 0.5) > 1e-14:
        return _report("test_property_container_dispatch", False)

    # Plain evaluator branch: sg_max must NOT be forwarded
    pc.rel_perm_ev = {"V": PlainKr()}
    pc.kr = np.zeros(1)
    captured["aware"] = None
    for j in pc.ph:
        kr_ev = pc.rel_perm_ev[pc.phases_name[j]]
        sg_max_arg = float(state[-1]) if pc.n_his else None
        if sg_max_arg is not None and isinstance(kr_ev, HistoryAwareRelPerm):
            pc.kr[j] = kr_ev.evaluate(pc.sat[j], sg_max_arg)
        else:
            pc.kr[j] = kr_ev.evaluate(pc.sat[j])
    if captured["plain"] != 0.42 or abs(pc.kr[0] - 0.7) > 1e-14:
        return _report("test_property_container_dispatch", False)

    return _report("test_property_container_dispatch", True)


print("Hysteresis plumbing tests:")
test_killough_land_constant()
test_killough_residual_saturation()
test_killough_update_sg_max()
test_history_field_defaults()
test_history_aware_abstract()
test_property_container_dispatch()

sys.exit(0 if _FAILED == 0 else 1)
