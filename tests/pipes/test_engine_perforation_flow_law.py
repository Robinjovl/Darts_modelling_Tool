"""Engine-side perforation flow law: analytic derivatives, and A/B against the hook (M5).

The linear IPR is a well perforation flow law. ``LinearDFMWellIPRHook`` computes
it in Python once per Newton iteration and differentiates it by finite
differences; ``add_perforation(flow_law=LinearIPR(...))`` hands the same law to
the well assembler in C++, which computes it with ANALYTIC derivatives with
respect to both connected blocks. These tests establish that the analytic
derivatives are the derivatives of the flux the engine actually applies.

**How the contribution is isolated.** Two instances of the same model are built,
differing in exactly one thing: the productivity and the intercept of the flow
law, which the twin sets to zero. A zero law takes the identical code path and
contributes identically zero, so at any state ``X``

    dR(X) = R_ipr(X) - R_zero(X)     and     dJ(X) = J_ipr(X) - J_zero(X)

are exactly the flow law's residual and Jacobian contributions -- everything
else in the assembly cancels. ``dJ`` is then checked against a central finite
difference of ``dR`` at a step unrelated to anything inside the engine.

**Why an isothermal MOLAR law is the sharp case.** Its flux is
``q_c = (A + B dp) * z_c`` -- a function of the state alone, with no interpolated
operator in it -- so the finite difference is exact and the agreement is at
round-off. The MASS, VOLUMETRIC and thermal laws additionally convert with the
upstream mixture molecular weight, molar density and molar enthalpy, which are
read from the piecewise-multilinear OBL operator table; there the finite
difference carries the interpolation's own error, and the tolerance reflects it.

The zero-drawdown case is checked separately: the residual contribution is
identically zero there, and the one-sided derivative from the injecting side is
what the assembler must have written (the upstream switch is a kink, exactly as
the engine's own phase upwinding is).
"""

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

pytest.importorskip("darts.engines")
pytest.importorskip("dartsflash")

from darts.engines import ms_well, perforation_flow_law_type  # noqa: E402
from darts.models.conditions import BlockCSRView  # noqa: E402
from darts.pipes.linear_dfm_well_ipr import (  # noqa: E402
    LinearDFMWellIPRConnection,
    LinearDFMWellIPRHook,
    LinearIPR,
    PI_Type,
)

MODELS_DIR = pathlib.Path(__file__).resolve().parents[2] / "models" / "dfm_well"
ISOTHERMAL_DIR = MODELS_DIR / "2ph_2comp_isothermal_dfm_vertical_well_vs_olga"
THERMAL_DIR = MODELS_DIR / "1ph_1comp_thermal_dfm_well_vs_olga"

DT = 0.001  # days; scales the contribution, and cancels out of the comparison


def _load_model_class(model_dir: pathlib.Path, module_name: str):
    if not model_dir.is_dir():
        pytest.skip(f"model directory not found: {model_dir}")
    sys.path.insert(0, str(model_dir))
    sys.path.insert(0, str(MODELS_DIR))
    try:
        spec = importlib.util.spec_from_file_location(
            module_name, model_dir / "model.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(MODELS_DIR))
        sys.path.remove(str(model_dir))
    return module.Model


def _build(model_cls, formulation, law: LinearIPR):
    """One initialized model whose engine-side flow law is exactly ``law``.

    The 1e20 far-field boundary volume of the benchmark is dropped. It is a mesh
    property with no bearing on the flow law, but it puts ``PV * d(acc)/dX`` of
    the perforated cell at ~1e20, whose ULP is ~1e4 -- so the flow law's own
    contribution to that diagonal block (order 1) would be quantized away when
    the two twins are differenced. Without it the assembled system stays at the
    scale of the fluxes and the contribution is isolated to twelve digits.
    """

    class _Patched(model_cls):
        def set_reservoir(self):
            super().set_reservoir()
            for face in list(self.reservoir.boundary_volumes):
                self.reservoir.boundary_volumes[face] = None

        def get_ipr_flow_law(self):
            return law

    model = _Patched(formulation=formulation)
    model.init(platform="cpu", verbose=0)
    return model


class _Pair:
    """A model with the flow law, and its twin with the law zeroed out."""

    def __init__(self, model_cls, formulation, law: LinearIPR):
        self.law = law
        self.ipr = _build(model_cls, formulation, law)
        self.zero = _build(
            model_cls,
            formulation,
            LinearIPR(productivity=0.0, basis=law.basis, offset=law.offset),
        )
        self.n_vars = self.ipr.physics.n_vars
        self.nc = self.ipr.physics.nc
        well = self.ipr.reservoir.wells[0]
        perf_segment_local, res_block = well.perforations[0][:2]
        self.well = well
        self.well_block = int(well.well_body_idx + perf_segment_local)
        self.res_block = int(res_block)
        self.view = BlockCSRView(self.ipr.physics.engine, self.n_vars)
        self.X0 = np.array(self.ipr.physics.engine.X, copy=True)
        self.scale = 0.0  # magnitude of the raw residual, set by contribution()

    def _assemble(self, model, X):
        """Assemble at ``X`` with the accumulation term annihilated.

        ``R = PV (acc(X) - acc_n) - dt (...)``, and the outer reservoir cell of
        this benchmark carries a 1e20 boundary volume, so ``PV acc`` reaches 1e16
        and its round-off (ULP 4) buries a flow-law contribution of order 10.
        Overwriting the previous-step operator values with the current ones makes
        every accumulation term identically zero, which leaves the assembled
        residual at the scale of the fluxes -- where the isolated contribution is
        resolved to twelve digits. The flow law itself does not read ``acc_n``, so
        the term under test is untouched, and both twins get the same treatment.
        """
        engine = model.physics.engine
        np.asarray(engine.X)[:] = X
        for _ in range(2):
            # the DFM momentum solution the engine assembly reads; the model's
            # own Newton loop refreshes it once per iteration
            model.update_dfm_well_vels_and_ders(DT, 0.0, 0)
            engine.assemble_linear_system(DT)
            np.asarray(engine.op_vals_arr_n)[:] = np.asarray(engine.op_vals_arr)
        return np.array(engine.RHS, copy=True), np.array(engine.jac_vals, copy=True)

    def contribution(self, X):
        """``(dR, dJ)``: exactly what the flow law adds at state ``X``.

        Both models are driven in lock-step through the same states, so the DFM
        pipe velocities -- which the engine assembly reads and which depend on
        ``X`` -- are identical on the two sides and cancel in the difference,
        together with every other term of the assembly.
        """
        r_ipr, j_ipr = self._assemble(self.ipr, X)
        r_zero, j_zero = self._assemble(self.zero, X)
        self.scale = max(self.scale, np.max(np.abs(r_ipr)))
        return r_ipr - r_zero, j_ipr - j_zero

    def block(self, d_jac, row_block, col_block):
        pos = (
            self.view.diag_pos(row_block)
            if row_block == col_block
            else self.view.block_pos(row_block, col_block)
        )
        bs = self.view.block_size
        return d_jac[pos * bs : (pos + 1) * bs].reshape(self.n_vars, self.n_vars)

    #: interior compositions for the two blocks, deliberately NOT round numbers.
    #: They must be away from the ``sim_eps`` clip, where the composition
    #: derivative of the law is legitimately zero and a finite difference would
    #: step off the clip -- and away from an OBL grid node, where the
    #: piecewise-multilinear operator table has a kink and its one-sided
    #: derivative (which is what the engine reports and uses) differs from a
    #: central difference by a step-independent 1.6e-4. Both are properties of
    #: the interpolation, not of the flow law.
    Z_WELL = 0.313372951
    Z_RES = 0.051372951

    def state_at(self, dp_well: float, dp_res: float = 0.0):
        """The initial state with the two block pressures shifted."""
        X = self.X0.copy()
        n = self.n_vars
        X[self.well_block * n] += dp_well
        X[self.res_block * n] += dp_res
        if self.nc > 1:
            X[self.well_block * n + 1] = self.Z_WELL
            X[self.res_block * n + 1] = self.Z_RES
        return X


# ---------------------------------------------------------------------------
# fixtures: each pair costs two model initializations, so they are module-scoped
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def isothermal_cls():
    return _load_model_class(ISOTHERMAL_DIR, "engine_ipr_isothermal_model")


@pytest.fixture(scope="module")
def thermal_cls():
    return _load_model_class(THERMAL_DIR, "engine_ipr_thermal_model")


@pytest.fixture(scope="module")
def pair_molar(isothermal_cls):
    # Reservoir-upstream (BHP producer) variant, molar PI: no operator enters the
    # flux, so the finite difference of dR is exact.
    return _Pair(
        isothermal_cls,
        "ipr_engine_producer",
        LinearIPR(productivity=5.0e3, basis=PI_Type.MOLAR),
    )


@pytest.fixture(scope="module")
def pair_mass(isothermal_cls):
    return _Pair(
        isothermal_cls, "ipr_engine", LinearIPR(productivity=1.0e5, basis=PI_Type.MASS)
    )


@pytest.fixture(scope="module")
def pair_volumetric(isothermal_cls):
    return _Pair(
        isothermal_cls,
        "ipr_engine_volumetric",
        LinearIPR(
            productivity=100.0,
            basis=PI_Type.VOLUMETRIC,
            offset=0.05,
            intercept=10.0,
        ),
    )


@pytest.fixture(scope="module")
def pair_thermal(thermal_cls):
    return _Pair(
        thermal_cls, "ipr_engine", LinearIPR(productivity=1.0e5, basis=PI_Type.MASS)
    )


# ---------------------------------------------------------------------------


def _check_fd(pair, dp_well, steps, rtol, dp_res=0.0):
    """Central FD of the flow law's residual vs the Jacobian it wrote.

    Entries below the round-off floor of the isolation (the ULP of the assembled
    residual, propagated through the difference quotient) are skipped: a
    component whose rate is ten orders of magnitude below the dominant one
    cannot be resolved by differencing two assemblies, and its derivative is not
    what this test is about.
    """
    n = pair.n_vars
    wb, rb = pair.well_block, pair.res_block
    X = pair.state_at(dp_well, dp_res)

    d_rhs, d_jac = pair.contribution(X)

    # the law touches these two blocks and nothing else
    touched = np.zeros(d_rhs.size, dtype=bool)
    touched[wb * n : (wb + 1) * n] = True
    touched[rb * n : (rb + 1) * n] = True
    assert np.all(d_rhs[~touched] == 0.0), "the flow law wrote outside its two blocks"

    # what leaves the well enters the reservoir; the two rows are the same
    # number with opposite signs, up to the round-off of the isolation
    noise = 64.0 * np.finfo(float).eps * pair.scale
    well_rows = d_rhs[wb * n : (wb + 1) * n]
    res_rows = d_rhs[rb * n : (rb + 1) * n]
    assert res_rows == pytest.approx(-well_rows, rel=1e-11, abs=noise)
    assert np.max(np.abs(well_rows)) > 0.0, "the flow law contributed nothing"

    blocks = {
        (wb, wb): pair.block(d_jac, wb, wb),
        (wb, rb): pair.block(d_jac, wb, rb),
        (rb, wb): pair.block(d_jac, rb, wb),
        (rb, rb): pair.block(d_jac, rb, rb),
    }

    # the reservoir row is the negative of the well row, block by block
    assert blocks[(rb, wb)] == pytest.approx(-blocks[(wb, wb)], rel=1e-11, abs=noise)
    assert blocks[(rb, rb)] == pytest.approx(-blocks[(wb, rb)], rel=1e-11, abs=noise)

    checked = 0
    for target in (wb, rb):
        for v in range(n):
            h = steps[v]
            Xp, Xm = X.copy(), X.copy()
            Xp[target * n + v] += h
            Xm[target * n + v] -= h
            fd = (pair.contribution(Xp)[0] - pair.contribution(Xm)[0]) / (2.0 * h)
            floor = 32.0 * np.finfo(float).eps * pair.scale / h

            for row in (wb, rb):
                analytic = blocks[(row, target)][:, v]
                numeric = fd[row * n : (row + 1) * n]
                for c in range(n):
                    mag = max(abs(analytic[c]), abs(numeric[c]))
                    if mag <= floor:
                        continue
                    checked += 1
                    assert numeric[c] == pytest.approx(analytic[c], rel=rtol), (
                        f"row block {row}, column block {target}, variable {v}, "
                        f"equation {c}: analytic {analytic[c]!r} vs FD {numeric[c]!r} "
                        f"(floor {floor:.3e})"
                    )
    assert checked > 0, "every entry was below the noise floor -- nothing was checked"


@pytest.mark.parametrize("dp_well", [8.0, -8.0], ids=["well-upstream", "res-upstream"])
def test_molar_derivatives_are_exact(pair_molar, dp_well):
    """An isothermal MOLAR law has no interpolated operator in it -- the flux is
    a bilinear function of the two block states -- so the central difference is
    exact and the agreement is at the round-off floor of the isolation."""
    _check_fd(pair_molar, dp_well, steps=[0.05, 1e-4], rtol=1e-7)


@pytest.mark.parametrize("dp_well", [8.0, -8.0], ids=["well-upstream", "res-upstream"])
def test_mass_derivatives_match_fd(pair_mass, dp_well):
    """MASS: the conversion divides by the upstream mixture molecular weight,
    which comes from the interpolated operator table."""
    _check_fd(pair_mass, dp_well, steps=[5e-3, 1e-5], rtol=1e-6)


@pytest.mark.parametrize("dp_well", [8.0, -8.0], ids=["well-upstream", "res-upstream"])
def test_volumetric_derivatives_match_fd(pair_volumetric, dp_well):
    """VOLUMETRIC: the conversion multiplies by the upstream total molar density,
    and this law also carries a non-zero intercept and pressure offset."""
    _check_fd(pair_volumetric, dp_well, steps=[5e-3, 1e-5], rtol=1e-6)


@pytest.mark.parametrize("dp_well", [8.0, -8.0], ids=["well-upstream", "res-upstream"])
def test_thermal_derivatives_match_fd(pair_thermal, dp_well):
    """Thermal, single component: the energy rate carries the upstream molar
    enthalpy and the potential-energy term ``spe * Mw``."""
    _check_fd(pair_thermal, dp_well, steps=[5e-3, 1e-3], rtol=1e-6)


def test_zero_drawdown_contributes_nothing_and_is_one_sided(pair_molar):
    """At zero drawdown the flux vanishes, and the derivative the assembler wrote
    is the one-sided derivative of the branch it selected (well upstream)."""
    pair = pair_molar
    n, wb, rb = pair.n_vars, pair.well_block, pair.res_block
    X = pair.state_at(0.0)
    X[rb * n] = X[wb * n]  # exactly zero drawdown

    d_rhs, d_jac = pair.contribution(X)
    assert np.max(np.abs(d_rhs)) == 0.0, "a zero-drawdown IPR must contribute nothing"

    # a forward step in p_well stays on the well-upstream branch the assembler chose
    h = 0.05
    Xp = X.copy()
    Xp[wb * n] += h
    fd = pair.contribution(Xp)[0] / h
    floor = 32.0 * np.finfo(float).eps * pair.scale / h
    checked = 0
    for row in (wb, rb):
        analytic = pair.block(d_jac, row, wb)[:, 0]
        numeric = fd[row * n : (row + 1) * n]
        for c in range(n):
            if max(abs(analytic[c]), abs(numeric[c])) <= floor:
                continue
            checked += 1
            assert numeric[c] == pytest.approx(analytic[c], rel=1e-7)
    assert checked > 0


def test_engine_law_and_python_hook_refuse_to_coexist(pair_mass):
    """Registering the Python hook on a perforation that already carries an
    engine-side law would apply the IPR twice."""
    model = pair_mass.ipr
    well = model.reservoir.wells[0]
    assert well.get_perforation_flow_law(0).law == perforation_flow_law_type.LINEAR_IPR
    hook = LinearDFMWellIPRHook(
        model,
        [
            LinearDFMWellIPRConnection(
                well_name=well.name,
                perforation_index=0,
                pi=1.0e5,
                pi_type=PI_Type.MASS,
            )
        ],
    )
    with pytest.raises(ValueError, match="ENGINE-SIDE flow law"):
        hook.bind(model)


# ---------------------------------------------------------------------------
# validation that needs no model at all
# ---------------------------------------------------------------------------


def test_linear_ipr_rejects_negative_productivity():
    with pytest.raises(ValueError, match="negative"):
        LinearIPR(productivity=-1.0, basis=PI_Type.MASS)


def test_linear_ipr_rejects_a_non_pi_type_basis():
    with pytest.raises(ValueError, match="PI_Type"):
        LinearIPR(productivity=1.0, basis="mass")


def test_flow_law_requires_a_zero_well_index():
    well = ms_well()
    well.name = "I1"
    well.perforations = [(3, 7, 12.5, 0.0)]
    with pytest.raises(RuntimeError, match="non-zero well index"):
        well.set_perforation_flow_law(0, LinearIPR(productivity=1.0).to_engine())


def test_flow_law_requires_an_existing_perforation():
    well = ms_well()
    well.name = "I1"
    well.perforations = [(3, 7, 0.0, 0.0)]
    with pytest.raises(RuntimeError, match="out of bounds"):
        well.set_perforation_flow_law(1, LinearIPR(productivity=1.0).to_engine())


def test_a_well_without_a_law_reports_darcy():
    well = ms_well()
    well.name = "I1"
    well.perforations = [(3, 7, 0.0, 0.0)]
    assert not well.has_non_darcy_perforation()
    assert well.get_perforation_flow_law(0).law == perforation_flow_law_type.DARCY

    well.set_perforation_flow_law(
        0, LinearIPR(productivity=2.5, basis=PI_Type.VOLUMETRIC, offset=1.5).to_engine()
    )
    assert well.has_non_darcy_perforation()
    law = well.get_perforation_flow_law(0)
    assert law.law == perforation_flow_law_type.LINEAR_IPR
    assert law.productivity == 2.5
    assert law.offset == 1.5
