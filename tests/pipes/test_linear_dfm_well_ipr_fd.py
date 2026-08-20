"""Engine-level FD consistency tests for :class:`LinearDFMWellIPRHook` (M1b).

The hook applies a linear IPR flux between a DFM well segment and a reservoir
cell, writing both the RHS and (FD-based) Jacobian blocks straight into the
engine. These tests verify that what the hook WROTE into the block-CSR
Jacobian is consistent with what it APPLIES to the RHS:

1. build the real coupled DFM model (imported from
   ``models/dfm_well/2ph_2comp_isothermal_dfm_vertical_well_vs_olga`` the way
   ``run_test_suite2.py`` does -- by file path; the model class imports darts
   and dartsflash only) and ``init()`` it once per module;
2. for each ``PI_Type`` (MASS / MOLAR / VOLUMETRIC) and both an injector-like
   (p_well > p_res) and a producer-like (p_well < p_res) state: zero the
   engine RHS and ``jac_vals``, set the well/reservoir block states, apply a
   fresh hook, and record the RHS contribution plus the four written Jacobian
   blocks (extracted through ``BlockCSRView`` positions);
3. re-apply the hook at externally perturbed states (a DIFFERENT step size
   than the hook's internal FD uses, so the check is not tautological) and
   compare the external FD of the RHS contribution against the written
   Jacobian columns, entry-wise on dominant entries with rtol 2e-4.

The engine assembly itself is never re-run inside the FD loop -- the hook is
a pure post-assembly contribution, so zeroing RHS/jac isolates exactly what
it adds. The wellhead control rows must never be touched (the hook writes to
the perforated body segment and the reservoir cell only).

Validation coverage: duplicate connections, unknown well, out-of-bounds
perforation, negative PI, non-zero WI rejection, and WID gating via
``allow_nonzero_well_indexD`` (exercised by temporarily rewriting
``well.perforations``).
"""

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

pytest.importorskip("darts.engines")
pytest.importorskip("dartsflash")

from darts.models.conditions import BlockCSRView  # noqa: E402
from darts.pipes.linear_dfm_well_ipr import (  # noqa: E402
    LinearDFMWellIPRConnection,
    LinearDFMWellIPRHook,
    PI_Type,
)

MODEL_DIR = (
    pathlib.Path(__file__).resolve().parents[2]
    / "models"
    / "dfm_well"
    / "2ph_2comp_isothermal_dfm_vertical_well_vs_olga"
)

DT = 0.001  # days; only scales the hook contribution


@pytest.fixture(scope="module")
def dfm_model():
    """The OLGA-benchmark DFM model (20-segment vertical well, 2-cell radial
    reservoir, isothermal CO2/H2O), initialized once. Fast: < 2 s."""
    if not MODEL_DIR.is_dir():
        pytest.skip(f"model directory not found: {MODEL_DIR}")
    sys.path.insert(0, str(MODEL_DIR))
    try:
        spec = importlib.util.spec_from_file_location(
            "dfm_olga_ipr_model", MODEL_DIR / "model.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.Model()
        model.init(platform="cpu", verbose=0)
    finally:
        sys.path.remove(str(MODEL_DIR))
    return model


@pytest.fixture(scope="module")
def harness(dfm_model):
    """Static indices shared by all cases."""
    engine = dfm_model.physics.engine
    n_vars = dfm_model.physics.n_vars
    well = dfm_model.reservoir.wells[0]
    perf_segment_local, res_block, well_index, well_indexD = well.perforations[0]
    assert well_index == 0.0 and well_indexD == 0.0  # IPR manages the flux
    well_block = well.well_body_idx + perf_segment_local
    view = BlockCSRView(engine, n_vars)
    return {
        "model": dfm_model,
        "engine": engine,
        "n_vars": n_vars,
        "well": well,
        "well_block": well_block,
        "res_block": res_block,
        "view": view,
    }


def _fresh_hook(model, pi, pi_type, **kwargs):
    return LinearDFMWellIPRHook(
        model,
        [
            LinearDFMWellIPRConnection(
                well_name="I1", perforation_index=0, pi=pi, pi_type=pi_type
            )
        ],
        **kwargs,
    )


def _get_block(view, pos, n_vars):
    start = pos * view.block_size
    return view.jac_vals[start : start + view.block_size].reshape(n_vars, n_vars).copy()


CASES = [
    (PI_Type.MASS, 1.0e3, 110.0, 102.0),  # injector-like: p_well > p_res
    (PI_Type.MASS, 1.0e3, 95.0, 102.0),  # producer-like: p_well < p_res
    (PI_Type.MOLAR, 50.0, 110.0, 102.0),
    (PI_Type.MOLAR, 50.0, 95.0, 102.0),
    (PI_Type.VOLUMETRIC, 1.0, 110.0, 102.0),
    (PI_Type.VOLUMETRIC, 1.0, 95.0, 102.0),
]


@pytest.mark.parametrize(
    "pi_type,pi,p_well,p_res",
    CASES,
    ids=[f"{c[0].value}-{'inj' if c[2] > c[3] else 'prod'}" for c in CASES],
)
def test_ipr_jacobian_consistent_with_rhs_fd(harness, pi_type, pi, p_well, p_res):
    model = harness["model"]
    engine = harness["engine"]
    n_vars = harness["n_vars"]
    well = harness["well"]
    wb, rb = harness["well_block"], harness["res_block"]
    view = harness["view"]

    X = np.asarray(engine.X)
    rhs = np.asarray(engine.RHS)
    hook = _fresh_hook(model, pi, pi_type)

    # base states: interior compositions well away from the z bounds and from
    # the total_rate = 0 upstream switch (|p_well - p_res| >> h)
    x_well_saved = X[wb * n_vars : (wb + 1) * n_vars].copy()
    x_res_saved = X[rb * n_vars : (rb + 1) * n_vars].copy()
    try:
        X[wb * n_vars : (wb + 1) * n_vars] = [p_well, 0.3]
        X[rb * n_vars : (rb + 1) * n_vars] = [p_res, 0.05]

        rhs[:] = 0.0
        view.jac_vals[:] = 0.0
        hook.apply(DT)
        rhs_base = rhs.copy()

        resolved = hook._resolved_connections[0]
        J_ww = _get_block(view, resolved["diag_well"], n_vars)
        J_wr = _get_block(view, resolved["off_well_res"], n_vars)
        J_rw = _get_block(view, resolved["off_res_well"], n_vars)
        J_rr = _get_block(view, resolved["diag_res"], n_vars)

        # (i) the wellhead control rows and every other block stay untouched
        touched = np.zeros(len(rhs), dtype=bool)
        touched[wb * n_vars : (wb + 1) * n_vars] = True
        touched[rb * n_vars : (rb + 1) * n_vars] = True
        assert np.all(rhs_base[~touched] == 0.0)
        head_rows = slice(
            well.well_head_idx * n_vars, (well.well_head_idx + 1) * n_vars
        )
        assert np.all(rhs_base[head_rows] == 0.0)

        # (ii) conservation: what leaves the well enters the reservoir
        rhs_well_rows = rhs_base[wb * n_vars : (wb + 1) * n_vars]
        rhs_res_rows = rhs_base[rb * n_vars : (rb + 1) * n_vars]
        assert rhs_res_rows == pytest.approx(-rhs_well_rows, rel=1e-12)
        assert np.any(rhs_well_rows != 0.0)  # the flux is actually active

        # (iii) external FD of the RHS contribution vs the written Jacobian.
        # Steps deliberately differ from the hook's internal FD deltas
        # (max(|p| * 1e-7, 1e-7) and max(|z| * 1e-7, 1e-8)).
        rows = np.r_[
            np.arange(wb * n_vars, (wb + 1) * n_vars),
            np.arange(rb * n_vars, (rb + 1) * n_vars),
        ]
        steps = [3e-5, 1e-6]  # pressure [bar], composition
        for target_block, J_cols in (
            (wb, np.vstack([J_ww, J_rw])),
            (rb, np.vstack([J_wr, J_rr])),
        ):
            for var in range(n_vars):
                h = steps[var]
                idx = target_block * n_vars + var
                x_saved = X[idx]
                X[idx] += h
                rhs[:] = 0.0
                hook.apply(DT)
                fd_col = (rhs[rows] - rhs_base[rows]) / h
                X[idx] = x_saved

                j_col = J_cols[:, var]
                scale = max(np.abs(j_col).max(), np.abs(fd_col).max())
                if scale == 0.0:
                    # both identically zero (e.g. the downstream-composition
                    # column: the flux only uses the upstream composition)
                    continue
                # dominant entries at rtol 2e-4; tiny entries against an
                # absolute floor tied to the column scale
                np.testing.assert_allclose(
                    fd_col,
                    j_col,
                    rtol=2e-4,
                    atol=1e-6 * scale,
                    err_msg=f"{pi_type} block {target_block} var {var}",
                )
    finally:
        X[wb * n_vars : (wb + 1) * n_vars] = x_well_saved
        X[rb * n_vars : (rb + 1) * n_vars] = x_res_saved


# ------------------------------------------------------------- validation
def test_duplicate_connection_raises(harness):
    model = harness["model"]
    connection = LinearDFMWellIPRConnection(
        well_name="I1", perforation_index=0, pi=1.0, pi_type=PI_Type.MASS
    )
    hook = LinearDFMWellIPRHook(model, [connection, connection])
    with pytest.raises(ValueError, match="Duplicate"):
        hook._get_resolved_connections(harness["view"])


def test_unknown_well_raises_key_error(harness):
    hook = LinearDFMWellIPRHook(
        harness["model"],
        [
            LinearDFMWellIPRConnection(
                well_name="NOPE", perforation_index=0, pi=1.0, pi_type=PI_Type.MASS
            )
        ],
    )
    with pytest.raises(KeyError, match="NOPE"):
        hook._get_resolved_connections(harness["view"])


def test_out_of_bounds_perforation_raises(harness):
    hook = LinearDFMWellIPRHook(
        harness["model"],
        [
            LinearDFMWellIPRConnection(
                well_name="I1", perforation_index=99, pi=1.0, pi_type=PI_Type.MASS
            )
        ],
    )
    with pytest.raises(IndexError, match="99"):
        hook._get_resolved_connections(harness["view"])


def test_negative_pi_raises(harness):
    hook = LinearDFMWellIPRHook(
        harness["model"],
        [
            LinearDFMWellIPRConnection(
                well_name="I1", perforation_index=0, pi=-1.0, pi_type=PI_Type.MASS
            )
        ],
    )
    with pytest.raises(ValueError, match="negative productivity"):
        hook._get_resolved_connections(harness["view"])


@pytest.fixture
def perforation_editor(harness):
    """Temporarily rewrite the (single) perforation of the DFM well."""
    well = harness["well"]
    saved = list(well.perforations)

    def _set(well_index, well_indexD):
        seg, cell, _, _ = saved[0]
        well.perforations = [(seg, cell, well_index, well_indexD)]

    yield _set
    well.perforations = saved


def test_nonzero_well_index_rejected(harness, perforation_editor):
    perforation_editor(well_index=65.5, well_indexD=0.0)
    hook = _fresh_hook(harness["model"], 1.0, PI_Type.MASS)
    with pytest.raises(ValueError, match="double-counting"):
        hook._get_resolved_connections(harness["view"])


def test_nonzero_well_indexd_gated_by_flag(harness, perforation_editor):
    perforation_editor(well_index=0.0, well_indexD=12.5)
    hook = _fresh_hook(harness["model"], 1.0, PI_Type.MASS)
    with pytest.raises(ValueError, match="allow_nonzero_well_indexD"):
        hook._get_resolved_connections(harness["view"])
    # explicit opt-in resolves fine (WID drives an independent heat term)
    allowed = _fresh_hook(
        harness["model"], 1.0, PI_Type.MASS, allow_nonzero_well_indexD=True
    )
    resolved = allowed._get_resolved_connections(harness["view"])
    assert resolved[0]["well_block_idx"] == harness["well_block"]
    assert resolved[0]["res_block_idx"] == harness["res_block"]
